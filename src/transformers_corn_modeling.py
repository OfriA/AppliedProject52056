"""CORN ordinal transformer modeling for CBCL severity prediction.

This module adapts a pretrained transformer for ordinal classification
of CBCL severity:

    Low < Medium < High

The implementation uses the official CORN loss from ``coral-pytorch``.

The module supports:
- Event + EER text
- Event-only or EER-only input
- optional structured metadata appended as text
- stratified inner train / validation splitting
- optional oversampling of the inner training set
- optional freezing of the pretrained transformer backbone
- validation-based threshold selection
- final evaluation on a held-out test set

The held-out test set is never used for training, oversampling,
or threshold selection.
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from scipy.special import expit
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)


LABELS = ["Low", "Medium", "High"]

label2id = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

id2label = {
    0: "Low",
    1: "Medium",
    2: "High"
}




def metadata_to_text(df, metadata_columns=None):
    if metadata_columns is None or len(metadata_columns) == 0:
        return [""] * len(df)

    meta_texts = []

    for _, row in df.iterrows():
        parts = []

        for col in metadata_columns:
            if col not in df.columns:
                continue

            value = row[col]

            if pd.isna(value):
                value = "missing"

            parts.append(f"{col}={value}")

        meta_texts.append(" ; ".join(parts))

    return meta_texts


def build_input_text(df, mode="event_eer", metadata_columns=None):
    df = df.copy()

    event = df["Event"].fillna("").astype(str)
    eer = df["EER_text"].fillna("").astype(str)

    if mode == "event_eer":
        texts = "EVENT: " + event + "\n\nEER_TEXT: " + eer

    elif mode == "event_only":
        texts = "EVENT: " + event

    elif mode == "eer_only":
        texts = "EER_TEXT: " + eer

    else:
        raise ValueError("mode must be one of: event_eer, event_only, eer_only")

    texts = texts.tolist()

    if metadata_columns is not None and len(metadata_columns) > 0:
        meta_texts = metadata_to_text(df, metadata_columns)

        texts = [
            text + "\n\nMETADATA: " + meta
            for text, meta in zip(texts, meta_texts)
        ]

    return texts


def encode_labels(df):
    labels = df["CBCL_label"].map(label2id)

    if labels.isna().any():
        bad_values = df.loc[labels.isna(), "CBCL_label"].unique()
        raise ValueError(f"Unknown labels found: {bad_values}")

    return labels.astype(int).values


def oversample_text_labels(
    texts,
    labels,
    random_state=42,
    target_counts=None
):
    tmp_df = pd.DataFrame({
        "text": list(texts),
        "labels": list(labels)
    })

    original_counts = tmp_df["labels"].value_counts().sort_index()

    if target_counts is None:
        max_count = original_counts.max()
        target_counts = {
            label_value: max_count
            for label_value in original_counts.index
        }

    balanced_parts = []

    for label_value in sorted(tmp_df["labels"].unique()):
        class_df = tmp_df[tmp_df["labels"] == label_value]

        target_n = target_counts.get(label_value, len(class_df))

        if target_n > len(class_df):
            class_resampled = resample(
                class_df,
                replace=True,
                n_samples=target_n,
                random_state=random_state
            )
        else:
            class_resampled = class_df.sample(
                n=target_n,
                replace=False,
                random_state=random_state
            )

        balanced_parts.append(class_resampled)

    balanced_df = pd.concat(balanced_parts, axis=0)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return (
        balanced_df["text"].tolist(),
        balanced_df["labels"].astype(int).values
    )


def freeze_transformer_body(model):
    if hasattr(model, "roberta"):
        for param in model.roberta.parameters():
            param.requires_grad = False

    elif hasattr(model, "bert"):
        for param in model.bert.parameters():
            param.requires_grad = False

    elif hasattr(model, "base_model"):
        for param in model.base_model.parameters():
            param.requires_grad = False

    else:
        print("Warning: Could not identify transformer body. No freezing applied.")

    return model



def corn_loss(logits, labels, num_classes):
    """
    CORN loss for ordinal labels.

    For K classes, the model has K-1 output logits.

    In our case:
    classes: Low=0, Medium=1, High=2
    outputs:
        logit 0: P(y > Low)
        logit 1: P(y > Medium | y > Low)

    This implementation follows the conditional-subset logic of CORN.
    """

    total_loss = 0.0
    total_tasks_used = 0

    for task_index in range(num_classes - 1):
        # task_index = 0:
        # train on all examples, label is y > 0
        #
        # task_index = 1:
        # train only on examples where y > 0, label is y > 1

        mask = labels > (task_index - 1)

        if mask.sum() == 0:
            continue

        task_logits = logits[mask, task_index]
        task_targets = (labels[mask] > task_index).float()

        task_loss = F.binary_cross_entropy_with_logits(
            task_logits,
            task_targets,
            reduction="mean"
        )

        total_loss = total_loss + task_loss
        total_tasks_used += 1

    if total_tasks_used == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Average over tasks for more stable scale.
    return total_loss / total_tasks_used


class CORNTrainer(Trainer):
    """
    HuggingFace Trainer with custom CORN ordinal loss.

    Important:
    We remove labels before calling the HuggingFace model.
    Otherwise, the model tries to compute its internal CE loss with num_labels=2
    while labels are 0/1/2, which can cause CUDA device-side assert.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = corn_loss(
            logits=logits,
            labels=labels,
            num_classes=len(LABELS)
        )

        return (loss, outputs) if return_outputs else loss


def corn_logits_to_ordinal_probs(logits):
    """
    Convert CORN logits into rank-consistent cumulative probabilities.

    Returns:
        p_y_gt_low:
            P(y > Low)

        p_y_gt_medium:
            P(y > Medium)
            = P(y > Low) * P(y > Medium | y > Low)
    """

    cond_probs = expit(logits)

    p_y_gt_low = cond_probs[:, 0]
    p_y_gt_medium = cond_probs[:, 0] * cond_probs[:, 1]

    ordinal_probs = np.vstack([
        p_y_gt_low,
        p_y_gt_medium
    ]).T

    return ordinal_probs


def compute_corn_predictions(logits, threshold=0.5):
    """
    Convert CORN logits to final class predictions.

    Prediction rule:
        pred = 0
        if P(y > Low) > threshold:
            pred += 1
        if P(y > Medium) > threshold:
            pred += 1

    Therefore:
        0 -> Low
        1 -> Medium
        2 -> High
    """

    ordinal_probs = corn_logits_to_ordinal_probs(logits)

    p_y_gt_low = ordinal_probs[:, 0]
    p_y_gt_medium = ordinal_probs[:, 1]

    pred_ids = (
        (p_y_gt_low > threshold).astype(int) +
        (p_y_gt_medium > threshold).astype(int)
    )

    return pred_ids, ordinal_probs


def evaluate_predictions_from_ids(y_true_ids, y_pred_ids):
    y_true_ids = np.array(y_true_ids)
    y_pred_ids = np.array(y_pred_ids)

    y_true_labels = np.array([id2label[int(i)] for i in y_true_ids])
    y_pred_labels = np.array([id2label[int(i)] for i in y_pred_ids])

    high_id = label2id["High"]

    results = {
        "accuracy": accuracy_score(y_true_ids, y_pred_ids),

        "macro_f1": f1_score(
            y_true_labels,
            y_pred_labels,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "high_precision": precision_score(
            y_true_ids == high_id,
            y_pred_ids == high_id,
            zero_division=0
        ),

        "high_recall": recall_score(
            y_true_ids == high_id,
            y_pred_ids == high_id,
            zero_division=0
        ),

        "high_f2": fbeta_score(
            y_true_ids == high_id,
            y_pred_ids == high_id,
            beta=2,
            zero_division=0
        ),

        "mae": mean_absolute_error(y_true_ids, y_pred_ids),

        "rmse": mean_squared_error(
            y_true_ids,
            y_pred_ids
        ) ** 0.5,

        "classification_report": classification_report(
            y_true_labels,
            y_pred_labels,
            labels=LABELS,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true_labels,
            y_pred_labels,
            labels=LABELS
        )
    }

    return results


def make_compute_metrics(threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        pred_ids, _ = compute_corn_predictions(
            logits,
            threshold=threshold
        )

        results = evaluate_predictions_from_ids(
            labels,
            pred_ids
        )

        return {
            "accuracy": results["accuracy"],
            "macro_f1": results["macro_f1"],
            "high_precision": results["high_precision"],
            "high_recall": results["high_recall"],
            "high_f2": results["high_f2"],
            "mae": results["mae"],
            "rmse": results["rmse"]
        }

    return compute_metrics


def run_corn_experiment(
    train_df,
    test_df,
    mode="event_eer",
    model_name="xlm-roberta-base",
    output_dir="./corn_results",
    max_length=384,
    num_epochs=20,
    batch_size=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    validation_size=0.20,
    random_state=42,
    freeze_base_model=True,
    oversample_train=True,
    oversample_target_counts=None,
    metadata_columns=None,
    threshold=0.5
):
    print("=" * 80)
    print("Running CORN ordinal transformer experiment")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Freeze base model: {freeze_base_model}")
    print(f"Oversample inner train: {oversample_train}")
    print(f"Metadata columns: {metadata_columns}")
    print(f"Threshold: {threshold}")
    print("Output layer: 2 logits for CORN ordinal tasks")
    print("=" * 80)

    set_seed(random_state)

    train_texts = build_input_text(
        train_df,
        mode=mode,
        metadata_columns=metadata_columns
    )

    test_texts = build_input_text(
        test_df,
        mode=mode,
        metadata_columns=metadata_columns
    )

    train_labels = encode_labels(train_df)
    test_labels = encode_labels(test_df)

    inner_train_texts, val_texts, inner_train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_labels
    )

    print("\nBefore oversampling:")
    print(pd.Series(inner_train_labels).map(id2label).value_counts())

    if oversample_train:
        inner_train_texts, inner_train_labels = oversample_text_labels(
            inner_train_texts,
            inner_train_labels,
            random_state=random_state,
            target_counts=oversample_target_counts
        )

        print("\nAfter oversampling:")
        print(pd.Series(inner_train_labels).map(id2label).value_counts())

    print("\nFinal data sizes:")
    print("Inner train size:", len(inner_train_texts))
    print("Validation size:", len(val_texts))
    print("Test size:", len(test_texts))

    train_dataset = Dataset.from_dict({
        "text": list(inner_train_texts),
        "labels": list(inner_train_labels)
    })

    val_dataset = Dataset.from_dict({
        "text": list(val_texts),
        "labels": list(val_labels)
    })

    test_dataset = Dataset.from_dict({
        "text": list(test_texts),
        "labels": list(test_labels)
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_outputs = len(LABELS) - 1

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_outputs,
        ignore_mismatched_sizes=True
    )

    if freeze_base_model:
        print("\nFreezing base transformer model. Training CORN ordinal head only.")
        model = freeze_transformer_body(model)

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="epoch",
        logging_strategy="epoch",

        save_strategy="no",
        load_best_model_at_end=False,
        save_total_limit=0,

        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,

        report_to="none",
        seed=random_state
    )

    trainer = CORNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(threshold=threshold)
    )

    trainer.train()

    print("\nValidation evaluation after final epoch:")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions

    test_pred_ids, ordinal_probs = compute_corn_predictions(
        test_logits,
        threshold=threshold
    )

    test_true_ids = np.array(test_labels)

    final_results = evaluate_predictions_from_ids(
        test_true_ids,
        test_pred_ids
    )

    y_true_labels = [id2label[int(i)] for i in test_true_ids]
    y_pred_labels = [id2label[int(i)] for i in test_pred_ids]

    predictions_df = pd.DataFrame({
        "y_true": y_true_labels,
        "y_pred": y_pred_labels,
        "p_y_gt_low": ordinal_probs[:, 0],
        "p_y_gt_medium": ordinal_probs[:, 1],
        "correct": np.array(y_true_labels) == np.array(y_pred_labels)
    })

    predictions_df["input_text"] = build_input_text(
        test_df,
        mode=mode,
        metadata_columns=metadata_columns
    )

    if "CBCL_score" in test_df.columns:
        predictions_df["CBCL_score"] = test_df["CBCL_score"].values

    print("\nTEST CLASSIFICATION REPORT")
    print(final_results["classification_report"])

    print("\nTEST SUMMARY METRICS")
    print(f"Accuracy:        {final_results['accuracy']:.3f}")
    print(f"Macro F1:        {final_results['macro_f1']:.3f}")
    print(f"High Precision:  {final_results['high_precision']:.3f}")
    print(f"High Recall:     {final_results['high_recall']:.3f}")
    print(f"High F2:         {final_results['high_f2']:.3f}")
    print(f"MAE:             {final_results['mae']:.3f}")
    print(f"RMSE:            {final_results['rmse']:.3f}")

    print("\nTEST CONFUSION MATRIX")
    cm_df = pd.DataFrame(
        final_results["confusion_matrix"],
        index=[f"True {x}" for x in LABELS],
        columns=[f"Pred {x}" for x in LABELS]
    )
    print(cm_df)

    results = {
        "architecture": "CORN",
        "mode": mode,
        "model_name": model_name,
        "features": mode,
        "method": "Frozen XLM-R + CORN ordinal loss",
        "freeze_base_model": freeze_base_model,
        "oversample_train": oversample_train,
        "oversample_target_counts": oversample_target_counts,
        "metadata_columns": metadata_columns,
        "threshold": threshold,

        "accuracy": final_results["accuracy"],
        "macro_f1": final_results["macro_f1"],
        "high_precision": final_results["high_precision"],
        "high_recall": final_results["high_recall"],
        "high_f2": final_results["high_f2"],
        "mae": final_results["mae"],
        "rmse": final_results["rmse"],

        "classification_report": final_results["classification_report"],
        "confusion_matrix": final_results["confusion_matrix"],
        "validation_metrics": val_metrics,

        "test_logits": test_logits,
        "ordinal_probs": ordinal_probs,
        "test_true_ids": test_true_ids,
        "test_pred_ids": test_pred_ids,
        "y_true": y_true_labels,
        "y_pred": y_pred_labels,
        "predictions_df": predictions_df,

        "trainer": trainer,
        "tokenizer": tokenizer
    }

    return results


def evaluate_corn_thresholds(
    results_corn,
    thresholds=np.arange(0.20, 0.56, 0.05)
):
    """
    Evaluate the same trained CORN model under different decision thresholds.
    Useful because High is rare and threshold=0.5 may be too conservative.
    """

    test_logits = results_corn["test_logits"]
    y_true_ids = np.array(results_corn["test_true_ids"])

    rows = []

    for threshold in thresholds:
        pred_ids, ordinal_probs = compute_corn_predictions(
            test_logits,
            threshold=threshold
        )

        metrics = evaluate_predictions_from_ids(
            y_true_ids,
            pred_ids
        )

        pred_labels = np.array([id2label[int(i)] for i in pred_ids])

        rows.append({
            "threshold": float(threshold),
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "High Precision": metrics["high_precision"],
            "High Recall": metrics["high_recall"],
            "High F2": metrics["high_f2"],
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "Pred Low": int(np.sum(pred_labels == "Low")),
            "Pred Medium": int(np.sum(pred_labels == "Medium")),
            "Pred High": int(np.sum(pred_labels == "High"))
        })

    return pd.DataFrame(rows)


def get_confusion_matrix_for_threshold(results_corn, threshold):
    test_logits = results_corn["test_logits"]
    y_true_ids = np.array(results_corn["test_true_ids"])

    pred_ids, _ = compute_corn_predictions(
        test_logits,
        threshold=threshold
    )

    metrics = evaluate_predictions_from_ids(
        y_true_ids,
        pred_ids
    )

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"True {x}" for x in LABELS],
        columns=[f"Pred {x}" for x in LABELS]
    )

    return cm_df, metrics


def summarize_results(results_list, baseline_dict=None):
    rows = []

    if baseline_dict is not None:
        rows.append(baseline_dict)

    for res in results_list:
        rows.append({
            "architecture": res.get("architecture", "CORN"),
            "method": res.get("method", "CORN ordinal loss"),
            "model": res["model_name"],
            "features": res["mode"],
            "metadata_columns": res.get("metadata_columns"),
            "threshold": res.get("threshold"),
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "high_precision": res["high_precision"],
            "high_recall": res["high_recall"],
            "high_f2": res["high_f2"],
            "mae": res["mae"],
            "rmse": res["rmse"]
        })

    return pd.DataFrame(rows)
