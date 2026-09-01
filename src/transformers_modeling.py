"""Transformer modeling utilities for CBCL severity prediction.

This module provides the main XLM-RoBERTa classification pipeline used
in the project, including:

- construction of transformer input text
- optional inclusion of structured metadata as text
- stratified inner train / validation splitting
- optional oversampling of the inner training set
- class-weighted cross-entropy loss
- optional freezing of the pretrained transformer backbone
- validation and held-out test evaluation

The held-out test set is never used for training, oversampling, or
model selection.
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
    confusion_matrix
)

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
    """
    Convert structured metadata columns into a textual block.

    Missing values are written explicitly as 'missing',
    instead of being converted to 0.
    """

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
    """
    Build text input for transformer experiments.

    mode = "event_eer":
        Uses Event + EER_text.

    mode = "event_only":
        Uses only Event.

    mode = "eer_only":
        Uses only EER_text.

    If metadata_columns is provided, metadata is appended as text.
    """

    df = df.copy()

    event = df["Event"].fillna("").astype(str)
    eer = df["EER_text"].fillna("").astype(str)

    if mode == "event_eer":
        texts = (
            "EVENT: " + event +
            "\n\nEER_TEXT: " + eer
        )

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
    """
    Oversample minority classes in the inner training split only.

    If target_counts is None:
        fully balances all classes to the majority class size.

    If target_counts is a dict:
        oversamples each class to the requested target count.
        Example:
            {0: 89, 1: 70, 2: 50}
    """

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    high_id = label2id["High"]

    return {
        "accuracy": accuracy_score(labels, preds),

        "macro_f1": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "high_precision": precision_score(
            labels == high_id,
            preds == high_id,
            zero_division=0
        ),

        "high_recall": recall_score(
            labels == high_id,
            preds == high_id,
            zero_division=0
        ),

        "high_f2": fbeta_score(
            labels == high_id,
            preds == high_id,
            beta=2,
            zero_division=0
        )
    }


def evaluate_predictions(y_true, y_pred):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    return {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),

        "macro_f1": f1_score(
            y_true_arr,
            y_pred_arr,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "high_precision": precision_score(
            y_true_arr == "High",
            y_pred_arr == "High",
            zero_division=0
        ),

        "high_recall": recall_score(
            y_true_arr == "High",
            y_pred_arr == "High",
            zero_division=0
        ),

        "high_f2": fbeta_score(
            y_true_arr == "High",
            y_pred_arr == "High",
            beta=2,
            zero_division=0
        ),

        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=LABELS,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=LABELS
        )
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights,
                dtype=torch.float
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


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
        print("Warning: Could not identify base model automatically. No freezing applied.")

    return model


def run_transformer_experiment(
    train_df,
    test_df,
    mode="event_eer",
    model_name="xlm-roberta-base",
    output_dir="./transformer_results",
    max_length=256,
    num_epochs=4,
    batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    validation_size=0.20,
    random_state=42,
    freeze_base_model=False,
    oversample_train=False,
    oversample_target_counts=None,
    metadata_columns=None
):
    """
    Fine-tune or train a classification head on a pretrained transformer.

    Oversampling, if used, is applied only to inner_train.
    Metadata, if used, is appended to the text input.
    Checkpoint saving is disabled to avoid filling Google Drive.
    """

    print("=" * 80)
    print("Running transformer experiment")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Freeze base model: {freeze_base_model}")
    print(f"Oversample inner train: {oversample_train}")
    print(f"Metadata columns: {metadata_columns}")
    print("Checkpoint saving: DISABLED")
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

    class_counts = np.bincount(inner_train_labels, minlength=len(LABELS))

    if np.any(class_counts == 0):
        raise ValueError(f"Missing class in inner train split: {class_counts}")

    class_weights = len(inner_train_labels) / (
        len(LABELS) * class_counts
    )

    print("\nClass counts used for training:")
    print(class_counts)
    print("Class weights:")
    print(class_weights)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    if freeze_base_model:
        print("\nFreezing base transformer model. Training classification head only.")
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()

    print("\nValidation evaluation after final epoch:")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    test_output = trainer.predict(test_dataset)

    test_logits = test_output.predictions

    test_probabilities = torch.softmax(
        torch.tensor(test_logits),
        dim=1
    ).numpy()

    test_pred_ids = np.argmax(test_logits, axis=1)
    test_true_ids = np.array(test_labels)

    y_true = [id2label[int(i)] for i in test_true_ids]
    y_pred = [id2label[int(i)] for i in test_pred_ids]

    final_results = evaluate_predictions(y_true, y_pred)

    print("\nTEST CLASSIFICATION REPORT")
    print(final_results["classification_report"])

    print("\nTEST SUMMARY METRICS")
    print(f"Accuracy:        {final_results['accuracy']:.3f}")
    print(f"Macro F1:        {final_results['macro_f1']:.3f}")
    print(f"High Precision:  {final_results['high_precision']:.3f}")
    print(f"High Recall:     {final_results['high_recall']:.3f}")
    print(f"High F2:         {final_results['high_f2']:.3f}")

    print("\nTEST CONFUSION MATRIX")
    cm_df = pd.DataFrame(
        final_results["confusion_matrix"],
        index=[f"True {x}" for x in LABELS],
        columns=[f"Pred {x}" for x in LABELS]
    )
    print(cm_df)

    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    for i, label in id2label.items():
        predictions_df[f"proba_{label}"] = test_probabilities[:, i]

    predictions_df["correct"] = (
        predictions_df["y_true"] == predictions_df["y_pred"]
    )

    predictions_df["input_text"] = build_input_text(
        test_df,
        mode=mode,
        metadata_columns=metadata_columns
    )

    if "CBCL_score" in test_df.columns:
        predictions_df["CBCL_score"] = test_df["CBCL_score"].values

    results = {
        "mode": mode,
        "model_name": model_name,
        "freeze_base_model": freeze_base_model,
        "oversample_train": oversample_train,
        "oversample_target_counts": oversample_target_counts,
        "metadata_columns": metadata_columns,

        "accuracy": final_results["accuracy"],
        "macro_f1": final_results["macro_f1"],
        "high_precision": final_results["high_precision"],
        "high_recall": final_results["high_recall"],
        "high_f2": final_results["high_f2"],

        "classification_report": final_results["classification_report"],
        "confusion_matrix": final_results["confusion_matrix"],

        "validation_metrics": val_metrics,

        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": test_probabilities,
        "predictions_df": predictions_df,

        "trainer": trainer,
        "tokenizer": tokenizer
    }

    return results


def summarize_results(results_list, baseline_dict=None):
    rows = []

    if baseline_dict is not None:
        rows.append(baseline_dict)

    for res in results_list:
        rows.append({
            "model": res["model_name"],
            "features": res["mode"],
            "freeze_base_model": res.get("freeze_base_model", None),
            "oversample_train": res.get("oversample_train", None),
            "metadata": bool(res.get("metadata_columns", None)),
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "high_precision": res["high_precision"],
            "high_recall": res["high_recall"],
            "high_f2": res["high_f2"]
        })

    return pd.DataFrame(rows)
