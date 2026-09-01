"""Classical modeling and evaluation utilities.

This module contains:
- a majority-class dummy baseline
- multinomial logistic regression
- cross-validated hyperparameter selection using High-class F2
- evaluation metrics for Low, Medium, and High CBCL severity classes
"""

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    fbeta_score,
    f1_score,
    make_scorer
)


LABELS = ["Low", "Medium", "High"]
POS_LABEL = "High"

def high_recall_metric(
    y_true,
    y_pred
):

    y_true_high = (
        np.asarray(y_true)
        == "High"
    ).astype(int)

    y_pred_high = (
        np.asarray(y_pred)
        == "High"
    ).astype(int)

    return recall_score(
        y_true_high,
        y_pred_high,
        zero_division=0
    )

def high_f2_metric(y_true, y_pred):
    return fbeta_score(
        y_true,
        y_pred,
        labels=[POS_LABEL],
        beta=2,
        average="macro",
        zero_division=0
    )


def evaluate_model(name, y_true, y_pred):
    p, r, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0
    )

    f2 = fbeta_score(
        y_true,
        y_pred,
        labels=LABELS,
        beta=2,
        average=None,
        zero_division=0
    )

    idx_low = LABELS.index("Low")
    idx_med = LABELS.index("Medium")
    idx_high = LABELS.index("High")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    return {
        "Model": name,

        "Macro F1": float(f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )),

        "Low Precision": float(p[idx_low]),
        "Low Recall": float(r[idx_low]),
        "Low F2": float(f2[idx_low]),

        "Medium Precision": float(p[idx_med]),
        "Medium Recall": float(r[idx_med]),
        "Medium F2": float(f2[idx_med]),

        "High Precision": float(p[idx_high]),
        "High Recall": float(r[idx_high]),
        "High F2": float(f2[idx_high]),

        "ConfusionMatrix": cm
    }


def train_majority_baseline(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, use_grid_search=True):
    base_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced"
    )

    if not use_grid_search:
        base_model.set_params(C=1)
        base_model.fit(X_train, y_train)
        return base_model

    scorer = make_scorer(high_f2_metric)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid={"C": [0.1, 0.3, 1, 3, 10]},
        scoring=scorer,
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV High F2:", grid.best_score_)

    return grid.best_estimator_
