"""Experiment utilities for baseline model analysis.

This module contains reusable procedures for evaluating the logistic
regression baseline beyond standard test-set performance.

It currently includes a null-distribution analysis based on repeated
permutation of the training labels.
"""

import numpy as np
import pandas as pd

from modeling import (
    train_logistic_regression,
    evaluate_model
)


def run_null_distribution(
    X_train,
    y_train,
    X_test,
    y_test,
    n_runs=100,
    random_state=42,
    use_grid_search=False
):
    rng = np.random.default_rng(random_state)

    rows = []

    for i in range(n_runs):
        print(f"Null run {i + 1}/{n_runs}")

        y_perm = rng.permutation(y_train)

        model = train_logistic_regression(
            X_train,
            y_perm,
            use_grid_search=use_grid_search
        )

        y_pred = model.predict(X_test)

        results = evaluate_model(
            f"Null {i + 1}",
            y_test,
            y_pred
        )

        rows.append({
            "run": i + 1,
            "Macro F1": results["Macro F1"],
            "High Precision": results["High Precision"],
            "High Recall": results["High Recall"],
            "High F2": results["High F2"]
        })

    return pd.DataFrame(rows)
