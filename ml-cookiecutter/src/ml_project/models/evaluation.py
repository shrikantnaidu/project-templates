"""Model evaluation helpers for classification and regression tasks."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_classification(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard regression metrics."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }
