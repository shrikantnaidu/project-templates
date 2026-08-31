"""
Model training, evaluation, and persistence utilities.
"""

import pickle
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ml_project.config import settings


@runtime_checkable
class Estimator(Protocol):
    """Protocol for sklearn-compatible estimators."""

    def fit(self, X: Any, y: Any) -> "Estimator": ...
    def predict(self, X: Any) -> Any: ...


def train_model(
    model: Estimator,
    X_train: Any,
    y_train: Any,
    **fit_kwargs: Any,
) -> Estimator:
    """
    Train a model on the given data.

    Args:
        model: Any sklearn-compatible estimator
        X_train: Training features
        y_train: Training labels
        **fit_kwargs: Additional arguments for fit()

    Returns:
        Trained model
    """
    logger.info(f"Training model: {model.__class__.__name__}")
    model.fit(X_train, y_train, **fit_kwargs)
    logger.info("Training complete")
    return model


def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
) -> dict[str, float]:
    """
    Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    logger.info(f"Classification metrics: {metrics}")
    return metrics


def evaluate_regression(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """
    Evaluate regression model performance.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    logger.info(f"Regression metrics: {metrics}")
    return metrics


def save_model(
    model: Any,
    name: str,
    models_dir: Path | None = None,
) -> Path:
    """
    Save a trained model to disk.

    Args:
        model: Trained model object
        name: Model name (without extension)
        models_dir: Directory to save (default: settings.models_dir)

    Returns:
        Path to saved model
    """
    save_dir = models_dir or settings.models_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to: {model_path}")
    return model_path


def load_model(
    name: str,
    models_dir: Path | None = None,
) -> Any:
    """
    Load a trained model from disk.

    Args:
        name: Model name (with or without .pkl extension)
        models_dir: Directory to load from (default: settings.models_dir)

    Returns:
        Loaded model object
    """
    load_dir = models_dir or settings.models_dir

    # Handle both with and without extension
    if not name.endswith(".pkl"):
        name = f"{name}.pkl"

    model_path = load_dir / name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded from: {model_path}")
    return model
