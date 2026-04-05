"""Model training helpers."""

from typing import Any

import pandas as pd


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit a model and return the fitted estimator."""
    return model.fit(X_train, y_train)
