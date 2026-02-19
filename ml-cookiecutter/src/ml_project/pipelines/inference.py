"""
Inference Pipeline - Real-time and Batch Prediction.

Loads models from Registry and performs inference.
"""

from typing import Any

import mlflow
import pandas as pd
from loguru import logger

from ml_project.config import settings


class InferencePipeline:
    """
    Loads a model and generates predictions.
    """

    def __init__(self, model_uri: str | None = None):
        # Default to latest production model if not specified
        self.model_uri = model_uri or f"models:/default_model/latest"
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading model from: {self.model_uri}")
            self._model = mlflow.sklearn.load_model(self.model_uri)
        return self._model

    def predict(self, data: pd.DataFrame) -> Any:
        """Generate predictions for the given data."""
        model = self._load_model()
        logger.debug(f"Running inference on {len(data)} samples")
        return model.predict(data)
