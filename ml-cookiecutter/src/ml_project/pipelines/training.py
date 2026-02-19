"""
Training Pipeline - Model Training and Registration.

Integrates with MLflow for experiment tracking and model versioning.
"""

from typing import Any

import mlflow
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ml_project.config import settings
from ml_project.models import evaluate_classification, evaluate_regression, train_model


class TrainingPipeline:
    """
    Trains a model and logs artifacts/metrics to MLflow.
    """

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri or "http://localhost:5000")
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        model: Any,
        task_type: str = "classification",
        params: dict[str, Any] | None = None,
    ) -> str:
        """Train model, log to MLflow, and return run ID."""
        logger.info(f"Starting Training Pipeline for experiment: {self.experiment_name}")

        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        with mlflow.start_run() as run:
            # Log params
            if params:
                mlflow.log_params(params)
            
            # Train
            trained_model = train_model(model, X_train, y_train)
            
            # Evaluate
            y_pred = trained_model.predict(X_test)
            if task_type == "classification":
                metrics = evaluate_classification(y_test, y_pred)
            else:
                metrics = evaluate_regression(y_test, y_pred)
            
            mlflow.log_metrics(metrics)
            
            # Register Model (MLflow handles persistence)
            mlflow.sklearn.log_model(
                sk_model=trained_model,
                artifact_path="model",
                registered_model_name=f"{self.experiment_name}_model"
            )
            
            logger.info(f"Training Pipeline complete. Run ID: {run.info.run_id}")
            return run.info.run_id
