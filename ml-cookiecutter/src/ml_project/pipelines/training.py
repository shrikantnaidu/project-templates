"""
Training Pipeline - Model Training and Registration.

Integrates with MLflow for experiment tracking and model versioning.
"""

from typing import Any, Literal

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_project.config import settings
from ml_project.models import evaluate_classification, evaluate_regression, train_model
from ml_project.pipelines.feature import FeaturePipeline


class TrainingPipeline:
    """
    Trains a model and logs artifacts/metrics to MLflow.
    """

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri or "./mlruns")
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        model: Any,
        task_type: Literal["classification", "regression"] = "classification",
        params: dict[str, Any] | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        random_state: int = 42,
    ) -> str:
        """Train model, log to MLflow, and return run ID."""
        logger.info(
            f"Starting Training Pipeline for experiment: {self.experiment_name}"
        )

        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unknown task type: {task_type}")
        if target_column not in data.columns:
            raise ValueError(f"Target column not found: {target_column}")
        X = data.drop(columns=[target_column])
        y = data[target_column]
        split_kwargs: dict[str, Any] = {
            "test_size": 0.2,
            "random_state": random_state,
        }
        if task_type == "classification":
            split_kwargs["stratify"] = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

        model_to_train = model
        if feature_pipeline is not None:
            model_to_train = Pipeline(
                [
                    ("features", feature_pipeline),
                    ("model", model),
                ]
            )

        with mlflow.start_run() as run:
            # Log params
            if params:
                mlflow.log_params(params)

            # Train
            trained_model = train_model(model_to_train, X_train, y_train)

            # Evaluate
            y_pred = trained_model.predict(X_test)
            if task_type == "classification":
                metrics = evaluate_classification(y_test, y_pred)
            else:
                metrics = evaluate_regression(y_test, y_pred)

            mlflow.log_metrics(metrics)

            # Register Model (MLflow handles persistence)
            registered_name = f"{self.experiment_name}_model"
            model_info = mlflow.sklearn.log_model(
                sk_model=trained_model,
                artifact_path="model",
                registered_model_name=registered_name,
            )
            registered_version = getattr(model_info, "registered_model_version", None)
            if registered_version is not None:
                MlflowClient().set_registered_model_alias(
                    registered_name,
                    settings.model_alias,
                    str(registered_version),
                )

            logger.info(f"Training Pipeline complete. Run ID: {run.info.run_id}")
            return str(run.info.run_id)
