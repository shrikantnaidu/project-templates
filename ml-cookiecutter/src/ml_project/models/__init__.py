"""Model training and evaluation utilities."""

from ml_project.models.training import train_model
from ml_project.models.evaluation import evaluate_classification, evaluate_regression

__all__ = ["train_model", "evaluate_classification", "evaluate_regression"]
