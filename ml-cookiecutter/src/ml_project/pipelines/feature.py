"""
Feature Pipeline - Ingestion and Transformation.

Responsible for converting raw data into features/labels.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ml_project.config import settings
from ml_project.data import load_csv, save_parquet
from ml_project.features import BaseFeatureProcessor


class FeaturePipeline:
    """
    Standardizes raw data into features.
    In a real system, this might write to a Feature Store.
    """

    def __init__(self, processors: list[BaseFeatureProcessor] | None = None):
        self.processors = processors or []

    def fit(self, df: pd.DataFrame, y: object = None) -> "FeaturePipeline":
        """Fit processors in order on feature columns."""
        transformed = df
        for processor in self.processors:
            transformed = processor.fit_transform(transformed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted processors in order."""
        transformed = df.copy()
        for processor in self.processors:
            transformed = processor.transform(transformed)
        return transformed

    def fit_transform(self, df: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit processors in order and transform the input."""
        transformed = df.copy()
        for processor in self.processors:
            transformed = processor.fit_transform(transformed)
        return transformed

    def run(
        self,
        raw_data_path: str | Path,
        output_name: str = "features.parquet",
        target_column: str | None = None,
    ) -> Path:
        """Execute feature engineering and save results."""
        logger.info(f"Starting Feature Pipeline: {raw_data_path}")

        df = load_csv(raw_data_path)
        target = None
        if target_column is not None:
            if target_column not in df.columns:
                raise ValueError(f"Target column not found: {target_column}")
            target = df[target_column].copy()
            df = df.drop(columns=[target_column])

        for processor in self.processors:
            logger.info(f"Applying processor: {processor.name}")
        df = self.fit_transform(df)
        if target is not None:
            df[target_column] = target

        output_path = settings.processed_data_dir / output_name
        save_parquet(df, output_path)

        logger.info(f"Feature Pipeline complete. Saved to: {output_path}")
        return output_path
