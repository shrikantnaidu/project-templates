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

    def run(self, raw_data_path: str | Path, output_name: str = "features.parquet") -> Path:
        """Execute feature engineering and save results."""
        logger.info(f"Starting Feature Pipeline: {raw_data_path}")
        
        df = load_csv(raw_data_path)
        
        for processor in self.processors:
            logger.info(f"Applying processor: {processor.name}")
            df = processor.fit_transform(df)
            
        output_path = settings.processed_data_dir / output_name
        save_parquet(df, output_path)
        
        logger.info(f"Feature Pipeline complete. Saved to: {output_path}")
        return output_path
