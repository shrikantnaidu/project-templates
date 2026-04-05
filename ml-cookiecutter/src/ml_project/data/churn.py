"""Utilities for preparing the Telco churn demo dataset."""

from pathlib import Path

import pandas as pd
from loguru import logger

from ml_project.config import settings
from ml_project.data.loader import load_csv, save_csv

TELCO_CHURN_URL = (
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


TARGET_COLUMN = "Churn"


def download_telco_dataset(output_path: str | Path = "raw/telco_churn.csv") -> Path:
    """Download the public Telco churn CSV and save it under the data directory."""
    logger.info("Downloading Telco churn dataset")
    df = pd.read_csv(TELCO_CHURN_URL)
    return save_csv(df, output_path)


def prepare_churn_features(
    input_path: str | Path = "raw/telco_churn.csv",
    output_path: str | Path = "processed/churn_features.parquet",
) -> Path:
    """
    Clean and encode the Telco churn dataset for model training.

    Processing includes:
    - numeric coercion for TotalCharges
    - missing value handling
    - dropping identifier columns
    - one-hot encoding categoricals
    - binary target conversion (Yes/No -> 1/0)
    """
    df = load_csv(input_path).copy()

    # Convert the known problematic numeric column.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill numeric missing values with median.
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode.
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(replacement)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    target = (df[TARGET_COLUMN].str.strip().str.lower() == "yes").astype(int)
    feature_df = df.drop(columns=[TARGET_COLUMN])

    # Remove row identifier if present.
    if "customerID" in feature_df.columns:
        feature_df = feature_df.drop(columns=["customerID"])

    encoded = pd.get_dummies(feature_df, drop_first=True)
    encoded[TARGET_COLUMN] = target.values

    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = settings.data_dir / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    encoded.to_parquet(output_file, index=False)
    logger.info(f"Prepared churn features with shape={encoded.shape} -> {output_file}")
    return output_file.resolve()
