"""
Data loading and preprocessing utilities.

Supports both pandas and polars for flexibility.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ml_project.config import settings


def load_csv(
    path: Path | str,
    use_polars: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV file as a DataFrame.

    Args:
        path: Path to CSV file (relative to data_dir or absolute)
        use_polars: If True, use polars instead of pandas
        **kwargs: Additional arguments passed to read_csv

    Returns:
        DataFrame (pandas or polars)
    """
    # Resolve path relative to data directory if not absolute
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = settings.data_dir / file_path

    logger.debug(f"Loading CSV from: {file_path}")

    if use_polars:
        import polars as pl

        return pl.read_csv(file_path, **kwargs)

    return pd.read_csv(file_path, **kwargs)


def save_csv(
    df: pd.DataFrame,
    path: Path | str,
    use_polars: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a DataFrame to CSV.

    Args:
        df: DataFrame to save
        path: Path for CSV file (relative to data_dir or absolute)
        use_polars: If True, assume df is a polars DataFrame
        **kwargs: Additional arguments passed to to_csv

    Returns:
        Absolute path to saved file
    """
    # Resolve path relative to data directory if not absolute
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = settings.data_dir / file_path

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving CSV to: {file_path}")

    if use_polars:
        df.write_csv(file_path, **kwargs)
    else:
        df.to_csv(file_path, index=False, **kwargs)

    return file_path.resolve()


def load_parquet(
    path: Path | str,
    use_polars: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a Parquet file as a DataFrame.

    Args:
        path: Path to Parquet file
        use_polars: If True, use polars instead of pandas
        **kwargs: Additional arguments

    Returns:
        DataFrame
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = settings.data_dir / file_path

    logger.debug(f"Loading Parquet from: {file_path}")

    if use_polars:
        import polars as pl

        return pl.read_parquet(file_path, **kwargs)

    return pd.read_parquet(file_path, **kwargs)


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    use_polars: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a DataFrame to Parquet.

    Args:
        df: DataFrame to save
        path: Path for Parquet file
        use_polars: If True, assume df is a polars DataFrame
        **kwargs: Additional arguments

    Returns:
        Absolute path to saved file
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = settings.data_dir / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving Parquet to: {file_path}")

    if use_polars:
        df.write_parquet(file_path, **kwargs)
    else:
        df.to_parquet(file_path, index=False, **kwargs)

    return file_path.resolve()
