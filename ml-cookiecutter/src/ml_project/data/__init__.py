"""Data loading and preprocessing module."""

from ml_project.data.loader import (
    load_csv,
    load_parquet,
    save_csv,
    save_parquet,
)

__all__ = ["load_csv", "load_parquet", "save_csv", "save_parquet"]
