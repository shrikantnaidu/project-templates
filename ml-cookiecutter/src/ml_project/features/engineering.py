"""
Feature engineering utilities.

Provides common feature transformations and a base class
for custom feature processors.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from loguru import logger


class BaseFeatureProcessor(ABC):
    """Base class for feature processors."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self._is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureProcessor":
        """Fit the processor on training data."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted processor."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before transform")


class ColumnSelector(BaseFeatureProcessor):
    """Select specific columns from a DataFrame."""

    def __init__(self, columns: list[str]):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame) -> "ColumnSelector":
        missing = set(self.columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return df[self.columns].copy()


class NullFiller(BaseFeatureProcessor):
    """Fill null values with specified strategy."""

    def __init__(
        self,
        strategy: str = "mean",  # mean, median, mode, constant
        fill_value: Any = None,
        columns: list[str] | None = None,
    ):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._fill_values: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "NullFiller":
        if self.strategy not in {"mean", "median", "mode", "constant"}:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        cols = self.columns or df.select_dtypes(include=["number"]).columns.tolist()
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value is required for the constant strategy")

        for col in cols:
            if self.strategy == "mean":
                self._fill_values[col] = df[col].mean()
            elif self.strategy == "median":
                self._fill_values[col] = df[col].median()
            elif self.strategy == "mode":
                modes = df[col].mode(dropna=True)
                if modes.empty:
                    raise ValueError(f"Cannot infer a mode for empty column: {col}")
                self._fill_values[col] = modes.iloc[0]
            elif self.strategy == "constant":
                self._fill_values[col] = self.fill_value

            if pd.isna(self._fill_values[col]):
                raise ValueError(f"Cannot infer a fill value for column: {col}")

        self._is_fitted = True
        logger.debug(f"Fitted NullFiller with {len(self._fill_values)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        result = df.copy()
        for col, value in self._fill_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)
        return result


class StandardScaler(BaseFeatureProcessor):
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self, columns: list[str] | None = None):
        super().__init__()
        self.columns = columns
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        cols = self.columns or df.select_dtypes(include=["number"]).columns.tolist()
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        for col in cols:
            self._means[col] = df[col].mean()
            # Match the population standard deviation used by sklearn.
            self._stds[col] = df[col].std(ddof=0)

        self._is_fitted = True
        logger.debug(f"Fitted StandardScaler with {len(self._means)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        result = df.copy()
        for col in self._means:
            if col in result.columns:
                result[col] = (result[col] - self._means[col]) / (
                    self._stds[col] + 1e-8
                )
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the standardization."""
        self._require_fitted()
        result = df.copy()
        for col in self._means:
            if col in result.columns:
                result[col] = result[col] * self._stds[col] + self._means[col]
        return result
