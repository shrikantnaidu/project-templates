"""Tests for feature engineering module."""

import pandas as pd
import pytest

from ml_project.features import ColumnSelector, NullFiller, StandardScaler
from ml_project.pipelines import FeaturePipeline


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, None, 40.0],
            "c": ["x", "y", "z", "w"],
        }
    )


class TestColumnSelector:
    def test_select_columns(self, sample_df):
        selector = ColumnSelector(columns=["a", "b"])
        result = selector.fit_transform(sample_df)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 4

    def test_missing_column_raises(self, sample_df):
        selector = ColumnSelector(columns=["a", "missing"])
        with pytest.raises(ValueError, match="Columns not found"):
            selector.fit(sample_df)


class TestNullFiller:
    def test_fill_with_mean(self, sample_df):
        filler = NullFiller(strategy="mean", columns=["b"])
        result = filler.fit_transform(sample_df)
        # Mean of [10, 20, 40] = 23.33...
        assert result["b"].isna().sum() == 0
        assert abs(result["b"].iloc[2] - 23.333) < 0.01

    def test_fill_with_constant(self, sample_df):
        filler = NullFiller(strategy="constant", fill_value=0, columns=["b"])
        result = filler.fit_transform(sample_df)
        assert result["b"].iloc[2] == 0


class TestStandardScaler:
    def test_scale_columns(self):
        df = pd.DataFrame({"a": [0.0, 10.0, 20.0, 30.0]})
        scaler = StandardScaler(columns=["a"])
        result = scaler.fit_transform(df)

        # Check mean is approximately 0
        assert abs(result["a"].mean()) < 0.01
        # Check std is approximately 1
        assert abs(result["a"].std(ddof=0) - 1.0) < 0.1

    def test_inverse_transform(self):
        df = pd.DataFrame({"a": [0.0, 10.0, 20.0, 30.0]})
        scaler = StandardScaler(columns=["a"])
        scaled = scaler.fit_transform(df)
        restored = scaler.inverse_transform(scaled)

        # Should be close to original
        for i in range(len(df)):
            assert abs(restored["a"].iloc[i] - df["a"].iloc[i]) < 0.01


def test_feature_pipeline_transforms_feature_columns(sample_df):
    pipeline = FeaturePipeline(
        processors=[NullFiller(strategy="mean"), StandardScaler()]
    )
    result = pipeline.fit_transform(sample_df.drop(columns=["c"]))

    assert result["a"].mean() == pytest.approx(0.0)
    assert result["b"].isna().sum() == 0


def test_feature_processor_requires_fit(sample_df):
    with pytest.raises(RuntimeError, match="must be fitted"):
        StandardScaler(columns=["a"]).transform(sample_df)
