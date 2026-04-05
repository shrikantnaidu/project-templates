"""Tests for churn demo data preparation."""

from pathlib import Path

import pandas as pd

from ml_project.data.churn import download_telco_dataset, prepare_churn_features


def _sample_telco_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [1, 12, 24],
            "MonthlyCharges": [29.85, 56.95, 42.30],
            "TotalCharges": ["29.85", "1889.5", " "],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_download_telco_dataset(monkeypatch, tmp_path):
    """Download helper should write a CSV using pandas.read_csv."""
    sample = _sample_telco_df()

    def fake_read_csv(url: str):
        assert "Telco-Customer-Churn.csv" in url
        return sample

    monkeypatch.setattr("pandas.read_csv", fake_read_csv)

    out = download_telco_dataset(tmp_path / "raw.csv")
    assert out.exists()


def test_prepare_churn_features(tmp_path):
    """Prepared churn dataset should be fully numeric with target column."""
    raw_path = tmp_path / "raw.csv"
    _sample_telco_df().to_csv(raw_path, index=False)

    out = prepare_churn_features(raw_path, tmp_path / "processed.parquet")
    prepared = pd.read_parquet(out)

    assert "Churn" in prepared.columns
    assert prepared["Churn"].isin([0, 1]).all()
    assert "customerID" not in prepared.columns
    assert prepared.select_dtypes(exclude=["number"]).empty
