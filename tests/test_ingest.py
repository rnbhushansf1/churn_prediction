"""Unit tests for ingestion validation logic — no Azure credentials needed."""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ingestion.ingest import validate_schema, clean_raw, SCHEMA


def make_valid_df(n: int = 10) -> pd.DataFrame:
    """Return a DataFrame that passes schema validation."""
    return pd.DataFrame({col: (["Yes"] * n if col not in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
                                else [1] * n)
                         for col in SCHEMA})


def test_validate_schema_passes_valid_df():
    df = make_valid_df()
    # Should not raise
    validate_schema(df)


def test_validate_schema_fails_missing_column():
    df = make_valid_df().drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)


def test_validate_schema_fails_high_null_rate():
    df = make_valid_df(20)
    # Make 50% of a required column null (exceeds 5% threshold)
    df.loc[:9, "tenure"] = None
    with pytest.raises(ValueError, match="null rate"):
        validate_schema(df)


def test_clean_raw_converts_total_charges():
    df = make_valid_df(5)
    df["TotalCharges"] = ["100.5", " ", "200.0", "300", " "]
    df["Churn"] = ["Yes", "No", "Yes", "No", "Yes"]
    result = clean_raw(df.copy())
    assert result["TotalCharges"].dtype == float
    # Spaces should become NaN
    assert result["TotalCharges"].isna().sum() == 2


def test_clean_raw_encodes_churn():
    df = make_valid_df(4)
    df["Churn"] = ["Yes", "No", "Yes", "No"]
    result = clean_raw(df.copy())
    assert set(result["Churn"].unique()).issubset({0, 1})


def test_clean_raw_drops_duplicates():
    df = make_valid_df(4)
    df["customerID"] = ["A", "A", "B", "C"]
    result = clean_raw(df.copy())
    assert len(result) == 3
