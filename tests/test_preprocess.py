"""Unit tests for preprocessing logic — no Azure credentials needed."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing.preprocess import handle_nulls, encode_categoricals, scale_numerics, NUMERIC_COLS, CATEGORICAL_COLS, TARGET


def make_sample_df(n: int = 100) -> pd.DataFrame:
    """Build a minimal Telco-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "gender":          rng.choice(["Male", "Female"], n),
        "SeniorCitizen":   rng.integers(0, 2, n),
        "Partner":         rng.choice(["Yes", "No"], n),
        "Dependents":      rng.choice(["Yes", "No"], n),
        "tenure":          rng.integers(1, 72, n).astype(float),
        "PhoneService":    rng.choice(["Yes", "No"], n),
        "MultipleLines":   rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":  rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":    rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":     rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":     rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract":        rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling":rng.choice(["Yes", "No"], n),
        "PaymentMethod":   rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "MonthlyCharges":  rng.uniform(18, 120, n),
        "TotalCharges":    rng.uniform(18, 8000, n),
        TARGET:            rng.integers(0, 2, n),
    })
    return df


def test_handle_nulls_fills_numeric():
    df = make_sample_df(50)
    df.loc[0, "tenure"] = np.nan
    df.loc[1, "MonthlyCharges"] = np.nan
    result = handle_nulls(df.copy())
    assert result["tenure"].isna().sum() == 0
    assert result["MonthlyCharges"].isna().sum() == 0


def test_handle_nulls_fills_categorical():
    df = make_sample_df(50)
    df.loc[0, "gender"] = np.nan
    result = handle_nulls(df.copy())
    assert result["gender"].isna().sum() == 0


def test_encode_categoricals_returns_integers():
    df = make_sample_df(50)
    X = df.drop(columns=[TARGET])
    X_enc, encoders = encode_categoricals(X.copy())
    for col in CATEGORICAL_COLS:
        assert X_enc[col].dtype in [np.int64, np.int32, int], f"{col} should be integer after encoding"
    assert len(encoders) == len(CATEGORICAL_COLS)


def test_encode_categoricals_reuse_encoder():
    """Ensure applying an existing encoder doesn't re-fit on new data."""
    df = make_sample_df(100)
    X = df.drop(columns=[TARGET])
    X_train = X.iloc[:70].copy()
    X_val   = X.iloc[70:].copy()
    X_train_enc, encoders = encode_categoricals(X_train)
    X_val_enc, _          = encode_categoricals(X_val, encoders)
    # Val set should not cause KeyError or NaN
    assert X_val_enc[CATEGORICAL_COLS[0]].isna().sum() == 0


def test_scale_numerics_zero_mean():
    df = make_sample_df(200)
    X = df.drop(columns=[TARGET])
    X_scaled, scaler = scale_numerics(X.copy())
    for col in NUMERIC_COLS:
        assert abs(X_scaled[col].mean()) < 0.1, f"{col} mean should be near 0 after scaling"


def test_scale_numerics_reuse_scaler():
    df = make_sample_df(200)
    X = df.drop(columns=[TARGET])
    X_train = X.iloc[:140].copy()
    X_test  = X.iloc[140:].copy()
    _, scaler = scale_numerics(X_train)
    X_test_scaled, _ = scale_numerics(X_test, scaler)
    # Should not raise and should not contain NaN
    assert X_test_scaled[NUMERIC_COLS].isna().sum().sum() == 0
