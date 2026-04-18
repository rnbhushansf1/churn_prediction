"""
Preprocessing: handle nulls, encode categoricals, scale numerics,
split into train/val/test (70/15/15) and save as parquet files.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Column type definitions ──────────────────────────────────────────────────
TARGET = "Churn"
DROP_COLS = ["customerID"]   # identifier — not a feature

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median, categorical NaNs with mode."""
    for col in NUMERIC_COLS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("Filled %d nulls in '%s' with median %.4f", n_null, col, median_val)

    for col in CATEGORICAL_COLS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            log.info("Filled %d nulls in '%s' with mode '%s'", n_null, col, mode_val)

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    LabelEncode each categorical column.
    If encoders dict is provided, reuse fitted encoders (for val/test sets).
    Returns (transformed df, encoders dict).
    """
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in encoders:
            # Fit a new encoder on training data
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # Apply existing encoder — unseen labels fall back to -1
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


def scale_numerics(df: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale numeric columns.
    If scaler is None, fit a new one (training set).
    Returns (transformed df, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df, scaler


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load curated parquet ──────────────────────────────────────────────
    log.info("Loading parquet from %s", args.input_parquet)
    df = pd.read_parquet(args.input_parquet)
    log.info("Loaded %d rows, %d cols", *df.shape)

    # ── 2. Drop identifier column ─────────────────────────────────────────────
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # ── 3. Separate features and target ─────────────────────────────────────
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # ── 4. Train / val / test split (70 / 15 / 15) ──────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    log.info("Split sizes — train: %d  val: %d  test: %d", len(X_train), len(X_val), len(X_test))

    # ── 5. Handle nulls (fit on train, apply to all) ─────────────────────────
    # Compute fill values from training set only to prevent leakage
    null_fills = {}
    for col in NUMERIC_COLS:
        null_fills[col] = X_train[col].median()
    for col in CATEGORICAL_COLS:
        null_fills[col] = X_train[col].mode()[0] if len(X_train[col].mode()) else "Unknown"

    for split in [X_train, X_val, X_test]:
        for col in NUMERIC_COLS:
            split[col] = split[col].fillna(null_fills[col])
        for col in CATEGORICAL_COLS:
            split[col] = split[col].fillna(null_fills[col])

    # ── 6. Encode categoricals (fit on train) ────────────────────────────────
    X_train, encoders = encode_categoricals(X_train.copy())
    X_val, _ = encode_categoricals(X_val.copy(), encoders)
    X_test, _ = encode_categoricals(X_test.copy(), encoders)

    # ── 7. Scale numerics (fit on train) ─────────────────────────────────────
    X_train, scaler = scale_numerics(X_train.copy())
    X_val, _ = scale_numerics(X_val.copy(), scaler)
    X_test, _ = scale_numerics(X_test.copy(), scaler)

    # ── 8. Save splits as parquet ────────────────────────────────────────────
    splits = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }
    for split_name, (X_split, y_split) in splits.items():
        combined = X_split.copy()
        combined[TARGET] = y_split.values
        path = out_dir / f"{split_name}.parquet"
        combined.to_parquet(path, index=False)
        log.info("Saved %s split → %s (%d rows)", split_name, path, len(combined))

    # ── 9a. Write MLTable dirs for AutoML (train + val) ─────────────────────
    if args.train_mltable_dir or args.val_mltable_dir:
        import shutil
        mltable_template = (
            "$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json\n"
            "paths:\n"
            "  - file: ./{name}.parquet\n"
            "transformations:\n"
            "  - read_parquet:\n"
            "      include_path_column: false\n"
        )
        for split_name, mltable_dir_arg in [("train", args.train_mltable_dir), ("val", args.val_mltable_dir)]:
            if not mltable_dir_arg:
                continue
            mltable_dir = Path(mltable_dir_arg)
            mltable_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(out_dir / f"{split_name}.parquet", mltable_dir / f"{split_name}.parquet")
            (mltable_dir / "MLTable").write_text(
                mltable_template.format(name=split_name), encoding="utf-8"
            )
            log.info("Wrote MLTable → %s", mltable_dir / "MLTable")

    # ── 9. Persist fitted encoders and scaler for inference ──────────────────
    joblib.dump(encoders, artifacts_dir / "label_encoders.pkl")
    joblib.dump(scaler,   artifacts_dir / "scaler.pkl")
    joblib.dump(null_fills, artifacts_dir / "null_fills.pkl")
    log.info("Saved preprocessing artifacts to %s", artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telco churn parquet")
    parser.add_argument("--input-parquet", default="data/processed/telco_curated.parquet")
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--artifacts-dir", default="data/processed/artifacts")
    parser.add_argument("--train-mltable-dir", default=None, help="If set, write MLTable for AutoML train split here")
    parser.add_argument("--val-mltable-dir",   default=None, help="If set, write MLTable for AutoML val split here")
    main(parser.parse_args())
