"""
Scoring script for Azure ML Managed Online Endpoint.
Handles inference requests and logs inputs/outputs for monitoring (Phase 10).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)

MODEL = None
ENCODERS = None
SCALER = None
NULL_FILLS = None

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def init():
    global MODEL, ENCODERS, SCALER, NULL_FILLS

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "."))

    # Find model.json (XGBoost) or MLmodel (AutoML)
    model_files = list(model_dir.rglob("model.json"))
    if model_files:
        MODEL = xgb.XGBClassifier()
        MODEL.load_model(str(model_files[0]))
        log.info("Loaded XGBoost model from %s", model_files[0])
    else:
        import mlflow.sklearn
        MODEL = mlflow.sklearn.load_model(str(model_dir))
        log.info("Loaded AutoML mlflow model from %s", model_dir)

    # Load preprocessing artifacts if present
    enc_file = next(model_dir.rglob("label_encoders.pkl"), None)
    scaler_file = next(model_dir.rglob("scaler.pkl"), None)
    null_file = next(model_dir.rglob("null_fills.pkl"), None)
    if enc_file:
        ENCODERS = joblib.load(enc_file)
    if scaler_file:
        SCALER = joblib.load(scaler_file)
    if null_file:
        NULL_FILLS = joblib.load(null_file)


def run(raw_data: str) -> str:
    try:
        payload = json.loads(raw_data)

        # Support both {"input_data": {"columns": [...], "data": [[...]]}}
        # and {"data": [[...]]} formats
        if "input_data" in payload:
            columns = payload["input_data"].get("columns", FEATURE_COLS)
            data = payload["input_data"]["data"]
        else:
            columns = payload.get("columns", FEATURE_COLS)
            data = payload["data"]

        df = pd.DataFrame(data, columns=columns)

        # Apply preprocessing if artifacts were loaded
        if NULL_FILLS:
            for col, val in NULL_FILLS.items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

        if ENCODERS:
            for col, le in ENCODERS.items():
                if col in df.columns:
                    df[col] = df[col].astype(str).map(
                        lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                    )

        if SCALER:
            numeric_cols = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
            df[numeric_cols] = SCALER.transform(df[numeric_cols])

        proba = MODEL.predict_proba(df)[:, 1]
        predictions = (proba >= 0.5).astype(int).tolist()

        result = {
            "predictions": predictions,
            "churn_probability": proba.tolist(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log for monitoring (Phase 10) — writes to stdout, captured by Azure ML
        log.info("INFERENCE_LOG | inputs=%d | churn_rate=%.3f | ts=%s",
                 len(predictions), float(np.mean(predictions)), result["timestamp"])

        return json.dumps(result)

    except Exception as exc:
        log.error("Scoring error: %s", exc)
        return json.dumps({"error": str(exc)})
