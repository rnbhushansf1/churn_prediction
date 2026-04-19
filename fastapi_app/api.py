"""
FastAPI scoring service — serves both XGBoost and AutoML models.
Endpoints:
  POST /predict/xgboost
  POST /predict/automl
  GET  /health
  GET  /docs  (Swagger UI — free from FastAPI)
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn using XGBoost and AutoML models trained on Azure ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
XGBOOST_MODEL_PATH   = next((BASE / "models" / "xgboost").rglob("model.json"), None)
AUTOML_MODEL_PATH    = BASE / "models" / "automl"
ARTIFACTS_PATH       = next((BASE / "models" / "artifacts").rglob("label_encoders.pkl"), BASE / "models" / "artifacts" / "label_encoders.pkl").parent

# ── Blob logging setup ────────────────────────────────────────────────────────
STORAGE_CONN_STR  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
INFERENCE_CONTAINER = "inference-logs"
blob_client = None

def _init_blob():
    global blob_client
    if STORAGE_CONN_STR:
        try:
            from azure.storage.blob import BlobServiceClient
            blob_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
            log.info("Blob logging enabled")
        except Exception as e:
            log.warning("Blob logging disabled: %s", e)

def log_inference(model: str, inputs: dict, prediction: int, probability: float, risk: str):
    if not blob_client:
        return
    record = {
        "id":          str(uuid.uuid4()),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "model":       model,
        "inputs":      inputs,
        "prediction":  prediction,
        "probability": probability,
        "risk_level":  risk,
    }
    try:
        blob_name = f"{model}/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/predictions.jsonl"
        container = blob_client.get_container_client(INFERENCE_CONTAINER)
        try:
            existing = container.get_blob_client(blob_name).download_blob().readall().decode()
        except Exception:
            existing = ""
        content = existing + json.dumps(record) + "\n"
        container.get_blob_client(blob_name).upload_blob(content, overwrite=True)
    except Exception as e:
        log.warning("Failed to log inference: %s", e)

# ── Global model store ────────────────────────────────────────────────────────
models = {}
preprocessors = {}


@app.on_event("startup")
def load_models():
    _init_blob()

    # XGBoost
    if XGBOOST_MODEL_PATH and XGBOOST_MODEL_PATH.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(XGBOOST_MODEL_PATH))
        models["xgboost"] = m
        log.info("XGBoost model loaded")
    else:
        log.warning("XGBoost model not found at %s", XGBOOST_MODEL_PATH)

    # AutoML (mlflow model)
    mlmodel_file = next(AUTOML_MODEL_PATH.rglob("MLmodel"), None) if AUTOML_MODEL_PATH.exists() else None
    if mlmodel_file:
        import mlflow.sklearn
        try:
            models["automl"] = mlflow.sklearn.load_model(str(mlmodel_file.parent))
            log.info("AutoML model loaded")
        except Exception:
            import mlflow.pyfunc
            models["automl"] = mlflow.pyfunc.load_model(str(mlmodel_file.parent))
            log.info("AutoML model loaded via pyfunc")
    else:
        log.warning("AutoML model not found at %s", AUTOML_MODEL_PATH)

    # Preprocessing artifacts
    enc_file    = ARTIFACTS_PATH / "label_encoders.pkl"
    scaler_file = ARTIFACTS_PATH / "scaler.pkl"
    null_file   = ARTIFACTS_PATH / "null_fills.pkl"
    if enc_file.exists():
        preprocessors["encoders"]   = joblib.load(enc_file)
        preprocessors["scaler"]     = joblib.load(scaler_file)
        preprocessors["null_fills"] = joblib.load(null_file)
        log.info("Preprocessing artifacts loaded")


# ── Request / Response schemas ────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender:           int = Field(..., ge=0, le=1,  description="0=Female, 1=Male")
    SeniorCitizen:    int = Field(..., ge=0, le=1,  description="0=No, 1=Yes")
    Partner:          int = Field(..., ge=0, le=1,  description="0=No, 1=Yes")
    Dependents:       int = Field(..., ge=0, le=1,  description="0=No, 1=Yes")
    tenure:           int = Field(..., ge=0, le=72, description="Months as customer")
    PhoneService:     int = Field(..., ge=0, le=1)
    MultipleLines:    int = Field(..., ge=0, le=1)
    InternetService:  int = Field(..., ge=0, le=2,  description="0=DSL, 1=Fiber, 2=No")
    OnlineSecurity:   int = Field(..., ge=0, le=1)
    OnlineBackup:     int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport:      int = Field(..., ge=0, le=1)
    StreamingTV:      int = Field(..., ge=0, le=1)
    StreamingMovies:  int = Field(..., ge=0, le=1)
    Contract:         int = Field(..., ge=0, le=2,  description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling: int = Field(..., ge=0, le=1)
    PaymentMethod:    int = Field(..., ge=0, le=3,  description="0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card")
    MonthlyCharges:   float = Field(..., ge=0, le=200)
    TotalCharges:     float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
                "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
                "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
                "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
                "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
                "PaymentMethod": 0, "MonthlyCharges": 65.5, "TotalCharges": 786.0
            }
        }


class PredictionResponse(BaseModel):
    prediction:        int
    churn_probability: float
    risk_level:        str
    model:             str


# ── Preprocessing ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def preprocess(features: CustomerFeatures) -> pd.DataFrame:
    df = pd.DataFrame([features.model_dump()])
    if preprocessors:
        for col, val in preprocessors.get("null_fills", {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, le in preprocessors.get("encoders", {}).items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
        scaler = preprocessors.get("scaler")
        if scaler:
            cols = [c for c in NUMERIC_COLS if c in df.columns]
            df[cols] = scaler.transform(df[cols])
    return df


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.post("/predict/xgboost", response_model=PredictionResponse)
def predict_xgboost(features: CustomerFeatures):
    if "xgboost" not in models:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    df   = preprocess(features)
    prob = float(models["xgboost"].predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    risk = risk_label(prob)
    log_inference("xgboost", features.model_dump(), pred, round(prob, 4), risk)
    return PredictionResponse(prediction=pred, churn_probability=round(prob, 4), risk_level=risk, model="xgboost")


@app.post("/predict/automl", response_model=PredictionResponse)
def predict_automl(features: CustomerFeatures):
    if "automl" not in models:
        raise HTTPException(status_code=503, detail="AutoML model not loaded yet")
    df = preprocess(features)
    try:
        prob = float(models["automl"].predict_proba(df)[0, 1])
    except Exception:
        result = models["automl"].predict(df)
        prob   = float(np.array(result).flatten()[0])
    pred = int(prob >= 0.5)
    risk = risk_label(prob)
    log_inference("automl", features.model_dump(), pred, round(prob, 4), risk)
    return PredictionResponse(prediction=pred, churn_probability=round(prob, 4), risk_level=risk, model="automl")
