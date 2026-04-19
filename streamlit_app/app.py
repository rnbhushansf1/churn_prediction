"""
Telco Churn Prediction Dashboard
Calls FastAPI scoring service for XGBoost and AutoML predictions.
Monitoring tab reads inference logs from Azure Blob Storage.
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io"
)
STORAGE_CONN_STR    = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
INFERENCE_CONTAINER = "inference-logs"

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def call_endpoint(model: str, payload: dict):
    url  = f"{API_BASE_URL}/predict/{model}"
    body = json.dumps(payload)
    req  = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)


def churn_badge(prob: float):
    if prob >= 0.7:
        return f"🔴 High churn risk ({prob:.0%})"
    elif prob >= 0.4:
        return f"🟡 Medium churn risk ({prob:.0%})"
    return f"🟢 Low churn risk ({prob:.0%})"


@st.cache_data(ttl=60)
def load_inference_logs() -> pd.DataFrame:
    if not STORAGE_CONN_STR:
        return pd.DataFrame()
    try:
        from azure.storage.blob import BlobServiceClient
        client    = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        container = client.get_container_client(INFERENCE_CONTAINER)
        records   = []
        for blob in container.list_blobs():
            raw = container.get_blob_client(blob.name).download_blob().readall().decode()
            for line in raw.strip().splitlines():
                if line:
                    records.append(json.loads(line))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"Could not load logs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_baseline() -> dict:
    if not STORAGE_CONN_STR:
        return {}
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        raw = client.get_container_client(INFERENCE_CONTAINER) \
                    .get_blob_client("baseline/feature_baseline.json") \
                    .download_blob().readall()
        return json.loads(raw)
    except Exception:
        return {}


def compute_drift(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """PSI for categoricals, normalised mean-shift for numerics. Returns per-feature drift score."""
    if df.empty or not baseline:
        return pd.DataFrame()

    inputs_df = pd.DataFrame(df["inputs"].tolist())
    rows = []

    # Numeric: normalised absolute mean shift (|live_mean - base_mean| / base_std)
    for col, stats in baseline.get("numeric", {}).items():
        if col not in inputs_df.columns:
            continue
        live_mean = inputs_df[col].mean()
        shift     = abs(live_mean - stats["mean"]) / (stats["std"] + 1e-9)
        rows.append({"feature": col, "type": "numeric",
                     "drift_score": round(min(shift, 2.0), 4),
                     "baseline_mean": round(stats["mean"], 3),
                     "live_mean": round(live_mean, 3)})

    # Categorical: Population Stability Index (PSI)
    for col, base_dist in baseline.get("categorical", {}).items():
        if col not in inputs_df.columns:
            continue
        live_dist = inputs_df[col].value_counts(normalize=True).to_dict()
        psi = 0.0
        for k, base_p in base_dist.items():
            live_p = live_dist.get(k, 0.0001)
            base_p = max(base_p, 0.0001)
            psi   += (live_p - base_p) * np.log(live_p / base_p)
        rows.append({"feature": col, "type": "categorical",
                     "drift_score": round(abs(psi), 4),
                     "baseline_mean": "-", "live_mean": "-"})

    result = pd.DataFrame(rows).sort_values("drift_score", ascending=False)
    return result


@st.cache_data(ttl=30)
def get_health():
    try:
        with urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {"status": "unreachable", "models_loaded": []}


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_monitor = st.tabs(["🔍 Predict", "📊 Monitoring"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.title("📡 Telco Customer Churn Predictor")
    st.caption("Azure ML · XGBoost & AutoML · MLOps Capstone Project")
    st.divider()

    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender", ["Female", "Male"])
        senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner         = st.selectbox("Has Partner", ["No", "Yes"])
        dependents      = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        phone           = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup   = st.selectbox("Online Backup", ["No", "Yes"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support      = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes"])
        contract          = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless         = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment           = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=1.0)

    payload = {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "tenure":           tenure,
        "PhoneService":     1 if phone == "Yes" else 0,
        "MultipleLines":    1 if multiple_lines == "Yes" else 0,
        "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        "OnlineSecurity":   1 if online_security == "Yes" else 0,
        "OnlineBackup":     1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport":      1 if tech_support == "Yes" else 0,
        "StreamingTV":      1 if streaming_tv == "Yes" else 0,
        "StreamingMovies":  1 if streaming_movies == "Yes" else 0,
        "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "PaymentMethod":    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment),
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    st.divider()

    if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("XGBoost Model")
            with st.spinner("Calling API..."):
                result, err = call_endpoint("xgboost", payload)
            if err:
                st.warning(err)
            else:
                prob = result["churn_probability"]
                pred = result["prediction"]
                st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
                st.progress(prob)
                st.write(churn_badge(prob))

        with col_b:
            st.subheader("AutoML Model")
            with st.spinner("Calling API..."):
                result, err = call_endpoint("automl", payload)
            if err:
                st.warning(f"AutoML: {err}")
            else:
                prob = result["churn_probability"]
                pred = result["prediction"]
                st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
                st.progress(prob)
                st.write(churn_badge(prob))

    st.divider()
    st.caption("Built with Azure ML · Models: telco-churn-xgboost & telco-churn-automl · github.com/rnbhushansf1/churn_prediction")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.title("📊 MLOps Monitoring Dashboard")
    st.caption("Live inference logs · Endpoint health · Model performance")

    if st.button("🔄 Refresh", key="refresh"):
        st.cache_data.clear()

    # ── Endpoint health ───────────────────────────────────────────────────────
    st.subheader("Endpoint Health")
    health = get_health()
    h_col1, h_col2, h_col3 = st.columns(3)
    h_col1.metric("API Status", "🟢 Online" if health["status"] == "ok" else "🔴 Down")
    h_col2.metric("Models Loaded", len(health.get("models_loaded", [])))
    h_col3.metric("Models", ", ".join(health.get("models_loaded", [])) or "None")

    st.divider()

    # ── Inference logs ────────────────────────────────────────────────────────
    st.subheader("Live Inference Logs")
    df = load_inference_logs()

    if df.empty:
        st.info("No predictions logged yet. Make some predictions in the Predict tab first.")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", len(df))
        m2.metric("XGBoost Calls",  int((df["model"] == "xgboost").sum()))
        m3.metric("AutoML Calls",   int((df["model"] == "automl").sum()))
        churn_rate = df["prediction"].mean()
        m4.metric("Overall Churn Rate", f"{churn_rate:.1%}")

        st.divider()

        # Churn rate over time
        st.subheader("Churn Rate Over Time")
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby(["date", "model"])["prediction"].mean().reset_index()
        daily.columns = ["date", "model", "churn_rate"]
        pivot = daily.pivot(index="date", columns="model", values="churn_rate")
        st.line_chart(pivot)

        # Prediction volume
        st.subheader("Prediction Volume by Model")
        vol = df.groupby(["date", "model"]).size().reset_index(name="count")
        vol_pivot = vol.pivot(index="date", columns="model", values="count").fillna(0)
        st.bar_chart(vol_pivot)

        # Risk distribution
        st.subheader("Risk Level Distribution")
        r1, r2 = st.columns(2)
        for col, model_name in zip([r1, r2], ["xgboost", "automl"]):
            model_df = df[df["model"] == model_name]
            if not model_df.empty:
                dist = model_df["risk_level"].value_counts()
                col.write(f"**{model_name.upper()}**")
                col.dataframe(dist.rename("count"), use_container_width=True)

        # Avg churn probability
        st.subheader("Average Churn Probability by Model")
        avg = df.groupby("model")["probability"].mean().reset_index()
        avg.columns = ["Model", "Avg Churn Probability"]
        avg["Avg Churn Probability"] = avg["Avg Churn Probability"].round(4)
        st.dataframe(avg, use_container_width=True, hide_index=True)

        st.divider()

        # Recent predictions table
        st.subheader("Recent Predictions")
        display_cols = ["timestamp", "model", "prediction", "probability", "risk_level"]
        recent = df[display_cols].head(20).copy()
        recent["prediction"] = recent["prediction"].map({1: "Churn", 0: "Stay"})
        recent["timestamp"]  = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(recent, use_container_width=True, hide_index=True)

    st.divider()

    # ── Data Drift ────────────────────────────────────────────────────────────
    st.subheader("Data Drift Detection")
    st.caption("Numeric: normalised mean-shift · Categorical: Population Stability Index (PSI) · Threshold: 0.20")

    baseline = load_baseline()
    drift_df = compute_drift(df, baseline) if not df.empty else pd.DataFrame()

    DRIFT_THRESHOLD = 0.20

    if drift_df.empty:
        st.info("Not enough inference data to compute drift yet. Make more predictions first.")
    else:
        # Summary alert
        drifted = drift_df[drift_df["drift_score"] > DRIFT_THRESHOLD]
        if drifted.empty:
            st.success(f"No drift detected — all {len(drift_df)} features within threshold ({DRIFT_THRESHOLD})")
        else:
            st.error(f"Drift detected in {len(drifted)} feature(s): {', '.join(drifted['feature'].tolist())}")

        # Bar chart of drift scores
        chart_df = drift_df.set_index("feature")[["drift_score"]]
        st.bar_chart(chart_df)

        # Threshold line annotation
        st.caption(f"Red threshold line = {DRIFT_THRESHOLD} (JSdivergence / normalised shift)")

        # Full table
        with st.expander("Full drift scores per feature"):
            display = drift_df.copy()
            display["status"] = display["drift_score"].apply(
                lambda x: "⚠️ Drifted" if x > DRIFT_THRESHOLD else "✅ OK"
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── Model info ────────────────────────────────────────────────────────────
    st.subheader("Model Registry")
    reg_data = {
        "Model":      ["telco-churn-xgboost", "telco-churn-automl (RF)"],
        "Type":       ["XGBoost", "RandomForest"],
        "Val ROC-AUC":["~0.85", "0.842"],
        "Status":     ["Production", "Production"],
        "Serving":    ["FastAPI /predict/xgboost", "FastAPI /predict/automl"],
    }
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Logs stored in Azure Blob Storage · Refreshes every 60 seconds · github.com/rnbhushansf1/churn_prediction")
