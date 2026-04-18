"""
Telco Churn Prediction Dashboard
Calls Azure ML Managed Online Endpoints for both XGBoost and AutoML models.
"""

import json
import os
import urllib.request
import urllib.error

import streamlit as st

# ── Endpoint config (set via environment variables or .env) ──────────────────
MANUAL_ENDPOINT_URL = os.getenv("MANUAL_ENDPOINT_URL", "")
MANUAL_ENDPOINT_KEY = os.getenv("MANUAL_ENDPOINT_KEY", "")
AUTOML_ENDPOINT_URL = os.getenv("AUTOML_ENDPOINT_URL", "")
AUTOML_ENDPOINT_KEY = os.getenv("AUTOML_ENDPOINT_KEY", "")

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered",
)

st.title("📡 Telco Customer Churn Predictor")
st.caption("Azure ML · XGBoost & AutoML · MLOps Capstone Project")
st.divider()


# ── Helper ───────────────────────────────────────────────────────────────────
def call_endpoint(url: str, key: str, payload: dict):
    if not url or not key:
        return None, "Endpoint not yet live (quota pending)"
    body = json.dumps({"input_data": {"columns": list(payload.keys()), "data": [list(payload.values())]}})
    req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)


def churn_badge(prob: float):
    if prob >= 0.7:
        return f"🔴 High churn risk ({prob:.0%})"
    elif prob >= 0.4:
        return f"🟡 Medium churn risk ({prob:.0%})"
    else:
        return f"🟢 Low churn risk ({prob:.0%})"


# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=1.0)

# ── Encode inputs ────────────────────────────────────────────────────────────
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

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("XGBoost Model")
        with st.spinner("Calling endpoint..."):
            result, err = call_endpoint(MANUAL_ENDPOINT_URL, MANUAL_ENDPOINT_KEY, payload)
        if err:
            st.warning(err)
        else:
            prob = result["churn_probability"][0]
            pred = result["predictions"][0]
            st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
            st.progress(prob)
            st.write(churn_badge(prob))

    with col_b:
        st.subheader("AutoML Model")
        with st.spinner("Calling endpoint..."):
            result, err = call_endpoint(AUTOML_ENDPOINT_URL, AUTOML_ENDPOINT_KEY, payload)
        if err:
            st.warning(err)
        else:
            prob = result["churn_probability"][0]
            pred = result["predictions"][0]
            st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
            st.progress(prob)
            st.write(churn_badge(prob))

st.divider()
st.caption("Built with Azure ML · Model: telco-churn-xgboost & telco-churn-automl · github.com/rnbhushansf1/churn_prediction")
