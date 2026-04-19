# Telco Customer Churn Prediction — End-to-End MLOps on Azure ML

Capstone project implementing a complete 15-phase MLOps lifecycle on Azure Machine Learning
with two model tracks, automated retraining, inference logging, data drift detection,
and a live Streamlit monitoring dashboard.

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | https://churn-dashboard.mangobeach-a2290557.southeastasia.azurecontainerapps.io |
| FastAPI + Swagger UI | https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io/docs |
| API Health | https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io/health |

## Architecture Overview

```
Kaggle CSV → Azure Blob Storage → Azure ML Pipeline
                                         │
                          ┌──────────────┴──────────────┐
                   Manual Track                   AutoML Track
              (LR Baseline + XGBoost)          (Random Forest)
                          │                              │
                   MLflow logging                 MLflow logging
                          │                              │
                   Azure ML Registry ←─────────────────┘
                          │
                   FastAPI (Azure Container Apps)
                   POST /predict/xgboost
                   POST /predict/automl
                          │                    │
                   Streamlit Dashboard    Azure Blob
                   • Predict tab         inference-logs/
                   • Monitoring tab      predictions.jsonl
                          │
                   Data Drift Detection
                   (PSI + mean-shift vs baseline)
```

> Full detailed architecture diagrams (data flow, training tracks, monitoring, CI/CD):
> see [docs/architecture.md](docs/architecture.md)

---

## 15-Phase Completion

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Use Case Definition | ✅ |
| 2 | Cloud Resource Setup | ✅ |
| 3 | Repository Structure | ✅ |
| 4 | Data Ingestion | ✅ |
| 5 | Preprocessing | ✅ |
| 6 | Model Training (LR baseline + XGBoost + AutoML) | ✅ |
| 7 | Model Evaluation | ✅ |
| 8 | Model Registry + version tags | ✅ |
| 9 | Model Deployment (FastAPI + Container Apps) | ✅ |
| 10 | Production Data Capture (Blob inference logs) | ✅ |
| 11 | Model Monitoring (drift detection + App Insights) | ✅ |
| 12 | Automated Retraining + Redeploy | ✅ |
| 13 | CI/CD Integration (GitHub Actions) | ✅ |
| 14 | Approval Workflow (production environment gate) | ✅ |
| 15 | Dashboard & Reporting (Streamlit monitoring tab) | ✅ |

---

## Prerequisites

- Azure subscription with ML Workspace `mlops-churn-ws` (resource group `mlops-churn-rg`)
- Compute cluster `cpu-cluster` (Standard_DS3_v2, 0–4 nodes)
- Python 3.10 + `pip install -r requirements.txt`
- Azure CLI: `az extension add --name ml`
- Kaggle Telco dataset in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Quick Start (local)

```bash
pip install -r requirements.txt

# Ingest and validate
python src/ingestion/ingest.py \
  --input-csv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --no-register

# Preprocess (creates train/val/test parquet splits)
python src/preprocessing/preprocess.py

# Train baseline Logistic Regression
python src/training/train_baseline.py \
  --splits-dir data/processed/splits \
  --model-dir outputs/baseline_model

# Train XGBoost
python src/training/train_manual.py \
  --splits-dir data/processed/splits \
  --model-dir outputs/manual_model

# Run unit tests
pytest tests/ -v
```

---

## Azure ML Pipelines

```bash
# Manual (LR Baseline + XGBoost) pipeline
JOB=$(az ml job create --file pipelines/manual_pipeline.yaml \
  --workspace-name mlops-churn-ws --resource-group mlops-churn-rg \
  --query name -o tsv)
az ml job stream --name $JOB \
  --workspace-name mlops-churn-ws --resource-group mlops-churn-rg

# AutoML pipeline
JOB=$(az ml job create --file pipelines/automl_pipeline.yaml \
  --workspace-name mlops-churn-ws --resource-group mlops-churn-rg \
  --query name -o tsv)
az ml job stream --name $JOB \
  --workspace-name mlops-churn-ws --resource-group mlops-churn-rg
```

---

## FastAPI Scoring Service

```bash
# Local
cd fastapi_app
pip install -r requirements.txt
uvicorn api:app --reload

# Docker
docker build -t churn-api fastapi_app/
docker run -p 8000:8000 \
  -e AZURE_STORAGE_CONNECTION_STRING=<conn_str> \
  churn-api
```

Endpoints:
- `POST /predict/xgboost` — XGBoost prediction + logs to Azure Blob
- `POST /predict/automl` — Random Forest prediction + logs to Azure Blob
- `GET  /health` — model load status
- `GET  /docs` — Swagger UI

---

## Streamlit Dashboard

```bash
cd streamlit_app
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 \
AZURE_STORAGE_CONNECTION_STRING=<conn_str> \
streamlit run app.py
```

**Predict tab** — fill in 19 customer features, get side-by-side predictions from both models with churn probability and risk badge.

**Monitoring tab** — live metrics from Azure Blob inference logs:
- Endpoint health status
- Total predictions, churn rate, per-model volume
- Churn rate over time (line chart)
- Prediction volume by model (bar chart)
- Risk level distribution (Low / Medium / High)
- **Data drift detection** — PSI for categorical features, normalised mean-shift for numeric; alert if any feature exceeds 0.20 threshold
- Recent predictions table
- Model registry info

---

## Monitoring

### Application Insights (Infrastructure)
Already wired to Container Apps automatically. Open Azure Portal →
`mlopschurnws1585469262` → Application Dashboard to see:
- Request volume and latency per endpoint
- Failed request rate
- Server response time (P50/P95)

### Streamlit Monitoring Tab (ML Metrics)
- Reads inference logs from `inference-logs/` Azure Blob container
- Computes data drift vs `baseline/feature_baseline.json` (training distribution)
- Refreshes every 60 seconds (manual refresh button available)

### Drift Detection Method
| Feature Type | Method | Threshold |
|-------------|--------|-----------|
| Numeric (tenure, charges) | Normalised mean-shift: `|live_mean - train_mean| / train_std` | 0.20 |
| Categorical (contract, payment, etc.) | Population Stability Index (PSI) | 0.20 |

---

## Automated Retraining & Redeployment

| Time (UTC) | What happens |
|------------|-------------|
| 02:00 daily | Azure ML Schedule fires `daily-churn-retrain` pipeline |
| | → Ingest → Preprocess → Train XGBoost → Evaluate → Register new version |
| 03:00 daily | GitHub Actions `redeploy.yml` fires |
| | → Download latest XGBoost from Azure ML registry |
| | → Retrain Random Forest on latest raw data |
| | → Rebuild FastAPI Docker image with both fresh models |
| | → Redeploy to Azure Container Apps |
| | → Smoke test both endpoints |

To recreate the schedule:
```bash
python pipelines/create_retrain_schedule.py
```

---

## CI/CD Setup

1. Create GitHub secret `AZURE_CREDENTIALS`:
   ```bash
   az ad sp create-for-rbac --name churn-cicd --role contributor \
     --scopes /subscriptions/<sub-id>/resourceGroups/mlops-churn-rg \
     --sdk-auth
   ```
2. Add GitHub environment `production` with required reviewers (approval gate).
3. Push to `main` → Actions runs: lint → tests → approval → pipelines → smoke tests.

---

## Project Structure

```
churn_prediction/
├── data/
│   ├── raw/                            # Kaggle CSV
│   └── processed/                      # parquet splits + preprocessing artifacts
├── src/
│   ├── ingestion/ingest.py             # CSV → parquet → Azure ML Data Asset
│   ├── preprocessing/preprocess.py     # nulls, encoding, scaling, MLTable outputs
│   ├── training/
│   │   ├── train_baseline.py           # Logistic Regression baseline (Phase 6)
│   │   ├── train_manual.py             # XGBoost + MLflow + model registration
│   │   └── train_automl.py             # Azure AutoML job submission
│   ├── evaluation/
│   │   ├── evaluate.py                 # XGBoost candidate vs production
│   │   └── evaluate_automl.py          # AutoML evaluation + registration
│   ├── deployment/score.py             # Azure ML endpoint scoring script
│   └── monitoring/monitor.py           # Azure ML Monitor config (quota-dependent)
├── components/                         # Azure ML reusable components (YAML)
│   ├── train_baseline.yaml             # LR baseline component
│   ├── train_manual.yaml               # XGBoost component
│   └── ...
├── pipelines/
│   ├── manual_pipeline.yaml            # LR + XGBoost end-to-end pipeline
│   ├── automl_pipeline.yaml            # AutoML end-to-end pipeline
│   ├── retrain_pipeline.yaml           # Daily retraining pipeline
│   └── create_retrain_schedule.py      # Creates Azure ML Schedule (SDK)
├── fastapi_app/
│   ├── api.py                          # FastAPI — /predict/xgboost, /predict/automl
│   ├── models/                         # Bundled XGBoost + RF model files
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit_app/
│   ├── app.py                          # Predict tab + Monitoring tab with drift
│   ├── requirements.txt
│   └── Dockerfile
├── docs/
│   └── architecture.md                 # Full Mermaid architecture diagrams
├── deployments/                        # Azure ML endpoint/deployment YAMLs
├── configs/
├── .github/workflows/
│   ├── ci_cd.yaml                      # On push: test → approve → deploy → smoke
│   └── redeploy.yml                    # Daily 03:00 UTC model refresh + redeploy
├── tests/
└── requirements.txt
```

---

## Key Metrics

| Model | Val ROC-AUC | Val F1 | Role |
|-------|-------------|--------|------|
| Logistic Regression | 0.844 | 0.626 | Baseline reference |
| XGBoost | ~0.850 | ~0.630 | Production (Manual track) |
| Random Forest | 0.842 | 0.628 | Production (AutoML track) |

**Monitoring thresholds:** drift alert > 0.20 · retraining trigger: daily schedule
