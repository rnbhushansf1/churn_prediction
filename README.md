# Telco Customer Churn Prediction — End-to-End MLOps on Azure ML

Capstone project implementing a full MLOps lifecycle on Azure ML with **two parallel model tracks**, a FastAPI scoring service, and a Streamlit dashboard — all hosted on Azure Container Apps.

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | https://churn-dashboard.mangobeach-a2290557.southeastasia.azurecontainerapps.io |
| FastAPI + Swagger UI | https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io/docs |

## Architecture

```
Kaggle CSV → Azure Blob → Ingestion → Preprocessing
                                           │
                          ┌────────────────┴────────────────┐
                     Manual Track                      AutoML Track
                   (XGBoost, tuned)             (Azure AutoML, best model)
                          │                                  │
                     MLflow Registry                  MLflow Registry
                          │                                  │
                          └────────────┬────────────────────┘
                                  FastAPI App
                              (Azure Container Apps)
                                       │
                              Streamlit Dashboard
                              (Azure Container Apps)
```

**CI/CD**: GitHub Actions → runs tests → submits both pipelines → approval gate → deploy  
**Retraining**: Automated daily schedule (`daily-churn-retrain`) at 02:00 UTC via Azure ML Schedule

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

# Train XGBoost locally (MLflow logs to ./mlruns)
python src/training/train_manual.py \
  --splits-dir data/processed/splits \
  --model-dir outputs/manual_model

# Run unit tests
pytest tests/ -v
```

---

## Azure ML Pipelines

```bash
# Set workspace env vars
export AZURE_SUBSCRIPTION_ID=bc906f50-e57d-4464-bfb5-5285937d2b4a

# Manual (XGBoost) pipeline
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

Serves both models locally or via Docker:

```bash
# Local
cd fastapi_app
pip install -r requirements.txt
uvicorn api:app --reload

# Docker
docker build -t churn-api fastapi_app/
docker run -p 8000:8000 churn-api
```

Endpoints:
- `POST /predict/xgboost` — XGBoost prediction
- `POST /predict/automl` — AutoML prediction
- `GET  /health` — model load status
- `GET  /docs` — Swagger UI

---

## Streamlit Dashboard

```bash
cd streamlit_app
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 streamlit run app.py
```

---

## CI/CD Setup

1. Create GitHub secret `AZURE_CREDENTIALS`:
   ```bash
   az ad sp create-for-rbac --name churn-cicd --role contributor \
     --scopes /subscriptions/<sub-id>/resourceGroups/mlops-churn-rg \
     --sdk-auth
   ```
2. Add a GitHub environment named `production` with required reviewers for deployment approval gate.
3. Push to `main` — Actions runs: tests → manual pipeline → AutoML pipeline → smoke tests.

---

## Retraining Schedule

A daily schedule (`daily-churn-retrain`) runs the full retrain pipeline at 02:00 UTC. To recreate:

```bash
python pipelines/create_retrain_schedule.py
```

---

## Project Structure

```
churn_prediction/
├── data/
│   ├── raw/                          # drop Kaggle CSV here
│   └── processed/                    # parquet splits + preprocessing artifacts
├── src/
│   ├── ingestion/ingest.py           # CSV → parquet → Azure ML Data Asset
│   ├── preprocessing/preprocess.py   # nulls, encoding, scaling, MLTable outputs
│   ├── training/
│   │   ├── train_manual.py           # XGBoost + MLflow autolog + model registration
│   │   └── train_automl.py           # AutoML job submission
│   ├── evaluation/
│   │   ├── evaluate.py               # XGBoost candidate vs production comparison
│   │   └── evaluate_automl.py        # AutoML evaluation + registration
│   ├── deployment/score.py           # Azure ML endpoint scoring script
│   └── monitoring/monitor.py         # data drift + prediction drift monitors
├── components/                       # Azure ML reusable components (YAML)
├── pipelines/
│   ├── manual_pipeline.yaml          # XGBoost end-to-end pipeline
│   ├── automl_pipeline.yaml          # AutoML end-to-end pipeline
│   ├── retrain_pipeline.yaml         # Daily retraining pipeline
│   └── create_retrain_schedule.py    # Creates Azure ML Schedule (SDK)
├── fastapi_app/
│   ├── api.py                        # FastAPI app (both models)
│   ├── models/                       # Bundled model files
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit_app/
│   ├── app.py                        # Streamlit dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── deployments/                      # Azure ML endpoint/deployment YAMLs
├── configs/
│   ├── automl_config.yaml
│   └── endpoint_config.yaml
├── .github/workflows/ci_cd.yaml      # GitHub Actions CI/CD
├── tests/
└── requirements.txt
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary metric (AUC_weighted for AutoML) |
| F1 Score | Secondary metric |
| Precision / Recall | Tracked per run |
| Promotion threshold | AUC improvement > 1% over production |
| Drift alert | JS-divergence > 0.2 on daily monitor |
