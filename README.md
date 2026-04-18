# Telco Customer Churn Prediction — End-to-End MLOps on Azure ML

Capstone project implementing a full MLOps lifecycle with **two parallel deployment tracks**:

| Track | Model | Endpoint |
|-------|-------|----------|
| Manual | XGBoost (hand-tuned) | `churn-manual-endpoint` |
| AutoML | Best Azure AutoML model | `churn-automl-endpoint` |

---

## Prerequisites

- Azure subscription `bc906f50-e57d-4464-bfb5-5285937d2b4a`
- Azure ML Workspace `mlops-churn-ws` in resource group `mlops-churn-rg`
- Compute cluster named `cpu-cluster` (Standard_DS3_v2, min 0 / max 4 nodes)
- Python 3.10 + `pip install -r requirements.txt`
- Azure CLI with ML extension: `az extension add --name ml`
- Kaggle Telco dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`

---

## Quick Start (run locally, Week 1)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest and validate (no Azure registration needed locally)
python src/ingestion/ingest.py \
  --input-csv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --no-register

# 3. Preprocess (creates train/val/test parquet splits)
python src/preprocessing/preprocess.py

# 4. Train XGBoost locally (MLflow logs to ./mlruns)
python src/training/train_manual.py \
  --splits-dir data/processed/splits \
  --model-dir outputs/manual_model

# 5. Run unit tests
pytest tests/ -v
```

---

## Azure ML Execution (Week 2+)

### Set environment variables

```bash
export AZURE_SUBSCRIPTION_ID=bc906f50-e57d-4464-bfb5-5285937d2b4a
export AZURE_TENANT_ID=<your-tenant-id>
export MLFLOW_TRACKING_URI=$(az ml workspace show \
  --name mlops-churn-ws --resource-group mlops-churn-rg \
  --query mlFlowTrackingUri -o tsv)
```

### Run the Manual Pipeline

```bash
az ml job create \
  --file pipelines/manual_pipeline.yaml \
  --workspace-name mlops-churn-ws \
  --resource-group mlops-churn-rg \
  --stream
```

### Run the AutoML Pipeline

```bash
az ml job create \
  --file pipelines/automl_pipeline.yaml \
  --workspace-name mlops-churn-ws \
  --resource-group mlops-churn-rg \
  --stream
```

### Deploy endpoints (if not using pipelines)

```bash
# Manual XGBoost endpoint
python src/deployment/deploy_manual.py

# AutoML endpoint
python src/deployment/deploy_automl.py \
  --job-name-file outputs/automl_job_name.txt
```

### Configure monitoring

```bash
python src/monitoring/monitor.py \
  --baseline-dataset-id azureml:telco-churn-curated:1
```

---

## CI/CD Setup

1. Create a GitHub secret `AZURE_CREDENTIALS` (JSON from `az ad sp create-for-rbac`)
2. Push to `main` — GitHub Actions runs tests → deploys both pipelines → smoke tests endpoints

---

## Project Structure

```
mlops-churn-project/
├── data/
│   ├── raw/                    # drop Kaggle CSV here
│   └── processed/              # parquet files + preprocessing artifacts
├── src/
│   ├── ingestion/ingest.py     # CSV → parquet → Azure ML Data Asset
│   ├── preprocessing/preprocess.py  # nulls, encoding, scaling, splits
│   ├── training/
│   │   ├── train_manual.py     # XGBoost + MLflow logging
│   │   └── train_automl.py     # Azure AutoML job submission
│   ├── evaluation/evaluate.py  # candidate vs production comparison
│   ├── deployment/
│   │   ├── deploy_manual.py    # XGBoost → Managed Online Endpoint
│   │   └── deploy_automl.py    # AutoML best model → Managed Online Endpoint
│   └── monitoring/monitor.py   # data drift + prediction drift monitors
├── pipelines/
│   ├── manual_pipeline.yaml    # Azure ML pipeline (manual track)
│   └── automl_pipeline.yaml    # Azure ML pipeline (AutoML track)
├── configs/
│   ├── automl_config.yaml      # AutoML job parameters
│   └── endpoint_config.yaml    # Endpoint/deployment settings
├── .github/workflows/ci_cd.yaml
├── tests/
└── requirements.txt
```

---

## Key Metrics

- **Primary**: ROC-AUC (AUC_weighted for AutoML)
- **Secondary**: F1 score, Precision, Recall
- **Promotion threshold**: AUC improvement > 1% over production model
- **Monitoring alerts**: data drift JSdivergence > 0.2, daily schedule at 06:00 UTC
```
