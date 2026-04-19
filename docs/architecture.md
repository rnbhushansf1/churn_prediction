# Project Architecture — Telco Churn Prediction MLOps

## 1. End-to-End System Overview

```mermaid
flowchart TD
    subgraph SOURCE["Data Source"]
        CSV["Kaggle Telco CSV\nWA_Fn-UseC_-Telco-Customer-Churn.csv"]
    end

    subgraph STORAGE["Azure Blob Storage"]
        RAW["Raw Data\ntelco-churn-raw:1"]
        SPLITS["Processed Splits\ntrain / val / test parquet"]
        LOGS["Inference Logs\ninference-logs/\nxgboost/YYYY/MM/DD/predictions.jsonl\nautoml/YYYY/MM/DD/predictions.jsonl"]
        BASELINE["Drift Baseline\nbaseline/feature_baseline.json"]
    end

    subgraph AML["Azure Machine Learning Workspace"]
        INGEST["Ingest Component\ntelco_ingest:3\nSchema validation + null checks"]
        PREPROCESS["Preprocess Component\ntelco_preprocess:4\nEncode + Scale + Split + MLTable"]

        subgraph TRAIN["Training — Two Tracks"]
            LR["Logistic Regression\nBaseline\ntrain_baseline.py\nROC-AUC ~0.84"]
            XGB["XGBoost\nAdvanced Model\ntrain_manual.py\nROC-AUC ~0.85"]
            RF["Random Forest\nAutoML Stand-in\nROC-AUC 0.842"]
        end

        EVAL_XGB["Evaluate XGBoost\nConfusion matrix\nROC-AUC, F1, Precision, Recall\nCandidate vs Production"]
        EVAL_RF["Evaluate AutoML\nevaluate_automl.py"]

        REGISTRY["Azure ML Model Registry\ntelco-churn-xgboost v1 → Production\ntelco-churn-automl"]

        SCHEDULE["Azure ML Schedule\ndaily-churn-retrain\n02:00 UTC daily"]
    end

    subgraph SERVE["Serving Layer — Azure Container Apps"]
        FASTAPI["FastAPI Scoring Service\nchurn-api\n/predict/xgboost\n/predict/automl\n/health  /docs"]
        STREAMLIT["Streamlit Dashboard\nchurn-dashboard\nPredict tab + Monitoring tab"]
    end

    subgraph MONITOR["Monitoring"]
        APPINSIGHTS["Application Insights\nLatency · Failures\nRequest volume · Errors"]
        DRIFT["Streamlit Drift Detection\nNumeric: mean-shift\nCategorical: PSI\nThreshold: 0.20"]
    end

    subgraph CICD["CI/CD — GitHub Actions"]
        CI["ci_cd.yaml\nOn push to main:\nLint → Test → Deploy pipelines\nApproval gate → Smoke test"]
        REDEPLOY["redeploy.yml\n03:00 UTC daily:\nDownload XGBoost from registry\nRetrain RF on latest data\nRebuild Docker → Redeploy"]
    end

    CSV --> RAW
    RAW --> INGEST
    INGEST --> PREPROCESS
    PREPROCESS --> SPLITS
    SPLITS --> LR
    SPLITS --> XGB
    SPLITS --> RF
    XGB --> EVAL_XGB
    RF --> EVAL_RF
    EVAL_XGB --> REGISTRY
    EVAL_RF --> REGISTRY
    REGISTRY --> FASTAPI
    FASTAPI --> STREAMLIT
    FASTAPI --> LOGS
    LOGS --> DRIFT
    BASELINE --> DRIFT
    APPINSIGHTS --> FASTAPI
    SCHEDULE --> INGEST
    REDEPLOY --> REGISTRY
    REDEPLOY --> FASTAPI
    CI --> AML
```

---

## 2. Data Flow

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        CSV2["Raw CSV\n7,043 customers\n21 features"]
    end

    subgraph PIPELINE["Azure ML Pipeline"]
        direction TB
        A["Ingest\n• Schema check\n• Null threshold\n• Parquet output"]
        B["Preprocess\n• Fill nulls\n• Label encode\n• StandardScaler\n• 70/15/15 split"]
        A --> B
    end

    subgraph OUTPUTS["Outputs"]
        TRAIN2["train.parquet\n~4,930 rows"]
        VAL["val.parquet\n~1,056 rows"]
        TEST["test.parquet\n~1,057 rows"]
        MLTABLE["MLTable dirs\nfor AutoML input"]
        ARTIFACTS["artifacts/\nlabel_encoders.pkl\nscaler.pkl\nnull_fills.pkl"]
    end

    subgraph MODELS["Models Trained"]
        M1["Logistic Regression\nBaseline reference\nAUC 0.844"]
        M2["XGBoost\nProduction model\nAUC ~0.85"]
        M3["Random Forest\nAutoML track\nAUC 0.842"]
    end

    CSV2 --> PIPELINE
    B --> TRAIN2 & VAL & TEST & MLTABLE & ARTIFACTS
    TRAIN2 --> M1 & M2 & M3
    VAL --> M1 & M2 & M3
```

---

## 3. Training Tracks

```mermaid
flowchart TD
    SPLITS2["Preprocessed Splits\ntrain / val / test"]

    subgraph MANUAL["Manual Track — XGBoost"]
        LR2["Step 1: Logistic Regression Baseline\n• C=1.0, class_weight=balanced\n• AUC 0.844, F1 0.626\n• Logged to MLflow"]
        XGB2["Step 2: XGBoost\n• n_estimators=300, max_depth=6\n• learning_rate=0.05, scale_pos_weight=3\n• AUC ~0.85\n• MLflow autolog"]
        EVALX["Step 3: Evaluate\n• Confusion matrix\n• Compare vs production\n• Register if improved"]
        LR2 --> XGB2 --> EVALX
    end

    subgraph AUTOML["AutoML Track — Random Forest"]
        RF2["Random Forest\n• n_estimators=300, max_depth=8\n• class_weight=balanced\n• AUC 0.842, F1 0.628\n• MLflow sklearn model"]
        EVALA["Evaluate\n• Same metrics\n• Register as telco-churn-automl"]
        RF2 --> EVALA
    end

    subgraph REG["Azure ML Model Registry"]
        XGB_REG["telco-churn-xgboost\nv1 → Stage: Production\ntags: framework=xgboost\n      dataset_version=v1"]
        RF_REG["telco-churn-automl\nv1 → Stage: Production\ntags: framework=randomforest"]
    end

    SPLITS2 --> MANUAL
    SPLITS2 --> AUTOML
    EVALX --> XGB_REG
    EVALA --> RF_REG
```

---

## 4. Serving Architecture

```mermaid
flowchart LR
    USER["User / Browser"]

    subgraph DASH["Streamlit Dashboard\nchurn-dashboard.mangobeach-a2290557\n.southeastasia.azurecontainerapps.io"]
        TAB1["Predict Tab\n• 19 feature inputs\n• Dropdowns + sliders\n• Side-by-side model results"]
        TAB2["Monitoring Tab\n• Endpoint health\n• Churn rate over time\n• Prediction volume\n• Risk distribution\n• Data drift chart\n• Recent predictions table\n• Model registry info"]
    end

    subgraph API["FastAPI\nchurn-api.mangobeach-a2290557\n.southeastasia.azurecontainerapps.io"]
        EP1["POST /predict/xgboost\nXGBoost inference\n+ Blob logging"]
        EP2["POST /predict/automl\nRandom Forest inference\n+ Blob logging"]
        EP3["GET /health\nModels loaded status"]
        EP4["GET /docs\nSwagger UI"]
    end

    subgraph BLOB["Azure Blob Storage"]
        ILOG["inference-logs/\nxgboost/YYYY/MM/DD/predictions.jsonl\nautoml/YYYY/MM/DD/predictions.jsonl"]
        BASE2["baseline/feature_baseline.json\nTraining distribution stats"]
    end

    USER --> TAB1
    USER --> TAB2
    TAB1 -->|"POST JSON payload"| EP1
    TAB1 -->|"POST JSON payload"| EP2
    TAB2 -->|"GET /health"| EP3
    TAB2 -->|"Reads logs"| ILOG
    TAB2 -->|"Reads baseline"| BASE2
    EP1 -->|"Log prediction"| ILOG
    EP2 -->|"Log prediction"| ILOG
```

---

## 5. Monitoring Architecture

```mermaid
flowchart TD
    subgraph INFRA["Infrastructure Monitoring\nApplication Insights — mlopschurnws1585469262"]
        AI1["Request Volume\nPer endpoint per minute"]
        AI2["Server Response Time\nP50 / P95 latency"]
        AI3["Failed Requests\nHTTP 4xx / 5xx rate"]
        AI4["Availability\nEndpoint up/down"]
    end

    subgraph MLMON["ML Monitoring — Streamlit Monitoring Tab"]
        subgraph PERF["Prediction Performance"]
            P1["Total predictions"]
            P2["Churn rate over time"]
            P3["XGBoost vs AutoML volume"]
            P4["Risk level distribution\nLow / Medium / High"]
        end

        subgraph DRIFT["Data Drift Detection"]
            D1["Numeric Features\nNormalised mean-shift\n|live_mean - train_mean| / train_std\nFeatures: tenure, MonthlyCharges\n         TotalCharges, SeniorCitizen"]
            D2["Categorical Features\nPopulation Stability Index (PSI)\nΣ (live_p - base_p) × log(live_p/base_p)\nFeatures: Contract, PaymentMethod\n          InternetService, etc."]
            D3["Alert Banner\n⚠️ Drifted if score > 0.20\n✅ OK otherwise"]
            D1 --> D3
            D2 --> D3
        end
    end

    FASTAPI2["FastAPI\nAll requests"] --> INFRA
    FASTAPI2 --> BLOB2["Azure Blob\ninference-logs"]
    BLOB2 --> MLMON
    BASELINE2["Baseline Stats\nfeature_baseline.json"] --> DRIFT
```

---

## 6. CI/CD & Automation Flow

```mermaid
flowchart TD
    subgraph TRIGGER["Triggers"]
        PUSH["Git push to main"]
        SCHED1["Azure ML Schedule\n02:00 UTC daily"]
        SCHED2["GitHub Actions Schedule\n03:00 UTC daily"]
        MANUAL["Manual workflow_dispatch"]
    end

    subgraph CICD2["ci_cd.yaml — On Push to Main"]
        T["Lint + Unit Tests\nflake8 + pytest"]
        AP["Approval Gate\nGitHub 'production' environment\nManual reviewer required"]
        DP1["Submit Manual Pipeline\nAzure ML Job"]
        DP2["Submit AutoML Pipeline\nAzure ML Job"]
        SM["Smoke Tests\nBoth endpoints"]
        T --> AP --> DP1 & DP2 --> SM
    end

    subgraph RETRAIN["Azure ML — Daily Retrain (02:00 UTC)"]
        R1["Ingest latest data"]
        R2["Preprocess"]
        R3["Train XGBoost"]
        R4["Evaluate + Register\ntelco-churn-xgboost vN"]
        R1 --> R2 --> R3 --> R4
    end

    subgraph REDEPLOY2["redeploy.yml — Daily Redeploy (03:00 UTC)"]
        RD1["Download latest XGBoost\nfrom Azure ML registry"]
        RD2["Download raw data\nRetrain RF model"]
        RD3["az acr build\nNew Docker image"]
        RD4["az containerapp update\n+ restart"]
        RD5["Smoke test\nboth endpoints"]
        RD1 --> RD2 --> RD3 --> RD4 --> RD5
    end

    PUSH --> CICD2
    MANUAL --> CICD2
    SCHED1 --> RETRAIN
    RETRAIN --> REDEPLOY2
    SCHED2 --> REDEPLOY2
```

---

## 7. Key Metrics Summary

| Metric | Logistic Regression | XGBoost | Random Forest (AutoML) |
|--------|--------------------|---------|-----------------------|
| Val ROC-AUC | 0.844 | ~0.850 | 0.842 |
| Val F1 | 0.626 | ~0.630 | 0.628 |
| Val Precision | 0.504 | ~0.55 | ~0.52 |
| Val Recall | 0.825 | ~0.70 | ~0.72 |
| Role | Baseline reference | Production | AutoML track |

## 8. Live URLs

| Service | URL |
|---------|-----|
| Streamlit Dashboard | https://churn-dashboard.mangobeach-a2290557.southeastasia.azurecontainerapps.io |
| FastAPI + Swagger | https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io/docs |
| FastAPI Health | https://churn-api.mangobeach-a2290557.southeastasia.azurecontainerapps.io/health |
