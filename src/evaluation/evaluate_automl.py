"""
AutoML model evaluation: load the best AutoML mlflow model, evaluate on test set,
register it in Azure ML registry, and promote if AUC beats production model.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET = "Churn"
PROMOTE_THRESHOLD = 0.01


def get_ml_client(subscription_id, resource_group, workspace):
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def load_automl_model(model_path: str):
    """Load AutoML mlflow model, preferring sklearn flavor for predict_proba."""
    try:
        model = mlflow.sklearn.load_model(model_path)
        log.info("Loaded AutoML model via sklearn flavor")
        return model, "sklearn"
    except Exception:
        model = mlflow.pyfunc.load_model(model_path)
        log.info("Loaded AutoML model via pyfunc flavor")
        return model, "pyfunc"


def compute_metrics(model, flavor: str, X: pd.DataFrame, y: pd.Series) -> dict:
    if flavor == "sklearn":
        proba = model.predict_proba(X)[:, 1]
    else:
        # pyfunc predict returns a DataFrame or array
        result = model.predict(X)
        if isinstance(result, pd.DataFrame):
            # AutoML pyfunc may return a column named after the target or 'probability'
            prob_cols = [c for c in result.columns if "prob" in c.lower() or c == "1" or c == 1]
            proba = result[prob_cols[0]].values if prob_cols else result.iloc[:, -1].values
        else:
            proba = np.array(result).flatten()

    y_pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc":   float(roc_auc_score(y, proba)),
        "f1":        float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall":    float(recall_score(y, y_pred)),
    }


def update_model_tag(ml_client, model_name, version, key, value):
    model = ml_client.models.get(model_name, version=version)
    if model.tags is None:
        model.tags = {}
    model.tags[key] = value
    ml_client.models.create_or_update(model)
    log.info("Tag '%s'='%s' on %s v%s", key, value, model_name, version)


def main(args):
    # ── 1. Load test data ────────────────────────────────────────────────────
    test_df = pd.read_parquet(Path(args.splits_dir) / "test.parquet")
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    log.info("Test set: %d rows", len(test_df))

    # ── 2. Load AutoML best model ────────────────────────────────────────────
    model, flavor = load_automl_model(args.automl_model_path)
    candidate_metrics = compute_metrics(model, flavor, X_test, y_test)
    log.info("Candidate metrics: %s", candidate_metrics)

    # ── 3. Register model in AML registry ────────────────────────────────────
    try:
        ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)
        aml_model = Model(
            path=args.automl_model_path,
            type=AssetTypes.MLFLOW_MODEL,
            name=args.registered_model_name,
            tags={"stage": "candidate"},
            description="Best AutoML model for Telco churn prediction",
        )
        registered = ml_client.models.create_or_update(aml_model)
        log.info("Registered model '%s' v%s", args.registered_model_name, registered.version)
        registered_version = registered.version
    except Exception as exc:
        log.warning("Could not register model: %s", exc)
        ml_client = None
        registered_version = "1"

    # ── 4. Compare vs production ─────────────────────────────────────────────
    should_promote = True
    production_metrics = {k: 0.0 for k in candidate_metrics}

    if ml_client:
        try:
            versions = list(ml_client.models.list(name=args.registered_model_name))
            prod_versions = [v for v in versions if v.tags and v.tags.get("stage") == "production"]
            if prod_versions:
                latest_prod = max(prod_versions, key=lambda m: int(m.version))
                prod_path = Path(args.splits_dir).parent / "prod_model_download"
                ml_client.models.download(
                    name=args.registered_model_name,
                    version=latest_prod.version,
                    download_path=str(prod_path),
                )
                prod_model_path = next(prod_path.rglob("MLmodel"), None)
                if prod_model_path:
                    prod_model, prod_flavor = load_automl_model(str(prod_model_path.parent))
                    production_metrics = compute_metrics(prod_model, prod_flavor, X_test, y_test)
                    auc_delta = candidate_metrics["roc_auc"] - production_metrics["roc_auc"]
                    should_promote = auc_delta > PROMOTE_THRESHOLD
                    log.info("AUC delta: %.4f → %s", auc_delta, "PROMOTE" if should_promote else "REJECT")
        except Exception as exc:
            log.warning("Could not compare vs production (%s). Auto-promoting.", exc)

    # ── 5. Tag promoted model ─────────────────────────────────────────────────
    if should_promote and ml_client:
        try:
            versions = list(ml_client.models.list(name=args.registered_model_name))
            for old in [v for v in versions if v.tags and v.tags.get("stage") == "production"]:
                update_model_tag(ml_client, args.registered_model_name, old.version, "stage", "archived")
            update_model_tag(ml_client, args.registered_model_name, registered_version, "stage", "production")
        except Exception as exc:
            log.warning("Could not update model tags: %s", exc)

    # ── 6. Write evaluation report ────────────────────────────────────────────
    report = {
        "candidate_metrics":  candidate_metrics,
        "production_metrics": production_metrics,
        "promote":            should_promote,
        "registered_model":   args.registered_model_name,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "evaluation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report → %s", report_path)

    if args.fail_if_no_promotion and not should_promote:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir",            required=True)
    parser.add_argument("--automl-model-path",     required=True, help="Path to mlflow model folder")
    parser.add_argument("--registered-model-name", default="telco-churn-automl")
    parser.add_argument("--subscription-id",       default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",        default="mlops-churn-rg")
    parser.add_argument("--workspace",             default="mlops-churn-ws")
    parser.add_argument("--output-dir",            default="outputs")
    parser.add_argument("--fail-if-no-promotion",  action="store_true")
    main(parser.parse_args())
