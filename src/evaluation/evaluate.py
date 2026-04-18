"""
Model evaluation: compare a candidate model (new training run) against the
currently registered production model. Promote if AUC improves by > 1 %.
Works for both the manual XGBoost track and the AutoML track.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET = "Churn"
PROMOTE_THRESHOLD = 0.01   # promote if AUC gain > 1 percentage point


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def load_candidate_model(model_dir: str) -> xgb.XGBClassifier:
    """Load XGBoost model from local model.json saved by train step."""
    model = xgb.XGBClassifier()
    model.load_model(str(Path(model_dir) / "model.json"))
    log.info("Loaded candidate model from %s/model.json", model_dir)
    return model


def load_production_model_aml(ml_client: MLClient, registered_name: str, model_dir: str):
    """Download and load the production-tagged model from Azure ML registry."""
    try:
        versions = list(ml_client.models.list(name=registered_name))
        prod_versions = [v for v in versions if v.tags and v.tags.get("stage") == "production"]
        if not prod_versions:
            log.warning("No production model found in registry. Will auto-promote candidate.")
            return None
        latest_prod = max(prod_versions, key=lambda m: int(m.version))
        log.info("Found production model: %s v%s", registered_name, latest_prod.version)

        # Download model file
        prod_dir = Path(model_dir) / "production_model"
        ml_client.models.download(name=registered_name, version=latest_prod.version, download_path=str(prod_dir))
        model_file = next(prod_dir.rglob("model.json"), None)
        if model_file is None:
            log.warning("Could not find model.json in downloaded production model.")
            return None
        prod_model = xgb.XGBClassifier()
        prod_model.load_model(str(model_file))
        return prod_model
    except Exception as exc:
        log.warning("Could not load production model (%s). Will auto-promote candidate.", exc)
        return None


def compute_metrics(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate an XGBoost model and return a metrics dict."""
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc":   roc_auc_score(y, proba),
        "f1":        f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall":    recall_score(y, y_pred),
    }


def update_model_tag(
    ml_client: MLClient,
    model_name: str,
    version: str,
    tag_key: str,
    tag_value: str,
) -> None:
    """Update a tag on a registered model version."""
    model = ml_client.models.get(model_name, version=version)
    if model.tags is None:
        model.tags = {}
    model.tags[tag_key] = tag_value
    ml_client.models.create_or_update(model)
    log.info("Set tag '%s'='%s' on model '%s' v%s", tag_key, tag_value, model_name, version)


def main(args: argparse.Namespace) -> None:
    # ── 1. Load test split ────────────────────────────────────────────────────
    test_df = pd.read_parquet(Path(args.splits_dir) / "test.parquet")
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]
    log.info("Test set: %d rows", len(test_df))

    # ── 2. Load candidate model from local model.json ────────────────────────
    candidate_model = load_candidate_model(args.model_dir)
    candidate_metrics = compute_metrics(candidate_model, X_test, y_test)
    log.info("Candidate metrics: %s", candidate_metrics)

    # ── 3. Load production model (may not exist on first run) ─────────────────
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)
    production_model = load_production_model_aml(ml_client, args.registered_model_name, args.model_dir)

    if production_model is None:
        # First-ever run — promote candidate automatically
        should_promote = True
        production_metrics = {k: 0.0 for k in candidate_metrics}
        log.info("No production model exists. Auto-promoting candidate.")
    else:
        production_metrics = compute_metrics(production_model, X_test, y_test)
        log.info("Production metrics: %s", production_metrics)
        auc_delta = candidate_metrics["roc_auc"] - production_metrics["roc_auc"]
        should_promote = auc_delta > PROMOTE_THRESHOLD
        log.info(
            "AUC delta: %.4f (threshold: %.4f) → %s",
            auc_delta, PROMOTE_THRESHOLD,
            "PROMOTE" if should_promote else "REJECT",
        )

    # ── 4. Write evaluation report ─────────────────────────────────────────
    report = {
        "candidate_metrics":   candidate_metrics,
        "production_metrics":  production_metrics,
        "promote":             should_promote,
        "registered_model":    args.registered_model_name,
    }
    report_path = Path(args.output_dir) / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Evaluation report saved to %s", report_path)

    # ── 5. Tag model version if promoting ─────────────────────────────────────
    if should_promote:
        try:
            versions = list(ml_client.models.list(name=args.registered_model_name))
            if versions:
                latest = max(versions, key=lambda m: int(m.version))
                try:
                    old_prod = [m for m in versions if m.tags and m.tags.get("stage") == "production"]
                    for old in old_prod:
                        update_model_tag(ml_client, args.registered_model_name, old.version, "stage", "archived")
                except Exception as exc:
                    log.warning("Could not demote old production model: %s", exc)
                update_model_tag(ml_client, args.registered_model_name, latest.version, "stage", "production")
                log.info("Model '%s' v%s promoted to production.", args.registered_model_name, latest.version)
            else:
                log.warning("No registered model versions found to tag.")
        except Exception as exc:
            log.warning("Could not tag model version (will still write report): %s", exc)
    else:
        log.info("Candidate not promoted. Production model unchanged.")

    # Exit with non-zero code if not promoting (useful in CI to gate deployment)
    if args.fail_if_no_promotion and not should_promote:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and optionally promote churn model")
    parser.add_argument("--splits-dir",              default="data/processed/splits")
    parser.add_argument("--model-dir",               required=True, help="Folder containing model.json from train step")
    parser.add_argument("--registered-model-name",   default="telco-churn-xgboost")
    parser.add_argument("--subscription-id",         default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",          default="mlops-churn-rg")
    parser.add_argument("--workspace",               default="mlops-churn-ws")
    parser.add_argument("--output-dir",              default="outputs")
    parser.add_argument("--fail-if-no-promotion",    action="store_true")
    main(parser.parse_args())
