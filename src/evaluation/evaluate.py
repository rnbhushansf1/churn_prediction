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


def load_model_from_run(tracking_uri: str, run_id: str, artifact_path: str):
    """Load an MLflow model from a specific run."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    log.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def load_production_model(registered_name: str, tracking_uri: str):
    """Load the model tagged 'production' from the MLflow model registry."""
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model_uri = f"models:/{registered_name}/production"
        log.info("Loading production model: %s", model_uri)
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        log.warning("No production model found (%s). Will auto-promote candidate.", exc)
        return None


def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate a pyfunc model and return a metrics dict."""
    preds = model.predict(X)

    # pyfunc.predict returns class labels by default; handle proba if available
    if hasattr(preds, "columns"):
        # DataFrame output (e.g., AutoML) — assume column '1' is churn probability
        proba = preds.get("1", preds.iloc[:, 1]).values
    else:
        proba = preds.astype(float)

    # Binarize predictions at 0.5 threshold
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

    # ── 2. Load candidate model ───────────────────────────────────────────────
    candidate_model = load_model_from_run(
        args.tracking_uri,
        args.candidate_run_id,
        args.candidate_artifact_path,
    )
    candidate_metrics = compute_metrics(candidate_model, X_test, y_test)
    log.info("Candidate metrics: %s", candidate_metrics)

    # ── 3. Load production model (may not exist on first run) ─────────────────
    production_model = load_production_model(args.registered_model_name, args.tracking_uri)

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
        "candidate_run_id":    args.candidate_run_id,
        "registered_model":    args.registered_model_name,
    }
    report_path = Path(args.output_dir) / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Evaluation report saved to %s", report_path)

    # ── 5. Tag model version if promoting ─────────────────────────────────────
    if should_promote:
        ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)

        # Retrieve latest version of the candidate model
        versions = ml_client.models.list(name=args.registered_model_name)
        latest = max(versions, key=lambda m: int(m.version))

        # Update tags: demote old production → staging, promote new → production
        try:
            old_prod_versions = [
                m for m in ml_client.models.list(name=args.registered_model_name)
                if m.tags and m.tags.get("stage") == "production"
            ]
            for old in old_prod_versions:
                update_model_tag(ml_client, args.registered_model_name, old.version, "stage", "archived")
        except Exception as exc:
            log.warning("Could not demote old production model: %s", exc)

        update_model_tag(ml_client, args.registered_model_name, latest.version, "stage", "production")
        log.info("Model '%s' v%s promoted to production.", args.registered_model_name, latest.version)
    else:
        log.info("Candidate not promoted. Production model unchanged.")

    # Exit with non-zero code if not promoting (useful in CI to gate deployment)
    if args.fail_if_no_promotion and not should_promote:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and optionally promote churn model")
    parser.add_argument("--splits-dir",              default="data/processed/splits")
    parser.add_argument("--tracking-uri",            default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--candidate-run-id",        required=True)
    parser.add_argument("--candidate-artifact-path", default="xgboost-churn-model")
    parser.add_argument("--registered-model-name",   default="telco-churn-xgboost")
    parser.add_argument("--subscription-id",         default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",          default="mlops-churn-rg")
    parser.add_argument("--workspace",               default="mlops-churn-ws")
    parser.add_argument("--output-dir",              default="outputs")
    parser.add_argument("--fail-if-no-promotion",    action="store_true")
    main(parser.parse_args())
