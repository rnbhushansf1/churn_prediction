"""
Manual training: XGBoost classifier on preprocessed train/val splits.
Logs all parameters, metrics, and artifacts to MLflow (Azure ML experiment).
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET = "Churn"

# ── Default XGBoost hyperparameters ─────────────────────────────────────────
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,   # handles class imbalance (~27 % churn)
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> str:
    """Save confusion matrix heatmap and return file path."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def main(args: argparse.Namespace) -> None:
    # ── 1. Connect to Azure ML experiment via MLflow ─────────────────────────
    # MLFLOW_TRACKING_URI is set automatically inside Azure ML jobs.
    # For local runs, set it to your AzureML workspace tracking URI.
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # ── 2. Load preprocessed splits ──────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df   = pd.read_parquet(splits_dir / "val.parquet")
    log.info("Train size: %d  Val size: %d", len(train_df), len(val_df))

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_val,   y_val   = val_df.drop(columns=[TARGET]),   val_df[TARGET]

    # ── 3. Build hyperparameter dict (allow CLI overrides) ───────────────────
    params = DEFAULT_PARAMS.copy()
    if args.n_estimators:
        params["n_estimators"] = args.n_estimators
    if args.max_depth:
        params["max_depth"] = args.max_depth
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate

    # ── 4. Train with MLflow autolog ─────────────────────────────────────────
    mlflow.xgboost.autolog(log_models=False)   # we log manually for full control

    with mlflow.start_run(run_name="xgboost-manual") as run:
        log.info("MLflow run ID: %s", run.info.run_id)

        # Log hyperparameters explicitly
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size",   len(val_df))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # ── 5. Evaluate on validation set ───────────────────────────────────
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred       = model.predict(X_val)

        roc_auc   = roc_auc_score(y_val, y_pred_proba)
        f1        = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall    = recall_score(y_val, y_pred)
        cm        = confusion_matrix(y_val, y_pred)

        log.info("ROC-AUC: %.4f  F1: %.4f  Precision: %.4f  Recall: %.4f",
                 roc_auc, f1, precision, recall)
        log.info("\n%s", classification_report(y_val, y_pred, target_names=["No Churn", "Churn"]))

        # Log metrics
        mlflow.log_metrics({
            "val_roc_auc":   roc_auc,
            "val_f1":        f1,
            "val_precision": precision,
            "val_recall":    recall,
        })

        # ── 6. Log confusion matrix as artifact ─────────────────────────────
        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(cm_path)

        # ── 7. Save and log model artifact ───────────────────────────────────
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Log model with signature for deployment
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict_proba(X_train))

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost-churn-model",
            signature=signature,
            registered_model_name=args.registered_model_name,
        )
        log.info("Model registered as '%s'", args.registered_model_name)

        # Also save locally for the evaluation step
        model.save_model(str(model_dir / "model.json"))
        log.info("Model saved locally to %s/model.json", model_dir)

        # Write run_id to file so the evaluation step can reference it
        with open(model_dir / "run_id.txt", "w") as fh:
            fh.write(run.info.run_id)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost churn model")
    parser.add_argument("--splits-dir",   default="data/processed/splits")
    parser.add_argument("--model-dir",    default="outputs/manual_model")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--experiment-name", default="telco-churn-manual")
    parser.add_argument("--registered-model-name", default="telco-churn-xgboost")
    parser.add_argument("--n-estimators",  type=int,   default=None)
    parser.add_argument("--max-depth",     type=int,   default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    main(parser.parse_args())
