"""
Baseline training: Logistic Regression on preprocessed train/val splits.
Logged to MLflow as a reference point for comparing advanced models.
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET = "Churn"


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix — Logistic Regression",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def main(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    splits_dir = Path(args.splits_dir)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df   = pd.read_parquet(splits_dir / "val.parquet")
    log.info("Train size: %d  Val size: %d", len(train_df), len(val_df))

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_val,   y_val   = val_df.drop(columns=[TARGET]),   val_df[TARGET]

    params = {
        "C": args.C,
        "max_iter": args.max_iter,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": 42,
        "n_jobs": -1,
    }

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="logistic-regression-baseline") as run:
        log.info("MLflow run ID: %s", run.info.run_id)
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size",   len(val_df))
        mlflow.log_param("n_features", X_train.shape[1])

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

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

        mlflow.log_metrics({
            "val_roc_auc":   roc_auc,
            "val_f1":        f1,
            "val_precision": precision,
            "val_recall":    recall,
        })

        cm_path = "confusion_matrix_baseline.png"
        plot_confusion_matrix(cm, cm_path)
        try:
            mlflow.log_artifact(cm_path)
        except Exception as e:
            log.warning("Could not log confusion matrix: %s", e)

        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(model, str(model_dir / "baseline_model"))
        log.info("Baseline model saved to %s", model_dir)

    log.info("Baseline training complete. ROC-AUC=%.4f (XGBoost target: ~0.85+)", roc_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression baseline")
    parser.add_argument("--splits-dir",     default="data/processed/splits")
    parser.add_argument("--model-dir",      default="outputs/baseline_model")
    parser.add_argument("--tracking-uri",   default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--experiment-name", default="telco-churn-manual")
    parser.add_argument("--C",              type=float, default=1.0)
    parser.add_argument("--max-iter",       type=int,   default=1000)
    main(parser.parse_args())
