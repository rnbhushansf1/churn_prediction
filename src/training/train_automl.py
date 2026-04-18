"""
AutoML training: configure and submit an Azure ML AutoML classification job
targeting 'Churn', primary metric AUC_weighted, max 20 trials, 60-min timeout.
"""

import argparse
import logging
import os
import time

from azure.ai.ml import MLClient, Input
from azure.ai.ml.automl import classification
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    """Build and return an authenticated Azure ML client."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def build_automl_job(
    training_data_asset: str,
    validation_data_asset: str,
    compute_target: str,
    experiment_name: str,
    max_trials: int,
    timeout_minutes: int,
):
    """
    Construct an AutoML classification job.
    Azure ML AutoML handles featurization, algorithm search, and ensembling automatically.
    """
    # Reference registered Data Assets as inputs
    training_data   = Input(type=AssetTypes.MLTABLE, path=training_data_asset)
    validation_data = Input(type=AssetTypes.MLTABLE, path=validation_data_asset)

    classification_job = classification(
        compute=compute_target,
        experiment_name=experiment_name,
        training_data=training_data,
        validation_data=validation_data,
        target_column_name="Churn",
        primary_metric="AUC_weighted",

        # ── Featurization ────────────────────────────────────────────────────
        # Azure ML AutoML will auto-detect column types and apply appropriate
        # transformations (encoding, scaling, imputation) unless overridden.

        # ── Search limits ────────────────────────────────────────────────────
        # max_trials: how many algorithm/hyperparameter combinations to try
        # timeout_minutes: wall-clock budget for the entire job
    )

    # Set limits on the job object
    classification_job.set_limits(
        max_trials=max_trials,
        timeout_minutes=timeout_minutes,
        max_concurrent_trials=4,          # parallel trials per compute cluster
        trial_timeout_minutes=15,         # kill any single trial after 15 min
        enable_early_termination=True,    # stop unpromising trials early
    )

    # Allow a curated list of algorithms — prevents exploration of very slow models
    classification_job.set_training(
        allowed_training_algorithms=[
            "LogisticRegression",
            "RandomForest",
            "GradientBoosting",
            "LightGBM",
            "XGBoostClassifier",
        ],
        enable_stack_ensemble=True,       # stack top models for extra performance
        enable_vote_ensemble=True,
    )

    # Featurization settings
    classification_job.set_featurization(mode="auto")

    return classification_job


def wait_for_completion(ml_client: MLClient, job_name: str, poll_interval: int = 60) -> str:
    """Poll job status until terminal state and return final status string."""
    terminal_states = {"Completed", "Failed", "Canceled"}
    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        log.info("Job '%s' status: %s", job_name, status)
        if status in terminal_states:
            return status
        time.sleep(poll_interval)


def main(args: argparse.Namespace) -> None:
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)

    # ── 1. Build the AutoML job ───────────────────────────────────────────────
    job = build_automl_job(
        training_data_asset=args.training_data,
        validation_data_asset=args.validation_data,
        compute_target=args.compute_target,
        experiment_name=args.experiment_name,
        max_trials=args.max_trials,
        timeout_minutes=args.timeout_minutes,
    )

    # ── 2. Submit ─────────────────────────────────────────────────────────────
    returned_job = ml_client.jobs.create_or_update(job)
    log.info("Submitted AutoML job: %s", returned_job.name)
    log.info("Studio URL: %s", returned_job.studio_url)

    # ── 3. Optionally wait for completion ────────────────────────────────────
    if args.wait:
        final_status = wait_for_completion(ml_client, returned_job.name)
        log.info("AutoML job finished with status: %s", final_status)

        if final_status == "Completed":
            # Retrieve best child run details
            best_run = ml_client.jobs.get(returned_job.name)
            log.info("Best model details available in Azure ML Studio")

            # Write job name for downstream steps (evaluate, deploy)
            with open(args.output_job_name_file, "w") as fh:
                fh.write(returned_job.name)
            log.info("Job name written to %s", args.output_job_name_file)
        else:
            raise RuntimeError(f"AutoML job ended with non-successful status: {final_status}")
    else:
        log.info("Job submitted. Check Studio URL above. Re-run with --wait to block.")
        with open(args.output_job_name_file, "w") as fh:
            fh.write(returned_job.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Azure AutoML classification job")
    parser.add_argument("--subscription-id",  default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",   default="mlops-churn-rg")
    parser.add_argument("--workspace",        default="mlops-churn-ws")
    parser.add_argument("--training-data",    required=True,  help="Azure ML Data Asset URI for training MLTable")
    parser.add_argument("--validation-data",  required=True,  help="Azure ML Data Asset URI for validation MLTable")
    parser.add_argument("--compute-target",   default="cpu-cluster")
    parser.add_argument("--experiment-name",  default="telco-churn-automl")
    parser.add_argument("--max-trials",       type=int, default=20)
    parser.add_argument("--timeout-minutes",  type=int, default=60)
    parser.add_argument("--wait",             action="store_true", help="Block until job completes")
    parser.add_argument("--output-job-name-file", default="outputs/automl_job_name.txt")
    main(parser.parse_args())
