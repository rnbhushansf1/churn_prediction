"""
Deploy best AutoML model to its own Azure ML Managed Online Endpoint.
Fetches the best child run from a completed AutoML job, registers it,
and deploys to a separate endpoint for side-by-side comparison.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENDPOINT_NAME   = "churn-automl-endpoint"
DEPLOYMENT_NAME = "automl-blue"

SAMPLE_PAYLOAD = {
    "input_data": {
        "columns": [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ],
        "data": [[
            "Male", 0, "Yes", "No", 12,
            "Yes", "No", "Fiber optic", "No",
            "No", "No", "No", "No",
            "No", "Month-to-month", "Yes", "Electronic check",
            65.5, 786.0,
        ]],
    }
}


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def register_best_automl_model(
    ml_client: MLClient,
    automl_job_name: str,
    registered_model_name: str,
) -> str:
    """
    Retrieve best child run of an AutoML job, download its model artifact,
    and register it in the Azure ML model registry.
    Returns the registered model version string.
    """
    log.info("Fetching best child run for AutoML job '%s'...", automl_job_name)

    # The best child run has the tag 'automl_best_child_run_id' on the parent job
    parent_job = ml_client.jobs.get(automl_job_name)

    # Use MLflow to find the best run in the parent experiment
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client._operation_scope.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment = mlflow.get_experiment_by_name(parent_job.experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{parent_job.experiment_name}' not found")

    # Query all child runs ordered by AUC_weighted descending
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_job.name}'",
        order_by=["metrics.AUC_weighted DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No child runs found for AutoML job")

    best_run_id = runs.iloc[0]["run_id"]
    best_auc    = runs.iloc[0].get("metrics.AUC_weighted", "N/A")
    log.info("Best child run: %s  AUC_weighted: %s", best_run_id, best_auc)

    # Register the best model from MLflow
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )
    log.info("Registered AutoML model: %s v%s", registered.name, registered.version)
    return str(registered.version)


def create_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    """Create Managed Online Endpoint if it doesn't already exist."""
    try:
        ep = ml_client.online_endpoints.get(endpoint_name)
        log.info("Endpoint '%s' already exists.", endpoint_name)
        return ep
    except Exception:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="AutoML Telco Churn Prediction endpoint",
            auth_mode="key",
            tags={"track": "automl"},
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        log.info("Created endpoint '%s'", endpoint_name)
        return ml_client.online_endpoints.get(endpoint_name)


def deploy_model(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    registered_model_name: str,
    model_version: str,
    instance_type: str,
    instance_count: int,
) -> None:
    """Deploy the registered AutoML model to the endpoint."""
    log.info(
        "Deploying '%s' v%s → endpoint '%s' deployment '%s'",
        registered_model_name, model_version, endpoint_name, deployment_name,
    )
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"{registered_model_name}:{model_version}",
        instance_type=instance_type,
        instance_count=instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route all traffic to this deployment
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    log.info("Traffic routed 100 %% to '%s'", deployment_name)


def smoke_test(ml_client: MLClient, endpoint_name: str) -> None:
    """Invoke endpoint with sample data to verify it's serving correctly."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(SAMPLE_PAYLOAD, tmp)
        tmp_path = tmp.name

    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=tmp_path,
    )
    log.info("Smoke test response: %s", response)
    Path(tmp_path).unlink(missing_ok=True)


def main(args: argparse.Namespace) -> None:
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)

    # ── 1. Read AutoML job name from file or CLI arg ──────────────────────────
    if args.automl_job_name:
        job_name = args.automl_job_name
    elif Path(args.job_name_file).exists():
        job_name = Path(args.job_name_file).read_text().strip()
    else:
        raise ValueError("Provide --automl-job-name or ensure job_name_file exists")

    # ── 2. Register best AutoML model ─────────────────────────────────────────
    model_version = register_best_automl_model(
        ml_client, job_name, args.registered_model_name
    )

    # ── 3. Create endpoint ────────────────────────────────────────────────────
    create_endpoint(ml_client, args.endpoint_name)

    # ── 4. Deploy ─────────────────────────────────────────────────────────────
    deploy_model(
        ml_client,
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        registered_model_name=args.registered_model_name,
        model_version=model_version,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    # ── 5. Smoke test ─────────────────────────────────────────────────────────
    if not args.skip_smoke_test:
        smoke_test(ml_client, args.endpoint_name)

    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    log.info("AutoML endpoint scoring URI: %s", endpoint.scoring_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register and deploy best AutoML model")
    parser.add_argument("--subscription-id",       default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",        default="mlops-churn-rg")
    parser.add_argument("--workspace",             default="mlops-churn-ws")
    parser.add_argument("--automl-job-name",       default=None,  help="Override job name from file")
    parser.add_argument("--job-name-file",         default="outputs/automl_job_name.txt")
    parser.add_argument("--registered-model-name", default="telco-churn-automl")
    parser.add_argument("--endpoint-name",         default=ENDPOINT_NAME)
    parser.add_argument("--deployment-name",       default=DEPLOYMENT_NAME)
    parser.add_argument("--instance-type",         default="Standard_DS3_v2")
    parser.add_argument("--instance-count",        type=int, default=1)
    parser.add_argument("--skip-smoke-test",       action="store_true")
    main(parser.parse_args())
