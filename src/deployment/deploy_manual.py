"""
Deploy manual XGBoost model to an Azure ML Managed Online Endpoint.
Creates the endpoint, deploys the scoring container, and tests via REST.
"""

import argparse
import json
import logging
import os
import time

import requests
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENDPOINT_NAME = "churn-manual-endpoint"
DEPLOYMENT_NAME = "xgboost-blue"

# Sample payload for smoke testing the endpoint
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
            0, 0, 1, 0, 12,
            1, 0, 1, 0,
            0, 0, 0, 0,
            0, 0, 1, 2,
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


def create_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    """Create a Managed Online Endpoint if it doesn't exist."""
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        log.info("Endpoint '%s' already exists (state: %s)", endpoint_name, endpoint.provisioning_state)
        return endpoint
    except Exception:
        log.info("Creating endpoint '%s'...", endpoint_name)
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="XGBoost Telco Churn Prediction (manual track)",
            auth_mode="key",            # use API key authentication
            tags={"track": "manual", "model": "xgboost"},
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        log.info("Endpoint created: %s", endpoint_name)
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
    """Deploy the registered model to the endpoint."""
    # Resolve the model to deploy
    if model_version == "latest":
        versions = list(ml_client.models.list(name=registered_model_name))
        model_ver = str(max(int(v.version) for v in versions))
    else:
        model_ver = model_version

    log.info("Deploying model '%s' version '%s'...", registered_model_name, model_ver)

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"{registered_model_name}:{model_ver}",
        instance_type=instance_type,
        instance_count=instance_count,

        # Azure ML automatically serves MLflow models without a custom scoring script.
        # If you need custom logic, uncomment code_configuration below.
        # code_configuration=CodeConfiguration(
        #     code="src/deployment/scoring",
        #     scoring_script="score.py",
        # ),
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    log.info("Deployment '%s' is live.", deployment_name)

    # Route 100 % of traffic to this deployment (blue/green pattern)
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    log.info("Traffic routed 100 %% to '%s'", deployment_name)


def smoke_test(ml_client: MLClient, endpoint_name: str) -> dict:
    """Send a sample request and verify the response."""
    log.info("Running smoke test against endpoint '%s'...", endpoint_name)
    import tempfile, json as _json, pathlib

    # Write sample payload to a temp file (Azure ML SDK invoke method)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        _json.dump(SAMPLE_PAYLOAD, tmp)
        tmp_path = tmp.name

    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=tmp_path,
    )
    log.info("Smoke test response: %s", response)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    return response


def main(args: argparse.Namespace) -> None:
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)

    # ── 1. Create endpoint ────────────────────────────────────────────────────
    create_endpoint(ml_client, args.endpoint_name)

    # ── 2. Deploy model ───────────────────────────────────────────────────────
    deploy_model(
        ml_client,
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        registered_model_name=args.registered_model_name,
        model_version=args.model_version,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    # ── 3. Smoke test ─────────────────────────────────────────────────────────
    if not args.skip_smoke_test:
        smoke_test(ml_client, args.endpoint_name)
    else:
        log.info("Skipping smoke test (--skip-smoke-test flag set)")

    # ── 4. Print endpoint URI for reference ──────────────────────────────────
    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    log.info("Scoring URI: %s", endpoint.scoring_uri)
    log.info("Swagger URI: %s", endpoint.openapi_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy XGBoost model to Azure ML Endpoint")
    parser.add_argument("--subscription-id",       default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",        default="mlops-churn-rg")
    parser.add_argument("--workspace",             default="mlops-churn-ws")
    parser.add_argument("--endpoint-name",         default=ENDPOINT_NAME)
    parser.add_argument("--deployment-name",       default=DEPLOYMENT_NAME)
    parser.add_argument("--registered-model-name", default="telco-churn-xgboost")
    parser.add_argument("--model-version",         default="latest")
    parser.add_argument("--instance-type",         default="Standard_DS3_v2")
    parser.add_argument("--instance-count",        type=int, default=1)
    parser.add_argument("--skip-smoke-test",       action="store_true")
    main(parser.parse_args())
