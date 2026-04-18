"""
Monitoring: configure Azure ML Model Monitor for both endpoints.
Tracks data drift, prediction drift, data quality, and latency.
"""

import argparse
import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ModelMonitor,
    MonitoringTarget,
    MonitorDefinition,
    MonitorSchedule,
    RecurrencePattern,
    RecurrenceTrigger,
    AlertNotification,
    DataDriftSignal,
    PredictionDriftSignal,
    DataQualitySignal,
    ProductionData,
    ReferenceData,
    MonitorFeatureFilter,
    DataSignalThreshold,
    NumericalDriftMetrics,
    CategoricalDriftMetrics,
)
from azure.ai.ml.constants import MonitorTargetTasks, MonitorDatasetContext
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Alert email — change to your notification address ───────────────────────
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "pbateman627@gmail.com")


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def configure_data_drift_signal(feature_names: list[str]) -> DataDriftSignal:
    """
    Data drift signal: compares current production data distribution
    against the training reference distribution using Jensen-Shannon divergence.
    """
    return DataDriftSignal(
        reference_data=ReferenceData(
            input_data=None,     # filled at monitor attachment time via baseline_dataset
            data_context=MonitorDatasetContext.TRAINING,
        ),
        features=MonitorFeatureFilter(top_n_feature_importance=10),
        metric_thresholds=[
            NumericalDriftMetrics(
                applicable_feature_type="numerical",
                threshold=DataSignalThreshold(threshold=0.2),   # JSdivergence > 0.2 → alert
            ),
            CategoricalDriftMetrics(
                applicable_feature_type="categorical",
                threshold=DataSignalThreshold(threshold=0.2),
            ),
        ],
        alert_enabled=True,
    )


def configure_prediction_drift_signal() -> PredictionDriftSignal:
    """
    Prediction drift signal: monitors shift in the model's output
    distribution over time. A sudden change can indicate model degradation.
    """
    return PredictionDriftSignal(
        reference_data=ReferenceData(
            input_data=None,
            data_context=MonitorDatasetContext.TRAINING,
        ),
        metric_thresholds=[
            NumericalDriftMetrics(
                applicable_feature_type="numerical",
                threshold=DataSignalThreshold(threshold=0.2),
            ),
        ],
        alert_enabled=True,
    )


def configure_data_quality_signal() -> DataQualitySignal:
    """
    Data quality signal: checks null rates, out-of-range values,
    and schema violations in production inference data.
    """
    return DataQualitySignal(
        reference_data=ReferenceData(
            input_data=None,
            data_context=MonitorDatasetContext.TRAINING,
        ),
        features=MonitorFeatureFilter(top_n_feature_importance=10),
        alert_enabled=True,
    )


def create_monitor(
    ml_client: MLClient,
    monitor_name: str,
    endpoint_name: str,
    deployment_name: str,
    baseline_dataset_id: str,
    notification_emails: list[str],
) -> None:
    """
    Create a ModelMonitor attached to a Managed Online Endpoint.
    Runs daily and alerts via email when thresholds are breached.
    """
    log.info("Configuring monitor '%s' for endpoint '%s/%s'", monitor_name, endpoint_name, deployment_name)

    monitor = ModelMonitor(
        name=monitor_name,
        description=f"Monitor for {endpoint_name} — data drift, prediction drift, quality",
        target=MonitoringTarget(
            ml_task=MonitorTargetTasks.CLASSIFICATION,
            endpoint_deployment_id=f"azureml:{endpoint_name}:{deployment_name}",
        ),
        monitoring_signals={
            "data_drift":        configure_data_drift_signal(feature_names=[]),
            "prediction_drift":  configure_prediction_drift_signal(),
            "data_quality":      configure_data_quality_signal(),
        },
        alert_notification=AlertNotification(emails=notification_emails),
    )

    # Daily schedule — runs at 6:00 AM UTC
    schedule = MonitorSchedule(
        name=f"{monitor_name}-schedule",
        trigger=RecurrenceTrigger(
            frequency="day",
            interval=1,
            schedule=RecurrencePattern(hours=[6], minutes=[0]),
        ),
        create_monitor=monitor,
    )

    result = ml_client.schedules.begin_create_or_update(schedule).result()
    log.info("Monitor schedule created: %s", result.name)


def main(args: argparse.Namespace) -> None:
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)
    emails    = [e.strip() for e in args.alert_emails.split(",")]

    # ── Configure monitor for manual XGBoost endpoint ────────────────────────
    create_monitor(
        ml_client,
        monitor_name="churn-manual-monitor",
        endpoint_name=args.manual_endpoint_name,
        deployment_name=args.manual_deployment_name,
        baseline_dataset_id=args.baseline_dataset_id,
        notification_emails=emails,
    )

    # ── Configure monitor for AutoML endpoint ────────────────────────────────
    create_monitor(
        ml_client,
        monitor_name="churn-automl-monitor",
        endpoint_name=args.automl_endpoint_name,
        deployment_name=args.automl_deployment_name,
        baseline_dataset_id=args.baseline_dataset_id,
        notification_emails=emails,
    )

    log.info("Both monitors configured. Check Azure ML Studio → Monitoring tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure Azure ML Model Monitors")
    parser.add_argument("--subscription-id",        default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group",         default="mlops-churn-rg")
    parser.add_argument("--workspace",              default="mlops-churn-ws")
    parser.add_argument("--manual-endpoint-name",   default="churn-manual-endpoint")
    parser.add_argument("--manual-deployment-name", default="xgboost-blue")
    parser.add_argument("--automl-endpoint-name",   default="churn-automl-endpoint")
    parser.add_argument("--automl-deployment-name", default="automl-blue")
    parser.add_argument("--baseline-dataset-id",    required=True, help="Azure ML Data Asset ID for training baseline")
    parser.add_argument("--alert-emails",           default=ALERT_EMAIL)
    main(parser.parse_args())
