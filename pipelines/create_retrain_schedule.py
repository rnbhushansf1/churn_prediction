"""
Create a daily retraining schedule using the Azure ML Python SDK.
Run with: python pipelines/create_retrain_schedule.py
"""

import logging
from pathlib import Path
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import JobSchedule, CronTrigger
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBSCRIPTION_ID = "bc906f50-e57d-4464-bfb5-5285937d2b4a"
RESOURCE_GROUP  = "mlops-churn-rg"
WORKSPACE       = "mlops-churn-ws"


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )

    # Load the pipeline job from YAML
    pipeline_yaml = Path(__file__).parent / "retrain_pipeline.yaml"
    pipeline_job = load_job(source=str(pipeline_yaml))
    log.info("Loaded pipeline from %s", pipeline_yaml)

    schedule = JobSchedule(
        name="daily-churn-retrain",
        display_name="Daily Churn Retraining",
        description="Retrain XGBoost churn model daily at 02:00 UTC, promote if AUC improves >1%",
        trigger=CronTrigger(
            expression="0 2 * * *",
            start_time="2026-04-19T02:00:00",
            time_zone="UTC",
        ),
        create_job=pipeline_job,
    )

    poller = ml_client.schedules.begin_create_or_update(schedule)
    result = poller.result()
    log.info("Schedule created: %s (status: %s)", result.name, result.provisioning_state)
    log.info("Next run at 02:00 UTC daily.")


if __name__ == "__main__":
    main()
