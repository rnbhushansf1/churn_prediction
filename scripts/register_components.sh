#!/usr/bin/env bash
# register_components.sh
# Run this ONCE before submitting any pipeline jobs.
# It registers the shared environment and all pipeline components in Azure ML.
#
# Usage:
#   chmod +x scripts/register_components.sh
#   ./scripts/register_components.sh

set -euo pipefail

WS="mlops-churn-ws"
RG="mlops-churn-rg"
SUB="bc906f50-e57d-4464-bfb5-5285937d2b4a"

COMMON="--workspace-name $WS --resource-group $RG --subscription $SUB"

echo "=== Registering Azure ML Environment ==="
az ml environment create -f components/environment.yaml $COMMON

echo ""
echo "=== Registering Pipeline Components ==="

components=(
  "components/ingest.yaml"
  "components/preprocess.yaml"
  "components/train_manual.yaml"
  "components/evaluate.yaml"
  "components/evaluate_automl.yaml"
  "components/deploy_manual.yaml"
  "components/deploy_automl.yaml"
)

for component in "${components[@]}"; do
  echo "Registering $component ..."
  az ml component create -f "$component" $COMMON
done

echo ""
echo "=== All components registered successfully ==="
echo "You can now submit pipeline jobs:"
echo "  az ml job create -f pipelines/manual_pipeline.yaml $COMMON"
echo "  az ml job create -f pipelines/automl_pipeline.yaml $COMMON"
