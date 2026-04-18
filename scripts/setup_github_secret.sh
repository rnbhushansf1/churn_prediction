#!/usr/bin/env bash
# setup_github_secret.sh
# Creates an Azure Service Principal and adds AZURE_CREDENTIALS to GitHub.
#
# Prerequisites:
#   - Azure CLI logged in:  az login
#   - GitHub CLI installed: gh auth login
#
# Usage:
#   chmod +x scripts/setup_github_secret.sh
#   ./scripts/setup_github_secret.sh

set -euo pipefail

SUBSCRIPTION_ID="bc906f50-e57d-4464-bfb5-5285937d2b4a"
RESOURCE_GROUP="mlops-churn-rg"
SP_NAME="mlops-churn-sp"
GITHUB_REPO="rnbhushansf1/churn_prediction"

echo "=== Creating Service Principal ==="
# Scope to the resource group — least-privilege principle
CREDENTIALS=$(az ad sp create-for-rbac \
  --name "$SP_NAME" \
  --role "Contributor" \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth)

echo "Service Principal created."

echo ""
echo "=== Adding AZURE_CREDENTIALS secret to GitHub ==="
echo "$CREDENTIALS" | gh secret set AZURE_CREDENTIALS \
  --repo "$GITHUB_REPO"

echo ""
echo "=== Done ==="
echo "GitHub Actions can now authenticate to Azure."
echo "Verify at: https://github.com/$GITHUB_REPO/settings/secrets/actions"
