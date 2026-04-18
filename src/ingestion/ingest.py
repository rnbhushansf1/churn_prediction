"""
Data ingestion: load Telco CSV from local/blob, validate schema,
convert to parquet, and register as an Azure ML v2 Data Asset.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Expected Telco schema: column → nullable ────────────────────────────────
SCHEMA = {
    "customerID": False,
    "gender": False,
    "SeniorCitizen": False,
    "Partner": False,
    "Dependents": False,
    "tenure": False,
    "PhoneService": False,
    "MultipleLines": True,
    "InternetService": False,
    "OnlineSecurity": True,
    "OnlineBackup": True,
    "DeviceProtection": True,
    "TechSupport": True,
    "StreamingTV": True,
    "StreamingMovies": True,
    "Contract": False,
    "PaperlessBilling": False,
    "PaymentMethod": False,
    "MonthlyCharges": False,
    "TotalCharges": True,   # contains spaces that become NaN
    "Churn": False,
}

NULL_THRESHOLD = 0.05   # fail if any required column exceeds 5 % nulls


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all expected columns exist and null rates are within threshold."""
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, nullable in SCHEMA.items():
        null_rate = df[col].isna().mean()
        if not nullable and null_rate > NULL_THRESHOLD:
            raise ValueError(
                f"Column '{col}' null rate {null_rate:.2%} exceeds threshold {NULL_THRESHOLD:.0%}"
            )
    log.info("Schema validation passed (%d rows, %d cols)", len(df), len(df.columns))


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning before saving to parquet."""
    # TotalCharges has ' ' strings instead of NaN — coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Binary target: Yes→1, No→0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop duplicate customers
    before = len(df)
    df = df.drop_duplicates(subset=["customerID"])
    log.info("Dropped %d duplicate rows", before - len(df))
    return df


def register_data_asset(
    ml_client: MLClient,
    parquet_path: str,
    name: str = "telco-churn-curated",
    version: str = "1",
    description: str = "Curated Telco churn dataset (parquet)",
) -> Data:
    """Register a local parquet file as an Azure ML Data Asset (uri_file)."""
    data_asset = Data(
        name=name,
        version=version,
        description=description,
        path=parquet_path,
        type=AssetTypes.URI_FILE,
    )
    registered = ml_client.data.create_or_update(data_asset)
    log.info("Registered data asset '%s' version '%s'", registered.name, registered.version)
    return registered


def main(args: argparse.Namespace) -> None:
    # ── 1. Load raw CSV ──────────────────────────────────────────────────────
    log.info("Loading CSV from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    log.info("Loaded %d rows", len(df))

    # ── 2. Validate schema ───────────────────────────────────────────────────
    validate_schema(df)

    # ── 3. Clean ─────────────────────────────────────────────────────────────
    df = clean_raw(df)

    # ── 4. Save as parquet ───────────────────────────────────────────────────
    out_path = Path(args.output_dir) / "telco_curated.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Saved parquet to %s (%d rows)", out_path, len(df))

    # ── 5. Register in Azure ML (optional — skip if --no-register) ───────────
    if not args.no_register:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace,
        )
        register_data_asset(
            ml_client,
            parquet_path=str(out_path),
            name=args.asset_name,
            version=args.asset_version,
        )
    else:
        log.info("Skipping Azure ML registration (--no-register flag set)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Telco Churn CSV → parquet → Azure ML")
    parser.add_argument("--input-csv", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for output parquet")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID", "bc906f50-e57d-4464-bfb5-5285937d2b4a"))
    parser.add_argument("--resource-group", default="mlops-churn-rg")
    parser.add_argument("--workspace", default="mlops-churn-ws")
    parser.add_argument("--asset-name", default="telco-churn-curated")
    parser.add_argument("--asset-version", default="1")
    parser.add_argument("--no-register", action="store_true", help="Skip Azure ML registration")
    main(parser.parse_args())
