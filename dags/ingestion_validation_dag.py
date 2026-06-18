"""
DAG: Data Ingestion + Validation.
1. Load raw data
2. Validate with Great Expectations-like checks
3. If valid, trigger feature engineering
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

BASE = "/opt/airflow"
log = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def validate_raw_data(**context):
    """Validate raw input data before processing."""
    import sys
    sys.path.insert(0, BASE)
    from monitoring.data_validator import validator
    import pandas as pd
    try:
        df = pd.read_parquet(f"{BASE}/data/features/features_engineered.parquet")
    except Exception:
        df = pd.read_parquet(f"{BASE}/data/raw/sample_data.parquet")
    report = validator.validate_features(df, dataset_name="raw_input")
    validator.save_report(report, "raw_validation")
    context["ti"].xcom_push(key="validation_report", value=json.dumps(report))
    return "proceed" if report["passed"] else "validation_failed"


def handle_validation_failure(**context):
    """Handle data validation failure."""
    ti = context["ti"]
    report = json.loads(ti.xcom_pull(task_ids="validate_raw_data", key="validation_report"))
    log.error("Validation FAILED: %s", json.dumps(report["errors"], indent=2))
    return "validation_failed"


with DAG(
    dag_id="ingestion_validation_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Étape 1: Ingestion + Validation des données (Great Expectations style)",
    tags=["mlops", "ingestion", "validation", "data-quality"],
) as dag:

    validate_raw_data = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    handle_failure = PythonOperator(
        task_id="handle_validation_failure",
        python_callable=handle_validation_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {BASE} && PYTHONPATH={BASE} python3 src/feature_engineering.py",
    )

    validate_raw_data >> feature_engineering
    validate_raw_data >> handle_failure
