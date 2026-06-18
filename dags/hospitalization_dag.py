"""
DAG: Pipeline MLOps complet hebdomadaire.
Nettoyage -> Features -> Validation -> Training -> Registry -> Drift -> Monitoring
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

BASE = "/opt/airflow"
PYTHON = f"cd {BASE} && PYTHONPATH={BASE} python3"

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def validate_after_engineering(**context):
    import sys
    sys.path.insert(0, BASE)
    from monitoring.data_validator import validator
    import pandas as pd
    df = pd.read_parquet(f"{BASE}/data/features/features_engineered.parquet")
    report = validator.validate_features(df, dataset_name="feature_engineering")
    validator.save_report(report, "features_engineered")
    if not report["passed"]:
        raise ValueError(f"Feature validation failed: {report['errors']}")
    return "validation_passed"


with DAG(
    dag_id="hospitalization_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "hospitalisation", "pipeline-complet"],
) as dag:

    data_cleaning = BashOperator(
        task_id="data_cleaning",
        bash_command=f"{PYTHON} src/data_cleaning.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"{PYTHON} src/feature_engineering.py",
    )

    validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=validate_after_engineering,
    )

    training = BashOperator(
        task_id="training",
        bash_command=f"{PYTHON} src/train.py",
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=f"{PYTHON} monitoring/drift_detector.py || true",
    )

    healthcheck_api = BashOperator(
        task_id="healthcheck_api",
        bash_command="curl -sf http://api:8000/health || curl -sf http://mlflow:5000/",
    )

    data_cleaning >> feature_engineering >> validate_features >> training >> drift_check >> healthcheck_api
