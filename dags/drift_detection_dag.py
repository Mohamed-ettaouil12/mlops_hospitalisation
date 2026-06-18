"""
DAG: Drift Detection (Evidently AI + PSI).
Daily drift check -> decision -> trigger retrain if needed.
Pushes drift scores to API metrics endpoint.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

BASE = "/opt/airflow"
log = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}


def run_drift_detection(**context):
    """Run full drift detection pipeline."""
    import sys
    sys.path.insert(0, BASE)
    from monitoring.drift_detector import detect_drift
    import config.settings as settings
    result = detect_drift()
    context["ti"].xcom_push(key="drift_result", value=json.dumps(result))
    return "drift_check_complete"


def decide_retrain(**context):
    """Decide whether to trigger retraining based on drift results."""
    ti = context["ti"]
    result = json.loads(ti.xcom_pull(task_ids="run_drift_detection", key="drift_result"))
    drift_detected = result.get("drift_detected", False)
    severity = result.get("drift_severity", "NONE")
    drift_pct = result.get("drift_pct", 0)
    log.info("Drift decision: detected=%s, severity=%s, pct=%.1f%%",
             drift_detected, severity, drift_pct)
    context["ti"].xcom_push(key="drift_severity", value=severity)
    if drift_detected and severity in ("CRITICAL", "HIGH"):
        return "trigger_retrain"
    return "skip_retrain"


def push_drift_to_api(**context):
    """Push drift metrics to API for Prometheus scraping."""
    import requests
    ti = context["ti"]
    result = json.loads(ti.xcom_pull(task_ids="run_drift_detection", key="drift_result"))
    try:
        ver = requests.get("http://api:8000/model/info", timeout=5).json().get("version", "unknown")
    except Exception:
        ver = "unknown"
    try:
        requests.post(
            "http://api:8000/drift/score",
            json={"drift_max_psi": result.get("max_psi", 0), "model_version": ver},
            timeout=5,
        )
    except Exception as e:
        log.warning("Could not push drift to API: %s", e)
    return "drift_pushed"


def save_drift_report(**context):
    """Archive drift report with timestamp."""
    import shutil
    ti = context["ti"]
    result = json.loads(ti.xcom_pull(task_ids="run_drift_detection", key="drift_result"))
    report_path = Path(f"{BASE}/reports/drift/drift_check_result.json")
    archive_path = Path(f"{BASE}/reports/drift/archive/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        shutil.copy(str(report_path), str(archive_path))
        log.info("Drift report archived: %s", archive_path)
    return "report_saved"


with DAG(
    dag_id="drift_detection_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Détection quotidienne de dérive (Evidently + PSI) -> décision retrain",
    tags=["mlops", "drift", "evidently", "monitoring", "production"],
) as dag:

    run_drift_check = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
    )

    push_drift = PythonOperator(
        task_id="push_drift_to_api",
        python_callable=push_drift_to_api,
    )

    save_report = PythonOperator(
        task_id="save_drift_report",
        python_callable=save_drift_report,
    )

    decide = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_model_dag",
        wait_for_completion=False,
    )

    skip_retrain = BashOperator(
        task_id="skip_retrain",
        bash_command="echo 'No significant drift detected, skipping retrain'",
    )

    run_drift_check >> push_drift >> save_report >> decide >> [trigger_retrain, skip_retrain]
