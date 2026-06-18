"""
DAG: Auto-Retraining complet.
Pipeline end-to-end: validation -> training -> registry -> monitoring.
Runs daily to ensure model stays fresh.
"""
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

BASE = "/opt/airflow"
log = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def run_full_pipeline(**context):
    """Execute full retraining pipeline: validate -> train -> promote."""
    import sys
    sys.path.insert(0, BASE)
    import lightgbm as lgb
    import pandas as pd
    import config.settings as settings
    from src.model_registry import registry
    from src.preprocessing import load_train_val_test
    from monitoring.data_validator import validator
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_train_val_test()
    val_report = validator.validate_features(X_val, dataset_name="training_features")
    if not val_report["passed"]:
        log.warning("Validation warnings during training: %s", val_report["warnings"])
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": settings.RANDOM_STATE,
        "n_jobs": -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    version = registry.log_training_run(
        model, X_val, y_val, X_test, y_test,
        params, feature_cols,
        dataset_version=f"v{datetime.now().strftime('%Y%m%d')}",
    )
    decision = registry.compare_and_decide(X_val, y_val, version)
    log.info("Pipeline complete: version=%s, decision=%s", version, decision["reason"])
    context["ti"].xcom_push(key="pipeline_result", value=json.dumps(decision))
    return decision["reason"]


with DAG(
    dag_id="auto_retraining",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Pipeline auto-retraining complet: train -> compare -> promote -> monitor",
    tags=["mlops", "retraining", "mlflow", "auto", "production"],
) as dag:

    run_pipeline = PythonOperator(
        task_id="run_full_pipeline",
        python_callable=run_full_pipeline,
    )

    run_pipeline
