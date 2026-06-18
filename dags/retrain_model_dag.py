"""
DAG: Retraining automatique avec champion/challenger.
- Récupère les dernières features
- Entraîne un nouveau modèle LightGBM
- Évalue et compare avec le champion en production
- Promouvoir si meilleur, archive sinon
- Push les métriques d'accuracy vers l'API Prometheus
- Rollback si performance dégrade
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

BASE = "/opt/airflow"
log = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def load_and_preprocess_data(**context):
    """Load and preprocess training data via feature store."""
    import sys
    sys.path.insert(0, BASE)
    import pandas as pd
    import config.settings as settings
    from src.feature_store import preprocess_train
    X_train = pd.read_parquet(f"{BASE}/data/features/X_train.parquet")
    y_train = pd.read_parquet(f"{BASE}/data/features/y_train.parquet")
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train[settings.TARGET_COL] if settings.TARGET_COL in y_train.columns else y_train.iloc[:, 0]
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
    X_val = pd.read_parquet(f"{BASE}/data/features/X_val.parquet")
    y_val = pd.read_parquet(f"{BASE}/data/features/y_val.parquet")
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val[settings.TARGET_COL] if settings.TARGET_COL in y_val.columns else y_val.iloc[:, 0]
    y_val = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int)
    X_test = pd.read_parquet(f"{BASE}/data/features/X_test.parquet")
    y_test = pd.read_parquet(f"{BASE}/data/features/y_test.parquet")
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test[settings.TARGET_COL] if settings.TARGET_COL in y_test.columns else y_test.iloc[:, 0]
    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(int)
    feature_cols = pd.read_csv(f"{BASE}/data/features/feature_names.csv").iloc[:, 0].tolist()
    for df in [X_train, X_val, X_test]:
        df.columns = df.columns.astype(str)
    context["ti"].xcom_push(key="feature_cols", value=json.dumps(feature_cols))
    return {
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }


def train_new_model(**context):
    """Train a new LightGBM model and register in MLflow."""
    import sys
    sys.path.insert(0, BASE)
    import json
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import config.settings as settings
    from src.model_registry import registry
    from src.preprocessing import load_train_val_test
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_train_val_test()
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
    context["ti"].xcom_push(key="new_model_version", value=version)
    log.info("New model trained and registered: v%s", version)
    return version


def compare_and_promote(**context):
    """Compare with champion and decide promotion."""
    import sys
    sys.path.insert(0, BASE)
    import json
    import pandas as pd
    import config.settings as settings
    from src.model_registry import registry
    ti = context["ti"]
    new_version = ti.xcom_pull(task_ids="train_new_model", key="new_model_version")
    X_val = pd.read_parquet(f"{BASE}/data/features/X_val.parquet")
    y_val = pd.read_parquet(f"{BASE}/data/features/y_val.parquet")
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]
    y_val = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    decision = registry.compare_and_decide(X_val, y_val, new_version)
    context["ti"].xcom_push(key="promotion_decision", value=json.dumps(decision))
    log.info("Promotion decision: %s", decision["reason"])
    return decision["reason"]


def push_metrics_to_api(**context):
    """Push accuracy metrics to API Prometheus endpoint."""
    import requests
    import sys
    sys.path.insert(0, BASE)
    import json
    import pandas as pd
    config = __import__("config.settings", fromlist=["settings"])
    ti = context["ti"]
    decision = json.loads(ti.xcom_pull(task_ids="compare_and_promote", key="promotion_decision"))
    metrics = decision.get("new_metrics", {})
    version = decision.get("new_version", "0")
    try:
        requests.post(
            "http://api:8000/accuracy/update",
            json={
                "model_version": f"v{version}",
                "values": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            },
            timeout=5,
        )
    except Exception as e:
        log.warning("Could not push accuracy to API: %s", e)
    return "metrics_pushed"


def cleanup_old_models(**context):
    """Archive old model versions beyond the last 10."""
    import sys
    sys.path.insert(0, BASE)
    from src.model_registry import registry
    registry.cleanup_old_versions(keep_last=10)
    return "cleanup_done"


with DAG(
    dag_id="retrain_model_dag",
    default_args=default_args,
    schedule_interval="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Réentraînement automatique: train -> compare -> promote (champion/challenger)",
    tags=["mlops", "retraining", "mlflow", "champion", "production"],
) as dag:

    load_data = PythonOperator(
        task_id="load_and_preprocess",
        python_callable=load_and_preprocess_data,
    )

    train = PythonOperator(
        task_id="train_new_model",
        python_callable=train_new_model,
    )

    compare = PythonOperator(
        task_id="compare_and_promote",
        python_callable=compare_and_promote,
    )

    push_metrics = PythonOperator(
        task_id="push_metrics_to_api",
        python_callable=push_metrics_to_api,
    )

    cleanup = PythonOperator(
        task_id="cleanup_old_models",
        python_callable=cleanup_old_models,
    )

    load_data >> train >> compare >> push_metrics >> cleanup
