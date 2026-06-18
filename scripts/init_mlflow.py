#!/usr/bin/env python3
"""
Seed MLflow Model Registry with initial champion model.
Uses ModelRegistry (champion/challenger system).

Usage:
    docker compose run --rm init
    docker compose run --rm api python /app/scripts/init_mlflow.py
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config.settings as settings
from src.model_registry import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("init_mlflow")

import mlflow
from mlflow.tracking import MlflowClient


def wait_for_mlflow(max_retries=30, delay=5):
    for i in range(max_retries):
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_URI)
            client = MlflowClient()
            client.search_experiments()
            log.info("MLflow available after %ds", i * delay)
            return True
        except Exception as e:
            if i % 5 == 0:
                log.warning("Waiting for MLflow (%d/%d): %s", i + 1, max_retries, e)
            time.sleep(delay)
    log.error("MLflow not available after %d retries", max_retries)
    return False


def load_data():
    X_train = pd.read_parquet(settings.FEATURES_DIR / "X_train.parquet")
    X_val = pd.read_parquet(settings.FEATURES_DIR / "X_val.parquet")
    X_test = pd.read_parquet(settings.FEATURES_DIR / "X_test.parquet")

    def load_target(p):
        df = pd.read_parquet(p)
        if isinstance(df, pd.DataFrame):
            c = settings.TARGET_COL if settings.TARGET_COL in df.columns else df.columns[0]
            y = df[c]
        else:
            y = df
        return pd.to_numeric(y, errors="coerce").fillna(0).astype(int).reset_index(drop=True)

    y_train = load_target(settings.FEATURES_DIR / "y_train.parquet")
    y_val = load_target(settings.FEATURES_DIR / "y_val.parquet")
    y_test = load_target(settings.FEATURES_DIR / "y_test.parquet")

    feat_path = settings.FEATURES_DIR / "feature_names.csv"
    feature_cols = pd.read_csv(feat_path).iloc[:, 0].tolist() if feat_path.exists() else X_train.columns.tolist()

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_model(X_train, y_train, X_val, y_val):
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
    return model, params


def main():
    log.info("=" * 60)
    log.info("INIT MLFLOW MODEL REGISTRY")
    log.info("=" * 60)
    log.info("URI: %s", settings.MLFLOW_URI)
    log.info("Model: %s", settings.MLFLOW_MODEL_NAME)

    if not wait_for_mlflow():
        sys.exit(1)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
        log.info("Data: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)
    except Exception as e:
        log.error("Data load failed: %s", e)
        sys.exit(1)

    try:
        model, params = train_model(X_train, y_train, X_val, y_val)
    except Exception as e:
        log.error("Training failed: %s", e)
        sys.exit(1)

    try:
        version = registry.log_training_run(
            model, X_val, y_val, X_test, y_test,
            params, feature_cols,
            dataset_version="v1",
        )
        log.info("Model registered: v%s", version)
        decision = registry.compare_and_decide(X_val, y_val, version)
        log.info("Decision: %s (promoted=%s)", decision["reason"], decision["promoted"])
        log.info("=" * 60)
        log.info("INIT COMPLETE: champion model v%s", version)
        log.info("=" * 60)
    except Exception as e:
        log.error("Registration failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
