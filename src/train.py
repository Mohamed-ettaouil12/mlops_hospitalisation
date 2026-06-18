"""
Production-grade training pipeline.
Uses ModelRegistry (champion/challenger) and FeatureStore.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config.settings as settings
from src.model_registry import registry
from src.preprocessing import load_train_val_test

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    if params is None:
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


def main(dataset_version: str = "v1"):
    log.info("=" * 60)
    log.info("TRAINING PIPELINE - Champion/Challenger")
    log.info("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_train_val_test()
    log.info("Data: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)
    model, params = train_lightgbm(X_train, y_train, X_val, y_val)
    version = registry.log_training_run(
        model, X_val, y_val, X_test, y_test,
        params, feature_cols,
        dataset_version=dataset_version,
    )
    log.info("Model registered: v%s", version)
    decision = registry.compare_and_decide(X_val, y_val, version)
    log.info("Decision: %s (promoted=%s)", decision["reason"], decision["promoted"])
    report = {
        "version": version,
        "decision": decision,
        "dataset_version": dataset_version,
        "timestamp": datetime.now().isoformat(),
    }
    report_path = settings.TRAINING_REPORTS_DIR / "retrain_result.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    dv = sys.argv[1] if len(sys.argv) > 1 else f"v{datetime.now().strftime('%Y%m%d')}"
    main(dv)
