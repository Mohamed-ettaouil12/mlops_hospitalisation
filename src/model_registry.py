"""
Professional MLflow Model Registry.
Champion/Challenger pattern with automatic rollback.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score, roc_auc_score

import config.settings as settings

log = logging.getLogger(__name__)

CHAMPION_TAG = "champion"
CHALLENGER_TAG = "challenger"
ROLLBACK_TAG = "rolled_back"

MODEL_NAME = settings.MLFLOW_MODEL_NAME
MIN_AUC = settings.MIN_AUC_FOR_PRODUCTION


class ModelRegistry:
    """
    Gère le cycle de vie complet des modèles:
    - Champion (production)
    - Challenger (candidat à la promotion)
    - Rollback automatique si performance dégrade
    - Archivage des anciennes versions
    - Métriques comparatives
    """

    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
        self.client = MlflowClient()

    def log_training_run(self, model, X_val, y_val, X_test, y_test,
                         params: dict, features: list[str],
                         dataset_version: str = "v1") -> str:
        """Log train run in MLflow and register model."""
        val_metrics = self._eval_metrics(model, X_val, y_val)
        test_metrics = self._eval_metrics(model, X_test, y_test)

        tags = {
            "dataset_version": dataset_version,
            "training_date": datetime.now().isoformat(),
            "model_type": "lightgbm",
            "project": "hospitalization_risk",
            "val_auc": str(round(val_metrics["auc_roc"], 4)),
            "test_auc": str(round(test_metrics["auc_roc"], 4)),
            "val_recall": str(round(val_metrics["recall"], 4)),
            "n_features": str(len(features)),
        }

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("n_features", len(features))
            mlflow.set_tags(tags)

            mlflow.lightgbm.log_model(
                model, artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, MODEL_NAME)
            version = result.version

            for k, v in tags.items():
                self.client.set_model_version_tag(MODEL_NAME, version, k, v)

            log.info("Model registered: %s v%s (run=%s)", MODEL_NAME, version, run_id)
        return version

    def promote_to_challenger(self, version: str):
        """Tag a model version as challenger for A/B testing."""
        self.client.set_model_version_tag(MODEL_NAME, version, "role", CHALLENGER_TAG)
        self.client.set_model_version_tag(MODEL_NAME, version, "promoted_at", datetime.now().isoformat())
        log.info("Model v%s promoted to Challenger", version)

    def promote_to_champion(self, version: str, old_champion_version: Optional[str] = None):
        """Promote challenger to champion. Archive previous champion."""
        if old_champion_version:
            self.client.set_model_version_tag(MODEL_NAME, old_champion_version, "role", "archived")
            self.client.set_model_version_tag(MODEL_NAME, old_champion_version, "archived_at", datetime.now().isoformat())
            try:
                self.client.transition_model_version_stage(
                    name=MODEL_NAME, version=old_champion_version, stage="Archived",
                )
            except Exception:
                pass
            log.info("Previous champion v%s archived", old_champion_version)

        self.client.set_model_version_tag(MODEL_NAME, version, "role", CHAMPION_TAG)
        self.client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Production",
        )
        self.client.set_model_version_tag(MODEL_NAME, version, "promoted_at", datetime.now().isoformat())
        log.info("Model v%s promoted to Champion (Production)", version)

    def get_champion(self) -> Optional[dict]:
        """Get current production champion model info."""
        try:
            versions = self.client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if versions:
                mv = versions[0]
                return {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "tags": mv.tags,
                }
        except Exception:
            pass
        return None

    def get_challenger(self) -> Optional[dict]:
        """Get current challenger model info."""
        try:
            versions = self.client.get_latest_versions(MODEL_NAME, stages=["Staging"])
            if versions:
                mv = versions[0]
                return {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "tags": mv.tags,
                }
        except Exception:
            pass
        return None

    def get_all_versions(self) -> list[dict]:
        """List all versions with their roles and metrics."""
        try:
            versions = self.client.search_model_versions(f"name='{MODEL_NAME}'")
            result = []
            for v in versions:
                result.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "tags": v.tags,
                    "role": v.tags.get("role", "none"),
                })
            return sorted(result, key=lambda x: int(x["version"]), reverse=True)
        except Exception:
            return []

    def load_model(self, version: str):
        """Load a specific model version."""
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")

    def load_champion_model(self):
        """Load the current champion model."""
        info = self.get_champion()
        if info:
            return self.load_model(info["version"])
        return None

    def should_rollback(self, version: str, X_val: pd.DataFrame, y_val: pd.Series,
                        recall_threshold: float = 0.75) -> tuple[bool, str]:
        """Check if a model version should be rolled back."""
        try:
            model = self.load_model(version)
            metrics = self._eval_metrics(model, X_val, y_val)
            if metrics["recall"] < recall_threshold:
                return True, f"Recall {metrics['recall']:.4f} < threshold {recall_threshold}"
            champion = self.get_champion()
            if champion and champion["version"] != version:
                champ_model = self.load_model(champion["version"])
                champ_metrics = self._eval_metrics(champ_model, X_val, y_val)
                if metrics["auc_roc"] < champ_metrics["auc_roc"] - 0.05:
                    return True, f"AUC {metrics['auc_roc']:.4f} < champion {champ_metrics['auc_roc']:.4f} - 0.05"
            return False, "OK"
        except Exception as e:
            return True, f"Load failed: {e}"

    def compare_and_decide(self, X_val: pd.DataFrame, y_val: pd.Series,
                           new_version: str) -> dict:
        """Compare new model vs champion and decide promotion."""
        new_model = self.load_model(new_version)
        new_metrics = self._eval_metrics(new_model, X_val, y_val)

        champion_info = self.get_champion()
        decision = {
            "new_version": new_version,
            "new_metrics": new_metrics,
            "champion_version": None,
            "champion_metrics": None,
            "promoted": True,
            "reason": "first_model",
        }

        if new_metrics["auc_roc"] < MIN_AUC:
            decision["promoted"] = False
            decision["reason"] = f"AUC {new_metrics['auc_roc']:.4f} < min {MIN_AUC}"
            return decision

        self.promote_to_challenger(new_version)

        if champion_info:
            champ_model = self.load_model(champion_info["version"])
            champ_metrics = self._eval_metrics(champ_model, X_val, y_val)
            decision["champion_version"] = champion_info["version"]
            decision["champion_metrics"] = champ_metrics

            new_better_recall = new_metrics["recall"] >= champ_metrics["recall"]
            new_better_auc = new_metrics["auc_roc"] >= champ_metrics["auc_roc"]

            if new_better_recall or new_better_auc:
                self.promote_to_champion(new_version, champion_info["version"])
                decision["promoted"] = True
                decision["reason"] = "better_performance"
            else:
                decision["promoted"] = False
                decision["reason"] = "champion_better"
                log.info("Champion v%s outperforms challenger v%s — keeping champion",
                         champion_info["version"], new_version)
        else:
            self.promote_to_champion(new_version)
            decision["promoted"] = True
            decision["reason"] = "first_champion"

        return decision

    def _eval_metrics(self, model, X_val, y_val) -> dict:
        y_proba = model.predict(X_val)
        if hasattr(y_proba, "ndim") and y_proba.ndim > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "auc_roc": float(roc_auc_score(y_val, y_proba)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "n_samples": len(y_val),
        }

    def cleanup_old_versions(self, keep_last: int = 10):
        """Archive versions beyond the most recent N."""
        versions = self.get_all_versions()
        to_archive = versions[keep_last:]
        for v in to_archive:
            try:
                if v["stage"] not in ("Production", "Staging"):
                    self.client.transition_model_version_stage(
                        name=MODEL_NAME, version=v["version"], stage="Archived",
                    )
                    log.info("Cleaned up v%s -> Archived", v["version"])
            except Exception:
                pass


registry = ModelRegistry()
