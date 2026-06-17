#!/usr/bin/env python3
"""Optuna tuning helpers with optional MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT_NAME = "hospitalisation_prediction"
TARGET_COL = "HOSPITALIZED_IN_6M"
RANDOM_STATE = 42

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mlops_hospitalisation_matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="The filesystem tracking backend.*",
    category=FutureWarning,
)


def _as_series(y: Any) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if TARGET_COL in y.columns:
            y = y[TARGET_COL]
        else:
            y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        y = pd.Series(y)

    return pd.to_numeric(y, errors="coerce").fillna(0).astype(int).reset_index(drop=True)


def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.reset_index(drop=True)


def _safe_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _json_safe(data: Dict[str, Any]) -> Dict[str, Any]:
    safe = {}
    for key, value in data.items():
        if isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        else:
            safe[key] = value
    return safe


def load_training_data(
    features_dir: Path = FEATURES_DIR,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    paths = {
        "X_train": features_dir / "X_train.parquet",
        "y_train": features_dir / "y_train.parquet",
        "X_val": features_dir / "X_val.parquet",
        "y_val": features_dir / "y_val.parquet",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers features manquants: "
            + ", ".join(missing)
            + ". Lance d'abord la phase preprocessing/feature engineering."
        )

    X_train = _sanitize_features(pd.read_parquet(paths["X_train"]))
    X_val = _sanitize_features(pd.read_parquet(paths["X_val"]))
    y_train = _as_series(pd.read_parquet(paths["y_train"]))
    y_val = _as_series(pd.read_parquet(paths["y_val"]))

    if len(X_train) != len(y_train):
        raise ValueError(f"Mismatch train: X={len(X_train)} y={len(y_train)}")
    if len(X_val) != len(y_val):
        raise ValueError(f"Mismatch val: X={len(X_val)} y={len(y_val)}")

    return X_train, y_train, X_val, y_val


class OptunaTuner:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: Iterable[int],
        X_val: pd.DataFrame,
        y_val: Iterable[int],
        *,
        metric: str = "f1",
        threshold: float = 0.5,
        enable_mlflow: bool = True,
        tracking_uri: str = MLFLOW_URI,
        experiment_name: str = EXPERIMENT_NAME,
        random_state: int = RANDOM_STATE,
    ):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna n'est pas installe. Active le venv puis installe requirements.txt.")

        if metric not in {"f1", "auc"}:
            raise ValueError("metric doit etre 'f1' ou 'auc'.")

        self.X_train = _sanitize_features(X_train)
        self.y_train = _as_series(y_train)
        self.X_val = _sanitize_features(X_val).reindex(columns=self.X_train.columns, fill_value=0)
        self.y_val = _as_series(y_val)
        self.metric = metric
        self.threshold = threshold
        self.random_state = random_state
        self.enable_mlflow = bool(enable_mlflow and MLFLOW_AVAILABLE)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.scale_pos_weight = self._scale_pos_weight(self.y_train)
        self.last_studies: Dict[str, Any] = {}

        if len(self.X_train) != len(self.y_train):
            raise ValueError(f"Mismatch train: X={len(self.X_train)} y={len(self.y_train)}")
        if len(self.X_val) != len(self.y_val):
            raise ValueError(f"Mismatch val: X={len(self.X_val)} y={len(self.y_val)}")

        self._setup_mlflow()

    @staticmethod
    def _scale_pos_weight(y: pd.Series) -> float:
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        return float(negatives / max(positives, 1))

    def _setup_mlflow(self) -> None:
        if not self.enable_mlflow:
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow non disponible: tracking desactive.")
            return

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow experiment: %s", self.experiment_name)
        except Exception as exc:
            logger.warning("MLflow setup echoue: %s", exc)
            self.enable_mlflow = False

    @contextmanager
    def _mlflow_run(self, run_name: str, *, nested: bool = False):
        if not self.enable_mlflow:
            yield
            return

        run_started = False
        try:
            mlflow.start_run(
                run_name=run_name,
                nested=nested or mlflow.active_run() is not None,
            )
            run_started = True
        except Exception as exc:
            logger.warning("MLflow start_run echoue pour %s: %s", run_name, exc)
            yield
            return

        try:
            yield
        finally:
            if run_started:
                try:
                    mlflow.end_run()
                except Exception as exc:
                    logger.warning("MLflow end_run echoue pour %s: %s", run_name, exc)

    def _log_params(self, params: Dict[str, Any]) -> None:
        if not self.enable_mlflow:
            return
        try:
            mlflow.log_params(_json_safe(params))
        except Exception as exc:
            logger.warning("MLflow log_params echoue: %s", exc)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.enable_mlflow:
            return
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None and np.isfinite(float(value))
        }
        if not clean:
            return
        try:
            mlflow.log_metrics(clean)
        except Exception as exc:
            logger.warning("MLflow log_metrics echoue: %s", exc)

    def _score(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        if self.metric == "auc":
            score = _safe_auc(y_true, y_proba)
        else:
            y_pred = (y_proba >= self.threshold).astype(int)
            score = float(f1_score(y_true, y_pred, zero_division=0))

        if not np.isfinite(score):
            return 0.0
        return float(score)

    def objective_xgb(self, trial: Any) -> float:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost n'est pas installe.")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": self.scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": -1,
            "early_stopping_rounds": 10,
        }

        with self._mlflow_run(f"xgboost_trial_{trial.number}", nested=True):
            model = xgb.XGBClassifier(**params)
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )
            val_proba = model.predict_proba(self.X_val)[:, 1]
            score = self._score(self.y_val, val_proba)
            auc = _safe_auc(self.y_val, val_proba)

            self._log_params({"model": "xgboost", "trial": trial.number, **params})
            self._log_metrics({self.metric: score, "auc": auc})

        return score

    def objective_lgb(self, trial: Any) -> float:
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM n'est pas installe.")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 128),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "scale_pos_weight": self.scale_pos_weight,
            "objective": "binary",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": -1,
            "force_row_wise": True,
        }

        with self._mlflow_run(f"lightgbm_trial_{trial.number}", nested=True):
            model = lgb.LGBMClassifier(**params)
            callbacks = [
                lgb.early_stopping(10, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=callbacks,
            )
            val_proba = model.predict_proba(self.X_val)[:, 1]
            score = self._score(self.y_val, val_proba)
            auc = _safe_auc(self.y_val, val_proba)

            self._log_params({"model": "lightgbm", "trial": trial.number, **params})
            self._log_metrics({self.metric: score, "auc": auc})

        return score

    def _optimize(self, model_name: str, objective: Any, n_trials: int) -> Tuple[Dict[str, Any], float]:
        logger.info("Start %s optimization with %d trials", model_name, n_trials)

        with self._mlflow_run(f"optuna_{model_name}"):
            self._log_params(
                {
                    "model": model_name,
                    "n_trials": n_trials,
                    "metric": self.metric,
                    "threshold": self.threshold,
                    "scale_pos_weight": self.scale_pos_weight,
                }
            )

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            self.last_studies[model_name] = study

            best_params = dict(study.best_params)
            best_value = float(study.best_value)

            self._log_params({f"best_{key}": value for key, value in best_params.items()})
            self._log_metrics({"best_score": best_value})

        return best_params, best_value

    def optimize_xgb(self, n_trials: int = 50) -> Tuple[Dict[str, Any], float]:
        return self._optimize("xgboost", self.objective_xgb, n_trials)

    def optimize_lgb(self, n_trials: int = 50) -> Tuple[Dict[str, Any], float]:
        return self._optimize("lightgbm", self.objective_lgb, n_trials)


def run_optimization(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    X_train, y_train, X_val, y_val = load_training_data(Path(args.features_dir))

    models = set(args.models)
    if "all" in models:
        models = {"xgb", "lgb"}

    tuner = OptunaTuner(
        X_train,
        y_train,
        X_val,
        y_val,
        metric=args.metric,
        threshold=args.threshold,
        enable_mlflow=not args.no_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )

    results: Dict[str, Dict[str, Any]] = {}

    if "xgb" in models:
        best_params, best_score = tuner.optimize_xgb(n_trials=args.n_trials)
        results["xgboost"] = {"best_params": best_params, f"best_{args.metric}": best_score}

    if "lgb" in models:
        best_params, best_score = tuner.optimize_lgb(n_trials=args.n_trials)
        results["lightgbm"] = {"best_params": best_params, f"best_{args.metric}": best_score}

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved best params to %s", output_path)

    if not args.no_mlflow and MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        try:
            mlflow.log_dict(results, "best_params.json")
        except Exception as exc:
            logger.warning("MLflow log_dict echoue: %s", exc)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimisation Optuna avec tracking MLflow.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--models", nargs="+", choices=["xgb", "lgb", "all"], default=["xgb"])
    parser.add_argument("--metric", choices=["f1", "auc"], default="f1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--output", default=str(MODELS_DIR / "best_params.json"))
    parser.add_argument("--tracking-uri", default=MLFLOW_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--no-mlflow", action="store_true", help="Desactive le tracking MLflow.")
    return parser


def main() -> Dict[str, Dict[str, Any]]:
    args = build_parser().parse_args()
    return run_optimization(args)


if __name__ == "__main__":
    main()
