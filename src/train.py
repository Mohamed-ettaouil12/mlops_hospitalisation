#!/usr/bin/env python3
"""Recall-first training pipeline with MLflow tracking."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mlops_hospitalisation_matplotlib"))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except Exception:
    SMOTE = None
    SMOTE_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    SHAP_AVAILABLE = True
except Exception:
    plt = None
    shap = None
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="The filesystem tracking backend.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The filesystem model registry backend.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

MLFLOW_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT_NAME = "maximisation_recall"

TARGET_COL = "HOSPITALIZED_IN_6M"
RANDOM_STATE = 42
N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "50"))
RECALL_TARGET = float(os.environ.get("RECALL_TARGET", "0.95"))
MIN_AUC_FOR_SELECTION = float(os.environ.get("MIN_AUC_FOR_SELECTION", "0.88"))
OPTUNA_RECALL_THRESHOLD = float(os.environ.get("OPTUNA_RECALL_THRESHOLD", "0.30"))
ENSEMBLE_THRESHOLD = float(os.environ.get("ENSEMBLE_THRESHOLD", "0.25"))

BEST_RECALL_MODEL_PATH = MODELS_DIR / "best_recall_model.pkl"
BEST_THRESHOLD_PATH = MODELS_DIR / "best_threshold.json"
BEST_RECALL_INFO_PATH = MODELS_DIR / "best_recall_model_info.json"
RECALL_REPORT_PATH = REPORTS_DIR / "recall_training_metrics.json"
THRESHOLD_ANALYSIS_PATH = REPORTS_DIR / "threshold_analysis.json"
THRESHOLD_ANALYSIS_CSV_PATH = REPORTS_DIR / "threshold_analysis.csv"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_importance_best_recall.json"
SHAP_LOCAL_PATH = REPORTS_DIR / "shap_local_high_risk.json"
SHAP_IMPORTANCE_CSV_PATH = REPORTS_DIR / "shap_importance_best_recall.csv"
PR_CURVE_PATH = FIGURES_DIR / "pr_curve_best_recall.png"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary_best_recall.png"
SHAP_IMPORTANCE_FIG_PATH = FIGURES_DIR / "shap_importance_best_recall.png"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    technique: str
    model: Any
    threshold: float
    val_metrics: Dict[str, float]
    val_confusion_matrix: Dict[str, int]
    params: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Optional[Dict[str, float]] = None
    test_confusion_matrix: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Data and metrics helpers
# ---------------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def safe_target(path: Path) -> pd.Series:
    require_file(path)
    df = pd.read_parquet(path)

    if isinstance(df, pd.DataFrame):
        if TARGET_COL in df.columns:
            y = df[TARGET_COL]
        else:
            y = df.iloc[:, 0]
    else:
        y = df

    return pd.to_numeric(y, errors="coerce").fillna(0).astype(int).reset_index(drop=True)


def sanitize_features(X: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    X = X.copy()
    if feature_cols is not None:
        X = X.reindex(columns=list(feature_cols), fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.reset_index(drop=True)


def load_feature_names() -> Optional[list[str]]:
    pkl_path = MODELS_DIR / "feature_names.pkl"
    csv_path = FEATURES_DIR / "feature_names.csv"

    if pkl_path.exists():
        return list(joblib.load(pkl_path))

    if csv_path.exists():
        return pd.read_csv(csv_path).iloc[:, 0].astype(str).tolist()

    return None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    log.info("Chargement des splits temporels train/val/test...")

    paths = {
        "X_train": FEATURES_DIR / "X_train.parquet",
        "X_val": FEATURES_DIR / "X_val.parquet",
        "X_test": FEATURES_DIR / "X_test.parquet",
        "y_train": FEATURES_DIR / "y_train.parquet",
        "y_val": FEATURES_DIR / "y_val.parquet",
        "y_test": FEATURES_DIR / "y_test.parquet",
    }

    for path in paths.values():
        require_file(path)

    X_train = pd.read_parquet(paths["X_train"])
    X_val = pd.read_parquet(paths["X_val"])
    X_test = pd.read_parquet(paths["X_test"])

    feature_cols = load_feature_names()
    if feature_cols is None:
        feature_cols = X_train.columns.astype(str).tolist()

    X_train = sanitize_features(X_train, feature_cols)
    X_val = sanitize_features(X_val, feature_cols)
    X_test = sanitize_features(X_test, feature_cols)

    y_train = safe_target(paths["y_train"])
    y_val = safe_target(paths["y_val"])
    y_test = safe_target(paths["y_test"])

    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        if len(X_split) != len(y_split):
            raise ValueError(f"Mismatch {split_name}: X={len(X_split)} y={len(y_split)}")

    log.info("  Train 2008 : %s | positifs %.2f%%", X_train.shape, y_train.mean() * 100)
    log.info("  Val   2009 : %s | positifs %.2f%%", X_val.shape, y_val.mean() * 100)
    log.info("  Test  2010 : %s | positifs %.2f%%", X_test.shape, y_test.mean() * 100)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def safe_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc": safe_auc(y_true, y_proba),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def confusion_matrix_dict(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def optimize_threshold_for_recall(
    y_true: pd.Series,
    y_proba: np.ndarray,
    recall_target: float = RECALL_TARGET,
) -> float:
    """Pick the threshold with recall >= target and best precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.5

    aligned_precisions = precisions[:-1]
    aligned_recalls = recalls[:-1]
    valid_indices = np.where(aligned_recalls >= recall_target)[0]

    if len(valid_indices) == 0:
        best_idx = int(np.nanargmax(aligned_recalls))
        return float(thresholds[best_idx])

    valid_precisions = aligned_precisions[valid_indices]
    best_precision = np.nanmax(valid_precisions)
    best_candidates = valid_indices[np.where(np.isclose(valid_precisions, best_precision))[0]]

    # Tie-breaker: keep the highest threshold to reduce false positives.
    best_idx = int(best_candidates[np.argmax(thresholds[best_candidates])])
    return float(thresholds[best_idx])


def optimize_threshold_from_proba(
    y_true: pd.Series,
    y_proba: np.ndarray,
    beta: float = 1.0,
) -> float:
    """Backward-compatible F-beta threshold helper used by older scripts."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5
    f_scores = (1 + beta**2) * precisions * recalls / ((beta**2 * precisions) + recalls + 1e-9)
    f_scores = f_scores[:-1]
    return float(thresholds[int(np.nanargmax(f_scores))])


def class_ratio(y_train: pd.Series) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return float(neg / max(pos, 1))


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def setup_mlflow() -> None:
    if not MLFLOW_AVAILABLE:
        log.warning("MLflow non disponible : tracking desactive.")
        return

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow experiment : %s", EXPERIMENT_NAME)


@contextmanager
def mlflow_run(run_name: str, nested: bool = True):
    if not MLFLOW_AVAILABLE:
        yield
        return

    active = mlflow.active_run() is not None
    mlflow.start_run(run_name=run_name, nested=nested or active)
    try:
        yield
    finally:
        mlflow.end_run()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_safe)


def mlflow_log_candidate(result: CandidateResult) -> None:
    if not MLFLOW_AVAILABLE:
        return

    with mlflow_run(result.name):
        mlflow.log_param("technique", result.technique)
        mlflow.log_param("seuil", float(result.threshold))
        for key, value in result.params.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                mlflow.log_param(key, value)

        mlflow.log_metric("recall", result.val_metrics["recall"])
        mlflow.log_metric("precision", result.val_metrics["precision"])
        mlflow.log_metric("f1", result.val_metrics["f1"])
        mlflow.log_metric("auc_roc", result.val_metrics["auc_roc"])
        mlflow.log_metric("faux_negatifs", result.val_confusion_matrix["fn"])


def evaluate_candidate(
    name: str,
    technique: str,
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float,
    params: Optional[Dict[str, Any]] = None,
    log_to_mlflow: bool = True,
) -> CandidateResult:
    y_proba = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_proba, threshold)
    cm = confusion_matrix_dict(y_val, y_proba, threshold)

    result = CandidateResult(
        name=name,
        technique=technique,
        model=model,
        threshold=float(threshold),
        val_metrics=metrics,
        val_confusion_matrix=cm,
        params=params or {},
    )

    if log_to_mlflow:
        mlflow_log_candidate(result)

    log.info(
        "%-24s | recall=%.4f precision=%.4f f1=%.4f auc=%.4f seuil=%.4f FN=%d",
        name,
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        metrics["auc_roc"],
        threshold,
        cm["fn"],
    )
    return result


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def ensure_training_dependencies() -> None:
    missing = []
    if not XGBOOST_AVAILABLE:
        missing.append("xgboost")
    if not LIGHTGBM_AVAILABLE:
        missing.append("lightgbm")
    if not OPTUNA_AVAILABLE:
        missing.append("optuna")
    if not SMOTE_AVAILABLE:
        missing.append("imbalanced-learn")

    if missing:
        raise ImportError(
            "Dependances manquantes: "
            + ", ".join(missing)
            + ". Installe requirements.txt dans le venv avant d'executer train.py."
        )


def base_xgb_params() -> Dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]) -> Any:
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]) -> Any:
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def optimize_xgb_for_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    log.info("Optuna XGBoost recall (%d essais, seuil objectif %.2f)...", N_OPTUNA_TRIALS, OPTUNA_RECALL_THRESHOLD)

    def objective(trial: Any) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 20.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = fit_xgboost(X_train, y_train, params)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= OPTUNA_RECALL_THRESHOLD).astype(int)
        return float(recall_score(y_val, y_pred, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )

    if MLFLOW_AVAILABLE:
        with mlflow_run("Optuna_Recall_XGBoost"):
            mlflow.log_param("technique", "optuna_recall_xgboost")
            mlflow.log_param("n_trials", N_OPTUNA_TRIALS)
            mlflow.log_param("objective_threshold", OPTUNA_RECALL_THRESHOLD)
            mlflow.log_metric("best_recall", float(study.best_value))
            for key, value in best_params.items():
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    mlflow.log_param(key, value)

    return best_params


def optimize_lgb_for_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    log.info("Optuna LightGBM recall (%d essais, seuil objectif %.2f)...", N_OPTUNA_TRIALS, OPTUNA_RECALL_THRESHOLD)

    def objective(trial: Any) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 20.0),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
            "force_row_wise": True,
        }
        model = fit_lightgbm(X_train, y_train, params)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= OPTUNA_RECALL_THRESHOLD).astype(int)
        return float(recall_score(y_val, y_pred, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
            "force_row_wise": True,
        }
    )

    if MLFLOW_AVAILABLE:
        with mlflow_run("Optuna_Recall_LightGBM"):
            mlflow.log_param("technique", "optuna_recall_lightgbm")
            mlflow.log_param("n_trials", N_OPTUNA_TRIALS)
            mlflow.log_param("objective_threshold", OPTUNA_RECALL_THRESHOLD)
            mlflow.log_metric("best_recall", float(study.best_value))
            for key, value in best_params.items():
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    mlflow.log_param(key, value)

    return best_params


def smote_resample_train(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    log.info("Application SMOTE uniquement sur train 2008...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=TARGET_COL).astype(int).reset_index(drop=True)
    X_res = sanitize_features(X_res, X_train.columns)

    log.info("  Avant SMOTE : %s | positifs %.2f%%", X_train.shape, y_train.mean() * 100)
    log.info("  Apres SMOTE : %s | positifs %.2f%%", X_res.shape, y_res.mean() * 100)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Selection and reporting
# ---------------------------------------------------------------------------

def add_test_metrics(result: CandidateResult, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_proba = result.model.predict_proba(X_test)[:, 1]
    result.test_metrics = compute_metrics(y_test, y_proba, result.threshold)
    result.test_confusion_matrix = confusion_matrix_dict(y_test, y_proba, result.threshold)


def select_best_recall_candidate(candidates: Dict[str, CandidateResult]) -> CandidateResult:
    valid = [
        candidate
        for candidate in candidates.values()
        if np.isfinite(candidate.val_metrics["auc_roc"])
        and candidate.val_metrics["auc_roc"] >= MIN_AUC_FOR_SELECTION
    ]

    if not valid:
        raise ValueError(
            f"Aucun candidat ne respecte AUC validation >= {MIN_AUC_FOR_SELECTION:.2f}."
        )

    return max(
        valid,
        key=lambda c: (
            c.val_metrics["recall"],
            c.val_metrics["precision"],
            c.val_metrics["f1"],
            c.val_metrics["auc_roc"],
        ),
    )


def candidate_to_dict(result: CandidateResult, include_params: bool = True) -> Dict[str, Any]:
    data = {
        "name": result.name,
        "technique": result.technique,
        "threshold": result.threshold,
        "validation_metrics": result.val_metrics,
        "validation_confusion_matrix": result.val_confusion_matrix,
        "test_metrics": result.test_metrics,
        "test_confusion_matrix": result.test_confusion_matrix,
    }
    if include_params:
        data["params"] = result.params
    return data


def format_comparison_table(candidates: Dict[str, CandidateResult]) -> str:
    ordered_names = [
        "Baseline XGBoost",
        "Seuil optimal",
        "scale_pos_weight",
        "Optuna Recall XGB",
        "SMOTE",
        "Ensemble seuil=0.25",
    ]
    lines = [
        "| Technique            | Recall | Precision | F1    | AUC   | Seuil | FN |",
        "|----------------------|--------|-----------|-------|-------|-------|----|",
    ]

    for name in ordered_names:
        if name not in candidates:
            continue
        result = candidates[name]
        m = result.val_metrics
        cm = result.val_confusion_matrix
        lines.append(
            f"| {name:<20} | {m['recall']:.3f}  | {m['precision']:.3f}     | "
            f"{m['f1']:.3f} | {m['auc_roc']:.3f} | {result.threshold:.3f} | {cm['fn']} |"
        )

    return "\n".join(lines)


def threshold_grid(selected_threshold: float) -> list[float]:
    base = [0.01, 0.02, 0.03, 0.04, 0.05]
    base.extend(np.round(np.arange(0.10, 0.96, 0.05), 4).tolist())
    base.append(float(selected_threshold))
    return sorted({float(np.round(threshold, 6)) for threshold in base if 0.0 < threshold < 1.0})


def build_threshold_analysis(
    best: CandidateResult,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    rows = []
    thresholds = threshold_grid(best.threshold)
    split_payloads = {
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split_name, (X_split, y_split) in split_payloads.items():
        y_proba = best.model.predict_proba(X_split)[:, 1]

        for threshold in thresholds:
            metrics = compute_metrics(y_split, y_proba, threshold)
            cm = confusion_matrix_dict(y_split, y_proba, threshold)
            rows.append(
                {
                    "split": split_name,
                    "threshold": threshold,
                    "auc_roc": metrics["auc_roc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "true_negative": cm["tn"],
                    "false_positive": cm["fp"],
                    "false_negative": cm["fn"],
                    "true_positive": cm["tp"],
                    "predicted_positive": int(cm["fp"] + cm["tp"]),
                }
            )

    payload = {
        "selected_model": best.name,
        "selected_threshold": best.threshold,
        "recall_target": RECALL_TARGET,
        "thresholds": thresholds,
        "rows": rows,
    }

    save_json(payload, THRESHOLD_ANALYSIS_PATH)
    pd.DataFrame(rows).to_csv(THRESHOLD_ANALYSIS_CSV_PATH, index=False)
    return payload


def plot_precision_recall_diagnostics(
    best: CandidateResult,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Optional[Path]:
    if plt is None:
        log.warning("Matplotlib non disponible : courbe PR ignoree.")
        return None

    plt.figure(figsize=(9, 6))

    for split_name, X_split, y_split, color in (
        ("Validation", X_val, y_val, "#2563eb"),
        ("Test", X_test, y_test, "#16a34a"),
    ):
        y_proba = best.model.predict_proba(X_split)[:, 1]
        precisions, recalls, _thresholds = precision_recall_curve(y_split, y_proba)
        metrics = compute_metrics(y_split, y_proba, best.threshold)

        plt.plot(
            recalls,
            precisions,
            label=f"{split_name} PR (AUC ROC={metrics['auc_roc']:.3f})",
            color=color,
            linewidth=2,
        )
        plt.scatter(
            metrics["recall"],
            metrics["precision"],
            color=color,
            edgecolor="black",
            s=80,
            zorder=4,
            label=f"{split_name} seuil={best.threshold:.3f}",
        )

    plt.axvline(RECALL_TARGET, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Recall cible={RECALL_TARGET:.2f}")
    plt.title("Courbe Precision-Recall - meilleur modele recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.01)
    plt.ylim(0.0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    return PR_CURVE_PATH


def select_shap_estimator(model: Any) -> Tuple[Optional[Any], Optional[str], str]:
    if hasattr(model, "named_estimators_"):
        named_estimators = model.named_estimators_
        for estimator_name in ("xgb", "lgbm"):
            if estimator_name in named_estimators:
                return (
                    named_estimators[estimator_name],
                    estimator_name,
                    "Le meilleur modele est un VotingClassifier; SHAP est calcule sur son sous-modele arbre.",
                )

    model_class_name = model.__class__.__name__.lower()
    if "xgb" in model_class_name or "lgbm" in model_class_name or "lightgbm" in model_class_name:
        return model, model.__class__.__name__, "SHAP calcule directement sur le modele arbre."

    return None, None, "Aucun sous-modele arbre compatible SHAP trouve."


def normalize_shap_values(raw_shap_values: Any) -> np.ndarray:
    shap_values = raw_shap_values
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values.mean(axis=2)

    return shap_values


def build_shap_diagnostics(
    best: CandidateResult,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = 500,
    n_local_cases: int = 5,
) -> Dict[str, Any]:
    if not SHAP_AVAILABLE:
        return {
            "available": False,
            "reason": "shap ou matplotlib non disponible.",
            "artifacts": [],
        }

    shap_model, shap_model_name, note = select_shap_estimator(best.model)
    if shap_model is None:
        return {
            "available": False,
            "reason": note,
            "artifacts": [],
        }

    try:
        y_proba = best.model.predict_proba(X_val)[:, 1]
        high_risk_positions = np.argsort(y_proba)[-n_local_cases:][::-1].tolist()
        base_positions = list(range(min(sample_size, len(X_val))))
        sample_positions = sorted(set(base_positions + high_risk_positions))
        sample_position_lookup = {position: idx for idx, position in enumerate(sample_positions)}
        X_sample = X_val.iloc[sample_positions].copy()

        explainer = shap.TreeExplainer(shap_model)
        shap_values = normalize_shap_values(explainer.shap_values(X_sample))

        if shap_values.shape[1] != X_sample.shape[1]:
            raise ValueError(
                f"Shape SHAP invalide: shap={shap_values.shape}, X={X_sample.shape}"
            )

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_sample.columns.astype(str),
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(SHAP_IMPORTANCE_CSV_PATH, index=False)
        save_json(
            {
                "model_explained": shap_model_name,
                "note": note,
                "sample_size": len(X_sample),
                "importance": importance_df.to_dict(orient="records"),
            },
            SHAP_IMPORTANCE_PATH,
        )

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH, dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(SHAP_IMPORTANCE_FIG_PATH, dpi=160, bbox_inches="tight")
        plt.close()

        local_cases = []
        for position in high_risk_positions:
            sample_idx = sample_position_lookup[position]
            contributions = (
                pd.DataFrame(
                    {
                        "feature": X_sample.columns.astype(str),
                        "value": X_sample.iloc[sample_idx].to_numpy(),
                        "shap_value": shap_values[sample_idx],
                        "abs_shap_value": np.abs(shap_values[sample_idx]),
                    }
                )
                .sort_values("abs_shap_value", ascending=False)
                .head(10)
            )
            local_cases.append(
                {
                    "validation_position": int(position),
                    "predicted_probability": float(y_proba[position]),
                    "top_contributions": contributions.to_dict(orient="records"),
                }
            )

        save_json(
            {
                "model_explained": shap_model_name,
                "note": note,
                "local_cases": local_cases,
            },
            SHAP_LOCAL_PATH,
        )

        artifacts = [
            SHAP_SUMMARY_PATH,
            SHAP_IMPORTANCE_FIG_PATH,
            SHAP_IMPORTANCE_PATH,
            SHAP_IMPORTANCE_CSV_PATH,
            SHAP_LOCAL_PATH,
        ]

        return {
            "available": True,
            "model_explained": shap_model_name,
            "note": note,
            "sample_size": len(X_sample),
            "top_features": importance_df.head(15).to_dict(orient="records"),
            "artifacts": [str(path) for path in artifacts],
        }

    except Exception as exc:
        log.warning("Generation SHAP echouee : %s", exc)
        return {
            "available": False,
            "reason": str(exc),
            "artifacts": [],
        }


def build_model_diagnostics(
    best: CandidateResult,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
) -> Dict[str, Any]:
    log.info("Generation diagnostics seuil PR et SHAP...")
    threshold_analysis = build_threshold_analysis(best, X_val, y_val, X_test, y_test)
    pr_curve_path = plot_precision_recall_diagnostics(best, X_val, y_val, X_test, y_test)
    shap_diagnostics = build_shap_diagnostics(best, X_val, feature_cols)

    artifact_paths = [
        THRESHOLD_ANALYSIS_PATH,
        THRESHOLD_ANALYSIS_CSV_PATH,
    ]
    if pr_curve_path is not None:
        artifact_paths.append(pr_curve_path)
    artifact_paths.extend(Path(path) for path in shap_diagnostics.get("artifacts", []))

    return {
        "threshold_analysis_path": str(THRESHOLD_ANALYSIS_PATH),
        "threshold_analysis_csv_path": str(THRESHOLD_ANALYSIS_CSV_PATH),
        "pr_curve_path": str(pr_curve_path) if pr_curve_path else None,
        "threshold_analysis_preview": threshold_analysis["rows"][:8],
        "shap": shap_diagnostics,
        "artifact_paths": [str(path) for path in artifact_paths if Path(path).exists()],
    }


def save_best_outputs(
    best: CandidateResult,
    candidates: Dict[str, CandidateResult],
    feature_cols: list[str],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, float]:
    joblib.dump(best.model, BEST_RECALL_MODEL_PATH)

    threshold_payload = {
        "threshold": best.threshold,
        "model": best.name,
        "technique": best.technique,
        "recall_target": RECALL_TARGET,
        "min_auc_for_selection": MIN_AUC_FOR_SELECTION,
    }
    save_json(threshold_payload, BEST_THRESHOLD_PATH)

    report = {
        "best_model": best.name,
        "best_model_path": str(BEST_RECALL_MODEL_PATH),
        "best_threshold_path": str(BEST_THRESHOLD_PATH),
        "best_threshold": best.threshold,
        "best_technique": best.technique,
        "recall_target": RECALL_TARGET,
        "min_auc_for_selection": MIN_AUC_FOR_SELECTION,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "best": candidate_to_dict(best),
        "all_candidates": {
            name: candidate_to_dict(candidate)
            for name, candidate in candidates.items()
        },
        "diagnostics": diagnostics or {},
    }

    save_json(report, BEST_RECALL_INFO_PATH)
    save_json(report, RECALL_REPORT_PATH)

    if MLFLOW_AVAILABLE:
        with mlflow_run("Best_Recall_Model"):
            mlflow.log_param("technique", best.technique)
            mlflow.log_param("model", best.name)
            mlflow.log_param("seuil", best.threshold)
            mlflow.log_metric("recall", best.val_metrics["recall"])
            mlflow.log_metric("precision", best.val_metrics["precision"])
            mlflow.log_metric("f1", best.val_metrics["f1"])
            mlflow.log_metric("auc_roc", best.val_metrics["auc_roc"])
            mlflow.log_metric("faux_negatifs", best.val_confusion_matrix["fn"])
            mlflow.log_artifact(str(BEST_THRESHOLD_PATH))
            mlflow.log_artifact(str(BEST_RECALL_INFO_PATH))
            if diagnostics:
                for artifact_path in diagnostics.get("artifact_paths", []):
                    artifact = Path(artifact_path)
                    if artifact.exists():
                        mlflow.log_artifact(str(artifact))
            try:
                mlflow.sklearn.log_model(best.model, name="model")
            except TypeError:
                mlflow.sklearn.log_model(best.model, "model")

    return best.model, best.name, float(best.val_metrics["auc_roc"])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> Tuple[Any, str, float]:
    ensure_training_dependencies()
    setup_mlflow()

    log.info("=" * 78)
    log.info("PIPELINE MAXIMISATION RECALL - DEBUT")
    log.info("Objectif : recall >= %.2f avec AUC validation >= %.2f", RECALL_TARGET, MIN_AUC_FOR_SELECTION)
    log.info("=" * 78)

    parent_run = None
    if MLFLOW_AVAILABLE:
        parent_run = mlflow.start_run(run_name="Pipeline_Maximisation_Recall")
        mlflow.log_param("experiment_goal", "maximize_recall")
        mlflow.log_param("recall_target", RECALL_TARGET)
        mlflow.log_param("min_auc_for_selection", MIN_AUC_FOR_SELECTION)
        mlflow.log_param("n_optuna_trials", N_OPTUNA_TRIALS)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
        scale_pos_weight = class_ratio(y_train)
        log.info("scale_pos_weight calcule depuis train : %.4f", scale_pos_weight)

        candidates: Dict[str, CandidateResult] = {}

        # Baseline XGBoost at threshold 0.5.
        baseline_params = base_xgb_params()
        baseline_model = fit_xgboost(X_train, y_train, baseline_params)
        candidates["Baseline XGBoost"] = evaluate_candidate(
            name="Baseline XGBoost",
            technique="baseline_xgboost",
            model=baseline_model,
            X_val=X_val,
            y_val=y_val,
            threshold=0.5,
            params=baseline_params,
        )

        # Technique 1: choose threshold for recall target using baseline probabilities.
        baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
        recall_threshold = optimize_threshold_for_recall(y_val, baseline_val_proba, RECALL_TARGET)
        candidates["Seuil optimal"] = evaluate_candidate(
            name="Seuil optimal",
            technique="seuil_optimal",
            model=baseline_model,
            X_val=X_val,
            y_val=y_val,
            threshold=recall_threshold,
            params={"recall_target": RECALL_TARGET, **baseline_params},
        )

        # Technique 2: XGBoost with scale_pos_weight from train only.
        weighted_params = base_xgb_params()
        weighted_params["scale_pos_weight"] = scale_pos_weight
        weighted_model = fit_xgboost(X_train, y_train, weighted_params)
        weighted_threshold = optimize_threshold_for_recall(
            y_val,
            weighted_model.predict_proba(X_val)[:, 1],
            RECALL_TARGET,
        )
        candidates["scale_pos_weight"] = evaluate_candidate(
            name="scale_pos_weight",
            technique="scale_pos_weight",
            model=weighted_model,
            X_val=X_val,
            y_val=y_val,
            threshold=weighted_threshold,
            params=weighted_params,
        )

        # Technique 3: Optuna objective centered on recall.
        best_xgb_params = optimize_xgb_for_recall(X_train, y_train, X_val, y_val)
        optuna_xgb_model = fit_xgboost(X_train, y_train, best_xgb_params)
        optuna_xgb_threshold = optimize_threshold_for_recall(
            y_val,
            optuna_xgb_model.predict_proba(X_val)[:, 1],
            RECALL_TARGET,
        )
        candidates["Optuna Recall XGB"] = evaluate_candidate(
            name="Optuna Recall XGB",
            technique="optuna_recall",
            model=optuna_xgb_model,
            X_val=X_val,
            y_val=y_val,
            threshold=optuna_xgb_threshold,
            params=best_xgb_params,
        )

        best_lgb_params = optimize_lgb_for_recall(X_train, y_train, X_val, y_val)
        lgb_model = fit_lightgbm(X_train, y_train, best_lgb_params)

        # Technique 4: SMOTE applied only on train 2008.
        X_train_smote, y_train_smote = smote_resample_train(X_train, y_train)
        smote_params = dict(best_xgb_params)
        smote_params["scale_pos_weight"] = 1.0
        smote_model = fit_xgboost(X_train_smote, y_train_smote, smote_params)
        smote_threshold = optimize_threshold_for_recall(
            y_val,
            smote_model.predict_proba(X_val)[:, 1],
            RECALL_TARGET,
        )
        candidates["SMOTE"] = evaluate_candidate(
            name="SMOTE",
            technique="smote",
            model=smote_model,
            X_val=X_val,
            y_val=y_val,
            threshold=smote_threshold,
            params={"smote_k_neighbors": 5, **smote_params},
        )

        # Technique 5: soft voting ensemble with low threshold.
        lr_model = build_lr_pipeline()
        ensemble = VotingClassifier(
            estimators=[
                ("lr", lr_model),
                ("xgb", xgb.XGBClassifier(**best_xgb_params)),
                ("lgbm", lgb.LGBMClassifier(**best_lgb_params)),
            ],
            voting="soft",
            n_jobs=-1,
        )
        ensemble.fit(X_train, y_train)
        candidates["Ensemble seuil=0.25"] = evaluate_candidate(
            name="Ensemble seuil=0.25",
            technique="ensemble",
            model=ensemble,
            X_val=X_val,
            y_val=y_val,
            threshold=ENSEMBLE_THRESHOLD,
            params={"threshold_fixed": ENSEMBLE_THRESHOLD},
        )

        best = select_best_recall_candidate(candidates)

        for candidate in candidates.values():
            add_test_metrics(candidate, X_test, y_test)

        table = format_comparison_table(candidates)
        log.info("\n%s", table)

        test_report = classification_report(
            y_test,
            (best.model.predict_proba(X_test)[:, 1] >= best.threshold).astype(int),
            output_dict=True,
            zero_division=0,
        )
        best.params["test_classification_report"] = test_report

        diagnostics = build_model_diagnostics(
            best=best,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
        )

        best_model, best_name, best_auc = save_best_outputs(
            best=best,
            candidates=candidates,
            feature_cols=feature_cols,
            diagnostics=diagnostics,
        )

        log.info("=" * 78)
        log.info("MEILLEUR MODELE RECALL : %s", best_name)
        log.info(
            "Validation | recall=%.4f precision=%.4f f1=%.4f auc=%.4f seuil=%.4f FN=%d",
            best.val_metrics["recall"],
            best.val_metrics["precision"],
            best.val_metrics["f1"],
            best.val_metrics["auc_roc"],
            best.threshold,
            best.val_confusion_matrix["fn"],
        )
        if best.test_metrics and best.test_confusion_matrix:
            log.info(
                "Test       | recall=%.4f precision=%.4f f1=%.4f auc=%.4f seuil=%.4f FN=%d",
                best.test_metrics["recall"],
                best.test_metrics["precision"],
                best.test_metrics["f1"],
                best.test_metrics["auc_roc"],
                best.threshold,
                best.test_confusion_matrix["fn"],
            )
        log.info("Modele sauvegarde : %s", BEST_RECALL_MODEL_PATH)
        log.info("Seuil sauvegarde  : %s", BEST_THRESHOLD_PATH)
        log.info("=" * 78)

        return best_model, best_name, best_auc

    finally:
        if MLFLOW_AVAILABLE and parent_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
