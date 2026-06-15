# ═══════════════════════════════════════════════════════════
# src/train.py
# Pipeline MLOps — Modélisation Production + MLflow Tracking
# Projet : Risque d'Hospitalisation Medicare CMS DE-SynPUF
# ═══════════════════════════════════════════════════════════

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble_models import ProbabilityAveragingEnsemble

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# IMPORTS OPTIONNELS ROBUSTES
# ═══════════════════════════════════════════════════════════

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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

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


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
LOG_DIR = Path("logs")

MLFLOW_URI = "mlruns"
EXPERIMENT_NAME = "hospitalisation_prediction"

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 100  # Augmenté de 50 à 100 pour meilleure optimisation
FAST_MODE = False  # mets True si tu veux test rapide
THRESHOLD_BETA = 1.0  # F1: plus precis que F2, sans sacrifier autant le recall que F0.5
MODEL_SELECTION_METRIC = "precision"
MIN_RECALL_FOR_SELECTION = 0.50
ENSEMBLE_WEIGHT_GRID = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2))
HIGH_PRECISION_TARGETS = (0.80, 0.85, 0.89, 0.90, 0.93)
MIN_TARGET_POSITIVE_PREDICTIONS = 50

TARGET_COL = "HOSPITALIZED_IN_6M"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def safe_target(path: Path) -> pd.Series:
    """
    Charge y_train / y_val / y_test proprement.
    Évite squeeze() instable.
    """
    require_file(path)

    df = pd.read_parquet(path)

    if isinstance(df, pd.DataFrame):
        if TARGET_COL in df.columns:
            y = df[TARGET_COL]
        else:
            y = df.iloc[:, 0]
    else:
        y = df

    y = (
        pd.to_numeric(y, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return y.reset_index(drop=True)


def sanitize_features(X: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Rend X 100% numérique et aligné avec feature_cols.
    """
    X = X.copy()

    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X.reset_index(drop=True)


def load_feature_names() -> Optional[list]:
    """
    Charge la liste officielle des features si disponible.
    Compatible avec preprocessing production :
    - models/feature_names.pkl
    - data/features/feature_names.csv
    """
    pkl_path = MODELS_DIR / "feature_names.pkl"
    csv_path = FEATURES_DIR / "feature_names.csv"

    if pkl_path.exists():
        return joblib.load(pkl_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.iloc[:, 0].astype(str).tolist()

    return None


def safe_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    Évite crash si une classe est absente.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_proba))


def optimize_threshold_from_proba(
    y_true: pd.Series,
    y_proba: np.ndarray,
    beta: float = THRESHOLD_BETA,
) -> float:
    """
    Seuil optimal basé sur F-beta validation.
    beta < 1 favorise la precision, beta > 1 favorise le recall.
    Version safe : évite bug de longueur thresholds.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.5

    f_scores = (
        (1 + beta**2) * precisions * recalls
        / ((beta**2 * precisions) + recalls + 1e-9)
    )

    # thresholds a longueur n, precisions/recalls longueur n+1
    f_scores = f_scores[:-1]

    best_idx = int(np.nanargmax(f_scores))
    best_threshold = float(thresholds[best_idx])

    return best_threshold


def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcule les métriques principales.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": safe_auc(y_true, y_proba),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    return metrics


def confusion_matrix_dict(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def find_threshold_for_target_precision(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_precision: float,
    min_positive_predictions: int = MIN_TARGET_POSITIVE_PREDICTIONS,
) -> Optional[Dict[str, Any]]:
    """
    Cherche un seuil qui atteint une precision cible sur validation.
    Parmi les seuils valides, garde celui qui conserve le plus de recall.
    """
    if len(np.unique(y_true)) < 2:
        return None

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return None

    precision_values = precisions[:-1]
    recall_values = recalls[:-1]
    predicted_positive_counts = np.array([
        int((y_proba >= threshold).sum())
        for threshold in thresholds
    ])

    valid = (
        (precision_values >= target_precision)
        & (predicted_positive_counts >= min_positive_predictions)
    )

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    best_idx = int(valid_indices[np.argmax(recall_values[valid_indices])])

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision_values[best_idx]),
        "recall": float(recall_values[best_idx]),
        "predicted_positive": int(predicted_positive_counts[best_idx]),
    }


def json_safe(obj: Any) -> Any:
    """
    Convertit objets numpy/pandas vers JSON natif.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, pd.Series):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_safe)


def log_metrics_readable(model_name: str, metrics: Dict[str, float]) -> None:
    log.info(
        f"  {model_name:22s} | "
        f"AUC={metrics['auc_roc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Threshold={metrics['threshold']:.3f}"
    )


def finite_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    MLflow refuse parfois NaN/inf.
    """
    clean = {}

    for k, v in metrics.items():
        try:
            v = float(v)
            if np.isfinite(v):
                clean[k] = v
        except Exception:
            pass

    return clean


# ═══════════════════════════════════════════════════════════
# MLFLOW SAFE
# ═══════════════════════════════════════════════════════════

def setup_mlflow() -> None:
    if not MLFLOW_AVAILABLE:
        log.warning("MLflow non disponible : tracking désactivé.")
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        log.info(f"MLflow experiment : {EXPERIMENT_NAME}")
    except Exception as e:
        log.warning(f"MLflow setup échoué : {e}")


def mlflow_log_model_safe(model: Any, model_name: str, flavor: str) -> None:
    if not MLFLOW_AVAILABLE:
        return

    try:
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, "model")
        elif flavor == "xgboost" and XGBOOST_AVAILABLE:
            mlflow.xgboost.log_model(model, "model")
        elif flavor == "lightgbm" and LIGHTGBM_AVAILABLE:
            mlflow.lightgbm.log_model(model, "model")
    except Exception as e:
        log.warning(f"MLflow log_model échoué pour {model_name}: {e}")


def mlflow_log_metrics_safe(metrics: Dict[str, float], prefix: str = "") -> None:
    if not MLFLOW_AVAILABLE:
        return

    try:
        clean_metrics = finite_metrics(metrics)

        if prefix:
            clean_metrics = {f"{prefix}_{k}": v for k, v in clean_metrics.items()}

        mlflow.log_metrics(clean_metrics)
    except Exception as e:
        log.warning(f"MLflow log_metrics échoué : {e}")


def mlflow_log_params_safe(params: Dict[str, Any]) -> None:
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_params(params)
    except Exception as e:
        log.warning(f"MLflow log_params échoué : {e}")


def mlflow_log_artifact_safe(path: Optional[Path]) -> None:
    if not MLFLOW_AVAILABLE or path is None:
        return

    try:
        if path.exists():
            mlflow.log_artifact(str(path))
    except Exception as e:
        log.warning(f"MLflow log_artifact échoué : {e}")


# ═══════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list]:
    log.info("Chargement des données features...")

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

    feature_cols = load_feature_names()

    X_train = pd.read_parquet(paths["X_train"])
    X_val = pd.read_parquet(paths["X_val"])
    X_test = pd.read_parquet(paths["X_test"])

    if feature_cols is None:
        feature_cols = X_train.columns.astype(str).tolist()

    X_train = sanitize_features(X_train, feature_cols)
    X_val = sanitize_features(X_val, feature_cols)
    X_test = sanitize_features(X_test, feature_cols)

    y_train = safe_target(paths["y_train"])
    y_val = safe_target(paths["y_val"])
    y_test = safe_target(paths["y_test"])

    if len(X_train) != len(y_train):
        raise ValueError(f"Mismatch train : X={len(X_train)} y={len(y_train)}")

    if len(X_val) != len(y_val):
        raise ValueError(f"Mismatch val : X={len(X_val)} y={len(y_val)}")

    if len(X_test) != len(y_test):
        raise ValueError(f"Mismatch test : X={len(X_test)} y={len(y_test)}")

    log.info(f"  Train : {X_train.shape} | Taux : {y_train.mean() * 100:.2f}%")
    log.info(f"  Val   : {X_val.shape} | Taux : {y_val.mean() * 100:.2f}%")
    log.info(f"  Test  : {X_test.shape} | Taux : {y_test.mean() * 100:.2f}%")
    log.info(f"  Features : {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ═══════════════════════════════════════════════════════════
# SHAP SAFE
# ═══════════════════════════════════════════════════════════

def plot_shap_safe(model: Any, X_train: pd.DataFrame, model_name: str) -> Optional[Path]:
    if not SHAP_AVAILABLE:
        log.warning("SHAP non disponible : graphique ignoré.")
        return None

    try:
        log.info(f"  Calcul SHAP pour {model_name}...")

        X_sample = X_train.iloc[: min(500, len(X_train))].copy()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            max_display=20,
        )

        path = FIGURES_DIR / f"shap_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        log.info(f"  → SHAP sauvegardé : {path}")
        return path

    except Exception as e:
        log.warning(f"  SHAP non disponible pour {model_name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# TRAIN LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════

def train_logistic(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> Dict[str, Any]:

    model_name = "LogisticRegression"
    log.info("\n── Régression Logistique Baseline ──")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run(run_name=model_name, nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        params = {
            "model": model_name,
            "class_weight": "balanced",
            "max_iter": 3000,
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
        }

        mlflow_log_params_safe(params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")
        mlflow_log_model_safe(model, model_name, "sklearn")

        model_path = MODELS_DIR / "logistic_regression.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": params,
        "path": str(model_path),
        "flavor": "sklearn",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN XGBOOST
# ═══════════════════════════════════════════════════════════

def train_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> Optional[Dict[str, Any]]:

    model_name = "XGBoost"

    if not XGBOOST_AVAILABLE:
        log.warning("XGBoost non installé : modèle ignoré.")
        return None

    if not OPTUNA_AVAILABLE:
        log.warning("Optuna non installé : XGBoost utilisera paramètres par défaut.")

    log.info("\n── XGBoost + Optuna ──")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    log.info(f"  scale_pos_weight : {scale_pos_weight:.2f}")

    if FAST_MODE:
        n_trials = 5
    else:
        n_trials = N_OPTUNA_TRIALS

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        return safe_auc(y_val, val_proba)

    if OPTUNA_AVAILABLE:
        log.info(f"  Optimisation Optuna ({n_trials} essais) sur VALIDATION seulement...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_auc = float(study.best_value)
    else:
        best_params = {
            "n_estimators": 350,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        best_auc = float("nan")

    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    })

    log.info(f"  Meilleurs paramètres XGBoost : {best_params}")

    model = xgb.XGBClassifier(**best_params)

    with mlflow.start_run(run_name="XGBoost_Optuna", nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train, verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe(best_params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")

        if np.isfinite(best_auc):
            mlflow_log_metrics_safe({"optuna_best_auc": best_auc})

        mlflow_log_model_safe(model, model_name, "xgboost")

        shap_path = plot_shap_safe(model, X_train, model_name)
        mlflow_log_artifact_safe(shap_path)

        model_path = MODELS_DIR / "xgboost_best.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": best_params,
        "path": str(model_path),
        "flavor": "xgboost",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN LIGHTGBM
# ═══════════════════════════════════════════════════════════

def train_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> Optional[Dict[str, Any]]:

    model_name = "LightGBM"

    if not LIGHTGBM_AVAILABLE:
        log.warning("LightGBM non installé : modèle ignoré.")
        return None

    if not OPTUNA_AVAILABLE:
        log.warning("Optuna non installé : LightGBM utilisera paramètres par défaut.")

    log.info("\n── LightGBM + Optuna ──")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    if FAST_MODE:
        n_trials = 5
    else:
        n_trials = N_OPTUNA_TRIALS

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
            "force_row_wise": True,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        return safe_auc(y_val, val_proba)

    if OPTUNA_AVAILABLE:
        log.info(f"  Optimisation Optuna ({n_trials} essais) sur VALIDATION seulement...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_auc = float(study.best_value)
    else:
        best_params = {
            "n_estimators": 350,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 30,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        best_auc = float("nan")

    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_row_wise": True,
    })

    model = lgb.LGBMClassifier(**best_params)

    with mlflow.start_run(run_name="LightGBM_Optuna", nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe(best_params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")

        if np.isfinite(best_auc):
            mlflow_log_metrics_safe({"optuna_best_auc": best_auc})

        mlflow_log_model_safe(model, model_name, "lightgbm")

        shap_path = plot_shap_safe(model, X_train, model_name)
        mlflow_log_artifact_safe(shap_path)

        model_path = MODELS_DIR / "lightgbm_best.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": best_params,
        "path": str(model_path),
        "flavor": "lightgbm",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN CATBOOST (NOUVEAU)
# ═══════════════════════════════════════════════════════════

def train_catboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> Optional[Dict[str, Any]]:

    model_name = "CatBoost"

    if not CATBOOST_AVAILABLE:
        log.warning("CatBoost non installé : modèle ignoré.")
        return None

    if not OPTUNA_AVAILABLE:
        log.warning("Optuna non installé : CatBoost utilisera paramètres par défaut.")

    log.info("\n── CatBoost + Optuna ──")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    if FAST_MODE:
        n_trials = 5
    else:
        n_trials = N_OPTUNA_TRIALS

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 150, 600),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "verbose": 0,
            "thread_count": -1,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        return safe_auc(y_val, val_proba)

    if OPTUNA_AVAILABLE:
        log.info(f"  Optimisation Optuna ({n_trials} essais) sur VALIDATION seulement...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_auc = float(study.best_value)
    else:
        best_params = {
            "iterations": 350,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 1.0,
            "subsample": 0.8,
        }
        best_auc = float("nan")

    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1,
    })

    log.info(f"  Meilleurs paramètres CatBoost : {best_params}")

    model = CatBoostClassifier(**best_params)

    with mlflow.start_run(run_name="CatBoost_Optuna", nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train, verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe(best_params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")

        if np.isfinite(best_auc):
            mlflow_log_metrics_safe({"optuna_best_auc": best_auc})

        mlflow_log_model_safe(model, model_name, "sklearn")

        shap_path = plot_shap_safe(model, X_train, model_name)
        mlflow_log_artifact_safe(shap_path)

        model_path = MODELS_DIR / "catboost_best.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": best_params,
        "path": str(model_path),
        "flavor": "sklearn",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN RANDOM FOREST BAGGING
# ═══════════════════════════════════════════════════════════

def train_random_forest_bagging(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> Dict[str, Any]:

    model_name = "RandomForest_Bagging"
    log.info("\n── Random Forest Bagging ──")

    n_estimators = 120 if FAST_MODE else 400
    params = {
        "n_estimators": n_estimators,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    model = RandomForestClassifier(**params)

    with mlflow.start_run(run_name=model_name, nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe({"model": model_name, **params})
        mlflow_log_metrics_safe(val_metrics, prefix="val")
        mlflow_log_model_safe(model, model_name, "sklearn")

        model_path = MODELS_DIR / "random_forest_bagging.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": params,
        "path": str(model_path),
        "flavor": "sklearn",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN SOFT VOTING ENSEMBLE
# ═══════════════════════════════════════════════════════════

def train_soft_voting_ensemble(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    xgb_result: Optional[Dict[str, Any]],
    lgb_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    model_name = "SoftVoting_XGB_LGBM"

    if xgb_result is None or lgb_result is None:
        log.warning("SoftVoting ignore : XGBoost et LightGBM sont requis.")
        return None

    log.info("\n── Soft Voting Ensemble XGBoost + LightGBM ──")

    xgb_val_proba = xgb_result["model"].predict_proba(X_val)[:, 1]
    lgb_val_proba = lgb_result["model"].predict_proba(X_val)[:, 1]

    best_weight = None
    best_score = -np.inf
    best_auc = -np.inf

    for xgb_weight in ENSEMBLE_WEIGHT_GRID:
        weights = [float(xgb_weight), float(1.0 - xgb_weight)]
        val_proba = (weights[0] * xgb_val_proba) + (weights[1] * lgb_val_proba)
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        metrics = compute_metrics(y_val, val_proba, threshold)

        score = metrics.get(MODEL_SELECTION_METRIC, float("nan"))
        auc = metrics.get("auc_roc", float("nan"))

        if not np.isfinite(score):
            continue

        if (
            score > best_score
            or (np.isclose(score, best_score) and np.isfinite(auc) and auc > best_auc)
        ):
            best_weight = weights
            best_score = score
            best_auc = auc if np.isfinite(auc) else -np.inf

    if best_weight is None:
        log.warning("SoftVoting ignore : aucun poids valide trouve.")
        return None

    log.info(
        "  Meilleurs poids SoftVoting : "
        f"XGBoost={best_weight[0]:.2f}, LightGBM={best_weight[1]:.2f}"
    )

    model = ProbabilityAveragingEnsemble(
        estimators=[
            ("xgboost", xgb_result["model"]),
            ("lightgbm", lgb_result["model"]),
        ],
        weights=best_weight,
    )

    params = {
        "model": model_name,
        "xgboost_weight": best_weight[0],
        "lightgbm_weight": best_weight[1],
        "threshold_beta": THRESHOLD_BETA,
        "weight_selection_metric": MODEL_SELECTION_METRIC,
    }

    with mlflow.start_run(run_name=model_name, nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe(params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")
        mlflow_log_model_safe(model, model_name, "sklearn")

        model_path = MODELS_DIR / "soft_voting_xgb_lgbm.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": params,
        "path": str(model_path),
        "flavor": "sklearn",
    }


# ═══════════════════════════════════════════════════════════
# TRAIN SOFT VOTING ENSEMBLE V2 (XGB + LGB + CatBoost)
# ═══════════════════════════════════════════════════════════

def train_soft_voting_ensemble_v2(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    xgb_result: Optional[Dict[str, Any]],
    lgb_result: Optional[Dict[str, Any]],
    cat_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Ensemble à 3 modèles (XGBoost + LightGBM + CatBoost).
    Optimise les poids pour maximiser la précision.
    """

    model_name = "SoftVoting_XGB_LGB_CAT"

    if xgb_result is None or lgb_result is None or cat_result is None:
        log.warning("SoftVoting V2 ignore : XGBoost, LightGBM et CatBoost sont requis.")
        return None

    log.info("\n── Soft Voting Ensemble V2 (XGB + LGB + CatBoost) ──")

    xgb_val_proba = xgb_result["model"].predict_proba(X_val)[:, 1]
    lgb_val_proba = lgb_result["model"].predict_proba(X_val)[:, 1]
    cat_val_proba = cat_result["model"].predict_proba(X_val)[:, 1]

    best_weights = None
    best_score = -np.inf
    best_auc = -np.inf

    # Optimiser en 3D: w_xgb, w_lgb, w_cat (somme = 1.0)
    for xgb_weight in ENSEMBLE_WEIGHT_GRID:
        for lgb_weight in ENSEMBLE_WEIGHT_GRID:
            cat_weight = 1.0 - xgb_weight - lgb_weight
            
            if cat_weight < 0:
                continue

            weights = [float(xgb_weight), float(lgb_weight), float(cat_weight)]
            val_proba = (
                weights[0] * xgb_val_proba + 
                weights[1] * lgb_val_proba + 
                weights[2] * cat_val_proba
            )
            
            threshold = optimize_threshold_from_proba(y_val, val_proba)
            metrics = compute_metrics(y_val, val_proba, threshold)

            score = metrics.get(MODEL_SELECTION_METRIC, float("nan"))
            auc = metrics.get("auc_roc", float("nan"))

            if not np.isfinite(score):
                continue

            if (
                score > best_score
                or (np.isclose(score, best_score) and np.isfinite(auc) and auc > best_auc)
            ):
                best_weights = weights
                best_score = score
                best_auc = auc if np.isfinite(auc) else -np.inf

    if best_weights is None:
        log.warning("SoftVoting V2 ignore : aucun poids valide trouve.")
        return None

    log.info(
        "  Meilleurs poids SoftVoting V2 : "
        f"XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}, CatBoost={best_weights[2]:.2f}"
    )

    model = ProbabilityAveragingEnsemble(
        estimators=[
            ("xgboost", xgb_result["model"]),
            ("lightgbm", lgb_result["model"]),
            ("catboost", cat_result["model"]),
        ],
        weights=best_weights,
    )

    params = {
        "model": model_name,
        "xgboost_weight": best_weights[0],
        "lightgbm_weight": best_weights[1],
        "catboost_weight": best_weights[2],
        "threshold_beta": THRESHOLD_BETA,
        "weight_selection_metric": MODEL_SELECTION_METRIC,
    }

    with mlflow.start_run(run_name=model_name, nested=True) if MLFLOW_AVAILABLE else nullcontext():
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimize_threshold_from_proba(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)

        mlflow_log_params_safe(params)
        mlflow_log_metrics_safe(val_metrics, prefix="val")
        mlflow_log_model_safe(model, model_name, "sklearn")

        model_path = MODELS_DIR / "soft_voting_xgb_lgb_cat.pkl"
        joblib.dump(model, model_path)

        log_metrics_readable(model_name, val_metrics)
        log.info(f"  → Modèle sauvegardé : {model_path}")

    return {
        "name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "params": params,
        "path": str(model_path),
        "flavor": "sklearn",
    }


# ═══════════════════════════════════════════════════════════
# CONTEXT MANAGER NULL
# ═══════════════════════════════════════════════════════════

class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


# ═══════════════════════════════════════════════════════════
# ÉVALUATION TEST FINALE
# ═══════════════════════════════════════════════════════════

def evaluate_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
) -> Dict[str, Any]:

    log.info(f"\n── Évaluation finale TEST : {model_name} ──")

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    cm = confusion_matrix_dict(y_test, test_proba, threshold)

    y_pred = (test_proba >= threshold).astype(int)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    log_metrics_readable(f"{model_name} TEST", test_metrics)
    log.info(f"  Confusion Matrix : {cm}")

    return {
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


# ═══════════════════════════════════════════════════════════
# RAPPORT SEUILS HAUTE PRÉCISION
# ═══════════════════════════════════════════════════════════

def build_high_precision_report(
    results: Dict[str, Dict[str, Any]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:

    log.info("\n── Recherche seuils haute précision ──")

    report = {
        "targets": list(HIGH_PRECISION_TARGETS),
        "min_positive_predictions_validation": MIN_TARGET_POSITIVE_PREDICTIONS,
        "note": (
            "Les seuils sont choisis sur validation uniquement. "
            "Le test est affiché pour controle, pas pour sélectionner le modèle."
        ),
        "models": {},
    }

    for model_name, result in results.items():
        model = result["model"]
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        model_report = {}

        for target in HIGH_PRECISION_TARGETS:
            threshold_info = find_threshold_for_target_precision(
                y_true=y_val,
                y_proba=val_proba,
                target_precision=target,
            )

            target_key = f"{target:.2f}"

            if threshold_info is None:
                model_report[target_key] = {
                    "achievable": False,
                    "reason": (
                        "Aucun seuil validation n'atteint cette précision "
                        f"avec au moins {MIN_TARGET_POSITIVE_PREDICTIONS} prédictions positives."
                    ),
                }
                continue

            threshold = float(threshold_info["threshold"])
            val_metrics = compute_metrics(y_val, val_proba, threshold)
            test_metrics = compute_metrics(y_test, test_proba, threshold)
            val_cm = confusion_matrix_dict(y_val, val_proba, threshold)
            test_cm = confusion_matrix_dict(y_test, test_proba, threshold)

            model_report[target_key] = {
                "achievable": True,
                "threshold": threshold,
                "validation_metrics": val_metrics,
                "validation_confusion_matrix": val_cm,
                "validation_predicted_positive": int(val_cm["fp"] + val_cm["tp"]),
                "test_metrics": test_metrics,
                "test_confusion_matrix": test_cm,
                "test_predicted_positive": int(test_cm["fp"] + test_cm["tp"]),
            }

            log.info(
                f"  {model_name:22s} | target={target:.2f} | "
                f"threshold={threshold:.3f} | "
                f"VAL precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}"
            )

        report["models"][model_name] = model_report

    return report


# ═══════════════════════════════════════════════════════════
# SÉLECTION MEILLEUR MODÈLE
# ═══════════════════════════════════════════════════════════

def select_best_model(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    log.info("\n── Comparaison des modèles sur VALIDATION ──")

    if not results:
        raise ValueError("Aucun modèle entraîné. Vérifie les dépendances.")

    best_name = None
    best_score = -np.inf
    best_auc = -np.inf
    fallback_name = None
    fallback_score = -np.inf
    fallback_auc = -np.inf

    for name, result in results.items():
        metrics = result["val_metrics"]
        score = metrics.get(MODEL_SELECTION_METRIC, float("nan"))
        auc = metrics.get("auc_roc", float("nan"))
        recall = metrics.get("recall", float("nan"))

        log_metrics_readable(name, metrics)

        if np.isfinite(score) and (
            score > fallback_score
            or (np.isclose(score, fallback_score) and np.isfinite(auc) and auc > fallback_auc)
        ):
            fallback_score = score
            fallback_auc = auc if np.isfinite(auc) else -np.inf
            fallback_name = name

        if not np.isfinite(score) or not np.isfinite(recall) or recall < MIN_RECALL_FOR_SELECTION:
            continue

        if (
            score > best_score
            or (np.isclose(score, best_score) and np.isfinite(auc) and auc > best_auc)
        ):
            best_score = score
            best_auc = auc if np.isfinite(auc) else -np.inf
            best_name = name

    if best_name is None:
        if fallback_name is None:
            raise ValueError("Impossible de sélectionner un meilleur modèle : métriques invalides partout.")

        log.warning(
            "Aucun modèle ne respecte recall >= %.2f ; fallback sur %s validation.",
            MIN_RECALL_FOR_SELECTION,
            MODEL_SELECTION_METRIC,
        )
        best_name = fallback_name
        best_score = fallback_score
        best_auc = fallback_auc

    best_result = results[best_name]

    log.info(
        f"\n🏆 Meilleur modèle : {best_name} | "
        f"{MODEL_SELECTION_METRIC} validation = {best_score:.4f} | "
        f"AUC validation = {best_auc:.4f}"
    )

    return best_result


def save_best_model(
    best_result: Dict[str, Any],
    test_report: Dict[str, Any],
    feature_cols: list,
    high_precision_report: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any, float]:

    model = best_result["model"]
    model_name = best_result["name"]
    threshold = float(best_result["threshold"])
    val_metrics = best_result["val_metrics"]
    test_metrics = test_report["test_metrics"]

    best_model_path = MODELS_DIR / "best_model.pkl"
    best_info_path = MODELS_DIR / "best_model_info.json"
    threshold_path = MODELS_DIR / "threshold.json"
    metrics_path = REPORTS_DIR / "training_metrics.json"

    joblib.dump(model, best_model_path)

    save_json(
        {
            "threshold": threshold,
            "model": model_name,
            "threshold_beta": THRESHOLD_BETA,
            "selection_metric": MODEL_SELECTION_METRIC,
        },
        threshold_path,
    )

    best_info = {
        "best_model": model_name,
        "best_model_path": str(best_model_path),
        "threshold": threshold,
        "threshold_beta": THRESHOLD_BETA,
        "selection_metric": MODEL_SELECTION_METRIC,
        "min_recall_for_selection": MIN_RECALL_FOR_SELECTION,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix_test": test_report["confusion_matrix"],
        "model_params": best_result.get("params", {}),
        "high_precision_report": high_precision_report or {},
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }

    save_json(best_info, best_info_path)

    save_json(
        {
            "best_model": model_name,
            "all_info": best_info,
        },
        metrics_path,
    )

    log.info(f"  → best_model sauvegardé : {best_model_path}")
    log.info(f"  → threshold sauvegardé : {threshold_path}")
    log.info(f"  → infos modèle sauvegardées : {best_info_path}")
    log.info(f"  → métriques sauvegardées : {metrics_path}")

    return model_name, model, float(val_metrics["auc_roc"])


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main() -> Tuple[Any, str, float]:
    log.info("=" * 70)
    log.info("PIPELINE MODÉLISATION — DÉBUT")
    log.info("=" * 70)

    setup_mlflow()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()

    results = {}

    log.info("\n" + "=" * 70)
    log.info("ENTRAÎNEMENT DES MODÈLES")
    log.info("=" * 70)

    # 1. Logistic Regression
    logistic_result = train_logistic(X_train, X_val, y_train, y_val)
    results[logistic_result["name"]] = logistic_result

    # 2. XGBoost
    xgb_result = train_xgboost(X_train, X_val, y_train, y_val)
    if xgb_result is not None:
        results[xgb_result["name"]] = xgb_result

    # 3. LightGBM
    lgb_result = train_lightgbm(X_train, X_val, y_train, y_val)
    if lgb_result is not None:
        results[lgb_result["name"]] = lgb_result

    # 3.5 CatBoost (NOUVEAU)
    cat_result = train_catboost(X_train, X_val, y_train, y_val)
    if cat_result is not None:
        results[cat_result["name"]] = cat_result

    # 4. Random Forest Bagging
    rf_result = train_random_forest_bagging(X_train, X_val, y_train, y_val)
    results[rf_result["name"]] = rf_result

    # 5. Soft Voting XGBoost + LightGBM
    ensemble_result = train_soft_voting_ensemble(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        xgb_result=xgb_result,
        lgb_result=lgb_result,
    )
    if ensemble_result is not None:
        results[ensemble_result["name"]] = ensemble_result

    # 5.5 Soft Voting V2 (XGB + LGB + CatBoost) - NOUVEAU
    ensemble_v2_result = train_soft_voting_ensemble_v2(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        xgb_result=xgb_result,
        lgb_result=lgb_result,
        cat_result=cat_result,
    )
    if ensemble_v2_result is not None:
        results[ensemble_v2_result["name"]] = ensemble_v2_result

    # 6. Rapport des seuils haute précision
    high_precision_report = build_high_precision_report(
        results=results,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # 7. Sélection sur validation uniquement
    best_result = select_best_model(results)

    # 8. Évaluation finale sur test uniquement après sélection
    test_report = evaluate_on_test(
        model=best_result["model"],
        X_test=X_test,
        y_test=y_test,
        threshold=best_result["threshold"],
        model_name=best_result["name"],
    )

    # 9. Sauvegarde production
    best_name, best_model, best_auc = save_best_model(
        best_result=best_result,
        test_report=test_report,
        feature_cols=feature_cols,
        high_precision_report=high_precision_report,
    )

    log.info("=" * 70)
    log.info("PIPELINE MODÉLISATION — TERMINÉ ✅")
    log.info(f"🏆 Meilleur modèle : {best_name} | AUC validation : {best_auc:.4f}")
    log.info("Prochaine étape : mlflow ui")
    log.info("=" * 70)

    return best_model, best_name, best_auc


if __name__ == "__main__":
    main()
