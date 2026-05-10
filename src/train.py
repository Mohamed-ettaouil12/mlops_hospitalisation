# ═══════════════════════════════════════════════════════════
# src/train.py
# Pipeline MLOps — Modélisation + MLflow Experiment Tracking
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import shap
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuration ──────────────────────────────────────────
FEATURES_DIR = 'data/features/'
MODELS_DIR   = 'models/'
FIGURES_DIR  = 'outputs/figures/'
MLFLOW_URI   = 'mlruns'
EXPERIMENT   = 'hospitalisation_prediction'
RANDOM_STATE = 42
N_OPTUNA     = 50   # nombre d'essais Optuna

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)


# ═══════════════════════════════════════════════════════════
# FONCTION 1 : Chargement des données
# ═══════════════════════════════════════════════════════════
def load_data():
    log.info("Chargement des données...")

    X_train = pd.read_parquet(os.path.join(FEATURES_DIR, 'X_train.parquet'))
    X_val   = pd.read_parquet(os.path.join(FEATURES_DIR, 'X_val.parquet'))
    X_test  = pd.read_parquet(os.path.join(FEATURES_DIR, 'X_test.parquet'))
    y_train = pd.read_parquet(os.path.join(FEATURES_DIR, 'y_train.parquet')).squeeze()
    y_val   = pd.read_parquet(os.path.join(FEATURES_DIR, 'y_val.parquet')).squeeze()
    y_test  = pd.read_parquet(os.path.join(FEATURES_DIR, 'y_test.parquet')).squeeze()

    log.info(f"  Train (2008) : {X_train.shape} | Taux : {y_train.mean()*100:.2f}%")
    log.info(f"  Val   (2009) : {X_val.shape}   | Taux : {y_val.mean()*100:.2f}%")
    log.info(f"  Test  (2010) : {X_test.shape}  | Taux : {y_test.mean()*100:.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════
# FONCTION 2 : Métriques complètes
def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'auc_roc'  : roc_auc_score(y_true, y_pred_proba),
        'f1'       : f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall'   : recall_score(y_true, y_pred, zero_division=0),
    }


# ═══════════════════════════════════════════════════════════
# FONCTION 3 : Graphique SHAP
# ═══════════════════════════════════════════════════════════
def plot_shap(model, X_train, model_name):
    try:
        log.info(f"  Calcul SHAP pour {model_name}...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train.iloc[:500])

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train.iloc[:500],
                          show=False, max_display=15)
        plt.title(f'SHAP — {model_name}')
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f'shap_{model_name.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  → SHAP sauvegardé : {path}")
        return path
    except Exception as e:
        log.warning(f"  SHAP non disponible : {e}")
        return None


# ═══════════════════════════════════════════════════════════
# MODÈLE 1 : Régression Logistique (Baseline)
# ═══════════════════════════════════════════════════════════
def train_logistic(X_train, X_test, y_train, y_test):
    log.info("\n── Régression Logistique (Baseline) ──")

    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)

        # Logger dans MLflow
        mlflow.log_params({'model': 'LogisticRegression', 'class_weight': 'balanced'})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        log.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        log.info(f"  F1-Score  : {metrics['f1']:.4f}")
        log.info(f"  Recall    : {metrics['recall']:.4f}")
        log.info(f"  Precision : {metrics['precision']:.4f}")

        # Sauvegarder
        joblib.dump(model, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))

    return model, metrics


# ═══════════════════════════════════════════════════════════
# MODÈLE 2 : XGBoost + Optuna
# ═══════════════════════════════════════════════════════════
def train_xgboost(X_train, X_test, y_train, y_test):
    log.info("\n── XGBoost + Optuna ──")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    log.info(f"  scale_pos_weight : {scale_pos_weight:.2f}")

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),
            'max_depth'        : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight' : scale_pos_weight,
            'random_state'     : RANDOM_STATE,
            'eval_metric'      : 'auc',
            'use_label_encoder': False
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_proba)

    log.info(f"  Optimisation Optuna ({N_OPTUNA} essais)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA)

    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state']     = RANDOM_STATE
    best_params['eval_metric']      = 'auc'
    best_params['use_label_encoder'] = False

    log.info(f"  Meilleurs paramètres : {best_params}")

    with mlflow.start_run(run_name="XGBoost_Optuna"):
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric('optuna_best_auc', study.best_value)
        mlflow.xgboost.log_model(model, "model")

        # SHAP
        shap_path = plot_shap(model, X_train, 'XGBoost')
        if shap_path:
            mlflow.log_artifact(shap_path)

        log.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        log.info(f"  F1-Score  : {metrics['f1']:.4f}")
        log.info(f"  Recall    : {metrics['recall']:.4f}")
        log.info(f"  Precision : {metrics['precision']:.4f}")

        joblib.dump(model, os.path.join(MODELS_DIR, 'xgboost_best.pkl'))

    return model, metrics, best_params


# ═══════════════════════════════════════════════════════════
# MODÈLE 3 : LightGBM + Optuna
# ═══════════════════════════════════════════════════════════
def train_lightgbm(X_train, X_test, y_train, y_test):
    log.info("\n── LightGBM + Optuna ──")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 100, 500),
            'max_depth'       : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves'      : trial.suggest_int('num_leaves', 20, 100),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state'    : RANDOM_STATE,
            'verbose'         : -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        y_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_proba)

    log.info(f"  Optimisation Optuna ({N_OPTUNA} essais)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA)

    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state']     = RANDOM_STATE
    best_params['verbose']          = -1

    with mlflow.start_run(run_name="LightGBM_Optuna"):
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train,
                  callbacks=[lgb.log_evaluation(-1)])

        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric('optuna_best_auc', study.best_value)
        mlflow.lightgbm.log_model(model, "model")

        shap_path = plot_shap(model, X_train, 'LightGBM')
        if shap_path:
            mlflow.log_artifact(shap_path)

        log.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        log.info(f"  F1-Score  : {metrics['f1']:.4f}")
        log.info(f"  Recall    : {metrics['recall']:.4f}")
        log.info(f"  Precision : {metrics['precision']:.4f}")

        joblib.dump(model, os.path.join(MODELS_DIR, 'lightgbm_best.pkl'))

    return model, metrics, best_params


# ═══════════════════════════════════════════════════════════
# FONCTION 4 : Sélection du meilleur modèle
# ═══════════════════════════════════════════════════════════
def select_best_model(results):
    log.info("\n── Comparaison des modèles ──")

    best_name  = None
    best_auc   = 0
    best_model = None

    for name, (model, metrics) in results.items():
        auc = metrics['auc_roc']
        log.info(f"  {name:25s} | AUC : {auc:.4f} | F1 : {metrics['f1']:.4f} | Recall : {metrics['recall']:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_name  = name
            best_model = model

    log.info(f"\n🏆 Meilleur modèle : {best_name} (AUC = {best_auc:.4f})")

    # Sauvegarder le meilleur modèle
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    pd.Series({'model': best_name, 'auc': best_auc}).to_json(
        os.path.join(MODELS_DIR, 'best_model_info.json')
    )
    log.info(f"  → Meilleur modèle sauvegardé : {MODELS_DIR}best_model.pkl")

    return best_name, best_model, best_auc


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("="*60)
    log.info("PIPELINE MODÉLISATION — DÉBUT")
    log.info("="*60)

    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    log.info(f"MLflow experiment : {EXPERIMENT}")

    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()


    # Entraîner les modèles
    results = {}

    log.info("\n" + "="*60)
    log.info("ENTRAÎNEMENT DES MODÈLES")
    log.info("="*60)

    # 1. Baseline
    model_lr, metrics_lr = train_logistic(X_train, X_test, y_train, y_test)
    results['LogisticRegression'] = (model_lr, metrics_lr)

    # 2. XGBoost
    model_xgb, metrics_xgb, _ = train_xgboost(X_train, X_test, y_train, y_test)
    results['XGBoost'] = (model_xgb, metrics_xgb)

    # 3. LightGBM
    model_lgb, metrics_lgb, _ = train_lightgbm(X_train, X_test, y_train, y_test)
    results['LightGBM'] = (model_lgb, metrics_lgb)

    # Sélectionner le meilleur
    best_name, best_model, best_auc = select_best_model(results)

    log.info("\n" + "="*60)
    log.info("PIPELINE MODÉLISATION — TERMINÉ ✅")
    log.info(f"🏆 Meilleur modèle : {best_name} | AUC : {best_auc:.4f}")
    log.info("Prochaine étape → mlflow ui")
    log.info("="*60)

    return best_model, best_name, best_auc


if __name__ == '__main__':
    main()