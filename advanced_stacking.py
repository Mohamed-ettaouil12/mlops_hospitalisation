#!/usr/bin/env python3
"""
STRATÉGIE AVANCÉE: Stacking avec Meta-Learner
Si les améliorations précédentes ne suffisent pas pour 80% de précision.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import load_data, compute_metrics, optimize_threshold_from_proba

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def create_stacking_ensemble(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    models_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Crée un ensemble de stacking en 2 niveaux:
    Niveau 1: Prédictions des modèles individuels
    Niveau 2: Meta-learner (LogisticRegression) entraîné sur niveau 1
    """
    
    log.info("\n" + "="*70)
    log.info("STACKING EN DEUX NIVEAUX (Advanced Strategy)")
    log.info("="*70)
    
    # Niveau 1: Récupérer les probas de tous les modèles
    log.info("\n📊 Génération des features de Niveau 1...")
    
    level1_models_names = [
        "LogisticRegression",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "RandomForest_Bagging",
    ]
    
    level1_train_features = []
    level1_val_features = []
    level1_test_features = []
    
    for model_name in level1_models_names:
        if model_name not in models_dict:
            log.warning(f"  ⚠️ {model_name} non trouvé, ignoré")
            continue
        
        model = models_dict[model_name]["model"]
        
        try:
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            
            level1_train_features.append(train_proba)
            level1_val_features.append(val_proba)
            level1_test_features.append(test_proba)
            
            log.info(f"  ✓ {model_name}: shape {train_proba.shape}")
        except Exception as e:
            log.warning(f"  ⚠️ Erreur avec {model_name}: {e}")
    
    # Convertir en DataFrames
    level1_train = pd.DataFrame(
        np.column_stack(level1_train_features),
        columns=[f"model_{i}" for i in range(len(level1_train_features))]
    )
    level1_val = pd.DataFrame(
        np.column_stack(level1_val_features),
        columns=[f"model_{i}" for i in range(len(level1_val_features))]
    )
    level1_test = pd.DataFrame(
        np.column_stack(level1_test_features),
        columns=[f"model_{i}" for i in range(len(level1_test_features))]
    )
    
    log.info(f"\n  Shapes Niveau 1:")
    log.info(f"    Train: {level1_train.shape}")
    log.info(f"    Val:   {level1_val.shape}")
    log.info(f"    Test:  {level1_test.shape}")
    
    # Niveau 2: Meta-learner
    log.info("\n🔗 Entraînement du Meta-Learner...")
    
    # Standardiser les features
    scaler = StandardScaler()
    level1_train_scaled = scaler.fit_transform(level1_train)
    level1_val_scaled = scaler.transform(level1_val)
    level1_test_scaled = scaler.transform(level1_test)
    
    # Meta-learner: Régression Logistique avec class_weight
    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    
    meta_model.fit(level1_train_scaled, y_train)
    
    # Prédictions du meta-learner
    meta_train_proba = meta_model.predict_proba(level1_train_scaled)[:, 1]
    meta_val_proba = meta_model.predict_proba(level1_val_scaled)[:, 1]
    meta_test_proba = meta_model.predict_proba(level1_test_scaled)[:, 1]
    
    # Optimiser le seuil sur validation
    threshold = optimize_threshold_from_proba(y_val, meta_val_proba)
    
    # Métriques
    val_metrics = compute_metrics(y_val, meta_val_proba, threshold)
    test_metrics = compute_metrics(y_test, meta_test_proba, threshold)
    
    log.info(f"\n📈 Résultats Meta-Learner:")
    log.info(f"  VALIDATION:")
    log.info(f"    Precision: {val_metrics['precision']:.4f}")
    log.info(f"    Recall:    {val_metrics['recall']:.4f}")
    log.info(f"    F1:        {val_metrics['f1']:.4f}")
    log.info(f"    AUC:       {val_metrics['auc_roc']:.4f}")
    
    log.info(f"\n  TEST:")
    log.info(f"    Precision: {test_metrics['precision']:.4f} 🎯")
    log.info(f"    Recall:    {test_metrics['recall']:.4f}")
    log.info(f"    F1:        {test_metrics['f1']:.4f}")
    log.info(f"    AUC:       {test_metrics['auc_roc']:.4f}")
    
    # Sauvegarder
    stacking_model = {
        "meta_model": meta_model,
        "scaler": scaler,
        "level1_models": level1_models_names,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    
    stacking_path = Path("models/stacking_meta_learner.pkl")
    joblib.dump(stacking_model, stacking_path)
    log.info(f"\n✅ Stacking ensemble sauvegardé: {stacking_path}")
    
    return stacking_model


def main():
    log.info("Chargement des données...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
    
    # Charger tous les modèles entraînés
    log.info("Chargement des modèles individuels...")
    models_dict = {}
    
    model_paths = {
        "LogisticRegression": "models/logistic_regression.pkl",
        "XGBoost": "models/xgboost_best.pkl",
        "LightGBM": "models/lightgbm_best.pkl",
        "CatBoost": "models/catboost_best.pkl",
        "RandomForest_Bagging": "models/random_forest_bagging.pkl",
    }
    
    for name, path in model_paths.items():
        if Path(path).exists():
            try:
                model = joblib.load(path)
                models_dict[name] = {"model": model}
                log.info(f"  ✓ {name} chargé")
            except Exception as e:
                log.warning(f"  ⚠️ Erreur chargement {name}: {e}")
        else:
            log.warning(f"  ⚠️ {name} non trouvé: {path}")
    
    if len(models_dict) < 3:
        log.error("❌ Au moins 3 modèles sont requis pour le stacking!")
        return
    
    # Créer l'ensemble de stacking
    stacking_model = create_stacking_ensemble(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        models_dict,
    )
    
    log.info("\n" + "="*70)
    log.info("STACKING STRATÉGIE TERMINÉE ✅")
    log.info("="*70)
    
    # Retourner les métriques
    return stacking_model["test_metrics"]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
