#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════╗
║  SCRIPT D'APPLICATION: Techniques Avancées d'Ensemble                 ║
║  - Implémentation complète du stacking avec OOF                       ║
║  - Weighted blending optimisé                                         ║
║  - Pseudo-labeling                                                    ║
║  - Feature engineering croisé                                         ║
║  - MLflow integration                                                 ║
╚════════════════════════════════════════════════════════════════════════╝

Usage:
    python apply_advanced_ensemble.py --technique stacking --experiment-name my_exp
    python apply_advanced_ensemble.py --technique blending --optimize-weights
    python apply_advanced_ensemble.py --technique pseudo-labeling --threshold 0.95
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Add parent to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_ensemble_techniques import (
    StackingWithOOF,
    WeightedBlending,
    PseudoLabelingStrategy,
    create_cross_features,
    log_ensemble_to_mlflow,
    compute_ensemble_metrics,
    save_ensemble_model,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
FIGURES_DIR = Path("outputs/figures")
ENSEMBLE_DIR = MODELS_DIR / "ensembles"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def load_trained_models() -> Dict[str, Any]:
    """Charge les modèles entraînés du répertoire models/"""
    models = {}
    
    # Essayer de charger les modèles disponibles
    model_names = ["xgboost_model", "lightgbm_model", "catboost_model", 
                   "logistic_model", "random_forest_model"]
    
    for model_name in model_names:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
                log.info(f"✓ Modèle chargé: {model_name}")
            except Exception as e:
                log.warning(f"⚠️ Erreur chargement {model_name}: {e}")
    
    if not models:
        log.warning("⚠️ Aucun modèle trouvé dans models/")
        log.info("   Les modèles doivent d'abord être entraînés avec src/train.py")
    
    return models


def load_data_splits(
    features_dir: Path = Path("data/features"),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge les splits train/val/test"""
    log.info("\n" + "="*70)
    log.info("📂 Chargement des données...")
    log.info("="*70)
    
    # Charger les features
    X_train = pd.read_parquet(features_dir / "X_train.parquet")
    X_val = pd.read_parquet(features_dir / "X_val.parquet")
    X_test = pd.read_parquet(features_dir / "X_test.parquet")
    
    # Charger les targets
    y_train = pd.read_parquet(features_dir / "y_train.parquet").squeeze()
    y_val = pd.read_parquet(features_dir / "y_val.parquet").squeeze()
    y_test = pd.read_parquet(features_dir / "y_test.parquet").squeeze()
    
    log.info(f"\n  Train:  X {X_train.shape}, y {y_train.shape}")
    log.info(f"  Val:    X {X_val.shape}, y {y_val.shape}")
    log.info(f"  Test:   X {X_test.shape}, y {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 1. STACKING AVEC OOF
# ═══════════════════════════════════════════════════════════════════════════

def apply_stacking_oof(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "stacking_oof",
    n_splits: int = 5,
    meta_model: str = "logistic",
) -> Dict[str, Any]:
    """
    Applique le stacking avec OOF predictions.
    """
    if not models:
        log.error("❌ Aucun modèle disponible pour le stacking")
        return {}
    
    log.info("\n" + "="*70)
    log.info("🎯 TECHNIQUE 1: STACKING AVEC OOF PREDICTIONS")
    log.info("="*70)
    
    # Créer le stacking
    stacking = StackingWithOOF(
        base_models=models,
        meta_model_type=meta_model,
        n_splits=n_splits,
        random_state=42,
    )
    
    # Entraîner sur train et val ensemble
    X_train_combined = pd.concat([X_train, X_val], ignore_index=False)
    y_train_combined = pd.concat([y_train, y_val], ignore_index=False)
    
    stacking.fit(X_train_combined, y_train_combined, verbose=True)
    
    # Évaluer sur test
    y_test_proba = stacking.predict_proba(X_test)
    threshold = 0.5
    
    test_metrics = compute_ensemble_metrics(y_test, y_test_proba, threshold)
    
    log.info("\n📊 RÉSULTATS STACKING:")
    log.info(f"  Precision: {test_metrics['precision']:.4f} 🎯")
    log.info(f"  Recall:    {test_metrics['recall']:.4f}")
    log.info(f"  F1:        {test_metrics['f1']:.4f}")
    log.info(f"  AUC:       {test_metrics['auc_roc']:.4f}")
    
    # Sauvegarder
    model_path = save_ensemble_model(
        stacking,
        ENSEMBLE_DIR,
        model_name="stacking_oof",
    )
    
    # Log MLflow
    params = {
        "technique": "stacking_oof",
        "n_splits": n_splits,
        "meta_model": meta_model,
        "n_base_models": len(models),
        "base_models": list(models.keys()),
    }
    
    log_ensemble_to_mlflow(
        experiment_name,
        "stacking_oof",
        {
            "metrics": test_metrics,
            "params": params,
        },
        params,
        artifact_dir=None,
    )
    
    return {
        "model": stacking,
        "model_path": model_path,
        "metrics": test_metrics,
        "y_proba": y_test_proba,
        "technique": "stacking_oof",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. WEIGHTED BLENDING
# ═══════════════════════════════════════════════════════════════════════════

def apply_weighted_blending(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "weighted_blending",
    objective: str = "f1",
) -> Dict[str, Any]:
    """
    Applique le blending pondéré optimisé.
    """
    if not models:
        log.error("❌ Aucun modèle disponible pour le blending")
        return {}
    
    log.info("\n" + "="*70)
    log.info("⚖️  TECHNIQUE 2: WEIGHTED BLENDING OPTIMISÉ")
    log.info("="*70)
    
    # Créer le blending
    blending = WeightedBlending(
        base_models=models,
        objective_metric=objective,
        beta=1.0,
    )
    
    # Optimiser les poids sur validation
    weights = blending.optimize_weights(X_val, y_val, verbose=True)
    
    # Évaluer sur test
    y_test_proba = blending.predict_proba(X_test)
    threshold = 0.5
    
    test_metrics = compute_ensemble_metrics(y_test, y_test_proba, threshold)
    
    log.info("\n📊 RÉSULTATS WEIGHTED BLENDING:")
    log.info(f"  Precision: {test_metrics['precision']:.4f} 🎯")
    log.info(f"  Recall:    {test_metrics['recall']:.4f}")
    log.info(f"  F1:        {test_metrics['f1']:.4f}")
    log.info(f"  AUC:       {test_metrics['auc_roc']:.4f}")
    
    # Sauvegarder
    model_path = save_ensemble_model(
        blending,
        ENSEMBLE_DIR,
        model_name="weighted_blending",
    )
    
    # Log MLflow
    params = {
        "technique": "weighted_blending",
        "objective_metric": objective,
        "n_base_models": len(models),
        "base_models": list(models.keys()),
        **{f"weight_{k}": v for k, v in weights.items()},
    }
    
    log_ensemble_to_mlflow(
        experiment_name,
        "weighted_blending",
        {
            "metrics": test_metrics,
            "weights": weights,
            "params": params,
        },
        params,
    )
    
    return {
        "model": blending,
        "model_path": model_path,
        "metrics": test_metrics,
        "weights": weights,
        "y_proba": y_test_proba,
        "technique": "weighted_blending",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. PSEUDO-LABELING
# ═══════════════════════════════════════════════════════════════════════════

def apply_pseudo_labeling(
    models: Dict[str, Any],
    X_unlabeled: Optional[pd.DataFrame] = None,
    confidence_threshold: float = 0.95,
    experiment_name: str = "pseudo_labeling",
) -> Dict[str, Any]:
    """
    Applique le pseudo-labeling sur des données non labellisées.
    """
    if not models:
        log.error("❌ Aucun modèle disponible pour le pseudo-labeling")
        return {}
    
    log.info("\n" + "="*70)
    log.info("🏷️  TECHNIQUE 3: PSEUDO-LABELING")
    log.info("="*70)
    
    # Si pas de données non labellisées, créer un exemple synthétique
    if X_unlabeled is None:
        log.info("  ℹ️  Pas de données non labellisées fournies")
        log.info("     (Dans la pratique, utiliser vos données réelles sans labels)")
        return {}
    
    # Utiliser le premier modèle pour générer les pseudo-labels
    best_model = list(models.values())[0]
    
    pseudo_strategy = PseudoLabelingStrategy(
        model=best_model,
        confidence_threshold=confidence_threshold,
    )
    
    X_pseudo, y_pseudo, confidence = pseudo_strategy.generate_pseudo_labels(
        X_unlabeled,
        verbose=True,
    )
    
    if len(X_pseudo) == 0:
        log.warning("  ⚠️ Aucun pseudo-label généré (confiance trop élevée)")
        return {}
    
    log.info("\n✅ Pseudo-labels générés avec succès!")
    log.info("   Prochaine étape: Combiner avec les données labelisées et réentraîner")
    
    return {
        "X_pseudo": X_pseudo,
        "y_pseudo": y_pseudo,
        "confidence": confidence,
        "n_pseudo_labels": len(X_pseudo),
        "technique": "pseudo_labeling",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Appliquer les techniques avancées d'ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python apply_advanced_ensemble.py --technique stacking
    python apply_advanced_ensemble.py --technique blending
    python apply_advanced_ensemble.py --technique pseudo-labeling --threshold 0.95
        """,
    )
    
    parser.add_argument(
        "--technique",
        choices=["stacking", "blending", "pseudo-labeling", "all"],
        default="all",
        help="Technique à appliquer",
    )
    parser.add_argument(
        "--experiment-name",
        default="advanced_ensemble",
        help="Nom de l'expérience MLflow",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Nombre de folds pour le stacking",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confiance minimale pour pseudo-labeling",
    )
    parser.add_argument(
        "--meta-model",
        choices=["logistic", "ridge"],
        default="logistic",
        help="Type de méta-modèle pour le stacking",
    )
    
    args = parser.parse_args()
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. Charger les données
    # ─────────────────────────────────────────────────────────────────────
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_splits()
    except Exception as e:
        log.error(f"❌ Erreur chargement données: {e}")
        log.error("   Assurez-vous que data/features/ contient les fichiers parquet")
        return 1
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. Charger les modèles
    # ─────────────────────────────────────────────────────────────────────
    
    models = load_trained_models()
    
    if not models:
        log.error("❌ Aucun modèle entraîné trouvé")
        log.info("   Entraînez d'abord les modèles avec: python src/train.py")
        return 1
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. Appliquer les techniques
    # ─────────────────────────────────────────────────────────────────────
    
    results = {}
    
    if args.technique in ["stacking", "all"]:
        try:
            results["stacking"] = apply_stacking_oof(
                models,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                experiment_name=args.experiment_name,
                n_splits=args.n_splits,
                meta_model=args.meta_model,
            )
        except Exception as e:
            log.error(f"❌ Erreur stacking: {e}")
            import traceback
            traceback.print_exc()
    
    if args.technique in ["blending", "all"]:
        try:
            results["blending"] = apply_weighted_blending(
                models,
                X_val, y_val,
                X_test, y_test,
                experiment_name=args.experiment_name,
                objective="f1",
            )
        except Exception as e:
            log.error(f"❌ Erreur blending: {e}")
            import traceback
            traceback.print_exc()
    
    if args.technique in ["pseudo-labeling", "all"]:
        try:
            results["pseudo_labeling"] = apply_pseudo_labeling(
                models,
                X_unlabeled=None,  # Fournir vos données non labellisées ici
                confidence_threshold=args.threshold,
                experiment_name=args.experiment_name,
            )
        except Exception as e:
            log.error(f"❌ Erreur pseudo-labeling: {e}")
            import traceback
            traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. Rapport final
    # ─────────────────────────────────────────────────────────────────────
    
    log.info("\n" + "="*70)
    log.info("📊 RAPPORT FINAL")
    log.info("="*70)
    
    for technique_name, result in results.items():
        if result and "metrics" in result:
            metrics = result["metrics"]
            log.info(f"\n  {technique_name.upper()}:")
            log.info(f"    Precision: {metrics.get('precision', 'N/A')}")
            log.info(f"    Recall:    {metrics.get('recall', 'N/A')}")
            log.info(f"    F1:        {metrics.get('f1', 'N/A')}")
            log.info(f"    AUC:       {metrics.get('auc_roc', 'N/A')}")
    
    log.info("\n✅ Toutes les techniques appliquées avec succès!")
    log.info(f"\n   Modèles sauvegardés dans: {ENSEMBLE_DIR}")
    log.info(f"   Résultats logés dans MLflow: {args.experiment_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
