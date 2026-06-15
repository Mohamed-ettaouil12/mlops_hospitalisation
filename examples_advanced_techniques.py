#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════╗
║  EXEMPLE COMPLET: Techniques Avancées d'Ensemble sur Données          ║
║  Synthétiques (démonstration et testing)                              ║
║                                                                        ║
║  Cet exemple montre:                                                  ║
║  1. Génération de données synthétiques                                ║
║  2. Entraînement de modèles de base                                   ║
║  3. Application du stacking avec OOF                                  ║
║  4. Weighted blending                                                 ║
║  5. Comparaison des résultats                                         ║
╚════════════════════════════════════════════════════════════════════════╝

Utilisation:
    python examples_advanced_techniques.py
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_ensemble_techniques import (
    StackingWithOOF,
    WeightedBlending,
    compute_ensemble_metrics,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# SETUP LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 50,
    n_informative: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Génère un dataset synthétique pour démonstration.
    """
    log.info("\n" + "="*70)
    log.info("📊 GÉNÉRATION DES DONNÉES SYNTHÉTIQUES")
    log.info("="*70)
    
    # Générer les données
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        weights=[0.7, 0.3],  # Déséquilibré (70% classe 0, 30% classe 1)
        random_state=random_state,
    )
    
    # Convertir en DataFrame
    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(y, name="target")
    
    # Splits: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    
    # Reset index
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    log.info(f"\n  Dataset créé:")
    log.info(f"    Total samples: {n_samples:,}")
    log.info(f"    Features: {n_features}")
    log.info(f"    Classes: 2 (déséquilibré: 70%-30%)")
    
    log.info(f"\n  Splits:")
    log.info(f"    Train: {X_train.shape[0]:,} ({X_train.shape[0]/n_samples*100:.1f}%)")
    log.info(f"    Val:   {X_val.shape[0]:,} ({X_val.shape[0]/n_samples*100:.1f}%)")
    log.info(f"    Test:  {X_test.shape[0]:,} ({X_test.shape[0]/n_samples*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DES MODÈLES DE BASE
# ═══════════════════════════════════════════════════════════════════════════

def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Entraîne les modèles de base.
    """
    log.info("\n" + "="*70)
    log.info("🔧 ENTRAÎNEMENT DES MODÈLES DE BASE")
    log.info("="*70)
    
    models = {}
    
    # 1. Logistic Regression
    log.info("\n  1. Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    log.info(f"     ✓ AUC validation: {val_auc:.4f}")
    models["LogisticRegression"] = lr
    
    # 2. Random Forest
    log.info("\n  2. Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    log.info(f"     ✓ AUC validation: {val_auc:.4f}")
    models["RandomForest"] = rf
    
    # 3. Gradient Boosting
    log.info("\n  3. Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1])
    log.info(f"     ✓ AUC validation: {val_auc:.4f}")
    models["GradientBoosting"] = gb
    
    return models


# ═══════════════════════════════════════════════════════════════════════════
# 3. ÉVALUATION DES MODÈLES INDIVIDUELS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_individual_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Évalue les modèles individuels sur le test set.
    """
    log.info("\n" + "="*70)
    log.info("📈 ÉVALUATION DES MODÈLES INDIVIDUELS")
    log.info("="*70)
    
    results = {}
    
    for model_name, model in models.items():
        log.info(f"\n  {model_name}:")
        
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            "auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        
        log.info(f"    AUC:       {metrics['auc']:.4f}")
        log.info(f"    Precision: {metrics['precision']:.4f}")
        log.info(f"    Recall:    {metrics['recall']:.4f}")
        log.info(f"    F1:        {metrics['f1']:.4f}")
        
        results[model_name] = metrics
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. STACKING AVEC OOF
# ═══════════════════════════════════════════════════════════════════════════

def apply_stacking(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Applique le stacking avec OOF predictions.
    """
    log.info("\n" + "="*70)
    log.info("🔗 STACKING AVEC OOF PREDICTIONS")
    log.info("="*70)
    
    # Combiner train et val
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    # Créer le stacking
    stacking = StackingWithOOF(
        base_models=models,
        meta_model_type="logistic",
        n_splits=5,
        random_state=42,
    )
    
    # Entraîner
    stacking.fit(X_combined, y_combined, verbose=True)
    
    # Évaluer
    y_proba = stacking.predict_proba(X_test)
    metrics = compute_ensemble_metrics(y_test, y_proba, threshold=0.5)
    
    log.info(f"\n📊 Résultats Stacking:")
    log.info(f"    AUC:       {metrics['auc_roc']:.4f}")
    log.info(f"    Precision: {metrics['precision']:.4f} 🎯")
    log.info(f"    Recall:    {metrics['recall']:.4f}")
    log.info(f"    F1:        {metrics['f1']:.4f}")
    
    return {
        "model": stacking,
        "metrics": metrics,
        "y_proba": y_proba,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. WEIGHTED BLENDING
# ═══════════════════════════════════════════════════════════════════════════

def apply_blending(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Applique le weighted blending optimisé.
    """
    log.info("\n" + "="*70)
    log.info("⚖️  WEIGHTED BLENDING")
    log.info("="*70)
    
    # Créer le blending
    blending = WeightedBlending(
        base_models=models,
        objective_metric="f1",
    )
    
    # Optimiser les poids
    weights = blending.optimize_weights(X_val, y_val, verbose=True)
    
    # Évaluer
    y_proba = blending.predict_proba(X_test)
    metrics = compute_ensemble_metrics(y_test, y_proba, threshold=0.5)
    
    log.info(f"\n📊 Résultats Blending:")
    log.info(f"    AUC:       {metrics['auc_roc']:.4f}")
    log.info(f"    Precision: {metrics['precision']:.4f} 🎯")
    log.info(f"    Recall:    {metrics['recall']:.4f}")
    log.info(f"    F1:        {metrics['f1']:.4f}")
    
    return {
        "model": blending,
        "metrics": metrics,
        "weights": weights,
        "y_proba": y_proba,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARAISON FINAL
# ═══════════════════════════════════════════════════════════════════════════

def compare_all_approaches(
    individual_results: Dict[str, Dict[str, float]],
    stacking_result: Dict[str, Any],
    blending_result: Dict[str, Any],
):
    """
    Compare tous les approches.
    """
    log.info("\n" + "="*70)
    log.info("🏆 COMPARAISON FINAL (Test Set)")
    log.info("="*70)
    
    # Créer un dataframe de comparaison
    comparison = []
    
    # Modèles individuels
    for model_name, metrics in individual_results.items():
        comparison.append({
            "Modèle": model_name,
            "AUC": metrics["auc"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Type": "Individual",
        })
    
    # Stacking
    stacking_metrics = stacking_result["metrics"]
    comparison.append({
        "Modèle": "Stacking (OOF)",
        "AUC": stacking_metrics["auc_roc"],
        "Precision": stacking_metrics["precision"],
        "Recall": stacking_metrics["recall"],
        "F1": stacking_metrics["f1"],
        "Type": "Ensemble",
    })
    
    # Blending
    blending_metrics = blending_result["metrics"]
    comparison.append({
        "Modèle": "Weighted Blending",
        "AUC": blending_metrics["auc_roc"],
        "Precision": blending_metrics["precision"],
        "Recall": blending_metrics["recall"],
        "F1": blending_metrics["f1"],
        "Type": "Ensemble",
    })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Afficher
    log.info("\n" + comparison_df.to_string(index=False))
    
    # Gains
    best_individual = comparison_df[comparison_df["Type"] == "Individual"]["Precision"].max()
    stacking_precision = stacking_metrics["precision"]
    blending_precision = blending_metrics["precision"]
    
    log.info("\n" + "="*70)
    log.info("📊 GAINS DE PRÉCISION")
    log.info("="*70)
    log.info(f"\n  Meilleur modèle individuel:  {best_individual:.4f}")
    log.info(f"  Stacking (OOF):              {stacking_precision:.4f} (+{(stacking_precision - best_individual)*100:+.2f}%)")
    log.info(f"  Weighted Blending:           {blending_precision:.4f} (+{(blending_precision - best_individual)*100:+.2f}%)")
    
    return comparison_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution"""
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  DÉMONSTRATION: Techniques Avancées d'Ensemble                ║")
    print("║  Données synthétiques pour testing complet                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # 1. Générer les données
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_data(
        n_samples=10000,
        n_features=50,
        n_informative=30,
    )
    
    # 2. Entraîner les modèles de base
    base_models = train_base_models(X_train, y_train, X_val, y_val)
    
    # 3. Évaluer les modèles individuels
    individual_results = evaluate_individual_models(base_models, X_test, y_test)
    
    # 4. Stacking
    stacking_result = apply_stacking(
        base_models,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )
    
    # 5. Blending
    blending_result = apply_blending(
        base_models,
        X_val, y_val,
        X_test, y_test,
    )
    
    # 6. Comparaison
    comparison = compare_all_approaches(
        individual_results,
        stacking_result,
        blending_result,
    )
    
    log.info("\n" + "="*70)
    log.info("✅ DÉMONSTRATION TERMINÉE")
    log.info("="*70)
    log.info("\n  Cette démonstration a montré:")
    log.info("    ✓ Entraînement de modèles de base hétérogènes")
    log.info("    ✓ Application du Stacking avec OOF (K-fold)")
    log.info("    ✓ Application du Weighted Blending")
    log.info("    ✓ Comparaison des approches")
    log.info("\n  Pour vos données réelles:")
    log.info("    → Charger les données avec load_data_splits()")
    log.info("    → Entraîner les modèles avec vos hyperparamètres")
    log.info("    → Appliquer les mêmes techniques")
    log.info("    → Logger dans MLflow pour le suivi")
    

if __name__ == "__main__":
    main()
