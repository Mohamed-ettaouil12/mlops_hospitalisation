#!/usr/bin/env python3
"""
Analyse : Est-ce que 80% de précision est atteignable ?
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import (
    load_data, 
    compute_metrics,
    confusion_matrix_dict,
)

def analyze_precision_target():
    """Analyse si 80% de précision est atteignable."""
    
    print("\n" + "="*70)
    print("ANALYSE: Peut-on atteindre 80% de précision?")
    print("="*70)
    
    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
    
    # Charger le meilleur modèle
    model = joblib.load("models/best_model.pkl")
    best_model_info = None
    if Path("models/best_model_info.json").exists():
        with open("models/best_model_info.json", "r") as f:
            best_model_info = json.load(f)
    
    print(f"\n📊 Modèle actuel: {best_model_info.get('best_model', 'Unknown') if best_model_info else 'Unknown'}")
    print(f"   Seuil actuel: {best_model_info.get('threshold', 0.5):.4f}")
    print(f"   Précision TEST actuelle: {best_model_info.get('test_metrics', {}).get('precision', 0):.4f}")
    print(f"   Rappel TEST actuel: {best_model_info.get('test_metrics', {}).get('recall', 0):.4f}")
    
    # Calculer probas sur validation ET test
    print(f"\n🔍 Analyse des courbes Précision-Rappel...")
    
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Sur VALIDATION
    print("\n--- VALIDATION SET ---")
    precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, val_proba)
    
    # Vérifier si 80% est atteignable sur validation
    target_precision = 0.80
    valid_mask = precisions_val[:-1] >= target_precision
    
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        # Garder le seuil qui maximize le recall
        best_idx = valid_indices[np.argmax(recalls_val[valid_indices])]
        threshold_for_80 = thresholds_val[best_idx]
        precision_at_80 = precisions_val[best_idx]
        recall_at_80 = recalls_val[best_idx]
        
        print(f"✅ 80% de précision EST ATTEIGNABLE sur validation!")
        print(f"   Seuil recommandé: {threshold_for_80:.4f}")
        print(f"   Précision: {precision_at_80:.4f}")
        print(f"   Rappel: {recall_at_80:.4f}")
        print(f"   Prédictions positives: {(val_proba >= threshold_for_80).sum()}")
        
        # Vérifier sur TEST
        print("\n--- TEST SET (pour information) ---")
        test_metrics = compute_metrics(y_test, test_proba, threshold_for_80)
        test_cm = confusion_matrix_dict(y_test, test_proba, threshold_for_80)
        
        print(f"   Précision TEST à ce seuil: {test_metrics['precision']:.4f}")
        print(f"   Rappel TEST: {test_metrics['recall']:.4f}")
        print(f"   F1 TEST: {test_metrics['f1']:.4f}")
        print(f"   Prédictions positives TEST: {test_cm['fp'] + test_cm['tp']}")
        
        # Afficher les top seuils pour différentes précisions cibles
        print(f"\n📈 Distribution des seuils pour différentes précisions cibles:")
        print(f"{'Précision cible':<20} {'Seuil':<15} {'Rappel max':<15} {'Prédictions +':<15}")
        print("-" * 65)
        
        for target in [0.70, 0.75, 0.80, 0.85, 0.90]:
            mask = precisions_val[:-1] >= target
            if np.any(mask):
                indices = np.where(mask)[0]
                best_recall_idx = indices[np.argmax(recalls_val[indices])]
                thresh = thresholds_val[best_recall_idx]
                recall = recalls_val[best_recall_idx]
                precision = precisions_val[best_recall_idx]
                n_pred_pos = (val_proba >= thresh).sum()
                print(f"{target:.2f}              {thresh:.4f}         {recall:.4f}          {n_pred_pos:<15}")
    else:
        print(f"❌ 80% de précision n'est PAS atteignable sur validation")
        print(f"   Précision maximale possible: {np.max(precisions_val[:-1]):.4f}")
        
    print("\n" + "="*70)
    print("RECOMMANDATIONS:")
    print("="*70)
    
    if np.any(valid_mask):
        print("""
✅ 80% de précision est POSSIBLE avec les modèles actuels!

Options pour améliorer davantage:
1. ✓ Utiliser le seuil optimisé calculé ci-dessus
2. Ajouter CatBoost à l'ensemble
3. Réoptimiser les poids du soft voting
4. Implémenter un stacking avec meta-learner
5. Ingénierie des features supplémentaire
        """)
    else:
        print("""
❌ 80% de précision n'est pas directement atteignable.

Options pour améliorer:
1. Ajouter des modèles plus puissants (CatBoost, HistGradientBoosting)
2. Implémenter un stacking avec meta-learner
3. Faire de l'ingénierie des features plus agressive
4. Rééquilibrer les classes (SMOTE)
5. Augmenter les hyperparamètres d'optuna
        """)

if __name__ == "__main__":
    try:
        analyze_precision_target()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
