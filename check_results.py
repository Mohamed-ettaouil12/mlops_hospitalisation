#!/usr/bin/env python3
"""
Script de Post-Traitement: Analyse des Résultats Finaux
À exécuter APRÈS que l'entraînement soit terminé.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def analyze_final_results() -> Dict[str, Any]:
    """Analyse les résultats d'entraînement final."""
    
    best_model_info_path = PROJECT_ROOT / "models" / "best_model_info.json"
    
    if not best_model_info_path.exists():
        print("❌ Entraînement non terminé. Lancez d'abord: python src/train.py")
        return {}
    
    print("\n" + "="*80)
    print("📊 RÉSULTATS FINAUX DE L'ENTRAÎNEMENT")
    print("="*80)
    
    with open(best_model_info_path, "r") as f:
        best_info = json.load(f)
    
    # Informations générales
    print(f"\n🏆 Meilleur Modèle: {best_info.get('best_model', 'Unknown')}")
    print(f"   Chemin: {best_info.get('best_model_path', 'N/A')}")
    print(f"   Seuil: {best_info.get('threshold', 0.5):.4f}")
    print(f"   Beta (F-score): {best_info.get('threshold_beta', 1.0)}")
    
    # Métriques TEST
    print(f"\n📈 MÉTRIQUES TEST (Performance Finale):")
    test_metrics = best_info.get("test_metrics", {})
    
    precision = test_metrics.get("precision", 0)
    recall = test_metrics.get("recall", 0)
    f1 = test_metrics.get("f1", 0)
    auc = test_metrics.get("auc_roc", 0)
    
    print(f"   Precision: {precision:.4f} (64% avant, objectif 80%)")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f} ⭐")
    
    # Matrice de Confusion
    print(f"\n🔍 Matrice de Confusion (Test):")
    cm = best_info.get("confusion_matrix_test", {})
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    
    print(f"   Vrais Négatifs:  {tn:,}")
    print(f"   Faux Positifs:   {fp:,}")
    print(f"   Faux Négatifs:   {fn:,}")
    print(f"   Vrais Positifs:  {tp:,}")
    
    # Verdict
    print(f"\n{'='*80}")
    if precision >= 0.80:
        print(f"✅ SUCCÈS! Précision {precision:.2%} ≥ 80% (Objectif atteint!)")
    elif precision >= 0.78:
        print(f"🟡 CLOSE! Précision {precision:.2%} (proche de 80%, peut nécessiter stacking)")
    else:
        print(f"🟠 PARTIEL. Précision {precision:.2%} (nécessite stratégies avancées)")
    
    print(f"{'='*80}")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    
    if precision >= 0.80:
        print(f"   ✅ Objectif 80% ATTEINT! Modèle prêt pour production.")
        print(f"   Considérez l'API de déploiement: src/api.py")
    elif precision >= 0.77:
        print(f"   🟡 Très proche! Essayez:")
        print(f"      1. Stacking avec meta-learner: python advanced_stacking.py")
        print(f"      2. Feature engineering supplémentaire")
        print(f"      3. SMOTE pour rééquilibrage")
    else:
        print(f"   🟠 Essayez les stratégies avancées:")
        print(f"      1. Stacking: python advanced_stacking.py")
        print(f"      2. Feature selection/engineering")
        print(f"      3. Hyperparamètres plus agressifs (N_OPTUNA_TRIALS=200)")
        print(f"      4. Rééquilibrage: SMOTE")
    
    # Comparaison avant/après
    print(f"\n📊 AVANT / APRÈS:")
    print(f"   Précision avant: 65.54% → après: {precision:.2%} (+{(precision-0.6554)*100:.1f}pp)")
    print(f"   AUC avant: 96.86% → après: {auc:.2%} (+{(auc-0.9686)*100:.2f}pp)")
    
    # Sauvegarder le rapport
    report = {
        "modele": best_info.get("best_model"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "objectif_atteint": precision >= 0.80,
        "precision_delta": precision - 0.6554,
    }
    
    report_path = PROJECT_ROOT / "models" / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: {report_path}")
    
    return report


if __name__ == "__main__":
    analyze_final_results()
