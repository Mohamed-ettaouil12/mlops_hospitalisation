#!/usr/bin/env python3
"""
Auto-Pipeline: Entraînement Principal + Stacking Automatique si besoin
Exécution: python auto_pipeline.py
"""

import json
import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    """Exécute le pipeline complet avec stacking optionnel."""
    
    print("\n" + "="*80)
    print("🚀 AUTO-PIPELINE: Entraînement + Amélioration Automatique")
    print("="*80)
    
    # Étape 1: Entraînement principal
    print("\n📚 ÉTAPE 1: Entraînement des Modèles (src/train.py)")
    print("-" * 80)
    print("⏳ Ceci peut prendre 30-50 minutes...")
    print("   Vous pouvez surveiller avec: tail -f training.log")
    
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 heure max
        )
        
        if result.returncode != 0:
            print(f"❌ Erreur dans l'entraînement principal:")
            print(result.stderr)
            return False
        
        print("✅ Entraînement principal terminé!")
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: L'entraînement a pris plus de 1 heure.")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Étape 2: Vérifier les résultats
    print("\n📊 ÉTAPE 2: Vérification des Résultats")
    print("-" * 80)
    
    best_model_info_path = PROJECT_ROOT / "models" / "best_model_info.json"
    
    if not best_model_info_path.exists():
        print("❌ Fichier best_model_info.json non trouvé!")
        return False
    
    with open(best_model_info_path, "r") as f:
        best_info = json.load(f)
    
    precision = best_info.get("test_metrics", {}).get("precision", 0)
    
    print(f"Meilleur modèle: {best_info.get('best_model')}")
    print(f"Précision TEST: {precision:.2%}")
    
    # Étape 3: Décider si stacking est nécessaire
    if precision >= 0.80:
        print(f"\n✅ EXCELLENT! Précision {precision:.2%} ≥ 80%")
        print("   Objectif atteint! Pas besoin de stacking.")
        return True
    
    elif precision >= 0.77:
        print(f"\n🟡 CLOSE! Précision {precision:.2%} (< 80% mais proche)")
        response = input("   Voulez-vous essayer le Stacking? (y/n): ").strip().lower()
        
        if response != 'y':
            print("   ❌ Stacking ignoré.")
            return True
    else:
        print(f"\n🟠 Précision {precision:.2%} < 80%. Tentative de Stacking...")
    
    # Étape 4: Stacking
    print("\n🔗 ÉTAPE 3: Stacking Meta-Learner (advanced_stacking.py)")
    print("-" * 80)
    print("⏳ Ceci prend environ 5-10 minutes...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "advanced_stacking.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"⚠️  Avertissement dans le stacking:")
            print(result.stderr)
        else:
            print("✅ Stacking terminé!")
        
        # Vérifier les résultats du stacking
        stacking_path = PROJECT_ROOT / "models" / "stacking_meta_learner.pkl"
        if stacking_path.exists():
            print("✅ Modèle stacking sauvegardé avec succès!")
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Stacking a pris plus de 10 minutes.")
    except Exception as e:
        print(f"⚠️  Erreur stacking: {e}")
    
    # Étape 5: Résumé final
    print("\n" + "="*80)
    print("✅ AUTO-PIPELINE TERMINÉ")
    print("="*80)
    
    print(f"\n📁 Fichiers Générés:")
    print(f"   models/best_model.pkl - Meilleur modèle")
    print(f"   models/best_model_info.json - Métriques finales")
    print(f"   models/catboost_best.pkl - CatBoost")
    print(f"   models/soft_voting_xgb_lgb_cat.pkl - Soft Voting V2")
    print(f"   models/stacking_meta_learner.pkl - Stacking (si exécuté)")
    
    print(f"\n📊 Vérifier les résultats:")
    print(f"   python check_results.py")
    
    print(f"\n🌐 Lancer l'API:")
    print(f"   python src/api.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
