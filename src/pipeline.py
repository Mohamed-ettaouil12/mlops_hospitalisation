# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — Lance toutes les étapes dans l'ordre
# ═══════════════════════════════════════════════════════════
# Usage : python src/pipeline.py

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

def main():
    log.info("🚀 PIPELINE MLOPS — DÉMARRAGE")
    log.info("="*60)

    # Étape 3 — Validation Great Expectations
    log.info("📋 Étape 1/3 : Validation des données raw...")
    try:
        import src.validate_data
        log.info("  ✅ Données validées")
    except Exception as e:
        log.error(f"  ❌ Validation échouée : {e}")
        sys.exit(1)

    # Étape 4 — Data Cleaning
    log.info("🧹 Étape 2/3 : Data Cleaning...")
    try:
        from src.data_cleaning import main as clean
        df_clean, claims = clean()
        log.info("  ✅ Cleaning terminé")
    except Exception as e:
        log.error(f"  ❌ Cleaning échoué : {e}")
        sys.exit(1)

    # Étape 5 — Data Preprocessing
    log.info("⚙️  Étape 3/3 : Data Preprocessing...")
    try:
        from src.data_preprocessing import main as preprocess
        X_train, X_test, y_train, y_test, features = preprocess()
        log.info("  ✅ Preprocessing terminé")
    except Exception as e:
        log.error(f"  ❌ Preprocessing échoué : {e}")
        sys.exit(1)

    log.info("="*60)
    log.info("✅ PIPELINE COMPLET — Prêt pour la modélisation")
    log.info(f"   Train : {len(X_train):,} | Test : {len(X_test):,}")
    log.info(f"   Features : {len(features)}")
    log.info("   Prochaine étape → python src/train.py")

if __name__ == '__main__':
    main()