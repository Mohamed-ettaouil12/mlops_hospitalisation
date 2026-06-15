#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 🚀 QUICK START: Commandes Rapides pour les Techniques Avancées
# ═══════════════════════════════════════════════════════════════════════════

# Aller au répertoire du projet
cd /home/tawil/mlops_hospitalisation

echo "═══════════════════════════════════════════════════════════════════════════"
echo "🚀 TECHNIQUES AVANCÉES D'ENSEMBLE"
echo "═══════════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────
# 1. VOIR LA DÉMO (30 SECONDES)
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "1️⃣  DÉMO RAPIDE (données synthétiques)"
echo "───────────────────────────────────────────────────────────────────────"
echo "Command: python examples_advanced_techniques.py"
echo ""
echo "Cette commande montre:"
echo "  ✓ Génération de données synthétiques"
echo "  ✓ Entraînement de 3 modèles de base"
echo "  ✓ Application du stacking avec OOF"
echo "  ✓ Application du weighted blending"
echo "  ✓ Comparaison des résultats"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 2. APPLIQUER À VOS DONNÉES
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "2️⃣  APPLIQUER AUX DONNÉES RÉELLES"
echo "───────────────────────────────────────────────────────────────────────"
echo ""

echo "🎯 Option A: Stacking avec OOF (meilleur gain de précision)"
echo "Command: python apply_advanced_ensemble.py --technique stacking --n-splits 5"
echo ""

echo "⚖️  Option B: Weighted Blending (plus rapide)"
echo "Command: python apply_advanced_ensemble.py --technique blending"
echo ""

echo "🏷️  Option C: Pseudo-Labeling (si données non labellisées)"
echo "Command: python apply_advanced_ensemble.py --technique pseudo-labeling --threshold 0.95"
echo ""

echo "🔥 Option D: Tout appliquer"
echo "Command: python apply_advanced_ensemble.py --technique all"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 3. VOIR LES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "3️⃣  TRACKER LES RÉSULTATS DANS MLflow"
echo "───────────────────────────────────────────────────────────────────────"
echo "Command: mlflow ui"
echo ""
echo "Puis ouvrir: http://localhost:5000"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 4. GUIDE DÉTAILLÉ
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "📖 DOCUMENTATION"
echo "───────────────────────────────────────────────────────────────────────"
echo "  • README_ADVANCED.md          ← Guide de démarrage (LISEZ CECI D'ABORD)"
echo "  • ADVANCED_ENSEMBLE_GUIDE.md  ← Guide complet avec best practices"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 5. FICHIERS CRÉÉS
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "✨ FICHIERS CRÉÉS"
echo "───────────────────────────────────────────────────────────────────────"
echo "  src/advanced_ensemble_techniques.py    → Classes principales"
echo "  apply_advanced_ensemble.py             → Script d'application"
echo "  examples_advanced_techniques.py        → Exemple complet"
echo "  ADVANCED_ENSEMBLE_GUIDE.md             → Guide détaillé"
echo "  README_ADVANCED.md                     → Guide de démarrage"
echo "  QUICKSTART.sh                          → Ce fichier"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 6. STRUCTURE DES TECHNIQUES
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "🎯 TECHNIQUES DISPONIBLES"
echo "───────────────────────────────────────────────────────────────────────"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ 1. STACKING AVEC OOF PREDICTIONS                            │"
echo "│    ├─ Gain: +2 à +3% de précision                          │"
echo "│    ├─ Complexité: Moyenne (K-fold CV)                      │"
echo "│    ├─ Temps: Élevé (n_splits * n_models)                  │"
echo "│    └─ Recommandé: OUI ✅ (meilleur gain)                   │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ 2. WEIGHTED BLENDING OPTIMISÉ                              │"
echo "│    ├─ Gain: +1 à +2% de précision                          │"
echo "│    ├─ Complexité: Basse (optimisation simple)              │"
echo "│    ├─ Temps: Très rapide                                   │"
echo "│    └─ Recommandé: OUI ✅ (rapide et efficace)              │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ 3. PSEUDO-LABELING                                         │"
echo "│    ├─ Gain: +1 à +2% de précision                          │"
echo "│    ├─ Complexité: Moyenne                                  │"
echo "│    ├─ Temps: Moyen                                         │"
echo "│    └─ Recommandé: OUI ✅ (si données non labellisées)      │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 7. GAINS ATTENDUS
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "📈 GAINS ATTENDUS"
echo "───────────────────────────────────────────────────────────────────────"
echo ""
echo "  Baseline (meilleur modèle seul):  0% (référence)"
echo "  ⬆️ Stacking:                      +2 à +3% 🎯"
echo "  ⬆️ Weighted Blending:             +1 à +2% 🎯"
echo "  ⬆️ Pseudo-Labeling:               +1 à +2% 🎯"
echo "  ⬆️ ENSEMBLE COMPLET:              +3 à +5% 🏆"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 8. EXÉCUTION PROGRESSIVE
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "🚀 ROADMAP D'EXÉCUTION RECOMMANDÉE"
echo "───────────────────────────────────────────────────────────────────────"
echo ""
echo "Jour 1 (Lundi):"
echo "  1. Lire: README_ADVANCED.md"
echo "  2. Exécuter: python examples_advanced_techniques.py"
echo ""
echo "Jour 2-3 (Mardi-Mercredi):"
echo "  1. Appliquer stacking: python apply_advanced_ensemble.py --technique stacking"
echo "  2. Comparer: mlflow ui"
echo ""
echo "Jour 4-5 (Jeudi-Vendredi):"
echo "  1. Optimiser hyperparamètres avec Optuna (si nécessaire)"
echo "  2. Mettre en place le monitoring en production"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 9. EXEMPLES DE CODE
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "💻 EXEMPLES DE CODE RAPIDES"
echo "───────────────────────────────────────────────────────────────────────"
echo ""

echo "📝 Stacking (exemple minimal):"
cat << 'EOF'

from src.advanced_ensemble_techniques import StackingWithOOF
import xgboost as xgb
import lightgbm as lgb

base_models = {
    "XGBoost": xgb.XGBClassifier(),
    "LightGBM": lgb.LGBMClassifier(),
}

stacking = StackingWithOOF(base_models, n_splits=5)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)

EOF
echo ""

echo "📝 Weighted Blending (exemple minimal):"
cat << 'EOF'

from src.advanced_ensemble_techniques import WeightedBlending

blending = WeightedBlending(base_models, objective_metric="f1")
weights = blending.optimize_weights(X_val, y_val)
y_pred = blending.predict_proba(X_test)

EOF
echo ""

# ─────────────────────────────────────────────────────────────────────────
# 10. RÉSUMÉ
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ RÉSUMÉ DES ACTIONS"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Vous avez maintenant:"
echo ""
echo "  ✓ 3 techniques avancées d'ensemble implémentées"
echo "  ✓ Scripts d'application prêts à l'emploi"
echo "  ✓ Exemple complet et fonctionnel"
echo "  ✓ Documentation détaillée"
echo ""
echo "Prochaines étapes:"
echo ""
echo "  1️⃣  python examples_advanced_techniques.py         (démo)"
echo "  2️⃣  python apply_advanced_ensemble.py --technique all  (appliquer)"
echo "  3️⃣  mlflow ui                                      (voir résultats)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
