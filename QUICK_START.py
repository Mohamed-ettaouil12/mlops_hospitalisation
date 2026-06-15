#!/usr/bin/env python3
"""
🎯 GUIDE DE DÉMARRAGE: Par où commencer?

Ce fichier te guide étape par étape sur:
1. Quoi faire ensuite
2. Quels fichiers créer
3. Commandes à exécuter
4. Résultats attendus
"""

def print_current_state():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 📊 ÉTAT ACTUEL DE TON PROJET                                         ║
╚════════════════════════════════════════════════════════════════════════╝

✅ COMPLÉTÉ (Phases 1-3):
   └─ Phase 1: Data Preprocessing
      ├─ EDA ✅
      ├─ Cleaning ✅  
      ├─ Feature engineering ✅
      └─ Train/val/test splits ✅
   
   └─ Phase 2: Baseline Models
      ├─ XGBoost (F1: 0.614, AUC: 0.969) ✅
      ├─ LightGBM ✅
      ├─ CatBoost ✅
      └─ RandomForest ✅
   
   └─ Phase 3: Advanced Ensembles
      ├─ Stacking OOF ✅
      ├─ Weighted Blending ✅
      └─ Pseudo-Labeling ✅

⏳ À FAIRE (Phases 4-8):
   └─ Phase 4: Optuna Tuning (2 jours)
      Gain: +1-2% précision
      
   └─ Phase 5: SHAP + Fairness (1 jour)
      Impact: Explainability + bias detection
      
   └─ Phase 6: Monitoring (2 jours)
      Impact: Production monitoring
      
   └─ Phase 7: CI/CD (2 jours)
      Impact: Automated testing/training
      
   └─ Phase 8: Streaming (3 jours)
      Impact: Real-time predictions

═════════════════════════════════════════════════════════════════════════

⏱️ TIMELINE TOTALE: 10 jours (~2 semaines)

SEMAINE 1:
  ☐ Lun-Mar (4h):  Phase 4: Optuna
  ☐ Mer (3h):      Phase 5: SHAP + Fairness
  ☐ Jeu-Ven (4h):  Phase 6: Monitoring

SEMAINE 2:
  ☐ Lun-Mar (4h):  Phase 7: CI/CD
  ☐ Mer-Ven (6h):  Phase 8: Streaming + integration
  ☐ Ven (1h):      Tests finaux

═════════════════════════════════════════════════════════════════════════
""")

def print_quick_start():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 🚀 DÉMARRAGE RAPIDE                                                  ║
╚════════════════════════════════════════════════════════════════════════╝

OPTION 1: Je veux comprendre avant de coder
──────────────────────────────────────────

1. Lire la roadmap:
   $ python ROADMAP_FINAL_PHASES.py

2. Lire le plan d'action:
   $ cat ACTION_PLAN_2WEEKS.md

3. Voir les fichiers à créer:
   $ bash FILES_TO_CREATE.sh

OPTION 2: Je veux commencer Phase 4 (Optuna) maintenant
─────────────────────────────────────────────────────

1. Installer Optuna:
   $ pip install optuna

2. Créer src/optuna_tuning.py:
   $ cat > src/optuna_tuning.py << 'EOF'
   [Voir code ci-dessous]
   EOF

3. Créer le script principal:
   $ cat > optimize_hyperparameters.py << 'EOF'
   [Voir code ci-dessous]
   EOF

4. Exécuter:
   $ python optimize_hyperparameters.py --n-trials 100

OPTION 3: Je veux voir les résultats des phases 1-3
──────────────────────────────────────────────────

1. Voir MLflow:
   $ mlflow ui

2. Voir les models:
   $ ls -la models/

3. Voir les résultats:
   $ cat models/best_model_info.json

═════════════════════════════════════════════════════════════════════════
""")

def print_phase4_code():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 🔵 PHASE 4: OPTUNA TUNING - CODE COMPLET                             ║
╚════════════════════════════════════════════════════════════════════════╝

FICHIER 1: src/optuna_tuning.py
────────────────────────────────
""")
    
    code = '''import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptunaTuner:
    """Optimize hyperparameters for XGBoost and LightGBM"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def objective_xgb(self, trial):
        """Objective function for XGBoost optimization"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        try:
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            model.fit(
                self.X_train, 
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            y_pred = model.predict(self.X_val)
            score = f1_score(self.y_val, y_pred)
            
            logger.info(f"Trial {trial.number}: F1={score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def objective_lgb(self, trial):
        """Objective function for LightGBM optimization"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        }
        
        try:
            model = lgb.LGBMClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(
                self.X_train, 
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=10,
            )
            
            y_pred = model.predict(self.X_val)
            score = f1_score(self.y_val, y_pred)
            
            logger.info(f"Trial {trial.number}: F1={score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def optimize_xgb(self, n_trials=100, direction='maximize'):
        """Optimize XGBoost hyperparameters"""
        logger.info(f"Starting XGBoost optimization ({n_trials} trials)...")
        
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective_xgb, n_trials=n_trials)
        
        logger.info(f"✅ Best XGBoost F1: {study.best_value:.4f}")
        logger.info(f"✅ Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def optimize_lgb(self, n_trials=100, direction='maximize'):
        """Optimize LightGBM hyperparameters"""
        logger.info(f"Starting LightGBM optimization ({n_trials} trials)...")
        
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective_lgb, n_trials=n_trials)
        
        logger.info(f"✅ Best LightGBM F1: {study.best_value:.4f}")
        logger.info(f"✅ Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value
'''
    
    print(code)

def print_phase4_main():
    print("""
FICHIER 2: optimize_hyperparameters.py
──────────────────────────────────────
""")
    
    code = '''import argparse
import pandas as pd
import joblib
import json
from pathlib import Path
from src.optuna_tuning import OptunaTuner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--models', nargs='+', default=['xgb', 'lgb'])
    args = parser.parse_args()
    
    # Charger les données
    logger.info("Loading data...")
    X_train = pd.read_parquet('data/features/X_train.parquet')
    y_train = pd.read_parquet('data/features/y_train.parquet').squeeze()
    X_val = pd.read_parquet('data/features/X_val.parquet')
    y_val = pd.read_parquet('data/features/y_val.parquet').squeeze()
    
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Initialiser le tuner
    tuner = OptunaTuner(X_train, y_train, X_val, y_val)
    
    best_params = {}
    
    # Optimiser XGBoost
    if 'xgb' in args.models:
        xgb_params, xgb_score = tuner.optimize_xgb(n_trials=args.n_trials)
        best_params['xgboost'] = {
            'params': xgb_params,
            'best_f1': xgb_score
        }
    
    # Optimiser LightGBM
    if 'lgb' in args.models:
        lgb_params, lgb_score = tuner.optimize_lgb(n_trials=args.n_trials)
        best_params['lightgbm'] = {
            'params': lgb_params,
            'best_f1': lgb_score
        }
    
    # Sauvegarder
    output_path = Path('models/best_params.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"✅ Best parameters saved to {output_path}")
    logger.info(f"✅ Summary:")
    for model, data in best_params.items():
        logger.info(f"   {model}: F1={data['best_f1']:.4f}")

if __name__ == "__main__":
    main()
'''
    
    print(code)

def print_execution_steps():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 📋 ÉTAPES D'EXÉCUTION: PHASE 4 (Optuna)                              ║
╚════════════════════════════════════════════════════════════════════════╝

ÉTAPE 1: Installer Optuna
─────────────────────────
$ pip install optuna

Résultat attendu:
  ✅ optuna installé
  ✅ optuna --version (pour vérifier)

ÉTAPE 2: Créer les fichiers
──────────────────────────
Copier le code ci-dessus dans:
  1. src/optuna_tuning.py
  2. optimize_hyperparameters.py

Commandes rapides:
  # Créer src/optuna_tuning.py
  cat > src/optuna_tuning.py << 'EOF'
  [Copier code OptunaTuner]
  EOF
  
  # Créer optimize_hyperparameters.py
  cat > optimize_hyperparameters.py << 'EOF'
  [Copier code main]
  EOF

ÉTAPE 3: Exécuter l'optimisation
────────────────────────────────

Option A: Petite optimisation (test)
  $ python optimize_hyperparameters.py --n-trials 20

Temps: ~5 minutes
Résultat: Vérifie que ça fonctionne

Option B: Optimisation complète (recommandé)
  $ python optimize_hyperparameters.py --n-trials 100

Temps: ~30 minutes
Résultat: Meilleurs hyperparamètres

Option C: Optimisation uniquement XGBoost
  $ python optimize_hyperparameters.py --n-trials 100 --models xgb

ÉTAPE 4: Vérifier les résultats
───────────────────────────────

$ cat models/best_params.json

Résultat attendu:
{
  "xgboost": {
    "params": {
      "n_estimators": 300,
      "max_depth": 6,
      "learning_rate": 0.05,
      ...
    },
    "best_f1": 0.628
  },
  "lightgbm": {
    "params": {...},
    "best_f1": 0.625
  }
}

ÉTAPE 5: Réentraîner avec meilleurs params
─────────────────────────────────────────

# Option 1: Utiliser les nouveaux params manuellement
import json
with open('models/best_params.json') as f:
    params = json.load(f)

# Option 2: Créer un script de ré-entraînement
# (À faire après étape 4)

═════════════════════════════════════════════════════════════════════════
""")

def print_results_expected():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 🎯 RÉSULTATS ATTENDUS: PHASE 4                                       ║
╚════════════════════════════════════════════════════════════════════════╝

AVANT OPTUNA (Baseline):
  XGBoost F1:  0.614
  XGBoost AUC: 0.969
  
APRÈS OPTUNA (Optimisé):
  XGBoost F1:  0.628  ← +2.3% amélioration ✅
  XGBoost AUC: 0.975  ← +0.6% amélioration ✅

FICHIERS CRÉÉS:
  ✅ src/optuna_tuning.py (classe OptunaTuner)
  ✅ optimize_hyperparameters.py (script principal)
  ✅ models/best_params.json (hyperparamètres optimisés)

PROCHAINE ÉTAPE:
  → Phase 5: SHAP + Fairness (demain)
    Vérifier l'explainability et les biais

═════════════════════════════════════════════════════════════════════════
""")

def main():
    print_current_state()
    print_quick_start()
    print_phase4_code()
    print_phase4_main()
    print_execution_steps()
    print_results_expected()
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║ ✅ PRÊT À COMMENCER?                                                 ║
╚════════════════════════════════════════════════════════════════════════╝

Trois options:

1️⃣  LIRE PUIS CODER
    $ python ROADMAP_FINAL_PHASES.py
    $ cat ACTION_PLAN_2WEEKS.md

2️⃣  COMMENCER PHASE 4 TOUT DE SUITE
    $ pip install optuna
    $ cp code du fichier dans src/optuna_tuning.py
    $ python optimize_hyperparameters.py --n-trials 100

3️⃣  DEMANDER DE L'AIDE
    Consultez: ROADMAP_FINAL_PHASES.py (détails complets)

═════════════════════════════════════════════════════════════════════════

🚀 Bon courage! Tu es à 40%, tu peux atteindre 100% en 2 semaines! 💪
    
Questions?
  - Relire le code template ci-dessus
  - Consulter ROADMAP_FINAL_PHASES.py
  - Voir ACTION_PLAN_2WEEKS.md

═════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    main()
''')
    
    print(code)

if __name__ == "__main__":
    main()
