# 🚀 Guide Complet: Techniques Avancées pour Améliorer la Précision

## 📋 Table des Matières
1. [Stacking avec OOF Predictions](#stacking-avec-oof)
2. [Weighted Blending Optimisé](#weighted-blending)
3. [Pseudo-Labeling](#pseudo-labeling)
4. [Feature Engineering Croisé](#feature-engineering)
5. [Hyperparameter Tuning avec Optuna](#optuna)
6. [SHAP Analysis](#shap-analysis)
7. [Roadmap MLOps](#roadmap-mlops)

---

## 1. Stacking avec OOF Predictions {#stacking-avec-oof}

### 🎯 Concept
Le stacking avec Out-of-Fold (OOF) predictions est LA technique la plus robuste pour améliorer la précision.

**Avantage clé**: Évite le **data leakage** (fuite de données)

### 📊 Processus

```
┌─────────────────────────────────────────────────────────┐
│ 1. DONNÉES TRAIN                                        │
│    ├─ Fold 1, 2, 3, 4                    │ Fold 5      │
│    └─ Entraîner modèles sur folds 1-4   │ Prédire 5   │
│                                                         │
│ 2. RÉPÉTER pour chaque fold              │             │
│    └─ Chaque fold génère des prédictions│             │
│       out-of-fold (OOF)                 │             │
│                                                         │
│ 3. RÉSULTAT: Prédictions OOF complètes  │             │
│    ├─ Une pour chaque base model         │             │
│    └─ Format: (n_samples, n_models)     │             │
│                                                         │
│ 4. MÉTA-MODÈLE                          │             │
│    └─ Entraîner un simple modèle sur    │             │
│       les prédictions OOF                │             │
│                                                         │
│ 5. INFÉRENCE FINALE                     │             │
│    └─ Réentraîner les base models sur   │             │
│       l'ensemble train complet           │             │
└─────────────────────────────────────────────────────────┘
```

### 💻 Utilisation

```python
from src.advanced_ensemble_techniques import StackingWithOOF
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Définir les modèles de base
base_models = {
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Créer le stacking
stacking = StackingWithOOF(
    base_models=base_models,
    meta_model_type="logistic",  # ou "ridge"
    n_splits=5,  # 5-fold ou 10-fold
    random_state=42,
)

# Entraîner
stacking.fit(X_train, y_train, verbose=True)

# Prédire
y_proba = stacking.predict_proba(X_test)
y_pred = stacking.predict(X_test, threshold=0.5)
```

### 🔍 Quand l'utiliser
- ✅ Besoin d'amélioration de précision (+1 à +3%)
- ✅ Avez 3+ modèles hétérogènes
- ✅ Assez de données (K-fold CV possible)
- ⚠️ Coût computationnel modéré (K * n_models d'entraînements)

### 📈 Résultats attendus
- **Précision**: +1 à +3% vs meilleur modèle seul
- **Stabilité**: Meilleure généralisation
- **Robustesse**: Moins sensible aux fluctuations

---

## 2. Weighted Blending Optimisé {#weighted-blending}

### 🎯 Concept
Trouver les **poids optimaux** pour combiner les prédictions.

Formule:
```
y_proba_final = w1 * y_proba_model1 + w2 * y_proba_model2 + ... 
                où sum(w) = 1 et w >= 0
```

### 💻 Utilisation

```python
from src.advanced_ensemble_techniques import WeightedBlending

# Créer le blending
blending = WeightedBlending(
    base_models={
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
    },
    objective_metric="f1",  # ou "precision", "recall", "auc"
)

# Optimiser les poids sur validation
weights = blending.optimize_weights(X_val, y_val, verbose=True)

# Résultat:
# {
#     "XGBoost": 0.6,
#     "LightGBM": 0.4,
# }

# Prédire
y_proba = blending.predict_proba(X_test)
```

### 🔍 Quand l'utiliser
- ✅ Solution **simple et rapide**
- ✅ Déjà 2-3 modèles entraînés
- ✅ Besoin de transparence (poids visibles)
- ✅ Peu de données (pas besoin de K-fold)

### 🎛️ Optimisation des poids
Les poids sont optimisés via **scipy.optimize.minimize** en cherchant les poids qui **maximisent** votre métrique cible:

```python
# Exemple de résultat
weights = {
    "XGBoost": 0.55,    # Model fort
    "LightGBM": 0.45,   # Model bon
}

# Interprétation: XGBoost plus fiable +5% en confiance
```

---

## 3. Pseudo-Labeling {#pseudo-labeling}

### 🎯 Concept
Utiliser les modèles entraînés pour générer des **labels confiants** sur les données non labellisées.

### 📊 Processus

```
┌────────────────────────────────────────────────────┐
│ DONNÉES NON LABELLISÉES (test data, new data)    │
│                                                    │
│ 1. Prédictions du modèle                         │
│    └─ XGBoost → [0.92, 0.08, 0.97, ...]         │
│                                                    │
│ 2. Filtrer par confiance (>0.95)                 │
│    └─ Garder: 0.92 (non), 0.97 (OUI)            │
│                                                    │
│ 3. Générer pseudo-labels                         │
│    └─ Samples confiants: label = argmax(proba)   │
│                                                    │
│ 4. Combiner avec données labellisées             │
│    └─ X_combined = [X_labeled + X_pseudo]        │
│       y_combined = [y_labeled + y_pseudo]        │
│                                                    │
│ 5. Réentraîner les modèles                       │
│    └─ Sur l'ensemble augmenté (⬆️ données!)      │
└────────────────────────────────────────────────────┘
```

### 💻 Utilisation

```python
from src.advanced_ensemble_techniques import PseudoLabelingStrategy

# Votre modèle entraîné
best_model = xgb_model

# Créer la stratégie
pseudo_labeler = PseudoLabelingStrategy(
    model=best_model,
    confidence_threshold=0.95,  # 95% confiance minimum
)

# Générer les pseudo-labels
X_pseudo, y_pseudo, confidence = pseudo_labeler.generate_pseudo_labels(
    X_unlabeled,
    verbose=True,
)

# Combiner avec données labelisées
X_combined = pd.concat([X_train, X_pseudo])
y_combined = pd.concat([y_train, y_pseudo])

# Réentraîner sur l'ensemble augmenté
best_model.fit(X_combined, y_combined)
```

### 🎛️ Hyperparamètres
- **confidence_threshold**: 0.90 à 0.99
  - 0.90 = permissif (+ samples mais moins fiables)
  - 0.99 = strict (- samples mais très fiables)

### 🚨 Pièges à éviter
- ❌ Threshold trop bas → Erreurs accumulées
- ❌ Samples déséquilibrés → Bias dans pseudo-labels
- ❌ Trop de pseudo-labels → Overfitting

### ✅ Bonnes pratiques
- Valider les pseudo-labels sur un small labeled set
- Utiliser une **confiance minimale élevée** (0.95+)
- Monitorer la distribution des classes dans les pseudo-labels

---

## 4. Feature Engineering Croisé {#feature-engineering}

### 🎯 Concept
Créer des **features d'interaction** entre les prédictions des modèles.

### 📊 Exemple

Si vous avez 2 modèles (XGBoost et LightGBM):

```
Features créées:
├─ xgboost_pred                    # Prédictions brutes
├─ lightgbm_pred
├─ xgb_x_lgb_prod                 # Produit
├─ xgb_x_lgb_mean                 # Moyenne
├─ xgb_x_lgb_max                  # Maximum
├─ xgb_x_lgb_diff                 # Différence
├─ pred_mean                      # Moyenne de tous
├─ pred_std                       # Écart-type
├─ pred_max                       # Max de tous
└─ pred_min                       # Min de tous

Total: 10 features pour le méta-modèle
```

### 💻 Utilisation

```python
from src.advanced_ensemble_techniques import create_cross_features

# Générer les prédictions des modèles
predictions = {}
for model_name, model in models.items():
    predictions[model_name] = model.predict_proba(X_test)[:, 1]

# Créer les features d'interaction
cross_features = create_cross_features(
    predictions_dict=predictions,
    feature_importance_dict={},  # Optionnel: importances des features
    X=X_test,
    verbose=True,
)

# Utiliser pour un nouveau modèle
meta_model.fit(cross_features, y_test)
```

### 🎯 Avantages
- Capture les **interactions** entre modèles
- Permet au méta-modèle d'apprendre les **patterns**
- Améliore la **robustesse** du stacking

---

## 5. Hyperparameter Tuning avec Optuna {#optuna}

### 🎯 Concept
Optimiser les hyperparamètres des modèles de base **individuellement** avant le stacking.

### 💻 Utilisation

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Fonction objective pour Optuna"""
    
    # Hyperparamètres à optimiser
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    
    # Score via cross-validation
    score = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="f1"
    ).mean()
    
    return score

# Lancer l'optimisation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Meilleurs hyperparamètres
print(study.best_params)
# {
#     'n_estimators': 250,
#     'max_depth': 6,
#     'learning_rate': 0.05,
#     'subsample': 0.8,
#     'colsample_bytree': 0.85,
# }
```

### 🎛️ Hyperparamètres clés par modèle

**XGBoost**:
```
n_estimators: 100-500
max_depth: 3-10
learning_rate: 0.001-0.3
subsample: 0.5-1.0
```

**LightGBM**:
```
num_leaves: 20-100
max_depth: 5-15
learning_rate: 0.001-0.3
feature_fraction: 0.7-1.0
```

**CatBoost**:
```
iterations: 100-500
depth: 4-10
learning_rate: 0.001-0.3
```

### 🔍 Configuration recommandée
- **n_trials**: 100 pour bonne exploration
- **cv_folds**: 5 ou 10 selon taille données
- **metric**: "f1" ou "precision" selon objectif

---

## 6. SHAP Analysis {#shap-analysis}

### 🎯 Concept
**SHAP (SHapley Additive exPlanations)** explique les prédictions et identifie les **features bruitées**.

### 💻 Installation
```bash
pip install shap
```

### 💻 Utilisation

```python
import shap
import xgboost as xgb

# Créer l'explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculer les SHAP values
shap_values = explainer.shap_values(X_test)

# Visualiser l'importance globale
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Visualiser l'impact d'une feature
shap.summary_plot(shap_values, X_test)

# Dépendance d'une feature
shap.dependence_plot("feature_name", shap_values, X_test)
```

### 🔍 Utilisation pour le Feature Selection

```python
# Calculer l'importance moyenne
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Garder les top features
important_features = X_test.columns[mean_abs_shap > threshold]

# Supprimer les features bruitées
X_cleaned = X_test[important_features]
```

### 📊 Interprétation
- **SHAP value positif**: Feature ↑ → Prédiction ↑
- **SHAP value négatif**: Feature ↑ → Prédiction ↓
- **Magnitude**: Importance de l'impact

---

## 7. Roadmap MLOps {#roadmap-mlops}

### 📋 Checklist Optimisation

#### Phase 1: Modèles Individuels (Semaine 1)
```
☐ Entraîner XGBoost, LightGBM, CatBoost
☐ Optimiser hyperparamètres avec Optuna
☐ Analyser feature importance (SHAP)
☐ Éliminer les features bruitées
☐ Logger dans MLflow
```

#### Phase 2: Ensemble (Semaine 2)
```
☐ Implémenter Stacking avec OOF
  └─ n_splits=5, meta_model="logistic"
☐ Essayer Weighted Blending
☐ Comparer les métriques
☐ Logger les poids et OOF predictions
```

#### Phase 3: Améliorations Supplémentaires (Semaine 3)
```
☐ Pseudo-labeling si données non labellisées dispo
☐ Feature engineering croisé
☐ Threshold optimization pour precision/recall
☐ Monitoring du drift en production
```

#### Phase 4: Production (Semaine 4)
```
☐ Sauvegarder le meilleur ensemble
☐ Documenter les versions
☐ Setup MLflow server
☐ Mettre en place le monitoring
☐ CI/CD pour réentraînement automatique
```

---

## 8. Ordre d'Exécution Recommandé

### Commandes MLOps

```bash
# 1. Entraîner les modèles individuels
python src/train.py --experiment-name baseline

# 2. Appliquer les techniques avancées
python apply_advanced_ensemble.py --technique stacking --n-splits 10
python apply_advanced_ensemble.py --technique blending

# 3. Analyser les résultats dans MLflow
mlflow ui  # http://localhost:5000

# 4. Comparer les performances
python scripts/compare_models.py
```

---

## 9. Gains Attendus par Technique

| Technique | Gain Precision | Complexité | Temps Entraînement | Recommandé |
|-----------|---|---|----|---|
| **Baseline** | 0% | 1/5 | 1x | Départ |
| **Hyperparameter Tuning** | +1-2% | 2/5 | 5-10x | ✅ |
| **Weighted Blending** | +1-2% | 1/5 | 1x | ✅ |
| **Stacking OOF** | +2-3% | 3/5 | K x (1x) | ✅✅ |
| **Ensemble Complet** | +3-5% | 5/5 | 10x | ✅✅✅ |

---

## 10. Troubleshooting

### ❌ Problème: Stacking n'améliore pas
```
Causes possibles:
✓ Modèles de base trop corrélés
  → Utiliser des architectures différentes
  
✓ Pas assez de données pour K-fold
  → Réduire n_splits (5 au lieu de 10)
  
✓ Méta-modèle overfit
  → Utiliser Ridge au lieu de Logistic Regression
  → Augmenter la régularisation
```

### ❌ Problème: Blending donne poids égaux
```
Causes possibles:
✓ Modèles trop similaires en performa
  → Vérifier que base_models sont vraiment différents
  
✓ Metric_objective inadapté
  → Essayer "precision" au lieu de "f1"
  
✓ Optimisation ne converge pas
  → Augmenter maxiter (100 → 500)
```

### ❌ Problème: Pseudo-labeling = faible coverage
```
Causes possibles:
✓ Threshold trop élevé (0.99)
  → Réduire à 0.95 ou 0.90
  
✓ Modèles pas assez confiants
  → Entraîner plus (n_estimators ↑)
  → Augmenter n_epochs
```

---

## 🎯 Résumé des Meilleurs Pratiques

### ✅ À FAIRE
1. **Stacking avec OOF** pour +2-3% gain
2. **Modèles de base hétérogènes** (XGB, LGB, RF)
3. **Hyperparameter tuning** avant stacking
4. **Logger tout dans MLflow** pour tracking
5. **Valider sur holdout set** séparé
6. **Monitorer le drift** en production

### ❌ À ÉVITER
1. ❌ Modèles trop similaires dans l'ensemble
2. ❌ Stacking sur données sans assez de samples
3. ❌ Pseudo-labeling sans validation
4. ❌ Features trop corrélées
5. ❌ Ne pas normaliser les inputs du méta-modèle
6. ❌ Ignorer l'imbalance des classes

---

## 📚 Ressources

- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [Stacking Kaggle Guide](https://kaggle.com/willkoehrsen/stacked-ensembles)
- [SHAP Documentation](https://github.com/slundberg/shap)
- [Optuna Documentation](https://optuna.org/)

---

**Créé le**: 2024  
**Version**: 1.0  
**Auteur**: MLOps Team
