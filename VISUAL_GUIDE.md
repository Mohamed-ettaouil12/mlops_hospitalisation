# 🎯 VISUAL GUIDE: Les 3 Techniques Expliquées Visuellement

## 📊 Technique 1: Stacking avec OOF Predictions

### Processus Visuel

```
PHASE 1: Générer les prédictions Out-of-Fold
═════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ DONNÉES TRAIN (1000 samples)                                    │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─ Fold 1: [Entraîner sur 2-5] → Prédire sur [1]
   │           XGBoost: [0.2, 0.9, 0.1, ...]
   │           LightGBM: [0.3, 0.85, 0.15, ...]
   │
   ├─ Fold 2: [Entraîner sur 1,3-5] → Prédire sur [2]
   │           XGBoost: [0.4, 0.7, ...]
   │           LightGBM: [0.35, 0.75, ...]
   │
   ├─ Fold 3: [Entraîner sur 1-2,4-5] → Prédire sur [3]
   │
   ├─ Fold 4: [Entraîner sur 1-3,5] → Prédire sur [4]
   │
   └─ Fold 5: [Entraîner sur 1-4] → Prédire sur [5]
   
RÉSULTAT: Prédictions OOF complètes
┌─────────────────────────────────────────────────────────────────┐
│ DataFrame OOF (1000 rows, 3 colonnes pour 3 modèles)            │
│                                                                 │
│ index │ XGBoost_OOF │ LightGBM_OOF │ RandomForest_OOF          │
│───────┼─────────────┼──────────────┼──────────────             │
│ 0     │ 0.82        │ 0.80         │ 0.78                      │
│ 1     │ 0.15        │ 0.18         │ 0.20                      │
│ ...   │ ...         │ ...          │ ...                       │
│ 999   │ 0.91        │ 0.89         │ 0.85                      │
└─────────────────────────────────────────────────────────────────┘
   │
   PHASE 2: Entraîner le méta-modèle
   │
   v
┌─────────────────────────────────────────────────────────────────┐
│ META-MODÈLE (LogisticRegression)                               │
│ Apprend: Comment combiner les 3 prédictions?                   │
│                                                                 │
│ Entrée: [0.82, 0.80, 0.78]                                    │
│ Sortie: 0.81 (moyenne pondérée)                               │
│                                                                 │
│ Weights appris:                                                │
│   XGBoost: 0.50                                                │
│   LightGBM: 0.35                                               │
│   RandomForest: 0.15                                           │
└─────────────────────────────────────────────────────────────────┘
   │
   PHASE 3: Réentraîner les modèles de base
   │
   v
┌─────────────────────────────────────────────────────────────────┐
│ MODÈLES DE BASE RÉENTRAÎNÉS (sur 100% train)                   │
│ ├─ XGBoost (nouveau)                                            │
│ ├─ LightGBM (nouveau)                                           │
│ └─ RandomForest (nouveau)                                       │
│                                                                 │
│ → Prêts pour l'inférence en production                         │
└─────────────────────────────────────────────────────────────────┘
```

### Résultats

```
Sur Test Set:

Modèles individuels:
  XGBoost:      Precision = 0.80
  LightGBM:     Precision = 0.79
  RandomForest: Precision = 0.78
  
Stacking:       Precision = 0.82 ✅ (+2.5%)
```

### Code Rapide

```python
stacking = StackingWithOOF(
    base_models={"XGBoost": xgb, "LightGBM": lgb, "RF": rf},
    meta_model_type="logistic",
    n_splits=5,
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

---

## 📊 Technique 2: Weighted Blending

### Processus Visuel

```
ÉTAPE 1: Générer les prédictions sur validation set
═════════════════════════════════════════════════════

┌──────────────────────────────────┐
│ X_val (200 samples)              │
└──────────────────────────────────┘
       │
       ├─ XGBoost:     [0.2, 0.9, 0.1, ...]
       ├─ LightGBM:    [0.3, 0.85, 0.15, ...]
       └─ RandomForest: [0.1, 0.92, 0.08, ...]


ÉTAPE 2: Optimiser les poids
════════════════════════════════

Objectif: Maximiser F1 score sur validation

┌─────────────────────────────┐
│ Poids initiaux (égaux):     │
│  w_xgb = 0.333              │
│  w_lgb = 0.333              │
│  w_rf  = 0.333              │
│ F1 = 0.750                  │
└─────────────────────────────┘
       │
       v (Optimisation)
       │
┌─────────────────────────────┐
│ Poids optimisés:            │
│  w_xgb = 0.55  ✅ MEILLEUR │
│  w_lgb = 0.35               │
│  w_rf  = 0.10               │
│ F1 = 0.765                  │
└─────────────────────────────┘


ÉTAPE 3: Prédire sur test
═════════════════════════════

Pour chaque sample:
  y_pred = 0.55*y_xgb + 0.35*y_lgb + 0.10*y_rf
  
Exemple:
  XGBoost:     0.82
  LightGBM:    0.80
  RandomForest: 0.78
  ─────────────────────
  Blended:     0.55*0.82 + 0.35*0.80 + 0.10*0.78 = 0.807
```

### Visualisation des Poids

```
Importance de chaque modèle:

XGBoost     ████████████░░░░ (55%)  ← Meilleur pour cette tâche
LightGBM    ███████░░░░░░░░░░ (35%)
RandomForest ██░░░░░░░░░░░░░░ (10%)

Total: 100%
```

### Résultats

```
Baseline (moyenne simple):    Precision = 0.79
Blending avec poids optimisés: Precision = 0.81 ✅ (+2%)
```

### Code Rapide

```python
blending = WeightedBlending(
    base_models={"XGBoost": xgb, "LightGBM": lgb},
    objective_metric="f1",
)

weights = blending.optimize_weights(X_val, y_val)
y_pred = blending.predict_proba(X_test)

# weights = {"XGBoost": 0.55, "LightGBM": 0.45}
```

---

## 📊 Technique 3: Pseudo-Labeling

### Processus Visuel

```
DONNÉES DISPONIBLES
═══════════════════

┌──────────────────────────┐
│ Données labellisées:     │
│ 1000 samples avec labels │
│ ├─ Classe 0: 700         │
│ └─ Classe 1: 300         │
└──────────────────────────┘

┌──────────────────────────┐
│ Données NON labellisées: │
│ 5000 samples SANS labels │
│ ├─ Label? (inconnu)      │
│ └─ Label? (inconnu)      │
└──────────────────────────┘


ÉTAPE 1: Prédire sur les données non labellisées
═════════════════════════════════════════════════

Modèle entraîné (XGBoost):
  
  Sample 1: Confiance = 0.98 (Classe 1) ✅ Haute confiance!
  Sample 2: Confiance = 0.52 (Classe 1) ❌ Basse confiance
  Sample 3: Confiance = 0.96 (Classe 0) ✅ Haute confiance!
  Sample 4: Confiance = 0.51 (Classe 0) ❌ Basse confiance
  ...


ÉTAPE 2: Filtrer par confiance (threshold = 0.95)
══════════════════════════════════════════════════

Prédictions:
  Sample 1: 0.98 → Garder ✅
  Sample 2: 0.52 → Rejeter ✗
  Sample 3: 0.96 → Garder ✅
  Sample 4: 0.51 → Rejeter ✗
  ...

Résultat: 3500 samples gardés (70% de couverture)
          avec pseudo-labels confiants


ÉTAPE 3: Combiner avec données labelisées
═════════════════════════════════════════

┌──────────────────────────────────┐
│ Données augmentées:              │
│ ├─ Labelisées:     1000 samples  │
│ └─ Pseudo-labelisées: 3500      │
│ ──────────────────────────────── │
│ Total:             4500 samples  │
│ Augmentation: +350%              │
└──────────────────────────────────┘


ÉTAPE 4: Réentraîner
═════════════════════

XGBoost_v2 = XGBoost.fit(X_combined, y_combined)
             ├─ X: 4500 samples (au lieu de 1000)
             ├─ y: 1000 labelisées + 3500 pseudo
             └─ Performance: Meilleure généralisation


RÉSULTAT:
════════

Avant pseudo-labeling:  Precision = 0.80
Après pseudo-labeling:  Precision = 0.81 ✅ (+1%)
```

### Distribution des Pseudo-Labels

```
Before vs After:

Before:
  Classe 0: ██████████░░░░░░░░░░░░░░░░ (30%)
  Classe 1: ████████████████░░░░░░░░░░░ (70%)
  Total: 1000

After:
  Classe 0: █████████░░░░░░░░░░░░░░░░░░ (35%)
  Classe 1: ██████████░░░░░░░░░░░░░░░░░ (65%)
  Total: 4500

✓ Données augmentées ET mieux équilibrées!
```

### Code Rapide

```python
pseudo_labeler = PseudoLabelingStrategy(
    model=best_model,
    confidence_threshold=0.95,
)

X_pseudo, y_pseudo, conf = pseudo_labeler.generate_pseudo_labels(X_unlabeled)

X_combined = pd.concat([X_train, X_pseudo])
y_combined = pd.concat([y_train, y_pseudo])

best_model.fit(X_combined, y_combined)
```

---

## 🏆 Comparaison Visuelle

### Gain en Précision

```
0.75 ├─
0.76 ├─
0.77 ├─ ■ RandomForest
0.78 ├─ ■ RF
0.79 ├─ ■ LightGBM
0.80 ├─ ■ Meilleur modèle seul (BASELINE)
0.81 ├─ ■ Blending ✅ (+1%)
0.81 ├─ ■ Pseudo-Labeling ✅ (+1%)
0.82 ├─ ■ Stacking OOF ✅✅ (+2%) 🏆
0.83 ├─
```

### Complexité vs Gain

```
Complexité (CPU/Memory)
    ↑
    │                    ┌─ Stacking OOF (gain: +2-3%)
    │                    │ (K-fold, complexe)
    │            
    │         ┌──────────┘
    │         │
    │     ┌───┤ Pseudo-Labeling (gain: +1-2%)
    │     │   │ (modéré)
    │     │   │
    │   ┌─┘   │
    │   │     │
    ├───┤─────┼─ Blending (gain: +1-2%)
    │   │     │ (simple, rapide)
    │   │     │
    0   └─────┴─────────────→ Gain en Precision
        Basique  Simple  Moyen  Expert
```

---

## 🎯 Matrice de Décision

### Quel modèle choisir?

```
┌─────────────────────────────────────────────────────────────┐
│ Scénario                    │ Technique recommandée         │
├─────────────────────────────────────────────────────────────┤
│ Je veux max gain            │ → STACKING OOF (+2-3%) 🏆     │
│ J'ai peu de temps           │ → BLENDING (5 min) ⚡         │
│ J'ai données non labellisées│ → PSEUDO-LABELING (+data)     │
│ Je veux transparence        │ → BLENDING (poids visibles)   │
│ J'ai GPU/compute            │ → STACKING (n_splits=10)      │
│ Je débute                   │ → BLENDING (plus simple)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Résumé Visuel

```
┌──────────────────────────────────────────────────────────────┐
│                    LES 3 TECHNIQUES                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1️⃣  STACKING OOF                                            │
│     ├─ Processus: K-fold CV → OOF → Meta-learner           │
│     ├─ Complexité: ████░░░░░░ (5/10)                       │
│     ├─ Gain: +2-3% 🏆                                      │
│     └─ Quand: Besoin de max gain                           │
│                                                              │
│  2️⃣  WEIGHTED BLENDING                                      │
│     ├─ Processus: Poids optimisés → Moyenne pondérée       │
│     ├─ Complexité: ██░░░░░░░░ (2/10)                       │
│     ├─ Gain: +1-2% ⚡                                      │
│     └─ Quand: Besoin de rapidité                           │
│                                                              │
│  3️⃣  PSEUDO-LABELING                                        │
│     ├─ Processus: Prédire → Filtrer → Augmenter données   │
│     ├─ Complexité: ███░░░░░░░ (3/10)                       │
│     ├─ Gain: +1-2% (+ données)                            │
│     └─ Quand: Données non labellisées dispo               │
│                                                              │
│  GAIN TOTAL ENSEMBLE: +3-5% 🚀                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Prochaines Étapes Visuelles

```
Jour 1 (Lundi)
└─ Lecture: README_ADVANCED.md
   └─ Compréhension: 10 min ✓

Jour 2 (Mardi)
├─ Exécution: python examples_advanced_techniques.py
│  └─ Résultat: Voir les techniques en action
│
└─ Exécution: python apply_advanced_ensemble.py --technique all
   └─ Résultat: Modèles sauvegardés

Jour 3 (Mercredi)
└─ MLflow: mlflow ui
   └─ Résultat: Dashboard avec résultats

Jour 4-5 (Jeudi-Vendredi)
└─ Production: Intégration en prod
   └─ Résultat: Modèle d'ensemble en production
```

---

Voilà! Maintenant vous comprenez visuellement comment chaque technique fonctionne! 🎯
