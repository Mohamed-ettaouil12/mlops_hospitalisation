# Requirements Document

## Introduction

Ce document spécifie les exigences pour finaliser le projet MLOps de prédiction du risque d'hospitalisation Medicare (CMS DE-SynPUF). Le projet dispose déjà d'un pipeline d'entraînement complet (XGBoost/LightGBM/CatBoost, Optuna, MLflow), d'une API FastAPI et d'artefacts de modèles entraînés. L'objectif est d'ajouter les tests unitaires robustes, une suite de CI/CD GitHub Actions, et de garantir que le pipeline est déployable de bout en bout.

## Glossary

- **Pipeline** : Ensemble des étapes d'entraînement ML (prétraitement → feature engineering → entraînement → évaluation → sauvegarde)
- **API** : L'application FastAPI dans `src/api.py` qui expose les endpoints `/health`, `/predict`, `/model/info`
- **CI** : Continuous Integration — exécution automatique des tests à chaque push ou pull request
- **CD** : Continuous Deployment/Delivery — déploiement ou packaging automatique après validation CI
- **Test_Suite** : L'ensemble des fichiers de tests dans le répertoire `tests/`
- **Feature_Names** : La liste canonique des 28 features attendues par le modèle, stockée dans `data/features/feature_names.csv`
- **Threshold** : Le seuil de classification binaire stocké dans `models/threshold.json`
- **Ensemble** : La classe `ProbabilityAveragingEnsemble` dans `src/ensemble_models.py`
- **Scaler** : Le `StandardScaler` sklearn sauvegardé dans `models/scaler.pkl`
- **MLflow** : Le système de tracking des expériences ML, servi localement
- **Workflow_CI** : Le fichier `.github/workflows/ci.yml` déclenché sur push/PR
- **Workflow_Train** : Le fichier `.github/workflows/train.yml` déclenché manuellement ou sur schedule
- **PatientData** : Le schéma Pydantic défini dans `src/api.py` représentant les données d'entrée d'un patient
- **PredictionResult** : Le schéma Pydantic de sortie contenant `probabilite`, `risque`, `seuil_utilise`, `message`

---

## Requirements

### Requirement 1: Tests unitaires du module de données

**User Story:** En tant que développeur ML, je veux des tests unitaires couvrant les modules de prétraitement et de feature engineering, afin de garantir que les transformations de données sont correctes et reproductibles.

#### Acceptance Criteria

1. WHEN `feature_engineering.py` calcule le `CHARLSON_INDEX` pour un patient, THE Test_Suite SHALL vérifier que le score est égal à la somme pondérée des comorbidités selon les poids définis dans `CHARLSON_WEIGHTS`
2. WHEN `feature_engineering.py` calcule `NB_COMORBIDITES` pour un patient, THE Test_Suite SHALL vérifier que la valeur est égale au nombre de colonnes `SP_*` dont la valeur est 1
3. WHEN `data_preprocessing.py` applique `safe_numeric` à une série, THE Test_Suite SHALL vérifier que toutes les valeurs infinies et NaN sont remplacées par la valeur `fill_value` spécifiée, quelle que soit la composition initiale de la série
4. WHEN `data_preprocessing.py` génère les splits train/val/test, THE Test_Suite SHALL vérifier que les ensembles sont disjoints (aucun patient ne se retrouve dans deux ensembles à la fois)
5. IF `data_preprocessing.py` reçoit un fichier d'entrée manquant, THEN THE Test_Suite SHALL vérifier qu'une `FileNotFoundError` est levée avec un message incluant le chemin du fichier

---

### Requirement 2: Tests unitaires de l'Ensemble

**User Story:** En tant que développeur ML, je veux des tests unitaires pour la classe `ProbabilityAveragingEnsemble`, afin de garantir que les prédictions d'ensemble sont correctement calculées.

#### Acceptance Criteria

1. WHEN `ProbabilityAveragingEnsemble.predict_proba` est appelée avec N estimateurs, THE Test_Suite SHALL vérifier que les probabilités de sortie sont normalisées (chaque ligne somme à 1.0, à ±1e-6 près)
2. WHEN `ProbabilityAveragingEnsemble.predict_proba` est appelée avec des poids valides, THE Test_Suite SHALL vérifier que la moyenne pondérée des probabilités de chaque estimateur est correctement calculée
3. IF `ProbabilityAveragingEnsemble` est initialisée sans estimateurs, THEN THE Test_Suite SHALL vérifier qu'une `ValueError` est levée lors de l'appel à `predict_proba` et que `predict_proba` a bien été invoquée
4. IF `ProbabilityAveragingEnsemble` est initialisée avec des poids de longueur différente du nombre d'estimateurs, THEN THE Test_Suite SHALL vérifier qu'une `ValueError` est levée
5. IF `ProbabilityAveragingEnsemble` est initialisée avec des poids tous négatifs, THEN THE Test_Suite SHALL vérifier qu'une `ValueError` est levée
6. WHEN `ProbabilityAveragingEnsemble.predict` est appelé, THE Test_Suite SHALL vérifier que la classe prédite correspond à l'argmax des probabilités retournées par `predict_proba`

---

### Requirement 3: Tests unitaires de l'API FastAPI

**User Story:** En tant que développeur, je veux des tests unitaires pour tous les endpoints de l'API FastAPI, afin de garantir que l'API répond correctement à des entrées valides et invalides sans accéder aux fichiers modèles réels.

#### Acceptance Criteria

1. WHEN `/health` est appelé avec la méthode GET, THE Test_Suite SHALL vérifier que la réponse a le statut HTTP 200 et contient les clés `status`, `modele`, `features`, `threshold`, `version`
2. WHEN `/predict` est appelé avec un `PatientData` valide contenant les 28 features, THE Test_Suite SHALL vérifier que la réponse contient `probabilite` dans [0.0, 1.0], `risque` dans ["FAIBLE", "MODÉRÉ", "ÉLEVÉ"], `seuil_utilise` > 0, et `message` non vide
3. IF `/predict` est appelé avec un champ `AGE` hors de la plage [0, 120], THEN THE Test_Suite SHALL vérifier que la réponse a le statut HTTP 422 (validation error Pydantic)
4. WHEN `/model/info` est appelé avec la méthode GET, THE Test_Suite SHALL vérifier que la réponse contient `nb_features` égal à 28, `features` étant une liste de 28 chaînes, et `threshold` un float dans (0, 1)
5. WHEN `/predict` est appelé avec `probabilite` supérieure ou égale au `seuil_utilise`, THE Test_Suite SHALL vérifier que `risque` est "ÉLEVÉ"
6. WHEN `/predict` est appelé avec `probabilite` inférieure à `seuil_utilise * 0.5`, THE Test_Suite SHALL vérifier que `risque` est "FAIBLE"

---

### Requirement 4: Tests unitaires de l'Optuna Tuner

**User Story:** En tant que data scientist, je veux des tests unitaires pour le module de tuning Optuna, afin de garantir que les objectifs d'optimisation retournent des scores valides.

#### Acceptance Criteria

1. WHEN `OptunaTuner.objective_xgb` est appelée avec un trial Optuna, THE Test_Suite SHALL vérifier que la valeur retournée est un float dans [0.0, 1.0]
2. WHEN `OptunaTuner.objective_lgb` est appelée avec un trial Optuna, THE Test_Suite SHALL vérifier que la valeur retournée est un float dans [0.0, 1.0]
3. WHEN `OptunaTuner.optimize_xgb` est appelée avec `n_trials=2`, THE Test_Suite SHALL vérifier que la fonction retourne un tuple `(best_params, best_value)` avec `best_params` un dict non vide et `best_value` un float dans [0.0, 1.0]

---

### Requirement 5: Tests de validation du pipeline end-to-end

**User Story:** En tant qu'ingénieur MLOps, je veux des tests de validation qui vérifient la cohérence du pipeline de bout en bout (artefacts → API → prédiction), afin de garantir que les modèles sauvegardés sont compatibles avec l'API.

#### Acceptance Criteria

1. WHEN les fichiers `models/best_model.pkl`, `models/scaler.pkl` et `data/features/feature_names.csv` existent, THE Test_Suite SHALL vérifier que le modèle peut être chargé, que le scaler a le même nombre de features que `feature_names.csv`, et que `predict_proba` retourne une sortie de shape `(N, 2)`
2. WHEN `models/threshold.json` existe, THE Test_Suite SHALL vérifier que le fichier est un JSON valide contenant la clé `threshold` avec une valeur float dans (0, 1)
3. WHEN `models/best_model_info.json` existe, THE Test_Suite SHALL vérifier que le fichier contient les clés `best_model`, `threshold`, `validation_metrics`, `test_metrics`, et `n_features`, et que `n_features` est exactement 28
4. WHEN `data/features/feature_names.csv` est chargé, THE Test_Suite SHALL vérifier que le fichier contient exactement 28 noms de features et que ceux-ci correspondent à la liste définie dans `best_model_info.json`

---

### Requirement 6: Workflow CI GitHub Actions (tests automatiques)

**User Story:** En tant qu'ingénieur MLOps, je veux un workflow GitHub Actions qui exécute automatiquement tous les tests unitaires à chaque push ou pull request sur la branche `main`, afin de détecter les régressions rapidement.

#### Acceptance Criteria

1. WHEN un push ou une pull request est créé sur la branche `main`, THE Workflow_CI SHALL s'exécuter automatiquement sur `ubuntu-latest` avec Python 3.10
2. WHEN THE Workflow_CI s'exécute, THE Workflow_CI SHALL installer toutes les dépendances depuis `requirements.txt` et les dépendances de test (`pytest`, `pytest-cov`, `httpx`)
3. WHEN THE Workflow_CI exécute les tests, THE Workflow_CI SHALL lancer `pytest tests/ --cov=src --cov-report=xml` et échouer si un test unitaire échoue
4. WHEN THE Workflow_CI exécute les tests, THE Workflow_CI SHALL générer un rapport de couverture de code et le stocker comme artifact GitHub
5. WHEN THE Workflow_CI échoue, THE Workflow_CI SHALL retourner un exit code non-zéro afin de bloquer les merges sur la branche protégée
6. WHEN les tests passent avec succès, THE Workflow_CI SHALL valider le build et afficher le badge de statut

---

### Requirement 7: Workflow Train GitHub Actions (réentraînement)

**User Story:** En tant qu'ingénieur MLOps, je veux un workflow GitHub Actions de réentraînement qui peut être déclenché manuellement ou sur schedule, afin de mettre à jour le modèle avec de nouvelles données sans intervention manuelle.

#### Acceptance Criteria

1. WHEN le Workflow_Train est déclenché via `workflow_dispatch` ou par un schedule hebdomadaire (lundi 2h UTC), THE Workflow_Train SHALL exécuter le script `src/train.py` dans un environnement Python 3.10 propre
2. WHEN THE Workflow_Train s'exécute avec succès, THE Workflow_Train SHALL stocker les artefacts de modèles (`models/*.pkl`, `models/*.json`) comme artifacts GitHub avec une rétention de 30 jours
3. WHEN THE Workflow_Train s'exécute, THE Workflow_Train SHALL enregistrer les métriques de performance du modèle (AUC-ROC, F1, précision, rappel) dans le résumé du job GitHub Actions
4. IF THE Workflow_Train échoue spécifiquement pendant l'exécution de `src/train.py`, THEN THE Workflow_Train SHALL conserver les artefacts du modèle précédent et retourner un exit code non-zéro, même si la conservation des artefacts échoue

---

### Requirement 8: Fixtures et données de test synthétiques

**User Story:** En tant que développeur, je veux des fixtures pytest réutilisables qui fournissent des données synthétiques cohérentes pour tous les tests, afin d'éviter les dépendances aux fichiers de données réels lors des tests en CI.

#### Acceptance Criteria

1. THE Test_Suite SHALL fournir une fixture `sample_patient_dict` retournant un dictionnaire valide avec les 28 features attendues par le modèle, avec des valeurs dans les plages définies par les contraintes Pydantic
2. THE Test_Suite SHALL fournir une fixture `sample_dataframe` retournant un `pd.DataFrame` de 100 lignes avec les 28 colonnes de features, avec des valeurs numériques valides
3. THE Test_Suite SHALL fournir une fixture `mock_model` retournant un modèle sklearn mock (`DummyClassifier`) avec les méthodes `predict_proba` et `predict`
4. THE Test_Suite SHALL fournir une fixture `mock_scaler` retournant un `StandardScaler` fitté sur des données synthétiques compatibles avec les features scalées de l'API
5. WHEN les fixtures sont utilisées dans les tests API, THE Test_Suite SHALL mocker les appels `joblib.load` et `pd.read_csv` pour que les tests ne dépendent pas des fichiers `models/` et `data/` réels
