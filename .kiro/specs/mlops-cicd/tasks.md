# Implementation Plan: MLOps CI/CD & Tests Unitaires

## Overview

Implémenter la suite de tests complète (`tests/`) et les workflows GitHub Actions (`.github/workflows/`) pour finaliser le projet MLOps de prédiction du risque d'hospitalisation Medicare. Les tests utilisent pytest + Hypothesis (property-based testing). Le CI exécute les tests automatiquement sur chaque push/PR. Le workflow Train automatise le réentraînement.

---

## Tasks

- [x] 1. Mettre en place l'infrastructure de test
  - Créer le répertoire `tests/` avec un fichier `tests/__init__.py` vide
  - Ajouter les dépendances de test dans `requirements.txt` : `pytest>=7.0`, `pytest-cov>=4.0`, `httpx>=0.24`, `pytest-mock>=3.10`, `hypothesis>=6.80`
  - Créer `pytest.ini` (ou section `[tool.pytest.ini_options]` dans `pyproject.toml`) avec : `testpaths = ["tests"]`, `python_files = "test_*.py"`, `python_functions = "test_*"`, `addopts = "-v --tb=short"`
  - _Requirements: 6.2, 8.1_

- [ ] 2. Créer les fixtures partagées dans `tests/conftest.py`
  - [ ] 2.1 Implémenter la fixture `sample_patient_dict` retournant un dict valide avec les 28 features (SP_* à 0/1, AGE entre 0-120, GROUPE_AGE_ENC entre 0-3, RACE_ENC entre 0-5, valeurs numériques dans leurs plages Pydantic)
    - _Requirements: 8.1_
  - [ ] 2.2 Implémenter la fixture `sample_dataframe` retournant un `pd.DataFrame` de 100 lignes × 28 colonnes avec les mêmes contraintes de valeurs
    - _Requirements: 8.2_
  - [ ] 2.3 Implémenter la fixture `mock_model` retournant un `DummyClassifier(strategy="uniform")` fitté sur des données synthétiques binaires (100 lignes × 28 colonnes)
    - _Requirements: 8.3_
  - [ ] 2.4 Implémenter la fixture `mock_scaler` retournant un `StandardScaler` fitté sur 100 lignes de données synthétiques pour les 11 colonnes numériques scalées par l'API (`AGE`, `COUT_TOTAL`, `CHARLSON_INDEX`, etc.)
    - _Requirements: 8.4_
  - [ ] 2.5 Implémenter la fixture `mock_app` qui patche `src.api.joblib.load` (retourne mock_model puis mock_scaler) et `src.api.pd.read_csv` (retourne DataFrame avec les 28 FEATURE_NAMES), recharge le module `src.api` via `importlib.reload`, et retourne un `TestClient(app)`
    - _Requirements: 8.5, 3.1_

- [ ] 3. Checkpoint — Vérifier l'infrastructure de test
  - Lancer `pytest tests/ --collect-only` pour vérifier que pytest découvre les fichiers et fixtures sans erreur.

- [ ] 4. Écrire les tests du module de données dans `tests/test_data.py`
  - [ ] 4.1 Importer les fonctions de `src/feature_engineering.py` (`CHARLSON_WEIGHTS`, `CHRONIC_DISEASE_COLS`, et la logique de calcul) et `src/data_preprocessing.py` (`safe_numeric`, `require_file`)
    - Identifier les fonctions pures testables (CHARLSON_INDEX, NB_COMORBIDITES, safe_numeric) et les extraire si nécessaires comme fonctions standalone
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ]* 4.2 Écrire le test de propriété pour CHARLSON_INDEX
    - **Property 1 : CHARLSON_INDEX est une somme pondérée exacte**
    - Utiliser `@given(st.lists(st.integers(0, 1), min_size=11, max_size=11))` pour générer des vecteurs binaires SP_*
    - Construire une `pd.Series` avec les noms CHARLSON_WEIGHTS et vérifier que le score calculé == `sum(w * v for col, w in CHARLSON_WEIGHTS.items())`
    - Tag : `# Feature: mlops-cicd, Property 1: CHARLSON_INDEX est une somme pondérée exacte`
    - **Validates: Requirements 1.1**
  - [ ]* 4.3 Écrire le test de propriété pour NB_COMORBIDITES
    - **Property 2 : NB_COMORBIDITES est un comptage exact de colonnes actives**
    - Utiliser `@given(st.lists(st.integers(0, 1), min_size=11, max_size=11))` pour les colonnes SP_*
    - Vérifier que NB_COMORBIDITES == `sum(vecteur)`
    - Tag : `# Feature: mlops-cicd, Property 2: NB_COMORBIDITES est un comptage exact`
    - **Validates: Requirements 1.2**
  - [ ]* 4.4 Écrire le test de propriété pour `safe_numeric`
    - **Property 3 : safe_numeric élimine toutes les valeurs invalides**
    - Utiliser `@given(st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1))` pour générer des séries avec NaN/inf
    - Vérifier qu'après `safe_numeric`, la série ne contient ni NaN ni inf
    - Tag : `# Feature: mlops-cicd, Property 3: safe_numeric élimine toutes les valeurs invalides`
    - **Validates: Requirements 1.3**
  - [ ]* 4.5 Écrire le test de propriété pour la disjointure des splits
    - **Property 4 : Les splits train/val/test sont strictement disjoints**
    - Utiliser `@given(st.integers(min_value=50, max_value=500))` pour le nombre de patients synthétiques
    - Créer un DataFrame avec des DESYNPUF_ID uniques, appliquer la logique de split, vérifier que train ∩ val, train ∩ test, val ∩ test sont vides
    - Tag : `# Feature: mlops-cicd, Property 4: Les splits sont strictement disjoints`
    - **Validates: Requirements 1.4**
  - [ ]* 4.6 Écrire le test exemple pour `require_file` avec fichier manquant
    - Appeler `require_file(Path("/chemin/inexistant/fichier.parquet"))`, vérifier `FileNotFoundError` avec le chemin dans le message
    - **Validates: Requirements 1.5**

- [ ] 5. Écrire les tests de l'Ensemble dans `tests/test_ensemble.py`
  - [ ] 5.1 Créer des helpers locaux : `make_mock_estimator(proba_matrix)` retournant un objet avec `predict_proba` retournant une matrice fixe (utilisé dans tous les tests de l'Ensemble)
    - _Requirements: 2.1_
  - [ ]* 5.2 Écrire le test de propriété pour la normalisation de `predict_proba`
    - **Property 5 : predict_proba produit des probabilités normalisées**
    - Utiliser `@given(st.integers(1, 5), st.integers(1, 20))` pour N estimateurs et M samples
    - Générer des matrices de probabilités aléatoires (normalisées par construction), créer l'ensemble, vérifier que chaque ligne de la sortie somme à 1.0 ± 1e-6
    - Tag : `# Feature: mlops-cicd, Property 5: predict_proba produit des probabilités normalisées`
    - **Validates: Requirements 2.1, 2.2**
  - [ ]* 5.3 Écrire le test de propriété pour la cohérence predict/predict_proba
    - **Property 6 : predict est cohérent avec predict_proba**
    - Utiliser `@given(st.integers(1, 4), st.integers(1, 30))` pour N estimateurs et M samples
    - Vérifier que `predict(X)[i] == classes_[argmax(predict_proba(X)[i])]` pour tout i
    - Tag : `# Feature: mlops-cicd, Property 6: predict est cohérent avec predict_proba`
    - **Validates: Requirements 2.6**
  - [ ]* 5.4 Écrire les tests exemples pour les cas d'erreur de l'Ensemble
    - Test 2.3 : `ProbabilityAveragingEnsemble(estimators=[])` → vérifier `ValueError` lors de l'appel `predict_proba(X)` et que la méthode a bien été invoquée
    - Test 2.4 : 2 estimateurs + 3 poids → vérifier `ValueError`
    - Test 2.5 : poids tous négatifs `[-1.0, -2.0]` → vérifier `ValueError`
    - **Validates: Requirements 2.3, 2.4, 2.5**

- [ ] 6. Checkpoint — Vérifier les tests données et ensemble
  - Lancer `pytest tests/test_data.py tests/test_ensemble.py -v` et vérifier que tous les tests passent.

- [ ] 7. Écrire les tests de l'API FastAPI dans `tests/test_api.py`
  - [ ] 7.1 Écrire les tests exemples pour les endpoints structurels
    - Test 3.1 (`/health`) : Appeler `GET /health` via `mock_app`, vérifier HTTP 200 et présence des clés `status`, `modele`, `features`, `threshold`, `version`
    - Test 3.4 (`/model/info`) : Appeler `GET /model/info`, vérifier `nb_features==28`, `features` liste de 28 strings, `threshold` float dans (0, 1)
    - Test 3.3 (AGE invalide) : Appeler `POST /predict` avec `AGE=200`, vérifier HTTP 422
    - **Validates: Requirements 3.1, 3.3, 3.4**
  - [ ]* 7.2 Écrire le test de propriété pour la classification du risque
    - **Property 7 : La classification du risque est déterministe par rapport au seuil**
    - Utiliser `@given(st.floats(0.0, 1.0), st.floats(0.01, 0.99))` pour (probabilite, seuil)
    - Configurer le mock model pour retourner `probabilite` contrôlée, appeler `/predict`, vérifier la logique de classification : p >= t → "ÉLEVÉ", p >= t*0.5 → "MODÉRÉ", sinon → "FAIBLE"
    - Tag : `# Feature: mlops-cicd, Property 7: La classification du risque est correcte par rapport au seuil`
    - **Validates: Requirements 3.5, 3.6**
  - [ ]* 7.3 Écrire le test exemple pour un patient valide complet
    - Appeler `/predict` avec `sample_patient_dict` via `mock_app`, vérifier HTTP 200, `probabilite` dans [0,1], `risque` dans les 3 valeurs valides, `message` non vide, `seuil_utilise` > 0
    - **Validates: Requirements 3.2**

- [ ] 8. Écrire les tests Optuna dans `tests/test_optuna.py`
  - [ ]* 8.1 Écrire les tests exemples pour les objectifs Optuna
    - Créer un petit dataset synthétique (200 lignes × 10 features) avec target binaire équilibré
    - Test 4.1 : Créer un `optuna.trial.FixedTrial` avec des hyperparamètres valides XGBoost, appeler `objective_xgb`, vérifier retour float dans [0.0, 1.0]
    - Test 4.2 : Idem pour `objective_lgb`
    - Test 4.3 : Appeler `optimize_xgb(n_trials=2)`, vérifier retour tuple `(dict, float)` avec dict non vide et float dans [0.0, 1.0]
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [ ] 9. Écrire les tests pipeline/artefacts dans `tests/test_pipeline.py`
  - [ ]* 9.1 Écrire les tests d'intégration pour les artefacts (avec skip si fichiers absents)
    - Utiliser `pytest.mark.skipif` ou `@pytest.fixture(autouse=False)` conditionnel sur l'existence des fichiers
    - Test 5.1 : Charger `models/best_model.pkl`, `models/scaler.pkl`, `data/features/feature_names.csv` ; vérifier que `predict_proba(X_test_synthetic)` retourne shape `(N, 2)`
    - Test 5.2 : Charger `models/threshold.json`, vérifier JSON valide + clé `threshold` dans (0, 1)
    - Test 5.3 : Charger `models/best_model_info.json`, vérifier clés requises et `n_features==28`
    - Test 5.4 : Charger `data/features/feature_names.csv`, vérifier 28 noms et correspondance avec `best_model_info.json`
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [ ] 10. Checkpoint — Exécuter la suite complète de tests
  - Lancer `pytest tests/ --ignore=tests/test_pipeline.py -v --tb=short` et vérifier que tous les tests (sauf pipeline) passent.

- [ ] 11. Créer le workflow CI GitHub Actions
  - [ ] 11.1 Créer le fichier `.github/workflows/ci.yml` avec :
    - Trigger : `push: branches: [main]` et `pull_request: branches: [main]`
    - Job `test` sur `ubuntu-latest`
    - Steps : `actions/checkout@v4`, `actions/setup-python@v5` (version 3.10), cache pip avec clé `${{ hashFiles('requirements.txt') }}`, `pip install -r requirements.txt pytest pytest-cov httpx pytest-mock hypothesis`
    - Étape test : `pytest tests/ --cov=src --cov-report=xml --ignore=tests/test_pipeline.py`
    - Upload artifact `coverage.xml` avec `actions/upload-artifact@v4` (retention-days: 7)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [ ] 11.2 Vérifier la syntaxe YAML du workflow avec `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` et corriger tout problème de syntaxe
    - _Requirements: 6.1_

- [ ] 12. Créer le workflow Train GitHub Actions
  - [ ] 12.1 Créer le fichier `.github/workflows/train.yml` avec :
    - Trigger : `workflow_dispatch` (inputs optionnels) et `schedule: - cron: '0 2 * * 1'`
    - Job `retrain` sur `ubuntu-latest`
    - Steps : checkout, setup-python 3.10, pip install requirements.txt
    - Étape train : `python src/train.py` avec `continue-on-error: false`
    - Étape upload artefacts (conditionnelle sur succès) : `models/*.pkl` et `models/*.json` avec retention-days 30
    - Étape résumé métriques : lire `models/best_model_info.json` et afficher AUC-ROC, F1, précision, rappel dans `$GITHUB_STEP_SUMMARY`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ] 12.2 Vérifier la syntaxe YAML du workflow train
    - _Requirements: 7.1_

- [ ] 13. Checkpoint final — Valider l'ensemble du projet
  - Lancer `pytest tests/ --ignore=tests/test_pipeline.py --cov=src --cov-report=term-missing` et vérifier que la couverture atteint les cibles définies dans le design (src/ensemble_models.py ≥95%, src/api.py ≥85%).
  - Vérifier que les deux fichiers YAML sont syntaxiquement valides.
  - Ensure all tests pass, ask the user if questions arise.

---

## Task Dependency Graph

```json
{
  "waves": [
    {"wave": 1, "tasks": ["1"]},
    {"wave": 2, "tasks": ["2"]},
    {"wave": 3, "tasks": ["3"]},
    {"wave": 4, "tasks": ["4"]},
    {"wave": 5, "tasks": ["5"]},
    {"wave": 6, "tasks": ["6"]},
    {"wave": 7, "tasks": ["7", "8", "9"]},
    {"wave": 8, "tasks": ["10"]},
    {"wave": 9, "tasks": ["11", "12"]},
    {"wave": 10, "tasks": ["13"]}
  ]
}
```

---

## Notes

- Les tâches marquées avec `*` sont optionnelles (tests) — elles peuvent être sautées pour un MVP plus rapide, mais sont fortement recommandées pour la qualité de production
- `tests/test_pipeline.py` est toujours exclu du CI (`--ignore`) car il nécessite les fichiers `.pkl` et `.parquet` réels non présents dans les runners GitHub
- Chaque test de propriété Hypothesis est configuré avec `@settings(max_examples=100)` minimum
- Pour `test_api.py`, le mocking doit se faire avant l'import/rechargement du module `src.api` car il charge les modèles au niveau module
- Les workflows GitHub Actions nécessitent que le repo soit hébergé sur GitHub pour fonctionner
