# MLOps Hospitalisation Risk Prediction

Projet MLOps pour la detection precoce du risque d'hospitalisation a partir du dataset Medicare CMS DE-SynPUF. L'objectif principal est de construire une chaine reproductible de preparation des donnees, entrainement, tracking MLflow, versioning DVC, interpretation SHAP et deploiement API.

Le modele final est optimise pour maximiser le recall, car dans un contexte medical il est plus critique de reduire les faux negatifs que de maximiser uniquement la precision.

## Objectif metier

Predire le risque d'hospitalisation d'un patient Medicare afin d'identifier les patients a haut risque et prioriser une intervention ou une surveillance medicale.

Contraintes principales :

- Maximiser le recall de la classe positive.
- Maintenir une AUC-ROC acceptable.
- Respecter un split temporel strict.
- Eviter toute fuite de donnees entre train, validation et test.
- Tracker les experiences avec MLflow.
- Versionner les donnees et modeles avec DVC.

## Dataset

Dataset utilise : CMS DE-SynPUF Medicare 2008-2010.

Structure temporelle :

| Split | Annee | Role |
|---|---:|---|
| Train | 2008 | Entrainement |
| Validation | 2009 | Optimisation seuils et choix modele |
| Test | 2010 | Evaluation finale |

Le taux de positifs est faible, autour de 9%, ce qui rend le probleme fortement desequilibre.

## Resultats principaux

Meilleur modele selectionne :

- Modele : ensemble soft voting
- Seuil final : 0.25
- Objectif recall : >= 0.95
- Contrainte AUC minimale : >= 0.88

### Performance finale

| Split | Recall | Precision | F1 | AUC-ROC | Faux negatifs |
|---|---:|---:|---:|---:|---:|
| Validation 2009 | 0.984 | 0.283 | 0.440 | 0.941 | 43 |
| Test 2010 | 0.953 | 0.407 | 0.571 | 0.961 | 104 |

Interpretation : le modele atteint l'objectif de recall sur le test. La precision reste plus faible, ce qui est attendu avec une strategie orientee reduction des faux negatifs.

### Comparaison des techniques

| Technique | Recall val | Precision val | F1 val | AUC val | Seuil |
|---|---:|---:|---:|---:|---:|
| Baseline XGBoost | 0.241 | 0.571 | 0.339 | 0.944 | 0.500 |
| Seuil optimal | 0.950 | 0.397 | 0.560 | 0.944 | 0.050 |
| scale_pos_weight | 0.950 | 0.393 | 0.556 | 0.943 | 0.193 |
| Optuna Recall XGB | 0.950 | 0.408 | 0.571 | 0.952 | 0.591 |
| SMOTE | 0.950 | 0.407 | 0.570 | 0.949 | 0.402 |
| Ensemble seuil 0.25 | 0.984 | 0.283 | 0.440 | 0.941 | 0.250 |

## Fonctionnalites MLOps

- Data validation et cleaning.
- Feature engineering medical et temporel.
- Split temporel train 2008, validation 2009, test 2010.
- Scaling fit uniquement sur train.
- Entrainement de plusieurs modeles : Logistic Regression, XGBoost, LightGBM, ensembles.
- Optimisation Optuna orientee recall.
- Gestion du desequilibre avec `scale_pos_weight` et SMOTE sur train uniquement.
- Tracking MLflow dans l'experience `maximisation_recall`.
- Sauvegarde du meilleur modele recall.
- Analyse Precision-Recall.
- Explicabilite avec SHAP.
- Versioning DVC des donnees et modeles.
- API FastAPI pour l'inference.
- Docker et docker-compose pour le deploiement local.
- Tests unitaires avec pytest.

## Structure du projet

```text
.
|-- data/
|   |-- cleaned.dvc
|   `-- features.dvc
|-- models.dvc
|-- outputs/
|   |-- figures/
|   |   |-- pr_curve_best_recall.png
|   |   |-- shap_importance_best_recall.png
|   |   `-- shap_summary_best_recall.png
|   `-- reports/
|       |-- recall_training_metrics.json
|       |-- threshold_analysis.json
|       |-- threshold_analysis.csv
|       |-- shap_importance_best_recall.json
|       |-- shap_importance_best_recall.csv
|       `-- shap_local_high_risk.json
|-- src/
|   |-- api.py
|   |-- data_cleaning.py
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- optuna_tuning.py
|   |-- pipeline.py
|   `-- train.py
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Installation locale

Creer et activer un environnement virtuel :

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Recuperer les donnees et modeles avec DVC

Les dossiers lourds `data/cleaned`, `data/features` et `models` ne doivent pas etre versionnes directement dans Git. Ils sont suivis via DVC.

Si un remote DVC est configure :

```bash
dvc pull
```

Pour verifier l'etat DVC :

```bash
dvc status
```

Pour ajouter une modification de donnees ou modeles :

```bash
dvc add data/cleaned data/features models
git add data/cleaned.dvc data/features.dvc models.dvc
```

Puis pousser vers le remote DVC :

```bash
dvc push
```

Note : un remote local comme `../../dvc_remote_storage` ne suffit pas pour partager les donnees avec d'autres utilisateurs GitHub. Pour un partage complet, utiliser un remote cloud ou DagsHub, S3, Google Drive, Azure Blob, etc.

## Lancer le pipeline complet

```bash
PYTHONPATH=. python3 src/pipeline.py
```

Le pipeline execute :

1. Validation des donnees.
2. Nettoyage.
3. Preprocessing et split temporel.
4. Entrainement et selection du meilleur modele.

## Entrainement recall-first

Pour relancer uniquement l'entrainement du modele optimise recall :

```bash
PYTHONPATH=. python3 src/train.py
```

Artefacts generes :

- `models/best_recall_model.pkl`
- `models/best_threshold.json`
- `models/best_recall_model_info.json`
- `outputs/reports/recall_training_metrics.json`
- `outputs/reports/threshold_analysis.json`
- `outputs/reports/threshold_analysis.csv`
- `outputs/reports/shap_importance_best_recall.json`
- `outputs/reports/shap_local_high_risk.json`
- `outputs/figures/pr_curve_best_recall.png`
- `outputs/figures/shap_summary_best_recall.png`
- `outputs/figures/shap_importance_best_recall.png`

## MLflow tracking

Les experiences sont loggees avec MLflow.

Lancer l'interface MLflow :

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

Experience principale :

```text
maximisation_recall
```

Si le port 5000 est deja utilise :

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5001
```

## Explicabilite et seuil PR

Le projet genere :

- Une courbe Precision-Recall.
- Une analyse de seuils pour comparer recall, precision, F1 et faux negatifs.
- Une analyse SHAP globale.
- Une analyse SHAP locale pour des patients a haut risque.

Fichiers principaux :

```text
outputs/figures/pr_curve_best_recall.png
outputs/figures/shap_summary_best_recall.png
outputs/figures/shap_importance_best_recall.png
outputs/reports/threshold_analysis.csv
outputs/reports/shap_importance_best_recall.csv
```

Top features importantes d'apres SHAP :

- `COUT_TOTAL`
- `CHARLSON_INDEX`
- `SP_COPD`
- `SP_CHF`
- `SP_CHRNKIDN`

## API FastAPI

Lancer l'API localement :

```bash
PYTHONPATH=. python3 src/api.py
```

Endpoints disponibles :

| Endpoint | Methode | Description |
|---|---|---|
| `/health` | GET | Verifie l'etat de l'API |
| `/predict` | POST | Predire le risque d'hospitalisation |
| `/model/info` | GET | Informations sur le modele charge |

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

## Docker

Construire et lancer les services :

```bash
docker compose up --build
```

Services :

- API FastAPI : `http://127.0.0.1:8000`
- MLflow UI : `http://127.0.0.1:5000`

## Tests

Lancer les tests :

```bash
PYTHONPATH=. pytest
```

Ou avec le venv :

```bash
PYTHONPATH=. venv/bin/python -m pytest
```

## Commandes GitHub recommandees

Ne pas faire :

```bash
git add .
```

Ajouter seulement les fichiers utiles :

```bash
git add README.md .gitignore requirements.txt Dockerfile docker-compose.yml
git add src tests
git add data/cleaned.dvc data/features.dvc models.dvc .dvcignore
git add outputs/figures/pr_curve_best_recall.png
git add outputs/figures/shap_summary_best_recall.png
git add outputs/figures/shap_importance_best_recall.png
git add outputs/reports/recall_training_metrics.json
git add outputs/reports/threshold_analysis.json outputs/reports/threshold_analysis.csv
git add outputs/reports/shap_importance_best_recall.json outputs/reports/shap_importance_best_recall.csv
git add outputs/reports/shap_local_high_risk.json
```

Verifier avant commit :

```bash
git diff --cached --name-only
```

Commit :

```bash
git commit -m "Add MLOps recall model documentation"
```

Connexion au repository GitHub :

```bash
git branch -M main
git remote add origin https://github.com/TON_USERNAME/mlops_hospitalisation.git
git push -u origin main
```

Remplacer `TON_USERNAME` par le nom de votre compte GitHub.

## Fichiers a ne pas envoyer directement dans Git

Ces elements doivent rester ignores ou suivis via DVC :

- `venv/`
- `mlruns/`
- `logs/`
- `catboost_info/`
- `models/`
- `data/cleaned/`
- `data/features/`
- `__pycache__/`
- `.pytest_cache/`

## Limites

- Le modele est optimise pour le recall, donc il genere plus de faux positifs.
- Les predictions ne remplacent pas une decision clinique.
- Le dataset CMS DE-SynPUF est synthetique et doit etre utilise pour experimentation, formation et demonstration MLOps.
