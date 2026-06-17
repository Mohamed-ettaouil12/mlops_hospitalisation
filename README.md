# 🏥 MLOps — Détection Précoce du Risque d'Hospitalisation

> Plateforme MLOps end-to-end pour prédire le risque d'hospitalisation à 6 mois à partir des données Medicare synthétiques CMS DE-SynPUF (2008–2010).

---

## 🎯 Objectif

Identifier en avance les patients Medicare à haut risque d'hospitalisation afin de prioriser une intervention médicale précoce.

**Contrainte médicale principale :** maximiser le **Recall** — manquer un patient à risque est bien plus grave que générer une fausse alerte.

| Métrique | Objectif | Résultat final |
|---|---|---|
| Recall | ≥ 0.95 | **0.953** (test 2010) |
| AUC-ROC | ≥ 0.88 | **0.961** (test 2010) |
| Faux négatifs | Minimiser | **104** sur 29 075 patients |

---

## 📊 Dataset

**CMS DE-SynPUF** — données Medicare synthétiques, 2.3M bénéficiaires, 60 Go.

| Split | Année | Patients | Taux positifs |
|---|---|---|---|
| Train | 2008 | 30 000 | 7.54 % |
| Validation | 2009 | 29 510 | 9.18 % |
| Test | 2010 | 29 075 | 7.58 % |

> ⚠️ Split **temporel strict** — jamais de données futures dans les features (anti-leakage).

---

## 🏆 Résultats

### Modèle final : Ensemble soft voting, seuil = 0.25

| Split | Recall | Précision | F1 | AUC-ROC | Faux négatifs |
|---|---|---|---|---|---|
| Validation 2009 | 0.984 | 0.283 | 0.440 | 0.941 | 43 |
| **Test 2010** | **0.953** | **0.407** | **0.571** | **0.961** | **104** |

### Comparaison des techniques testées

| Technique | Recall val | AUC val | Seuil |
|---|---|---|---|
| Baseline XGBoost | 0.241 | 0.944 | 0.500 |
| Seuil optimal | 0.950 | 0.944 | 0.050 |
| scale_pos_weight | 0.950 | 0.943 | 0.193 |
| Optuna Recall XGB | 0.950 | 0.952 | 0.591 |
| SMOTE | 0.950 | 0.949 | 0.402 |
| **Ensemble seuil 0.25** | **0.984** | **0.941** | **0.250** |

---

## 🏗️ Architecture du pipeline

```
CMS DE-SynPUF (CSV)
        │
        ▼
┌─────────────────┐
│  Data Cleaning  │  dates · ESRD · cible anti-leakage · dedup (ID+ANNEE)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  29 features · fenêtres 3M/6M/12M · Charlson
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│ Entraînement + Optuna    │  LogReg · XGBoost · LightGBM · Ensemble
│ Optimisation Recall-first│  SMOTE · scale_pos_weight · seuil PR
└────────┬─────────────────┘
         │
         ▼
┌─────────────────┐
│ MLflow + SHAP   │  tracking · courbe PR · explicabilité locale/globale
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ FastAPI + Docker    │  /predict · /health · /model/info
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Monitoring Evidently│  PSI drift · rollback auto · AUC prod
└─────────────────────┘
```

---

## 📂 Structure du projet

```
mlops-hospitalization/
├── src/
│   ├── data_cleaning.py        # Nettoyage, construction cible anti-leakage
│   ├── data_preprocessing.py   # Encodage, normalisation, split temporel
│   ├── feature_engineering.py  # Variables par fenêtres temporelles
│   ├── optuna_tuning.py        # Optimisation Optuna orientée Recall
│   ├── train.py                # Entraînement multi-modèles + MLflow
│   ├── api.py                  # API FastAPI de prédiction
│   └── pipeline.py             # Orchestration bout en bout
├── data/
│   ├── cleaned.dvc
│   └── features.dvc
├── models.dvc
├── outputs/
│   ├── figures/                # Courbes PR, SHAP plots
│   └── reports/                # JSON/CSV métriques et analyses
├── tests/                      # Tests pytest
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Lancer le projet

### Pipeline complet
```bash
PYTHONPATH=. python3 src/pipeline.py
```

### Entraînement seul (recall-first)
```bash
PYTHONPATH=. python3 src/train.py
```

### API FastAPI en local
```bash
PYTHONPATH=. python3 src/api.py
# → http://127.0.0.1:8000/docs
```

### MLflow UI
```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
# → http://127.0.0.1:5000
```

### Docker (API + MLflow ensemble)
```bash
docker compose up --build
# API    → http://127.0.0.1:8000
# MLflow → http://127.0.0.1:5000
```

### Tests
```bash
PYTHONPATH=. pytest
```

---

## 📡 API — Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Statut API + modèle chargé |
| `/predict` | POST | Prédire le risque d'hospitalisation |
| `/model/info` | GET | Métadonnées du modèle actif |

**Exemple de requête :**
```json
POST /predict
{
  "AGE": 72.5,
  "SP_DIABETES": 1,
  "NB_HOSP_6M": 2,
  "COUT_TOTAL": 12500
}
```

**Réponse :**
```json
{
  "risk_score": 0.87,
  "prediction": 1,
  "risque": "ÉLEVÉ",
  "model_version": "ensemble_v1"
}
```

---

## 🔍 Explicabilité SHAP

Top 5 variables les plus prédictives (XGBoost SHAP) :

| Rang | Variable | Importance |
|---|---|---|
| 1 | `COUT_TOTAL` | 0.16 |
| 2 | `CHARLSON_INDEX` | 0.15 |
| 3 | `SP_COPD` | 0.13 |
| 4 | `SP_CHF` | 0.12 |
| 5 | `SP_CHRNKIDN` | 0.11 |

Figures générées :
- `outputs/figures/pr_curve_best_recall.png`
- `outputs/figures/shap_summary_best_recall.png`
- `outputs/figures/shap_importance_best_recall.png`

---

## 📦 DVC — Versionnement données et modèles

```bash
# Récupérer données et modèles
dvc pull

# Après modification
dvc add data/cleaned data/features models
git add data/cleaned.dvc data/features.dvc models.dvc
git commit -m "feat: update data/models"
dvc push
```

---

## 🔄 GitHub — Commits propres

```bash
# Ajouter uniquement les fichiers utiles
git add README.md .gitignore requirements.txt Dockerfile docker-compose.yml
git add src/ tests/
git add data/cleaned.dvc data/features.dvc models.dvc .dvcignore
git add outputs/figures/ outputs/reports/

git commit -m "feat: MLOps recall pipeline — XGBoost + Ensemble + SHAP"
git push origin main
```

---

## ⚠️ Limites

- Modèle optimisé pour le Recall → précision volontairement sacrifiée (faux positifs élevés).
- Dataset CMS DE-SynPUF **synthétique** — usage académique et démonstration MLOps uniquement.
- Les prédictions ne remplacent **jamais** une décision clinique.

---

## 🛠️ Stack technique

`Python 3.11` · `XGBoost` · `LightGBM` · `scikit-learn` · `Optuna` · `MLflow` · `SHAP` · `FastAPI` · `Docker` · `DVC` · `Great Expectations` · `pandas` · `pyarrow`

---

*ENSIAS — Projet de Fin d'Année 2IA · Mohamed Ettaouil · 2024-2025*
