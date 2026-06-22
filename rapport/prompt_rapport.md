# Génération du Rapport PFA MLOps — Style Académique Professionnel

## Rôle
Tu es un étudiant ingénieur en 2ème année IA qui rédige son rapport 
de PFA. Tu dois écrire en prose académique naturelle, comme un vrai 
étudiant, pas comme une IA.

## Fichiers disponibles
- `rapportENSIAS.cls` : classe LaTeX de l'école
- `refs.bib` : bibliographie
- `figures/mlflow.png` : capture interface MLflow réelle
- `figures/swagger.png` : capture API Swagger /docs + /predict
- `figures/docker.png` : capture conteneurs Docker actifs
- `figures/grafana.png` : capture Dashboard Grafana/Evidently
- `figures/prefect.png` : capture interface Prefect
- `figures/evidently.png` : capture rapport Evidently drift

## Page de garde
- Titre : Plateforme MLOps pour la Détection Précoce du Risque
  d'Hospitalisation
- Sous-titre : Prévenir les hospitalisations évitables chez les 
  patients polypathologiques grâce à l'intelligence artificielle
- Étudiant : Mohamed Ettaouil
- École : École Nationale d'Intelligence Artificielle — Promotion 2026
- Spécialité : Machine Learning Operations (MLOps)
- Secteur : Santé — Intelligence Artificielle Médicale
- Année : 2025-2026
- Encadrant : [NOM ENCADRANT]
- Jury : [NOM JURY 1] / [NOM JURY 2]

---

## RÈGLES DE RÉDACTION — ABSOLUMENT OBLIGATOIRES

### Règle 1 — AUCUN code dans le corps du rapport
Le code Python, YAML, Bash est STRICTEMENT INTERDIT dans les 
chapitres 1, 2, 3 et 4.
Le code va UNIQUEMENT dans les annexes.

### Règle 2 — Prose académique fluide
Chaque section doit être rédigée en paragraphes complets et 
connectés. Pas de listes à puces pour expliquer une technique.
Les listes sont acceptées uniquement pour énumérer des éléments 
courts (ex: liste d'outils).

### Règle 3 — Expliquer le POURQUOI avant le COMMENT
Pour chaque choix technique, toujours expliquer :
- Pourquoi ce choix a été fait
- Quel problème il résout
- Ce qu'on a observé comme résultat concret

### Règle 4 — Figures uniquement réelles
Les figures sont UNIQUEMENT des captures d'écran réelles avec 
\includegraphics. Pas de TikZ complexe, pas de pgfplots dans 
le corps. Une seule figure TikZ simple est acceptée pour 
l'architecture globale au Chapitre 1.

### Règle 5 — Analyser chaque tableau
Chaque tableau doit être suivi d'un paragraphe d'analyse qui 
explique ce que les chiffres signifient concrètement.

### Règle 6 — Transitions entre sections
Chaque section se termine par une phrase de transition vers 
la section suivante. Chaque chapitre se termine par une 
synthèse d'un paragraphe.

### Règle 7 — Ton naturel d'étudiant ingénieur
Évite les formulations trop génériques comme "il est important 
de noter que" ou "dans le cadre de ce projet". 
Écris directement et précisément.

---

## CONTRAINTES FORMAT

- Maximum 65 pages
- 4 chapitres exactement avec ces titres :
  * Chapitre 1 : Comprendre le Problème avant de le Résoudre (12p)
  * Chapitre 2 : De la Donnée Brute à la Donnée Prête (18p)
  * Chapitre 3 : Apprendre à Prédire : Modèles et Expérimentation (16p)
  * Chapitre 4 : Du Modèle au Service : Mise en Production (14p)
  * Introduction (2p) + Conclusion (2p) + Annexes (1p)
- Maximum 9 figures :
  * Figure 1.1 : Architecture pipeline (TikZ simple — la seule)
  * Figure 2.1 : Schéma fenêtres temporelles anti-leakage (TikZ)
  * Figure 3.1 : \includegraphics{figures/mlflow.png}
  * Figure 3.2 : \includegraphics{figures/swagger.png} 
  * Figure 4.1 : \includegraphics{figures/docker.png}
  * Figure 4.2 : \includegraphics{figures/grafana.png}
  * Figure 4.3 : \includegraphics{figures/evidently.png}
  * Figure 4.4 : \includegraphics{figures/prefect.png}
  * (réserver 1 figure pour fairness ou monitoring)
- Tableaux avec booktabs uniquement
- Boîtes tcolorbox pour les points critiques uniquement
  (pas plus de 2 par chapitre)

---

## DONNÉES RÉELLES DU PROJET

### Dataset CMS DE-SynPUF
- Période : 2008 à 2010
- Volume : 60 Go, 20 samples, ~116K patients/sample
- Total : 2.3 millions de bénéficiaires
- Développement : 30,000 patients (Sample 1)

### Statistiques EDA réelles
- 30,000 patients, aucun doublon
- 55.3% de femmes, 44.7% d'hommes
- 4 groupes raciaux (Blanc majoritaire)
- 490 décès enregistrés
- Dates corrigées : 2007-12-10 à 2010-12-30

### Volume claims (30K patients)
- Inpatient : 17,325 lignes
- Outpatient : 205,642 lignes
- Carrier A+B : 1,233,861 lignes
- Prescription : 1,436,994 lignes
- Total : ~2.9 millions de lignes

### Bénéficiaires par année
- 2008 : 30,000 | 2009 : 29,510 | 2010 : 29,075
- Total fusionné : 88,585 lignes, 33 colonnes

### Taux d'hospitalisation par année
- 2008 Train : 7.54% (2,261 hospitalisés)
- 2009 Validation : 9.18% (2,710 hospitalisés)
- 2010 Test : 7.58% (2,203 hospitalisés)
- Taux global : 8.10%

### Les 11 comorbidités chroniques
| Code | Maladie | Prévalence | Poids Charlson |
|---|---|---|---|
| SP_ALZHDMTA | Alzheimer | 43.8% | 1 |
| SP_CHF | Insuffisance cardiaque | 41.3% | 1 |
| SP_RA_OA | Arthrite | 38.5% | 1 |
| SP_ISCHMCHT | Coronaropathie | 31.2% | 1 |
| SP_DIABETES | Diabète | 28.7% | 1 |
| SP_OSTEOPRS | Ostéoporose | 22.4% | 0 |
| SP_COPD | BPCO | 18.9% | 1 |
| SP_DEPRESSN | Dépression | 14.2% | 1 |
| SP_STRKETIA | AVC/AIT | 8.3% | 2 |
| SP_CNCR | Cancer | 5.8% | 2 |
| SP_CHRNKIDN | Maladie rénale | 2.1% | 2 |

### Association comorbidités → hospitalisation
| Comorbidité | Taux présente | Taux absente | Odds Ratio |
|---|---|---|---|
| Insuffisance cardiaque | 18.4% | 4.2% | 5.1 |
| AVC/AIT | 16.2% | 6.3% | 3.0 |
| BPCO | 15.8% | 5.9% | 3.1 |
| Diabète | 12.3% | 5.8% | 2.3 |
| Coronaropathie | 11.7% | 5.1% | 2.5 |
| Dépression | 10.9% | 6.2% | 1.9 |

### Résultats ACP et KMeans
- CP1 : 31.4% variance | CP2 : 18.7% | 3CP : 62.1%
- Cluster 0 Faible risque : 14,820 patients | 3.2% | 0.8 comorbidités
- Cluster 1 Risque modéré : 10,940 patients | 7.8% | 2.3 comorbidités
- Cluster 2 Haut risque : 4,240 patients | 22.1% | 5.1 comorbidités

### Bugs corrigés
| Problème | Solution |
|---|---|
| Dates flottants 20100312.0 | Strip .0 + format %Y%m%d |
| BENE_ESRD_IND mixte Y/0 | Mapping Y→1 |
| Coûts extrêmes | Winsorisation P99 |
| Doublons multi-années | Clé (DESYNPUF_ID, ANNEE) |

### Data Cleaning résultats
- 88,585 lignes, 33 colonnes
- 0 doublon sur clé (ID, ANNEE)
- 6,216 patients cold-start (IS_NEW_PATIENT=1)
- 9 colonnes de coûts winsorisées

### Les 29 features
Statiques (16) : AGE, SEXE_ENC, RACE_ENC, BENE_ESRD_IND,
GROUPE_AGE_ENC + 11 comorbidités SP_*

Dynamiques (8) : NB_HOSP_PASSEES, NB_OP_3M, NB_OP_6M,
NB_OP_12M, NB_CAR_6M, NB_PRESCRIPTIONS,
NB_MOLECULES_UNIQUES, COUT_HOSP_PASSE

Composites (5) : NB_COMORBIDITES(moy=2.26),
CHARLSON_INDEX(moy=2.34), COUT_TOTAL,
POLYPHARMACIE, IS_NEW_PATIENT

### Hyperparamètres optimaux Optuna (50 essais)
| Paramètre | XGBoost | LightGBM |
|---|---|---|
| n_estimators | 423 | 387 |
| max_depth | 6 | 7 |
| learning_rate | 0.042 | 0.038 |
| subsample | 0.82 | 0.79 |
| colsample_bytree | 0.74 | 0.71 |
| scale_pos_weight | 10.94 | 10.94 |
| Seuil optimal | 0.87 | 0.83 |

### Résultats comparatifs
| Modèle | AUC Val.2009 | AUC Test.2010 | Recall | F1 |
|---|---|---|---|---|
| Régression Logistique | 0.9082 | 0.8847 | 0.9339 | 0.3825 |
| XGBoost + Optuna | 0.9245 | 0.8973 | 0.9150 | 0.4210 |
| LightGBM + Optuna | 0.9210 | 0.8934 | 0.9200 | 0.4150 |

### Matrice de confusion XGBoost (Test 2010)
- VP=1,965 | FN=238 | FP=7,230 | VN=19,642
- Recall=89.2% | Précision=21.4%

### Seuil de décision
- Logistique : 0.961 | XGBoost : 0.87 | LightGBM : 0.83
- Justification : minimiser faux négatifs (coût médical)

### Top 15 features SHAP
NB_HOSP_12M=0.198, NB_OP_6M=0.178, NB_HOSP_6M=0.161,
NB_MOLECULES_UNIQUES=0.147, NB_COMORBIDITES=0.132,
COUT_TOTAL=0.118, CHARLSON_INDEX=0.103, AGE=0.091

### ROI
- VP=1,965 | FP=7,230
- Coût prévention : 9.20 M$
- Économies : 11.79 M$
- ROI net : +2.59 M$

### API FastAPI
- /health : statut + version
- /predict : score < 100ms
- /model/info : métadonnées
- Niveaux : FAIBLE(<0.40) MODÉRÉ(0.40-0.70) ÉLEVÉ(≥0.70)

### Stack technologique
MVP : XGBoost, LightGBM, Logistique, MLflow, FastAPI,
Docker, Evidently AI, Grafana, GitHub Actions, DVC,
Great Expectations, Prefect

### Seuils monitoring
- PSI > 0.20 → Ré-entraînement auto
- AUC < 0.75 → Rollback automatique
- Recall < 0.70 → Ré-entraînement
- Latence P99 > 150ms → Alerte

### Simulation dérive 2010→2026
| Variable | Perturbation | PSI | Alerte |
|---|---|---|---|
| COUT_TOTAL | +30% | 0.28 | Oui |
| AGE | +3 ans | 0.19 | Non |
| NB_COMORBIDITES | +0.5 | 0.31 | Oui |
| Taux hosp. | 7.5%→12% | 0.44 | Oui |

### Fairness par groupe racial
| Groupe | N | AUC | Recall | FP rate |
|---|---|---|---|---|
| Blanc | 21,840 | 0.901 | 0.894 | 0.241 |
| Noir | 4,210 | 0.887 | 0.871 | 0.268 |
| Autre | 1,520 | 0.883 | 0.863 | 0.274 |
| Asiatique | 830 | 0.879 | 0.858 | 0.281 |
| Hispanique | 520 | 0.876 | 0.851 | 0.289 |
| Natif Américain | 155 | 0.871 | 0.839 | 0.295 |
| Écart max Recall | | | 5.5% | |

### Tests réalisés
| Test | Résultat |
|---|---|
| Great Expectations | 2/2 réussis |
| Data Cleaning | 88,585 lignes |
| Feature Engineering | 29 features |
| XGBoost | AUC 0.9245 |
| API /predict | < 100ms |
| Docker compose | 4 services actifs |
| Drift PSI simulé | 0.31 détecté |
| Rollback | Fonctionnel |
| Fairness | 5.5% écart |

### Pourquoi pas Kafka/Spark/Redis/Kubernetes
- Kafka : dataset historique statique, batch suffit
- Spark : 2.9M lignes en quelques secondes avec Pandas
- Redis : API déjà < 100ms sans cache
- Kubernetes : Docker Compose suffit pour ce scale

---

## STRUCTURE DES FICHIERS À GÉNÉRER
