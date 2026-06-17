# ═══════════════════════════════════════════════════════════
# src/api.py
# Pipeline MLOps — API REST FastAPI
# ═══════════════════════════════════════════════════════════

import joblib
import json
import pandas as pd
import numpy as np
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
MODELS_DIR   = 'models/'
FEATURES_DIR = 'data/features/'


def load_model_threshold(default: float = 0.5) -> float:
    threshold_path = os.path.join(MODELS_DIR, 'threshold.json')
    try:
        with open(threshold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return float(data.get('threshold', default))
    except Exception as e:
        log.warning(f"Seuil introuvable ou invalide, fallback {default}: {e}")
        return default

# ── Chargement du modèle au démarrage ─────────────────────
log.info("Chargement du modèle...")
model   = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
features = pd.read_csv(os.path.join(FEATURES_DIR, 'feature_names.csv'))
FEATURE_NAMES = features.iloc[:, 0].tolist()
MODEL_THRESHOLD = load_model_threshold()
log.info(f"Modèle chargé : {type(model).__name__}")
log.info(f"Nombre de features : {len(FEATURE_NAMES)}")
log.info(f"Seuil modèle : {MODEL_THRESHOLD:.4f}")

# ── Application FastAPI ────────────────────────────────────
app = FastAPI(
    title="API MLOps — Risque d'Hospitalisation",
    description="Prédit le risque d'hospitalisation à 6 mois pour un patient Medicare",
    version="1.0.0"
)


# ═══════════════════════════════════════════════════════════
# SCHÉMA D'ENTRÉE — Données d'un patient
# ═══════════════════════════════════════════════════════════
class PatientData(BaseModel):
    # Démographie
    AGE               : float = Field(..., ge=0, le=120, description="Âge du patient")
    SEXE_ENC          : int   = Field(..., ge=0, le=1,   description="Sexe (0=Femme, 1=Homme)")
    RACE_ENC          : int   = Field(..., ge=0, le=5,   description="Race encodée (0-5)")
    BENE_ESRD_IND     : int   = Field(0,  ge=0, le=1,   description="Indicateur ESRD")
    GROUPE_AGE_ENC    : int   = Field(..., ge=0, le=3,   description="Groupe âge (0=<65, 1=65-74, 2=75-84, 3=85+)")

    # Comorbidités
    SP_ALZHDMTA       : int = Field(0, ge=0, le=1)
    SP_CHF            : int = Field(0, ge=0, le=1)
    SP_CHRNKIDN       : int = Field(0, ge=0, le=1)
    SP_CNCR           : int = Field(0, ge=0, le=1)
    SP_COPD           : int = Field(0, ge=0, le=1)
    SP_DEPRESSN       : int = Field(0, ge=0, le=1)
    SP_DIABETES       : int = Field(0, ge=0, le=1)
    SP_ISCHMCHT       : int = Field(0, ge=0, le=1)
    SP_OSTEOPRS       : int = Field(0, ge=0, le=1)
    SP_RA_OA          : int = Field(0, ge=0, le=1)
    SP_STRKETIA       : int = Field(0, ge=0, le=1)
    SP_STATE_CODE     : int   = Field(0, ge=0, description="Code état (non utilisé par le modèle)")

    # Features composites
    NB_COMORBIDITES   : float = Field(0, ge=0)
    CHARLSON_INDEX    : float = Field(0, ge=0)
    COUT_TOTAL        : float = Field(0, ge=0)
    IS_NEW_PATIENT    : int   = Field(0, ge=0, le=1)

    # Features dynamiques
    NB_HOSP_PASSEES   : float = Field(0, ge=0)
    NB_OP_3M          : float = Field(0, ge=0)
    NB_OP_6M          : float = Field(0, ge=0)
    NB_OP_12M         : float = Field(0, ge=0)
    NB_CAR_6M         : float = Field(0, ge=0)
    NB_PRESCRIPTIONS  : float = Field(0, ge=0)
    NB_MOLECULES_UNIQUES : float = Field(0, ge=0)
    POLYPHARMACIE     : int   = Field(0, ge=0, le=1)


# ═══════════════════════════════════════════════════════════
# SCHÉMA DE SORTIE — Résultat de la prédiction
# ═══════════════════════════════════════════════════════════
class PredictionResult(BaseModel):
    probabilite    : float
    risque         : str
    seuil_utilise  : float
    message        : str


# ═══════════════════════════════════════════════════════════
# ENDPOINT 1 : Health Check
# ═══════════════════════════════════════════════════════════
@app.get("/health")
def health_check():
    return {
        "status"  : "ok",
        "modele"  : type(model).__name__,
        "features": len(FEATURE_NAMES),
        "threshold": MODEL_THRESHOLD,
        "version" : "1.0.0"
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 2 : Prédiction
# ═══════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientData):
    try:
        # Convertir en DataFrame
        data = pd.DataFrame([patient.dict()])

        # Garder seulement les features du modèle dans le bon ordre
        data = data.reindex(columns=FEATURE_NAMES, fill_value=0)

        # Normaliser avec le scaler
        cols_to_scale = [
            'AGE', 'COUT_TOTAL', 'CHARLSON_INDEX', 'NB_COMORBIDITES',
            'NB_HOSP_PASSEES', 'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
            'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES'
        ]
        cols_to_scale = [c for c in cols_to_scale if c in data.columns]
        data[cols_to_scale] = scaler.transform(data[cols_to_scale])

        # Prédiction
        probabilite = float(model.predict_proba(data)[0, 1])
        seuil       = MODEL_THRESHOLD
        seuil_modere = max(0.0, min(1.0, seuil * 0.5))

        # Niveau de risque
        if probabilite >= seuil:
            risque  = "ÉLEVÉ"
            message = "⚠️ Risque élevé — Intervention médicale recommandée"
        elif probabilite >= seuil_modere:
            risque  = "MODÉRÉ"
            message = "⚡ Risque modéré — Surveillance renforcée recommandée"
        else:
            risque  = "FAIBLE"
            message = "✅ Risque faible — Suivi standard suffisant"

        log.info(f"Prédiction : {probabilite:.3f} | Risque : {risque}")

        return PredictionResult(
            probabilite   = round(probabilite, 4),
            risque        = risque,
            seuil_utilise = seuil,
            message       = message
        )

    except Exception as e:
        log.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ENDPOINT 3 : Informations sur le modèle
# ═══════════════════════════════════════════════════════════
@app.get("/model/info")
def model_info():
    return {
        "type"        : type(model).__name__,
        "nb_features" : len(FEATURE_NAMES),
        "threshold"   : MODEL_THRESHOLD,
        "features"    : FEATURE_NAMES,
        "description" : "Modèle de prédiction du risque d'hospitalisation Medicare"
    }


# ═══════════════════════════════════════════════════════════
# LANCEMENT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
