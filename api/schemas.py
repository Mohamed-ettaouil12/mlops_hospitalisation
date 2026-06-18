from pydantic import BaseModel, Field


class PatientData(BaseModel):
    AGE: float = Field(..., ge=0, le=120)
    SEXE_ENC: int = Field(..., ge=0, le=1)
    RACE_ENC: int = Field(..., ge=0, le=5)
    BENE_ESRD_IND: int = Field(0, ge=0, le=1)
    GROUPE_AGE_ENC: int = Field(..., ge=0, le=3)
    SP_ALZHDMTA: int = Field(0, ge=0, le=1)
    SP_CHF: int = Field(0, ge=0, le=1)
    SP_CHRNKIDN: int = Field(0, ge=0, le=1)
    SP_CNCR: int = Field(0, ge=0, le=1)
    SP_COPD: int = Field(0, ge=0, le=1)
    SP_DEPRESSN: int = Field(0, ge=0, le=1)
    SP_DIABETES: int = Field(0, ge=0, le=1)
    SP_ISCHMCHT: int = Field(0, ge=0, le=1)
    SP_OSTEOPRS: int = Field(0, ge=0, le=1)
    SP_RA_OA: int = Field(0, ge=0, le=1)
    SP_STRKETIA: int = Field(0, ge=0, le=1)
    SP_STATE_CODE: int = Field(0, ge=0)
    NB_COMORBIDITES: float = Field(0, ge=0)
    CHARLSON_INDEX: float = Field(0, ge=0)
    COUT_TOTAL: float = Field(0, ge=0)
    IS_NEW_PATIENT: int = Field(0, ge=0, le=1)
    NB_HOSP_PASSEES: float = Field(0, ge=0)
    NB_OP_3M: float = Field(0, ge=0)
    NB_OP_6M: float = Field(0, ge=0)
    NB_OP_12M: float = Field(0, ge=0)
    NB_CAR_6M: float = Field(0, ge=0)
    NB_PRESCRIPTIONS: float = Field(0, ge=0)
    NB_MOLECULES_UNIQUES: float = Field(0, ge=0)
    POLYPHARMACIE: int = Field(0, ge=0, le=1)


class PredictionResult(BaseModel):
    probabilite: float
    risque: str
    seuil_utilise: float
    version_modele: str


class ModelInfo(BaseModel):
    type: str
    version: str
    stage: str
    nb_features: int
    features: list[str]


class DriftScoreRequest(BaseModel):
    drift_max_psi: float = 0.0
    model_version: str = "unknown"


class AccuracyUpdateRequest(BaseModel):
    model_version: str
    values: dict[str, float]
