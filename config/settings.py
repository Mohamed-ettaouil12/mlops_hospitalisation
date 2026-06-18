import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", str(MLFLOW_DIR / "artifacts"))
MLFLOW_EXPERIMENT = "hospitalization_risk"
MLFLOW_MODEL_NAME = "hospitalization_risk_model"

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
TARGET_COL = "HOSPITALIZED_IN_6M"
RANDOM_STATE = 42

RECALL_TARGET = 0.95
MIN_AUC_FOR_PRODUCTION = 0.85
PSI_THRESHOLD = 0.20

LOG_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
TRAINING_REPORTS_DIR = REPORTS_DIR / "training"
VALIDATION_REPORTS_DIR = REPORTS_DIR / "validation"

for d in [LOG_DIR, DRIFT_REPORTS_DIR, TRAINING_REPORTS_DIR, VALIDATION_REPORTS_DIR,
          MLFLOW_DIR, Path(MLFLOW_ARTIFACT_ROOT)]:
    os.makedirs(d, exist_ok=True)
