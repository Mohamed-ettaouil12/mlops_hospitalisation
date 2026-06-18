import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
TARGET_COL = "HOSPITALIZED_IN_6M"


def main():
    prod_path = MODELS_DIR / "production_model.pkl"
    if not prod_path.exists():
        prod_path = MODELS_DIR / "best_recall_model.pkl"
    if not prod_path.exists():
        result = {"model_available": False, "error": "no_model_found"}
        print(json.dumps(result))
        log.warning("Aucun modèle trouvé")
        return result

    model = joblib.load(prod_path)
    threshold = 0.5
    thresh_path = MODELS_DIR / "production_threshold.json"
    if thresh_path.exists():
        threshold = float(json.loads(thresh_path.read_text()).get("threshold", 0.5))
    elif (MODELS_DIR / "best_threshold.json").exists():
        threshold = float(json.loads((MODELS_DIR / "best_threshold.json").read_text()).get("threshold", 0.5))

    try:
        X_val = pd.read_parquet(FEATURES_DIR / "X_val.parquet")
        y_val = pd.read_parquet(FEATURES_DIR / "y_val.parquet")
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val[TARGET_COL] if TARGET_COL in y_val.columns else y_val.iloc[:, 0]
        y_val = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int).reset_index(drop=True)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_val, y_proba))

        result = {
            "model_available": True,
            "recall": round(recall, 4),
            "auc_roc": round(auc, 4),
            "threshold": threshold,
            "n_samples": len(y_val),
            "model_path": str(prod_path),
        }
    except Exception as e:
        result = {"model_available": False, "error": str(e)}
        log.error("Erreur évaluation performance: %s", e)

    result_path = PROJECT_ROOT / "outputs" / "reports" / "performance_check.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))

    log.info("Performance: recall=%.4f, AUC=%.4f", result.get("recall", 0), result.get("auc_roc", 0))
    print(json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    main()
