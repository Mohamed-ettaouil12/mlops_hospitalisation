import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config.settings as settings
from src.model_registry import compare_and_promote, register_model

MODELS_DIR = PROJECT_ROOT / "models"


def main():
    model_path = MODELS_DIR / "best_recall_model.pkl"
    if not model_path.exists():
        log.warning("Aucun nouveau modèle trouvé: %s", model_path)
        result = {"registered": False, "reason": "no_new_model"}
        print(json.dumps(result))
        return result

    entry = register_model(
        name=settings.MLFLOW_MODEL_NAME,
        model_path=model_path,
        stage="Staging",
    )
    log.info("Modèle enregistré en Staging: %s", entry["version_id"])

    result = compare_and_promote(entry["version_id"], metric="recall")

    result["version_id"] = entry["version_id"]
    log.info("Résultat promotion: %s", json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
