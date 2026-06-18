"""
Compare models using the champion/challenger registry.
Kept for backward compatibility - delegates to model_registry.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config.settings as settings
from src.model_registry import registry
from src.preprocessing import load_train_val_test

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    _, X_val, _, _, y_val, _, _ = load_train_val_test()
    report_path = settings.TRAINING_REPORTS_DIR / "retrain_result.json"
    if not report_path.exists():
        log.warning("No retrain result found")
        print(json.dumps({"compared": False, "reason": "no_retrain_result"}))
        return
    report = json.loads(report_path.read_text())
    new_version = report["version"]
    log.info("Comparing model v%s with champion", new_version)
    decision = registry.compare_and_decide(X_val, y_val, new_version)
    print(json.dumps(decision, indent=2, default=str))
    return decision


if __name__ == "__main__":
    main()
