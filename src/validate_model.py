"""
Validate model performance before deployment.
Kept for backward compatibility.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config.settings as settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MIN_AUC = float(sys.argv[1]) if len(sys.argv) > 1 else settings.MIN_AUC_FOR_PRODUCTION


def main():
    report_path = settings.TRAINING_REPORTS_DIR / "retrain_result.json"
    if not report_path.exists():
        log.warning("No retrain result found")
        print(json.dumps({"validated": False, "reason": "no_report"}))
        return
    report = json.loads(report_path.read_text())
    val_auc = report.get("decision", {}).get("new_metrics", {}).get("auc_roc", 0)
    version = report.get("version", "unknown")
    promoted = val_auc >= MIN_AUC
    result = {
        "validated": True,
        "version": version,
        "val_auc": val_auc,
        "min_auc": MIN_AUC,
        "promoted_to_production": promoted,
    }
    log.info("Model v%s: AUC=%.4f >= %.2f -> %s", version, val_auc, MIN_AUC, "PASS" if promoted else "FAIL")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
