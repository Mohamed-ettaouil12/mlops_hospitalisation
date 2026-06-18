import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECALL_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85


def main():
    drift_path = PROJECT_ROOT / "outputs" / "reports" / "drift_check_result.json"
    perf_path = PROJECT_ROOT / "outputs" / "reports" / "performance_check.json"

    drift = json.loads(drift_path.read_text()) if drift_path.exists() else {"drift_detected": False}
    perf = json.loads(perf_path.read_text()) if perf_path.exists() else {"model_available": False}

    reasons = []
    do_retrain = False

    if drift.get("drift_detected"):
        do_retrain = True
        reasons.append(f"drift: {drift['n_drifted']} features (max PSI={drift['max_psi']:.4f})")

    if perf.get("model_available"):
        recall = perf.get("recall", 1.0)
        if recall < RECALL_MIN:
            do_retrain = True
            reasons.append(f"performance degradation: recall={recall:.4f} < {RECALL_MIN}")
    else:
        do_retrain = True
        reasons.append("no production model available")

    result = {"retrain": do_retrain, "reasons": reasons}
    print(json.dumps(result))

    if do_retrain:
        log.warning("DÉCISION: réentraînement nécessaire — %s", " | ".join(reasons))
        sys.exit(0)
    else:
        log.info("DÉCISION: pas de réentraînement nécessaire")
        sys.exit(1)


if __name__ == "__main__":
    main()
