"""
Production-grade Drift Detection pipeline (Evidently AI + PSI).
Generates structured reports, triggers alerts, and stores drift history.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config.settings as settings
from src.feature_store import feature_store

log = logging.getLogger(__name__)


def safe_numeric(series, fill=0):
    if not isinstance(series, pd.Series):
        return fill
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)


def load_feature_names() -> list[str]:
    p = settings.FEATURES_DIR / "feature_names.csv"
    return pd.read_csv(p)["feature"].tolist() if p.exists() else []


def load_data():
    df = pd.read_parquet(settings.FEATURES_DIR / "features_engineered.parquet")
    df["SEXE_ENC"] = safe_numeric(df.get("BENE_SEX_IDENT_CD", 0))
    df["RACE_ENC"] = safe_numeric(df.get("BENE_RACE_CD", 0))
    df["BENE_ESRD_IND"] = safe_numeric(df.get("BENE_ESRD_IND", 0))
    df["GROUPE_AGE_ENC"] = (
        pd.cut(df["AGE"], bins=[0, 65, 75, 85, 120], labels=[0, 1, 2, 3], include_lowest=True)
        .astype(float).fillna(1)
    )
    features = load_feature_names()
    target_col = settings.TARGET_COL
    ref = df[df["ANNEE"] == 2008].copy()
    prod = df[df["ANNEE"] == 2010].copy()
    y_ref = pd.to_numeric(ref.get(target_col, 0), errors="coerce").fillna(0).astype(int)
    y_prod = pd.to_numeric(prod.get(target_col, 0), errors="coerce").fillna(0).astype(int)
    for col in features:
        ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(0).astype(float)
        prod[col] = pd.to_numeric(prod[col], errors="coerce").fillna(0).astype(float)
    return ref[features], prod[features], y_ref, y_prod


def calc_psi(expected, actual, n_bins=10):
    combined = pd.concat([expected, actual])
    lo, hi = combined.min(), combined.max()
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    e = np.histogram(expected, bins=bins)[0].astype(float) / max(len(expected), 1)
    a = np.histogram(actual, bins=bins)[0].astype(float) / max(len(actual), 1)
    e = np.clip(e, 0.0001, 1)
    a = np.clip(a, 0.0001, 1)
    return float(np.sum((a - e) * np.log(a / e)))


def build_evidently_reports(ref, prod, y_ref, y_prod, features):
    """Generate Evidently AI reports if available."""
    try:
        from evidently import Report
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
        dd = Report(metrics=[DataDriftPreset(num_method="psi")])
        dd.run(current_data=prod, reference_data=ref)
        dd.save_html(str(settings.DRIFT_REPORTS_DIR / "drift_report.html"))
        td = Report(metrics=[TargetDriftPreset()])
        td.run(current_data=prod.assign(**{settings.TARGET_COL: y_prod}),
               reference_data=ref.assign(**{settings.TARGET_COL: y_ref}))
        td.save_html(str(settings.DRIFT_REPORTS_DIR / "target_drift_report.html"))
        dq = Report(metrics=[DataQualityPreset()])
        dq.run(current_data=prod, reference_data=ref)
        dq.save_html(str(settings.DRIFT_REPORTS_DIR / "data_quality_report.html"))
        log.info("Evidently reports generated -> %s", settings.DRIFT_REPORTS_DIR)
        return True
    except ImportError:
        log.warning("Evidently not installed, skipping HTML reports")
        return False
    except Exception as e:
        log.warning("Evidently report error: %s", e)
        return False


def detect_drift(psi_threshold: Optional[float] = None) -> dict:
    if psi_threshold is None:
        psi_threshold = settings.PSI_THRESHOLD
    settings.DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features = load_feature_names()
    ref, prod, y_ref, y_prod = load_data()
    log.info("Reference (2008): %s | Production (2010): %s", ref.shape, prod.shape)
    psi_results = {}
    for col in features:
        psi = calc_psi(ref[col], prod[col])
        psi_results[col] = psi
    drifted = [c for c, v in psi_results.items() if v > psi_threshold]
    max_psi = max(psi_results.values()) if psi_results else 0.0
    n_drift = len(drifted)
    drift_pct = round(100 * n_drift / max(len(features), 1), 1)
    ref_rate = float(y_ref.mean())
    prod_rate = float(y_prod.mean())
    target_drift = abs(ref_rate - prod_rate) > 0.02
    build_evidently_reports(ref, prod, y_ref, y_prod, features)
    drift_severity = "NONE"
    if n_drift > 0:
        if drift_pct > 30:
            drift_severity = "CRITICAL"
        elif drift_pct > 15:
            drift_severity = "HIGH"
        else:
            drift_severity = "MODERATE"
    result = {
        "drift_detected": n_drift > 0 or target_drift,
        "drift_severity": drift_severity,
        "psi_threshold": psi_threshold,
        "n_features": len(features),
        "n_drifted": n_drift,
        "drift_pct": drift_pct,
        "drifted_features": drifted,
        "max_psi": round(max_psi, 4),
        "target_drift": target_drift,
        "target_ref_rate": round(ref_rate, 4),
        "target_prod_rate": round(prod_rate, 4),
        "top5_psi": sorted(psi_results.items(), key=lambda x: -x[1])[:5],
        "timestamp": datetime.now().isoformat(),
        "reports": {
            "drift_html": str(settings.DRIFT_REPORTS_DIR / "drift_report.html"),
            "target_drift_html": str(settings.DRIFT_REPORTS_DIR / "target_drift_report.html"),
            "data_quality_html": str(settings.DRIFT_REPORTS_DIR / "data_quality_report.html"),
        },
    }
    path = settings.DRIFT_REPORTS_DIR / "drift_check_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    drift_history_path = settings.DRIFT_REPORTS_DIR / "drift_history.jsonl"
    with open(drift_history_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    if n_drift > 0:
        log.warning("DRIFT: %s/%s features (%.1f%%) severity=%s",
                     n_drift, len(features), drift_pct, drift_severity)
    else:
        log.info("No drift detected (PSI <= %.2f)", psi_threshold)
    return result


def main():
    result = detect_drift()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if not result["drift_detected"] else 1)


if __name__ == "__main__":
    import sys
    main()
