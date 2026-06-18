import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
PSI_THRESHOLD = 0.20


def safe_numeric(series, fill_value=0):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
    )


def load_feature_names():
    return pd.read_csv(FEATURES_DIR / "feature_names.csv")["feature"].tolist()


def load_and_prepare():
    df = pd.read_parquet(FEATURES_DIR / "features_engineered.parquet")

    df["SEXE_ENC"] = safe_numeric(
        df["BENE_SEX_IDENT_CD"]
        if "BENE_SEX_IDENT_CD" in df.columns
        else pd.Series(0, index=df.index)
    )
    df["RACE_ENC"] = safe_numeric(
        df["BENE_RACE_CD"]
        if "BENE_RACE_CD" in df.columns
        else pd.Series(0, index=df.index)
    )
    df["BENE_ESRD_IND"] = safe_numeric(
        df.get("BENE_ESRD_IND", pd.Series(0, index=df.index))
    )
    df["GROUPE_AGE_ENC"] = (
        pd.cut(df["AGE"], bins=[0, 65, 75, 85, 120], labels=[0, 1, 2, 3], include_lowest=True)
        .astype(float)
        .fillna(1)
    )

    features = load_feature_names()
    df_ref = df[df["ANNEE"] == 2008][features].copy()
    df_prod = df[df["ANNEE"] == 2010][features].copy()

    for col in features:
        df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce").fillna(0).astype(float)
        df_prod[col] = pd.to_numeric(df_prod[col], errors="coerce").fillna(0).astype(float)

    return df_ref, df_prod


def calc_psi(expected, actual, n_bins=10):
    combined = pd.concat([expected, actual])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, n_bins + 1)
    exp_hist = np.histogram(expected, bins=bins)[0].astype(float) / len(expected)
    act_hist = np.histogram(actual, bins=bins)[0].astype(float) / len(actual)
    exp_hist = np.clip(exp_hist, 0.0001, 1)
    act_hist = np.clip(act_hist, 0.0001, 1)
    return float(np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist)))


def check_drift(psi_threshold: float = PSI_THRESHOLD) -> dict:
    log.info("=" * 60)
    log.info("CHECK DRIFT — DÉTECTION DE DÉRIVE (PSI)")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_feature_names()
    log.info("Features chargées : %s", len(features))

    df_ref, df_prod = load_and_prepare()
    log.info("Référence (2008) : %s lignes", f"{len(df_ref):,}")
    log.info("Production (2010) : %s lignes", f"{len(df_prod):,}")

    psi_results = {}
    has_drift = False
    drifted_features = []

    for col in features:
        psi_val = calc_psi(df_ref[col], df_prod[col])
        psi_results[col] = psi_val
        if psi_val > psi_threshold:
            log.warning("⚠ DRIFT %s — PSI = %.4f (> %.2f)", col, psi_val, psi_threshold)
            has_drift = True
            drifted_features.append(col)
        else:
            log.info("✓ %s — PSI = %.4f", col, psi_val)

    result = {
        "drift_detected": has_drift,
        "psi_threshold": psi_threshold,
        "n_features": len(features),
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "max_psi": max(psi_results.values()) if psi_results else 0.0,
        "psi_results": psi_results,
    }

    result_path = OUTPUT_DIR / "drift_check_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Résultat sauvegardé → %s", result_path)

    if has_drift:
        log.warning("DRIFT DÉTECTÉ — %s features au-dessus du seuil PSI=%.2f", len(drifted_features), psi_threshold)
    else:
        log.info("Aucun drift détecté — PSI ≤ %.2f sur toutes les features", psi_threshold)

    return result


def main():
    result = check_drift()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if not result["drift_detected"] else 1)


if __name__ == "__main__":
    main()
