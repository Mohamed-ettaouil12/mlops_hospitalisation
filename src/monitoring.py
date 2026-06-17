import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets.drift import DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
OUTPUT_DIR = Path("outputs/reports")
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


def main():
    log.info("=" * 60)
    log.info("MONITORING — DÉTECTION DE DRIFT (PSI)")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_feature_names()
    log.info("Features chargées : %s", len(features))

    df_ref, df_prod = load_and_prepare()
    log.info("Référence (2008) : %s lignes", f"{len(df_ref):,}")
    log.info("Production (2010) : %s lignes", f"{len(df_prod):,}")

    psi_results = {}
    has_drift = False
    for col in features:
        psi_val = calc_psi(df_ref[col], df_prod[col])
        psi_results[col] = psi_val
        if psi_val > PSI_THRESHOLD:
            log.warning("⚠️  DRIFT %s — PSI = %.4f (> %.2f)", col, psi_val, PSI_THRESHOLD)
            has_drift = True
        else:
            log.info("✓ %s — PSI = %.4f", col, psi_val)

    log.info("Génération du rapport Evidently...")
    preset = DataDriftPreset(columns=features, num_method="psi")
    report = Report(metrics=[preset])
    snapshot = report.run(current_data=df_prod, reference_data=df_ref)
    report_path = OUTPUT_DIR / "drift_report.html"
    snapshot.save_html(str(report_path))
    log.info("Rapport sauvegardé → %s", report_path)

    if has_drift:
        log.warning("=" * 60)
        log.warning("DRIFT DÉTECTÉ — PSI > %.2f sur au moins une feature", PSI_THRESHOLD)
        log.warning("Arrêt du pipeline (exit 1)")
        sys.exit(1)

    log.info("Aucun drift détecté — PSI ≤ %.2f sur toutes les features", PSI_THRESHOLD)
    log.info("MONITORING TERMINÉ")


if __name__ == "__main__":
    main()
