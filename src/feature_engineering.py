# ═══════════════════════════════════════════════════════════
# src/feature_engineering.py
# Pipeline MLOps — Feature Engineering
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
CLEANED_DIR  = 'data/cleaned/'
FEATURES_DIR = 'data/features/'
OBS_DATE     = pd.Timestamp('2009-01-01')

os.makedirs(FEATURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# FONCTION 1 : Features temporelles glissantes
# ═══════════════════════════════════════════════════════════
def build_temporal_features(claims):
    """
    Calcule les features dans des fenêtres temporelles
    STRICTEMENT avant OBS_DATE (anti-leakage).
    """
    log.info("Construction des features temporelles...")
    pid = 'DESYNPUF_ID'

    ip  = claims['inpatient'].copy()
    op  = claims['outpatient'].copy()
    car = claims['carrier'].copy()
    rx  = claims['prescription'].copy()

    results = []

    # ── Fenêtres : 3M, 6M, 12M avant OBS_DATE ─────────────
    for months in [3, 6, 12]:
        start = OBS_DATE - pd.DateOffset(months=months)

        # Hospitalisations dans la fenêtre
        ip_w = ip[
            (ip['admission'] >= start) &
            (ip['admission'] < OBS_DATE)
        ].groupby(pid).size().reset_index(name=f'NB_HOSP_{months}M')

        # Consultations externes dans la fenêtre
        op_w = op[
            (op['CLM_FROM_DT'] >= start) &
            (op['CLM_FROM_DT'] < OBS_DATE)
        ].groupby(pid).size().reset_index(name=f'NB_OP_{months}M')

        # Consultations médecins dans la fenêtre
        car_w = car[
            (car['CLM_FROM_DT'] >= start) &
            (car['CLM_FROM_DT'] < OBS_DATE)
        ].groupby(pid).size().reset_index(name=f'NB_CAR_{months}M')

        # Prescriptions dans la fenêtre
        rx_w = rx[
            (rx['SRVC_DT'] >= start) &
            (rx['SRVC_DT'] < OBS_DATE)
        ].groupby(pid).size().reset_index(name=f'NB_RX_{months}M')

        results.extend([ip_w, op_w, car_w, rx_w])

    log.info(f"  → {len(results)} features temporelles construites")
    return results


# ═══════════════════════════════════════════════════════════
# FONCTION 2 : Features de coûts
# ═══════════════════════════════════════════════════════════
def build_cost_features(claims):
    """Coûts cumulés avant OBS_DATE par type de soin."""
    log.info("Construction des features de coûts...")
    pid = 'DESYNPUF_ID'

    ip  = claims['inpatient'].copy()
    op  = claims['outpatient'].copy()
    car = claims['carrier'].copy()

    ip_past  = ip[ip['admission'] < OBS_DATE]
    op_past  = op[op['CLM_FROM_DT'] < OBS_DATE]
    car_past = car[car['CLM_FROM_DT'] < OBS_DATE]

    cost_features = []

    # Coût hospitalier total passé
    if 'CLM_PMT_AMT' in ip_past.columns:
        ip_cost = ip_past.groupby(pid)['CLM_PMT_AMT'].sum().reset_index(
            name='COUT_HOSP_PASSE'
        )
        cost_features.append(ip_cost)

    # Coût ambulatoire total passé
    if 'CLM_PMT_AMT' in op_past.columns:
        op_cost = op_past.groupby(pid)['CLM_PMT_AMT'].sum().reset_index(
            name='COUT_OP_PASSE'
        )
        cost_features.append(op_cost)

    log.info(f"  → {len(cost_features)} features de coûts construites")
    return cost_features


# ═══════════════════════════════════════════════════════════
# FONCTION 3 : Features de prescriptions
# ═══════════════════════════════════════════════════════════
def build_prescription_features(claims):
    """Nombre de molécules uniques et polypharmacie."""
    log.info("Construction des features de prescriptions...")
    pid = 'DESYNPUF_ID'

    rx = claims['prescription'].copy()
    rx_past = rx[rx['SRVC_DT'] < OBS_DATE]

    rx_feats = []

    if 'PROD_SRVC_ID' in rx_past.columns:
        # Nombre de molécules uniques
        mol = rx_past.groupby(pid)['PROD_SRVC_ID'].nunique().reset_index(
            name='NB_MOLECULES_UNIQUES'
        )
        rx_feats.append(mol)

    # Nombre total de prescriptions
    nb_rx = rx_past.groupby(pid).size().reset_index(name='NB_PRESCRIPTIONS_TOTAL')
    rx_feats.append(nb_rx)

    log.info(f"  → {len(rx_feats)} features prescriptions construites")
    return rx_feats


# ═══════════════════════════════════════════════════════════
# FONCTION 4 : Fusion de toutes les features
# ═══════════════════════════════════════════════════════════
def merge_all_features(df_patients, all_feature_dfs):
    """Fusionne toutes les features sur le DataFrame patients."""
    log.info("Fusion de toutes les features...")

    df = df_patients.copy()
    pid = 'DESYNPUF_ID'

    for feat_df in all_feature_dfs:
        df = df.merge(feat_df, on=pid, how='left')

    # Remplir les NaN par 0 (patients sans historique)
    feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ['NB_', 'COUT_', 'POLYPHARMACIE', 'IS_NEW']
    )]
    df[feature_cols] = df[feature_cols].fillna(0)

    # Polypharmacie : > 5 molécules uniques
    if 'NB_MOLECULES_UNIQUES' in df.columns:
        df['POLYPHARMACIE'] = (df['NB_MOLECULES_UNIQUES'] > 5).astype(int)

    # IS_NEW_PATIENT : aucun soin avant OBS_DATE
    df['IS_NEW_PATIENT'] = (
        (df.get('NB_HOSP_12M', pd.Series([0]*len(df))) == 0) &
        (df.get('NB_OP_12M',   pd.Series([0]*len(df))) == 0) &
        (df.get('NB_RX_12M',   pd.Series([0]*len(df))) == 0)
    ).astype(int)

    log.info(f"  → DataFrame final : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("="*60)
    log.info("FEATURE ENGINEERING — DÉBUT")
    log.info("="*60)

    # Charger données nettoyées
    df_patients = pd.read_parquet(os.path.join(CLEANED_DIR, 'patients_cleaned.parquet'))
    log.info(f"Patients chargés : {len(df_patients):,}")

    claims = {}
    for key in ['inpatient', 'outpatient', 'carrier', 'prescription']:
        path = os.path.join(CLEANED_DIR, f'claims_{key}_cleaned.parquet')
        if os.path.exists(path):
            claims[key] = pd.read_parquet(path)
            log.info(f"  {key} : {len(claims[key]):,} lignes")

    # Construire toutes les features
    temporal_feats    = build_temporal_features(claims)
    cost_feats        = build_cost_features(claims)
    prescription_feats = build_prescription_features(claims)

    all_feats = temporal_feats + cost_feats + prescription_feats

    # Fusionner
    df_final = merge_all_features(df_patients, all_feats)

    # Sauvegarder
    output_path = os.path.join(FEATURES_DIR, 'features_engineered.parquet')
    df_final.to_parquet(output_path, index=False)
    log.info(f"\n✅ Features sauvegardées → {output_path}")
    log.info(f"   Nombre de features : {df_final.shape[1]}")

    log.info("="*60)
    log.info("FEATURE ENGINEERING — TERMINÉ ✅")
    log.info("="*60)

    return df_final


if __name__ == '__main__':
    main()