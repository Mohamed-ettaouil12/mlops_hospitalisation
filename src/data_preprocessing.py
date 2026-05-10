# ═══════════════════════════════════════════════════════════
# src/data_preprocessing.py
# Pipeline MLOps — Data Preprocessing + Split Temporel
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
CLEANED_DIR  = 'data/cleaned/'
FEATURES_DIR = 'data/features/'
MODELS_DIR   = 'models/'
OBS_DATE     = pd.Timestamp('2009-01-01')

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# FONCTION 1 : Features dynamiques (anti-leakage)
# ═══════════════════════════════════════════════════════════
def build_dynamic_features(df_patients, claims):
    log.info("Construction des features dynamiques (anti-leakage)...")

    pid = 'DESYNPUF_ID'
    results = []

    for annee in [2008, 2009, 2010]:
        obs_date = pd.Timestamp(f'{annee}-01-01')
        df_annee = df_patients[df_patients['ANNEE'] == annee].copy()

        # ── Inpatient : hospitalisations passées ───────────
        ip = claims['inpatient'].copy()
        ip_past = ip[ip['admission'] < obs_date]

        hosp_past = (
            ip_past.groupby(pid)
            .agg(NB_HOSP_PASSEES=('admission', 'count'))
            .reset_index()
        )

        # ── Outpatient : consultations externes ────────────
        op = claims['outpatient'].copy()

        def count_window(df, col, months):
            start = obs_date - pd.DateOffset(months=months)
            mask  = (df[col] >= start) & (df[col] < obs_date)
            return df[mask].groupby(pid).size().reset_index(
                name=f'NB_OP_{months}M'
            )

        op_3m  = count_window(op, 'CLM_FROM_DT', 3)
        op_6m  = count_window(op, 'CLM_FROM_DT', 6)
        op_12m = count_window(op, 'CLM_FROM_DT', 12)

        # ── Carrier : consultations médecins ───────────────
        car = claims['carrier'].copy()
        car_6m = (
            car[
                (car['CLM_FROM_DT'] >= obs_date - pd.DateOffset(months=6)) &
                (car['CLM_FROM_DT'] < obs_date)
            ]
            .groupby(pid).size().reset_index(name='NB_CAR_6M')
        )

        # ── Prescriptions ──────────────────────────────────
        rx = claims['prescription'].copy()
        rx_past = rx[rx['SRVC_DT'] < obs_date]

        rx_features = (
            rx_past.groupby(pid)
            .agg(NB_PRESCRIPTIONS=('SRVC_DT', 'count'))
            .reset_index()
        )

        if 'PROD_SRVC_ID' in rx_past.columns:
            mol = (
                rx_past.groupby(pid)['PROD_SRVC_ID']
                .nunique().reset_index(name='NB_MOLECULES_UNIQUES')
            )
        else:
            mol = rx_features[['DESYNPUF_ID']].copy()
            mol['NB_MOLECULES_UNIQUES'] = 0

        # ── Fusion pour cette année ────────────────────────
        df_a = df_annee.copy()
        for feat_df in [hosp_past, op_3m, op_6m, op_12m, car_6m, rx_features, mol]:
            df_a = df_a.merge(feat_df, on=pid, how='left')

        results.append(df_a)

    # Concaténer les 3 années
    df = pd.concat(results, ignore_index=True)

    # Remplir NaN par 0
    dynamic_cols = [
        'NB_HOSP_PASSEES', 'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
        'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES'
    ]
    for col in dynamic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Cold start
    df['IS_NEW_PATIENT'] = (
        (df.get('NB_HOSP_PASSEES', pd.Series([0]*len(df))) == 0) &
        (df.get('NB_OP_12M',       pd.Series([0]*len(df))) == 0) &
        (df.get('NB_PRESCRIPTIONS',pd.Series([0]*len(df))) == 0)
    ).astype(int)

    n_new = df['IS_NEW_PATIENT'].sum()
    log.info(f"  → Nouveaux patients sans historique : {n_new:,}")
    return df


# ═══════════════════════════════════════════════════════════
# FONCTION 2 : Features composites
# ═══════════════════════════════════════════════════════════
def build_composite_features(df):
    log.info("Construction des features composites...")

    chronic_cols = [c for c in df.columns if c.startswith('SP_')]
    df['NB_COMORBIDITES'] = df[chronic_cols].sum(axis=1)

    charlson_weights = {
        'SP_CHF': 1, 'SP_DIABETES': 1, 'SP_CHRNKIDN': 2,
        'SP_CNCR': 2, 'SP_COPD': 1, 'SP_STRKETIA': 2,
        'SP_ALZHDMTA': 1, 'SP_DEPRESSN': 1, 'SP_ISCHMCHT': 1,
        'SP_OSTEOPRS': 0, 'SP_RA_OA': 1
    }
    df['CHARLSON_INDEX'] = sum(
        df[col] * weight
        for col, weight in charlson_weights.items()
        if col in df.columns
    )

    if 'NB_MOLECULES_UNIQUES' in df.columns:
        df['POLYPHARMACIE'] = (df['NB_MOLECULES_UNIQUES'] > 5).astype(int)

    df['GROUPE_AGE'] = pd.cut(
        df['AGE'],
        bins=[0, 65, 75, 85, 120],
        labels=['<65', '65-74', '75-84', '85+']
    )

    log.info(f"  NB_COMORBIDITES moyen : {df['NB_COMORBIDITES'].mean():.2f}")
    log.info(f"  CHARLSON_INDEX moyen  : {df['CHARLSON_INDEX'].mean():.2f}")
    return df


# ═══════════════════════════════════════════════════════════
# FONCTION 3 : Encodage + Normalisation
# ═══════════════════════════════════════════════════════════
def encode_and_scale(df):
    log.info("Encodage et normalisation...")

    # Correction BENE_ESRD_IND
    df['BENE_ESRD_IND'] = df['BENE_ESRD_IND'].map(
        {'Y': 1, '0': 0, 0: 0, 1: 1}
    ).fillna(0)

    # Encodage
    le_sexe = LabelEncoder()
    df['SEXE_ENC'] = le_sexe.fit_transform(df['SEXE'].fillna('Inconnu'))

    le_race = LabelEncoder()
    df['RACE_ENC'] = le_race.fit_transform(df['RACE'].fillna('Autre'))

    age_map = {'<65': 0, '65-74': 1, '75-84': 2, '85+': 3}
    df['GROUPE_AGE_ENC'] = df['GROUPE_AGE'].map(age_map).fillna(1)

    # Normalisation
    cols_to_scale = [
        'AGE', 'COUT_TOTAL', 'CHARLSON_INDEX', 'NB_COMORBIDITES',
        'NB_HOSP_PASSEES', 'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
        'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES'
    ]
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    log.info(f"  → Scaler sauvegardé dans {MODELS_DIR}scaler.pkl")

    return df_scaled, scaler


# ═══════════════════════════════════════════════════════════
# FONCTION 4 : Split Temporel Strict
# ═══════════════════════════════════════════════════════════
def split_and_save(df_scaled):
    log.info("Découpage temporel strict (2008/2009/2010)...")

    chronic_cols = [c for c in df_scaled.columns if c.startswith('SP_')]
    feature_cols = (
        chronic_cols +
        ['AGE', 'SEXE_ENC', 'RACE_ENC', 'BENE_ESRD_IND',
         'GROUPE_AGE_ENC', 'NB_COMORBIDITES', 'CHARLSON_INDEX',
         'COUT_TOTAL', 'IS_NEW_PATIENT'] +
        [c for c in [
            'NB_HOSP_PASSEES', 'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
            'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES',
            'POLYPHARMACIE'
         ] if c in df_scaled.columns]
    )
    feature_cols = [c for c in feature_cols if c in df_scaled.columns]
    target_col   = 'HOSPITALIZED_IN_6M'

    X = df_scaled[feature_cols]
    y = df_scaled[target_col]

    # ── Split temporel strict ─────────────────────────────
    X_train = X[df_scaled['ANNEE'] == 2008]
    y_train = y[df_scaled['ANNEE'] == 2008]

    X_val   = X[df_scaled['ANNEE'] == 2009]
    y_val   = y[df_scaled['ANNEE'] == 2009]

    X_test  = X[df_scaled['ANNEE'] == 2010]
    y_test  = y[df_scaled['ANNEE'] == 2010]

    log.info(f"  Train (2008) : {len(X_train):,} | Taux : {y_train.mean()*100:.2f}%")
    log.info(f"  Val   (2009) : {len(X_val):,}   | Taux : {y_val.mean()*100:.2f}%")
    log.info(f"  Test  (2010) : {len(X_test):,}  | Taux : {y_test.mean()*100:.2f}%")
    log.info(f"  Nombre de features : {len(feature_cols)}")

    # ── Sauvegarde ─────────────────────────────────────────
    X_train.to_parquet(os.path.join(FEATURES_DIR, 'X_train.parquet'), index=False)
    X_val.to_parquet(os.path.join(FEATURES_DIR, 'X_val.parquet'),     index=False)
    X_test.to_parquet(os.path.join(FEATURES_DIR, 'X_test.parquet'),   index=False)

    y_train.to_frame().to_parquet(os.path.join(FEATURES_DIR, 'y_train.parquet'), index=False)
    y_val.to_frame().to_parquet(os.path.join(FEATURES_DIR, 'y_val.parquet'),     index=False)
    y_test.to_frame().to_parquet(os.path.join(FEATURES_DIR, 'y_test.parquet'),   index=False)

    pd.Series(feature_cols).to_csv(
        os.path.join(FEATURES_DIR, 'feature_names.csv'), index=False
    )

    log.info(f"  → Données sauvegardées dans {FEATURES_DIR}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("="*60)
    log.info("PIPELINE DATA PREPROCESSING — DÉBUT")
    log.info("="*60)

    # 1. Charger les données nettoyées
    log.info("Chargement des données nettoyées...")
    df_patients = pd.read_parquet(
        os.path.join(CLEANED_DIR, 'patients_cleaned.parquet')
    )
    log.info(f"  → {len(df_patients):,} patients | Années : {sorted(df_patients['ANNEE'].unique())}")

    # 2. Charger les claims
    claims = {}
    for key in ['inpatient', 'outpatient', 'carrier', 'prescription']:
        path = os.path.join(CLEANED_DIR, f'claims_{key}_cleaned.parquet')
        if os.path.exists(path):
            claims[key] = pd.read_parquet(path)

    # 3. Features dynamiques (par année, anti-leakage)
    df = build_dynamic_features(df_patients, claims)

    # 4. Features composites
    df = build_composite_features(df)

    # 5. Encodage + Normalisation
    df_scaled, scaler = encode_and_scale(df)

    # 6. Split temporel + sauvegarde
    X_train, X_val, X_test, y_train, y_val, y_test, features = split_and_save(df_scaled)

    log.info("="*60)
    log.info("PIPELINE DATA PREPROCESSING — TERMINÉ ✅")
    log.info("="*60)

    return X_train, X_val, X_test, y_train, y_val, y_test, features


if __name__ == '__main__':
    main()