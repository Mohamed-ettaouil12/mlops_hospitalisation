# ═══════════════════════════════════════════════════════════
# ÉTAPE 5 — Data Preprocessing + Feature Engineering
# Pipeline MLOps — Risque d'Hospitalisation Medicare
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── Logging ────────────────────────────────────────────────
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
# FONCTION 1 : Features dynamiques à partir des claims
#              ⚠️ Anti-leakage : uniquement données AVANT OBS_DATE
# ═══════════════════════════════════════════════════════════
def build_dynamic_features(df_patients, claims):
    log.info("Construction des features dynamiques (anti-leakage)...")

    pid = 'DESYNPUF_ID'

    # ── Inpatient : hospitalisations passées ───────────────
    ip = claims['inpatient'].copy()
    ip_past = ip[ip['admission'] < OBS_DATE]

    hosp_past = (
        ip_past.groupby(pid)
        .agg(
            NB_HOSP_PASSEES=('admission', 'count'),
            COUT_HOSP_TOTAL=('CLM_PMT_AMT', 'sum') if 'CLM_PMT_AMT' in ip.columns else ('CLM_FROM_DT', 'count')
        )
        .reset_index()
    )

    # ── Outpatient : consultations externes ────────────────
    op = claims['outpatient'].copy()
    op_past = op[op['CLM_FROM_DT'] < OBS_DATE]

    # Fenêtres temporelles : 3M, 6M, 12M avant OBS_DATE
    def count_in_window(df, date_col, months):
        start = OBS_DATE - pd.DateOffset(months=months)
        mask = (df[date_col] >= start) & (df[date_col] < OBS_DATE)
        return df[mask].groupby(pid).size().reset_index(name=f'NB_OP_{months}M')

    op_3m  = count_in_window(op, 'CLM_FROM_DT', 3)
    op_6m  = count_in_window(op, 'CLM_FROM_DT', 6)
    op_12m = count_in_window(op, 'CLM_FROM_DT', 12)

    # ── Carrier : consultations médecins ───────────────────
    car = claims['carrier'].copy()
    car_past = car[car['CLM_FROM_DT'] < OBS_DATE]

    car_6m = (
        car_past[car_past['CLM_FROM_DT'] >= OBS_DATE - pd.DateOffset(months=6)]
        .groupby(pid).size().reset_index(name='NB_CAR_6M')
    )

    # ── Prescriptions ──────────────────────────────────────
    rx = claims['prescription'].copy()
    rx_past = rx[rx['SRVC_DT'] < OBS_DATE]

    rx_features = (
        rx_past.groupby(pid)
        .agg(
            NB_PRESCRIPTIONS=('SRVC_DT', 'count'),
            NB_MOLECULES_UNIQUES=('PROD_SRVC_ID', 'nunique') if 'PROD_SRVC_ID' in rx.columns else ('SRVC_DT', 'count')
        )
        .reset_index()
    )

    # ── Coût total 6 derniers mois ─────────────────────────
    cost_cols = [c for c in df_patients.columns if 'MEDREIMB' in c]
    df_patients['COUT_TOTAL'] = df_patients[cost_cols].sum(axis=1)

    # ── Fusion de toutes les features dynamiques ───────────
    log.info("  Fusion des features dynamiques...")
    df = df_patients.copy()
    for feat_df in [hosp_past, op_3m, op_6m, op_12m, car_6m, rx_features]:
        df = df.merge(feat_df, on=pid, how='left')

    # Remplir les patients sans historique par 0
    dynamic_cols = ['NB_HOSP_PASSEES', 'COUT_HOSP_TOTAL',
                    'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
                    'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES']
    for col in dynamic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Cold start : patients sans aucun historique ────────
    df['IS_NEW_PATIENT'] = (
        (df.get('NB_HOSP_PASSEES', 0) == 0) &
        (df.get('NB_OP_12M', 0) == 0) &
        (df.get('NB_PRESCRIPTIONS', 0) == 0)
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

    # Nombre total de comorbidités
    df['NB_COMORBIDITES'] = df[chronic_cols].sum(axis=1)

    # Index de Charlson simplifié (pondération des comorbidités)
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

    # Polypharmacie : > 5 molécules uniques
    if 'NB_MOLECULES_UNIQUES' in df.columns:
        df['POLYPHARMACIE'] = (df['NB_MOLECULES_UNIQUES'] > 5).astype(int)

    # Groupe d'âge
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

    # ── Encodage des variables catégorielles ───────────────
    le_sexe = LabelEncoder()
    df['SEXE_ENC'] = le_sexe.fit_transform(df['SEXE'].fillna('Inconnu'))

    le_race = LabelEncoder()
    df['RACE_ENC'] = le_race.fit_transform(df['RACE'].fillna('Autre'))

    # Encodage GROUPE_AGE
    age_map = {'<65': 0, '65-74': 1, '75-84': 2, '85+': 3}
    df['GROUPE_AGE_ENC'] = df['GROUPE_AGE'].map(age_map).fillna(1)

    # ── Colonnes à normaliser ─────────────────────────────
    cols_to_scale = ['AGE', 'COUT_TOTAL', 'CHARLSON_INDEX',
                     'NB_COMORBIDITES', 'NB_HOSP_PASSEES',
                     'NB_OP_3M', 'NB_OP_6M', 'NB_OP_12M',
                     'NB_CAR_6M', 'NB_PRESCRIPTIONS', 'NB_MOLECULES_UNIQUES']

    cols_to_scale = [c for c in cols_to_scale if c in df.columns]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Sauvegarder le scaler pour l'inférence en production
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    log.info(f"  → Scaler sauvegardé dans {MODELS_DIR}scaler.pkl")

    return df_scaled, scaler


# ═══════════════════════════════════════════════════════════
# FONCTION 4 : Sélection des features finales + Train/Test split
#              ⚠️ Validation temporelle stricte (pas de K-Fold)
# ═══════════════════════════════════════════════════════════
def split_and_save(df_scaled):
    log.info("Découpage Train / Validation / Test (temporel)...")

    # Features finales
    chronic_cols = [c for c in df_scaled.columns if c.startswith('SP_')]
    feature_cols = (
        chronic_cols +
        ['AGE', 'SEXE_ENC', 'RACE_ENC', 'BENE_ESRD_IND',
         'GROUPE_AGE_ENC', 'NB_COMORBIDITES', 'CHARLSON_INDEX',
         'COUT_TOTAL', 'IS_NEW_PATIENT'] +
        [c for c in ['NB_HOSP_PASSEES', 'NB_OP_3M', 'NB_OP_6M',
                     'NB_OP_12M', 'NB_CAR_6M', 'NB_PRESCRIPTIONS',
                     'NB_MOLECULES_UNIQUES', 'POLYPHARMACIE']
         if c in df_scaled.columns]
    )

    feature_cols = [c for c in feature_cols if c in df_scaled.columns]
    target_col   = 'HOSPITALIZED_IN_6M'

    X = df_scaled[feature_cols]
    y = df_scaled[target_col]

    # ── Découpage 80% Train / 20% Test (stratifié sur cible) ──
    # Note : en production réelle, le split sera temporel (2008/2009/2010)
    # Ici sur 30K patients d'une même année → split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    log.info(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")
    log.info(f"  Taux hosp. Train : {y_train.mean()*100:.2f}% | Test : {y_test.mean()*100:.2f}%")
    log.info(f"  Nombre de features : {len(feature_cols)}")

    # ── Sauvegarde ─────────────────────────────────────────
    X_train.to_parquet(os.path.join(FEATURES_DIR, 'X_train.parquet'), index=False)
    X_test.to_parquet(os.path.join(FEATURES_DIR, 'X_test.parquet'), index=False)
    y_train.to_frame().to_parquet(os.path.join(FEATURES_DIR, 'y_train.parquet'), index=False)
    y_test.to_frame().to_parquet(os.path.join(FEATURES_DIR, 'y_test.parquet'), index=False)

    # Sauvegarder la liste des features
    pd.Series(feature_cols).to_csv(os.path.join(FEATURES_DIR, 'feature_names.csv'), index=False)

    log.info(f"  → Données sauvegardées dans {FEATURES_DIR}")
    return X_train, X_test, y_train, y_test, feature_cols


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("="*60)
    log.info("PIPELINE DATA PREPROCESSING — DÉBUT")
    log.info("="*60)

    # 1. Charger les données nettoyées
    log.info("Chargement des données nettoyées...")
    df_patients = pd.read_parquet(os.path.join(CLEANED_DIR, 'patients_cleaned.parquet'))
    log.info(f"  → {len(df_patients):,} patients")

    claims = {}
    for key in ['inpatient', 'outpatient', 'carrier', 'prescription']:
        path = os.path.join(CLEANED_DIR, f'claims_{key}_cleaned.parquet')
        if os.path.exists(path):
            claims[key] = pd.read_parquet(path)

    # 2. Features dynamiques
    df = build_dynamic_features(df_patients, claims)

    # 3. Features composites
    df = build_composite_features(df)

    # 4. Encodage + Normalisation
    df_scaled, scaler = encode_and_scale(df)

    # 5. Split et sauvegarde
    X_train, X_test, y_train, y_test, features = split_and_save(df_scaled)

    log.info("="*60)
    log.info("PIPELINE DATA PREPROCESSING — TERMINÉ ✅")
    log.info("="*60)

    return X_train, X_test, y_train, y_test, features


if __name__ == '__main__':
    main()