# ═══════════════════════════════════════════════════════════
# ÉTAPE 4 — Data Cleaning
# Pipeline MLOps — Risque d'Hospitalisation Medicare
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaning.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
DATA_DIR   = '/home/tawil/Bureau/pfa/data set/sample 1/'
OUTPUT_DIR = 'data/cleaned/'
SAMPLE_NUM = 1
NB_PATIENTS = 30_000
RANDOM_STATE = 42
OBS_DATE     = pd.Timestamp('2009-01-01')   # date d'observation
HORIZON_MOIS = 6                             # fenêtre de prédiction

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ═══════════════════════════════════════════════════════════
# FONCTION 1 : Correction des dates
# ═══════════════════════════════════════════════════════════
def clean_and_parse_date(series):
    """Corrige les dates au format YYYYMMDD.0 → datetime."""
    series = series.astype(str).str.strip()
    series = series.str.replace(r'\.0$', '', regex=True)
    series = series.replace(['', 'nan', 'None', 'NaT'], pd.NA)
    return pd.to_datetime(series, format='%Y%m%d', errors='coerce')


# ═══════════════════════════════════════════════════════════
# FONCTION 2 : Chargement et échantillonnage des patients
# ═══════════════════════════════════════════════════════════
def load_patients(year=2008):
    log.info(f"Chargement Beneficiary {year}...")
    fname = f'DE1_0_{year}_Beneficiary_Summary_File_Sample_{SAMPLE_NUM}.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, fname), low_memory=False)

    # Échantillonnage reproductible
    df = df.sample(n=min(NB_PATIENTS, len(df)), random_state=RANDOM_STATE)
    log.info(f"  → {len(df):,} patients chargés")
    return df


# ═══════════════════════════════════════════════════════════
# FONCTION 3 : Chargement des claims filtrés sur les IDs
# ═══════════════════════════════════════════════════════════
def load_claims(patient_ids):
    log.info("Chargement des claims (filtrés sur les 30K patients)...")

    date_cols = ['CLM_FROM_DT', 'CLM_THRU_DT', 'CLM_ADMSN_DT',
                 'NCH_BENE_DSCHRG_DT', 'SRVC_DT']

    files = {
        'inpatient'   : f'DE1_0_2008_to_2010_Inpatient_Claims_Sample_{SAMPLE_NUM}.csv',
        'outpatient'  : f'DE1_0_2008_to_2010_Outpatient_Claims_Sample_{SAMPLE_NUM}.csv',
        'carrier_A'   : f'DE1_0_2008_to_2010_Carrier_Claims_Sample_{SAMPLE_NUM}A.csv',
        'carrier_B'   : f'DE1_0_2008_to_2010_Carrier_Claims_Sample_{SAMPLE_NUM}B.csv',
        'prescription': f'DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_{SAMPLE_NUM}.csv',
    }

    claims = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)

        # Lire uniquement les colonnes de date en str
        with open(path) as f:
            header = f.readline().strip().split(',')
        dtype_dict = {col: str for col in date_cols if col in header}

        df = pd.read_csv(path, dtype=dtype_dict, low_memory=False)
        df = df[df['DESYNPUF_ID'].isin(patient_ids)]
        claims[key] = df
        log.info(f"  {key}: {len(df):,} lignes")

    # Fusionner carrier A + B
    claims['carrier'] = pd.concat(
        [claims.pop('carrier_A'), claims.pop('carrier_B')],
        ignore_index=True
    )

    return claims


# ═══════════════════════════════════════════════════════════
# FONCTION 4 : Conversion des dates dans tous les fichiers
# ═══════════════════════════════════════════════════════════
def convert_all_dates(ben08, ben09, ben10, claims):
    log.info("Conversion des dates...")

    date_cols_map = {
        'ben08'       : ['BENE_BIRTH_DT', 'BENE_DEATH_DT'],
        'ben09'       : ['BENE_BIRTH_DT', 'BENE_DEATH_DT'],
        'ben10'       : ['BENE_BIRTH_DT', 'BENE_DEATH_DT'],
        'inpatient'   : ['CLM_FROM_DT', 'CLM_THRU_DT', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT'],
        'outpatient'  : ['CLM_FROM_DT', 'CLM_THRU_DT'],
        'carrier'     : ['CLM_FROM_DT', 'CLM_THRU_DT'],
        'prescription': ['SRVC_DT'],
    }

    tables = {
        'ben08': ben08, 'ben09': ben09, 'ben10': ben10,
        **{k: claims[k] for k in ['inpatient', 'outpatient', 'carrier', 'prescription']}
    }

    for name, df in tables.items():
        for col in date_cols_map[name]:
            if col in df.columns:
                df[col] = clean_and_parse_date(df[col])

    # Colonne admission unifiée pour inpatient
    claims['inpatient']['admission'] = (
        claims['inpatient']['CLM_ADMSN_DT']
        .fillna(claims['inpatient']['CLM_FROM_DT'])
    )

    log.info("  → Dates converties avec succès")
    return ben08, ben09, ben10, claims


# ═══════════════════════════════════════════════════════════
# FONCTION 5 : Construction de la cible HOSPITALIZED_IN_6M
#              ⚠️ Anti-leakage strict : uniquement admissions
#              urgentes (code 1 ou 2) APRÈS OBS_DATE
# ═══════════════════════════════════════════════════════════
def build_target(ben08, claims):
    log.info(f"Construction de la cible (OBS_DATE={OBS_DATE.date()}, horizon={HORIZON_MOIS}M)...")

    ip = claims['inpatient'].copy()
    future_date = OBS_DATE + pd.DateOffset(months=HORIZON_MOIS)

    # Filtrer : admissions dans la fenêtre future
    mask_window = (ip['admission'] >= OBS_DATE) & (ip['admission'] <= future_date)

    # Exclure les hospitalisations programmées (garder urgence=1, urgente=2)
    if 'CLM_IP_ADMSN_TYPE_CD' in ip.columns:
        mask_urgence = ip['CLM_IP_ADMSN_TYPE_CD'].isin([1, 2])
        future_ip = ip[mask_window & mask_urgence]
        log.info("  → Hospitalisations programmées exclues")
    else:
        future_ip = ip[mask_window]
        log.info("  → Colonne admission type absente, toutes admissions conservées")

    hosp_ids = set(future_ip['DESYNPUF_ID'])
    ben08['HOSPITALIZED_IN_6M'] = ben08['DESYNPUF_ID'].isin(hosp_ids).astype(int)

    taux = ben08['HOSPITALIZED_IN_6M'].mean() * 100
    log.info(f"  → {len(hosp_ids):,} patients hospitalisés | Taux : {taux:.2f}%")
    return ben08


# ═══════════════════════════════════════════════════════════
# FONCTION 6 : Nettoyage principal
# ═══════════════════════════════════════════════════════════
def clean_patients(ben08):
    log.info("Nettoyage des données patients...")
    df = ben08.copy()

    # ── Colonnes utiles ────────────────────────────────────
    demo_cols    = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT',
                    'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_ESRD_IND']
    chronic_cols = [c for c in df.columns if c.startswith('SP_')]
    cost_cols    = [c for c in df.columns if any(k in c for k in ['MEDREIMB', 'BENRES', 'PPPYMT'])]
    target_col   = ['HOSPITALIZED_IN_6M']

    df = df[demo_cols + chronic_cols + cost_cols + target_col].copy()

    # ── 1. Suppression des doublons ────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset='DESYNPUF_ID', keep='first')
    log.info(f"  Doublons supprimés : {n_before - len(df)}")

    # ── 2. Calcul de l'âge ────────────────────────────────
    ref_date = datetime(2009, 1, 1)
    df['AGE'] = df['BENE_BIRTH_DT'].apply(
        lambda x: (ref_date - x).days / 365.25 if pd.notnull(x) else np.nan
    )

    # ── 3. Valeurs manquantes ──────────────────────────────
    # Comorbidités : 1=oui, 2=non dans le dataset → convertir en 0/1
    for col in chronic_cols:
        df[col] = df[col].fillna(2)         # manquant → supposé absent
        df[col] = (df[col] == 1).astype(int)

    # Coûts manquants → 0
    for col in cost_cols:
        df[col] = df[col].fillna(0)

    # Âge manquant → médiane
    age_median = df['AGE'].median()
    df['AGE'] = df['AGE'].fillna(age_median)
    log.info(f"  Âge médian utilisé pour imputation : {age_median:.1f} ans")

    # ── 4. Valeurs aberrantes sur l'âge ───────────────────
    n_age_invalid = ((df['AGE'] < 0) | (df['AGE'] > 120)).sum()
    df = df[(df['AGE'] >= 0) & (df['AGE'] <= 120)]
    log.info(f"  Âges aberrants supprimés : {n_age_invalid}")

    # ── 5. Winsorisation des coûts (99e percentile) ────────
    for col in cost_cols:
        p99 = df[col].quantile(0.99)
        n_outliers = (df[col] > p99).sum()
        df[col] = df[col].clip(upper=p99)
        if n_outliers > 0:
            log.info(f"  Winsorisation {col} : {n_outliers} valeurs plafonnées à {p99:.0f}")

    # ── 6. Cold start : patients sans historique ──────────
    # (pas d'hospitalisations antérieures à OBS_DATE)
    df['IS_NEW_PATIENT'] = 0   # sera mis à jour dans feature_engineering.py

    # ── 7. Encodage sexe et race ──────────────────────────
    df['SEXE'] = df['BENE_SEX_IDENT_CD'].map({1: 'Homme', 2: 'Femme'})
    race_map = {1:'Blanc', 2:'Noir', 3:'Autre', 4:'Asiatique', 5:'Hispanique', 6:'Natif Amér.'}
    df['RACE'] = df['BENE_RACE_CD'].map(race_map)

    log.info(f"  → Table finale : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("="*60)
    log.info("PIPELINE DATA CLEANING — DÉBUT")
    log.info("="*60)

    # 1. Charger les patients de référence (2008)
    ben08 = load_patients(2008)
    patient_ids = set(ben08['DESYNPUF_ID'])

    # 2. Charger les bénéficiaires 2009/2010 filtrés
    fname09 = f'DE1_0_2009_Beneficiary_Summary_File_Sample_{SAMPLE_NUM}.csv'
    fname10 = f'DE1_0_2010_Beneficiary_Summary_File_Sample_{SAMPLE_NUM}.csv'
    ben09 = pd.read_csv(os.path.join(DATA_DIR, fname09), low_memory=False)
    ben10 = pd.read_csv(os.path.join(DATA_DIR, fname10), low_memory=False)
    ben09 = ben09[ben09['DESYNPUF_ID'].isin(patient_ids)]
    ben10 = ben10[ben10['DESYNPUF_ID'].isin(patient_ids)]
    log.info(f"Ben09 : {len(ben09):,} | Ben10 : {len(ben10):,}")

    # 3. Charger les claims
    claims = load_claims(patient_ids)

    # 4. Convertir les dates
    ben08, ben09, ben10, claims = convert_all_dates(ben08, ben09, ben10, claims)

    # 5. Construire la cible (anti-leakage)
    ben08 = build_target(ben08, claims)

    # 6. Nettoyer
    df_clean = clean_patients(ben08)

    # 7. Sauvegarder
    output_path = os.path.join(OUTPUT_DIR, 'patients_cleaned.parquet')
    df_clean.to_parquet(output_path, index=False)
    log.info(f"\n✅ Données sauvegardées → {output_path}")

    # Sauvegarder aussi les claims nettoyés
    for key, df in claims.items():
        path = os.path.join(OUTPUT_DIR, f'claims_{key}_cleaned.parquet')
        df.to_parquet(path, index=False)
        log.info(f"  Claims {key} → {path}")

    log.info("="*60)
    log.info("PIPELINE DATA CLEANING — TERMINÉ ✅")
    log.info("="*60)

    return df_clean, claims


if __name__ == '__main__':
    df_clean, claims = main()