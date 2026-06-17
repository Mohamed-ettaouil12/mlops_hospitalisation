# ═══════════════════════════════════════════════════════════
# MLOPS CLEANING — PRODUCTION SAFE FIXED VERSION (v2)
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import logging

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/data_cleaning.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "/home/tawil/Bureau/pfa/data set/sample 1/"
OUTPUT_DIR = "data/cleaned/"
SAMPLE_NUM = 1

HORIZON = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SAFE DATE PARSER
# ═══════════════════════════════════════════════════════════
def parse_date(col):
    col = col.astype(str).str.replace(r"\.0$", "", regex=True)
    col = col.replace(["nan", "None", "", "NaT"], pd.NA)
    return pd.to_datetime(col, format="%Y%m%d", errors="coerce")


# ═══════════════════════════════════════════════════════════
# LOAD PATIENTS
# ═══════════════════════════════════════════════════════════
def load_patients(year, patient_ids=None):
    file = f"DE1_0_{year}_Beneficiary_Summary_File_Sample_{SAMPLE_NUM}.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)

    if patient_ids is None:
        df = df.sample(min(30000, len(df)), random_state=42)
    else:
        df = df[df["DESYNPUF_ID"].isin(patient_ids)]

    df["ANNEE"] = year

    log.info(f"Patients {year}: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════
# SAFE CLAIMS LOADING (FIX ALL TYPES)
# ═══════════════════════════════════════════════════════════
def load_claims(patient_ids):

    files = {
        "inpatient": "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
        "outpatient": "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
        "carrier_A": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv",
        "carrier_B": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv",
        "prescription": "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv",
    }

    claims = {}

    for k, f in files.items():
        path = os.path.join(DATA_DIR, f)

        df = pd.read_csv(path, low_memory=False)
        df = df[df["DESYNPUF_ID"].isin(patient_ids)]

        # 🔥 CRITICAL FIX PYARROW ERROR
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype("string")

        claims[k] = df
        log.info(f"{k}: {len(df):,}")

    # merge carrier
    claims["carrier"] = pd.concat(
        [claims["carrier_A"], claims["carrier_B"]],
        ignore_index=True
    )

    claims.pop("carrier_A")
    claims.pop("carrier_B")

    return claims


# ═══════════════════════════════════════════════════════════
# TARGET (FIX TIMESTAMP BUG + DIAGNOSTIC NaT)
# ═══════════════════════════════════════════════════════════
def build_target(df, claims, obs_date):

    ip = claims["inpatient"].copy()

    obs_date = pd.to_datetime(obs_date)
    future = obs_date + pd.DateOffset(months=int(HORIZON))

    # 🔍 DIAGNOSTIC : combien de dates d'admission sont NaT après parsing ?
    n_nat = ip["admission"].isna().sum()
    n_total = len(ip)
    if n_nat > 0:
        log.warning(
            f"build_target({obs_date.date()}) : {n_nat:,}/{n_total:,} "
            f"dates d'admission NaT (seront exclues du filtre temporel)"
        )

    mask = (ip["admission"] > obs_date) & (ip["admission"] <= future)

    if "CLM_IP_ADMSN_TYPE_CD" in ip.columns:
        mask = mask & ip["CLM_IP_ADMSN_TYPE_CD"].isin([1, 2])

    n_matched_claims = mask.sum()
    ids = ip.loc[mask, "DESYNPUF_ID"].unique()

    log.info(
        f"build_target({obs_date.date()}) : fenêtre [{obs_date.date()} -> "
        f"{future.date()}] | {n_matched_claims:,} admissions urgentes matchées "
        f"| {len(ids):,} patients uniques"
    )

    df["HOSPITALIZED_IN_6M"] = df["DESYNPUF_ID"].isin(ids).astype(int)

    log.info(f"Target rate: {df['HOSPITALIZED_IN_6M'].mean():.3f}")

    return df


# ═══════════════════════════════════════════════════════════
# CLEAN PATIENTS (FIX PYARROW + NUMERIC SAFETY + AGE FIX)
# ═══════════════════════════════════════════════════════════
def clean_patients(df):

    df = df.copy()

    chronic_cols = [c for c in df.columns if c.startswith("SP_")]
    cost_cols = [c for c in df.columns if any(x in c for x in ["MEDREIMB", "BENRES", "PPPYMT"])]

    keep = [
        "DESYNPUF_ID", "ANNEE",
        "BENE_BIRTH_DT",
        "BENE_SEX_IDENT_CD",
        "BENE_RACE_CD",
        "HOSPITALIZED_IN_6M"
    ]

    df = df[keep + chronic_cols + cost_cols]

    df = df.drop_duplicates(["DESYNPUF_ID", "ANNEE"])

    # 🔥 AGE SAFE — calculé par rapport au 1er janvier de l'ANNEE de la ligne
    # (et non une date fixe, sinon incohérence anti-leakage entre 2008/2009/2010)
    obs_dates = pd.to_datetime(df["ANNEE"].astype(str) + "-01-01")
    birth_dt = pd.to_datetime(df["BENE_BIRTH_DT"], errors="coerce")

    df["AGE"] = (obs_dates - birth_dt).dt.days / 365.25
    df["AGE"] = df["AGE"].fillna(df["AGE"].median())

    # numeric fix
    for c in chronic_cols + cost_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 🔥 FINAL FIX PYARROW
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string")

    return df


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def main():

    log.info("PIPELINE CLEANING START")

    ben08 = load_patients(2008)
    ids = ben08["DESYNPUF_ID"].unique()
    ben09 = load_patients(2009, ids)
    ben10 = load_patients(2010, ids)

    claims = load_claims(ids)

    # parse dates SAFE
    for k in ["inpatient", "outpatient", "carrier", "prescription"]:
        for c in claims[k].columns:
            if "DT" in c:
                claims[k][c] = parse_date(claims[k][c])

    claims["inpatient"]["admission"] = claims["inpatient"]["CLM_ADMSN_DT"]

    # 🔍 DIAGNOSTIC GLOBAL : qualité du parsing des dates d'admission
    n_total = len(claims["inpatient"])
    n_valid = claims["inpatient"]["admission"].notna().sum()
    log.info(
        f"Dates d'admission inpatient : {n_valid:,}/{n_total:,} valides "
        f"({100 * n_valid / n_total:.1f}%)"
    )

    if n_total > 0:
        log.info(
            f"Plage des dates d'admission valides : "
            f"{claims['inpatient']['admission'].min()} -> "
            f"{claims['inpatient']['admission'].max()}"
        )

    # targets
    ben08 = build_target(ben08, claims, "2008-01-01")
    ben09 = build_target(ben09, claims, "2009-01-01")
    ben10 = build_target(ben10, claims, "2010-01-01")

    df = pd.concat([ben08, ben09, ben10], ignore_index=True)

    df = clean_patients(df)

    # SAVE SAFE PARQUET
    df.to_parquet(
        os.path.join(OUTPUT_DIR, "patients_cleaned.parquet"),
        index=False
    )

    for k, v in claims.items():
        for c in v.columns:
            if v[c].dtype == "object":
                v[c] = v[c].astype("string")

        output_path = os.path.join(OUTPUT_DIR, f"claims_{k}_cleaned.parquet")
        v.to_parquet(output_path, index=False)
        log.info(f"Claims {k} sauvegardés -> {output_path}")

    log.info("PIPELINE CLEANING DONE SUCCESSFULLY ✅")

    return df, claims


if __name__ == "__main__":
    main()
