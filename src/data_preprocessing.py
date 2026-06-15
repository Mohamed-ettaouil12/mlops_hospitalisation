# ═══════════════════════════════════════════════════════════
# Data Preprocessing — Split temporel production
# ═══════════════════════════════════════════════════════════

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LOG_DIR = Path("logs")
CLEANED_DIR = Path("data/cleaned")
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")

LOG_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PATIENT_ID = "DESYNPUF_ID"
TARGET = "HOSPITALIZED_IN_6M"
YEARS = [2008, 2009, 2010]

CHRONIC_COLS = [
    "SP_ALZHDMTA",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_CNCR",
    "SP_COPD",
    "SP_DEPRESSN",
    "SP_DIABETES",
    "SP_ISCHMCHT",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def safe_numeric(series, fill_value=0):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
    )


def load_inputs():
    patients_path = CLEANED_DIR / "patients_cleaned.parquet"
    require_file(patients_path)

    log.info("Chargement des donnees nettoyees...")
    patients = pd.read_parquet(patients_path)
    log.info(
        "  -> %s patients | Annees : %s",
        f"{len(patients):,}",
        sorted(patients["ANNEE"].unique().tolist()),
    )

    claims = {}
    for key in ["inpatient", "outpatient", "carrier", "prescription"]:
        path = CLEANED_DIR / f"claims_{key}_cleaned.parquet"
        require_file(path)
        claims[key] = pd.read_parquet(path)

    return patients, claims


def prepare_patients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CHRONIC_COLS:
        if col in df.columns:
            df[col] = (safe_numeric(df[col], 2) == 1).astype(int)

    cost_cols = [c for c in df.columns if any(x in c for x in ["MEDREIMB", "BENRES", "PPPYMT"])]
    for col in cost_cols:
        df[col] = safe_numeric(df[col])

    df["COUT_TOTAL"] = df[cost_cols].sum(axis=1) if cost_cols else 0
    df["AGE"] = safe_numeric(df.get("AGE", pd.Series(0, index=df.index)))
    df["SEXE_ENC"] = safe_numeric(df.get("BENE_SEX_IDENT_CD", pd.Series(0, index=df.index)))
    df["RACE_ENC"] = safe_numeric(df.get("BENE_RACE_CD", pd.Series(0, index=df.index)))
    df["BENE_ESRD_IND"] = safe_numeric(df.get("BENE_ESRD_IND", pd.Series(0, index=df.index)))

    df[TARGET] = safe_numeric(df[TARGET]).astype(int)
    return df


def ensure_claim_dates(claims):
    for key, df in claims.items():
        df = df.copy()
        for col in df.columns:
            if "DT" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        claims[key] = df

    if "admission" not in claims["inpatient"].columns:
        claims["inpatient"]["admission"] = claims["inpatient"].get("CLM_ADMSN_DT")

    return claims


def build_dynamic_features(patients: pd.DataFrame, claims) -> pd.DataFrame:
    log.info("Construction des features dynamiques anti-leakage...")
    claims = ensure_claim_dates(claims)
    results = []

    for year in YEARS:
        obs_date = pd.Timestamp(f"{year}-01-01")
        df_year = patients[patients["ANNEE"] == year].copy()

        inpatient = claims["inpatient"]
        ip_past = inpatient[inpatient["admission"] < obs_date]
        hosp_past = (
            ip_past.groupby(PATIENT_ID)
            .size()
            .reset_index(name="NB_HOSP_PASSEES")
        )

        outpatient = claims["outpatient"]
        op_features = []
        for months in [3, 6, 12]:
            start = obs_date - pd.DateOffset(months=months)
            mask = (outpatient["CLM_FROM_DT"] >= start) & (outpatient["CLM_FROM_DT"] < obs_date)
            op_features.append(
                outpatient[mask].groupby(PATIENT_ID).size().reset_index(name=f"NB_OP_{months}M")
            )

        carrier = claims["carrier"]
        car_start = obs_date - pd.DateOffset(months=6)
        car_mask = (carrier["CLM_FROM_DT"] >= car_start) & (carrier["CLM_FROM_DT"] < obs_date)
        car_6m = carrier[car_mask].groupby(PATIENT_ID).size().reset_index(name="NB_CAR_6M")

        prescription = claims["prescription"]
        rx_past = prescription[prescription["SRVC_DT"] < obs_date]
        rx_count = rx_past.groupby(PATIENT_ID).size().reset_index(name="NB_PRESCRIPTIONS")

        if "PROD_SRVC_ID" in rx_past.columns:
            rx_unique = (
                rx_past.groupby(PATIENT_ID)["PROD_SRVC_ID"]
                .nunique()
                .reset_index(name="NB_MOLECULES_UNIQUES")
            )
        else:
            rx_unique = pd.DataFrame(columns=[PATIENT_ID, "NB_MOLECULES_UNIQUES"])

        for feat in [hosp_past, *op_features, car_6m, rx_count, rx_unique]:
            df_year = df_year.merge(feat, on=PATIENT_ID, how="left")

        results.append(df_year)

    df = pd.concat(results, ignore_index=True)
    dynamic_cols = [
        "NB_HOSP_PASSEES",
        "NB_OP_3M",
        "NB_OP_6M",
        "NB_OP_12M",
        "NB_CAR_6M",
        "NB_PRESCRIPTIONS",
        "NB_MOLECULES_UNIQUES",
    ]

    for col in dynamic_cols:
        df[col] = safe_numeric(df.get(col, pd.Series(0, index=df.index)))

    df["IS_NEW_PATIENT"] = (
        (df["NB_HOSP_PASSEES"] == 0)
        & (df["NB_OP_12M"] == 0)
        & (df["NB_PRESCRIPTIONS"] == 0)
    ).astype(int)

    log.info("  -> Nouveaux patients sans historique : %s", f"{df['IS_NEW_PATIENT'].sum():,}")
    return df


def build_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Construction des features composites...")

    chronic_cols = [c for c in CHRONIC_COLS if c in df.columns]
    df["NB_COMORBIDITES"] = df[chronic_cols].sum(axis=1)

    charlson_weights = {
        "SP_CHF": 1,
        "SP_DIABETES": 1,
        "SP_CHRNKIDN": 2,
        "SP_CNCR": 2,
        "SP_COPD": 1,
        "SP_STRKETIA": 2,
        "SP_ALZHDMTA": 1,
        "SP_DEPRESSN": 1,
        "SP_ISCHMCHT": 1,
        "SP_OSTEOPRS": 0,
        "SP_RA_OA": 1,
    }
    df["CHARLSON_INDEX"] = sum(
        df[col] * weight for col, weight in charlson_weights.items() if col in df.columns
    )
    df["POLYPHARMACIE"] = (df["NB_MOLECULES_UNIQUES"] > 5).astype(int)
    df["GROUPE_AGE_ENC"] = pd.cut(
        df["AGE"],
        bins=[0, 65, 75, 85, 120],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float).fillna(1)

    log.info("  NB_COMORBIDITES moyen : %.2f", df["NB_COMORBIDITES"].mean())
    log.info("  CHARLSON_INDEX moyen  : %.2f", df["CHARLSON_INDEX"].mean())
    return df


def split_and_save(df: pd.DataFrame):
    log.info("Encodage, normalisation et split temporel strict...")

    feature_cols = [
        *[c for c in CHRONIC_COLS if c in df.columns],
        "AGE",
        "SEXE_ENC",
        "RACE_ENC",
        "BENE_ESRD_IND",
        "GROUPE_AGE_ENC",
        "NB_COMORBIDITES",
        "CHARLSON_INDEX",
        "COUT_TOTAL",
        "IS_NEW_PATIENT",
        "NB_HOSP_PASSEES",
        "NB_OP_3M",
        "NB_OP_6M",
        "NB_OP_12M",
        "NB_CAR_6M",
        "NB_PRESCRIPTIONS",
        "NB_MOLECULES_UNIQUES",
        "POLYPHARMACIE",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(float)
    )
    y = df[TARGET].astype(int)

    scale_cols = [
        c for c in [
            "AGE",
            "COUT_TOTAL",
            "CHARLSON_INDEX",
            "NB_COMORBIDITES",
            "NB_HOSP_PASSEES",
            "NB_OP_3M",
            "NB_OP_6M",
            "NB_OP_12M",
            "NB_CAR_6M",
            "NB_PRESCRIPTIONS",
            "NB_MOLECULES_UNIQUES",
        ] if c in X.columns
    ]

    train_mask = df["ANNEE"] == 2008
    val_mask = df["ANNEE"] == 2009
    test_mask = df["ANNEE"] == 2010

    scaler = StandardScaler()
    X_scaled = X.copy()
    if scale_cols:
        X_scaled.loc[:, scale_cols] = scaler.fit_transform(X.loc[:, scale_cols])

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]

    log.info("  Train (2008) : %s | Taux : %.2f%%", f"{len(X_train):,}", y_train.mean() * 100)
    log.info("  Val   (2009) : %s | Taux : %.2f%%", f"{len(X_val):,}", y_val.mean() * 100)
    log.info("  Test  (2010) : %s | Taux : %.2f%%", f"{len(X_test):,}", y_test.mean() * 100)
    log.info("  Nombre de features : %s", len(feature_cols))

    X_train.to_parquet(FEATURES_DIR / "X_train.parquet", index=False)
    X_val.to_parquet(FEATURES_DIR / "X_val.parquet", index=False)
    X_test.to_parquet(FEATURES_DIR / "X_test.parquet", index=False)
    y_train.to_frame(name=TARGET).to_parquet(FEATURES_DIR / "y_train.parquet", index=False)
    y_val.to_frame(name=TARGET).to_parquet(FEATURES_DIR / "y_val.parquet", index=False)
    y_test.to_frame(name=TARGET).to_parquet(FEATURES_DIR / "y_test.parquet", index=False)

    pd.Series(feature_cols, name="feature").to_csv(FEATURES_DIR / "feature_names.csv", index=False)
    joblib.dump(feature_cols, MODELS_DIR / "feature_names.pkl")
    joblib.dump(scale_cols, MODELS_DIR / "scale_cols.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    report = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_target_rate": float(y_train.mean()),
        "val_target_rate": float(y_val.mean()),
        "test_target_rate": float(y_test.mean()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "scale_cols": scale_cols,
    }
    with open(FEATURES_DIR / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("  -> Donnees sauvegardees dans %s", FEATURES_DIR)
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def main():
    log.info("=" * 60)
    log.info("PIPELINE DATA PREPROCESSING — DEBUT")
    log.info("=" * 60)

    patients, claims = load_inputs()
    patients = prepare_patients(patients)
    df = build_dynamic_features(patients, claims)
    df = build_composite_features(df)
    result = split_and_save(df)

    log.info("=" * 60)
    log.info("PIPELINE DATA PREPROCESSING — TERMINE")
    log.info("=" * 60)
    return result


if __name__ == "__main__":
    main()
