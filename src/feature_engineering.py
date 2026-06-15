# ═══════════════════════════════════════════════════════════
# src/feature_engineering.py
# Pipeline MLOps — Feature Engineering Production
# Projet : Risque d'Hospitalisation Medicare CMS DE-SynPUF
# ═══════════════════════════════════════════════════════════

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CLEANED_DIR = Path("data/cleaned")
FEATURES_DIR = Path("data/features")
LOG_DIR = Path("logs")

PATIENT_ID_COL = "DESYNPUF_ID"
YEAR_COL = "ANNEE"
TARGET_COL = "HOSPITALIZED_IN_6M"

PATIENTS_FILE = CLEANED_DIR / "patients_cleaned.parquet"

CLAIMS_FILES = {
    "inpatient": CLEANED_DIR / "claims_inpatient_cleaned.parquet",
    "outpatient": CLEANED_DIR / "claims_outpatient_cleaned.parquet",
    "carrier": CLEANED_DIR / "claims_carrier_cleaned.parquet",
    "prescription": CLEANED_DIR / "claims_prescription_cleaned.parquet",
}

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


CHRONIC_DISEASE_COLS = [
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


CHARLSON_WEIGHTS = {
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


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "feature_engineering.log"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def safe_numeric(series, fill_value: float = 0) -> pd.Series:
    """
    Convertit proprement en numérique.
    Fonction robuste pour Series, list, array.
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
    )


def get_num_col(df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    """
    Retourne toujours une Series numérique de même longueur que df.
    Évite l'erreur : 'bool' object has no attribute astype.
    """
    if col in df.columns:
        return safe_numeric(df[col], fill_value=default)

    return pd.Series(default, index=df.index, dtype="float64")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def clean_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if PATIENT_ID_COL not in df.columns:
        raise ValueError(f"Colonne obligatoire absente : {PATIENT_ID_COL}")

    df[PATIENT_ID_COL] = df[PATIENT_ID_COL].astype(str)

    return df


def empty_feature(name: str) -> pd.DataFrame:
    """
    Retourne un DataFrame vide mais avec les bonnes colonnes.
    On le merge quand même pour créer la colonne.
    """
    return pd.DataFrame(columns=[PATIENT_ID_COL, name])


def validate_inputs(
    df_patients: pd.DataFrame,
    claims: Dict[str, pd.DataFrame],
) -> None:
    required_patient_cols = [PATIENT_ID_COL, YEAR_COL, TARGET_COL]

    for col in required_patient_cols:
        if col not in df_patients.columns:
            raise ValueError(f"Colonne manquante dans patients_cleaned : {col}")

    required_claims = {
        "inpatient": [PATIENT_ID_COL, "admission"],
        "outpatient": [PATIENT_ID_COL, "CLM_FROM_DT"],
        "carrier": [PATIENT_ID_COL, "CLM_FROM_DT"],
        "prescription": [PATIENT_ID_COL, "SRVC_DT"],
    }

    for table, cols in required_claims.items():
        if table not in claims:
            raise KeyError(f"Claims manquant : {table}")

        for col in cols:
            if col not in claims[table].columns:
                raise ValueError(f"Colonne manquante dans claims['{table}'] : {col}")


# ═══════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════

def load_inputs() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    log.info("Chargement des données nettoyées...")

    require_file(PATIENTS_FILE)

    df_patients = pd.read_parquet(PATIENTS_FILE)
    df_patients = clean_patient_id(df_patients)

    claims: Dict[str, pd.DataFrame] = {}

    for key, path in CLAIMS_FILES.items():
        require_file(path)

        df_claim = pd.read_parquet(path)
        df_claim = clean_patient_id(df_claim)

        claims[key] = df_claim

        log.info(f"  {key}: {len(df_claim):,} lignes")

    claims = normalize_claim_dates(claims)

    validate_inputs(df_patients, claims)

    years = sorted(df_patients[YEAR_COL].dropna().astype(int).unique())

    log.info(f"Patients chargés : {len(df_patients):,}")
    log.info(f"Années disponibles : {years}")

    return df_patients, claims


def normalize_claim_dates(claims: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    claims = {key: value.copy() for key, value in claims.items()}

    # inpatient admission
    if "admission" not in claims["inpatient"].columns:
        if "CLM_ADMSN_DT" in claims["inpatient"].columns:
            claims["inpatient"]["admission"] = claims["inpatient"]["CLM_ADMSN_DT"]
        elif "CLM_FROM_DT" in claims["inpatient"].columns:
            claims["inpatient"]["admission"] = claims["inpatient"]["CLM_FROM_DT"]
        else:
            raise ValueError("Aucune date d'admission trouvée dans inpatient.")

    date_map = {
        "inpatient": ["admission", "CLM_FROM_DT", "CLM_THRU_DT"],
        "outpatient": ["CLM_FROM_DT", "CLM_THRU_DT"],
        "carrier": ["CLM_FROM_DT", "CLM_THRU_DT"],
        "prescription": ["SRVC_DT"],
    }

    for table, cols in date_map.items():
        for col in cols:
            if col in claims[table].columns:
                claims[table][col] = pd.to_datetime(
                    claims[table][col],
                    errors="coerce"
                )

    return claims


# ═══════════════════════════════════════════════════════════
# AGRÉGATIONS GÉNÉRIQUES ANTI-LEAKAGE
# ═══════════════════════════════════════════════════════════

def count_before(
    df: pd.DataFrame,
    date_col: str,
    obs_date: pd.Timestamp,
    name: str,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return empty_feature(name)

    tmp = df[df[date_col] < obs_date].copy()

    if tmp.empty:
        return empty_feature(name)

    return (
        tmp.groupby(PATIENT_ID_COL)
        .size()
        .reset_index(name=name)
    )


def count_window(
    df: pd.DataFrame,
    date_col: str,
    obs_date: pd.Timestamp,
    months: int,
    name: str,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return empty_feature(name)

    start = obs_date - pd.DateOffset(months=months)

    tmp = df[
        (df[date_col] >= start) &
        (df[date_col] < obs_date)
    ].copy()

    if tmp.empty:
        return empty_feature(name)

    return (
        tmp.groupby(PATIENT_ID_COL)
        .size()
        .reset_index(name=name)
    )


def sum_before(
    df: pd.DataFrame,
    date_col: str,
    amount_col: str,
    obs_date: pd.Timestamp,
    name: str,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or amount_col not in df.columns:
        return empty_feature(name)

    tmp = df[df[date_col] < obs_date].copy()

    if tmp.empty:
        return empty_feature(name)

    tmp[amount_col] = safe_numeric(tmp[amount_col], fill_value=0)

    return (
        tmp.groupby(PATIENT_ID_COL)[amount_col]
        .sum()
        .reset_index(name=name)
    )


def unique_before(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    obs_date: pd.Timestamp,
    name: str,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return empty_feature(name)

    tmp = df[df[date_col] < obs_date].copy()

    if tmp.empty:
        return empty_feature(name)

    return (
        tmp.groupby(PATIENT_ID_COL)[value_col]
        .nunique()
        .reset_index(name=name)
    )


def days_since_last(
    df: pd.DataFrame,
    date_col: str,
    obs_date: pd.Timestamp,
    name: str,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return empty_feature(name)

    tmp = df[df[date_col] < obs_date].copy()

    if tmp.empty:
        return empty_feature(name)

    last_dates = (
        tmp.groupby(PATIENT_ID_COL)[date_col]
        .max()
        .reset_index(name="LAST_EVENT_DATE")
    )

    last_dates[name] = (obs_date - last_dates["LAST_EVENT_DATE"]).dt.days

    return last_dates[[PATIENT_ID_COL, name]]


# ═══════════════════════════════════════════════════════════
# FEATURES PAR ANNÉE
# ═══════════════════════════════════════════════════════════

def build_year_features(
    df_patients_year: pd.DataFrame,
    claims: Dict[str, pd.DataFrame],
    year: int,
) -> pd.DataFrame:
    obs_date = pd.Timestamp(f"{year}-01-01")

    log.info(f"Construction features année {year} | OBS_DATE={obs_date.date()}")

    df_year = df_patients_year.copy()
    df_year = clean_patient_id(df_year)

    ip = claims["inpatient"]
    op = claims["outpatient"]
    car = claims["carrier"]
    rx = claims["prescription"]

    feature_tables: List[pd.DataFrame] = []

    # Hospitalisations passées
    feature_tables.append(
        count_before(
            ip,
            "admission",
            obs_date,
            "NB_HOSP_PASSEES"
        )
    )

    feature_tables.append(
        days_since_last(
            ip,
            "admission",
            obs_date,
            "DAYS_SINCE_LAST_HOSP"
        )
    )

    # Fenêtres temporelles 3M, 6M, 12M
    for months in [3, 6, 12]:
        feature_tables.append(
            count_window(
                ip,
                "admission",
                obs_date,
                months,
                f"NB_HOSP_{months}M"
            )
        )

        feature_tables.append(
            count_window(
                op,
                "CLM_FROM_DT",
                obs_date,
                months,
                f"NB_OP_{months}M"
            )
        )

        feature_tables.append(
            count_window(
                car,
                "CLM_FROM_DT",
                obs_date,
                months,
                f"NB_CAR_{months}M"
            )
        )

        feature_tables.append(
            count_window(
                rx,
                "SRVC_DT",
                obs_date,
                months,
                f"NB_RX_{months}M"
            )
        )

    # Totaux passés
    feature_tables.append(
        count_before(
            op,
            "CLM_FROM_DT",
            obs_date,
            "NB_OP_TOTAL_PASSE"
        )
    )

    feature_tables.append(
        count_before(
            car,
            "CLM_FROM_DT",
            obs_date,
            "NB_CAR_TOTAL_PASSE"
        )
    )

    feature_tables.append(
        count_before(
            rx,
            "SRVC_DT",
            obs_date,
            "NB_PRESCRIPTIONS"
        )
    )

    # Dernier événement
    feature_tables.append(
        days_since_last(
            op,
            "CLM_FROM_DT",
            obs_date,
            "DAYS_SINCE_LAST_OP"
        )
    )

    feature_tables.append(
        days_since_last(
            car,
            "CLM_FROM_DT",
            obs_date,
            "DAYS_SINCE_LAST_CAR"
        )
    )

    feature_tables.append(
        days_since_last(
            rx,
            "SRVC_DT",
            obs_date,
            "DAYS_SINCE_LAST_RX"
        )
    )

    # Nombre de molécules uniques
    feature_tables.append(
        unique_before(
            rx,
            "SRVC_DT",
            "PROD_SRVC_ID",
            obs_date,
            "NB_MOLECULES_UNIQUES"
        )
    )

    # Coûts claims passés si disponibles
    feature_tables.append(
        sum_before(
            ip,
            "admission",
            "CLM_PMT_AMT",
            obs_date,
            "COUT_HOSP_PASSE"
        )
    )

    feature_tables.append(
        sum_before(
            op,
            "CLM_FROM_DT",
            "CLM_PMT_AMT",
            obs_date,
            "COUT_OP_PASSE"
        )
    )

    feature_tables.append(
        sum_before(
            car,
            "CLM_FROM_DT",
            "CLM_PMT_AMT",
            obs_date,
            "COUT_CAR_PASSE"
        )
    )

    # Merge toutes les tables même vides pour garder les colonnes
    for feat_df in feature_tables:
        if feat_df is not None:
            df_year = df_year.merge(
                feat_df,
                on=PATIENT_ID_COL,
                how="left"
            )

    # Colonnes générées obligatoires
    generated_cols = [
        "NB_HOSP_PASSEES",
        "DAYS_SINCE_LAST_HOSP",
        "NB_HOSP_3M",
        "NB_HOSP_6M",
        "NB_HOSP_12M",
        "NB_OP_3M",
        "NB_OP_6M",
        "NB_OP_12M",
        "NB_CAR_3M",
        "NB_CAR_6M",
        "NB_CAR_12M",
        "NB_RX_3M",
        "NB_RX_6M",
        "NB_RX_12M",
        "NB_OP_TOTAL_PASSE",
        "NB_CAR_TOTAL_PASSE",
        "NB_PRESCRIPTIONS",
        "DAYS_SINCE_LAST_OP",
        "DAYS_SINCE_LAST_CAR",
        "DAYS_SINCE_LAST_RX",
        "NB_MOLECULES_UNIQUES",
        "COUT_HOSP_PASSE",
        "COUT_OP_PASSE",
        "COUT_CAR_PASSE",
    ]

    for col in generated_cols:
        if col not in df_year.columns:
            df_year[col] = 0

        df_year[col] = safe_numeric(df_year[col], fill_value=0)

    # Features dérivées 100% Series-safe
    nb_hosp_passees = get_num_col(df_year, "NB_HOSP_PASSEES")
    nb_op_12m = get_num_col(df_year, "NB_OP_12M")
    nb_prescriptions = get_num_col(df_year, "NB_PRESCRIPTIONS")
    nb_molecules = get_num_col(df_year, "NB_MOLECULES_UNIQUES")

    nb_hosp_3m = get_num_col(df_year, "NB_HOSP_3M")
    nb_hosp_6m = get_num_col(df_year, "NB_HOSP_6M")
    nb_hosp_12m = get_num_col(df_year, "NB_HOSP_12M")

    nb_op_3m = get_num_col(df_year, "NB_OP_3M")
    nb_op_6m = get_num_col(df_year, "NB_OP_6M")

    nb_car_3m = get_num_col(df_year, "NB_CAR_3M")
    nb_car_6m = get_num_col(df_year, "NB_CAR_6M")
    nb_car_12m = get_num_col(df_year, "NB_CAR_12M")

    nb_rx_3m = get_num_col(df_year, "NB_RX_3M")
    nb_rx_6m = get_num_col(df_year, "NB_RX_6M")
    nb_rx_12m = get_num_col(df_year, "NB_RX_12M")

    nb_op_total = get_num_col(df_year, "NB_OP_TOTAL_PASSE")
    nb_car_total = get_num_col(df_year, "NB_CAR_TOTAL_PASSE")

    df_year["HAS_HOSP_HISTORY"] = (nb_hosp_passees > 0).astype(int)

    df_year["POLYPHARMACIE"] = (nb_molecules > 5).astype(int)

    df_year["UTILISATION_3M"] = (
        nb_hosp_3m + nb_op_3m + nb_car_3m + nb_rx_3m
    )

    df_year["UTILISATION_6M"] = (
        nb_hosp_6m + nb_op_6m + nb_car_6m + nb_rx_6m
    )

    df_year["UTILISATION_12M"] = (
        nb_hosp_12m + nb_op_12m + nb_car_12m + nb_rx_12m
    )

    df_year["IS_NEW_PATIENT"] = (
        (nb_hosp_passees == 0)
        & (nb_op_total == 0)
        & (nb_car_total == 0)
        & (nb_prescriptions == 0)
    ).astype(int)

    df_year["RATIO_RX_OP_12M"] = nb_rx_12m / (nb_op_12m + 1)
    df_year["RATIO_CAR_OP_12M"] = nb_car_12m / (nb_op_12m + 1)

    log.info(
        f"  → Année {year} terminée : "
        f"{df_year.shape[0]:,} lignes, {df_year.shape[1]} colonnes"
    )

    return df_year


# ═══════════════════════════════════════════════════════════
# FEATURES COMPOSITES PATIENTS
# ═══════════════════════════════════════════════════════════

def add_composite_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Ajout des features composites patients...")

    df = df.copy()

    # Comorbidités
    for col in CHRONIC_DISEASE_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = (
            safe_numeric(df[col], fill_value=0)
            .astype(int)
            .clip(0, 1)
        )

    df["NB_COMORBIDITES"] = df[CHRONIC_DISEASE_COLS].sum(axis=1)

    # Charlson
    df["CHARLSON_INDEX"] = 0

    for col, weight in CHARLSON_WEIGHTS.items():
        if col in df.columns:
            df["CHARLSON_INDEX"] += (
                safe_numeric(df[col], fill_value=0).astype(int) * weight
            )

    # Coûts annuels beneficiary
    cost_cols = [
        col for col in df.columns
        if any(k in col for k in ["MEDREIMB", "BENRES", "PPPYMT"])
    ]

    if cost_cols:
        for col in cost_cols:
            df[col] = safe_numeric(df[col], fill_value=0)

        df["COUT_BENEFICIARY_ANNUEL"] = df[cost_cols].sum(axis=1)
    else:
        df["COUT_BENEFICIARY_ANNUEL"] = 0

    # Coûts claims passés
    df["COUT_HOSP_PASSE"] = get_num_col(df, "COUT_HOSP_PASSE")
    df["COUT_OP_PASSE"] = get_num_col(df, "COUT_OP_PASSE")
    df["COUT_CAR_PASSE"] = get_num_col(df, "COUT_CAR_PASSE")

    df["COUT_CLAIMS_PASSE"] = (
        df["COUT_HOSP_PASSE"]
        + df["COUT_OP_PASSE"]
        + df["COUT_CAR_PASSE"]
    )

    df["COUT_TOTAL"] = (
        df["COUT_BENEFICIARY_ANNUEL"]
        + df["COUT_CLAIMS_PASSE"]
    )

    # Âge
    if "AGE" in df.columns:
        df["AGE"] = safe_numeric(df["AGE"], fill_value=np.nan)

        age_median = df["AGE"].median()

        if pd.isna(age_median):
            age_median = 70

        df["AGE"] = df["AGE"].fillna(age_median).clip(0, 120)
    else:
        df["AGE"] = 70

    df["GROUPE_AGE"] = pd.cut(
        df["AGE"],
        bins=[0, 65, 75, 85, 120],
        labels=["<65", "65-74", "75-84", "85+"],
        include_lowest=True,
    )

    log.info(f"  NB_COMORBIDITES moyen : {df['NB_COMORBIDITES'].mean():.2f}")
    log.info(f"  CHARLSON_INDEX moyen : {df['CHARLSON_INDEX'].mean():.2f}")
    log.info(f"  COUT_TOTAL moyen : {df['COUT_TOTAL'].mean():.2f}")

    return df


# ═══════════════════════════════════════════════════════════
# BUILD GLOBAL
# ═══════════════════════════════════════════════════════════

def build_features(
    df_patients: pd.DataFrame,
    claims: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("FEATURE ENGINEERING — DÉBUT")
    log.info("=" * 70)

    df_patients = df_patients.copy()
    df_patients = clean_patient_id(df_patients)

    years = sorted(
        df_patients[YEAR_COL]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not years:
        raise ValueError("Aucune année trouvée dans patients_cleaned.")

    all_years: List[pd.DataFrame] = []

    for year in years:
        df_year = df_patients[
            df_patients[YEAR_COL].astype(int) == int(year)
        ].copy()

        if df_year.empty:
            log.warning(f"Aucune ligne pour l'année {year}")
            continue

        df_year_features = build_year_features(
            df_patients_year=df_year,
            claims=claims,
            year=int(year),
        )

        all_years.append(df_year_features)

    if not all_years:
        raise ValueError("Aucun DataFrame de features généré.")

    df_final = pd.concat(all_years, ignore_index=True)

    df_final = add_composite_patient_features(df_final)

    df_final = df_final.loc[:, ~df_final.columns.duplicated()].copy()

    # Nettoyage numérique final
    numeric_prefixes = [
        "NB_",
        "COUT_",
        "DAYS_",
        "HAS_",
        "IS_",
        "UTILISATION_",
        "RATIO_",
        "CHARLSON",
    ]

    for col in df_final.columns:
        if any(col.startswith(prefix) for prefix in numeric_prefixes):
            df_final[col] = safe_numeric(df_final[col], fill_value=0)

    if TARGET_COL in df_final.columns:
        df_final[TARGET_COL] = (
            safe_numeric(df_final[TARGET_COL], fill_value=0)
            .astype(int)
        )

    log.info(
        f"DataFrame final features : "
        f"{df_final.shape[0]:,} lignes, {df_final.shape[1]} colonnes"
    )

    log.info("=" * 70)
    log.info("FEATURE ENGINEERING — TERMINÉ ✅")
    log.info("=" * 70)

    return df_final


# ═══════════════════════════════════════════════════════════
# SAUVEGARDE
# ═══════════════════════════════════════════════════════════

def save_features(df_final: pd.DataFrame) -> None:
    output_path = FEATURES_DIR / "features_engineered.parquet"
    columns_path = FEATURES_DIR / "features_engineered_columns.csv"

    df_final.to_parquet(output_path, index=False)

    pd.DataFrame({"column": df_final.columns}).to_csv(
        columns_path,
        index=False
    )

    log.info(f"✅ Features sauvegardées → {output_path}")
    log.info(f"✅ Liste colonnes sauvegardée → {columns_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    df_patients, claims = load_inputs()

    df_final = build_features(df_patients, claims)

    save_features(df_final)

    return df_final


if __name__ == "__main__":
    main()