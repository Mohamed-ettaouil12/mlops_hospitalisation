import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_patients():
    return pd.DataFrame({
        "DESYNPUF_ID": ["A1", "A1", "B2", "C3", "D4"],
        "ANNEE": [2008, 2009, 2008, 2008, 2009],
        "BENE_BIRTH_DT": pd.to_datetime(["1940-06-15", "1940-06-15", "1955-01-01", "1970-10-10", "1980-03-20"]),
        "BENE_SEX_IDENT_CD": [1, 1, 2, 1, 2],
        "BENE_RACE_CD": [1, 1, 2, 3, 1],
        "HOSPITALIZED_IN_6M": [0, 0, 0, 0, 0],
        "SP_CHF": [0, 0, 1, 0, 0],
        "SP_CHRONKIDNEY": [0, 0, 0, 0, 1],
        "BENE_MDCR_PYMT_AMT": [1000.0, 1200.0, 800.0, 900.0, 1100.0],
    })


@pytest.fixture
def sample_patients_with_dup_different_year():
    """Same DESYNPUF_ID across different ANNEE — should NOT be deduplicated."""
    return pd.DataFrame({
        "DESYNPUF_ID": ["X1", "X1", "X1", "Y2", "Y2"],
        "ANNEE": [2008, 2009, 2010, 2008, 2009],
        "BENE_BIRTH_DT": pd.to_datetime(["1950-01-01"] * 5),
        "BENE_SEX_IDENT_CD": [1] * 5,
        "BENE_RACE_CD": [1] * 5,
        "HOSPITALIZED_IN_6M": [0] * 5,
        "SP_CHF": [0, 1, 0, 0, 0],
        "BENE_MDCR_PYMT_AMT": [500.0] * 5,
    })


@pytest.fixture
def sample_patients_with_dup_same_year():
    """Two identical rows for same DESYNPUF_ID + ANNEE — should deduplicate to 1."""
    return pd.DataFrame({
        "DESYNPUF_ID": ["A1", "A1", "B2", "B2"],
        "ANNEE": [2008, 2008, 2009, 2009],
        "BENE_BIRTH_DT": pd.to_datetime(["1940-06-15", "1940-06-15", "1955-01-01", "1955-01-01"]),
        "BENE_SEX_IDENT_CD": [1, 1, 2, 2],
        "BENE_RACE_CD": [1, 1, 2, 2],
        "HOSPITALIZED_IN_6M": [0, 0, 0, 0],
        "SP_CHF": [0, 0, 1, 1],
        "BENE_MDCR_PYMT_AMT": [1000.0, 1000.0, 800.0, 800.0],
    })


@pytest.fixture
def sample_patients_with_esrd():
    """Includes BENE_ESRD_IND with mixed types (bug #3 scenario)."""
    return pd.DataFrame({
        "DESYNPUF_ID": ["E1", "E2", "E3", "E4", "E5"],
        "ANNEE": [2008] * 5,
        "BENE_BIRTH_DT": pd.to_datetime(["1940-01-01"] * 5),
        "BENE_SEX_IDENT_CD": [1] * 5,
        "BENE_RACE_CD": [1] * 5,
        "HOSPITALIZED_IN_6M": [0] * 5,
        "BENE_ESRD_IND": ["Y", "0", 0, "Y", 0],
        "SP_CHF": [0] * 5,
        "BENE_MDCR_PYMT_AMT": [1000.0] * 5,
    })


@pytest.fixture
def sample_claims_inpatient():
    claims = pd.DataFrame({
        "DESYNPUF_ID": ["A1", "A1", "A1", "B2", "C3", "D4", "E1"],
        "CLM_ADMSN_DT": [
            "2008-01-15", "2008-07-01", "2008-01-01",
            "2008-02-01", "2008-03-15", "2009-06-15", "2008-06-01",
        ],
        "CLM_IP_ADMSN_TYPE_CD": [1, 2, 1, 1, 1, 1, 1],
        "NCH_BENE_DSCHRG_DT": [
            "2008-01-20", "2008-07-10", "2008-01-05",
            "2008-02-10", "2008-03-20", "2009-06-25", "2008-06-10",
        ],
    })
    for c in ["CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"]:
        claims[c] = pd.to_datetime(claims[c])
    claims["admission"] = claims["CLM_ADMSN_DT"]
    return {"inpatient": claims, "outpatient": pd.DataFrame(), "carrier": pd.DataFrame(), "prescription": pd.DataFrame()}


@pytest.fixture
def sample_claims_same_day():
    """Admissions exactly on obs_date 2008-01-01."""
    claims = pd.DataFrame({
        "DESYNPUF_ID": ["SAME_DAY", "FUTURE", "BEFORE", "BEYOND"],
        "CLM_ADMSN_DT": pd.to_datetime([
            "2008-01-01", "2008-01-02", "2007-12-31", "2008-07-02"
        ]),
        "CLM_IP_ADMSN_TYPE_CD": [1, 1, 1, 1],
        "NCH_BENE_DSCHRG_DT": pd.to_datetime([
            "2008-01-05", "2008-01-10", "2008-01-05", "2008-07-10"
        ]),
    })
    claims["admission"] = claims["CLM_ADMSN_DT"]
    return {"inpatient": claims, "outpatient": pd.DataFrame(), "carrier": pd.DataFrame(), "prescription": pd.DataFrame()}
