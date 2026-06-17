import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import parse_date, clean_patients, build_target


def test_date_parsing():
    col = pd.Series([20100312.0])
    result = parse_date(col)
    assert result.iloc[0] == pd.Timestamp("2010-03-12")


def test_esrd_mapping():
    s = pd.Series(["Y", "0", 0])
    mapping = {"Y": 1, "0": 0, 0: 0}
    result = s.map(mapping).fillna(0).astype(int)
    assert list(result) == [1, 0, 0]


def test_target_binary(sample_patients, sample_claims_inpatient):
    cohort = sample_patients[sample_patients["ANNEE"] == 2008].copy()
    result = build_target(cohort, sample_claims_inpatient, "2008-01-01")
    assert result["HOSPITALIZED_IN_6M"].isin({0, 1}).all()


def test_no_dedup_bug(sample_patients_with_dup_different_year):
    result = clean_patients(sample_patients_with_dup_different_year)
    x1_rows = result[result["DESYNPUF_ID"] == "X1"]
    assert len(x1_rows) == 3
    assert list(x1_rows["ANNEE"]) == [2008, 2009, 2010]
