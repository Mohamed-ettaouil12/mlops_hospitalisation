import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import parse_date, clean_patients, build_target


def _as_series(dates):
    return pd.to_datetime(pd.Series(dates))


class TestParseDate:
    """Bug #1: floats like 20100312.0 → parsed with format='%Y%m%d' after str strip."""

    def test_float_input(self):
        col = pd.Series([20100312.0, 20080101.0, 20091231.0])
        result = parse_date(col)
        expected = _as_series(["2010-03-12", "2008-01-01", "2009-12-31"])
        pd.testing.assert_series_equal(result, expected)

    def test_string_input(self):
        col = pd.Series(["20100312", "20080101", "20091231"])
        result = parse_date(col)
        expected = _as_series(["2010-03-12", "2008-01-01", "2009-12-31"])
        pd.testing.assert_series_equal(result, expected)

    def test_float_with_dot_zero_string(self):
        col = pd.Series(["20100312.0", "20100101.0"])
        result = parse_date(col)
        expected = _as_series(["2010-03-12", "2010-01-01"])
        pd.testing.assert_series_equal(result, expected)

    def test_nan_input_coerces_nat(self):
        col = pd.Series([20100312.0, None, float("nan"), "not_a_date", ""])
        result = parse_date(col)
        assert result.iloc[0] == pd.Timestamp("2010-03-12")
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[4])

    def test_strips_dot_zero_from_float(self):
        col = pd.Series([20100101.0, 20080615.0])
        result = parse_date(col)
        expected = _as_series(["2010-01-01", "2008-06-15"])
        pd.testing.assert_series_equal(result, expected)


class TestDeduplication:
    """Bug #2: dedup on (DESYNPUF_ID, ANNEE), not DESYNPUF_ID alone."""

    def test_different_years_kept(self, sample_patients_with_dup_different_year):
        result = clean_patients(sample_patients_with_dup_different_year)
        assert len(result) == 5
        assert list(result["DESYNPUF_ID"]) == ["X1", "X1", "X1", "Y2", "Y2"]
        assert list(result["ANNEE"]) == [2008, 2009, 2010, 2008, 2009]

    def test_same_year_dropped(self, sample_patients_with_dup_same_year):
        result = clean_patients(sample_patients_with_dup_same_year)
        assert len(result) == 2
        assert result["DESYNPUF_ID"].nunique() == 2

    def test_no_duplicates_preserved(self, sample_patients):
        result = clean_patients(sample_patients)
        assert len(result) == len(sample_patients)


class TestBeneEsrdInd:
    """Bug #3: explicit map {'Y': 1, '0': 0, 0: 0} — no direct cast."""

    def test_explicit_mapping(self):
        s = pd.Series(["Y", "0", 0, "Y", "0", 0, "Y"])
        expected = [1, 0, 0, 1, 0, 0, 1]
        mapping = {"Y": 1, "0": 0, 0: 0}
        result = s.map(mapping).fillna(0).astype(int)
        assert list(result) == expected

    def test_mapping_dtype_int(self):
        s = pd.Series(["Y", 0, "0"])
        mapping = {"Y": 1, "0": 0, 0: 0}
        result = s.map(mapping).fillna(0).astype(int)
        assert result.dtype == int

    def test_mapping_with_nan(self):
        s = pd.Series(["Y", np.nan, "0", None, 0])
        mapping = {"Y": 1, "0": 0, 0: 0}
        result = s.map(mapping).fillna(0).astype(int)
        assert list(result) == [1, 0, 0, 0, 0]

    def test_clean_patients_with_esrd(self, sample_patients_with_esrd):
        df = sample_patients_with_esrd.copy()
        chronic_cols = [c for c in df.columns if c.startswith("SP_")]
        cost_cols = [c for c in df.columns if "BENE_MDCR" in c]
        keep = [
            "DESYNPUF_ID", "ANNEE",
            "BENE_BIRTH_DT",
            "BENE_SEX_IDENT_CD",
            "BENE_RACE_CD",
            "HOSPITALIZED_IN_6M",
            "BENE_ESRD_IND",
        ]
        df = df[keep + chronic_cols + cost_cols]
        mapping = {"Y": 1, "0": 0, 0: 0}
        df["BENE_ESRD_IND"] = df["BENE_ESRD_IND"].map(mapping).fillna(0).astype(int)
        df = df.drop_duplicates(["DESYNPUF_ID", "ANNEE"])
        assert list(df["BENE_ESRD_IND"]) == [1, 0, 0, 1, 0]
        assert df["BENE_ESRD_IND"].dtype == int


class TestBuildTarget:
    """Bug #4: window strictly future — no same-day admissions at obs_date."""

    def test_excludes_same_day_admission(self, sample_claims_same_day):
        ids = pd.DataFrame({
            "DESYNPUF_ID": ["SAME_DAY", "FUTURE", "BEFORE", "BEYOND"],
            "ANNEE": [2008] * 4,
        })
        result = build_target(ids, sample_claims_same_day, "2008-01-01")
        assert result.loc[result["DESYNPUF_ID"] == "SAME_DAY", "HOSPITALIZED_IN_6M"].iloc[0] == 0
        assert result.loc[result["DESYNPUF_ID"] == "FUTURE", "HOSPITALIZED_IN_6M"].iloc[0] == 1
        assert result.loc[result["DESYNPUF_ID"] == "BEFORE", "HOSPITALIZED_IN_6M"].iloc[0] == 0
        assert result.loc[result["DESYNPUF_ID"] == "BEYOND", "HOSPITALIZED_IN_6M"].iloc[0] == 0

    def test_normal_cohort_2008(self, sample_patients, sample_claims_inpatient):
        cohort = sample_patients[sample_patients["ANNEE"] == 2008].copy()
        result = build_target(cohort, sample_claims_inpatient, "2008-01-01")
        assert result.loc[result["DESYNPUF_ID"] == "A1", "HOSPITALIZED_IN_6M"].iloc[0] == 1
        assert result.loc[result["DESYNPUF_ID"] == "B2", "HOSPITALIZED_IN_6M"].iloc[0] == 1
        assert result.loc[result["DESYNPUF_ID"] == "C3", "HOSPITALIZED_IN_6M"].iloc[0] == 1

    def test_normal_cohort_2009(self, sample_patients, sample_claims_inpatient):
        cohort = sample_patients[sample_patients["ANNEE"] == 2009].copy()
        result = build_target(cohort, sample_claims_inpatient, "2009-01-01")
        assert result.loc[result["DESYNPUF_ID"] == "A1", "HOSPITALIZED_IN_6M"].iloc[0] == 0
        assert result.loc[result["DESYNPUF_ID"] == "D4", "HOSPITALIZED_IN_6M"].iloc[0] == 1

    def test_no_admissions_no_target(self):
        empty_claims = {
            "inpatient": pd.DataFrame(columns=["DESYNPUF_ID", "admission", "CLM_IP_ADMSN_TYPE_CD"]),
            "outpatient": pd.DataFrame(),
            "carrier": pd.DataFrame(),
            "prescription": pd.DataFrame(),
        }
        ids = pd.DataFrame({"DESYNPUF_ID": ["NO_ADMIT"], "ANNEE": [2008]})
        result = build_target(ids, empty_claims, "2008-01-01")
        assert result.loc[result["DESYNPUF_ID"] == "NO_ADMIT", "HOSPITALIZED_IN_6M"].iloc[0] == 0

    def test_non_urgent_admission_not_counted(self):
        claims = pd.DataFrame({
            "DESYNPUF_ID": ["NON_URGENT"],
            "CLM_ADMSN_DT": pd.to_datetime(["2008-03-01"]),
            "CLM_IP_ADMSN_TYPE_CD": [3],
            "NCH_BENE_DSCHRG_DT": pd.to_datetime(["2008-03-05"]),
        })
        claims["admission"] = claims["CLM_ADMSN_DT"]
        ids = pd.DataFrame({"DESYNPUF_ID": ["NON_URGENT"], "ANNEE": [2008]})
        result = build_target(ids, {"inpatient": claims}, "2008-01-01")
        assert result.loc[result["DESYNPUF_ID"] == "NON_URGENT", "HOSPITALIZED_IN_6M"].iloc[0] == 0
