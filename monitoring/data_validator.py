"""
Production-grade Data Validation (Great Expectations style).
Validates data quality before ingestion and training.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config.settings as settings

log = logging.getLogger(__name__)


VALIDATION_RULES = {
    "sp_features": {
        "cols": ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
                 "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
                 "SP_RA_OA", "SP_STRKETIA"],
        "dtype": "int",
        "min": 0, "max": 1,
    },
    "numerical_features": {
        "cols": ["AGE", "COUT_TOTAL", "CHARLSON_INDEX", "NB_COMORBIDITES",
                 "NB_HOSP_PASSEES", "NB_OP_3M", "NB_OP_6M", "NB_OP_12M",
                 "NB_CAR_6M", "NB_PRESCRIPTIONS", "NB_MOLECULES_UNIQUES"],
        "dtype": "float",
        "min": 0,
    },
    "categorical_features": {
        "cols": ["SEXE_ENC", "RACE_ENC", "BENE_ESRD_IND", "GROUPE_AGE_ENC",
                 "IS_NEW_PATIENT", "POLYPHARMACIE"],
        "dtype": "int",
        "min": 0,
    },
}


class DataValidator:
    """
    Validation des données en entrée.
    - Vérifie types, ranges, colonnes manquantes
    - Détecte les anomalies statistiques
    - Génère rapport structuré
    """

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or VALIDATION_RULES

    def validate_features(self, df: pd.DataFrame, dataset_name: str = "unknown") -> dict:
        """Validate a feature DataFrame against rules."""
        report = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "passed": True,
            "checks": [],
            "errors": [],
            "warnings": [],
        }

        self._check_missing_columns(df, report)
        self._check_dtypes(df, report)
        self._check_value_ranges(df, report)
        self._check_missing_values(df, report)
        self._check_constant_features(df, report)

        report["passed"] = len(report["errors"]) == 0
        report["quality_score"] = self._compute_quality_score(report)
        return report

    def validate_target(self, y: pd.Series, dataset_name: str = "target") -> dict:
        """Validate target variable."""
        report = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y),
            "passed": True,
            "checks": [],
            "errors": [],
            "warnings": [],
        }
        y_clean = pd.to_numeric(y, errors="coerce")
        nulls = y_clean.isna().sum()
        if nulls > 0:
            report["errors"].append(f"{nulls} null values in target")
            report["passed"] = False
        unique = y_clean.dropna().unique()
        if not set(unique).issubset({0, 1}):
            report["errors"].append(f"Target has values outside {{0,1}}: {unique}")
            report["passed"] = False
        pos_rate = y_clean.mean()
        report["checks"].append(f"Positive rate: {pos_rate:.4f}")
        if pos_rate < 0.01 or pos_rate > 0.99:
            report["warnings"].append(f"Unbalanced target: {pos_rate:.4f}")
        return report

    def _check_missing_columns(self, df: pd.DataFrame, report: dict):
        for group_name, group_rules in self.rules.items():
            for col in group_rules["cols"]:
                if col not in df.columns:
                    report["errors"].append(f"Missing column: {col} ({group_name})")
                    report["passed"] = False

    def _check_dtypes(self, df: pd.DataFrame, report: dict):
        for group_name, group_rules in self.rules.items():
            for col in group_rules["cols"]:
                if col in df.columns:
                    inferred = str(df[col].infer_objects().dtype)
                    expected = group_rules["dtype"]
                    if expected == "int" and "float" in inferred:
                        if df[col].dropna().apply(lambda x: x != int(x)).any():
                            report["warnings"].append(f"Column {col} has float values but expected int")
                    elif expected == "int" and "int" not in inferred:
                        report["warnings"].append(f"Column {col} expected int, got {inferred}")

    def _check_value_ranges(self, df: pd.DataFrame, report: dict):
        for group_name, group_rules in self.rules.items():
            for col in group_rules["cols"]:
                if col not in df.columns:
                    continue
                col_min = group_rules.get("min")
                col_max = group_rules.get("max")
                if col_min is not None and df[col].min() < col_min:
                    report["errors"].append(f"Column {col} has min {df[col].min()} < {col_min}")
                    report["passed"] = False
                if col_max is not None and df[col].max() > col_max:
                    report["errors"].append(f"Column {col} has max {df[col].max()} > {col_max}")
                    report["passed"] = False

    def _check_missing_values(self, df: pd.DataFrame, report: dict):
        for col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > 0.5:
                report["errors"].append(f"Column {col} has {null_pct:.1%} missing")
                report["passed"] = False
            elif null_pct > 0.1:
                report["warnings"].append(f"Column {col} has {null_pct:.1%} missing")

    def _check_constant_features(self, df: pd.DataFrame, report: dict):
        for col in df.columns:
            if df[col].nunique() == 1:
                report["warnings"].append(f"Constant feature: {col} (value={df[col].iloc[0]})")

    def _compute_quality_score(self, report: dict) -> float:
        error_penalty = len(report["errors"]) * 15
        warning_penalty = len(report["warnings"]) * 5
        score = max(0, 100 - error_penalty - warning_penalty)
        return score

    def save_report(self, report: dict, name: str = "validation"):
        path = settings.REPORTS_DIR / "validation" / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Validation report saved: %s", path)
        return path


validator = DataValidator()
