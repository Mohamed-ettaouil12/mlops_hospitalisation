"""
Production-grade Feature Store.
Versioned, consistent preprocessing between training and inference.
"""
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config.settings as settings

log = logging.getLogger(__name__)

VERSION_FILE = "store_version.json"
FEATURE_STORE_FILE = "feature_store.parquet"
METADATA_FILE = "store_metadata.json"
SCALER_FILE = "feature_scaler.pkl"
COLUMN_ORDER_FILE = "column_order.json"


def _features_dir() -> Path:
    return settings.FEATURES_DIR


def _feature_store_path() -> Path:
    return _features_dir() / FEATURE_STORE_FILE


def _metadata_path() -> Path:
    return _features_dir() / METADATA_FILE


def _version_path() -> Path:
    return _features_dir() / VERSION_FILE


def _scaler_path() -> Path:
    return _features_dir() / SCALER_FILE


def _column_order_path() -> Path:
    return _features_dir() / COLUMN_ORDER_FILE


def compute_dataset_hash(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(raw.tobytes()).hexdigest()[:16]


class FeatureStore:
    """
    Feature Store versionné.

    - stocke les features sous format Parquet avec métadonnées
    - scaler sauvegardé pour inference cohérente
    - preprocessing pipeline identique train/inference
    - rollback possible vers version antérieure
    """

    def __init__(self):
        self._version = None
        self._metadata = {}
        self._scaler: Optional[StandardScaler] = None
        self._column_order: list[str] = []
        self._load_or_init()

    def _load_or_init(self):
        if _version_path().exists():
            with open(_version_path()) as f:
                ver = json.load(f)
                self._version = ver.get("version", "v0")
        else:
            self._version = "v0"

        if _scaler_path().exists():
            self._scaler = joblib.load(_scaler_path())

        if _column_order_path().exists():
            with open(_column_order_path()) as f:
                self._column_order = json.load(f)

        if _metadata_path().exists():
            with open(_metadata_path()) as f:
                self._metadata = json.load(f)

    @property
    def version(self) -> str:
        return self._version

    @property
    def scaler(self) -> Optional[StandardScaler]:
        return self._scaler

    @property
    def column_order(self) -> list[str]:
        return self._column_order

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def save(self, df: pd.DataFrame, scaler: StandardScaler, dataset_version: str = "v1"):
        _features_dir().mkdir(parents=True, exist_ok=True)

        df = df.reset_index(drop=True)
        self._column_order = list(df.columns)

        temp_path = _feature_store_path().with_suffix(".tmp.parquet")
        df.to_parquet(temp_path, index=False)
        shutil.move(str(temp_path), str(_feature_store_path()))

        joblib.dump(scaler, _scaler_path())
        with open(_column_order_path(), "w") as f:
            json.dump(self._column_order, f)

        new_version = f"v{int(self._version[1:]) + 1}" if self._version.startswith("v") else "v1"
        self._version = new_version
        self._scaler = scaler

        dataset_hash = compute_dataset_hash(df)
        self._metadata = {
            "version": new_version,
            "previous_version": self._version,
            "created_at": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_columns": len(self._column_order),
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "columns": self._column_order,
        }

        with open(_version_path(), "w") as f:
            json.dump({"version": new_version}, f)
        with open(_metadata_path(), "w") as f:
            json.dump(self._metadata, f, indent=2)

        log.info("Feature Store saved: %s (%d rows, %d cols)", new_version, len(df), len(self._column_order))

    def load_features(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        if not _feature_store_path().exists():
            raise FileNotFoundError(f"Feature store not found at {_feature_store_path()}")
        df = pd.read_parquet(_feature_store_path())
        if columns:
            missing = set(columns) - set(df.columns)
            if missing:
                for c in missing:
                    df[c] = 0.0
            df = df[columns]
        return df

    def transform_inference(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._column_order:
            data = data.reindex(columns=self._column_order, fill_value=0)
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        if self._scaler:
            numeric_cols = self._scaler.feature_names_in_.tolist()
            scale_cols = [c for c in numeric_cols if c in data.columns]
            if scale_cols:
                data[scale_cols] = self._scaler.transform(data[scale_cols])
        return data

    def get_version_info(self) -> dict:
        return {
            "version": self._version,
            "n_features": len(self._column_order),
            "has_scaler": self._scaler is not None,
            "metadata": self._metadata,
        }


feature_store = FeatureStore()


def preprocess_train(df_raw: pd.DataFrame, target_col: str = settings.TARGET_COL) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Preprocess training data and update feature store."""
    from src.preprocessing import sanitize_features

    feature_cols = [c for c in df_raw.columns if c != target_col]
    X = sanitize_features(df_raw[feature_cols], feature_cols)
    y = pd.to_numeric(df_raw[target_col], errors="coerce").fillna(0).astype(int).reset_index(drop=True)

    scale_cols = [
        "AGE", "COUT_TOTAL", "CHARLSON_INDEX", "NB_COMORBIDITES",
        "NB_HOSP_PASSEES", "NB_OP_3M", "NB_OP_6M", "NB_OP_12M",
        "NB_CAR_6M", "NB_PRESCRIPTIONS", "NB_MOLECULES_UNIQUES",
    ]
    scale_cols = [c for c in scale_cols if c in X.columns]
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    feature_store.save(X, scaler)
    return X, y, scaler


def preprocess_inference(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess inference data using stored feature store."""
    return feature_store.transform_inference(data)
