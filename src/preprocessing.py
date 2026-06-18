import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config.settings as settings

log = logging.getLogger(__name__)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable")


def sanitize_features(X: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame:
    X = X.copy()
    if feature_cols is not None:
        X = X.reindex(columns=list(feature_cols), fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.reset_index(drop=True)


def load_feature_names() -> list[str]:
    pkl = settings.MODELS_DIR / "feature_names.pkl"
    csv = settings.FEATURES_DIR / "feature_names.csv"
    if pkl.exists():
        import joblib
        return list(joblib.load(pkl))
    if csv.exists():
        return pd.read_csv(csv).iloc[:, 0].astype(str).tolist()
    return []


def load_train_val_test():
    log.info("Chargement des splits temporels...")
    paths = {
        "X_train": settings.FEATURES_DIR / "X_train.parquet",
        "X_val": settings.FEATURES_DIR / "X_val.parquet",
        "X_test": settings.FEATURES_DIR / "X_test.parquet",
        "y_train": settings.FEATURES_DIR / "y_train.parquet",
        "y_val": settings.FEATURES_DIR / "y_val.parquet",
        "y_test": settings.FEATURES_DIR / "y_test.parquet",
    }
    for p in paths.values():
        require_file(p)

    X_train = pd.read_parquet(paths["X_train"])
    X_val = pd.read_parquet(paths["X_val"])
    X_test = pd.read_parquet(paths["X_test"])

    feature_cols = load_feature_names()
    if not feature_cols:
        feature_cols = X_train.columns.astype(str).tolist()

    X_train = sanitize_features(X_train, feature_cols)
    X_val = sanitize_features(X_val, feature_cols)
    X_test = sanitize_features(X_test, feature_cols)

    def safe_target(p):
        df = pd.read_parquet(p)
        if isinstance(df, pd.DataFrame):
            c = settings.TARGET_COL if settings.TARGET_COL in df.columns else df.columns[0]
            y = df[c]
        else:
            y = df
        return pd.to_numeric(y, errors="coerce").fillna(0).astype(int).reset_index(drop=True)

    y_train = safe_target(paths["y_train"])
    y_val = safe_target(paths["y_val"])
    y_test = safe_target(paths["y_test"])

    log.info("Train: %s (pos: %.2f%%)", X_train.shape, y_train.mean() * 100)
    log.info("Val:   %s (pos: %.2f%%)", X_val.shape, y_val.mean() * 100)
    log.info("Test:  %s (pos: %.2f%%)", X_test.shape, y_test.mean() * 100)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
