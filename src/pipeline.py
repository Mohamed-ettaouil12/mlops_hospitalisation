# ═══════════════════════════════════════════════════════════
# Pipeline MLOps principal
# ═══════════════════════════════════════════════════════════

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = Path("logs")
DATA_CLEANED_DIR = Path("data/cleaned")
DATA_FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

for directory in [LOG_DIR, DATA_CLEANED_DIR, DATA_FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def print_header(title: str) -> None:
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


def run_step(step_name: str, func: Callable, stop_on_error: bool = True) -> Any:
    log.info("")
    log.info("-" * 70)
    log.info("START %s", step_name)
    log.info("-" * 70)

    start = time.time()
    try:
        result = func()
        duration = time.time() - start
        log.info("OK %s termine en %.2fs", step_name, duration)
        return result
    except Exception as exc:
        duration = time.time() - start
        log.error("ECHEC %s apres %.2fs", step_name, duration)
        log.error("Erreur : %s", exc)
        log.error(traceback.format_exc())
        if stop_on_error:
            sys.exit(1)
        return None


def check_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} introuvable : {path}")
    log.info("OK fichier %s : %s", label, path)


def check_cleaning_outputs() -> None:
    expected_files = {
        "patients_cleaned": DATA_CLEANED_DIR / "patients_cleaned.parquet",
        "claims_inpatient": DATA_CLEANED_DIR / "claims_inpatient_cleaned.parquet",
        "claims_outpatient": DATA_CLEANED_DIR / "claims_outpatient_cleaned.parquet",
        "claims_carrier": DATA_CLEANED_DIR / "claims_carrier_cleaned.parquet",
        "claims_prescription": DATA_CLEANED_DIR / "claims_prescription_cleaned.parquet",
    }
    for label, path in expected_files.items():
        check_file(path, label)


def check_preprocessing_outputs() -> None:
    expected_files = {
        "X_train": DATA_FEATURES_DIR / "X_train.parquet",
        "X_val": DATA_FEATURES_DIR / "X_val.parquet",
        "X_test": DATA_FEATURES_DIR / "X_test.parquet",
        "y_train": DATA_FEATURES_DIR / "y_train.parquet",
        "y_val": DATA_FEATURES_DIR / "y_val.parquet",
        "y_test": DATA_FEATURES_DIR / "y_test.parquet",
        "feature_names": DATA_FEATURES_DIR / "feature_names.csv",
        "preprocessing_report": DATA_FEATURES_DIR / "preprocessing_report.json",
        "scaler": MODELS_DIR / "scaler.pkl",
        "feature_names_pkl": MODELS_DIR / "feature_names.pkl",
    }
    for label, path in expected_files.items():
        check_file(path, label)


def check_training_outputs() -> None:
    expected_files = {
        "best_model": MODELS_DIR / "best_model.pkl",
        "best_model_info": MODELS_DIR / "best_model_info.json",
    }
    for label, path in expected_files.items():
        check_file(path, label)


def step_validation() -> bool:
    import src.validate_data  # noqa: F401

    return True


def step_cleaning() -> Tuple[Any, Any]:
    from src.data_cleaning import main as clean

    df_clean, claims = clean()
    check_cleaning_outputs()
    log.info("Patients clean shape : %s", df_clean.shape)
    log.info("Claims disponibles : %s", list(claims.keys()))
    return df_clean, claims


def step_preprocessing():
    from src.data_preprocessing import main as preprocess

    result = preprocess()
    if not isinstance(result, tuple) or len(result) != 7:
        raise ValueError(
            "data_preprocessing.main() doit retourner "
            "(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)"
        )

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = result
    check_preprocessing_outputs()
    log.info("X_train : %s", X_train.shape)
    log.info("X_val   : %s", X_val.shape)
    log.info("X_test  : %s", X_test.shape)
    log.info("y_train : %s", len(y_train))
    log.info("y_val   : %s", len(y_val))
    log.info("y_test  : %s", len(y_test))
    log.info("Features : %s", len(feature_cols))
    return result


def step_training():
    from src.train import main as train

    result = train()
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("train.main() doit retourner (best_model, best_name, best_auc)")

    best_model, best_name, best_auc = result
    check_training_outputs()
    log.info("Best model : %s", best_name)
    log.info("Best AUC   : %.4f", best_auc)
    return best_model, best_name, best_auc


def main():
    total_start = time.time()
    print_header("PIPELINE MLOPS — DEMARRAGE PRODUCTION")
    log.info("Ordre : Validation -> Cleaning -> Preprocessing -> Training")

    run_step("Etape 1/4 : Validation des donnees raw", step_validation)
    df_clean, _claims = run_step("Etape 2/4 : Data Cleaning", step_cleaning)
    X_train, X_val, X_test, _y_train, _y_val, _y_test, feature_cols = run_step(
        "Etape 3/4 : Data Preprocessing",
        step_preprocessing,
    )
    _best_model, best_name, best_auc = run_step("Etape 4/4 : Training modeles", step_training)

    total_duration = time.time() - total_start
    print_header("PIPELINE MLOPS COMPLET — TERMINE")
    log.info("Duree totale : %.2fs", total_duration)
    log.info("Patients cleaned : %s", f"{df_clean.shape[0]:,}")
    log.info("Train : %s", X_train.shape)
    log.info("Validation : %s", X_val.shape)
    log.info("Test : %s", X_test.shape)
    log.info("Nombre features : %s", len(feature_cols))
    log.info("Meilleur modele : %s", best_name)
    log.info("AUC validation : %.4f", best_auc)
    log.info("Prochaine commande : mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
