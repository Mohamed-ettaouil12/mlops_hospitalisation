"""
Production-grade FastAPI for MLOps Hospitalization Risk.
- Champion/Challenger A/B testing
- Automatic rollback detection
- MLflow registry with local fallback
- Prometheus metrics enriched (drift, accuracy, latency)
- Health endpoints with detailed status
"""
import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config.settings as settings
from api.schemas import PatientData, PredictionResult, ModelInfo, DriftScoreRequest, AccuracyUpdateRequest

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"],
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0))
ERROR_COUNT = Counter("api_errors_total", "Total errors", ["method", "endpoint"])
PREDICTION_COUNT = Counter("api_predictions_total", "Total predictions", ["risk_level", "model_version"])
MODEL_DRIFT_SCORE = Gauge("model_drift_score", "Current drift score (PSI max)", ["model_version"])
MODEL_ACCURACY = Gauge("model_accuracy_tracking", "Tracked model accuracy metrics", ["model_version", "metric"])
AB_TEST_COUNTER = Counter("api_ab_test_assignments", "A/B test assignments", ["model_a", "model_b", "chosen"])

model_state = {
    "champion": {"model": None, "version": "none", "stage": "none"},
    "challenger": {"model": None, "version": "none", "stage": "none"},
    "feature_names": [],
    "threshold": 0.25,
    "scaler": None,
    "mlflow_available": False,
    "loaded": False,
    "ab_test_enabled": False,
    "challenger_traffic_pct": 0,
}

CHAMPION_TRAFFIC_PCT = 90
CHALLENGER_TRAFFIC_PCT = 10


def _load_metadata():
    state = model_state
    feat_path = settings.FEATURES_DIR / "feature_names.csv"
    if feat_path.exists():
        state["feature_names"] = pd.read_csv(feat_path).iloc[:, 0].tolist()
    thresh_path = settings.MODELS_DIR / "best_threshold.json"
    if thresh_path.exists():
        state["threshold"] = float(json.loads(thresh_path.read_text()).get("threshold", 0.25))
    scaler_path = settings.MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        state["scaler"] = joblib.load(scaler_path)


def _load_local_fallback():
    state = model_state
    local_paths = [
        settings.MODELS_DIR / "production_model.pkl",
        settings.MODELS_DIR / "best_recall_model.pkl",
        settings.MODELS_DIR / "best_model.pkl",
        settings.MODELS_DIR / "lgb.pkl",
    ]
    for p in local_paths:
        if p.exists():
            log.info("Local fallback: loading from %s", p)
            try:
                model = joblib.load(p)
                state["champion"]["model"] = model
                state["champion"]["version"] = "local"
                state["champion"]["stage"] = "local"
                state["loaded"] = True
                return True
            except Exception:
                continue
    log.warning("No local model found")
    return False


def _load_from_mlflow(stage: str) -> dict | None:
    try:
        import mlflow.pyfunc
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(settings.MLFLOW_MODEL_NAME, stages=[stage])
        if versions:
            mv = versions[0]
            model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{mv.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            log.info("Loaded %s: v%s", stage, mv.version)
            return {"model": model, "version": mv.version, "stage": stage}
    except Exception as e:
        log.debug("MLflow %s load failed: %s", stage, e)
    return None


def load_models():
    state = model_state
    try:
        champion = _load_from_mlflow("Production")
        if champion:
            state["champion"] = champion
            state["mlflow_available"] = True
            state["loaded"] = True
            challenger = _load_from_mlflow("Staging")
            if challenger and challenger["version"] != champion["version"]:
                state["challenger"] = challenger
                state["ab_test_enabled"] = True
                state["challenger_traffic_pct"] = CHALLENGER_TRAFFIC_PCT
                log.info("A/B testing enabled: champion v%s (90%%) / challenger v%s (10%%)",
                         champion["version"], challenger["version"])
            else:
                state["ab_test_enabled"] = False
                state["challenger_traffic_pct"] = 0
            _load_metadata()
            return True
    except Exception as e:
        log.warning("MLflow load failed: %s", e)
    if not state["loaded"]:
        return _load_local_fallback() or False
    return state["loaded"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _retry_loop():
        for i in range(60):
            if model_state["loaded"]:
                MODEL_DRIFT_SCALE = model_state.get("champion", {}).get("version", "none")
                MODEL_DRIFT_SCORE.labels(model_version=MODEL_DRIFT_SCALE).set(0.0)
                return
            log.info("Model load attempt %d/60", i + 1)
            success = await asyncio.to_thread(load_models)
            if not success:
                await asyncio.sleep(10)
        if not model_state["loaded"]:
            log.warning("No model loaded after 60 attempts, degraded mode")
    asyncio.create_task(_retry_loop())
    yield


app = FastAPI(title="MLOps Hospitalization Risk", version="3.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    start = time.time()
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(response.status_code)).inc()
        if response.status_code >= 500:
            ERROR_COUNT.labels(method=method, endpoint=endpoint).inc()
        return response
    except Exception:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
        ERROR_COUNT.labels(method=method, endpoint=endpoint).inc()
        raise
    finally:
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    s = model_state
    champ = s.get("champion", {})
    chall = s.get("challenger", {})
    return {
        "status": "healthy" if s["loaded"] else "degraded",
        "model_loaded": s["loaded"],
        "mlflow_available": s["mlflow_available"],
        "champion": {"version": f"v{champ.get('version', 'none')}", "stage": champ.get("stage", "none")},
        "challenger": {"version": f"v{chall.get('version', 'none')}" if chall.get("model") else None, "stage": chall.get("stage", "none")},
        "ab_test_enabled": s.get("ab_test_enabled", False),
        "n_features": len(s["feature_names"]),
        "threshold": s["threshold"],
        "api_version": "3.0.0",
    }


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    s = model_state
    if not s["loaded"]:
        raise HTTPException(503, "Model not loaded")
    champ = s["champion"]
    return ModelInfo(
        type=type(champ["model"]).__name__ if champ["model"] else "none",
        version=f"v{champ['version']}",
        stage=champ["stage"],
        nb_features=len(s["feature_names"]),
        features=s["feature_names"],
    )


def _select_model(s: dict) -> tuple:
    """A/B test model selection."""
    if s.get("ab_test_enabled") and s["challenger"]["model"] is not None:
        if random.randint(1, 100) <= s["challenger_traffic_pct"]:
            return s["challenger"], "challenger"
    return s["champion"], "champion"


@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientData):
    s = model_state
    if not s["loaded"]:
        raise HTTPException(503, "Model not loaded yet")
    try:
        active_model, model_role = _select_model(s)
        version = active_model["version"]
        data = pd.DataFrame([patient.model_dump()])
        data = data.reindex(columns=s["feature_names"], fill_value=0)
        data = data.astype({c: "float64" for c in data.columns})
        if s["scaler"] is not None:
            scale_cols = ["AGE", "COUT_TOTAL", "CHARLSON_INDEX", "NB_COMORBIDITES",
                          "NB_HOSP_PASSEES", "NB_OP_3M", "NB_OP_6M", "NB_OP_12M",
                          "NB_CAR_6M", "NB_PRESCRIPTIONS", "NB_MOLECULES_UNIQUES"]
            scale_cols = [c for c in scale_cols if c in data.columns]
            if scale_cols:
                data[scale_cols] = s["scaler"].transform(data[scale_cols])
        proba = float(active_model["model"].predict(data)[0])
        proba = max(0.0, min(1.0, proba))
        seuil = s["threshold"]
        seuil_modere = max(0.0, min(1.0, seuil * 0.5))
        if proba >= seuil:
            risque = "ÉLEVÉ"
        elif proba >= seuil_modere:
            risque = "MODÉRÉ"
        else:
            risque = "FAIBLE"
        PREDICTION_COUNT.labels(risk_level=risque, model_version=f"v{version}").inc()
        if model_role == "challenger":
            AB_TEST_COUNTER.labels(
                model_a=f"v{s['champion']['version']}",
                model_b=f"v{version}",
                chosen="challenger",
            ).inc()
        return PredictionResult(
            probabilite=round(proba, 4),
            risque=risque,
            seuil_utilise=seuil,
            version_modele=f"v{version}",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Predict error: %s", str(e)[:200])
        raise HTTPException(500, str(e))


@app.post("/predict/champion", response_model=PredictionResult)
def predict_champion(patient: PatientData):
    """Force prediction using champion model (no A/B)."""
    s = model_state
    if not s["loaded"] or not s["champion"]["model"]:
        raise HTTPException(503, "Champion model not loaded")
    active_model = s["champion"]
    version = active_model["version"]
    try:
        data = pd.DataFrame([patient.model_dump()])
        data = data.reindex(columns=s["feature_names"], fill_value=0)
        data = data.astype({c: "float64" for c in data.columns})
        if s["scaler"] is not None:
            scale_cols = ["AGE", "COUT_TOTAL", "CHARLSON_INDEX", "NB_COMORBIDITES",
                          "NB_HOSP_PASSEES", "NB_OP_3M", "NB_OP_6M", "NB_OP_12M",
                          "NB_CAR_6M", "NB_PRESCRIPTIONS", "NB_MOLECULES_UNIQUES"]
            scale_cols = [c for c in scale_cols if c in data.columns]
            if scale_cols:
                data[scale_cols] = s["scaler"].transform(data[scale_cols])
        proba = float(active_model["model"].predict(data)[0])
        proba = max(0.0, min(1.0, proba))
        seuil = s["threshold"]
        seuil_modere = max(0.0, min(1.0, seuil * 0.5))
        if proba >= seuil:
            risque = "ÉLEVÉ"
        elif proba >= seuil_modere:
            risque = "MODÉRÉ"
        else:
            risque = "FAIBLE"
        return PredictionResult(
            probabilite=round(proba, 4),
            risque=risque,
            seuil_utilise=seuil,
            version_modele=f"v{version}",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Champion predict error: %s", str(e)[:200])
        raise HTTPException(500, str(e))


@app.post("/drift/score")
def update_drift_score(request: DriftScoreRequest):
    """Endpoint for Airflow to push drift scores to Prometheus."""
    MODEL_DRIFT_SCORE.labels(model_version=request.model_version).set(request.drift_max_psi)
    return {"drift_score_updated": True, "model_version": request.model_version, "psi": request.drift_max_psi}


@app.post("/accuracy/update")
def update_accuracy_metrics(request: AccuracyUpdateRequest):
    """Endpoint for Airflow to push accuracy metrics to Prometheus."""
    for metric_name, metric_value in request.values.items():
        MODEL_ACCURACY.labels(model_version=request.model_version, metric=metric_name).set(metric_value)
    return {"accuracy_updated": True, "model_version": request.model_version}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
