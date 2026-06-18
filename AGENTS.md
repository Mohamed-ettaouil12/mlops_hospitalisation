# AGENTS.md — MLOps Hospitalisation (Production v3.0)

## Architecture (Container)
- Docker Compose: FastAPI (v3.0), MLflow (v2.16.0), Airflow (2.8.0), Prometheus, Grafana (11.2.0)
- Start order: MLflow (healthy) → API + Airflow (depends_on service_healthy)

## Model Lifecycle
- **Champion** (Production) / **Challenger** (Staging) pattern
- Automatic promotion if challenger AUC >= champion
- Automatic rollback check via `should_rollback()` (falls back to local pkl)
- Cleanup: keeps last 10 versions, archives older ones
- Registry: `src/model_registry.py` → `ModelRegistry` class

## Feature Store
- Versioned Parquet store at `data/features/feature_store.parquet`
- Scaler saved as `data/features/feature_scaler.pkl`
- Column order saved as `data/features/column_order.json`
- Consistent preprocessing between train/inference via `FeatureStore.transform_inference()`

## API Endpoints (FastAPI v3.0.0)
- `GET /health` — full status with champion/challenger info
- `GET /model/info` — current model metadata
- `POST /predict` — prediction with A/B testing (90% champion, 10% challenger)
- `POST /predict/champion` — force champion prediction
- `POST /drift/score` — receive drift metrics from Airflow → Prometheus
- `POST /accuracy/update` — receive accuracy metrics from Airflow → Prometheus
- `GET /metrics` — Prometheus format

## Prometheus Metrics
- `api_requests_total` (method, endpoint, status)
- `api_request_latency_seconds` (method, endpoint) — buckets to 10s
- `api_errors_total` (method, endpoint)
- `api_predictions_total` (risk_level, model_version)
- `api_ab_test_assignments` (model_a, model_b, chosen)
- `model_drift_score` (model_version)
- `model_accuracy_tracking` (model_version, metric)

## Airflow DAGs (4 production DAGs)
1. **ingestion_validation_dag** — daily: validate raw data (Great Expectations style) → feature engineering
2. **drift_detection_dag** — daily: Evidently AI + PSI → push drift to API → decide retrain
3. **retrain_model_dag** — weekly (Sun 2am): train → compare → promote → push metrics → cleanup
4. **auto_retraining** — daily: full pipeline (val → train → registry → compare → promote)
5. **hospitalization_pipeline** — weekly: clean → features → validate → train → drift → healthcheck

## Drift Detection
- Evidently AI reports: DataDrift, TargetDrift, DataQuality
- PSI-based feature drift per column
- Severity levels: NONE / MODERATE / HIGH / CRITICAL
- Archived history in `reports/drift/archive/`
- Scores pushed to Prometheus via API

## Data Validation
- Custom DataValidator class (Great Expectations pattern)
- Checks: column presence, dtypes, value ranges, missing %, constant features
- Quality score (0-100) with error/warning penalties
- Reports saved to `reports/validation/`

## Key Commands
```bash
# Start full stack
docker compose up -d

# Seed MLflow with initial model
docker compose run --rm init

# Rebuild and start
docker compose up -d --build

# Test prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"AGE":75,"SEXE_ENC":0,"RACE_ENC":2,"BENE_ESRD_IND":0,"GROUPE_AGE_ENC":2,"SP_ALZHDMTA":0,"SP_CHF":0,"SP_CHRNKIDN":0,"SP_CNCR":0,"SP_COPD":0,"SP_DEPRESSN":0,"SP_DIABETES":1,"SP_ISCHMCHT":0,"SP_OSTEOPRS":0,"SP_RA_OA":0,"SP_STRKETIA":0,"NB_COMORBIDITES":2,"CHARLSON_INDEX":3,"COUT_TOTAL":12000,"IS_NEW_PATIENT":0,"NB_HOSP_PASSEES":1,"NB_OP_3M":0,"NB_OP_6M":1,"NB_OP_12M":2,"NB_CAR_6M":0,"NB_PRESCRIPTIONS":5,"NB_MOLECULES_UNIQUES":4,"POLYPHARMACIE":0,"SP_STATE_CODE":10}'

# Check health
curl -s http://localhost:8000/health | python3 -m json.tool

# Airflow (admin/admin)
open http://localhost:8080

# Grafana (admin/admin)
open http://localhost:3000

# Re-seed MLflow (if DB cleared)
docker compose run --rm init
```
