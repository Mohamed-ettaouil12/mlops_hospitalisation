# Requirements Document

## Introduction

This document defines the requirements to make the existing MLOps hospitalisation project fully production-ready. The project serves a CatBoost/LightGBM-based hospital admission risk prediction API backed by MLflow model registry, Airflow orchestration, Prometheus metrics collection, and Grafana dashboards, all running in Docker Compose. The current system has several broken or incomplete subsystems: the FastAPI is unstable under load, Prometheus metrics are incomplete, Grafana shows "No data" on critical panels, Evidently drift detection is not active end-to-end, Airflow DAGs are not fully wired, MLflow model loading is unreliable, A/B testing metrics are not populated, and the traffic simulation script does not match the real API schema. All of these defects must be resolved with production-grade, non-pseudocode Python 3.11 compatible with the existing Docker Compose stack.

---

## Glossary

- **API**: The FastAPI application in `api/main.py`, served on port 8000.
- **Champion**: The model version currently in MLflow stage `Production`, used for ≥80% of traffic.
- **Challenger**: The model version currently in MLflow stage `Staging`, used for ≤20% of traffic when A/B testing is enabled.
- **Prometheus**: The Prometheus metrics server scraping `api:8000/metrics`, configured in `prometheus/prometheus.yml`.
- **Grafana**: The Grafana dashboard server reading from Prometheus, provisioned from `grafana/provisioning/`.
- **Drift_Detector**: The module `monitoring/drift_detector.py` implementing PSI-based feature drift and Evidently AI reports.
- **Model_Registry**: The class `ModelRegistry` in `src/model_registry.py` managing MLflow model lifecycle.
- **Airflow_Scheduler**: The Apache Airflow scheduler running the DAGs in `dags/`.
- **Traffic_Generator**: The Python script `scripts/traffic_generator.py` that simulates realistic API traffic.
- **Model_State**: The in-process dictionary in `api/main.py` holding all loaded model objects and configuration.
- **PSI**: Population Stability Index, a numeric drift measure per feature column.
- **A/B_Test**: The routing mechanism that sends a configurable fraction of `/predict` traffic to the Challenger model.

---

## Requirements

### Requirement 1: FastAPI Stability and Health Endpoint

**User Story:** As an on-call engineer, I want the `/health` endpoint to always return a valid JSON response with HTTP 200, so that load balancers and monitoring systems never receive a connection error or 5xx from a health probe.

#### Acceptance Criteria

1. THE API SHALL expose `GET /health` that always returns HTTP 200 with a JSON body containing at minimum the keys `status` (one of `"ok"` or `"degraded"`), `model_loaded`, `mlflow_available`, `champion`, `challenger`, `ab_test_enabled`, `n_features`, `threshold`, and `api_version`.
2. WHEN the Champion model is not yet loaded during startup OR after all retry attempts are exhausted, THE API SHALL return a JSON body containing all 9 required keys with `"status": "degraded"` and `"model_loaded": false` with HTTP 200 rather than HTTP 503 or a connection error.
3. IF an unhandled exception occurs inside a request handler, THEN THE API SHALL log the error with its traceback, increment the `api_errors_total` counter, and return an HTTP 500 JSON body with an `"error"` key instead of dropping the TCP connection.
4. THE API lifespan SHALL retry model loading up to 60 times with a 10-second delay between attempts; WHEN all 60 attempts fail, THE API SHALL enter degraded mode and cease retry attempts so that the process does not loop indefinitely.
5. WHILE the model retry loop is running, THE API SHALL respond to `GET /health` and `GET /metrics` within 2 seconds without waiting for the retry loop to complete.

---

### Requirement 2: Complete Prometheus Metrics Endpoint

**User Story:** As a platform engineer, I want the `/metrics` endpoint to expose all defined Prometheus metrics in the standard text format, so that Prometheus can scrape them and Grafana can display data on every dashboard panel.

#### Acceptance Criteria

1. THE API SHALL expose `GET /metrics` returning the Prometheus text exposition format with `Content-Type: text/plain; version=0.0.4`.
2. THE API SHALL register and maintain the following metrics families at startup: `api_requests_total` (Counter, labels: method, endpoint, status), `api_request_latency_seconds` (Histogram, labels: method, endpoint, buckets to 10 s), `api_errors_total` (Counter, labels: method, endpoint), `api_predictions_total` (Counter, labels: risk_level, model_version), `model_drift_score` (Gauge, label: model_version), `model_accuracy_tracking` (Gauge, labels: model_version, metric), and `api_ab_test_assignments` (Counter, labels: model_a, model_b, chosen).
3. WHEN the API starts and the Champion version is known, THE API SHALL call `model_drift_score.labels(model_version=<version>).set(0.0)` to initialise the gauge so it appears immediately in Prometheus before any drift push occurs.
4. WHEN `api_requests_total` is scraped, THE metric SHALL contain at least one sample per `(method, endpoint, status)` combination that has been observed since the last restart.
5. THE Prometheus scrape job named `mlops_api` in `prometheus/prometheus.yml` SHALL target `api:8000` on path `/metrics` with a scrape interval of 10 seconds and the label `job="mlops_api"` applied, so Grafana label filters work without additional relabeling.

---

### Requirement 3: Grafana "No Data" Resolution

**User Story:** As a data scientist, I want every Grafana dashboard panel to show live data instead of "No data", so that I can monitor prediction rates, latency, drift, and A/B testing in real time.

#### Acceptance Criteria

1. THE Grafana provisioning SHALL include a datasource named `Prometheus` with `uid: prometheus_ds` pointing to `http://prometheus:9090`, matching the `uid` referenced in every panel target in `grafana/dashboards/api_dashboard.json`.
2. WHEN Prometheus has been scraping for at least 60 seconds, THE panels for "Requests per second", "Latency P50/P95/P99", "Error rate (5xx)", "Predictions by risk level", "Total requests / Errors / Latency", and "A/B test assignments" SHALL have at least one data series visible if traffic has been generated.
3. THE "Drift score (PSI max)" panel SHALL display data after `POST /drift/score` has been called at least once, because the gauge is initialised at startup and updated on each drift push.
4. THE "Model accuracy (AUC / Recall)" panel SHALL display data after `POST /accuracy/update` has been called at least once with metric keys `auc_roc` and `recall`.
5. THE Grafana dashboard JSON SHALL set `"schemaVersion": 38` and use panel-level `"datasource": {"type": "prometheus", "uid": "prometheus_ds"}` on every target, not only on the panel root, so Grafana 11.x correctly resolves the datasource.
6. THE Grafana provisioning dashboards YAML SHALL set `"updateIntervalSeconds": 30` and `"allowUiUpdates": true` so dashboards reload and users can edit panels in the UI.

---

### Requirement 4: Drift Detection — End-to-End Activation

**User Story:** As an MLOps engineer, I want drift detection to run end-to-end — from PSI calculation through Prometheus gauge update — so that I receive an accurate, up-to-date drift signal without manual intervention.

#### Acceptance Criteria

1. THE Drift_Detector SHALL calculate PSI for every feature in `feature_names.csv` using reference data (year 2008) and production data (year 2010) from `features_engineered.parquet`.
2. WHEN a feature's PSI exceeds `settings.PSI_THRESHOLD` (default 0.20), THE Drift_Detector SHALL include that feature in the `drifted_features` list of the result dict.
3. THE Drift_Detector SHALL return a result dict containing at minimum: `drift_detected` (bool), `drift_severity` (one of NONE / MODERATE / HIGH / CRITICAL), `max_psi` (float, rounded to 4 decimal places), `n_drifted` (int), `drift_pct` (float), `drifted_features` (list), `timestamp` (ISO-8601 string).
4. WHEN `detect_drift()` completes, THE Drift_Detector SHALL write the result dict to `reports/drift/drift_check_result.json` and append a single JSON line to `reports/drift/drift_history.jsonl`.
5. THE `drift_detection_dag` in Airflow SHALL execute the tasks in the order: `run_drift_detection` → `push_drift_to_api` → `save_drift_report` → `decide_retrain` → `[trigger_retrain | skip_retrain]`.
6. WHEN `push_drift_to_api` executes, THE Airflow_Scheduler SHALL make an HTTP POST to `http://api:8000/drift/score` with JSON body `{"drift_max_psi": <float>, "model_version": <string>}`, and THE API SHALL store the value in the `model_drift_score` Prometheus gauge labelled with the model version.
7. IF the HTTP POST to `/drift/score` fails with a connection error, THEN THE Airflow_Scheduler SHALL log a warning and continue without failing the task, so the DAG run is not marked as failed due to a transient API unavailability.
8. WHERE Evidently AI is installed, THE Drift_Detector SHALL generate HTML reports for DataDrift, TargetDrift, and DataQuality and save them under `reports/drift/`.

---

### Requirement 5: Airflow DAG Completeness

**User Story:** As an MLOps engineer, I want both the retraining DAG and the drift detection DAG to be fully wired and triggerable, so that the system can retrain automatically when drift exceeds the threshold and models are kept current.

#### Acceptance Criteria

1. THE `retrain_model_dag` SHALL execute tasks in the order: `load_and_preprocess` → `train_new_model` → `compare_and_promote` → `push_metrics_to_api` → `cleanup_old_models`.
2. WHEN `push_metrics_to_api` executes, THE Airflow_Scheduler SHALL POST to `http://api:8000/accuracy/update` with the metric dict from `compare_and_promote`'s XCom output, containing at minimum the keys `auc_roc` and `recall` as float values.
3. THE `drift_detection_dag` SHALL be scheduled with `schedule_interval="@daily"` and `catchup=False`, and SHALL be unpaused in the Airflow web UI on first deployment.
4. THE `retrain_model_dag` SHALL be scheduled with `schedule_interval="0 2 * * 0"` (weekly Sunday 2 AM) and `catchup=False`.
5. WHEN the `decide_retrain` BranchPythonOperator executes, THE Airflow_Scheduler SHALL route to `trigger_retrain` if `drift_severity` is `HIGH` or `CRITICAL`, and SHALL route to `skip_retrain` otherwise.
6. THE `trigger_retrain` TriggerDagRunOperator SHALL reference `trigger_dag_id="retrain_model_dag"` with `wait_for_completion=False` so the drift DAG does not block on retraining.
7. IF any task fails with a retryable error, THEN THE Airflow_Scheduler SHALL retry up to 2 times with a 3-minute delay before marking the task as failed.

---

### Requirement 6: MLflow Model Loading Reliability

**User Story:** As a DevOps engineer, I want the API to reliably load the production model from MLflow on startup, and fall back to a local `.pkl` file when MLflow is temporarily unavailable, so that the API always serves predictions without a manual restart.

#### Acceptance Criteria

1. WHEN the API starts, THE Model_State SHALL attempt to load the Champion model from MLflow stage `Production` using `mlflow.pyfunc.load_model`.
2. IF MLflow is unreachable or returns no `Production` model, THEN THE Model_State SHALL attempt to load from local fallback paths in the order: `models/production_model.pkl`, `models/best_recall_model.pkl`, `models/best_model.pkl`, `models/lgb.pkl`.
3. WHEN a local fallback model is loaded, THE Model_State SHALL set `champion.version = "local"` and `mlflow_available = False` so `/health` accurately reflects the source.
4. THE `scripts/init_mlflow.py` seeding script SHALL register a trained LightGBM model as version 1, transition it to `Production` stage, and exit with code 0, so the `init` Docker Compose profile can be used to seed a fresh MLflow database.
5. THE Model_Registry `compare_and_decide` method SHALL transition the winning model version to MLflow stage `Production` using `client.transition_model_version_stage`, and SHALL transition the previous champion to `Archived`, so that `client.get_latest_versions(stage=["Production"])` always returns exactly one version.
6. WHEN `cleanup_old_versions(keep_last=10)` executes, THE Model_Registry SHALL transition all versions beyond the most recent 10 that are not in `Production` or `Staging` stage to `Archived`.

---

### Requirement 7: A/B Testing Metrics

**User Story:** As a data scientist, I want A/B testing assignments to be logged as Prometheus metrics so that I can compare champion and challenger prediction rates on the Grafana dashboard.

#### Acceptance Criteria

1. WHEN the API loads both a Champion and a Challenger model, THE Model_State SHALL set `ab_test_enabled = True` and `challenger_traffic_pct = 10` (10% of `/predict` traffic routed to the Challenger).
2. WHEN a `/predict` request is routed to the Challenger, THE API SHALL increment `api_ab_test_assignments` Counter with labels `model_a=<champion_version>`, `model_b=<challenger_version>`, `chosen="challenger"`.
3. WHEN a `/predict` request is routed to the Champion while A/B testing is active, THE API SHALL increment `api_ab_test_assignments` Counter with labels `model_a=<champion_version>`, `model_b=<challenger_version>`, `chosen="champion"`.
4. WHEN only the Champion model is loaded (no Challenger), THE Model_State SHALL set `ab_test_enabled = False` and `challenger_traffic_pct = 0`, and THE API SHALL not increment `api_ab_test_assignments`.
5. THE `POST /predict` endpoint SHALL include `"ab_test_role"` in its response JSON, indicating which model (`"champion"` or `"challenger"`) served the request, so callers can log per-model outcomes.
6. WHEN `GET /health` is called, THE API SHALL include `"ab_test_enabled"` and `"challenger_traffic_pct"` in the response body.

---

### Requirement 8: Production-Grade Traffic Simulation Script

**User Story:** As a developer, I want a realistic traffic simulation script that sends correctly-formed requests to the live API at configurable rates, so that I can seed Prometheus with real data and verify Grafana panels end-to-end.

#### Acceptance Criteria

1. THE Traffic_Generator SHALL send HTTP requests to a configurable `BASE_URL` (default `http://localhost:8000`) using the distribution: 60% `POST /predict`, 20% `GET /health`, 10% `GET /metrics`, 5% `POST /drift/score`, 5% `GET /model/info`.
2. WHEN generating a `/predict` payload, THE Traffic_Generator SHALL produce a JSON object matching the `PatientData` schema with all required fields (`AGE`, `SEXE_ENC`, `RACE_ENC`, `BENE_ESRD_IND`, `GROUPE_AGE_ENC`, all `SP_*` binary fields, `SP_STATE_CODE`, `NB_COMORBIDITES`, `CHARLSON_INDEX`, `COUT_TOTAL`, `IS_NEW_PATIENT`, `NB_HOSP_PASSEES`, `NB_OP_3M`, `NB_OP_6M`, `NB_OP_12M`, `NB_CAR_6M`, `NB_PRESCRIPTIONS`, `NB_MOLECULES_UNIQUES`, `POLYPHARMACIE`) with values in their valid ranges.
3. THE Traffic_Generator SHALL introduce a simulated latency by sleeping a random duration between 0.05 s and 2.0 s between requests.
4. THE Traffic_Generator SHALL introduce deliberate error scenarios: 5% of `/predict` calls SHALL send an intentionally malformed payload (missing required field or out-of-range value) to generate 4xx responses and exercise the error counter.
5. THE Traffic_Generator SHALL support a `--burst` CLI flag that sends 20 concurrent requests in a thread pool before returning to normal rate, to test API behaviour under load.
6. WHEN a request returns a non-2xx status, THE Traffic_Generator SHALL print the status code and response body to stdout without raising an exception, so the simulation continues uninterrupted.
7. THE Traffic_Generator SHALL be compatible with Python 3.11 and SHALL only import from the standard library and `requests`, so it can run outside the Docker containers without additional dependencies.

---

### Requirement 9: Docker Compose Stack Correctness

**User Story:** As a DevOps engineer, I want `docker compose up -d` to start all services in the correct order with correct health probes, so that the full stack reaches a stable, healthy state within 3 minutes on a clean first boot.

#### Acceptance Criteria

1. THE `docker-compose.yml` SHALL define services `mlflow`, `api`, `airflow`, `prometheus`, and `grafana`, with `api` and `airflow` having `depends_on.mlflow.condition: service_healthy`.
2. THE `mlflow` service healthcheck SHALL use a Python socket connection to port 5000 with a 3-second timeout, `interval: 15s`, `retries: 3`, and `start_period: 20s`.
3. THE `api` service healthcheck SHALL make an HTTP GET to `/health` and accept HTTP 200 as healthy, with `interval: 15s`, `timeout: 5s`, `retries: 5`, and `start_period: 45s`.
4. THE `docker-compose.yml` SHALL define an `init` service under `profiles: [init]` that runs `scripts/init_mlflow.py`, depends on `mlflow` being healthy, and exits cleanly, so it can be used to seed the registry without leaving a running container.
5. THE `prometheus` service SHALL mount `./prometheus:/etc/prometheus` so that `prometheus/prometheus.yml` is used as the scrape configuration.
6. THE `grafana` service SHALL mount `./grafana/provisioning:/etc/grafana/provisioning` and `./grafana/dashboards:/var/lib/grafana/dashboards` so dashboards and datasources are provisioned automatically on first start.

---

### Requirement 10: Production Model Prediction Correctness

**User Story:** As a clinician-facing system, I want the `/predict` endpoint to always return a valid probability and risk level for any well-formed `PatientData` input, so that downstream systems never receive an unexpected error for a valid request.

#### Acceptance Criteria

1. WHEN a valid `PatientData` JSON body is POSTed to `/predict`, THE API SHALL return HTTP 200 with a JSON body containing `probabilite` (float in [0.0, 1.0]), `risque` (one of "ÉLEVÉ", "MODÉRÉ", "FAIBLE"), `seuil_utilise` (float), `version_modele` (string), and `ab_test_role` (string).
2. THE API SHALL clamp the raw model output to the range [0.0, 1.0] before classifying risk level.
3. WHEN the model output is ≥ `threshold`, THE API SHALL classify `risque` as `"ÉLEVÉ"`.
4. WHEN the model output is ≥ `threshold * 0.5` and < `threshold`, THE API SHALL classify `risque` as `"MODÉRÉ"`.
5. WHEN the model output is < `threshold * 0.5`, THE API SHALL classify `risque` as `"FAIBLE"`.
6. WHEN the `PatientData` contains feature columns not present in `feature_names.csv`, THE API SHALL ignore those extra columns by re-indexing to the known feature list and filling unknown columns with `0`.
7. IF the model raises an exception during inference, THEN THE API SHALL return HTTP 500 with a JSON error body and SHALL NOT propagate the exception to the ASGI server in a way that crashes the worker process.
