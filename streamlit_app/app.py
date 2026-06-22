import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
PAGE_TITLE = "MLOps Hospitalization Risk Prediction"
PAGE_ICON = "🏥"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ── API Helpers ────────────────────────────────────────────────────────────────


@st.cache_data(ttl=10)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def predict(data: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def predict_champion(data: dict):
    try:
        r = requests.post(f"{API_BASE}/predict/champion", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .risk-low { color: #28a745; font-weight: 700; font-size: 1.2rem; }
    .risk-moderate { color: #ffc107; font-weight: 700; font-size: 1.2rem; }
    .risk-high { color: #dc3545; font-weight: 700; font-size: 1.2rem; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1.2rem; text-align: center; border: 1px solid #e9ecef; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #6c757d; }
    .stButton>button { width: 100%; }
    div[data-testid="stSidebarNav"] { padding-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://img.icons8.com/fluency/96/hospital-3.png", width=80
)
st.sidebar.title("MLOps Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Prediction", "ℹ️ Model Info", "📊 Monitoring"],
    label_visibility="collapsed",
)

# Strip the icon prefix to get the page name
page_name = page.split(" ", 1)[1] if " " in page else page

st.sidebar.markdown("---")
st.sidebar.caption(f"API: {API_BASE}")
health = fetch_health()
if health:
    st.sidebar.success(f"🟢 API Online — v{health.get('api_version', '?')}")
else:
    st.sidebar.error("🔴 API Offline")

st.sidebar.caption(f"Built · Streamlit 1.40")
st.sidebar.caption(f"© {datetime.now().year} MLOps PFE")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def render_home():
    st.title("🏥 MLOps Hospitalization Risk Prediction")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
        ### System Overview
        This platform leverages a **production-grade MLOps pipeline** to predict
        patient hospitalization risk in real time. It follows industry best
        practices for model lifecycle management, continuous monitoring, and
        automated retraining.

        ### Key Capabilities
        - **Real-time inference** with A/B testing (90% champion / 10% challenger)
        - **Automated retraining** triggered by data drift
        - **Model registry** with champion/challenger promotion
        - **Continuous monitoring** of drift, accuracy, and latency
        """
        )

    with col2:
        st.markdown("### Architecture")
        st.markdown(
            """
        ```
        ┌──────────┐     ┌──────────┐
        │ Streamlit │────▶│  FastAPI │
        │   (UI)   │     │   (API)  │
        └──────────┘     └────┬─────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  MLflow  │   │ Airflow  │   │Prometheus│
        │ Registry │   │  (DAGs)  │   │ Metrics  │
        └──────────┘   └──────────┘   └────┬─────┘
                                           ▼
                                      ┌──────────┐
                                      │ Grafana  │
                                      │  (Dash)  │
                                      └──────────┘
        ```
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.markdown('<div class="metric-card"><div class="metric-value">FastAPI</div><div class="metric-label">Serving</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="metric-card"><div class="metric-value">MLflow</div><div class="metric-label">Registry</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="metric-card"><div class="metric-value">Airflow</div><div class="metric-label">Orchestration</div></div>', unsafe_allow_html=True)
    with col_d:
        st.markdown('<div class="metric-card"><div class="metric-value">Prometheus</div><div class="metric-label">Monitoring</div></div>', unsafe_allow_html=True)
    with col_e:
        st.markdown('<div class="metric-card"><div class="metric-value">Grafana</div><div class="metric-label">Visualization</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Links")
    qa, qb, qc, qd = st.columns(4)
    with qa:
        st.link_button("📡 API Health", f"{API_BASE}/health")
    with qb:
        st.link_button("📓 MLflow UI", "http://localhost:5000")
    with qc:
        st.link_button("📊 Grafana", "http://localhost:3000")
    with qd:
        st.link_button("⏰ Airflow", "http://localhost:8080")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

# 29 input fields definition — used for the form UI and payload building
INPUT_FIELDS = [
    # (label, key, type, min, max, default, help_text, section)
    ("Age", "AGE", 0.0, 120.0, 50.0, "Patient age in years", "Demographics"),
    ("Sex (0=F, 1=M)", "SEXE_ENC", 0, 1, 0, "Sex encoded", "Demographics"),
    ("Race (0-5)", "RACE_ENC", 0, 5, 0, "Race category encoded", "Demographics"),
    ("Age Group (0-3)", "GROUPE_AGE_ENC", 0, 3, 0, "Age group encoded", "Demographics"),
    ("ESRD Indicator", "BENE_ESRD_IND", 0, 1, 0, "End-stage renal disease", "Demographics"),
    ("State Code", "SP_STATE_CODE", 0, 100, 0, "State code", "Demographics"),
    ("Alzheimer's", "SP_ALZHDMTA", 0, 1, 0, "Alzheimer's disease", "Comorbidities"),
    ("CHF", "SP_CHF", 0, 1, 0, "Congestive heart failure", "Comorbidities"),
    ("Chronic Kidney Disease", "SP_CHRNKIDN", 0, 1, 0, "Chronic kidney disease", "Comorbidities"),
    ("Cancer", "SP_CNCR", 0, 1, 0, "Any cancer", "Comorbidities"),
    ("COPD", "SP_COPD", 0, 1, 0, "Chronic pulmonary disease", "Comorbidities"),
    ("Depression", "SP_DEPRESSN", 0, 1, 0, "Depression", "Comorbidities"),
    ("Diabetes", "SP_DIABETES", 0, 1, 0, "Diabetes mellitus", "Comorbidities"),
    ("Ischemic Heart Disease", "SP_ISCHMCHT", 0, 1, 0, "Ischemic heart disease", "Comorbidities"),
    ("Osteoporosis", "SP_OSTEOPRS", 0, 1, 0, "Osteoporosis", "Comorbidities"),
    ("Rheumatoid Arthritis / OA", "SP_RA_OA", 0, 1, 0, "Rheumatoid arthritis or osteoarthritis", "Comorbidities"),
    ("Stroke / TIA", "SP_STRKETIA", 0, 1, 0, "Stroke or transient ischemic attack", "Comorbidities"),
    ("Number of Comorbidities", "NB_COMORBIDITES", 0.0, 20.0, 0.0, "Total number of comorbid conditions", "Clinical Metrics"),
    ("Charlson Index", "CHARLSON_INDEX", 0.0, 40.0, 0.0, "Charlson comorbidity index score", "Clinical Metrics"),
    ("Total Cost", "COUT_TOTAL", 0.0, None, 0.0, "Total healthcare cost", "Utilization"),
    ("New Patient", "IS_NEW_PATIENT", 0, 1, 0, "Is this a new patient?", "Utilization"),
    ("Past Hospitalizations", "NB_HOSP_PASSEES", 0.0, None, 0.0, "Number of past hospitalizations", "Utilization"),
    ("Surgery in 3M", "NB_OP_3M", 0.0, None, 0.0, "Surgeries in last 3 months", "Surgery History"),
    ("Surgery in 6M", "NB_OP_6M", 0.0, None, 0.0, "Surgeries in last 6 months", "Surgery History"),
    ("Surgery in 12M", "NB_OP_12M", 0.0, None, 0.0, "Surgeries in last 12 months", "Surgery History"),
    ("Cardiac Events 6M", "NB_CAR_6M", 0.0, None, 0.0, "Cardiac events in last 6 months", "Clinical Metrics"),
    ("Prescriptions", "NB_PRESCRIPTIONS", 0.0, None, 0.0, "Number of prescriptions", "Medications"),
    ("Unique Molecules", "NB_MOLECULES_UNIQUES", 0.0, None, 0.0, "Number of unique drug molecules", "Medications"),
    ("Polypharmacy", "POLYPHARMACIE", 0, 1, 0, "Polypharmacy indicator (>5 meds)", "Medications"),
]


def build_input_form():
    payload = {}
    last_section = None
    first_row = True

    for label, key, min_val, max_val, default, _help, section in INPUT_FIELDS:
        if section != last_section:
            if not first_row:
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"**{section}**")
            st.markdown('<div style="border-left: 3px solid #0d6efd; padding-left: 1rem; margin-bottom: 1rem;">', unsafe_allow_html=True)
            last_section = section
            first_row = True

        is_int = isinstance(min_val, int) and isinstance(default, int)
        step = 1.0 if is_int else 0.1
        fmt_min = min_val if min_val is not None else 0
        fmt_max = max_val if max_val is not None else 999999

        val = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=step,
            format="%d" if is_int else "%f",
            key=f"input_{key}",
            help=_help,
        )
        payload[key] = val
        first_row = False

    st.markdown("</div>", unsafe_allow_html=True)
    return payload


def render_prediction():
    st.title("🔮 Patient Risk Prediction")
    st.markdown("Enter patient data below to predict hospitalization risk.")

    with st.form("prediction_form"):
        st.subheader("Patient Data")
        payload = build_input_form()

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("🚀 Predict Risk", type="primary", use_container_width=True)
        with col2:
            use_champion = st.checkbox("Force champion model (no A/B)", value=False)

    if submitted:
        with st.spinner("Calling prediction API..."):
            if use_champion:
                result = predict_champion(payload)
            else:
                result = predict(payload)

        if "error" in result:
            st.error(f"❌ Prediction failed: {result['error']}")
            return

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        prob = result.get("probabilite", 0.0)
        risque = result.get("risque", "N/A")
        seuil = result.get("seuil_utilise", 0.25)
        version = result.get("version_modele", "?")

        risk_emoji = {"FAIBLE": "🟢", "MODÉRÉ": "🟡", "ÉLEVÉ": "🔴"}
        risk_icon = risk_emoji.get(risque, "⚪")

        res_cols = st.columns([1, 1, 1, 1])
        with res_cols[0]:
            prediction = 1 if prob >= seuil else 0
            st.metric("Prediction (0/1)", prediction)

        with res_cols[1]:
            st.metric("Probability", f"{prob:.4f}")

        with res_cols[2]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size: 2rem;">{risk_icon}</div>'
                f'<div class="{"risk-high" if risque == "ÉLEVÉ" else "risk-moderate" if risque == "MODÉRÉ" else "risk-low"}">{risque}</div>'
                f'<div class="metric-label">Risk Level</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with res_cols[3]:
            st.metric("Threshold", f"{seuil:.2f}")

        with st.expander("🔍 Details"):
            st.json(
                {
                    "model_version": version,
                    "threshold_used": seuil,
                    "risk_level": risque,
                    "probability": prob,
                    "prediction": int(prob >= seuil),
                    "A/B_testing": not use_champion,
                }
            )

            st.caption("Input data sent to API:")
            st.json(payload)

        latency = time.time() - time.time()  # placeholder
        st.caption(f"Model version: {version} · Threshold: {seuil}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════

def render_model_info():
    st.title("ℹ️ Model Information")
    st.markdown("---")

    with st.spinner("Fetching API health..."):
        h = fetch_health()

    if h is None:
        st.error("❌ Unable to reach the API. Make sure the FastAPI server is running.")
        st.info(f"Expected endpoint: {API_BASE}/health")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📡 API Status")
        status_color = "🟢" if h.get("status") == "healthy" else "🟡"
        st.markdown(f"**Status:** {status_color} {h.get('status', 'unknown').upper()}")
        st.markdown(f"**API Version:** `v{h.get('api_version', '?')}`")
        st.markdown(f"**Model Loaded:** {'✅ Yes' if h.get('model_loaded') else '❌ No'}")
        st.markdown(f"**MLflow Available:** {'✅ Yes' if h.get('mlflow_available') else '❌ No'}")
        st.markdown(f"**A/B Testing:** {'✅ Enabled' if h.get('ab_test_enabled') else '❌ Disabled'}")
        st.markdown(f"**Features:** {h.get('n_features', '?')}")
        st.markdown(f"**Threshold:** {h.get('threshold', '?')}")

    with col2:
        st.subheader("🏆 Champion Model")
        champ = h.get("champion", {})
        st.markdown(f"**Version:** `{champ.get('version', 'N/A')}`")
        st.markdown(f"**Stage:** `{champ.get('stage', 'N/A')}`")

        st.subheader("🥊 Challenger Model")
        chal = h.get("challenger", {})
        st.markdown(f"**Version:** `{chal.get('version', 'N/A') if chal else 'None'}`")
        st.markdown(f"**Stage:** `{chal.get('stage', 'N/A') if chal else '—'}`")

    st.markdown("---")
    st.subheader("⚙️ Raw Health Response")
    with st.expander("View raw JSON"):
        st.json(h)

    st.markdown("---")
    st.subheader("🔗 Quick Links")
    qa, qb = st.columns(2)
    with qa:
        st.link_button("📓 Open MLflow", "http://localhost:5000", use_container_width=True)
    with qb:
        st.link_button("📡 API Health Endpoint", f"{API_BASE}/health", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def simulate_request():
    """Measure API response time as a simple health check."""
    start = time.time()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        latency = time.time() - start
        return latency, r.status_code, r.ok
    except requests.exceptions.RequestException:
        return time.time() - start, 0, False


def render_monitoring():
    st.title("📊 Monitoring Dashboard")
    st.markdown("---")

    st.subheader("🏓 API Latency & Health")
    st.markdown("Live check against the FastAPI endpoint:")

    if st.button("🔄 Run Health Check", type="primary"):
        with st.spinner("Pinging API..."):
            latency, status_code, ok = simulate_request()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Response Time", f"{latency*1000:.1f} ms" if ok else "N/A")
        with col2:
            st.metric("HTTP Status", status_code)
        with col3:
            st.metric("Status", "✅ Online" if ok else "❌ Offline")

        if ok:
            st.success(f"API responded in {latency*1000:.1f} ms (HTTP {status_code})")
        else:
            st.error(f"API unreachable (HTTP {status_code})")
    else:
        st.info("Click **Run Health Check** to test API connectivity.")

    st.markdown("---")
    st.subheader("📈 Performance Metrics (Prometheus)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">📥</div><div class="metric-label">Requests</div><div style="font-size:0.8rem;color:#6c757d;">api_requests_total</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">⏱️</div><div class="metric-label">Latency</div><div style="font-size:0.8rem;color:#6c757d;">api_request_latency_seconds</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">❌</div><div class="metric-label">Errors</div><div style="font-size:0.8rem;color:#6c757d;">api_errors_total</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">📊</div><div class="metric-label">A/B Assignments</div><div style="font-size:0.8rem;color:#6c757d;">api_ab_test_assignments</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔗 External Dashboards")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("#### 📊 Grafana Dashboard")
        st.markdown(
            """
        - **URL:** [http://localhost:3000](http://localhost:3000)
        - **Login:** `admin` / `admin`
        - Pre-built dashboards for latency, errors, and drift metrics
        """
        )
        st.link_button("🔗 Open Grafana", "http://localhost:3000", use_container_width=True)

    with ecol2:
        st.markdown("#### 📓 MLflow Tracking")
        st.markdown(
            """
        - **URL:** [http://localhost:5000](http://localhost:5000)
        - View model runs, parameters, and metrics
        - Compare champion vs challenger performance
        """
        )
        st.link_button("🔗 Open MLflow", "http://localhost:5000", use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Drift & Accuracy Metrics")
    drift_cols = st.columns(2)
    with drift_cols[0]:
        st.markdown("**Drift Detection**")
        st.markdown(
            """
        - Evidently AI reports (DataDrift, TargetDrift, DataQuality)
        - PSI-based feature drift per column
        - Severity: NONE / MODERATE / HIGH / CRITICAL
        """
        )
    with drift_cols[1]:
        st.markdown("**Retraining Pipeline**")
        st.markdown(
            """
        - Champion/Challenger comparison
        - Automatic promotion if AUC improves
        - Rollback safety mechanism
        - Airflow DAG runs weekly (Sun 2am)
        """
        )

    st.markdown("---")
    st.caption("Prometheus metrics are scraped automatically. Grafana dashboards are updated in real time.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page_name == "Home":
    render_home()
elif page_name == "Prediction":
    render_prediction()
elif page_name == "Model Info":
    render_model_info()
elif page_name == "Monitoring":
    render_monitoring()
