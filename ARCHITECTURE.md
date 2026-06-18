```mermaid
graph TB
    subgraph INPUT["📥 Données"]
        DS[(Dataset<br/>Hospitalisation<br/>2008-2010)]
        DP[(Data/Features<br/>Parquet)]
    end

    subgraph ORCH["⏰ Orchestration - Airflow"]
        DDD[dift_detection_dag<br/>@daily]
        RMD[retrain_model_dag<br/>@weekly + on_trigger]
    end

    subgraph MON["📊 Monitoring & Drift"]
        ED[Evidently<br/>Data Drift<br/>Target Drift<br/>Data Quality]
        DR[Reports HTML<br/>+ JSON Résumé]
    end

    subgraph ML["🧠 ML Pipeline"]
        PRE[Preprocessing]
        TR[Training<br/>LightGBM]
        EV[Evaluate<br/>AUC / Recall / F1]
        CM[Compare Models]
    end

    subgraph REG["📦 MLflow Registry"]
        EXP[Experiment<br/>hospitalization_risk]
        REG_MOD[(Model Registry)]
        STG[Staging]
        PROD[Production]
        ARCH[Archived]
    end

    subgraph API["🌐 Serving - FastAPI"]
        API_MAIN[api/main.py]
        PRED[/predict]
        HLTH[/health]
        METR[/metrics]
        MINFO[/model/info]
    end

    subgraph MONIT["📈 System Monitoring"]
        PROM[Prometheus<br/>:9090]
        GRAF[Grafana<br/>:3000]
    end

    subgraph DOCKER["🐳 Docker Compose"]
        DC[docker-compose.yml<br/>5 services]
        NET[app-network]
    end

    %% Données → Pipeline ML
    DP --> PRE
    PRE --> TR

    %% Airflow orchestration
    DDD -->|run_drift_check| ED
    ED --> DR
    DR -->|trigger_retrain| RMD
    RMD -->|extract| PRE
    RMD -->|train_model| TR
    RMD -->|evaluate| EV
    RMD -->|compare_and_register| CM

    %% MLflow
    TR -->|log_model| EXP
    EXP --> REG_MOD
    CM -->|promote| STG
    STG -->|AUC ≥ 0.85| PROD
    PROD -->|meilleur| ARCH

    %% API
    PROD -->|load_model| API_MAIN
    API_MAIN --> PRED
    API_MAIN --> HLTH
    API_MAIN --> METR
    API_MAIN --> MINFO
    PRED -->|predictions| PROM

    %% Prometheus → Grafana
    PROM -->|scrape| GRAF

    %% Réseau
    DDD -.-> DC
    RMD -.-> DC
    API_MAIN -.-> NET
    ED -.-> DR

    %% Styles
    classDef airflow fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef mlflow fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef api fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef monitoring fill:#fff3e0,stroke:#e65100,color:#000
    classDef infra fill:#f5f5f5,stroke:#616161,color:#000

    class DDD,RMD airflow
    class EXP,REG_MOD,STG,PROD,ARCH mlflow
    class API_MAIN,PRED,HLTH,METR,MINFO api
    class ED,DR,PROM,GRAF monitoring
    class DC,NET infra
```
