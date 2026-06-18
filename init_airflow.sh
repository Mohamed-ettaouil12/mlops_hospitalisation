#!/usr/bin/env bash
set -euo pipefail

echo "=== Initialisation Airflow ==="

echo "→ db init"
docker compose run --rm airflow airflow db init

echo "→ création utilisateur admin / admin"
docker compose run --rm airflow airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

echo "✓ Airflow prêt — connecte-toi sur http://localhost:8080 (admin / admin)"
