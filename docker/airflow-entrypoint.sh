#!/usr/bin/env bash
set -e

airflow db init

exec airflow standalone
