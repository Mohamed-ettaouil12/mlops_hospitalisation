import requests
import random
import time

BASE_URL = "http://localhost:8000"

def generate_patient():
    return {
        "age": random.randint(20, 90),
        "bmi": round(random.uniform(18, 40), 1),
        "blood_pressure": random.randint(90, 160),
        "glucose": random.randint(70, 200),
        "smoker": random.randint(0, 1),
        "diabetes": random.randint(0, 1)
    }

def call_predict():
    try:
        data = generate_patient()
        r = requests.post(f"{BASE_URL}/predict", json=data, timeout=3)
        print("predict:", r.status_code, r.json())
    except Exception as e:
        print("error predict:", e)

def call_health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        print("health:", r.status_code)
    except Exception as e:
        print("error health:", e)

def call_metrics():
    try:
        r = requests.get(f"{BASE_URL}/metrics")
        print("metrics:", r.status_code)
    except Exception as e:
        print("error metrics:", e)

while True:
    choice = random.random()

    if choice < 0.6:
        call_predict()
    elif choice < 0.8:
        call_health()
    else:
        call_metrics()

    time.sleep(random.uniform(0.2, 1.5))

