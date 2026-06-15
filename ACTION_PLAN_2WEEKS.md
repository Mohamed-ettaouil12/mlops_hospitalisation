# 🚀 PLAN D'ACTION IMMÉDIAT: 2 Semaines pour Compléter le Projet

## 📊 État Actuel vs Complet

```
ACTUELLEMENT:      ███████░░░░░░░░░░░░░░░░  40% complété

À FAIRE:
  ⏳ Optuna Tuning
  ⏳ SHAP + Fairness
  ⏳ Monitoring
  ⏳ CI/CD
  ⏳ Streaming

OBJECTIF:          ██████████████████████░░  100% complet
```

---

## 📅 SEMAINE 1: Optimisation & Analyse

### 🔵 Lundi-Mardi: Optuna Tuning

**Objectif**: Augmenter la précision de +1-2% avec meilleurs hyperparamètres

**À faire**:
```bash
# 1. Créer le fichier d'optimisation
cat > src/optuna_tuning.py << 'EOF'
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score

class OptunaTuner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def objective_xgb(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_val)
        return f1_score(self.y_val, y_pred)
    
    def optimize_xgb(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_xgb, n_trials=n_trials)
        return study.best_params

EOF

# 2. Exécuter l'optimisation
python -c "
from src.optuna_tuning import OptunaTuner
import pandas as pd

# Charger données
X_train = pd.read_parquet('data/features/X_train.parquet')
y_train = pd.read_parquet('data/features/y_train.parquet').squeeze()
X_val = pd.read_parquet('data/features/X_val.parquet')
y_val = pd.read_parquet('data/features/y_val.parquet').squeeze()

# Optimiser
tuner = OptunaTuner(X_train, y_train, X_val, y_val)
best_params = tuner.optimize_xgb(n_trials=50)
print('Best params:', best_params)
"

# 3. Réentraîner avec les meilleurs paramètres
python src/train.py --use-optuna-params
```

**Résultat attendu**: 
- ✅ Best params pour XGBoost
- ✅ F1 score amélioré
- ✅ Params sauvegardés dans `models/best_params.json`

---

### 🔵 Mercredi: SHAP + Fairness

**Objectif**: Analyser les prédictions et détecter les biais

**À faire**:
```bash
# 1. Créer SHAP analyzer
cat > src/shap_analyzer.py << 'EOF'
import shap
import numpy as np

class SHAPAnalyzer:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.X_train = X_train
    
    def get_shap_values(self, X_test):
        return self.explainer.shap_values(X_test)
    
    def plot_summary(self, X_test, filename):
        shap_values = self.get_shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False, 
                         plot_type='bar')
        import matplotlib.pyplot as plt
        plt.savefig(filename)
        plt.close()
    
    def get_important_features(self, X_test, top_k=10):
        shap_values = self.get_shap_values(X_test)
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]
        return X_test.columns[top_indices].tolist()

EOF

# 2. Créer fairness analyzer
cat > src/fairness_analyzer.py << 'EOF'
from sklearn.metrics import confusion_matrix
import numpy as np

class FairnessAnalyzer:
    def __init__(self, y_true, y_pred, sensitive_attr):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_attr = sensitive_attr
    
    def demographic_parity(self):
        """Proportion de positifs par groupe"""
        groups = self.sensitive_attr.unique()
        parities = {}
        
        for group in groups:
            mask = self.sensitive_attr == group
            parity = self.y_pred[mask].mean()
            parities[f'group_{group}'] = parity
        
        return parities
    
    def equal_opportunity(self):
        """TPR par groupe"""
        groups = self.sensitive_attr.unique()
        tprs = {}
        
        for group in groups:
            mask = self.sensitive_attr == group
            tp = ((self.y_pred[mask] == 1) & (self.y_true[mask] == 1)).sum()
            fn = ((self.y_pred[mask] == 0) & (self.y_true[mask] == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tprs[f'group_{group}'] = tpr
        
        return tprs

EOF

# 3. Analyser
python -c "
from src.shap_analyzer import SHAPAnalyzer
from src.fairness_analyzer import FairnessAnalyzer
import pandas as pd
import joblib

# Charger
model = joblib.load('models/best_model.pkl')
X_test = pd.read_parquet('data/features/X_test.parquet')
y_test = pd.read_parquet('data/features/y_test.parquet').squeeze()

# SHAP
shap_analyzer = SHAPAnalyzer(model, X_test)
top_features = shap_analyzer.get_important_features(X_test)
print('Top 10 features:', top_features)

# Fairness
y_pred = model.predict(X_test)
fairness = FairnessAnalyzer(y_test, y_pred, X_test['SEXE_ENC'])
print('Demographic parity:', fairness.demographic_parity())
print('Equal opportunity:', fairness.equal_opportunity())
"
```

**Résultat attendu**:
- ✅ Top 10 features identifiées
- ✅ SHAP plots générés
- ✅ Fairness metrics calculées
- ✅ Biais détectés (si présents)

---

### 🔵 Jeudi-Vendredi: Monitoring Setup

**Objectif**: Configurer la détection de drift en production

**À faire**:
```bash
# 1. Installer Evidently
pip install evidently

# 2. Créer drift detector
cat > src/drift_detector.py << 'EOF'
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import pandas as pd

class DriftDetector:
    def __init__(self, reference_data, reference_target=None):
        self.reference_data = reference_data
        self.reference_target = reference_target
    
    def check_data_drift(self, current_data):
        """Détecte la dérive des données"""
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )
        
        return report
    
    def check_prediction_drift(self, current_data, current_predictions, current_target):
        """Détecte la dérive des prédictions"""
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        current_with_pred = current_data.copy()
        current_with_pred['prediction'] = current_predictions
        current_with_pred['target'] = current_target
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_with_pred,
        )
        
        return report

EOF

# 3. Setup monitoring
python -c "
from src.drift_detector import DriftDetector
import pandas as pd

# Charger
X_train = pd.read_parquet('data/features/X_train.parquet')
X_test = pd.read_parquet('data/features/X_test.parquet')

# Détecter drift
detector = DriftDetector(X_train)
report = detector.check_data_drift(X_test)

# Sauvegarder le rapport
report.save_html('outputs/reports/drift_report.html')
print('✅ Rapport sauvegardé: outputs/reports/drift_report.html')
"
```

**Résultat attendu**:
- ✅ Drift report généré
- ✅ Dashboard HTML créé
- ✅ Alertes configurées

---

## 📅 SEMAINE 2: Deployment & Streaming

### 🟢 Lundi-Mardi: CI/CD Setup

**Objectif**: Automatiser les tests et l'entraînement

**À faire**:
```bash
# 1. Créer tests
mkdir -p tests
cat > tests/test_models.py << 'EOF'
import pytest
import pandas as pd
import joblib

def test_model_loading():
    model = joblib.load('models/best_model.pkl')
    assert model is not None

def test_model_prediction():
    model = joblib.load('models/best_model.pkl')
    X_test = pd.read_parquet('data/features/X_test.parquet')
    
    predictions = model.predict(X_test.head())
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)

def test_model_performance():
    model = joblib.load('models/best_model.pkl')
    X_test = pd.read_parquet('data/features/X_test.parquet')
    y_test = pd.read_parquet('data/features/y_test.parquet').squeeze()
    
    from sklearn.metrics import f1_score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    assert f1 > 0.60, f"F1 score {f1} below threshold 0.60"

EOF

# 2. Créer GitHub Actions workflow
mkdir -p .github/workflows
cat > .github/workflows/test.yml << 'EOF'
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/

EOF

# 3. Push to GitHub
git add .
git commit -m "Add CI/CD pipeline"
git push
```

**Résultat attendu**:
- ✅ Tests créés et passants
- ✅ GitHub Actions workflow actif
- ✅ Tests exécutés automatiquement

---

### 🟢 Mercredi-Vendredi: Streaming Setup

**Objectif**: Setup Kafka pour les prédictions en temps réel

**À faire**:
```bash
# 1. Docker Compose pour Kafka
cat > docker-compose-kafka.yml << 'EOF'
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

EOF

# 2. Démarrer Kafka
docker-compose -f docker-compose-kafka.yml up -d

# 3. Créer streaming app
cat > streaming_app.py << 'EOF'
from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Charger le modèle
model = joblib.load('models/best_model.pkl')
feature_cols = pd.read_parquet('data/features/X_test.parquet').columns

# Consumer
consumer = KafkaConsumer(
    'predictions-input',
    bootstrap_servers=['kafka:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='prediction-service',
)

# Producer
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

log.info("🚀 Streaming app started, listening to Kafka...")

for message in consumer:
    try:
        # Récupérer les données
        data = message.value
        X = pd.DataFrame([data])[feature_cols]
        
        # Prédire
        prediction = model.predict_proba(X)[0, 1]
        
        # Envoyer le résultat
        result = {
            'input_id': data.get('id'),
            'prediction': float(prediction),
            'status': 'success'
        }
        producer.send('predictions-output', value=result)
        
        log.info(f"✅ Prediction: {prediction:.4f}")
        
    except Exception as e:
        log.error(f"❌ Error: {e}")
        producer.send('predictions-output', value={
            'status': 'error',
            'error': str(e)
        })

EOF

# 4. Démarrer l'app
python streaming_app.py
```

**Résultat attendu**:
- ✅ Kafka broker running
- ✅ Streaming app listening
- ✅ Real-time predictions working

---

## 🎯 COMMANDES RAPIDES

### État du Projet
```bash
python PROJECT_STATUS.py
```

### Voir la Roadmap Complète
```bash
python ROADMAP_FINAL_PHASES.py
```

### Exécuter Chaque Phase

**Phase 4: Optuna**
```bash
python src/optuna_tuning.py --n-trials 100
```

**Phase 5: SHAP + Fairness**
```bash
python src/shap_analyzer.py
python src/fairness_analyzer.py
```

**Phase 6: Monitoring**
```bash
python src/drift_detector.py
mlflow ui  # Voir les drifts
```

**Phase 7: CI/CD**
```bash
git push  # GitHub Actions déclenché automatiquement
pytest tests/  # Tests locaux
```

**Phase 8: Streaming**
```bash
docker-compose -f docker-compose-kafka.yml up
python streaming_app.py
```

---

## ✅ CHECKLIST 2 SEMAINES

### Semaine 1
- [ ] Lun-Mar: Optuna tuning complété
- [ ] Mer: SHAP analysis complété
- [ ] Mer: Fairness analysis complété
- [ ] Jeu-Ven: Monitoring setup complété

### Semaine 2
- [ ] Lun-Mar: CI/CD setup complété
- [ ] Mer-Ven: Kafka setup et streaming complété
- [ ] Ven: Tous les tests passants
- [ ] Ven: Projet 100% complet ✅

---

## 🚀 RÉSULTAT FINAL

Après 2 semaines:

```
✅ Baseline models: XGBoost, LightGBM, etc.
✅ Ensemble: Stacking OOF, Blending, Pseudo-labeling
✅ Optimization: Optuna tuning (+1-2%)
✅ Explainability: SHAP analysis + Fairness
✅ Monitoring: Drift detection + Alerts
✅ CI/CD: Automated testing + training
✅ Streaming: Real-time predictions via Kafka
✅ Full MLOps Pipeline: Production-ready! 🚀
```

**Précision finale attendue**: +3-5% vs baseline 🎯

---

## 📞 SUPPORT

Vous êtes bloqué? Consultez:
- `ROADMAP_FINAL_PHASES.py` pour les détails
- `src/` pour les implémentations
- MLflow UI pour les résultats

Bon courage! 💪
