"""
╔════════════════════════════════════════════════════════════════════════╗
║  TECHNIQUES AVANCÉES D'ENSEMBLE POUR AMÉLIORATION DE PRÉCISION        ║
║  - Stacking avec OOF predictions (évite data leakage)                 ║
║  - Weighted blending optimisé                                         ║
║  - Feature engineering croisé (Feature Importances)                   ║
║  - Pseudo-labeling pour données non labellisées                       ║
║  - SHAP-based feature selection                                       ║
║  - MLflow logging pour reproductibilité                               ║
╚════════════════════════════════════════════════════════════════════════╝
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in [str(p) for p in __import__("sys").path]:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. STACKING AVEC OOF PREDICTIONS (Out-of-Fold)
# ═══════════════════════════════════════════════════════════════════════════

class StackingWithOOF:
    """
    Stacking avec prédictions Out-of-Fold pour éviter le data leakage.
    
    Processus:
    1. Diviser train en K folds
    2. Pour chaque fold, entraîner les modèles de base sur K-1 folds
    3. Générer les prédictions OOF sur le fold restant
    4. Utiliser les prédictions OOF comme features pour le méta-modèle
    5. Réentraîner les modèles de base sur tout le train pour l'inférence
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model_type: str = "logistic",
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            base_models: Dict des modèles de base {nom: estimateur}
            meta_model_type: 'logistic' ou 'ridge'
            n_splits: Nombre de folds
            random_state: Seed pour reproducibilité
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.base_models_trained_ = {}
        self.meta_model_ = None
        self.scaler_ = None
        self.oof_predictions_ = None
        self.feature_names_ = None
    
    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Génère les prédictions Out-of-Fold.
        
        Returns:
            Array de shape (n_samples, n_base_models)
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        oof_preds = np.zeros((n_samples, n_models))
        
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            if verbose:
                log.info(f"\n  Génération OOF pour: {model_name}")
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                # Clone et entraîner le modèle
                model_fold = joblib.load(joblib.dumps(model))  # Deep copy
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Prédictions OOF
                oof_preds[val_idx, model_idx] = model_fold.predict_proba(X_val_fold)[:, 1]
                
                if verbose and (fold_idx + 1) % max(1, self.n_splits // 2) == 0:
                    log.info(f"    Fold {fold_idx + 1}/{self.n_splits} complété")
        
        self.oof_predictions_ = oof_preds
        if verbose:
            log.info(f"\n  ✓ OOF predictions shape: {oof_preds.shape}")
        
        return oof_preds
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> "StackingWithOOF":
        """
        Entraîne le stacking:
        1. Génère les OOF predictions sur train
        2. Entraîne le méta-modèle sur les OOF predictions
        3. Réentraîne les modèles de base sur tout le train
        """
        if verbose:
            log.info("\n" + "="*70)
            log.info("🔧 STACKING AVEC OOF PREDICTIONS")
            log.info("="*70)
        
        # Étape 1: Générer OOF predictions
        if verbose:
            log.info("\n1️⃣  Génération des prédictions Out-of-Fold...")
        
        oof_train = self.generate_oof_predictions(X_train, y_train, verbose=verbose)
        
        # Étape 2: Entraîner le méta-modèle
        if verbose:
            log.info("\n2️⃣  Entraînement du méta-modèle...")
        
        self.scaler_ = StandardScaler()
        oof_train_scaled = self.scaler_.fit_transform(oof_train)
        
        if self.meta_model_type == "logistic":
            self.meta_model_ = LogisticRegression(
                class_weight="balanced",
                max_iter=10000,
                solver="lbfgs",
                random_state=self.random_state,
            )
        elif self.meta_model_type == "ridge":
            self.meta_model_ = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            raise ValueError(f"meta_model_type {self.meta_model_type} non supporté")
        
        self.meta_model_.fit(oof_train_scaled, y_train)
        
        if verbose:
            log.info(f"  ✓ Méta-modèle ({self.meta_model_type}) entraîné")
        
        # Étape 3: Réentraîner les modèles de base sur tout le train
        if verbose:
            log.info("\n3️⃣  Réentraînement des modèles de base sur l'ensemble train...")
        
        for model_name, model in self.base_models.items():
            try:
                model.fit(X_train, y_train)
                self.base_models_trained_[model_name] = model
                if verbose:
                    log.info(f"  ✓ {model_name} réentraîné")
            except Exception as e:
                log.warning(f"  ⚠️ Erreur réentraînement {model_name}: {e}")
        
        # Sauvegarder les noms des features
        self.feature_names_ = [f"model_{i}" for i in range(len(self.base_models))]
        
        if verbose:
            log.info("\n✅ Stacking avec OOF complété!")
        
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        use_base_models: bool = True,
    ) -> np.ndarray:
        """
        Prédit les probabilités:
        1. Génère les prédictions des modèles de base
        2. Les combine via le méta-modèle
        """
        n_models = len(self.base_models_trained_)
        n_samples = X.shape[0]
        
        level1_preds = np.zeros((n_samples, n_models))
        
        for model_idx, (model_name, model) in enumerate(self.base_models_trained_.items()):
            try:
                level1_preds[:, model_idx] = model.predict_proba(X)[:, 1]
            except Exception as e:
                log.warning(f"  ⚠️ Erreur prédiction {model_name}: {e}")
                level1_preds[:, model_idx] = 0.5
        
        # Appliquer le scaler et le méta-modèle
        level1_scaled = self.scaler_.transform(level1_preds)
        
        if self.meta_model_type == "logistic":
            return self.meta_model_.predict_proba(level1_scaled)[:, 1]
        else:  # Ridge
            return np.clip(self.meta_model_.predict(level1_scaled), 0, 1)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Prédit les labels binaires"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# 2. WEIGHTED BLENDING OPTIMISÉ
# ═══════════════════════════════════════════════════════════════════════════

class WeightedBlending:
    """
    Blending pondéré avec optimisation des poids via validation.
    Cherche les poids optimaux qui maximisent une métrique cible sur validation.
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        objective_metric: str = "f1",
        beta: float = 1.0,
    ):
        """
        Args:
            base_models: Dict des modèles de base
            objective_metric: 'f1', 'precision', 'recall', 'auc'
            beta: Pour F-beta score (< 1 favorise precision, > 1 favorise recall)
        """
        self.base_models = base_models
        self.objective_metric = objective_metric
        self.beta = beta
        self.optimal_weights_ = None
    
    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Optimise les poids des modèles sur le validation set.
        """
        if verbose:
            log.info("\n" + "="*70)
            log.info("⚖️  OPTIMISATION DES POIDS (Weighted Blending)")
            log.info("="*70)
        
        n_models = len(self.base_models)
        model_names = list(self.base_models.keys())
        
        # Générer les prédictions de tous les modèles
        predictions = []
        for model_name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)[:, 1]
                predictions.append(pred)
                if verbose:
                    log.info(f"  ✓ Prédictions collectées pour: {model_name}")
            except Exception as e:
                log.warning(f"  ⚠️ Erreur {model_name}: {e}")
                predictions.append(np.full(X_val.shape[0], 0.5))
        
        predictions = np.array(predictions)
        
        # Fonction objective: négatif car on minimise
        def objective(weights):
            # Normaliser les poids
            weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
            
            # Blending pondéré
            blended = np.average(predictions, axis=0, weights=weights)
            
            # Seuil optimal
            threshold = 0.5
            y_pred = (blended >= threshold).astype(int)
            
            # Métrique cible
            if self.objective_metric == "f1":
                metric = f1_score(y_val, y_pred, zero_division=0)
            elif self.objective_metric == "precision":
                metric = precision_score(y_val, y_pred, zero_division=0)
            elif self.objective_metric == "recall":
                metric = recall_score(y_val, y_pred, zero_division=0)
            elif self.objective_metric == "auc":
                metric = roc_auc_score(y_val, blended)
            else:
                metric = f1_score(y_val, y_pred, zero_division=0)
            
            return -metric  # On minimise le négatif
        
        # Initialisation: poids égaux
        x0 = np.ones(n_models) / n_models
        
        # Contraintes: poids positifs et somme à 1
        constraints = {
            "type": "eq",
            "fun": lambda w: np.sum(np.abs(w)) - 1
        }
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimisation
        if verbose:
            log.info("\n  Optimisation en cours...")
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        
        # Normaliser les poids optimaux
        optimal_weights = np.abs(result.x) / (np.sum(np.abs(result.x)) + 1e-10)
        
        self.optimal_weights_ = dict(zip(model_names, optimal_weights))
        
        if verbose:
            log.info("\n  Poids optimaux:")
            for model_name, weight in self.optimal_weights_.items():
                log.info(f"    {model_name}: {weight:.4f}")
        
        return self.optimal_weights_
    
    def predict_proba(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Prédit avec les poids optimisés"""
        if self.optimal_weights_ is None:
            raise ValueError("Appeler optimize_weights() d'abord")
        
        predictions = []
        weights = []
        
        for model_name, model in self.base_models.items():
            if model_name in self.optimal_weights_:
                try:
                    pred = model.predict_proba(X)[:, 1]
                    predictions.append(pred)
                    weights.append(self.optimal_weights_[model_name])
                except Exception as e:
                    log.warning(f"  ⚠️ Erreur prédiction {model_name}: {e}")
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        return np.average(predictions, axis=0, weights=weights / weights.sum())


# ═══════════════════════════════════════════════════════════════════════════
# 3. PSEUDO-LABELING POUR DONNÉES NON LABELLISÉES
# ═══════════════════════════════════════════════════════════════════════════

class PseudoLabelingStrategy:
    """
    Pseudo-labeling: Utilise les modèles entraînés pour générer des labels
    sur les données non labellisées, puis les utilise pour l'entraînement.
    """
    
    def __init__(
        self,
        model,
        confidence_threshold: float = 0.95,
    ):
        """
        Args:
            model: Modèle entraîné
            confidence_threshold: Confiance minimale pour pseudo-label (0.95 = 95%)
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def generate_pseudo_labels(
        self,
        X_unlabeled: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Génère les pseudo-labels sur les données non labellisées.
        
        Returns:
            (X_pseudo, y_pseudo, confidence_scores)
        """
        if verbose:
            log.info("\n" + "="*70)
            log.info("🏷️  PSEUDO-LABELING")
            log.info("="*70)
        
        # Prédictions
        proba = self.model.predict_proba(X_unlabeled)
        confidence = np.max(proba, axis=1)
        predictions = np.argmax(proba, axis=1)
        
        # Filtrer par confiance
        high_conf_mask = confidence >= self.confidence_threshold
        
        X_pseudo = X_unlabeled[high_conf_mask].copy()
        y_pseudo = pd.Series(
            predictions[high_conf_mask],
            index=X_unlabeled[high_conf_mask].index,
        )
        confidence_pseudo = confidence[high_conf_mask]
        
        if verbose:
            log.info(f"\n  Total samples non labellisés: {len(X_unlabeled)}")
            log.info(f"  Samples avec haute confiance (>{self.confidence_threshold:.1%}): {len(X_pseudo)}")
            log.info(f"  Coverage: {len(X_pseudo) / len(X_unlabeled) * 100:.2f}%")
            log.info(f"\n  Distribution des pseudo-labels:")
            log.info(f"    Classe 0: {(y_pseudo == 0).sum()} ({(y_pseudo == 0).sum() / len(y_pseudo) * 100:.2f}%)")
            log.info(f"    Classe 1: {(y_pseudo == 1).sum()} ({(y_pseudo == 1).sum() / len(y_pseudo) * 100:.2f}%)")
        
        return X_pseudo, y_pseudo, confidence_pseudo


# ═══════════════════════════════════════════════════════════════════════════
# 4. CROSS-FEATURES ENGINEERING (Interactions entre modèles)
# ═══════════════════════════════════════════════════════════════════════════

def create_cross_features(
    predictions_dict: Dict[str, np.ndarray],
    feature_importance_dict: Dict[str, np.ndarray],
    X: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Crée des features d'interaction entre les prédictions des modèles.
    
    Args:
        predictions_dict: {model_name: predictions_array}
        feature_importance_dict: {model_name: importance_array}
        X: Features originales
        
    Returns:
        DataFrame avec les features d'interaction
    """
    if verbose:
        log.info("\n" + "="*70)
        log.info("🔗 CROSS-FEATURES ENGINEERING")
        log.info("="*70)
    
    cross_features = pd.DataFrame(index=X.index)
    
    model_names = list(predictions_dict.keys())
    preds = list(predictions_dict.values())
    
    # 1. Features de base: les prédictions
    for model_name, pred in zip(model_names, preds):
        cross_features[f"{model_name}_pred"] = pred
    
    # 2. Interactions deux à deux
    if verbose:
        log.info("\n  Création des features d'interaction:")
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_i, name_j = model_names[i], model_names[j]
            
            # Produit des prédictions
            cross_features[f"{name_i}_x_{name_j}_prod"] = preds[i] * preds[j]
            
            # Moyenne
            cross_features[f"{name_i}_x_{name_j}_mean"] = (preds[i] + preds[j]) / 2
            
            # Max
            cross_features[f"{name_i}_x_{name_j}_max"] = np.maximum(preds[i], preds[j])
            
            # Différence
            cross_features[f"{name_i}_x_{name_j}_diff"] = np.abs(preds[i] - preds[j])
            
            if verbose and i + j < len(model_names):
                log.info(f"    ✓ {name_i} ✕ {name_j}")
    
    # 3. Statistics globales
    cross_features["pred_mean"] = np.array(preds).mean(axis=0)
    cross_features["pred_std"] = np.array(preds).std(axis=0)
    cross_features["pred_max"] = np.array(preds).max(axis=0)
    cross_features["pred_min"] = np.array(preds).min(axis=0)
    
    if verbose:
        log.info(f"\n  Total features créées: {cross_features.shape[1]}")
    
    return cross_features


# ═══════════════════════════════════════════════════════════════════════════
# 5. MLFLOW LOGGING POUR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def log_ensemble_to_mlflow(
    experiment_name: str,
    run_name: str,
    ensemble_results: Dict[str, Any],
    model_params: Dict[str, Any],
    artifact_dir: Optional[Path] = None,
):
    """
    Log les résultats du stacking/blending dans MLflow.
    """
    try:
        import mlflow
        
        MLFLOW_URI = str(PROJECT_ROOT / "mlruns")
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log des paramètres
            mlflow.log_params(model_params)
            
            # Log des métriques
            for metric_name, metric_value in ensemble_results.get("metrics", {}).items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log des poids (si disponibles)
            if "weights" in ensemble_results:
                weights_dict = {
                    f"weight_{k}": v for k, v in ensemble_results["weights"].items()
                }
                mlflow.log_params(weights_dict)
            
            # Log des artefacts
            if artifact_dir and artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
            
            log.info("\n✅ Résultats loggés dans MLflow")
            
    except ImportError:
        log.warning("MLflow non disponible, résultats non loggés")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ensemble_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcule les métriques d'ensemble"""
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def save_ensemble_model(
    ensemble_obj: Any,
    output_path: Path,
    model_name: str = "ensemble",
):
    """Sauvegarde le modèle d'ensemble"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f"{model_name}.pkl"
    joblib.dump(ensemble_obj, model_path)
    
    log.info(f"✅ Modèle d'ensemble sauvegardé: {model_path}")
    
    return model_path


def load_ensemble_model(
    model_path: Path,
) -> Any:
    """Charge le modèle d'ensemble"""
    return joblib.load(model_path)
