"""Small sklearn-compatible ensemble helpers."""

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Prefit soft-voting classifier.

    It stores already-trained estimators and averages their predict_proba outputs.
    Keeping this class in an importable module makes joblib models loadable from
    FastAPI and notebooks.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, Any]],
        weights: Optional[Sequence[float]] = None,
    ):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: Any, y: Optional[Iterable[int]] = None) -> "ProbabilityAveragingEnsemble":
        if y is None:
            self.classes_ = np.array([0, 1])
        else:
            self.classes_ = np.unique(np.asarray(list(y)))
        return self

    def _weights_array(self) -> Optional[np.ndarray]:
        if self.weights is None:
            return None

        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(self.estimators):
            raise ValueError("weights doit avoir la meme longueur que estimators.")
        if np.any(weights < 0):
            raise ValueError("weights ne doit pas contenir de valeurs negatives.")
        if weights.sum() == 0:
            raise ValueError("La somme des weights doit etre positive.")
        return weights

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Aucun estimateur dans l'ensemble.")

        probas = []
        for name, estimator in self.estimators:
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(f"L'estimateur {name} ne supporte pas predict_proba.")

            proba = np.asarray(estimator.predict_proba(X), dtype=float)
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise ValueError(f"predict_proba invalide pour {name}: shape={proba.shape}")
            probas.append(proba)

        averaged = np.average(np.stack(probas, axis=0), axis=0, weights=self._weights_array())
        row_sums = averaged.sum(axis=1, keepdims=True)
        return averaged / np.clip(row_sums, 1e-12, None)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
