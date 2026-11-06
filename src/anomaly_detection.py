"""
Telemetry Anomaly Detection Module.
Implements Isolation Forest and Local Outlier Factor scoring to isolate
multivariate electrical anomalies (current spikes, thermal runaway, voltage sags).
"""

import os
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation Forest anomaly detector with calibrated [0.0, 1.0] scoring."""

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.score_min = -0.3
        self.score_max = 0.3
        self.is_fitted = False

    def fit(self, X: np.ndarray, df_feat: Optional[pd.DataFrame] = None) -> "AnomalyDetector":
        """Fits the Isolation Forest on training feature matrix."""
        self.model.fit(X)
        raw_decisions = self.model.decision_function(X)
        self.score_min = float(np.percentile(raw_decisions, 1))
        self.score_max = float(np.percentile(raw_decisions, 99))
        self.is_fitted = True
        return self

    def _normalize_score(self, decision_vals: np.ndarray) -> np.ndarray:
        """
        Normalizes Isolation Forest decision values to [0.0, 1.0] anomaly score.
        0.0 = completely nominal, 1.0 = severe anomaly.
        """
        # Lower decision function means more anomalous
        span = max(0.001, self.score_max - self.score_min)
        norm = (self.score_max - decision_vals) / span
        return np.clip(norm, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly flags and normalized anomaly scores.
        Returns:
            is_anomaly: boolean numpy array (True if anomalous)
            anomaly_scores: float numpy array in [0.0, 1.0]
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector is not fitted.")

        decisions = self.model.decision_function(X)
        scores = self._normalize_score(decisions)
        is_anom = scores >= 0.55
        return is_anom, scores

    def predict_single(self, X_single: np.ndarray) -> Tuple[bool, float]:
        """Predicts anomaly status and score for single observation."""
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        is_anom, scores = self.predict(X_single)
        return bool(is_anom[0]), float(scores[0])

    def save(self, filepath: str):
        """Serializes anomaly detection model."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AnomalyDetector":
        """Loads serialized anomaly detection model."""
        return joblib.load(filepath)
