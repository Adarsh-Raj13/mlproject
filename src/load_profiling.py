"""
Appliance Load Profiling Module.
Implements unsupervised K-Means clustering (k=4) and classification
to partition electrical telemetry into 4 behavioral energy regimes:
1. Eco / Light Duty (Low demand, standby)
2. Standard Normal (Nominal steady operation)
3. Heavy Continuous (High sustained power draw)
4. Peak Surge (Inrush transients, high mechanical stress)
"""

import os
from typing import List, Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


REGIME_CONFIG = {
    0: ("Eco / Light Duty", "Low"),
    1: ("Standard Normal", "Medium"),
    2: ("Heavy Continuous", "High"),
    3: ("Peak Surge", "Critical Surge")
}


class LoadProfiler:
    """Unsupervised K-Means clustering and supervised regime classifier."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.classifier = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=random_state)
        self.cluster_to_regime: Dict[int, Tuple[str, str]] = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray, df_feat: pd.DataFrame) -> "LoadProfiler":
        """Fits KMeans on feature space and orders clusters by mean active power."""
        raw_clusters = self.kmeans.fit_predict(X)

        # Compute mean active power per cluster to sort regimes systematically
        power_col = df_feat["Active_Power"].values if "Active_Power" in df_feat.columns else X[:, 2]
        mean_powers = []
        for c in range(self.n_clusters):
            mask = (raw_clusters == c)
            avg_p = float(np.mean(power_col[mask])) if np.any(mask) else 0.0
            mean_powers.append((c, avg_p))

        # Sort clusters by average power ascending
        sorted_clusters = [c for c, _ in sorted(mean_powers, key=lambda x: x[1])]

        # Map sorted indices to regime labels
        self.cluster_to_regime = {
            cluster_id: REGIME_CONFIG[rank]
            for rank, cluster_id in enumerate(sorted_clusters)
        }

        # Train fast classifier on mapped regime labels for ultra-fast prediction
        regime_labels = [self.cluster_to_regime[c][0] for c in raw_clusters]
        self.classifier.fit(X, regime_labels)

        self.is_fitted = True
        return self

    def predict_profile(self, X: np.ndarray) -> List[str]:
        """Predicts load profile regime labels for a batch of samples."""
        if not self.is_fitted:
            raise RuntimeError("LoadProfiler is not fitted.")
        return list(self.classifier.predict(X))

    def predict_single(self, X_single: np.ndarray) -> Tuple[str, str]:
        """Predicts (regime_label, consumption_category) for a single observation."""
        if not self.is_fitted:
            raise RuntimeError("LoadProfiler is not fitted.")
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        raw_cluster = self.kmeans.predict(X_single)[0]
        return self.cluster_to_regime.get(raw_cluster, ("Standard Normal", "Medium"))

    def save(self, filepath: str):
        """Saves model pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LoadProfiler":
        """Loads model pipeline from disk."""
        return joblib.load(filepath)

