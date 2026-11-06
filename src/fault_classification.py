"""
Multi-Class Electrical Fault Classification Module.
Benchmarks four supervised algorithms:
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (RBF Kernel)
- XGBoost Classifier
Selects the optimal model for production inference and provides actionable engineering actions.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


CORRECTIVE_ACTIONS = {
    "Normal": "Optimal steady operation. No intervention required. System operating nominally.",
    "Overload": "De-energize or shed load immediately. Verify mechanical bearings, thermal cut-off, and motor clearance.",
    "Overheating": "Power down immediately. Inspect thermal sensors, ventilation pathways, and heat sink dissipation.",
    "Overcurrent": "Critical: Immediate disconnect. Inspect winding insulation, check for locked rotor, and verify breaker ratings.",
    "Voltage Abnormality": "Check grid supply voltage, inspect stabilizer/regulator, and verify line neutral grounding.",
    "Abnormal Power Consumption": "Schedule preventive maintenance. Inspect duty cycle, thermostat calibration, and component aging.",
    "Potential Electrical Fault": "Perform insulation resistance Megger test. Check for phase imbalance and loose wiring connections."
}


class FaultClassifier:
    """Supervised fault classification benchmark and prediction engine."""

    def __init__(self, class_names: Optional[List[str]] = None, random_state: int = 42):
        self.class_names = class_names or [
            "Normal", "Overload", "Overheating", "Overcurrent",
            "Voltage Abnormality", "Abnormal Power Consumption", "Potential Electrical Fault"
        ]
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = "Random Forest"
        self.is_fitted = False

    def train_and_compare(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Trains and benchmarks Random Forest, Decision Tree, SVM, and XGBoost.
        Stores the top-performing model for deployment.
        """
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=self.random_state),
            "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=self.random_state),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state, eval_metric="mlogloss")
        }

        results = []
        best_f1 = -1.0

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "Algorithm": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4)
            })

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

        self.is_fitted = True
        comparison_df = pd.DataFrame(results)
        print(f"Optimal Model Selected: {self.best_model_name} (F1-Score: {best_f1:.4f})")
        return comparison_df

    def predict(self, X: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Predicts fault classes for a batch of feature vectors.
        Returns:
            predicted_labels: List of class strings
            confidences: Array of highest probability values
            probas: Full probability distribution array
        """
        if not self.is_fitted or self.best_model is None:
            raise RuntimeError("FaultClassifier has not been trained.")

        if hasattr(self.best_model, "predict_proba"):
            probas = self.best_model.predict_proba(X)
            class_indices = np.argmax(probas, axis=1)
            confidences = np.max(probas, axis=1)
        else:
            class_indices = self.best_model.predict(X)
            confidences = np.ones(len(class_indices))
            probas = np.zeros((len(class_indices), len(self.class_names)))

        predicted_labels = [self.class_names[idx] for idx in class_indices]
        return predicted_labels, confidences, probas

    def predict_single(self, X_single: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predicts (label, confidence, probas) for a single sample."""
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        labels, confs, probas = self.predict(X_single)
        return labels[0], float(confs[0]), probas[0]

    @staticmethod
    def get_corrective_action(fault_name: str) -> str:
        """Retrieves targeted engineering recommendation for a given fault."""
        return CORRECTIVE_ACTIONS.get(fault_name, "Inspect electrical parameters and check equipment maintenance manual.")

    def save(self, filepath: str):
        """Serializes trained classifier."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FaultClassifier":
        """Loads serialized classifier."""
        return joblib.load(filepath)
