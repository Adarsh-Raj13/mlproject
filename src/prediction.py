"""
Unified Machine Learning Inference Engine.
Coordinates Preprocessing, Load Profiling, Anomaly Detection, and Fault Classification.
Computes real-time parameter attribution, safety severity triage, and engineering recommendations.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.feature_engineering import APPLIANCE_SPECS, compute_electrical_features
from src.data_preprocessing import DataPreprocessor
from src.load_profiling import LoadProfiler
from src.anomaly_detection import AnomalyDetector
from src.fault_classification import FaultClassifier


class InferenceEngine:
    """Production inference engine for live and batch electrical diagnostics."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_dir = os.path.join(base_dir, "models")
        else:
            self.models_dir = models_dir

        self.preprocessor: Optional[DataPreprocessor] = None
        self.load_profiler: Optional[LoadProfiler] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.fault_classifier: Optional[FaultClassifier] = None

        self.load_models()

    @property
    def is_loaded(self) -> bool:
        """Checks whether all four core ML models and preprocessors are loaded."""
        return (
            self.preprocessor is not None
            and self.load_profiler is not None
            and self.anomaly_detector is not None
            and self.fault_classifier is not None
        )

    def load_models(self):
        """Loads serialized model artifacts from disk if available."""
        prep_path = os.path.join(self.models_dir, "preprocessing_pipeline.pkl")
        prof_path = os.path.join(self.models_dir, "load_profiling_model.pkl")
        anom_path = os.path.join(self.models_dir, "anomaly_detection_model.pkl")
        fault_path = os.path.join(self.models_dir, "fault_classification_model.pkl")

        try:
            if os.path.exists(prep_path):
                self.preprocessor = DataPreprocessor.load(prep_path)
            if os.path.exists(prof_path):
                self.load_profiler = LoadProfiler.load(prof_path)
            if os.path.exists(anom_path):
                self.anomaly_detector = AnomalyDetector.load(anom_path)
            if os.path.exists(fault_path):
                self.fault_classifier = FaultClassifier.load(fault_path)
        except Exception as e:
            print(f"Notice: InferenceEngine could not load some models: {e}")

    def _determine_abnormal_parameters(self, reading: Dict[str, Any]) -> List[str]:
        """Detects specific electrical sensor parameters violating physical boundaries."""
        app_name = reading.get("Appliance_Type", "Ceiling Fan")
        spec = APPLIANCE_SPECS.get(app_name, APPLIANCE_SPECS["Ceiling Fan"])

        v = float(reading.get("Voltage", 230.0))
        cur = float(reading.get("Current", spec["rated_current"]))
        p = float(reading.get("Active_Power", spec["rated_power"]))
        pf = float(reading.get("Power_Factor", spec["nominal_pf"]))
        temp = float(reading.get("Temperature", 30.0))

        abnormal = []
        if cur > spec["rated_current"] * 1.30:
            abnormal.append(f"Current drawn ({cur:.2f}A) exceeds rated current ({spec['rated_current']:.2f}A)")
        if temp > spec["max_safe_temp"]:
            abnormal.append(f"Housing temperature ({temp:.1f}°C) exceeds max limit ({spec['max_safe_temp']:.1f}°C)")
        if v < 205.0 or v > 250.0:
            abnormal.append(f"Grid voltage ({v:.1f}V) deviates from allowable nominal range (207-253V)")
        if p > spec["rated_power"] * 1.30:
            abnormal.append(f"Active power ({p:.0f}W) exceeds rated envelope ({spec['rated_power']:.0f}W)")
        if abs(pf - spec["nominal_pf"]) > 0.18 or pf < 0.60:
            abnormal.append(f"Power factor ({pf:.2f}) degraded relative to nominal specification ({spec['nominal_pf']:.2f})")

        return abnormal

    def _determine_severity(self, fault: str, current: float, rated_current: float,
                            temp: float, max_safe_temp: float,
                            anomaly_score: float) -> Tuple[str, str]:
        """
        Determines severity status and badge color according to IEEE/safety criteria:
        Critical (Red) -> High Risk (Orange) -> Warning (Yellow) -> Normal (Green).
        """
        if (current > rated_current * 2.0) or (fault in ["Overcurrent", "Potential Electrical Fault"]):
            return "Critical", "Red"

        if (temp > max_safe_temp + 5.0) or (current > rated_current * 1.35) or (fault in ["Overload", "Overheating", "Voltage Abnormality"]):
            return "High Risk", "Orange"

        if (anomaly_score > 0.55) or (fault == "Abnormal Power Consumption") or (temp > max_safe_temp):
            return "Warning", "Yellow"

        return "Normal", "Green"

    def predict_single(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes real-time inference pipeline on single telemetry reading.
        """
        if not self.is_loaded:
            raise RuntimeError("Inference models are not loaded. Run train_models.py first.")

        # Ensure dictionary is wrapped in single-row DataFrame
        df_single = pd.DataFrame([reading])

        # Feature Transformation
        X, _ = self.preprocessor.transform(df_single)

        # 1. Fault Classification
        pred_fault, confidence, _ = self.fault_classifier.predict_single(X[0])

        # 2. Anomaly Detection
        is_anom, anom_score = self.anomaly_detector.predict_single(X[0])

        # 3. Load Profiling
        profile_label, category = self.load_profiler.predict_single(X[0])

        # 4. Attribution and Severity
        abnormal_params = self._determine_abnormal_parameters(reading)
        app_name = reading.get("Appliance_Type", "Ceiling Fan")
        spec = APPLIANCE_SPECS.get(app_name, APPLIANCE_SPECS["Ceiling Fan"])
        cur = float(reading.get("Current", spec["rated_current"]))
        temp = float(reading.get("Temperature", 30.0))

        severity, severity_color = self._determine_severity(
            fault=pred_fault,
            current=cur,
            rated_current=spec["rated_current"],
            temp=temp,
            max_safe_temp=spec["max_safe_temp"],
            anomaly_score=anom_score
        )

        action = self.fault_classifier.get_corrective_action(pred_fault)

        return {
            "predicted_fault": pred_fault,
            "confidence": float(confidence),
            "severity": severity,
            "severity_color": severity_color,
            "is_anomaly": bool(is_anom),
            "anomaly_score": float(anom_score),
            "load_profile_label": profile_label,
            "consumption_category": category,
            "abnormal_parameters": abnormal_params,
            "corrective_action": action
        }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes vectorized batch inference over a DataFrame of telemetry records.
        """
        if not self.is_loaded:
            raise RuntimeError("Inference models are not loaded. Run train_models.py first.")

        df_out = df.copy()
        X, _ = self.preprocessor.transform(df_out)

        # Batch Predictions
        faults, confs, _ = self.fault_classifier.predict(X)
        is_anoms, anom_scores = self.anomaly_detector.predict(X)
        profiles = self.load_profiler.predict_profile(X)

        severities = []
        severity_colors = []
        actions = []
        abnormal_list = []

        for idx, row in df_out.iterrows():
            row_dict = row.to_dict()
            app_name = row_dict.get("Appliance_Type", "Ceiling Fan")
            spec = APPLIANCE_SPECS.get(app_name, APPLIANCE_SPECS["Ceiling Fan"])
            cur = float(row_dict.get("Current", spec["rated_current"]))
            temp = float(row_dict.get("Temperature", 30.0))
            f_label = faults[idx]
            a_score = float(anom_scores[idx])

            sev, color = self._determine_severity(
                fault=f_label,
                current=cur,
                rated_current=spec["rated_current"],
                temp=temp,
                max_safe_temp=spec["max_safe_temp"],
                anomaly_score=a_score
            )
            severities.append(sev)
            severity_colors.append(color)
            actions.append(self.fault_classifier.get_corrective_action(f_label))
            abnormal_list.append("; ".join(self._determine_abnormal_parameters(row_dict)))

        df_out["Predicted_Fault"] = faults
        df_out["Model_Confidence"] = np.round(confs, 4)
        df_out["Is_Anomaly"] = is_anoms
        df_out["Anomaly_Score"] = np.round(anom_scores, 4)
        df_out["Load_Profile"] = profiles
        df_out["Alert_Severity"] = severities
        df_out["Severity_Color"] = severity_colors
        df_out["Abnormal_Parameters"] = abnormal_list
        df_out["Corrective_Action"] = actions

        return df_out
