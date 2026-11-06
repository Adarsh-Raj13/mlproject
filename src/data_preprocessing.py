"""
Data Preprocessing Pipeline Module.
Performs data sanitization, physical boundary enforcement,
feature normalization, categorical encoding, and serialization.
"""

import os
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from src.feature_engineering import compute_electrical_features, APPLIANCE_SPECS


NUMERICAL_FEATURES = [
    "Voltage", "Current", "Active_Power", "Reactive_Power", "Apparent_Power",
    "Power_Factor", "Frequency", "Temperature", "Operating_Duration",
    "Impedance", "Power_Ratio", "Current_Ratio", "Voltage_Deviation",
    "Temp_Margin", "PF_Deviation"
]


class DataPreprocessor:
    """End-to-end preprocessing pipeline for electrical telemetry records."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.appliance_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.fault_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and sanitizes raw sensor observations within physical bounds."""
        df_clean = df.copy()

        # Handle numeric defaults and physical clipping
        if "Voltage" in df_clean.columns:
            df_clean["Voltage"] = pd.to_numeric(df_clean["Voltage"], errors="coerce").fillna(230.0).clip(120.0, 320.0)
        if "Current" in df_clean.columns:
            df_clean["Current"] = pd.to_numeric(df_clean["Current"], errors="coerce").fillna(0.1).clip(0.001, 60.0)
        if "Power_Factor" in df_clean.columns:
            df_clean["Power_Factor"] = pd.to_numeric(df_clean["Power_Factor"], errors="coerce").fillna(0.85).clip(0.1, 1.0)
        if "Frequency" in df_clean.columns:
            df_clean["Frequency"] = pd.to_numeric(df_clean["Frequency"], errors="coerce").fillna(50.0).clip(45.0, 55.0)
        if "Temperature" in df_clean.columns:
            df_clean["Temperature"] = pd.to_numeric(df_clean["Temperature"], errors="coerce").fillna(35.0).clip(10.0, 150.0)
        if "Operating_Duration" in df_clean.columns:
            df_clean["Operating_Duration"] = pd.to_numeric(df_clean["Operating_Duration"], errors="coerce").fillna(30.0).clip(0.0, 1440.0)

        # Fill unknown appliance
        if "Appliance_Type" in df_clean.columns:
            df_clean["Appliance_Type"] = df_clean["Appliance_Type"].fillna("Ceiling Fan")

        return df_clean

    def fit(self, df_raw: pd.DataFrame) -> "DataPreprocessor":
        """Fits scalers and encoders on raw or feature-engineered telemetry dataset."""
        df_clean = self.clean_raw_data(df_raw)
        df_feat = compute_electrical_features(df_clean)

        # 1. Fit numerical scaler
        self.scaler.fit(df_feat[NUMERICAL_FEATURES])

        # 2. Fit one-hot encoder on appliance types
        app_series = df_feat[["Appliance_Type"]]
        self.appliance_encoder.fit(app_series)

        # 3. Fit target label encoder if Operating_Condition is present
        if "Operating_Condition" in df_feat.columns:
            # Ensure "Normal" is present and canonical classes are captured
            all_classes = [
                "Normal", "Overload", "Overheating", "Overcurrent",
                "Voltage Abnormality", "Abnormal Power Consumption", "Potential Electrical Fault"
            ]
            self.fault_encoder.fit(all_classes)

        # Store feature names
        app_feature_names = [f"App_{cat}" for cat in self.appliance_encoder.categories_[0]]
        self.feature_names = NUMERICAL_FEATURES + app_feature_names
        self.is_fitted = True
        return self

    def transform(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms raw data into feature matrix X and target vector y.
        Returns (X, y) or (X, None) if Operating_Condition is absent.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transform.")

        df_clean = self.clean_raw_data(df_raw)
        df_feat = compute_electrical_features(df_clean)

        # Transform numerical features
        num_scaled = self.scaler.transform(df_feat[NUMERICAL_FEATURES])

        # Transform appliance categories
        app_series = df_feat[["Appliance_Type"]]
        app_encoded = self.appliance_encoder.transform(app_series)

        # Combine
        X = np.hstack([num_scaled, app_encoded])

        y = None
        if "Operating_Condition" in df_feat.columns and df_feat["Operating_Condition"].notna().any():
            # Filter or map known classes
            y = self.fault_encoder.transform(df_feat["Operating_Condition"].astype(str))

        return X, y

    def save(self, filepath: str):
        """Serializes preprocessor artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Loads preprocessor artifact from disk."""
        return joblib.load(filepath)

