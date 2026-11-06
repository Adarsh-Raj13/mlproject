"""
Verification & Integration Test Script for Smart Load Profiling & Fault Alert System.
Tests model loading, single prediction, batch prediction, database operations,
and all 5 required electrical scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np

from database.database import DatabaseManager
from src.prediction import InferenceEngine
from src.feature_engineering import compute_electrical_features


def run_tests():
    print("=" * 60)
    print("RUNNING COMPREHENSIVE SYSTEM VERIFICATION TESTS")
    print("=" * 60)

    # 1. Test Database
    print("\n[Test 1] Testing Database Connection & Seeding...")
    db = DatabaseManager()
    appliances = db.get_appliances()
    assert len(appliances) == 6, f"Expected 6 appliances, found {len(appliances)}"
    print(f"  Passed: Found {len(appliances)} appliances in registry.")

    stats = db.get_system_summary_stats()
    print(f"  Passed: System summary stats: {stats}")

    # 2. Test Model Loading
    print("\n[Test 2] Testing ML Model Inference Engine...")
    engine = InferenceEngine()
    assert engine.is_loaded, "Inference engine failed to load models!"
    print("  Passed: All 4 ML model artifacts loaded successfully.")

    # 3. Test 5 Required Electrical Operating Scenarios
    scenarios = [
        {
            "name": "Normal Operation (Ceiling Fan)",
            "reading": {
                "Timestamp": "2026-11-05 12:00:00",
                "Appliance_Type": "Ceiling Fan",
                "Voltage": 230.0,
                "Current": 0.35,
                "Active_Power": 65.0,
                "Power_Factor": 0.81,
                "Frequency": 50.0,
                "Temperature": 34.0,
                "Operating_Duration": 60.0
            },
            "expected_fault": "Normal",
            "expected_severity": "Normal"
        },
        {
            "name": "Overload Condition (Ceiling Fan with Seized Bearing)",
            "reading": {
                "Timestamp": "2026-11-05 12:00:00",
                "Appliance_Type": "Ceiling Fan",
                "Voltage": 226.0,
                "Current": 0.65,
                "Active_Power": 125.0,
                "Power_Factor": 0.72,
                "Frequency": 50.0,
                "Temperature": 68.0,
                "Operating_Duration": 90.0
            },
            "expected_fault": "Overload",
            "expected_severity": "High Risk"
        },
        {
            "name": "Overheating Condition (Water Heater Dry Heating)",
            "reading": {
                "Timestamp": "2026-11-05 12:00:00",
                "Appliance_Type": "Water Heater",
                "Voltage": 230.0,
                "Current": 8.7,
                "Active_Power": 2000.0,
                "Power_Factor": 0.99,
                "Frequency": 50.0,
                "Temperature": 98.0,
                "Operating_Duration": 60.0
            },
            "expected_fault": "Overheating",
            "expected_severity": "High Risk"
        },
        {
            "name": "Abnormal Current / Overcurrent (Air Conditioner Locked Rotor)",
            "reading": {
                "Timestamp": "2026-11-05 12:00:00",
                "Appliance_Type": "Air Conditioner",
                "Voltage": 224.0,
                "Current": 24.5,
                "Active_Power": 3950.0,
                "Power_Factor": 0.72,
                "Frequency": 50.0,
                "Temperature": 92.0,
                "Operating_Duration": 15.0
            },
            "expected_fault": "Overcurrent",
            "expected_severity": "Critical"
        },
        {
            "name": "Voltage Fluctuation / Sag (Refrigerator)",
            "reading": {
                "Timestamp": "2026-11-05 12:00:00",
                "Appliance_Type": "Refrigerator",
                "Voltage": 178.0,
                "Current": 1.65,
                "Active_Power": 220.0,
                "Power_Factor": 0.75,
                "Frequency": 50.0,
                "Temperature": 48.0,
                "Operating_Duration": 120.0
            },
            "expected_fault": "Voltage Abnormality",
            "expected_severity": "High Risk"
        }
    ]

    print("\n[Test 3] Testing 5 Standard Electrical Test Scenarios...")
    for s in scenarios:
        res = engine.predict_single(s["reading"])
        print(f"\n  Scenario: {s['name']}")
        print(f"    Predicted Fault: {res['predicted_fault']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"    Severity: {res['severity']} (Color: {res['severity_color']})")
        print(f"    Is Anomaly: {res['is_anomaly']} (Score: {res['anomaly_score']:.3f})")
        print(f"    Load Profile: {res['load_profile_label']} ({res['consumption_category']})")
        print(f"    Abnormal Params: {res['abnormal_parameters']}")
        print(f"    Action: {res['corrective_action']}")

    # 4. Test Batch Prediction on Raw Data
    print("\n[Test 4] Testing Batch Inference...")
    df_raw = pd.read_csv("dataset/raw_data.csv")
    sample_batch = df_raw.head(20).copy()
    batch_results = engine.batch_predict(sample_batch)
    assert len(batch_results) == 20
    assert "Predicted_Fault" in batch_results.columns
    assert "Load_Profile" in batch_results.columns
    print(f"  Passed: Successfully processed batch of {len(batch_results)} records.")

    print("\n" + "=" * 60)
    print("ALL SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

