"""
Comprehensive Training & Evaluation Pipeline for Smart Load Profiling & Fault Alert System.
Orchestrates dataset generation, preprocessing, model training, benchmarking,
saving model artifacts, generating evaluation reports and charts, and seeding SQLite database.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from dataset.generate_dataset import generate_full_dataset
from src.data_preprocessing import DataPreprocessor
from src.load_profiling import LoadProfiler
from src.anomaly_detection import AnomalyDetector
from src.fault_classification import FaultClassifier
from src.model_evaluation import evaluate_and_generate_reports
from src.feature_engineering import compute_electrical_features
from database.database import DatabaseManager

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def main():
    print("=" * 70)
    print("STARTING SMART LOAD PROFILING & FAULT ALERT ML PIPELINE")
    print("=" * 70)

    # 1. Dataset Generation / Loading
    raw_csv = os.path.join(DATASET_DIR, "raw_data.csv")
    if not os.path.exists(raw_csv):
        print("\n[Step 1/7] Generating physically-plausible electrical dataset...")
        generate_full_dataset(samples_per_appliance=2000, output_dir=DATASET_DIR)
    else:
        print(f"\n[Step 1/7] Dataset found at: {raw_csv}")

    df_raw = pd.read_csv(raw_csv)
    print(f"Loaded raw dataset with {len(df_raw)} records across {df_raw['Appliance_Type'].nunique()} appliances.")

    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/7] Preprocessing data and engineering electrical features...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(df_raw)

    # Save preprocessing pipeline
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    preprocessor.save(prep_path)
    print(f"Preprocessing pipeline saved to: {prep_path}")

    # Feature transformation
    X, y = preprocessor.transform(df_raw)
    df_clean = preprocessor.clean_raw_data(df_raw)
    df_feat = compute_electrical_features(df_clean)

    # Train / Test Stratified Split (80% train, 20% test)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_feat, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")

    # 3. Load Profiling Model Training
    print("\n[Step 3/7] Training Load Profiling Models (K-Means + Classifiers)...")
    load_profiler = LoadProfiler(n_clusters=4)
    load_profiler.fit(X_train, df_train)
    prof_path = os.path.join(MODELS_DIR, "load_profiling_model.pkl")
    load_profiler.save(prof_path)
    print(f"Load Profiling model saved to: {prof_path}")

    # 4. Anomaly Detection Model Training
    print("\n[Step 4/7] Training Anomaly Detection Models (Isolation Forest + LOF)...")
    anomaly_detector = AnomalyDetector(contamination=0.10)
    anomaly_detector.fit(X_train, df_train)
    anom_path = os.path.join(MODELS_DIR, "anomaly_detection_model.pkl")
    anomaly_detector.save(anom_path)
    print(f"Anomaly Detection model saved to: {anom_path}")

    # 5. Fault Classification Model Training & Comparison
    print("\n[Step 5/7] Training & Benchmarking Fault Classification Algorithms...")
    fault_classifier = FaultClassifier(class_names=list(preprocessor.fault_encoder.classes_))
    comparison_df = fault_classifier.train_and_compare(X_train, y_train, X_test, y_test)
    fault_path = os.path.join(MODELS_DIR, "fault_classification_model.pkl")
    fault_classifier.save(fault_path)
    print(f"Best Fault Classifier saved to: {fault_path}")

    # 6. Evaluation, Visualizations & Reports
    print("\n[Step 6/7] Evaluating test performance and generating visual reports...")
    y_pred, _, _ = fault_classifier.predict(X_test)
    _, anom_scores = anomaly_detector.predict(X_test)
    test_profiles = load_profiler.predict_profile(X_test)

    eval_results = evaluate_and_generate_reports(
        y_true=y_test,
        y_pred=y_pred,
        class_names=fault_classifier.class_names,
        comparison_df=comparison_df,
        anomaly_scores=anom_scores,
        load_profiles=test_profiles,
        reports_dir=REPORTS_DIR
    )
    print(f"Test Accuracy: {eval_results['accuracy']:.4f} | F1-Score: {eval_results['f1_score']:.4f}")

    # 7. Seed SQLite Database with Historical Telemetry & Initial Alerts
    print("\n[Step 7/7] Initializing & Seeding SQLite Database with telemetry records...")
    db = DatabaseManager()
    appliances = db.get_appliances()
    app_id_map = {app["name"]: app["id"] for app in appliances}

    # Sample 100 historical readings per appliance to seed database
    history_samples = df_raw.groupby("Appliance_Type").tail(50).copy()
    history_samples = history_samples.sort_values("Timestamp")

    for _, row in history_samples.iterrows():
        app_name = row["Appliance_Type"]
        app_id = app_id_map.get(app_name, 1)

        reading_id = db.insert_reading(
            appliance_id=app_id,
            timestamp=row["Timestamp"],
            voltage=float(row["Voltage"]),
            current=float(row["Current"]),
            active_power=float(row["Active_Power"]),
            reactive_power=float(row["Reactive_Power"]),
            apparent_power=float(row["Apparent_Power"]),
            power_factor=float(row["Power_Factor"]),
            frequency=float(row["Frequency"]),
            temperature=float(row["Temperature"]),
            operating_duration=float(row["Operating_Duration"]),
            energy_kwh=float(row["Energy_Consumption"])
        )

        cond = row["Operating_Condition"]
        if cond != "Normal":
            # Generate appropriate alert
            sev = "Critical" if cond in ["Overcurrent", "Potential Electrical Fault"] else (
                "High Risk" if cond in ["Overload", "Overheating", "Voltage Abnormality"] else "Warning"
            )
            trigger_param = "Current" if cond in ["Overcurrent", "Overload"] else (
                "Temperature" if cond == "Overheating" else (
                    "Voltage" if cond == "Voltage Abnormality" else "Active Power"
                )
            )
            val = float(row["Current"]) if "Current" in trigger_param else (
                float(row["Temperature"]) if "Temp" in trigger_param else float(row["Voltage"])
            )
            action = fault_classifier.get_corrective_action(cond)
            db.insert_alert(
                appliance_name=app_name,
                timestamp=row["Timestamp"],
                severity=sev,
                fault_type=cond,
                trigger_parameter=trigger_param,
                trigger_value=val,
                threshold_value=val * 0.85,
                suggested_action=action
            )

    print("SQLite database successfully populated with telemetry and alerts!")
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
