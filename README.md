# ⚡ SMART LOAD MONITOR
### Smart Load Profiling and Early Fault Alert System for Electrical Appliances

An end-to-end Machine Learning and Industrial IoT monitoring system that analyzes electrical telemetry from household and commercial appliances, understands baseline energy signatures, identifies anomalous operating behavior, classifies electrical faults, and generates early safety alerts.

---

## 🌟 Key Capabilities

- **6 Monitored Electrical Appliances**:
  1. Ceiling Fan (Inductive Single-Phase Induction Motor)
  2. Refrigerator (Cyclic Compressor Motor)
  3. Air Conditioner (1.5 Ton Heavy Inductive Compressor + Condenser Fan)
  4. Washing Machine (Dynamic Mixed Agitation / High-Speed Spin / Heater)
  5. Electric Iron (Pure Resistive Bang-Bang Thermostat)
  6. Water Heater (High-Demand Immersion Resistive Element)

- **AC Electrical Engineering Precision**:
  - $S = V \times I$ (Apparent Power in VA)
  - $P = V \times I \times \text{PF}$ (Active Power in W)
  - $Q = \sqrt{S^2 - P^2}$ (Reactive Power in VAR)
  - $\text{PF} = \frac{P}{S}$ (Power Factor)
  - $E = \int \frac{P}{1000} \, dt$ (Energy in kWh)
  - Physical boundary clipping, thermal dissipation dynamics, and nominal design envelopes.

- **3 Machine Learning Modules**:
  - **Load Profiling**: Unsupervised K-Means clustering ($k=4$) and Random Forest classifier for energy consumption regime categorization (Eco, Standard Normal, Heavy Load, Peak Surge).
  - **Anomaly Detection**: Isolation Forest and Local Outlier Factor (LOF) for telemetry outlier scoring with root-cause parameter attribution.
  - **Fault Classification**: Random Forest, Decision Tree, Support Vector Machine (SVM), and XGBoost benchmarking across 7 operational conditions (Normal, Overload, Overheating, Overcurrent, Voltage Abnormality, Abnormal Power Consumption, Potential Electrical Fault).

- **6-Page Streamlit Web Application**:
  - **Page 1: Home Dashboard**: High-level KPIs, system health percentage, appliance status cards, load distribution pie chart, and live power trends.
  - **Page 2: Appliance Monitoring**: Telemetry gauges (Voltage, Current, Power, Temperature), complex power curves ($P, Q, S$), and thermal gradient charts.
  - **Page 3: Load Profiling**: Power factor analysis, $P-Q$ plane scatter clusters, peak demand periods, and energy regime distributions.
  - **Page 4: Fault Detection**: Live manual parameter testing, CSV batch upload with ML inference, pre-configured field fault scenarios, and corrective action recommendations.
  - **Page 5: Early Alert Centre**: Alert triage by severity (Green, Yellow, Orange, Red), status management (Active, Acknowledged, Resolved), and parameter violation root-causes.
  - **Page 6: Analytics & Reports**: Model evaluation benchmarks, confusion matrices, anomaly score distributions, energy audit logs, and one-click CSV report exports.

- **Demonstration Mode**:
  - Live IoT Telemetry Simulator: Inject simulated telemetry packets with authentic noise and fault characteristics into the SQLite database at any time.

---

## 📁 Project Directory Structure

```
ml project/
├── app.py                            # Streamlit Application Entry Point
├── train_models.py                   # Complete ML Pipeline & DB Seeding Orchestration
├── test_system.py                    # Comprehensive Verification & Test Suite
├── requirements.txt                  # Python Dependencies
├── README.md                         # Installation & Execution Guide
├── PROJECT_DOCUMENTATION.md          # Technical & Electrical Engineering Methodology
├── smart_load_monitor.db             # SQLite Database (Auto-initialized)
│
├── dataset/
│   ├── generate_dataset.py           # Physically Plausible Synthetic Generator
│   ├── raw_data.csv                  # 12,000 Row Telemetry Dataset
│   ├── processed_data.csv            # Feature-Engineered Dataset
│   └── DATASET_DOCUMENTATION.md      # Feature Definitions & Sensor Specifications
│
├── models/
│   ├── load_profiling_model.pkl      # Trained K-Means & Classifier Pipeline
│   ├── anomaly_detection_model.pkl   # Trained Isolation Forest & LOF Pipeline
│   ├── fault_classification_model.pkl# Trained Fault Classifier (Random Forest / XGBoost)
│   └── preprocessing_pipeline.pkl    # Scaler & Categorical Encoders
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb# Jupyter EDA Notebook
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py        # Electrical Engineering Metrics ($S, Q, PF, |Z|$)
│   ├── data_preprocessing.py         # Pipeline Cleaning, Encoding, Scaling
│   ├── load_profiling.py             # K-Means Load Regime Modeling
│   ├── anomaly_detection.py          # Isolation Forest & Threshold Breaches
│   ├── fault_classification.py       # Multi-class Classifier Benchmarking
│   ├── model_evaluation.py           # Metric Calculations & Figure Generation
│   └── prediction.py                 # Production Inference Engine
│
├── database/
│   ├── __init__.py
│   └── database.py                   # SQLite Database Manager & Schemas
│
├── components/
│   ├── __init__.py
│   ├── dashboard.py                  # Page 1: Home Dashboard
│   ├── appliance_monitoring.py       # Page 2: Appliance Telemetry
│   ├── load_analytics.py             # Page 3: Load Profiling & Regimes
│   ├── fault_detection.py            # Page 4: Interactive Fault Analysis
│   ├── alert_centre.py               # Page 5: Alert Lifecycle Management
│   └── analytics_reports.py          # Page 6: Performance Reports & Export
│
├── assets/
│   └── style.css                     # Modern Industrial IoT UI Theme
│
└── reports/
    ├── model_evaluation.txt          # Benchmark Evaluation Summary Log
    ├── confusion_matrix.png          # Fault Confusion Matrix Heatmap
    ├── model_comparison.png          # Algorithm Comparison Bar Chart
    ├── anomaly_distribution.png      # Isolation Forest Anomaly Histogram
    └── cluster_distribution.png      # Discovered Operating Regimes Bar Chart
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites
- Python 3.10 or Python 3.11 installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models & Initialize Database (Optional if pre-trained)
The repository includes pre-trained models and a seeded database. To regenerate datasets and retrain all models from scratch:
```bash
python train_models.py
```

### 4. Run System Verification Tests
Execute the verification suite to test database access, model inference, and 5 electrical fault scenarios:
```bash
python test_system.py
```

### 5. Launch the Web Application
Start the interactive Streamlit engineering dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Testing Scenarios Included in the Dashboard

In **Page 4 (Fault Detection) -> Pre-configured Scenarios**, you can immediately test:
1. **Normal Operation**: Ceiling fan drawing nominal 0.35 A at 230 V (100% Normal confidence).
2. **Mechanical Overload**: Ceiling fan bearing seizure with elevated current (0.65 A) and thermal stress (68°C).
3. **Overheating**: Water heater thermostat failure causing dry heating and extreme temperature (98°C).
4. **Overcurrent / Locked Rotor**: Air conditioner compressor motor lock drawing 24.5 A (Critical alert).
5. **Voltage Abnormality / Sag**: Refrigerator running on a severe grid voltage sag (178 V) drawing compensation current.

---

## ⚖️ Safety & Engineering Disclaimer
This application is designed as an educational, diagnostic, and decision-support system. It is not a certified electrical protection device and cannot replace hardware circuit breakers (MCBs), fuses, residual current devices (RCDs), or qualified electrical inspections.

