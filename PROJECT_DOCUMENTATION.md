# PROJECT DOCUMENTATION
# Smart Load Profiling and Early Fault Alert System for Electrical Appliances

## 1. Executive Summary
The **Smart Load Profiling and Early Fault Alert System** is an end-to-end industrial IoT and machine learning software system designed to analyze electrical parameters of residential and light-commercial appliances, profile normal operating energy regimes, detect anomalous electrical signatures, classify electrical fault modes, and generate early warning alerts.

The system is built on electrical engineering AC circuit principles, integrating real-time telemetry processing, SQLite time-series persistence, multiple scikit-learn/xgboost machine learning models, and an interactive Streamlit engineering dashboard.

---

## 2. System Architecture & Component Interaction

```
[ Appliance Telemetry / Sensor Node / Simulation ]
                         │
                         ▼
             [ AC Engineering Validation ]
    (S = V * I,  P = S * PF,  Q = √(S² - P²),  E = ∫P dt)
                         │
                         ▼
            [ SQLite Database Layer ]
   (appliances, electrical_readings, alerts, predictions)
                         │
                         ▼
        [ Preprocessing & Feature Engineering ]
  (StandardScaler, One-Hot Encoding, Deviation Metrics)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
[ Load Profiler ] [ Anomaly Detector ] [ Fault Classifier ]
 (K-Means + RF)   (Isolation Forest)     (Random Forest)
        └────────────────┬────────────────┘
                         │
                         ▼
               [ Inference Engine ]
   (Severity Triage, Parameter Attribution, Actions)
                         │
                         ▼
          [ Streamlit Web Application ]
 ┌───────────────────────┬───────────────────────┐
 │ 🏠 Home Dashboard     │ 🔌 Appliance Monitor  │
 │ ⚡ Load Profiling     │ 🔍 Fault Detection    │
 │ 🚨 Alert Centre       │ 📊 Reports & Audits   │
 └───────────────────────┴───────────────────────┘
```

---

## 3. Electrical Engineering Fundamentals

### 3.1 AC Power Relationships (Single-Phase AC Grid, 230V RMS, 50 Hz)
For single-phase alternating current circuits:

1. **Apparent Power ($S$)**:
   $$S = V_{\text{RMS}} \times I_{\text{RMS}} \quad [\text{Volt-Amperes, VA}]$$
   Represents total complex power transmitted from grid to load.

2. **Power Factor ($\text{PF}$ or $\cos \phi$)**:
   $$\text{PF} = \frac{P}{S} \quad [0.0 \le \text{PF} \le 1.0]$$
   Quantifies the phase displacement between AC voltage and current.

3. **Active Real Power ($P$)**:
   $$P = V_{\text{RMS}} \times I_{\text{RMS}} \times \text{PF} \quad [\text{Watts, W}]$$
   Represents real work dissipated as mechanical torque, cooling, or heat.

4. **Reactive Quadrature Power ($Q$)**:
   $$Q = \sqrt{\max\left(0, \, S^2 - P^2\right)} \quad [\text{Volt-Amperes Reactive, VAR}]$$
   Alternates between source and load magnetic fields (e.g. induction motor windings).

5. **Electrical Energy Accumulation ($E$)**:
   $$E = \int_{0}^{t} \frac{P(t)}{1000 \times 3600} \, dt \approx \frac{P \times \Delta t_{\text{mins}}}{60 \times 1000} \quad [\text{Kilowatt-hours, kWh}]$$

6. **Impedance Magnitude ($|Z|$)**:
   $$|Z| = \frac{V_{\text{RMS}}}{I_{\text{RMS}}} \quad [\Omega]$$

### 3.2 Monitored Appliance Characteristics
- **Ceiling Fan (75 W, 0.40 A, PF 0.82)**: Inductive single-phase induction motor with starting capacitor. Susceptible to bearing seizure (causing overload) and capacitor degradation (causing low PF).
- **Refrigerator (160 W, 0.95 A, PF 0.76)**: Cyclic inductive compressor motor. Normal operation exhibits regular duty cycles (40-60%). Continuous non-stop running signals refrigerant leakage or failed thermostat.
- **Air Conditioner (1800 W, 8.50 A, PF 0.92)**: 1.5 Ton residential rotary compressor + fan motor. Draws high current; locked rotor draws up to $3\times$ nominal current ($>24\text{ A}$) with rapid thermal escalation.
- **Washing Machine (600 W - 2000 W, 3.20 A, PF 0.81)**: Dual-mode system; motor agitation is inductive ($300-600\text{ W}$, $\text{PF} \approx 0.80$), while heating cycle is resistive ($1800-2200\text{ W}$, $\text{PF} \approx 0.99$).
- **Electric Iron (1200 W, 5.22 A, PF 0.99)**: Pure resistive nichrome element with bimetallic thermostat switch. Thermostat weld causes continuous thermal rise beyond $90^\circ\text{C}$ casing limit.
- **Water Heater (2000 W, 8.70 A, PF 0.99)**: Pure resistive immersion element. Dry heating (running without water) produces rapid localized heating ($>95^\circ\text{C}$) and risk of element rupture.

---

## 4. Machine Learning Implementation

### Module A: Appliance Load Profiling
- **Unsupervised Modeling**: K-Means clustering ($k=4$) sorts electrical operating records into four distinct energy regimes:
  1. *Eco / Light Duty*: Low standby current, idle appliances.
  2. *Standard Normal*: Nominal steady-state operation within designed parameters.
  3. *Heavy Continuous*: High sustained demand (e.g. full-blast heating or cooling).
  4. *Peak Surge*: Motor inrush transients, severe mechanical load, or anomalous overload.
- **Supervised Classifier**: Random Forest and Decision Tree models predict regime membership in real time for immediate dashboard categorization.

### Module B: Anomaly Detection
- **Isolation Forest**: Builds an ensemble of 150 isolation trees to isolate anomalies. Normal data requires deep trees, while abnormal spikes ($V, I, P, T$) are isolated quickly at root levels.
- **Local Outlier Factor (LOF)**: Measures local density deviation relative to $k$-nearest neighbors.
- **Root-Cause Parameter Attribution**: Pinpoints exact parameters violating empirical 3-sigma / 99th percentile bounds per appliance type.

### Module C: Fault Classification
- Evaluates four competitive algorithms:
  - Random Forest Classifier (120 estimators, max depth 12)
  - Decision Tree Classifier (max depth 8)
  - Support Vector Machine (RBF kernel, calibrated probabilities)
  - XGBoost Classifier (100 estimators, learning rate 0.1)
- Model comparison is automatically benchmarked using stratified cross-validation on test partitions ($n=2,400$).
- Random Forest achieved top performance with high macro/weighted F1-scores across all 7 fault classes.

---

## 5. Early Warning Alert Logic & Severity Triage

| Severity | Color Code | Trigger Criteria | Example Condition | Corrective Engineering Action |
|---|---|---|---|---|
| **Critical** | Red | $I > 2.2 \times I_{\text{rated}}$ OR Fault $\in$ {Overcurrent, Potential Electrical Fault} | Compressor locked rotor, winding insulation short | Immediate disconnect. Inspect winding insulation and Megger resistance before re-energizing. |
| **High Risk** | Orange | $T > 75^\circ\text{C}$ OR $I > 1.4 \times I_{\text{rated}}$ OR Fault $\in$ {Overload, Overheating, Voltage Abnormality} | Bearing seizure, dry heating, grid voltage sag | De-energize or shed load. Verify ventilation clearance, thermostat cut-off, and supply line voltage. |
| **Warning** | Yellow | Anomaly Score $> 0.55$ OR Fault == Abnormal Power Consumption | Defective thermostat, refrigerant gas leak | Schedule preventive maintenance. Check thermal sensor calibration and duty cycle duration. |
| **Normal** | Green | All parameters within 3-sigma design bounds | Optimal steady operation | No intervention required. System operating nominally. |

---

## 6. Hardware Integration Architecture (ESP32 / Arduino Guide)

To stream real-time hardware telemetry into this platform:
1. **Sensors Required**:
   - Current: Non-invasive SCT-013-000 (100A:50mA) or ACS712 Hall-effect sensor.
   - Voltage: ZMPT101B AC voltage transformer module.
   - Temperature: DS18B20 digital 1-wire probe or PT100 RTD sensor.
2. **Microcontroller**: ESP32 with dual-core 240 MHz MCU and WiFi.
3. **Payload Structure (JSON over HTTP POST or MQTT)**:
   ```json
   {
     "appliance_type": "Air Conditioner",
     "voltage": 228.4,
     "current": 8.42,
     "active_power": 1780.2,
     "power_factor": 0.925,
     "frequency": 50.02,
     "temperature": 52.6,
     "operating_duration": 45.0
   }
   ```
4. **Backend Ingestion Endpoint**: The `DatabaseManager.insert_reading()` method accepts this payload directly, triggers `InferenceEngine.predict_single()`, and logs alerts when safety thresholds are breached.

---

## 7. Educational and Safety Disclaimer
This system is an educational, research, and predictive decision-support software platform. It is **not** a certified electrical safety device and **cannot** replace calibrated hardware thermal-magnetic circuit breakers (MCBs), residual current devices (RCDs), physical fuses, or periodic certified electrical safety audits.

