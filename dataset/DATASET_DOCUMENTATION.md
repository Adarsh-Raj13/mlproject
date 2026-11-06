# DATASET DOCUMENTATION
# Smart Load Profiling & Electrical Fault Telemetry Dataset

## 1. Overview
This dataset contains high-frequency electrical telemetry records simulated from 6 commercial and residential electrical loads under real-world single-phase 230V RMS AC operating regimes.

Total Records: 12,000 (2,000 observations per appliance)
Time Horizon: 30 days continuous telemetry with timestamps.
Operational Statuses: 7 conditions (Normal, Overload, Overheating, Overcurrent, Voltage Abnormality, Abnormal Power Consumption, Potential Electrical Fault).

---

## 2. Feature Schema & Unit Specifications

| Column Name | Data Type | Physical Unit | Description & AC Circuit Formula |
|---|---|---|---|
| `Timestamp` | DateTime | ISO 8601 | Telemetry packet transmission timestamp |
| `Appliance_Type` | String | Categorical | Specific monitored equipment name |
| `Voltage` | Float | Volts (V RMS) | Grid potential across appliance terminals |
| `Current` | Float | Amperes (A RMS) | Load current drawn through phase conductor |
| `Active_Power` | Float | Watts (W) | Real electrical power dissipated: $P = V \cdot I \cdot \text{PF}$ |
| `Reactive_Power` | Float | VAR | Quadrature magnetizing power: $Q = \sqrt{S^2 - P^2}$ |
| `Apparent_Power` | Float | Volt-Amperes (VA) | Total complex power: $S = V \cdot I$ |
| `Power_Factor` | Float | Dimensionless [0-1] | Displacement ratio $\cos\phi = P / S$ |
| `Frequency` | Float | Hertz (Hz) | AC line frequency (Nominal 50.0 Hz) |
| `Temperature` | Float | Degrees Celsius (°C) | Surface/housing thermal sensor measurement |
| `Operating_Duration` | Float | Minutes | Continuous active running duration |
| `Energy_Consumption` | Float | Kilowatt-hours (kWh) | Cumulative energy: $E = (P \cdot t_{\text{mins}}) / 60,000$ |
| `Operating_Condition` | String | Target Class | Ground-truth fault or normal label |

---

## 3. Engineered Features (`processed_data.csv`)

| Feature | Formula | Engineering Significance |
|---|---|---|
| `Impedance` | $\|Z\| = V / I$ ($\Omega$) | Reflects load impedance magnitude |
| `Power_Ratio` | $P / P_{\text{rated}}$ | Quantifies overload or light-duty ratio |
| `Current_Ratio` | $I / I_{\text{rated}}$ | Indicates inrush, locked rotor, or partial short |
| `Voltage_Deviation` | $\|V - 230\| / 230$ | Flags brownouts, sags, and voltage spikes |
| `Temp_Margin` | $T - T_{\text{max\_safe}}$ (°C) | Positive value triggers thermal early warning |
| `PF_Deviation` | $\|\text{PF} - \text{PF}_{\text{nominal}}\|$ | Diagnoses capacitor failure or non-linear distortion |
