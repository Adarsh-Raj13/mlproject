"""
Physically-Plausible Synthetic Electrical Telemetry Dataset Generator.
Simulates 12,000 electrical records across 6 household & industrial appliances
based on rigorous AC single-phase circuit physics and failure physics.
Generates 7 operational conditions:
- Normal
- Overload
- Overheating
- Overcurrent
- Voltage Abnormality
- Abnormal Power Consumption
- Potential Electrical Fault
"""

import os
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import APPLIANCE_SPECS, compute_electrical_features


def generate_single_reading(appliance_name: str, condition: str, base_time: datetime) -> Dict[str, Any]:
    """Generates a single physically plausible AC electrical measurement."""
    spec = APPLIANCE_SPECS[appliance_name]
    v_rated = spec["rated_voltage"]
    i_rated = spec["rated_current"]
    p_rated = spec["rated_power"]
    pf_nom = spec["nominal_pf"]
    t_safe = spec["max_safe_temp"]

    freq = round(random.gauss(50.0, 0.05), 2)
    duration = round(random.uniform(10.0, 180.0), 1)

    if condition == "Normal":
        voltage = round(random.gauss(230.0, 2.8), 1)
        # Normal operating current centered at ~85-95% of rated
        current = round(max(0.05, random.gauss(i_rated * 0.90, i_rated * 0.05)), 2)
        pf = round(min(1.0, max(0.65, random.gauss(pf_nom, 0.02))), 2)
        # Thermal rise proportional to current
        temp = round(max(26.0, random.gauss(t_safe * 0.60, 3.0)), 1)

    elif condition == "Overload":
        voltage = round(random.gauss(226.0, 3.0), 1)
        # Mechanical drag, locked load: 1.35x - 1.80x rated current
        current = round(i_rated * random.uniform(1.35, 1.80), 2)
        pf = round(max(0.60, pf_nom * random.uniform(0.85, 0.95)), 2)
        temp = round(t_safe + random.uniform(5.0, 16.0), 1)

    elif condition == "Overheating":
        voltage = round(random.gauss(230.0, 2.5), 1)
        # Normal or slightly elevated current, but thermal runaway
        current = round(i_rated * random.uniform(1.02, 1.25), 2)
        pf = round(min(1.0, max(0.70, random.gauss(pf_nom, 0.02))), 2)
        temp = round(t_safe + random.uniform(18.0, 36.0), 1)

    elif condition == "Overcurrent":
        # Locked rotor / short circuit: 2.1x - 3.2x rated current
        voltage = round(random.gauss(222.0, 4.0), 1)
        current = round(i_rated * random.uniform(2.10, 3.20), 2)
        pf = round(max(0.50, pf_nom * random.uniform(0.70, 0.85)), 2)
        temp = round(t_safe + random.uniform(16.0, 32.0), 1)
        duration = round(random.uniform(5.0, 35.0), 1)  # Tripped shortly

    elif condition == "Voltage Abnormality":
        # Severe grid sag (155-198V) or swell (250-280V)
        is_sag = random.random() < 0.65
        if is_sag:
            voltage = round(random.uniform(155.0, 198.0), 1)
            current = round(i_rated * random.uniform(1.10, 1.85), 2)
        else:
            voltage = round(random.uniform(250.0, 280.0), 1)
            current = round(i_rated * random.uniform(1.05, 1.45), 2)
        pf = round(min(1.0, max(0.55, pf_nom * random.uniform(0.85, 1.0))), 2)
        temp = round(max(25.0, t_safe * random.uniform(0.60, 0.88)), 1)

    elif condition == "Abnormal Power Consumption":
        # Thermostat stuck or continuous heavy cycle
        voltage = round(random.gauss(229.0, 2.5), 1)
        current = round(i_rated * random.uniform(1.15, 1.38), 2)
        pf = round(min(1.0, max(0.65, pf_nom * random.uniform(0.92, 1.05))), 2)
        temp = round(t_safe + random.uniform(2.0, 12.0), 1)
        duration = round(random.uniform(180.0, 360.0), 1)  # Continuous running

    elif condition == "Potential Electrical Fault":
        # Capacitor degradation (low PF), phase imbalance, arcing
        voltage = round(random.gauss(225.0, 4.5), 1)
        current = round(i_rated * random.uniform(1.20, 1.90), 2)
        pf = round(max(0.38, pf_nom * random.uniform(0.50, 0.68)), 2)
        temp = round(t_safe + random.uniform(8.0, 22.0), 1)

    else:
        # Fallback normal
        voltage = 230.0
        current = i_rated * 0.9
        pf = pf_nom
        temp = 32.0

    # AC Single-Phase Circuit Formulations
    apparent_power = round(voltage * current, 2)
    active_power = round(apparent_power * pf, 2)
    diff = max(0.0, apparent_power**2 - active_power**2)
    reactive_power = round(diff**0.5, 2)
    energy_kwh = round((active_power * duration) / 60000.0, 4)

    return {
        "Timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Appliance_Type": appliance_name,
        "Voltage": voltage,
        "Current": current,
        "Active_Power": active_power,
        "Reactive_Power": reactive_power,
        "Apparent_Power": apparent_power,
        "Power_Factor": pf,
        "Frequency": freq,
        "Temperature": temp,
        "Operating_Duration": duration,
        "Energy_Consumption": energy_kwh,
        "Operating_Condition": condition
    }


def generate_full_dataset(samples_per_appliance: int = 2000, output_dir: str = "dataset") -> pd.DataFrame:
    """
    Generates a full balanced-ratio synthetic electrical dataset across 6 appliances
    and exports raw_data.csv and processed_data.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # Condition distribution proportions (Total = 1.0)
    condition_weights = [
        ("Normal", 0.64),
        ("Overload", 0.07),
        ("Overheating", 0.07),
        ("Overcurrent", 0.06),
        ("Voltage Abnormality", 0.06),
        ("Abnormal Power Consumption", 0.05),
        ("Potential Electrical Fault", 0.05)
    ]
    conditions, weights = zip(*condition_weights)

    end_time = datetime(2026, 11, 6, 20, 0, 0)
    start_time = end_time - timedelta(days=30)
    total_seconds = int((end_time - start_time).total_seconds())

    rows: List[Dict[str, Any]] = []

    for app_name in APPLIANCE_SPECS.keys():
        assigned_conditions = random.choices(conditions, weights=weights, k=samples_per_appliance)

        # Distribute timestamps evenly with random jitter
        step_seconds = total_seconds / samples_per_appliance

        for i, cond in enumerate(assigned_conditions):
            t_offset = i * step_seconds + random.uniform(-step_seconds * 0.3, step_seconds * 0.3)
            current_time = start_time + timedelta(seconds=max(0, min(total_seconds, t_offset)))

            row = generate_single_reading(app_name, cond, current_time)
            rows.append(row)

    df_raw = pd.DataFrame(rows)
    df_raw = df_raw.sort_values("Timestamp").reset_index(drop=True)

    raw_path = os.path.join(output_dir, "raw_data.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"Generated raw dataset: {len(df_raw)} records -> {raw_path}")

    # Compute full feature engineered dataset
    df_processed = compute_electrical_features(df_raw)
    proc_path = os.path.join(output_dir, "processed_data.csv")
    df_processed.to_csv(proc_path, index=False)
    print(f"Generated processed dataset: {len(df_processed)} records -> {proc_path}")

    return df_raw


if __name__ == "__main__":
    generate_full_dataset()
