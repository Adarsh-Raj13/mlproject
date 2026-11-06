"""
Electrical Engineering Feature Calculations & Appliance Specifications.
Implements single-phase AC circuit theory:
- S = V * I (Apparent Power, VA)
- P = S * PF (Active Power, W)
- Q = sqrt(max(0, S^2 - P^2)) (Reactive Power, VAR)
- |Z| = V / I (Impedance Magnitude, Ohms)
- Normalized deviations from appliance nominal nameplate ratings.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

# Standard Nameplate Specifications for Monitored Appliances
APPLIANCE_SPECS: Dict[str, Dict[str, Any]] = {
    "Ceiling Fan": {
        "category": "Inductive Motor",
        "rated_power": 75.0,
        "rated_voltage": 230.0,
        "rated_current": 0.40,
        "nominal_pf": 0.82,
        "max_safe_temp": 60.0,
        "location": "Living Room / Lab Area"
    },
    "Refrigerator": {
        "category": "Cyclic Inductive",
        "rated_power": 160.0,
        "rated_voltage": 230.0,
        "rated_current": 0.95,
        "nominal_pf": 0.76,
        "max_safe_temp": 65.0,
        "location": "Kitchen / Utility"
    },
    "Air Conditioner": {
        "category": "Heavy Inductive",
        "rated_power": 1800.0,
        "rated_voltage": 230.0,
        "rated_current": 8.50,
        "nominal_pf": 0.92,
        "max_safe_temp": 75.0,
        "location": "Master Suite / Server Room"
    },
    "Washing Machine": {
        "category": "Dynamic Mixed",
        "rated_power": 600.0,
        "rated_voltage": 230.0,
        "rated_current": 3.20,
        "nominal_pf": 0.81,
        "max_safe_temp": 65.0,
        "location": "Laundry Room"
    },
    "Electric Iron": {
        "category": "Resistive Heating",
        "rated_power": 1200.0,
        "rated_voltage": 230.0,
        "rated_current": 5.22,
        "nominal_pf": 0.99,
        "max_safe_temp": 85.0,
        "location": "Utility Wardrobe"
    },
    "Water Heater": {
        "category": "Resistive Immersion",
        "rated_power": 2000.0,
        "rated_voltage": 230.0,
        "rated_current": 8.70,
        "nominal_pf": 0.99,
        "max_safe_temp": 78.0,
        "location": "Bathroom / Boiler Room"
    }
}


def compute_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous AC electrical engineering metrics and deviation features
    from raw telemetry parameters. Handles missing fields gracefully.
    """
    df = df.copy()

    # Ensure numerical datatypes
    v = pd.to_numeric(df.get("Voltage", 230.0), errors="coerce").fillna(230.0)
    i = pd.to_numeric(df.get("Current", 0.0), errors="coerce").fillna(0.0).clip(lower=0.001)

    # Apparent Power S (VA)
    if "Apparent_Power" not in df.columns or df["Apparent_Power"].isna().all():
        s = v * i
        df["Apparent_Power"] = s
    else:
        s = pd.to_numeric(df["Apparent_Power"], errors="coerce").fillna(v * i)

    # Power Factor PF
    if "Power_Factor" not in df.columns or df["Power_Factor"].isna().all():
        if "Active_Power" in df.columns and not df["Active_Power"].isna().all():
            pf = (pd.to_numeric(df["Active_Power"], errors="coerce") / s.replace(0, 0.001)).clip(0.05, 1.0)
        else:
            pf = pd.Series(0.85, index=df.index)
        df["Power_Factor"] = pf
    else:
        pf = pd.to_numeric(df["Power_Factor"], errors="coerce").fillna(0.85).clip(0.05, 1.0)

    # Active Power P (Watts)
    if "Active_Power" not in df.columns or df["Active_Power"].isna().all():
        p = s * pf
        df["Active_Power"] = p
    else:
        p = pd.to_numeric(df["Active_Power"], errors="coerce").fillna(s * pf)

    # Reactive Power Q (VAR)
    if "Reactive_Power" not in df.columns or df["Reactive_Power"].isna().all():
        diff = np.maximum(0.0, s**2 - p**2)
        q = np.sqrt(diff)
        df["Reactive_Power"] = q
    else:
        q = pd.to_numeric(df["Reactive_Power"], errors="coerce").fillna(0.0)

    # Impedance Magnitude |Z| (Ohms)
    df["Impedance"] = (v / i).clip(0.1, 10000.0)

    # Frequency default
    if "Frequency" not in df.columns:
        df["Frequency"] = 50.0
    else:
        df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(50.0)

    # Operating Duration (mins)
    if "Operating_Duration" not in df.columns:
        df["Operating_Duration"] = 60.0
    else:
        df["Operating_Duration"] = pd.to_numeric(df["Operating_Duration"], errors="coerce").fillna(60.0)

    # Energy Consumption (kWh)
    if "Energy_Consumption" not in df.columns:
        df["Energy_Consumption"] = (df["Active_Power"] * df["Operating_Duration"]) / 60000.0
    else:
        df["Energy_Consumption"] = pd.to_numeric(df["Energy_Consumption"], errors="coerce").fillna(
            (df["Active_Power"] * df["Operating_Duration"]) / 60000.0
        )

    # Temperature (°C)
    if "Temperature" not in df.columns:
        df["Temperature"] = 35.0
    else:
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(35.0)

    # Appliance-specific deviations from rated specs
    app_series = df.get("Appliance_Type", pd.Series(["Ceiling Fan"] * len(df), index=df.index))

    rated_power = app_series.map(lambda name: APPLIANCE_SPECS.get(name, {}).get("rated_power", 500.0))
    rated_current = app_series.map(lambda name: APPLIANCE_SPECS.get(name, {}).get("rated_current", 2.5))
    nominal_pf = app_series.map(lambda name: APPLIANCE_SPECS.get(name, {}).get("nominal_pf", 0.85))
    max_safe_temp = app_series.map(lambda name: APPLIANCE_SPECS.get(name, {}).get("max_safe_temp", 65.0))

    df["Power_Ratio"] = (p / rated_power).clip(0.0, 10.0)
    df["Current_Ratio"] = (i / rated_current).clip(0.0, 10.0)
    df["Voltage_Deviation"] = ((v - 230.0).abs() / 230.0).clip(0.0, 1.0)
    df["Temp_Margin"] = df["Temperature"] - max_safe_temp
    df["PF_Deviation"] = (pf - nominal_pf).abs().clip(0.0, 1.0)

    return df

