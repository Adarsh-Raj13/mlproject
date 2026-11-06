"""
Main Application Entry Point for Smart Load Profiling & Early Fault Alert System.
Title: SMART LOAD MONITOR
Subtitle: Intelligent Electrical Monitoring and Early Fault Detection System
"""

import streamlit as st
import os
import random
from datetime import datetime

# Initialize Streamlit Page Configuration
st.set_page_config(
    page_title="SMART LOAD MONITOR - Industrial IoT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database.database import DatabaseManager
from src.prediction import InferenceEngine
from src.feature_engineering import APPLIANCE_SPECS
from components.dashboard import render_dashboard_page
from components.appliance_monitoring import render_appliance_monitoring_page
from components.load_analytics import render_load_analytics_page
from components.fault_detection import render_fault_detection_page
from components.alert_centre import render_alert_centre_page
from components.analytics_reports import render_analytics_reports_page

# Load Custom CSS Styling
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_database() -> DatabaseManager:
    """Initializes and caches the SQLite database connection manager."""
    return DatabaseManager()


@st.cache_resource
def get_inference_engine() -> InferenceEngine:
    """Initializes and caches the trained ML model inference engine."""
    return InferenceEngine()


def inject_simulation_reading(db: DatabaseManager, engine: InferenceEngine):
    """Simulates an incoming IoT sensor measurement packet and runs live ML inference."""
    app_name = random.choice(list(APPLIANCE_SPECS.keys()))
    spec = APPLIANCE_SPECS[app_name]
    app_record = db.get_appliance_by_name(app_name)
    app_id = app_record["id"] if app_record else 1

    # Decide if this simulated packet is normal or abnormal (80% normal, 20% abnormal)
    is_faulty = random.random() < 0.25

    if is_faulty:
        # Choose a simulated fault condition
        fault_type = random.choice(["Overload", "Overheating", "Voltage Abnormality", "Overcurrent"])
        if fault_type == "Overload":
            voltage = round(random.gauss(226.0, 3.0), 1)
            current = round(spec["rated_current"] * random.uniform(1.35, 1.65), 2)
            temp = round(spec["max_safe_temp"] + random.uniform(4.0, 12.0), 1)
            pf = round(max(0.55, spec["nominal_pf"] * random.uniform(0.85, 0.95)), 2)
        elif fault_type == "Overheating":
            voltage = round(random.gauss(230.0, 2.0), 1)
            current = round(spec["rated_current"] * random.uniform(1.05, 1.25), 2)
            temp = round(spec["max_safe_temp"] + random.uniform(16.0, 30.0), 1)
            pf = round(spec["nominal_pf"], 2)
        elif fault_type == "Voltage Abnormality":
            voltage = round(random.choice([random.uniform(170.0, 192.0), random.uniform(255.0, 272.0)]), 1)
            current = round(spec["rated_current"] * random.uniform(1.15, 1.40), 2)
            temp = round(spec["max_safe_temp"] - 5.0, 1)
            pf = round(spec["nominal_pf"] * 0.9, 2)
        else:  # Overcurrent
            voltage = round(random.gauss(224.0, 4.0), 1)
            current = round(spec["rated_current"] * random.uniform(2.1, 2.8), 2)
            temp = round(spec["max_safe_temp"] + 15.0, 1)
            pf = round(spec["nominal_pf"] * 0.8, 2)
    else:
        voltage = round(random.gauss(230.0, 2.0), 1)
        current = round(max(0.05, random.gauss(spec["rated_current"] * 0.85, 0.05)), 2)
        temp = round(max(25.0, random.gauss(spec["max_safe_temp"] * 0.65, 2.0)), 1)
        pf = round(min(1.0, max(0.65, random.gauss(spec["nominal_pf"], 0.02))), 2)

    apparent_power = round(voltage * current, 2)
    active_power = round(apparent_power * pf, 2)
    diff = max(0.0, apparent_power**2 - active_power**2)
    reactive_power = round(diff**0.5, 2)
    duration = round(random.uniform(15.0, 240.0), 1)
    energy_kwh = round((active_power * duration) / 60000.0, 4)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert raw reading
    reading_id = db.insert_reading(
        appliance_id=app_id,
        timestamp=ts,
        voltage=voltage,
        current=current,
        active_power=active_power,
        reactive_power=reactive_power,
        apparent_power=apparent_power,
        power_factor=pf,
        frequency=50.0,
        temperature=temp,
        operating_duration=duration,
        energy_kwh=energy_kwh
    )

    # ML Inference
    if engine.is_loaded:
        reading_dict = {
            "Timestamp": ts,
            "Appliance_Type": app_name,
            "Voltage": voltage,
            "Current": current,
            "Active_Power": active_power,
            "Power_Factor": pf,
            "Temperature": temp,
            "Frequency": 50.0,
            "Operating_Duration": duration
        }
        res = engine.predict_single(reading_dict)

        db.insert_prediction(
            reading_id=reading_id,
            appliance_name=app_name,
            timestamp=ts,
            load_profile=res["load_profile_label"],
            is_anomaly=res["is_anomaly"],
            anomaly_score=res["anomaly_score"],
            predicted_fault=res["predicted_fault"],
            confidence=res["confidence"],
            abnormal_parameters=", ".join(res["abnormal_parameters"]),
            corrective_action=res["corrective_action"]
        )

        if res["severity"] in ["Warning", "High Risk", "Critical"]:
            db.insert_alert(
                appliance_name=app_name,
                timestamp=ts,
                severity=res["severity"],
                fault_type=res["predicted_fault"],
                trigger_parameter=res["abnormal_parameters"][0] if res["abnormal_parameters"] else "Current/Power",
                trigger_value=current,
                threshold_value=spec["rated_current"],
                suggested_action=res["corrective_action"]
            )


def main():
    db = get_database()
    engine = get_inference_engine()

    # Sidebar Navigation & IoT Controls
    with st.sidebar:
        st.markdown("""
        <div style="padding: 12px 0 18px 0; border-bottom: 1px solid #334155;">
            <div style="font-size: 20px; font-weight: 800; color: #38bdf8; letter-spacing: -0.02em;">
                ⚡ SMART LOAD MONITOR
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                Intelligent Electrical Monitoring & Early Fault Detection
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        menu_choice = st.radio(
            "Navigation Menu",
            [
                "🏠 Home Dashboard",
                "🔌 Appliance Monitoring",
                "⚡ Load Profiling",
                "🔍 Fault Detection",
                "🚨 Early Alert Centre",
                "📊 Analytics and Reports"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📡 Demonstration Mode")
        st.caption("Inject physically-calibrated telemetry into SQLite database to observe live ML reaction:")

        if st.button("⚡ Inject Telemetry Reading", use_container_width=True, type="secondary"):
            inject_simulation_reading(db, engine)
            st.toast("New telemetry packet recorded and processed!", icon="⚡")
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 12px; color: #94a3b8;">
            <b>System Status:</b> <span style="color:#10b981;">● Online (Ready)</span><br>
            <b>Grid Spec:</b> 230V RMS &bull; 50 Hz<br>
            <b>Monitored Assets:</b> 6 Appliances<br>
            <b>ML Models:</b> Active (Joblib)
        </div>
        """, unsafe_allow_html=True)

    # Main Route Execution
    if menu_choice == "🏠 Home Dashboard":
        render_dashboard_page(db)
    elif menu_choice == "🔌 Appliance Monitoring":
        render_appliance_monitoring_page(db)
    elif menu_choice == "⚡ Load Profiling":
        render_load_analytics_page(db, engine)
    elif menu_choice == "🔍 Fault Detection":
        render_fault_detection_page(db, engine)
    elif menu_choice == "🚨 Early Alert Centre":
        render_alert_centre_page(db)
    elif menu_choice == "📊 Analytics and Reports":
        render_analytics_reports_page(db)


if __name__ == "__main__":
    main()

