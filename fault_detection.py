"""
Page 4: Fault Detection Component.
Interactive fault analysis supporting manual parameter input, CSV batch upload,
actual ML inference (Load Profiler, Isolation Forest, Fault Classifier),
confidence output, abnormal parameter attribution, and engineering recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from database.database import DatabaseManager
from src.prediction import InferenceEngine
from src.feature_engineering import APPLIANCE_SPECS


def render_fault_detection_page(db: DatabaseManager, engine: InferenceEngine):
    st.markdown("""
    <div class="iot-header">
        <h1>🔍 INTERACTIVE FAULT DIAGNOSTICS & INFERENCE</h1>
        <p>Live Machine Learning Classification &bull; Anomaly Root-Cause Analysis &bull; Operational Safeguards</p>
    </div>
    """, unsafe_allow_html=True)

    if not engine.is_loaded:
        st.error("ML Models are not currently loaded into memory. Please ensure models are trained.")
        return

    tab_manual, tab_upload, tab_sim = st.tabs([
        "⚙️ Manual Parameter Diagnostics",
        "📂 CSV Telemetry Batch Upload",
        "🧪 Pre-configured Fault Scenarios"
    ])

    # ---------------- TAB 1: MANUAL PARAMETER ENTRY ----------------
    with tab_manual:
        st.subheader("Manual Electrical Telemetry Entry")
        st.caption("Provide live electrical readings to execute immediate ML diagnosis.")

        col_app, col_v, col_i = st.columns(3)
        with col_app:
            app_choice = st.selectbox("Appliance Type", list(APPLIANCE_SPECS.keys()), index=2)
            spec = APPLIANCE_SPECS[app_choice]
            st.caption(f"Rated: {spec['rated_power']}W | {spec['rated_current']}A | Max Temp: {spec['max_safe_temp']}°C")

        with col_v:
            voltage_input = st.number_input("Grid Voltage (V)", min_value=120.0, max_value=320.0, value=230.0, step=1.0)
        with col_i:
            current_input = st.number_input("Drawn Current (A)", min_value=0.01, max_value=40.0, value=float(spec["rated_current"]), step=0.1)

        col_p, col_pf, col_t, col_dur = st.columns(4)
        with col_p:
            # S = V * I
            default_p = min(4000.0, voltage_input * current_input * spec["nominal_pf"])
            power_input = st.number_input("Active Power (W)", min_value=1.0, max_value=6000.0, value=float(round(default_p, 1)), step=10.0)
        with col_pf:
            calculated_pf = min(1.0, max(0.1, power_input / max(1e-3, (voltage_input * current_input))))
            pf_input = st.slider("Power Factor (cos φ)", min_value=0.20, max_value=1.0, value=float(round(calculated_pf, 2)), step=0.01)
        with col_t:
            temp_input = st.number_input("Measured Temperature (°C)", min_value=15.0, max_value=140.0, value=45.0, step=1.0)
        with col_dur:
            duration_input = st.number_input("Operating Duration (mins)", min_value=1.0, max_value=1440.0, value=60.0, step=15.0)

        if st.button("🚀 Run Live ML Fault Diagnostics", key="btn_run_manual", type="primary"):
            reading_dict = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Appliance_Type": app_choice,
                "Voltage": voltage_input,
                "Current": current_input,
                "Active_Power": power_input,
                "Power_Factor": pf_input,
                "Temperature": temp_input,
                "Frequency": 50.0,
                "Operating_Duration": duration_input
            }

            with st.spinner("Executing ML Inference Pipeline..."):
                res = engine.predict_single(reading_dict)

            # Record in DB
            db_conn = db.get_appliance_by_name(app_choice)
            app_id = db_conn["id"] if db_conn else 1
            reading_id = db.insert_reading(
                appliance_id=app_id,
                timestamp=reading_dict["Timestamp"],
                voltage=voltage_input,
                current=current_input,
                active_power=power_input,
                reactive_power=res["reactive_power"],
                apparent_power=res["apparent_power"],
                power_factor=pf_input,
                frequency=50.0,
                temperature=temp_input,
                operating_duration=duration_input,
                energy_kwh=(power_input * duration_input) / 60000.0
            )

            db.insert_prediction(
                reading_id=reading_id,
                appliance_name=app_choice,
                timestamp=reading_dict["Timestamp"],
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
                    appliance_name=app_choice,
                    timestamp=reading_dict["Timestamp"],
                    severity=res["severity"],
                    fault_type=res["predicted_fault"],
                    trigger_parameter=res["abnormal_parameters"][0] if res["abnormal_parameters"] else "Power",
                    trigger_value=current_input if "Current" in str(res["abnormal_parameters"]) else power_input,
                    threshold_value=spec["rated_power"],
                    suggested_action=res["corrective_action"]
                )

            # Display Diagnostics Dashboard Results
            st.markdown("---")
            st.markdown("### 📋 Machine Learning Diagnostic Assessment")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                badge_class = f"badge-{res['severity_color'].lower()}"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Condition Classification</div>
                    <div style="margin-top:6px;"><span class="badge-pill {badge_class}" style="font-size:16px;">{res['predicted_fault']}</span></div>
                    <div class="kpi-subtext" style="margin-top:8px;">Severity: <b>{res['severity']}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with d2:
                conf_pct = res["confidence"] * 100
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Classifier Confidence</div>
                    <div class="kpi-value">{conf_pct:.1f}%</div>
                    <div class="kpi-subtext">Random Forest / Ensembled</div>
                </div>
                """, unsafe_allow_html=True)

            with d3:
                anom_text = "Anomaly Detected" if res["is_anomaly"] else "Normal Signature"
                anom_color = "#ef4444" if res["is_anomaly"] else "#10b981"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Isolation Forest Score</div>
                    <div class="kpi-value" style="color: {anom_color};">{res['anomaly_score']:.3f}</div>
                    <div class="kpi-subtext">{anom_text}</div>
                </div>
                """, unsafe_allow_html=True)

            with d4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Load Profile Regime</div>
                    <div class="kpi-value" style="font-size:20px;">{res['load_profile_label']}</div>
                    <div class="kpi-subtext">Category: {res['consumption_category']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Root cause abnormal parameters & recommendations
            col_params, col_action = st.columns(2)
            with col_params:
                st.markdown("#### ⚠️ Identified Parameter Deviations")
                if res["abnormal_parameters"]:
                    for p_dev in res["abnormal_parameters"]:
                        st.warning(f"🚩 {p_dev}")
                else:
                    st.success("✅ All monitored electrical parameters reside within allowable design tolerances.")

            with col_action:
                st.markdown("#### 🛠️ Recommended Corrective Action")
                st.info(f"👉 **Action Plan:** {res['corrective_action']}")

    # ---------------- TAB 2: CSV BATCH UPLOAD ----------------
    with tab_upload:
        st.subheader("Batch Telemetry File Upload")
        st.caption("Upload a CSV file containing multiple electrical measurement rows to run full vector ML inference.")

        sample_csv = "dataset/raw_data.csv"
        if os.path.exists(sample_csv):
            with open(sample_csv, "rb") as f:
                st.download_button(
                    "📥 Download Sample CSV Template",
                    data=f,
                    file_name="sample_electrical_telemetry.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(df_upload)}** rows. Preview:")
                st.dataframe(df_upload.head(5), use_container_width=True)

                if st.button("⚡ Process CSV with ML Models", key="btn_run_csv", type="primary"):
                    with st.spinner("Processing batch telemetry through ML models..."):
                        results_df = engine.batch_predict(df_upload)

                    st.success(f"Successfully processed {len(results_df)} telemetry entries!")

                    # Summary Metrics of Batch
                    anom_count = results_df["Is_Anomaly"].sum()
                    fault_summary = results_df["Predicted_Fault"].value_counts()

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total Rows Evaluated", len(results_df))
                    with c2:
                        st.metric("Total Anomalies Detected", anom_count, delta=f"{(anom_count/len(results_df))*100:.1f}%")
                    with c3:
                        st.metric("Predominant State", fault_summary.index[0], f"{fault_summary.iloc[0]} occurrences")

                    st.markdown("#### 📊 Diagnostic Results Table")
                    display_cols = ["Appliance_Type", "Voltage", "Current", "Active_Power", "Temperature",
                                    "Load_Profile", "Predicted_Fault", "Model_Confidence", "Alert_Severity", "Corrective_Action"]
                    valid_cols = [c for c in display_cols if c in results_df.columns]
                    st.dataframe(results_df[valid_cols], use_container_width=True)

                    # Download processed results
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Evaluated Diagnostic Report (CSV)",
                        data=csv_export,
                        file_name=f"fault_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

    # ---------------- TAB 3: PRE-CONFIGURED SCENARIOS ----------------
    with tab_sim:
        st.subheader("Simulate Authentic Field Scenarios")
        st.caption("Test the system against pre-configured electrical fault and normal scenarios.")

        scenarios = {
            "Air Conditioner - Compressor Locked Rotor (Overcurrent)": {
                "Appliance_Type": "Air Conditioner", "Voltage": 222.0, "Current": 24.5,
                "Active_Power": 3950.0, "Power_Factor": 0.72, "Temperature": 92.0, "Operating_Duration": 35.0
            },
            "Water Heater - Dry Heating / Thermostat Failure (Overheating)": {
                "Appliance_Type": "Water Heater", "Voltage": 230.0, "Current": 9.1,
                "Active_Power": 2080.0, "Power_Factor": 0.99, "Temperature": 98.0, "Operating_Duration": 90.0
            },
            "Ceiling Fan - Bearing Seizure (Mechanical Overload)": {
                "Appliance_Type": "Ceiling Fan", "Voltage": 228.0, "Current": 0.72,
                "Active_Power": 145.0, "Power_Factor": 0.78, "Temperature": 68.0, "Operating_Duration": 120.0
            },
            "Refrigerator - Severe Grid Voltage Sag (Voltage Abnormality)": {
                "Appliance_Type": "Refrigerator", "Voltage": 178.0, "Current": 1.65,
                "Active_Power": 220.0, "Power_Factor": 0.75, "Temperature": 49.0, "Operating_Duration": 180.0
            },
            "Electric Iron - Normal Thermostat Cycling": {
                "Appliance_Type": "Electric Iron", "Voltage": 231.0, "Current": 5.20,
                "Active_Power": 1200.0, "Power_Factor": 0.99, "Temperature": 55.0, "Operating_Duration": 30.0
            }
        }

        selected_scenario_name = st.selectbox("Choose a Scenario to Evaluate:", list(scenarios.keys()))
        selected_data = scenarios[selected_scenario_name]

        st.json(selected_data)

        if st.button("⚡ Test This Scenario", type="primary"):
            sim_dict = dict(selected_data)
            sim_dict["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sim_dict["Frequency"] = 50.0

            res = engine.predict_single(sim_dict)

            badge_class = f"badge-{res['severity_color'].lower()}"
            st.markdown(f"""
            <div class="telemetry-card" style="margin-top: 15px;">
                <div class="telemetry-card-title">
                    <span>Diagnostic Result: <b>{res['predicted_fault']}</b></span>
                    <span class="badge-pill {badge_class}">{res['severity']}</span>
                </div>
                <div style="font-size: 14px; margin-bottom: 8px;">
                    <b>Model Confidence:</b> {res['confidence']*100:.1f}% &bull; 
                    <b>Anomaly Score:</b> {res['anomaly_score']:.3f} &bull; 
                    <b>Profile:</b> {res['load_profile_label']}
                </div>
                <div style="color: #ea580c; font-weight:600; margin-bottom: 8px;">
                    Trigger Deviations: {', '.join(res['abnormal_parameters']) if res['abnormal_parameters'] else 'None (Nominal)'}
                </div>
                <div style="background: #f1f5f9; padding: 10px; border-radius: 6px;">
                    🛠️ <b>Action:</b> {res['corrective_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

