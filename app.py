"""
Smart Load Monitor - Streamlined Web Application
Upload CSV data â†’ Get instant ML fault predictions.
3 pages: Home Dashboard | Predict & Upload | Model Info
"""

import streamlit as st
import os
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Smart Load Monitor",
    page_icon="âš¡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# â”€â”€ Inject minimal clean CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
[data-testid="stSidebar"]          { background: #1e293b; }
[data-testid="stHeader"]           { background: transparent; }
section[data-testid="stMain"] > div { padding-top: 1rem; }

/* Top nav tabs */
div[data-testid="stTabs"] button {
    font-weight: 600; font-size: 14px; color: #94a3b8;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8; border-bottom: 2px solid #38bdf8;
}

/* KPI cards */
.kpi { background:#1e293b; border:1px solid #334155; border-radius:12px;
       padding:16px 20px; text-align:center; }
.kpi .label { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
              color:#64748b; margin-bottom:6px; }
.kpi .val   { font-size:30px; font-weight:800; line-height:1.1; }
.kpi .sub   { font-size:12px; color:#94a3b8; margin-top:4px; }

/* Severity badges */
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:12px; font-weight:700; }
.badge-green  { background:#064e3b; color:#34d399; }
.badge-yellow { background:#78350f; color:#fbbf24; }
.badge-orange { background:#7c2d12; color:#fb923c; }
.badge-red    { background:#7f1d1d; color:#f87171; }

/* Upload area */
[data-testid="stFileUploader"] { border:2px dashed #334155; border-radius:12px;
                                  padding:12px; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

/* Result card */
.result-card { background:#1e293b; border-radius:12px; border:1px solid #334155;
               padding:20px; margin-top:16px; }

/* Info box */
.info-box { background:#172554; border-left:4px solid #3b82f6;
            padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Load models & DB (cached) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_resource(show_spinner="Loading ML modelsâ€¦")
def load_engine():
    from src.prediction import InferenceEngine
    return InferenceEngine()

@st.cache_resource(show_spinner="Connecting to databaseâ€¦")
def load_db():
    from database.database import DatabaseManager
    return DatabaseManager()


def severity_badge(sev: str) -> str:
    color_map = {"Normal": "green", "Warning": "yellow",
                 "High Risk": "orange", "Critical": "red"}
    c = color_map.get(sev, "green")
    return f'<span class="badge badge-{c}">{sev}</span>'


# â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding:8px 0 20px 0;
            border-bottom:1px solid #334155;margin-bottom:24px;">
  <span style="font-size:28px;">âš¡</span>
  <div>
    <div style="font-size:22px;font-weight:800;color:#38bdf8;letter-spacing:-.02em;">
      SMART LOAD MONITOR
    </div>
    <div style="font-size:12px;color:#64748b;">
      Electrical Fault Detection &amp; Load Profiling â€” ML-Powered
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# â”€â”€ Top Navigation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab_home, tab_predict, tab_info = st.tabs([
    "ðŸ   Dashboard",
    "ðŸ”  Predict / Upload",
    "ðŸ“Š  Model Info"
])


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 â€” DASHBOARD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_home:
    db = load_db()
    engine = load_engine()

    stats = db.get_system_summary_stats()
    recent_alerts = db.get_recent_alerts(limit=10)
    df_hist = db.get_readings_history(limit=300)

    # â”€â”€ KPI Row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    k1, k2, k3, k4 = st.columns(4)
    health = stats.get("overall_health_score", 100)
    health_color = "#10b981" if health >= 80 else "#f59e0b" if health >= 60 else "#ef4444"

    with k1:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">System Health</div>
          <div class="val" style="color:{health_color};">{health:.0f}%</div>
          <div class="sub">Overall Status</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">Total Readings</div>
          <div class="val" style="color:#38bdf8;">{stats.get('total_readings', 0)}</div>
          <div class="sub">In database</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">Critical Alerts</div>
          <div class="val" style="color:#ef4444;">{stats.get('critical_alerts', 0)}</div>
          <div class="sub">Needs attention</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">Warning Alerts</div>
          <div class="val" style="color:#f59e0b;">{stats.get('warning_alerts', 0)}</div>
          <div class="sub">Monitor closely</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

    # â”€â”€ Charts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col_chart, col_alerts = st.columns([2, 1])

    with col_chart:
        st.markdown("#### ðŸ“ˆ Recent Power Trend")
        if not df_hist.empty and "active_power" in df_hist.columns and "timestamp" in df_hist.columns:
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            fig = px.line(
                df_hist.sort_values("timestamp").tail(100),
                x="timestamp", y="active_power",
                color="appliance_name" if "appliance_name" in df_hist.columns else None,
                template="plotly_dark",
                labels={"active_power": "Power (W)", "timestamp": "Time"}
            )
            fig.update_layout(
                height=280, paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No telemetry data yet. Inject a reading below or use the Predict tab.")

    with col_alerts:
        st.markdown("#### ðŸš¨ Recent Alerts")
        if recent_alerts:
            for a in recent_alerts[:6]:
                sev = a.get("severity", "Normal")
                badge = severity_badge(sev)
                name = a.get("appliance_name", "?")
                fault = a.get("fault_type", "?")
                st.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid #1e293b;font-size:13px;'>"
                    f"{badge} &nbsp;<b>{name}</b><br>"
                    f"<span style='color:#64748b;font-size:11px;'>{fault}</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div style='color:#64748b;font-size:13px;'>No alerts yet.</div>",
                        unsafe_allow_html=True)

    # â”€â”€ Inject Simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown("---")
    st.markdown("#### âš¡ Demo: Inject Simulated Telemetry")
    c_btn, c_msg = st.columns([1, 3])
    with c_btn:
        if st.button("Inject Reading", type="primary", width='stretch'):
            from src.feature_engineering import APPLIANCE_SPECS
            app_name = random.choice(list(APPLIANCE_SPECS.keys()))
            spec = APPLIANCE_SPECS[app_name]
            is_faulty = random.random() < 0.3
            if is_faulty:
                current = round(spec["rated_current"] * random.uniform(1.4, 2.5), 2)
                temp    = round(spec["max_safe_temp"] + random.uniform(5, 25), 1)
                voltage = round(random.choice([random.uniform(170, 195), random.uniform(255, 270)]), 1)
            else:
                current = round(max(0.05, random.gauss(spec["rated_current"] * 0.85, 0.05)), 2)
                temp    = round(max(25.0, random.gauss(spec["max_safe_temp"] * 0.65, 2.0)), 1)
                voltage = round(random.gauss(230.0, 2.0), 1)
            pf             = round(min(1.0, max(0.5, random.gauss(spec["nominal_pf"], 0.03))), 2)
            apparent_power = round(voltage * current, 2)
            active_power   = round(apparent_power * pf, 2)
            diff           = max(0.0, apparent_power**2 - active_power**2)
            reactive_power = round(diff**0.5, 2)
            duration       = round(random.uniform(15, 180), 1)
            energy_kwh     = round((active_power * duration) / 60000.0, 4)
            ts             = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            app_record = db.get_appliance_by_name(app_name)
            app_id     = app_record["id"] if app_record else 1

            reading_id = db.insert_reading(
                appliance_id=app_id, timestamp=ts,
                voltage=voltage, current=current,
                active_power=active_power, reactive_power=reactive_power,
                apparent_power=apparent_power, power_factor=pf,
                frequency=50.0, temperature=temp,
                operating_duration=duration, energy_kwh=energy_kwh
            )

            if engine.is_loaded:
                res = engine.predict_single({
                    "Timestamp": ts, "Appliance_Type": app_name,
                    "Voltage": voltage, "Current": current,
                    "Active_Power": active_power, "Power_Factor": pf,
                    "Temperature": temp, "Frequency": 50.0,
                    "Operating_Duration": duration
                })
                db.insert_prediction(
                    reading_id=reading_id, appliance_name=app_name, timestamp=ts,
                    load_profile=res["load_profile_label"], is_anomaly=res["is_anomaly"],
                    anomaly_score=res["anomaly_score"], predicted_fault=res["predicted_fault"],
                    confidence=res["confidence"],
                    abnormal_parameters=", ".join(res["abnormal_parameters"]),
                    corrective_action=res["corrective_action"]
                )
                if res["severity"] in ["Warning", "High Risk", "Critical"]:
                    db.insert_alert(
                        appliance_name=app_name, timestamp=ts, severity=res["severity"],
                        fault_type=res["predicted_fault"],
                        trigger_parameter=res["abnormal_parameters"][0] if res["abnormal_parameters"] else "Current",
                        trigger_value=current, threshold_value=spec["rated_current"],
                        suggested_action=res["corrective_action"]
                    )
                st.toast(f"âœ… {app_name} â€” {res['predicted_fault']} ({res['severity']})", icon="âš¡")
            st.rerun()

    with c_msg:
        st.caption("Injects one simulated sensor packet (80% normal, 20% fault) and runs ML inference instantly.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 â€” PREDICT / UPLOAD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_predict:
    engine = load_engine()
    db     = load_db()

    if not engine.is_loaded:
        st.error("âš ï¸ ML models are not loaded. Run `python train_models.py` first, then restart the app.")
        st.stop()

    from src.feature_engineering import APPLIANCE_SPECS

    mode = st.radio(
        "Choose Input Mode",
        ["ðŸ“‚ Upload CSV File", "âš™ï¸ Enter Values Manually"],
        horizontal=True
    )
    st.markdown("---")

    # â”€â”€ MODE A: CSV Upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if mode == "ðŸ“‚ Upload CSV File":
        st.markdown("#### Upload your telemetry CSV")

        col_up, col_tmpl = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Drop your CSV here (must contain: Appliance_Type, Voltage, Current, "
                "Active_Power, Power_Factor, Temperature, Operating_Duration)",
                type=["csv"], label_visibility="collapsed"
            )
        with col_tmpl:
            sample_path = os.path.join(os.path.dirname(__file__), "dataset", "raw_data.csv")
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    st.download_button(
                        "ðŸ“¥ Download Sample Template",
                        data=f, file_name="sample_telemetry.csv",
                        mime="text/csv", width='stretch'
                    )
            st.caption("Not sure of the format? Download the sample CSV above.")

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                st.markdown(f"**Loaded {len(df_in):,} rows** â€” Preview:")
                st.dataframe(df_in.head(5), width='stretch', height=200)

                if st.button("âš¡ Run ML Analysis", type="primary", width='stretch'):
                    with st.spinner(f"Running inference on {len(df_in):,} rowsâ€¦"):
                        df_out = engine.batch_predict(df_in)

                    # â”€â”€ Summary Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    anom_count  = int(df_out["Is_Anomaly"].sum())
                    fault_counts = df_out["Predicted_Fault"].value_counts()
                    crit_count  = int((df_out["Alert_Severity"] == "Critical").sum())
                    warn_count  = int((df_out["Alert_Severity"].isin(["Warning", "High Risk"])).sum())

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""<div class="kpi">
                          <div class="label">Rows Analysed</div>
                          <div class="val" style="color:#38bdf8;">{len(df_out):,}</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="kpi">
                          <div class="label">Anomalies Found</div>
                          <div class="val" style="color:#f59e0b;">{anom_count}</div>
                          <div class="sub">{anom_count/len(df_out)*100:.1f}% of data</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="kpi">
                          <div class="label">Critical Faults</div>
                          <div class="val" style="color:#ef4444;">{crit_count}</div>
                        </div>""", unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""<div class="kpi">
                          <div class="label">Warnings</div>
                          <div class="val" style="color:#fbbf24;">{warn_count}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

                    # â”€â”€ Charts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        fig_fault = px.pie(
                            values=fault_counts.values, names=fault_counts.index,
                            title="Fault Distribution", template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_fault.update_layout(height=280, paper_bgcolor="#1e293b",
                                                margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_fault, width='stretch')

                    with cc2:
                        sev_counts = df_out["Alert_Severity"].value_counts()
                        color_map  = {"Normal":"#10b981","Warning":"#f59e0b",
                                      "High Risk":"#f97316","Critical":"#ef4444"}
                        fig_sev = px.bar(
                            x=sev_counts.index, y=sev_counts.values,
                            title="Severity Breakdown", template="plotly_dark",
                            color=sev_counts.index,
                            color_discrete_map=color_map,
                            labels={"x": "Severity", "y": "Count"}
                        )
                        fig_sev.update_layout(height=280, paper_bgcolor="#1e293b",
                                              plot_bgcolor="#0f172a", showlegend=False,
                                              margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_sev, width='stretch')

                    # â”€â”€ Results Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    st.markdown("#### ðŸ“‹ Detailed Results")
                    show_cols = [c for c in [
                        "Appliance_Type", "Voltage", "Current", "Active_Power",
                        "Temperature", "Predicted_Fault", "Model_Confidence",
                        "Alert_Severity", "Is_Anomaly", "Corrective_Action"
                    ] if c in df_out.columns]

                    # Colour-code the severity column
                    def color_sev(val):
                        m = {"Critical":"background-color:#7f1d1d;color:#fca5a5",
                             "High Risk":"background-color:#7c2d12;color:#fdba74",
                             "Warning":"background-color:#78350f;color:#fde68a",
                             "Normal":"background-color:#064e3b;color:#6ee7b7"}
                        return m.get(val, "")

                    styled = df_out[show_cols].style.applymap(
                        color_sev, subset=["Alert_Severity"]
                    )
                    st.dataframe(styled, width='stretch', height=380)

                    # â”€â”€ Download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "ðŸ“¥ Download Full Results CSV",
                        data=csv_bytes,
                        file_name=f"ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", width='stretch', type="primary"
                    )

            except Exception as e:
                st.error(f"âŒ Error: {e}")
                st.caption("Make sure your CSV has the required columns. Download the sample template to check.")

    # â”€â”€ MODE B: Manual Entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    else:
        st.markdown("#### Enter Electrical Readings Manually")

        col_app, col_v, col_i = st.columns(3)
        with col_app:
            app_choice = st.selectbox("Appliance", list(APPLIANCE_SPECS.keys()))
            spec = APPLIANCE_SPECS[app_choice]
            st.caption(f"Rated: **{spec['rated_power']}W** | **{spec['rated_current']}A** | Max temp: **{spec['max_safe_temp']}Â°C**")
        with col_v:
            voltage = st.number_input("Voltage (V)", 120.0, 320.0, 230.0, 1.0)
        with col_i:
            current = st.number_input("Current (A)", 0.01, 40.0, float(spec["rated_current"]), 0.1)

        col_p, col_pf, col_t, col_dur = st.columns(4)
        with col_p:
            default_p = round(min(5000.0, voltage * current * spec["nominal_pf"]), 1)
            power = st.number_input("Active Power (W)", 1.0, 6000.0, default_p, 10.0)
        with col_pf:
            calc_pf = round(min(1.0, max(0.1, power / max(1e-3, voltage * current))), 2)
            pf = st.slider("Power Factor", 0.20, 1.0, calc_pf, 0.01)
        with col_t:
            temp = st.number_input("Temperature (Â°C)", 15.0, 140.0, 45.0, 1.0)
        with col_dur:
            duration = st.number_input("Duration (mins)", 1.0, 1440.0, 60.0, 15.0)

        if st.button("ðŸš€ Run Diagnosis", type="primary", width='stretch'):
            reading = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Appliance_Type": app_choice,
                "Voltage": voltage, "Current": current,
                "Active_Power": power, "Power_Factor": pf,
                "Temperature": temp, "Frequency": 50.0,
                "Operating_Duration": duration
            }
            with st.spinner("Running ML inferenceâ€¦"):
                res = engine.predict_single(reading)

            # Save to DB
            app_rec = db.get_appliance_by_name(app_choice)
            app_id  = app_rec["id"] if app_rec else 1
            apparent = round(voltage * current, 2)
            diff     = max(0.0, apparent**2 - (apparent * pf)**2)
            reactive = round(diff**0.5, 2)
            rid = db.insert_reading(
                appliance_id=app_id, timestamp=reading["Timestamp"],
                voltage=voltage, current=current, active_power=power,
                reactive_power=reactive, apparent_power=apparent,
                power_factor=pf, frequency=50.0, temperature=temp,
                operating_duration=duration, energy_kwh=(power * duration) / 60000.0
            )
            db.insert_prediction(
                reading_id=rid, appliance_name=app_choice, timestamp=reading["Timestamp"],
                load_profile=res["load_profile_label"], is_anomaly=res["is_anomaly"],
                anomaly_score=res["anomaly_score"], predicted_fault=res["predicted_fault"],
                confidence=res["confidence"],
                abnormal_parameters=", ".join(res["abnormal_parameters"]),
                corrective_action=res["corrective_action"]
            )

            # â”€â”€ Result Display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            sev_colors = {"Normal":"#10b981","Warning":"#f59e0b",
                          "High Risk":"#f97316","Critical":"#ef4444"}
            sev_color  = sev_colors.get(res["severity"], "#10b981")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"""<div class="kpi">
                  <div class="label">Fault Detected</div>
                  <div class="val" style="font-size:18px;color:{sev_color};">{res['predicted_fault']}</div>
                  <div class="sub">Severity: <b>{res['severity']}</b></div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="kpi">
                  <div class="label">Confidence</div>
                  <div class="val" style="color:#38bdf8;">{res['confidence']*100:.1f}%</div>
                  <div class="sub">Model certainty</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                anom_col = "#ef4444" if res["is_anomaly"] else "#10b981"
                anom_txt = "ANOMALY" if res["is_anomaly"] else "NORMAL"
                st.markdown(f"""<div class="kpi">
                  <div class="label">Anomaly Score</div>
                  <div class="val" style="color:{anom_col};">{res['anomaly_score']:.3f}</div>
                  <div class="sub">{anom_txt}</div>
                </div>""", unsafe_allow_html=True)
            with r4:
                st.markdown(f"""<div class="kpi">
                  <div class="label">Load Profile</div>
                  <div class="val" style="font-size:17px;color:#a78bfa;">{res['load_profile_label']}</div>
                  <div class="sub">{res['consumption_category']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

            col_ab, col_act = st.columns(2)
            with col_ab:
                st.markdown("**âš ï¸ Parameter Violations**")
                if res["abnormal_parameters"]:
                    for p in res["abnormal_parameters"]:
                        st.warning(p)
                else:
                    st.success("âœ… All parameters within safe limits")
            with col_act:
                st.markdown("**ðŸ› ï¸ Recommended Action**")
                st.info(f"ðŸ‘‰ {res['corrective_action']}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 â€” MODEL INFO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_info:
    st.markdown("#### ðŸ§  ML Models Overview")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model_files = {
        "preprocessing_pipeline.pkl": ("ðŸ”§ Preprocessing Pipeline", "Scaler + categorical encoder"),
        "load_profiling_model.pkl":   ("ðŸ“Š Load Profiler",          "K-Means (k=4) + Random Forest classifier"),
        "anomaly_detection_model.pkl":("ðŸ”Ž Anomaly Detector",       "Isolation Forest + Local Outlier Factor"),
        "fault_classification_model.pkl": ("âš ï¸ Fault Classifier",   "Random Forest / XGBoost (7 fault classes)"),
    }

    for fname, (title, desc) in model_files.items():
        fpath = os.path.join(models_dir, fname)
        exists = os.path.exists(fpath)
        size   = f"{os.path.getsize(fpath)/1024:.1f} KB" if exists else "â€”"
        status_col = "#10b981" if exists else "#ef4444"
        status_txt = "âœ… Loaded" if exists else "âŒ Missing"
        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                    padding:14px 18px;margin-bottom:10px;display:flex;
                    justify-content:space-between;align-items:center;">
          <div>
            <div style="font-weight:700;font-size:15px;">{title}</div>
            <div style="color:#64748b;font-size:12px;">{desc}</div>
          </div>
          <div style="text-align:right;">
            <div style="color:{status_col};font-weight:700;">{status_txt}</div>
            <div style="color:#475569;font-size:11px;">{size}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ðŸ” Retrain All Models")
    st.caption("This will regenerate the dataset, retrain all ML models, and save new `.pkl` files. Takes ~1â€“2 minutes.")

    if st.button("ðŸ” Retrain Models Now", type="secondary"):
        with st.spinner("Training models â€” please waitâ€¦"):
            import subprocess, sys
            train_script = os.path.join(os.path.dirname(__file__), "train_models.py")
            result = subprocess.run(
                [sys.executable, train_script],
                capture_output=True, text=True,
                cwd=os.path.dirname(__file__)
            )
        if result.returncode == 0:
            st.success("âœ… Models retrained successfully! Refresh the page to reload.")
            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        else:
            st.error("âŒ Training failed. See error below:")
            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    st.markdown("---")
    st.markdown("#### âš¡ Monitored Appliances")
    from src.feature_engineering import APPLIANCE_SPECS
    rows = []
    for name, spec in APPLIANCE_SPECS.items():
        rows.append({
            "Appliance": name,
            "Rated Power (W)": spec["rated_power"],
            "Rated Current (A)": spec["rated_current"],
            "Max Safe Temp (Â°C)": spec["max_safe_temp"],
            "Nominal PF": spec["nominal_pf"],
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("#### ðŸ“Œ Fault Classes")
    faults = [
        ("Normal", "green", "All parameters within design limits"),
        ("Overload", "orange", "Current/power exceeds rated capacity"),
        ("Overheating", "orange", "Housing temperature above safe limit"),
        ("Overcurrent", "red", "Severe current spike (>2Ã— rated)"),
        ("Voltage Abnormality", "yellow", "Grid voltage sag/surge outside 207â€“253V"),
        ("Abnormal Power Consumption", "yellow", "Power factor or energy usage out of spec"),
        ("Potential Electrical Fault", "red", "Multiple critical deviations â€” immediate action needed"),
    ]
    for f, color, desc in faults:
        badge = f'<span class="badge badge-{color}">{f}</span>'
        st.markdown(
            f"<div style='padding:6px 0;border-bottom:1px solid #1e293b;'>"
            f"{badge} &nbsp; <span style='color:#94a3b8;font-size:13px;'>{desc}</span></div>",
            unsafe_allow_html=True
        )

