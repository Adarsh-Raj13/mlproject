"""
Page 6: Analytics and Reports Component.
Comprehensive reporting dashboard displaying model performance benchmarks,
confusion matrix, anomaly distribution graphs, fault frequency analysis,
energy consumption summaries, and one-click CSV report exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from database.database import DatabaseManager

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def render_analytics_reports_page(db: DatabaseManager):
    st.markdown("""
    <div class="iot-header">
        <h1>📑 ANALYTICS, BENCHMARKS & AUDIT REPORTS</h1>
        <p>Model Evaluation Metrics &bull; Fault Frequency Distributions &bull; Energy Audits & CSV Exports</p>
    </div>
    """, unsafe_allow_html=True)

    tab_eval, tab_energy, tab_faults, tab_export = st.tabs([
        "🤖 ML Model Evaluation & Benchmarks",
        "⚡ Energy Consumption Reports",
        "📊 Fault Frequency & Anomalies",
        "💾 Data Export Center"
    ])

    # ---------------- TAB 1: MODEL EVALUATION & BENCHMARKS ----------------
    with tab_eval:
        st.subheader("Machine Learning Performance & Metrics")

        # Load text report if available
        summary_txt_path = os.path.join(REPORTS_DIR, "model_evaluation.txt")
        if os.path.exists(summary_txt_path):
            with open(summary_txt_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.text_area("Full Evaluation & Classification Benchmark Log", report_content, height=220)

        c1, c2 = st.columns(2)
        with c1:
            cm_img_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Multiclass Fault Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found. Run model training pipeline.")

        with c2:
            comp_img_path = os.path.join(REPORTS_DIR, "model_comparison.png")
            if os.path.exists(comp_img_path):
                st.image(comp_img_path, caption="Algorithm Performance Comparison (Accuracy, F1, Precision, Recall)", use_container_width=True)
            else:
                st.info("Model comparison plot not found.")

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            anom_img_path = os.path.join(REPORTS_DIR, "anomaly_distribution.png")
            if os.path.exists(anom_img_path):
                st.image(anom_img_path, caption="Isolation Forest Anomaly Score Distribution", use_container_width=True)

        with c4:
            clust_img_path = os.path.join(REPORTS_DIR, "cluster_distribution.png")
            if os.path.exists(clust_img_path):
                st.image(clust_img_path, caption="K-Means Energy Regime Distribution", use_container_width=True)

    # ---------------- TAB 2: ENERGY CONSUMPTION REPORTS ----------------
    with tab_energy:
        st.subheader("Appliance-Wise Energy Consumption Audit")
        df_history = db.get_readings_history(limit=500)

        if not df_history.empty:
            summary_stats = df_history.groupby("appliance_name").agg(
                Total_kWh=("energy_kwh", "sum"),
                Avg_Power_W=("active_power", "mean"),
                Max_Power_W=("active_power", "max"),
                Avg_PF=("power_factor", "mean"),
                Avg_Temp_C=("temperature", "mean"),
                Reading_Count=("id", "count")
            ).reset_index()

            st.dataframe(summary_stats.round(3), use_container_width=True, hide_index=True)

            # Energy Bar Chart
            fig_energy = px.bar(
                summary_stats,
                x="appliance_name",
                y="Total_kWh",
                color="appliance_name",
                title="Total Energy Consumption (kWh) by Monitored Asset",
                labels={"Total_kWh": "Energy (kWh)", "appliance_name": "Appliance"},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_energy.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_energy, use_container_width=True)
        else:
            st.info("No telemetry records available for energy audit.")

    # ---------------- TAB 3: FAULT FREQUENCY & ANOMALIES ----------------
    with tab_faults:
        st.subheader("Fault Occurrence Frequency & Distribution")
        alerts = db.get_recent_alerts(limit=200)

        if alerts:
            df_alerts = pd.DataFrame(alerts)
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                fault_counts = df_alerts["fault_type"].value_counts().reset_index()
                fault_counts.columns = ["Fault Type", "Count"]
                fig_fault_bar = px.bar(
                    fault_counts,
                    x="Fault Type",
                    y="Count",
                    color="Fault Type",
                    title="Fault Incidents by Category",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                fig_fault_bar.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_fault_bar, use_container_width=True)

            with col_f2:
                sev_counts = df_alerts["severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig_sev_pie = px.pie(
                    sev_counts,
                    names="Severity",
                    values="Count",
                    title="Alert Severity Breakdown",
                    hole=0.4,
                    color="Severity",
                    color_discrete_map={
                        "Critical": "#ef4444",
                        "High Risk": "#ea580c",
                        "Warning": "#f59e0b",
                        "Normal": "#10b981"
                    }
                )
                fig_sev_pie.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_sev_pie, use_container_width=True)
        else:
            st.info("No fault incidents recorded in alert logs.")

    # ---------------- TAB 4: DATA EXPORT CENTER ----------------
    with tab_export:
        st.subheader("Data Export & Audit Logs Download")
        st.caption("Generate downloadable CSV snapshots for regulatory, audit, or academic review.")

        exp_c1, exp_c2, exp_c3 = st.columns(3)

        with exp_c1:
            st.markdown("#### ⚡ Telemetry Readings Log")
            df_readings = db.get_readings_history(limit=1000)
            if not df_readings.empty:
                csv_readings = df_readings.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Readings CSV",
                    data=csv_readings,
                    file_name=f"electrical_readings_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with exp_c2:
            st.markdown("#### 🚨 Alert Incident History")
            all_alerts = db.get_recent_alerts(limit=500)
            if all_alerts:
                df_alerts_exp = pd.DataFrame(all_alerts)
                csv_alerts = df_alerts_exp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Alerts CSV",
                    data=csv_alerts,
                    file_name=f"alert_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with exp_c3:
            st.markdown("#### 📋 Processed Dataset")
            processed_csv_path = "dataset/processed_data.csv"
            if os.path.exists(processed_csv_path):
                with open(processed_csv_path, "rb") as f:
                    st.download_button(
                        "📥 Download Processed Dataset (CSV)",
                        data=f,
                        file_name="processed_electrical_dataset.csv",
                        mime="text/csv"
                    )

