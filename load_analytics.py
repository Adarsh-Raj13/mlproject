"""
Page 3: Load Profiling Component.
Analyzes energy consumption signatures, unsupervised K-Means load regimes,
daily and weekly energy curves, peak load periods, and appliance-wise comparative profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from database.database import DatabaseManager
from src.prediction import InferenceEngine


def render_load_analytics_page(db: DatabaseManager, engine: InferenceEngine):
    st.markdown("""
    <div class="iot-header">
        <h1>⚡ APPLIANCE LOAD PROFILING & REGIME ANALYTICS</h1>
        <p>Unsupervised K-Means Behavioral Regimes &bull; Energy Baseline Modeling &bull; Peak Demand Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    df_history = db.get_readings_history(limit=500)
    appliances = db.get_appliances()

    if df_history.empty:
        st.warning("No telemetry records available to build load profiles.")
        return

    # Top summary metrics on Load Profiling
    total_kwh = df_history["energy_kwh"].sum()
    avg_power = df_history["active_power"].mean()
    peak_power = df_history["active_power"].max()
    avg_pf = df_history["power_factor"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monitored Energy Total</div>
            <div class="kpi-value">{total_kwh:.2f} <span style="font-size:16px;">kWh</span></div>
            <div class="kpi-subtext">Cumulative across appliances</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Mean Active Power</div>
            <div class="kpi-value">{avg_power:.1f} <span style="font-size:16px;">W</span></div>
            <div class="kpi-subtext">Overall operational baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Observed Peak Surge</div>
            <div class="kpi-value" style="color:#ea580c;">{peak_power:.1f} <span style="font-size:16px;">W</span></div>
            <div class="kpi-subtext">Maximum transient recorded</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Aggregate System PF</div>
            <div class="kpi-value" style="color:#10b981;">{avg_pf:.3f}</div>
            <div class="kpi-subtext">Grid efficiency factor</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Section 1: Appliance Load Profiles & Baseline Ranges
    st.subheader("📋 Appliance Baseline Load Characterization")
    prof_data = []

    for a in appliances:
        app_name = a["name"]
        rated_p = a["rated_power"]
        rated_i = a["rated_current"]
        nom_pf = a["nominal_pf"]
        app_hist = df_history[df_history["appliance_name"] == app_name]

        if not app_hist.empty:
            min_p = app_hist["active_power"].min()
            max_p = app_hist["active_power"].max()
            mean_p = app_hist["active_power"].mean()
            tot_kwh = app_hist["energy_kwh"].sum()
        else:
            min_p, max_p, mean_p, tot_kwh = 0, rated_p, rated_p * 0.7, 1.0

        # Load behavior classification
        if rated_p > 1500:
            load_class = "Continuous High-Duty Heavy Load"
        elif rated_p > 800:
            load_class = "Intermittent High Resistive"
        elif app_name == "Refrigerator":
            load_class = "Cyclic Thermostat Inductive"
        else:
            load_class = "Low-Medium Variable Inductive"

        prof_data.append({
            "Appliance": app_name,
            "Rated (W)": rated_p,
            "Normal Range (W)": f"{min_p:.0f} - {max_p:.0f} W",
            "Avg Load (W)": f"{mean_p:.1f} W",
            "Nominal PF": nom_pf,
            "Total (kWh)": f"{tot_kwh:.3f}",
            "Load Regime Classification": load_class
        })

    st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)

    # Section 2: Visual Charts (Scatter Clusters & Comparison)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Active vs Reactive Power Signature (P-Q Plane)")
        fig_scatter = px.scatter(
            df_history,
            x="active_power",
            y="reactive_power",
            color="appliance_name",
            size="current",
            hover_data=["voltage", "power_factor", "temperature"],
            title="Appliance Impedance & Complex Power Clusters",
            labels={"active_power": "Active Power P (W)", "reactive_power": "Reactive Power Q (VAR)"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.subheader("⚡ Daily Consumption Comparison by Appliance")
        app_energy = df_history.groupby("appliance_name")["energy_kwh"].sum().reset_index()
        fig_bar = px.bar(
            app_energy,
            x="appliance_name",
            y="energy_kwh",
            color="appliance_name",
            text_auto=".2f",
            title="Total Energy Consumption (kWh)",
            labels={"energy_kwh": "Energy (kWh)", "appliance_name": "Appliance"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_bar.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Section 3: Machine Learning K-Means Load Regime Profiles
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("🤖 Unsupervised K-Means Behavioral Regimes")

    col_img, col_desc = st.columns([1, 1])
    cluster_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "cluster_distribution.png")

    with col_img:
        if os.path.exists(cluster_img_path):
            st.image(cluster_img_path, caption="Discovered Operating Regimes Distribution", use_container_width=True)
        else:
            st.info("Regime distribution chart will be shown here after model training.")

    with col_desc:
        st.markdown("""
        ### Operational Energy Regimes:
        - **Cluster 0: Eco / Standby Regime**
          Characterized by low idle current ($\le 0.35\\text{ A}$) and low thermal signature. Observed in idle appliances, fans at low speed, or refrigerator off-cycles.
        - **Cluster 1: Standard Normal Load Regime**
          Nominal operating band ($75\\text{ W} - 800\\text{ W}$) with optimal designed power factors ($0.75 - 0.95$). Appliances running within thermal equilibrium.
        - **Cluster 2: Heavy Continuous Load Regime**
          High sustained active power ($1200\\text{ W} - 2200\\text{ W}$), typical for Air Conditioners and Water Heaters during continuous heating/cooling duty.
        - **Cluster 3: Peak Surge / Extreme Demand Regime**
          Occurs during high inrush currents, motor jams, overload conditions, or combined simultaneous operations requiring immediate thermal monitoring.
        """)

