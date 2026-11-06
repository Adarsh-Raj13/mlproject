"""
Page 2: Appliance Monitoring Component.
Detailed telemetry inspection for each of the 6 monitored appliances:
Ceiling Fan, Refrigerator, Air Conditioner, Washing Machine, Electric Iron, Water Heater.
Shows live parameter gauges, wave/trend charts, power factor indicators, and load condition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from database.database import DatabaseManager


def render_appliance_monitoring_page(db: DatabaseManager):
    st.markdown("""
    <div class="iot-header">
        <h1>ðŸ“Š APPLIANCE TELEMETRY MONITORING</h1>
        <p>In-Depth Telemetry, Electrical Signatures & Dynamic Operational Metrics for Individual Equipment</p>
    </div>
    """, unsafe_allow_html=True)

    appliances = db.get_appliances()
    app_names = [a["name"] for a in appliances]

    if not app_names:
        st.error("No appliances found in registry.")
        return

    # Appliance selector in horizontal tabs or selectbox
    selected_app_name = st.selectbox(
        "ðŸŽ¯ Select Monitored Appliance for Detailed Telemetry Inspection:",
        app_names,
        index=0
    )

    app_meta = db.get_appliance_by_name(selected_app_name)
    df_history = db.get_readings_history(appliance_name=selected_app_name, limit=120)

    if df_history.empty:
        st.warning(f"No historical telemetry records for {selected_app_name}. Please initialize database or start data simulator.")
        return

    latest = df_history.iloc[0]

    # Calculate status & load condition
    rated_p = app_meta["rated_power"]
    rated_i = app_meta["rated_current"]
    max_t = app_meta["max_safe_temp"]

    p_val = float(latest["active_power"])
    i_val = float(latest["current"])
    v_val = float(latest["voltage"])
    pf_val = float(latest["power_factor"])
    t_val = float(latest["temperature"])
    e_val = float(latest["energy_kwh"])
    dur_val = float(latest["operating_duration"])

    # Load condition evaluation
    p_ratio = p_val / rated_p if rated_p > 0 else 1.0
    if p_ratio > 1.35 or i_val > rated_i * 1.5:
        load_condition = "Severe Overload"
        status_badge = '<span class="badge-pill badge-red">Critical Alert</span>'
    elif p_ratio > 1.1 or t_val > max_t:
        load_condition = "High Thermal / Electrical Stress"
        status_badge = '<span class="badge-pill badge-orange">Warning Condition</span>'
    elif p_ratio < 0.2:
        load_condition = "Low Duty / Standby"
        status_badge = '<span class="badge-pill badge-green">Normal / Eco</span>'
    else:
        load_condition = "Optimal Operational Load"
        status_badge = '<span class="badge-pill badge-green">Normal Steady</span>'

    # Appliance Specification & Operational Summary Header
    st.markdown(f"""
    <div class="telemetry-card" style="margin-top: 10px;">
        <div class="telemetry-card-title">
            <span>{selected_app_name} &bull; <span style="font-weight:400; font-size:14px; color:#64748b;">{app_meta['category']} ({app_meta['location']})</span></span>
            {status_badge}
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #475569; margin-bottom: 12px;">
            <div><b>Rated Power:</b> {rated_p} W</div>
            <div><b>Rated Current:</b> {rated_i} A</div>
            <div><b>Nominal PF:</b> {app_meta['nominal_pf']}</div>
            <div><b>Thermal Limit:</b> {max_t} Â°C</div>
            <div><b>Current Condition:</b> <b>{load_condition}</b></div>
            <div><b>Cumulative Energy:</b> {e_val:.3f} kWh</div>
            <div><b>Run Duration:</b> {dur_val:.1f} mins</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Gauges / Parameter Value Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fig_v = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v_val,
            title={'text': "Voltage (V)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [170, 280]},
                'bar': {'color': "#0284c7"},
                'steps': [
                    {'range': [170, 207], 'color': "#fee2e2"},
                    {'range': [207, 253], 'color': "#ecfdf5"},
                    {'range': [253, 280], 'color': "#fee2e2"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 255}
            }
        ))
        fig_v.update_layout(height=200, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_v, width='stretch')

    with c2:
        fig_i = go.Figure(go.Indicator(
            mode="gauge+number",
            value=i_val,
            title={'text': "Current (A)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, rated_i * 2.5]},
                'bar': {'color': "#2563eb"},
                'steps': [
                    {'range': [0, rated_i * 1.1], 'color': "#ecfdf5"},
                    {'range': [rated_i * 1.1, rated_i * 1.5], 'color': "#fef3c7"},
                    {'range': [rated_i * 1.5, rated_i * 2.5], 'color': "#fee2e2"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': rated_i * 1.4}
            }
        ))
        fig_i.update_layout(height=200, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_i, width='stretch')

    with c3:
        fig_p = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_val,
            title={'text': "Active Power (W)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, rated_p * 2.0]},
                'bar': {'color': "#0f766e"},
                'steps': [
                    {'range': [0, rated_p * 1.05], 'color': "#ecfdf5"},
                    {'range': [rated_p * 1.05, rated_p * 1.35], 'color': "#fef3c7"},
                    {'range': [rated_p * 1.35, rated_p * 2.0], 'color': "#fee2e2"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': rated_p * 1.25}
            }
        ))
        fig_p.update_layout(height=200, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_p, width='stretch')

    with c4:
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number",
            value=t_val,
            title={'text': "Temperature (Â°C)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [20, 110]},
                'bar': {'color': "#ea580c"},
                'steps': [
                    {'range': [20, max_t * 0.9], 'color': "#ecfdf5"},
                    {'range': [max_t * 0.9, max_t], 'color': "#fef3c7"},
                    {'range': [max_t, 110], 'color': "#fee2e2"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': max_t}
            }
        ))
        fig_t.update_layout(height=200, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_t, width='stretch')

    # Historical Telemetry Trends
    st.subheader(f"ðŸ“ˆ Historical Telemetry Time Series: {selected_app_name}")
    df_chart = df_history.sort_values("timestamp")

    tab1, tab2, tab3 = st.tabs(["âš¡ Power & Current Signature", "ðŸŒ¡ï¸ Thermal Dynamics", "ðŸ“ AC Phasor & Power Factor"])

    with tab1:
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(
            x=df_chart["timestamp"], y=df_chart["active_power"],
            name="Active Power (W)", line=dict(color="#0284c7", width=2.5)
        ))
        fig_power.add_trace(go.Scatter(
            x=df_chart["timestamp"], y=df_chart["apparent_power"],
            name="Apparent Power S (VA)", line=dict(color="#94a3b8", dash="dot")
        ))
        fig_power.add_trace(go.Scatter(
            x=df_chart["timestamp"], y=df_chart["reactive_power"],
            name="Reactive Power Q (VAR)", line=dict(color="#f59e0b", dash="dash")
        ))
        fig_power.update_layout(
            title="Complex Power Dynamics (P, Q, S)",
            xaxis_title="Time",
            yaxis_title="Power (W / VAR / VA)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_power, width='stretch')

    with tab2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df_chart["timestamp"], y=df_chart["temperature"],
            name="Measured Temp (Â°C)", line=dict(color="#ea580c", width=2.5)
        ))
        fig_temp.add_hline(
            y=max_t, line_dash="dash", line_color="red",
            annotation_text=f"Max Safe Threshold ({max_t}Â°C)"
        )
        fig_temp.update_layout(
            title="Thermal Gradient Profile",
            xaxis_title="Time",
            yaxis_title="Temperature (Â°C)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_temp, width='stretch')

    with tab3:
        fig_pf = px.line(
            df_chart, x="timestamp", y="power_factor",
            title="Power Factor Variation",
            labels={"power_factor": "Power Factor (cos Ï†)", "timestamp": "Timestamp"},
            template="plotly_white",
            color_discrete_sequence=["#10b981"]
        )
        fig_pf.add_hline(
            y=app_meta["nominal_pf"], line_dash="dot", line_color="blue",
            annotation_text=f"Nominal Design PF ({app_meta['nominal_pf']})"
        )
        fig_pf.update_yaxes(range=[0.4, 1.02])
        fig_pf.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pf, width='stretch')


