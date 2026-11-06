"""
Page 1: Home Dashboard Component.
Displays high-level KPI cards, overall system health, appliance status cards,
real-time telemetry trends, recent alerts, and load distribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from database.database import DatabaseManager


def render_dashboard_page(db: DatabaseManager):
    st.markdown("""
    <div class="iot-header">
        <h1>âš¡ SMART LOAD MONITOR</h1>
        <p>Intelligent Electrical Monitoring and Early Fault Detection System &bull; Live Facility Telemetry</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Fetch System Telemetry & Statistics
    stats = db.get_system_summary_stats()
    latest_readings = db.get_latest_readings_for_all_appliances()
    recent_alerts = db.get_recent_alerts(limit=5)
    df_history = db.get_readings_history(limit=250)

    # Compute operational counts
    total_apps = stats["total_appliances"]
    crit_count = stats["critical_alerts"]
    warn_count = stats["warning_alerts"]
    normal_count = max(0, total_apps - (crit_count + warn_count))
    health_score = stats["overall_health_score"]

    # 2. Top Metric KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monitored Appliances</div>
            <div class="kpi-value">{total_apps}</div>
            <div class="kpi-subtext">Active telemetry feeds</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Normal Operation</div>
            <div class="kpi-value" style="color: #10b981;">{normal_count}</div>
            <div class="kpi-subtext">Nominal operating state</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Warnings</div>
            <div class="kpi-value" style="color: #f59e0b;">{warn_count}</div>
            <div class="kpi-subtext">Requires inspection</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Critical Alerts</div>
            <div class="kpi-value" style="color: #ef4444;">{crit_count}</div>
            <div class="kpi-subtext">Immediate action needed</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Active Load</div>
            <div class="kpi-value">{stats['total_active_power']} <span style="font-size:16px;">W</span></div>
            <div class="kpi-subtext">Avg PF: {stats['average_power_factor']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        health_color = "#10b981" if health_score > 80 else ("#f59e0b" if health_score > 50 else "#ef4444")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">System Health</div>
            <div class="kpi-value" style="color: {health_color};">{health_score}%</div>
            <div class="kpi-subtext">Reliability Index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # 3. Monitored Appliance Status Cards
    st.subheader("ðŸ”Œ Monitored Appliances Status")
    cols = st.columns(3)

    for i, app in enumerate(latest_readings):
        col_idx = i % 3
        with cols[col_idx]:
            app_name = app["appliance_name"]
            v = app.get("voltage", 230.0) or 230.0
            cur = app.get("current", 0.0) or 0.0
            p = app.get("active_power", 0.0) or 0.0
            pf = app.get("power_factor", 0.85) or 0.85
            temp = app.get("temperature", 32.0) or 32.0

            # Calculate badge status
            if temp > app.get("max_safe_temp", 65.0) or cur > app.get("rated_current", 5.0) * 1.5:
                badge_html = '<span class="badge-pill badge-red">â— Critical</span>'
            elif p > app.get("rated_power", 1000.0) * 1.15 or v < 205.0 or v > 250.0:
                badge_html = '<span class="badge-pill badge-orange">â–² Warning</span>'
            else:
                badge_html = '<span class="badge-pill badge-green">âœ” Normal</span>'

            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-card-title">
                    <span>{app_name}</span>
                    {badge_html}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 10px;">
                    {app['category']} &bull; Rated: {app['rated_power']}W ({app['rated_current']}A)
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;">
                    <div class="param-box">
                        <div class="param-label">Voltage</div>
                        <div class="param-val">{v:.1f}V</div>
                    </div>
                    <div class="param-box">
                        <div class="param-label">Current</div>
                        <div class="param-val">{cur:.2f}A</div>
                    </div>
                    <div class="param-box">
                        <div class="param-label">Power</div>
                        <div class="param-val">{p:.0f}W</div>
                    </div>
                    <div class="param-box">
                        <div class="param-label">Temp</div>
                        <div class="param-val">{temp:.1f}Â°C</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Graphs & Visualizations: Load Distribution & Trends
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns([7, 5])

    with g1:
        st.subheader("ðŸ“ˆ Electrical Load & Power Trends")
        if not df_history.empty:
            # Sort ascending for time series line chart
            df_plot = df_history.sort_values("timestamp")
            fig_trend = px.line(
                df_plot,
                x="timestamp",
                y="active_power",
                color="appliance_name",
                title="Active Power Consumption (W) Over Time",
                labels={"active_power": "Active Power (W)", "timestamp": "Timestamp", "appliance_name": "Appliance"},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_trend.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend, width='stretch')
        else:
            st.info("No historical telemetry available yet. Start simulation or train models to generate records.")

    with g2:
        st.subheader("ðŸ¥§ Load Distribution")
        if latest_readings:
            power_dist = [
                {"Appliance": r["appliance_name"], "Active Power (W)": max(1.0, r.get("active_power", 0.0) or 1.0)}
                for r in latest_readings
            ]
            df_dist = pd.DataFrame(power_dist)
            fig_pie = px.pie(
                df_dist,
                names="Appliance",
                values="Active Power (W)",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_pie.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=20),
                legend=dict(orientation="v", yanchor="middle", y=0.5)
            )
            st.plotly_chart(fig_pie, width='stretch')

    # 5. Recent Alerts Section
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("ðŸš¨ Recent Early Warning Alerts")

    if recent_alerts:
        alert_rows = []
        for a in recent_alerts:
            sev = a["severity"]
            badge_class = "badge-red" if sev == "Critical" else ("badge-orange" if sev == "High Risk" else "badge-yellow")
            alert_rows.append({
                "Timestamp": a["timestamp"],
                "Appliance": a["appliance_name"],
                "Severity": f"<span class='badge-pill {badge_class}'>{sev}</span>",
                "Fault Type": a["fault_type"],
                "Trigger Parameter": f"{a['trigger_parameter']} ({a['trigger_value']:.2f})",
                "Recommended Action": a["suggested_action"],
                "Status": a["status"]
            })
        df_alert_table = pd.DataFrame(alert_rows)
        st.markdown(df_alert_table.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.success("All systems operating within normal parameters. No active alerts.")

    # Engineering Disclaimer
    st.markdown("""
    <div class="safety-disclaimer">
        â„¹ <b>Engineering Decision-Support Disclaimer:</b> This monitoring system is an educational decision-support
        and diagnostic tool powered by Machine Learning. It does not replace physical calibrated circuit breakers (MCBs),
        residual current devices (RCDs), fuses, or statutory electrical safety inspections.
    </div>
    """, unsafe_allow_html=True)


