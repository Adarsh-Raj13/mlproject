"""
Page 5: Early Alert Centre Component.
Centralized alert lifecycle management, status filtering (Active, Acknowledged, Resolved),
severity breakdowns (Green, Yellow, Orange, Red), parameter attribution, and corrective actions.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from database.database import DatabaseManager


def render_alert_centre_page(db: DatabaseManager):
    st.markdown("""
    <div class="iot-header">
        <h1>🚨 EARLY WARNING ALERT & DISPATCH CENTRE</h1>
        <p>Real-Time Severity Triage &bull; Electrical Safety Threshold Violations &bull; Maintenance Actions</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        status_filter = st.selectbox("Status Filter", ["All Statuses", "Active", "Acknowledged", "Resolved"], index=1)
    with f_col2:
        severity_filter = st.selectbox("Severity Filter", ["All Severities", "Critical", "High Risk", "Warning", "Normal"], index=0)
    with f_col3:
        limit_val = st.slider("Max Alerts to Display", min_value=10, max_value=100, value=30, step=10)

    # Fetch alerts from database
    st_arg = None if status_filter == "All Statuses" else status_filter
    alerts = db.get_recent_alerts(limit=limit_val, status=st_arg)

    if severity_filter != "All Severities":
        alerts = [a for a in alerts if a["severity"] == severity_filter]

    # Metrics Row
    total_alerts = len(alerts)
    crit_count = sum(1 for a in alerts if a["severity"] == "Critical")
    high_count = sum(1 for a in alerts if a["severity"] == "High Risk")
    warn_count = sum(1 for a in alerts if a["severity"] == "Warning")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Matching Alerts</div>
            <div class="kpi-value">{total_alerts}</div>
            <div class="kpi-subtext">Current view count</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Critical (Red)</div>
            <div class="kpi-value" style="color:#ef4444;">{crit_count}</div>
            <div class="kpi-subtext">Immediate shutdown risk</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Risk (Orange)</div>
            <div class="kpi-value" style="color:#ea580c;">{high_count}</div>
            <div class="kpi-subtext">Thermal/Overload stress</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Warnings (Yellow)</div>
            <div class="kpi-value" style="color:#f59e0b;">{warn_count}</div>
            <div class="kpi-subtext">Preventive attention</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    if not alerts:
        st.success("🎉 No alerts matching the selected filters. System operating within all safety margins.")
        return

    st.subheader("📋 Active Alert Queue & History")

    # Render each alert as an actionable card
    for alert in alerts:
        a_id = alert["id"]
        app_name = alert["appliance_name"]
        ts = alert["timestamp"]
        sev = alert["severity"]
        fault = alert["fault_type"]
        param = alert["trigger_parameter"]
        val = alert["trigger_value"]
        thresh = alert["threshold_value"]
        action = alert["suggested_action"]
        status = alert["status"]

        badge_class = "badge-red" if sev == "Critical" else (
            "badge-orange" if sev == "High Risk" else (
                "badge-yellow" if sev == "Warning" else "badge-green"
            )
        )

        with st.container():
            st.markdown(f"""
            <div class="telemetry-card" style="margin-bottom: 12px; border-left: 5px solid {'#ef4444' if sev == 'Critical' else ('#ea580c' if sev == 'High Risk' else '#f59e0b')};">
                <div class="telemetry-card-title">
                    <span><b>{app_name}</b> &bull; <span style="font-weight:500; font-size:14px;">{fault}</span></span>
                    <div>
                        <span class="badge-pill {badge_class}">{sev}</span>
                        <span style="font-size:12px; margin-left:8px; color:#64748b;">Status: <b>{status}</b></span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 2fr 3fr; gap: 16px; font-size: 13px;">
                    <div>
                        <div><b>Timestamp:</b> {ts}</div>
                        <div><b>Trigger Parameter:</b> <span style="color:#b91c1c; font-weight:600;">{param}</span> (Value: {val:.2f} | Limit: {thresh:.2f})</div>
                    </div>
                    <div>
                        <b>🛠️ Corrective Action:</b>
                        <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; margin-top:4px; color:#1e293b;">
                            {action}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick Action Buttons
            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            if status == "Active":
                with btn_col1:
                    if st.button("Acknowledge", key=f"ack_{a_id}"):
                        db.acknowledge_alert(a_id)
                        st.rerun()
                with btn_col2:
                    if st.button("Resolve Alert", key=f"res_{a_id}", type="primary"):
                        db.resolve_alert(a_id)
                        st.rerun()
            elif status == "Acknowledged":
                with btn_col1:
                    if st.button("Resolve Alert", key=f"res_ack_{a_id}", type="primary"):
                        db.resolve_alert(a_id)
                        st.rerun()

