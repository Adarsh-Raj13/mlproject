"""UI Page Components initialization."""
from components.dashboard import render_dashboard_page
from components.appliance_monitoring import render_appliance_monitoring_page
from components.load_analytics import render_load_analytics_page
from components.fault_detection import render_fault_detection_page
from components.alert_centre import render_alert_centre_page
from components.analytics_reports import render_analytics_reports_page

__all__ = [
    "render_dashboard_page",
    "render_appliance_monitoring_page",
    "render_load_analytics_page",
    "render_fault_detection_page",
    "render_alert_centre_page",
    "render_analytics_reports_page"
]


