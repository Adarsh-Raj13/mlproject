"""Analytics service."""
from backend.database import SessionLocal
from backend.models.appliance import Appliance
from backend.models.alert import Alert
from backend.models.reading import ElectricalReading

class AnalyticsService:
    def get_summary(self):
        with SessionLocal() as db:
            total_apps = db.query(Appliance).count()
            crit = db.query(Alert).filter(Alert.severity == "Critical", Alert.acknowledgement_status == "Active").count()
            warn = db.query(Alert).filter(Alert.severity.in_(["High Risk", "Warning"]), Alert.acknowledgement_status == "Active").count()
            return {
                "total_appliances": total_apps,
                "critical_alerts": crit,
                "warning_alerts": warn,
                "overall_health_score": max(10, min(100, int(100 - (crit * 15 + warn * 5))))
            }

analytics_service = AnalyticsService()

