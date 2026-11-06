"""Analytics summary endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.appliance import Appliance
from backend.models.alert import Alert
from backend.models.reading import ElectricalReading

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """Computes high level system stats."""
    total_apps = db.query(Appliance).count()
    crit_alerts = db.query(Alert).filter(Alert.severity == "Critical", Alert.acknowledgement_status == "Active").count()
    warn_alerts = db.query(Alert).filter(Alert.severity.in_(["High Risk", "Warning"]), Alert.acknowledgement_status == "Active").count()
    total_readings = db.query(ElectricalReading).count()
    return {
        "total_appliances": total_apps,
        "critical_alerts": crit_alerts,
        "warning_alerts": warn_alerts,
        "total_readings": total_readings,
        "health_score": max(10, min(100, int(100 - (crit_alerts * 15 + warn_alerts * 5))))
    }

