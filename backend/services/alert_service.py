"""Alert service."""
from backend.database import SessionLocal
from backend.models.alert import Alert

class AlertService:
    def get_recent(self, limit: int = 50):
        with SessionLocal() as db:
            return db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()

alert_service = AlertService()

