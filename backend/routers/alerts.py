"""Safety alerts endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database import get_db
from backend.models.alert import Alert

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("/")
def get_alerts(
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """Retrieves alerts with optional status filtering."""
    q = db.query(Alert)
    if status and status != "All Statuses":
        q = q.filter(Alert.acknowledgement_status == status)
    return q.order_by(Alert.timestamp.desc()).limit(limit).all()

@router.patch("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Marks an active alert as acknowledged."""
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledgement_status = "Acknowledged"
    db.commit()
    return {"message": "Alert acknowledged", "alert_id": alert_id}

@router.patch("/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    """Marks an alert as resolved."""
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledgement_status = "Resolved"
    alert.resolved_at = datetime.utcnow()
    db.commit()
    return {"message": "Alert resolved", "alert_id": alert_id}

