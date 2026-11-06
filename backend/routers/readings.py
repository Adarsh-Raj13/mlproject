"""Electrical telemetry readings endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.reading import ElectricalReading

router = APIRouter(prefix="/readings", tags=["Readings"])

@router.get("/")
def get_readings(
    appliance_id: Optional[int] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    """Retrieves recent telemetry measurements."""
    q = db.query(ElectricalReading)
    if appliance_id:
        q = q.filter(ElectricalReading.appliance_id == appliance_id)
    return q.order_by(ElectricalReading.timestamp.desc()).limit(limit).all()
