"""Appliance management endpoints."""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.appliance import Appliance

router = APIRouter(prefix="/appliances", tags=["Appliances"])

@router.get("/")
def list_appliances(db: Session = Depends(get_db)):
    """Returns all registered monitored electrical appliances."""
    return db.query(Appliance).all()

@router.get("/{appliance_id}")
def get_appliance(appliance_id: int, db: Session = Depends(get_db)):
    """Returns single appliance details."""
    app = db.query(Appliance).filter(Appliance.appliance_id == appliance_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Appliance not found")
    return app
