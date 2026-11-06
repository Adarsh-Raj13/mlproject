"""Common Pydantic request and response schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class SystemStatusResponse(BaseModel):
    backend_status: str
    database_status: str
    ml_models_loaded: bool
    models_status: Dict[str, Any]
    total_appliances: int
    total_readings: int
    active_alerts_count: int
    critical_alerts_count: int
    version: str

