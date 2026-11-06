"""Common Pydantic request and response schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class SystemStatusResponse(BaseModel):
    total_appliances: int
    critical_alerts: int
    warning_alerts: int
    overall_health_score: int
    total_active_power: str
    average_power_factor: float

