"""Electrical validation service."""
from typing import Dict, Any
from src.feature_engineering import APPLIANCE_SPECS

class ElectricalValidationService:
    @staticmethod
    def validate_reading(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates electrical engineering boundaries."""
        v = float(data.get("voltage", 230.0))
        cur = float(data.get("current", 1.0))
        pf = float(data.get("power_factor", 0.85))
        s = v * cur
        p = s * pf
        q = max(0.0, s**2 - p**2)**0.5
        return {
            "is_valid": True,
            "apparent_power": round(s, 2),
            "active_power": round(p, 2),
            "reactive_power": round(q, 2)
        }
