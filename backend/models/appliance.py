"""Appliance SQLAlchemy model."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.database import Base


class Appliance(Base):
    __tablename__ = "appliances"

    appliance_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    appliance_name = Column(String(100), unique=True, nullable=False, index=True)
    appliance_type = Column(String(100), nullable=False)
    rated_voltage = Column(Float, nullable=False, default=230.0)
    rated_power = Column(Float, nullable=False)
    rated_current = Column(Float, nullable=False)
    nominal_pf = Column(Float, nullable=False)
    max_safe_temp = Column(Float, nullable=False)
    location = Column(String(150))
    installation_date = Column(String(50))
    operating_status = Column(String(50), default="Online")
    created_at = Column(DateTime, default=datetime.utcnow)

