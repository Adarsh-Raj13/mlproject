"""ElectricalReading SQLAlchemy model."""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, Boolean, ForeignKey
from backend.database import Base


class ElectricalReading(Base):
    __tablename__ = "electrical_readings"

    reading_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    appliance_id = Column(Integer, ForeignKey("appliances.appliance_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    active_power = Column(Float, nullable=False)
    reactive_power = Column(Float, nullable=False)
    apparent_power = Column(Float, nullable=False)
    power_factor = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False, default=50.0)
    temperature = Column(Float, nullable=False)
    operating_duration = Column(Float, nullable=False, default=0.0)
    energy_consumption = Column(Float, nullable=False, default=0.0)
    is_simulated = Column(Boolean, default=False)
