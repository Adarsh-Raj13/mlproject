"""Alert SQLAlchemy model."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from backend.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    appliance_id = Column(Integer, ForeignKey("appliances.appliance_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    detected_parameter = Column(String(100), nullable=False)
    parameter_value = Column(Float, nullable=False)
    threshold_value = Column(Float)
    recommended_action = Column(Text, nullable=False)
    acknowledgement_status = Column(String(50), nullable=False, default="Active")
    resolved_at = Column(DateTime)
