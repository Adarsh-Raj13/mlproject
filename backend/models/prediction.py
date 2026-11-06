"""MLPrediction SQLAlchemy model."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from backend.database import Base


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    prediction_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    appliance_id = Column(Integer, ForeignKey("appliances.appliance_id", ondelete="CASCADE"), nullable=False, index=True)
    reading_id = Column(Integer, ForeignKey("electrical_readings.reading_id", ondelete="SET NULL"))
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    predicted_load_category = Column(String(100), nullable=False)
    predicted_fault = Column(String(100), nullable=False)
    anomaly_status = Column(String(50), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(50), default="v1.0")
    abnormal_parameters = Column(Text)
    corrective_action = Column(Text)
