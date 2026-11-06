"""Anomaly detection service."""
from backend.core.model_loader import model_loader

class AnomalyDetectionService:
    def predict(self, reading_dict):
        if not model_loader.engine:
            model_loader.load_all_models()
        res = model_loader.engine.predict_single(reading_dict)
        return res["is_anomaly"], res["anomaly_score"]

anomaly_detection_service = AnomalyDetectionService()

