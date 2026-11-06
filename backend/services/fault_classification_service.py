"""Fault classification service."""
from backend.core.model_loader import model_loader

class FaultClassificationService:
    def predict(self, reading_dict):
        if not model_loader.engine:
            model_loader.load_all_models()
        res = model_loader.engine.predict_single(reading_dict)
        return res["predicted_fault"], res["confidence"], res["corrective_action"]

fault_classification_service = FaultClassificationService()
