"""Load profiling service."""
from backend.core.model_loader import model_loader

class LoadProfilingService:
    def predict(self, reading_dict):
        if not model_loader.engine:
            model_loader.load_all_models()
        res = model_loader.engine.predict_single(reading_dict)
        return res["load_profile_label"], res["consumption_category"]

load_profiling_service = LoadProfilingService()
