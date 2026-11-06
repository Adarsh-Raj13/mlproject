"""Model Loader singleton."""
from typing import Dict, Any
from src.prediction import InferenceEngine

class ModelLoader:
    def __init__(self):
        self.engine = None

    def load_all_models(self) -> InferenceEngine:
        self.engine = InferenceEngine()
        return self.engine

    def check_health(self) -> Dict[str, Any]:
        if self.engine is None:
            self.load_all_models()
        loaded = self.engine is not None and self.engine.is_loaded
        return {
            "all_loaded": loaded,
            "components": {
                "preprocessor": self.engine.preprocessor is not None if self.engine else False,
                "load_profiler": self.engine.load_profiler is not None if self.engine else False,
                "anomaly_detector": self.engine.anomaly_detector is not None if self.engine else False,
                "fault_classifier": self.engine.fault_classifier is not None if self.engine else False
            }
        }

model_loader = ModelLoader()
__all__ = ["model_loader", "ModelLoader"]

