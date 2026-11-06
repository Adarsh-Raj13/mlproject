"""Model Loader singleton."""
from src.prediction import InferenceEngine

class ModelLoader:
    def __init__(self):
        self.engine = None

    def load_all_models(self) -> InferenceEngine:
        self.engine = InferenceEngine()
        return self.engine

model_loader = ModelLoader()
__all__ = ["model_loader", "ModelLoader"]

