"""ML Prediction diagnostic endpoints."""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from backend.core.model_loader import model_loader

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.post("/predict")
def predict_single_reading(reading: Dict[str, Any]):
    """Runs live ML inference over a sensor observation packet."""
    if not model_loader.engine or not model_loader.engine.is_loaded:
        model_loader.load_all_models()
    if not model_loader.engine.is_loaded:
        raise HTTPException(status_code=503, detail="ML Models are not loaded.")
    return model_loader.engine.predict_single(reading)

