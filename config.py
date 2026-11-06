"""
Backend Configuration Module.
Defines application settings, database paths, ML model artifact paths,
and runtime configuration options using Pydantic settings.
"""

import os
from typing import List
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseModel):
    """Application configuration settings."""
    APP_NAME: str = "SMART LOAD MONITOR API"
    APP_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Paths
    BASE_DIR: str = BASE_DIR
    DB_PATH: str = os.path.join(BASE_DIR, "smart_load_monitor.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'smart_load_monitor.db')}")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "http://localhost:8000",  # FastAPI server
        "http://127.0.0.1:8000",
        "*"
    ]

    # Machine Learning Model Filenames
    PREPROCESSING_PIPELINE: str = "preprocessing_pipeline.pkl"
    LOAD_PROFILING_MODEL: str = "load_profiling_model.pkl"
    ANOMALY_DETECTION_MODEL: str = "anomaly_detection_model.pkl"
    FAULT_CLASSIFICATION_MODEL: str = "fault_classification_model.pkl"


settings = Settings()

