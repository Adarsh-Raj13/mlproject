"""
FastAPI Backend Application Entry Point.
SMART LOAD PROFILING AND EARLY FAULT ALERT SYSTEM FOR ELECTRICAL APPLIANCES.
Provides REST APIs for appliance telemetry, ML predictions, alert triage, and analytics.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.config import settings
from backend.database import init_db, get_db
from backend.core.model_loader import model_loader
from backend.core.logging_config import logger
from backend.models.appliance import Appliance
from backend.models.reading import ElectricalReading
from backend.models.alert import Alert
from backend.schemas.common_schema import HealthResponse, SystemStatusResponse

# Import routers
from backend.routers import appliances, readings, predictions, alerts, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan management."""
    logger.info("Starting SMART LOAD MONITOR Backend Service...")
    # Initialize database tables and seed defaults
    init_db()
    # Load ML models into memory
    model_loader.load_all_models()
    logger.info("Database and Machine Learning models ready.")
    yield
    logger.info("Shutting down SMART LOAD MONITOR Backend Service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Electrical Monitoring, Load Profiling & Early Fault Detection REST API.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs for details.", "error": str(exc)}
    )


# System & Health Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["System"])
def check_health():
    """Returns application health and operational heartbeat."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION
    )


@app.get(f"{settings.API_PREFIX}/system/status", response_model=SystemStatusResponse, tags=["System"])
def get_system_status(db: Session = Depends(get_db)):
    """Provides operational readiness, database record counts, and ML model statuses."""
    app_count = db.query(Appliance).count()
    reading_count = db.query(ElectricalReading).count()
    active_alerts = db.query(Alert).filter(Alert.acknowledgement_status == "Active").count()
    crit_alerts = db.query(Alert).filter(Alert.severity == "Critical", Alert.acknowledgement_status == "Active").count()

    health_info = model_loader.check_health()

    return SystemStatusResponse(
        backend_status="online",
        database_status="connected",
        ml_models_loaded=health_info["all_loaded"],
        models_status=health_info["components"],
        total_appliances=app_count,
        total_readings=reading_count,
        active_alerts_count=active_alerts,
        critical_alerts_count=crit_alerts,
        version=settings.APP_VERSION
    )


# Register Feature Routers
app.include_router(appliances.router, prefix=settings.API_PREFIX)
app.include_router(readings.router, prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(alerts.router, prefix=settings.API_PREFIX)
app.include_router(analytics.router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

