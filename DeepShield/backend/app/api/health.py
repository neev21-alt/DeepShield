"""
Health and readiness verification endpoints for DeepShield backend.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.config import settings
from app.schemas.analysis import HealthResponse
from app.services.model_service import model_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Model Status",
    description="Reports API operational status, candidate model identifier, architecture, and memory load status without falsely reporting model availability.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="API v1 Health & Model Status",
    description="Reports API operational status, candidate model identifier, architecture, and memory load status without falsely reporting model availability.",
)
async def get_health() -> HealthResponse:
    """
    Return operational telemetry and AI model status.
    
    Reflects genuine model lifecycle status:
    - If model is loaded: model_loaded=True, architecture, and device.
    - If model loading failed or hasn't occurred: model_loaded=False.
    """
    is_loaded = model_service.is_loaded
    architecture = model_service.architecture if is_loaded else None

    return HealthResponse(
        status="ok",
        backend=f"{settings.app_name} API",
        version=settings.app_version,
        model_loaded=is_loaded,
        model_identifier=model_service.model_name,
        model_architecture=architecture,
        device=model_service.device,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
