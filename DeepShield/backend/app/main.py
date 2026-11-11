"""
DeepShield FastAPI Application Entry Point.

Explainable AI-Powered Facial Deepfake Detection Platform.
Project: CodeQuest — National Tech Hackathon 2026
Team: Code Ph@nt0m
"""

import logging
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.health import router as health_router
from app.api.analyze import router as analyze_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepshield.main")

# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=(
        "Explainable AI-Powered Facial Deepfake Detection API. "
        "Provides media inspection, MediaPipe face localization, ViT classification, "
        "and Grad-CAM visual explainability."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS) for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router, prefix="")
app.include_router(analyze_router, prefix="")


@app.get(
    "/",
    tags=["Root"],
    summary="DeepShield Platform Root",
    description="Returns platform identity, project context, active phase, and documentation links.",
)
async def root() -> Dict[str, Any]:
    """Root endpoint welcoming clients and confirming API reachability."""
    return {
        "name": settings.app_name,
        "tagline": settings.tagline,
        "version": settings.app_version,
        "project": settings.project_name,
        "team": settings.team_name,
        "phase": "Phase 4: Vision Transformer Grad-CAM Explainability",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "health_v1": "/api/v1/health",
            "analyze": "/analyze",
            "analyze_v1": "/api/v1/analyze",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler to ensure consistent JSON error response formatting.
    Prevents leaking internal stack traces in non-debug environments.
    """
    logger.exception(f"Unhandled server error on {request.method} {request.url.path}: {exc}")
    
    error_message = str(exc) if settings.debug else "An internal server error occurred."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": error_message,
                "path": str(request.url.path),
            },
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
