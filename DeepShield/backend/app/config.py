"""
DeepShield Configuration Module.

Centralizes environment configuration, API settings, and model identifiers.
Settings are loaded from environment variables and optional .env files.
Never hardcodes secrets or API keys.
"""

import os
from typing import List
from pathlib import Path


class Settings:
    """Application settings read from environment variables with sensible defaults."""

    def __init__(self) -> None:
        self.app_name: str = "DeepShield"
        self.app_version: str = "0.1.0"
        self.tagline: str = "See the fake before it fools you."
        self.project_name: str = "CodeQuest — National Tech Hackathon 2026"
        self.team_name: str = "Code Ph@nt0m"

        # Environment
        self.app_env: str = os.getenv("APP_ENV", "development").lower()
        self.debug: bool = self.app_env == "development"

        # Network settings
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))

        # CORS allowed origins
        raw_origins = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        )
        self.allowed_origins: List[str] = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

        # Model identifier (Candidate model for Phase 2 verification)
        self.model_name: str = os.getenv(
            "MODEL_NAME", "dima806/deepfake_vs_real_image_detection"
        )

        # Upload constraints
        self.max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
        self.max_upload_size_bytes: int = self.max_upload_size_mb * 1024 * 1024

        # Supported MIME types for analysis
        self.allowed_image_types: List[str] = [
            "image/jpeg",
            "image/png",
            "image/webp",
        ]

        # Face Detection & Classification Thresholds (Phase 3)
        self.min_face_detection_confidence: float = float(
            os.getenv("MIN_FACE_DETECTION_CONFIDENCE", "0.50")
        )
        self.face_padding_factor: float = float(
            os.getenv("FACE_PADDING_FACTOR", "0.20")
        )
        self.real_threshold: float = float(os.getenv("REAL_THRESHOLD", "0.70"))
        self.fake_threshold: float = float(os.getenv("FAKE_THRESHOLD", "0.70"))

        # External API Keys (Optional, future phases; never hardcoded)
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY")

        # Paths
        self.base_dir: Path = Path(__file__).resolve().parent.parent

    @property
    def is_production(self) -> bool:
        """Return True if running in production environment."""
        return self.app_env == "production"


# Singleton instance
settings = Settings()
