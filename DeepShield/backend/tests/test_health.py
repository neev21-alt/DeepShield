"""
Tests for DeepShield health and root endpoints (Phase 2).
"""

from fastapi.testclient import TestClient
from app.main import app
from app.services.model_service import model_service

client = TestClient(app)


def test_root_endpoint() -> None:
    """Verify that root endpoint responds with application metadata and 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DeepShield"
    assert data["tagline"] == "See the fake before it fools you."
    assert "Phase 4" in data["phase"]
    assert "health" in data["endpoints"]
    assert "analyze" in data["endpoints"]


def test_health_endpoint() -> None:
    """
    Verify that GET /health responds with 200 OK and valid health telemetry.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "DeepShield" in data["backend"]
    assert "model_loaded" in data
    assert data["model_identifier"] == "dima806/deepfake_vs_real_image_detection"
    assert "device" in data
    assert "timestamp" in data


def test_api_v1_health_endpoint() -> None:
    """
    Verify that GET /api/v1/health reports model status and architecture accurately.
    """
    # Ensure model is loaded to test full telemetry
    if not model_service.is_loaded:
        model_service.load_model()

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_identifier"] == "dima806/deepfake_vs_real_image_detection"
    assert data["model_architecture"] == "ViTForImageClassification"
    assert data["device"] in ("cpu", "cuda")
    assert "timestamp" in data
