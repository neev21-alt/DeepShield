"""
End-to-end integration tests for DeepShield /analyze and /api/v1/analyze endpoints (Phase 3).

Verifies:
- Single face analysis on authentic image
- Multiple faces processed independently in a single upload
- No-face image handling ('no_faces_detected')
- Missing file payload validation (422)
- Corrupted image rejection (400)
- Unsupported format rejection (400)
- Oversized payload rejection (>10MB -> 413)
- Backwards-compatible /analyze and /api/v1/analyze route parity
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REAL_SAMPLE = SAMPLES_DIR / "real" / "grace_hopper_navy_real.jpg"
MULTI_SAMPLE = SAMPLES_DIR / "real" / "multi_face_sample.jpg"
NO_FACE_SAMPLE = SAMPLES_DIR / "real" / "no_face_sample.jpg"


def test_analyze_requires_file() -> None:
    """Verify that omitting the file payload returns 422 Unprocessable Entity."""
    response = client.post("/api/v1/analyze")
    assert response.status_code == 422


def test_analyze_valid_single_face_endpoint() -> None:
    """
    Verify complete analysis pipeline on authentic single-subject facial portrait.
    Asserts detection of 1 face, valid bounding box, likely_real verdict, and latency telemetry.
    """
    assert REAL_SAMPLE.is_file(), f"Real test sample missing: {REAL_SAMPLE}"

    with open(REAL_SAMPLE, "rb") as f:
        files = {"file": ("grace_hopper.jpg", f.read(), "image/jpeg")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["faces_detected"] == 1
    assert len(data["faces"]) == 1

    face = data["faces"][0]
    assert face["face_id"] == 1
    assert face["verdict"] == "likely_real"
    assert face["predicted_label"] == "Real"
    assert face["real_score"] >= 0.70
    assert face["fake_score"] <= 0.30
    assert face["detection_confidence"] is not None
    assert face["detection_confidence"] > 0.50
    assert face["inference_time_ms"] is not None
    assert face["inference_time_ms"] > 0

    # Phase 4 Grad-CAM explainability assertions
    assert face["heatmap_available"] is True
    assert face["heatmap_dimensions"] is not None
    assert face["heatmap_dimensions"]["width"] > 0
    assert face["heatmap_dimensions"]["height"] > 0
    assert face["heatmap_base64"] is not None
    assert len(face["heatmap_base64"]) > 100
    assert face["explanation"] is not None
    assert "Grad-CAM" in face["explanation"] or "model" in face["explanation"]

    # Bounding box assertions
    bbox = face["bounding_box"]
    assert 0.0 <= bbox["x_min"] <= 1.0
    assert 0.0 <= bbox["y_min"] <= 1.0
    assert 0.0 < bbox["width"] <= 1.0
    assert 0.0 < bbox["height"] <= 1.0

    # Response envelope metadata
    assert data["overall_verdict"] == "likely_real"
    assert data["processing_time_ms"] > 0
    assert "dima806" in data["model_identifier"]


def test_analyze_multiple_faces_processed_independently() -> None:
    """
    Verify that an image with multiple subjects localizes and evaluates each face independently.
    """
    assert MULTI_SAMPLE.is_file(), f"Multi-face test sample missing: {MULTI_SAMPLE}"

    with open(MULTI_SAMPLE, "rb") as f:
        files = {"file": ("multi_face.jpg", f.read(), "image/jpeg")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["faces_detected"] >= 2
    assert len(data["faces"]) >= 2

    # Verify each face has a unique ID and independent score evaluation
    face_ids = [f["face_id"] for f in data["faces"]]
    assert len(face_ids) == len(set(face_ids)), "Face IDs must be unique"

    for face in data["faces"]:
        assert face["bounding_box"]["width"] > 0
        assert face["bounding_box"]["height"] > 0
        assert face["verdict"] in ("likely_real", "likely_fake", "uncertain")
        assert 0.0 <= face["real_score"] <= 1.0
        assert 0.0 <= face["fake_score"] <= 1.0
        # Phase 4 Grad-CAM assertions per face
        assert face["heatmap_available"] is True
        assert face["heatmap_base64"] is not None
        assert face["heatmap_dimensions"] is not None
        assert face["explanation"] is not None


def test_analyze_no_face_detected_response() -> None:
    """
    Verify that an image with zero faces cleanly returns 'no_faces_detected' status
    without crashing or returning false detections.
    """
    assert NO_FACE_SAMPLE.is_file(), f"No-face test sample missing: {NO_FACE_SAMPLE}"

    with open(NO_FACE_SAMPLE, "rb") as f:
        files = {"file": ("scenery.jpg", f.read(), "image/jpeg")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "no_faces_detected"
    assert data["faces_detected"] == 0
    assert data["faces"] == []
    assert data["overall_verdict"] is None
    assert "No human faces" in data["message"]


def test_analyze_corrupted_image_returns_400() -> None:
    """Verify that submitting corrupted byte payload returns HTTP 400 Bad Request."""
    corrupted_bytes = b"CORRUPTED_HEADER_NOT_A_VALID_JPEG_STREAM_123456"
    files = {"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["code"] == "CORRUPTED_IMAGE"


def test_analyze_unsupported_format_returns_400() -> None:
    """Verify that uploading an unsupported file format (e.g. text/binary) returns 400."""
    fake_doc = b"%PDF-1.4 sample pdf document header"
    files = {"file": ("document.pdf", fake_doc, "application/pdf")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["code"] == "CORRUPTED_IMAGE" or data["detail"]["code"] == "UNSUPPORTED_FORMAT"


def test_analyze_file_exceeding_10mb_limit_returns_413() -> None:
    """Verify that uploading a payload exceeding 10MB returns HTTP 413 Payload Too Large."""
    # 10.5 MB oversized payload
    oversized_bytes = b"\x00" * int(10.5 * 1024 * 1024)
    files = {"file": ("huge_image.jpg", oversized_bytes, "image/jpeg")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 413
    data = response.json()
    assert data["detail"]["code"] == "FILE_TOO_LARGE"


def test_analyze_route_alias_parity() -> None:
    """Verify that POST /analyze and POST /api/v1/analyze both work identically."""
    with open(REAL_SAMPLE, "rb") as f:
        content = f.read()

    res_root = client.post("/analyze", files={"file": ("grace.jpg", content, "image/jpeg")})
    res_v1 = client.post("/api/v1/analyze", files={"file": ("grace.jpg", content, "image/jpeg")})

    assert res_root.status_code == 200
    assert res_v1.status_code == 200
    assert res_root.json()["faces_detected"] == res_v1.json()["faces_detected"]
