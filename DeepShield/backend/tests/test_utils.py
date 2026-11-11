"""
Tests for DeepShield image utilities and schemas.
"""

from app.utils.image_utils import is_valid_image_extension, calculate_aspect_ratio_padding
from app.schemas.analysis import BoundingBox, FaceAnalysisResult


def test_is_valid_image_extension() -> None:
    """Check supported and unsupported file extensions."""
    assert is_valid_image_extension("photo.jpg") is True
    assert is_valid_image_extension("portrait.JPEG") is True
    assert is_valid_image_extension("sample.png") is True
    assert is_valid_image_extension("face.webp") is True
    assert is_valid_image_extension("document.pdf") is False
    assert is_valid_image_extension("script.py") is False


def test_calculate_aspect_ratio_padding() -> None:
    """Ensure bounding box expansion stays clamped within [0.0, 1.0]."""
    x, y, w, h = calculate_aspect_ratio_padding(0.2, 0.2, 0.4, 0.4, padding_factor=0.25)
    assert x >= 0.0
    assert y >= 0.0
    assert (x + w) <= 1.0
    assert (y + h) <= 1.0


def test_face_analysis_schema_validation() -> None:
    """Verify FaceAnalysisResult schema instantiates and validates field types."""
    bbox = BoundingBox(x_min=0.1, y_min=0.1, width=0.5, height=0.5, confidence=0.98)
    res = FaceAnalysisResult(
        face_id=1,
        bounding_box=bbox,
        verdict="likely_real",
        model_score=0.94,
        real_score=0.94,
        fake_score=0.06,
        heatmap_base64=None,
        explanation="High coherence in facial texture and reflection boundaries.",
    )
    assert res.verdict == "likely_real"
    assert res.model_score == 0.94
