"""
Unit tests for DeepShield Face Detection, Image Cropping, and Pipeline Logic (Phase 3).
"""

from pathlib import Path
import pytest
from PIL import Image

from app.schemas.analysis import BoundingBox
from app.services.face_service import face_service
from app.services.image_service import (
    CorruptedImageError,
    ImageTooLargeError,
    InvalidCropError,
    image_service,
)
from app.services.model_service import model_service

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REAL_SAMPLE = SAMPLES_DIR / "real" / "grace_hopper_navy_real.jpg"
MULTI_SAMPLE = SAMPLES_DIR / "real" / "multi_face_sample.jpg"
NO_FACE_SAMPLE = SAMPLES_DIR / "real" / "no_face_sample.jpg"


def test_face_detection_single_subject() -> None:
    """Ensure MediaPipe detects Grace Hopper portrait and returns valid coordinates."""
    img = Image.open(REAL_SAMPLE)
    faces = face_service.detect_faces(img)
    assert len(faces) == 1

    f0 = faces[0]
    assert f0.face_id == 1
    assert f0.confidence > 0.50
    assert 0.0 <= f0.bounding_box.x_min <= 1.0
    assert 0.0 <= f0.bounding_box.y_min <= 1.0
    assert 0.0 < f0.bounding_box.width <= 1.0
    assert 0.0 < f0.bounding_box.height <= 1.0
    assert f0.pixel_box["width"] > 50
    assert f0.pixel_box["height"] > 50


def test_face_detection_multi_subject() -> None:
    """Ensure MediaPipe detects multiple faces in a composite group image."""
    img = Image.open(MULTI_SAMPLE)
    faces = face_service.detect_faces(img)
    assert len(faces) >= 2

    # Check distinct coordinates
    xs = [f.bounding_box.x_min for f in faces]
    assert len(set(xs)) == len(faces), "Detected faces should have distinct spatial positions"


def test_face_detection_no_face() -> None:
    """Ensure MediaPipe returns an empty list for scenery/non-face images."""
    img = Image.open(NO_FACE_SAMPLE)
    faces = face_service.detect_faces(img)
    assert faces == []


def test_adaptive_face_cropping_with_padding() -> None:
    """Verify face cropping expands bounding box with padding and clamps to bounds."""
    img = Image.open(REAL_SAMPLE)
    bbox = BoundingBox(x_min=0.32, y_min=0.21, width=0.39, height=0.33, confidence=0.88)

    crop = image_service.crop_face(img, bbox, padding_factor=0.20)
    assert isinstance(crop, Image.Image)
    assert crop.mode == "RGB"
    assert crop.width > int(bbox.width * img.width)  # Padded width > unpadded width
    assert crop.height > int(bbox.height * img.height)


def test_face_crop_degenerate_box_raises_error() -> None:
    """Ensure degenerate or micro bounding boxes are rejected."""
    img = Image.open(REAL_SAMPLE)
    degenerate_bbox = BoundingBox(x_min=0.5, y_min=0.5, width=0.001, height=0.001, confidence=0.1)

    with pytest.raises(InvalidCropError):
        image_service.crop_face(img, degenerate_bbox, padding_factor=0.0)


def test_verdict_threshold_rules() -> None:
    """
    Verify threshold-based verdict logic:
    - Real score >= 0.70 -> 'likely_real'
    - Fake score >= 0.70 -> 'likely_fake'
    - Neither >= 0.70 -> 'uncertain'
    """
    if not model_service.is_loaded:
        model_service.load_model()

    img = Image.open(REAL_SAMPLE)

    # 1. Standard prediction on genuine face
    res_real = model_service.predict(img, real_threshold=0.70, fake_threshold=0.70)
    assert res_real["real_score"] >= 0.70
    assert res_real["verdict"] == "likely_real"

    # 2. Strict threshold test triggering 'uncertain'
    res_uncertain = model_service.predict(img, real_threshold=0.99999, fake_threshold=0.99999)
    assert res_uncertain["verdict"] == "uncertain"


def test_image_service_validation_errors() -> None:
    """Verify validation exceptions on empty payload or oversized uploads."""
    with pytest.raises(CorruptedImageError):
        image_service.validate_and_decode(b"")

    with pytest.raises(CorruptedImageError):
        image_service.validate_and_decode(b"not an image stream")

    huge_payload = b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(ImageTooLargeError):
        image_service.validate_and_decode(huge_payload)
