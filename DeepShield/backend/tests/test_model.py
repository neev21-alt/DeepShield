"""
Automated unit and integration tests for DeepShield Model Service (Phase 2).

Verifies:
- Successful Vision Transformer model loading
- Verified class-label mapping (id2label and label2id)
- Image preprocessing and tensor normalization
- Valid inference output shapes and finite uncalibrated scores
- Error handling on corrupted, empty, or nonexistent image inputs
- Error handling when model loading fails
"""

import os
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from app.services.model_service import (
    ModelService,
    InvalidImageError,
    ModelNotLoadedError,
    model_service,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"
REAL_SAMPLE = SAMPLE_DIR / "real" / "grace_hopper_navy_real.jpg"
FAKE_SAMPLE = SAMPLE_DIR / "fake" / "sample_fake_3.jpg"


# -----------------------------------------------------------------------------
# Model Initialization & Loading Tests
# -----------------------------------------------------------------------------

def test_model_service_initial_state() -> None:
    """Verify freshly initialized ModelService state before loading."""
    fresh_service = ModelService(model_name="dima806/deepfake_vs_real_image_detection")
    assert fresh_service.is_loaded is False
    assert fresh_service.load_error is None
    assert fresh_service.device in ("cpu", "cuda")
    assert fresh_service.architecture is None


def test_model_service_load_success() -> None:
    """Verify that candidate ViT checkpoint loads and is configured properly."""
    loaded = model_service.load_model()
    assert loaded is True
    assert model_service.is_loaded is True
    assert model_service.load_error is None
    assert model_service.architecture == "ViTForImageClassification"
    assert model_service.model_type == "vit"
    assert model_service.total_parameters > 80_000_000


def test_verified_class_label_mapping() -> None:
    """
    CRITICAL TEST: Ensure class-label mapping is verified directly from checkpoint.
    Confirms class index 0 is 'Real' and class index 1 is 'Fake'.
    """
    if not model_service.is_loaded:
        model_service.load_model()

    id2label = model_service.id2label
    label2id = model_service.label2id

    assert id2label[0] == "Real", f"Expected class 0 to be 'Real', got: {id2label.get(0)}"
    assert id2label[1] == "Fake", f"Expected class 1 to be 'Fake', got: {id2label.get(1)}"
    assert label2id["Real"] == 0
    assert label2id["Fake"] == 1
    assert len(id2label) == 2


# -----------------------------------------------------------------------------
# Preprocessing Tests
# -----------------------------------------------------------------------------

def test_image_preprocessing_shapes() -> None:
    """Verify preprocessing converts inputs to tensor of shape (1, 3, 224, 224)."""
    if not model_service.is_loaded:
        model_service.load_model()

    # Test 1: From PIL Image
    dummy_pil = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor_dict = model_service.preprocess_image(dummy_pil)
    assert "pixel_values" in tensor_dict
    pixel_values = tensor_dict["pixel_values"]
    assert pixel_values.shape == (1, 3, 224, 224)
    assert pixel_values.dtype == torch.float32

    # Test 2: From Numpy Array
    dummy_np = np.zeros((150, 150, 3), dtype=np.uint8)
    tensor_dict_np = model_service.preprocess_image(dummy_np)
    assert tensor_dict_np["pixel_values"].shape == (1, 3, 224, 224)

    # Test 3: From File Path
    assert REAL_SAMPLE.is_file(), f"Sample image missing at {REAL_SAMPLE}"
    tensor_dict_file = model_service.preprocess_image(REAL_SAMPLE)
    assert tensor_dict_file["pixel_values"].shape == (1, 3, 224, 224)


# -----------------------------------------------------------------------------
# Inference & Score Verification Tests
# -----------------------------------------------------------------------------

def test_inference_output_structure_and_finite_scores() -> None:
    """
    Verify inference produces complete structured schema with finite uncalibrated scores.
    """
    if not model_service.is_loaded:
        model_service.load_model()

    result = model_service.predict(REAL_SAMPLE)

    # Verify keys
    expected_keys = {
        "predicted_label",
        "verdict",
        "raw_scores",
        "confidence",
        "model_identifier",
        "model_architecture",
        "inference_time_ms",
        "score_type",
        "device",
    }
    assert expected_keys.issubset(result.keys())

    # Verify verdict and labels
    assert result["predicted_label"] in ("Real", "Fake")
    assert result["verdict"] in ("likely_real", "likely_fake")
    assert result["score_type"] == "uncalibrated_softmax_score"
    assert result["model_architecture"] == "ViTForImageClassification"
    assert result["inference_time_ms"] > 0

    # Verify scores are finite and sum to 1.0
    scores = result["raw_scores"]
    assert "Real" in scores
    assert "Fake" in scores
    assert 0.0 <= scores["Real"] <= 1.0
    assert 0.0 <= scores["Fake"] <= 1.0
    assert torch.isfinite(torch.tensor(scores["Real"])).item()
    assert torch.isfinite(torch.tensor(scores["Fake"])).item()
    assert abs((scores["Real"] + scores["Fake"]) - 1.0) < 1e-3


def test_inference_on_known_manipulated_sample() -> None:
    """Verify inference pipeline successfully evaluates known manipulated sample."""
    if not model_service.is_loaded:
        model_service.load_model()

    assert FAKE_SAMPLE.is_file(), f"Fake sample missing at {FAKE_SAMPLE}"
    result = model_service.predict(FAKE_SAMPLE)
    assert result["predicted_label"] in ("Real", "Fake")
    assert result["confidence"] > 0.5
    assert result["inference_time_ms"] > 0


# -----------------------------------------------------------------------------
# Error Handling & Edge Case Tests
# -----------------------------------------------------------------------------

def test_invalid_image_corrupted_bytes() -> None:
    """Ensure corrupted bytes payload raises InvalidImageError."""
    corrupted_data = b"NOT_A_VALID_IMAGE_DATA_12345"
    with pytest.raises(InvalidImageError):
        model_service.preprocess_image(corrupted_data)


def test_invalid_image_empty_payload() -> None:
    """Ensure empty bytes payload raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        model_service.preprocess_image(b"")


def test_invalid_image_nonexistent_file() -> None:
    """Ensure nonexistent file path raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        model_service.preprocess_image("/tmp/nonexistent_image_file_deepshield.jpg")


def test_invalid_image_dimensions_too_small() -> None:
    """Ensure abnormally small images (<10x10) are rejected."""
    tiny_img = Image.new("RGB", (4, 4), color="red")
    with pytest.raises(InvalidImageError):
        model_service.preprocess_image(tiny_img)


def test_model_loading_error_handling() -> None:
    """Ensure invalid checkpoint identifier fails gracefully without crashing."""
    bad_service = ModelService(model_name="nonexistent/checkpoint-that-does-not-exist-123")
    success = bad_service.load_model()
    assert success is False
    assert bad_service.is_loaded is False
    assert bad_service.load_error is not None

    with pytest.raises(ModelNotLoadedError):
        bad_service.predict(REAL_SAMPLE)
