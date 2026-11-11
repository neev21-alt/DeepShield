"""
Unit and integration tests for Vision Transformer Grad-CAM Explainability (Phase 4).
"""

import base64
import io
import os
import numpy as np
import pytest
import torch
from PIL import Image

from app.services.gradcam_service import (
    GradCAMService,
    generate_gradcam_explanation,
    gradcam_service,
)
from app.services.model_service import model_service

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
REAL_SAMPLE = os.path.join(SAMPLE_DIR, "real", "grace_hopper_navy_real.jpg")
FAKE_SAMPLE = os.path.join(SAMPLE_DIR, "fake", "sample_fake_3.jpg")


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """Ensure ViT model is loaded before running Grad-CAM tests."""
    if not model_service.is_loaded:
        success = model_service.load_model()
        assert success is True, f"Model failed to load: {model_service.load_error}"


def test_gradcam_explanation_phrasing():
    """Verify explainability strings are scientifically grounded and legally cautious."""
    fake_exp = generate_gradcam_explanation(verdict="likely_fake", predicted_label="Fake")
    assert "does not prove manipulation" in fake_exp
    assert "Authentic characteristics confirmed" not in fake_exp
    assert "Manipulation definitively proven" not in fake_exp

    real_exp = generate_gradcam_explanation(verdict="likely_real", predicted_label="Real")
    assert "does not definitively prove" in real_exp
    assert "AI artifacts detected" not in real_exp

    uncertain_exp = generate_gradcam_explanation(verdict="uncertain", predicted_label="Unknown")
    assert "caution" in uncertain_exp.lower()


def test_gradcam_target_layer_identification():
    """Verify target layer resolution identifies the final transformer encoder layernorm_before."""
    target_layer = gradcam_service._get_target_layer(model_service._model)
    assert target_layer is not None
    assert isinstance(target_layer, torch.nn.LayerNorm)


def test_gradcam_hook_lifecycle_and_cleanup():
    """Verify hooks are removed and model gradients zeroed after Grad-CAM execution."""
    target_layer = gradcam_service._get_target_layer(model_service._model)
    
    # Check baseline hook count
    initial_fwd_hooks = len(target_layer._forward_hooks)
    initial_bwd_hooks = len(target_layer._backward_hooks)

    test_img = Image.new("RGB", (128, 128), color=(120, 150, 180))
    result = gradcam_service.generate_heatmap(face_image=test_img, target_class=1)

    # Hooks must be completely detached
    assert len(target_layer._forward_hooks) == initial_fwd_hooks
    assert len(target_layer._backward_hooks) == initial_bwd_hooks

    # Gradients on model parameters must be None or zeroed
    for param in model_service._model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0).item() is True


def test_gradcam_real_sample():
    """Test Grad-CAM heatmap generation on real face sample."""
    assert os.path.exists(REAL_SAMPLE), f"Sample image not found: {REAL_SAMPLE}"
    img = Image.open(REAL_SAMPLE).convert("RGB")
    face_crop = img.crop((50, 50, 200, 200))

    result = gradcam_service.generate_heatmap(
        face_image=face_crop,
        target_class=0,
        verdict="likely_real",
    )

    assert result.heatmap_available is True
    assert result.heatmap_base64 is not None
    assert result.heatmap_dimensions == {"width": face_crop.width, "height": face_crop.height}
    assert result.grid_size == (14, 14)
    assert "genuine" in result.explanation or "contributed" in result.explanation

    # Decode and verify base64 PNG
    png_bytes = base64.b64decode(result.heatmap_base64)
    pil_overlay = Image.open(io.BytesIO(png_bytes))
    assert pil_overlay.format == "PNG"
    assert pil_overlay.size == face_crop.size


def test_gradcam_fake_sample():
    """Test Grad-CAM heatmap generation on synthetic/fake face sample."""
    assert os.path.exists(FAKE_SAMPLE), f"Sample image not found: {FAKE_SAMPLE}"
    img = Image.open(FAKE_SAMPLE).convert("RGB")
    face_crop = img.crop((20, 20, 180, 180))

    result = gradcam_service.generate_heatmap(
        face_image=face_crop,
        target_class=1,
        verdict="likely_fake",
    )

    assert result.heatmap_available is True
    assert result.heatmap_base64 is not None
    assert result.heatmap_dimensions == {"width": face_crop.width, "height": face_crop.height}
    assert "manipulated" in result.explanation or "contributed" in result.explanation

    # Ensure min < max or valid normalized range
    assert result.raw_cam_min <= result.raw_cam_max


def test_gradcam_tiny_crop_fallback():
    """Verify tiny images (< 10x10) safely return fallback without crashing."""
    tiny_img = Image.new("RGB", (5, 5), color=(200, 100, 50))
    result = gradcam_service.generate_heatmap(face_image=tiny_img)

    assert result.heatmap_available is False
    assert result.heatmap_base64 is None
    assert result.heatmap_dimensions is None
    assert "too small" in result.explanation.lower()
