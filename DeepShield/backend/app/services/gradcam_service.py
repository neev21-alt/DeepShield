"""
DeepShield Vision Transformer Grad-CAM Explainability Service (Phase 4).

Generates gradient-weighted class activation mapping (Grad-CAM) visualizations
for Vision Transformer (ViT) classification models.

Mathematical & Architectural Formulation:
1. Target Layer: `model.vit.encoder.layer[-1].layernorm_before`.
   - ViT pools exclusively from the [CLS] token (token 0) for classification.
   - Spatial patch tokens after the final encoder layernorm have zero gradient with respect
     to the classification head because they bypass the pooled output.
   - Hooking at `layernorm_before` of the final (12th) transformer block captures the layer
     where the [CLS] token attends across all spatial patch tokens, yielding rich,
     informative gradient signals from the target class logit.
2. Token Decomposition:
   - Total tokens: 197 = 1 [CLS] token + 196 spatial patch tokens.
   - [CLS] token at index 0 is excluded.
   - 196 patch tokens correspond to a 14x14 grid (224 / 16 = 14).
3. Gradient Weighting:
   - Patch activations: A in R^(1, 196, 768) -> spatial A in R^(1, 768, 14, 14).
   - Patch gradients:   G in R^(1, 196, 768) -> spatial G in R^(1, 768, 14, 14).
   - Channel weights: alpha_k = (1 / (14*14)) * sum_{h,w} G_{k, h, w}.
   - Saliency map:    CAM = ReLU( sum_k alpha_k * A_{k, h, w} ).
4. Spatial Interpolation & Blending:
   - Saliency map normalized to [0.0, 1.0].
   - Scaled to original face crop dimensions via bicubic interpolation.
   - Colorized via OpenCV COLORMAP_JET and alpha-blended with the facial crop.
   - Encoded as Base64 PNG.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from app.services.model_service import ModelService, model_service

logger = logging.getLogger("deepshield.gradcam")


@dataclass
class GradCAMResult:
    """Structured container for Grad-CAM explainability outputs."""
    heatmap_available: bool
    heatmap_base64: Optional[str]
    heatmap_dimensions: Optional[Dict[str, int]]
    target_class: Optional[int]
    target_label: Optional[str]
    explanation: str
    grid_size: Tuple[int, int] = (14, 14)
    raw_cam_min: float = 0.0
    raw_cam_max: float = 0.0
    error: Optional[str] = None


def generate_gradcam_explanation(verdict: str, predicted_label: str) -> str:
    """
    Generate scientifically grounded, legally cautious explainability text.
    Strictly avoids unsupported claims of definitive authenticity or artifact detection.
    """
    verdict_lower = (verdict or "").lower()
    label_lower = (predicted_label or "").lower()

    if "fake" in verdict_lower or "fake" in label_lower:
        return (
            "The model predicts this face is likely manipulated. The Grad-CAM visualization highlights "
            "image regions that contributed to this prediction. It does not prove manipulation or "
            "provide exact forensic localization."
        )
    elif "real" in verdict_lower or "real" in label_lower:
        return (
            "The model predicts this face is likely genuine. The Grad-CAM visualization highlights "
            "image regions that contributed to this prediction. It does not definitively prove "
            "authenticity or absence of manipulation."
        )
    else:
        return (
            "The model prediction is uncertain. The Grad-CAM visualization highlights image regions "
            "with the highest relative activation. These cues should be interpreted with caution."
        )


class GradCAMService:
    """Computes and renders ViT-compatible Grad-CAM activation heatmaps."""

    def __init__(self, service: Optional[ModelService] = None) -> None:
        self.model_service: ModelService = service or model_service

    def _get_target_layer(self, model: Any) -> Any:
        """
        Locate the verified target layer for Vision Transformer Grad-CAM.
        Supports both transformers v4 (vit.encoder.layer) and v5 (vit.layers).
        """
        try:
            if hasattr(model, "vit"):
                vit = model.vit
            else:
                vit = model
            if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
                return vit.encoder.layer[-1].layernorm_before
            elif hasattr(vit, "layers"):
                return vit.layers[-1].layernorm_before
            raise AttributeError("ViT model has neither encoder.layer nor layers attribute")
        except AttributeError as e:
            raise RuntimeError(f"Unable to resolve target layer in ViT architecture: {e}") from e

    def generate_heatmap(
        self,
        face_image: Image.Image,
        target_class: Optional[int] = None,
        verdict: str = "uncertain",
        alpha: float = 0.45,
    ) -> GradCAMResult:
        """
        Compute a Grad-CAM heatmap overlay for an individual face crop.

        Args:
            face_image: Cropped facial region as an RGB PIL Image.
            target_class: Optional class index to explain (0: Real, 1: Fake).
                          If None, explains the predicted class.
            verdict: Categorical verdict ('likely_real', 'likely_fake', 'uncertain').
            alpha: Heatmap blend opacity factor (default 0.45 = 45% heatmap, 55% original face).

        Returns:
            GradCAMResult: Structured heatmap payload with Base64 overlay, dimensions, and explanation.
        """
        if not self.model_service.is_loaded:
            loaded = self.model_service.load_model()
            if not loaded:
                return GradCAMResult(
                    heatmap_available=False,
                    heatmap_base64=None,
                    heatmap_dimensions=None,
                    target_class=None,
                    target_label=None,
                    explanation="Grad-CAM visualization is unavailable because the model could not be loaded.",
                    error=self.model_service.load_error,
                )

        model = self.model_service._model
        processor = self.model_service._processor

        if model is None or processor is None:
            return GradCAMResult(
                heatmap_available=False,
                heatmap_base64=None,
                heatmap_dimensions=None,
                target_class=None,
                target_label=None,
                explanation="Grad-CAM visualization is unavailable because model components are uninitialized.",
                error="Model or processor uninitialized",
            )

        orig_w, orig_h = face_image.size
        if orig_w < 10 or orig_h < 10:
            return GradCAMResult(
                heatmap_available=False,
                heatmap_base64=None,
                heatmap_dimensions=None,
                target_class=None,
                target_label=None,
                explanation="Face crop dimensions are too small to generate a reliable Grad-CAM heatmap.",
                error=f"Crop dimensions ({orig_w}x{orig_h}) too small",
            )

        # Hook storage
        hook_storage: Dict[str, Optional[torch.Tensor]] = {
            "activations": None,
            "gradients": None,
        }

        def forward_hook(module: Any, input_tensor: Any, output_tensor: Any) -> None:
            hook_storage["activations"] = output_tensor

        def backward_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            hook_storage["gradients"] = grad_output[0]

        target_layer = self._get_target_layer(model)
        h_forward = target_layer.register_forward_hook(forward_hook)
        h_backward = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Preprocess crop to 224x224 RGB tensor
            rgb_crop = face_image.convert("RGB")
            inputs = processor(images=rgb_crop, return_tensors="pt")
            device = self.model_service.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"].requires_grad = True

            # Enable gradient calculation for Grad-CAM
            with torch.enable_grad():
                model.zero_grad()
                outputs = model(**inputs)
                logits = outputs.logits

                if target_class is None:
                    target_class = int(torch.argmax(logits, dim=-1).item())

                target_label = self.model_service.id2label.get(target_class, f"Class_{target_class}")
                target_score = logits[0, target_class]
                target_score.backward()

            activations = hook_storage["activations"]
            gradients = hook_storage["gradients"]

            if activations is None or gradients is None:
                raise RuntimeError("Failed to capture activations or gradients from target ViT layer.")

            # Move to CPU for post-processing
            act_cpu = activations.detach().cpu()
            grad_cpu = gradients.detach().cpu()

            # Exclude CLS token at index 0 (shape: [1, 196, 768])
            patch_act = act_cpu[:, 1:, :]
            patch_grad = grad_cpu[:, 1:, :]

            batch_size, num_patches, hidden_dim = patch_act.shape
            grid_h = grid_w = int(num_patches ** 0.5)
            if grid_h * grid_w != num_patches:
                raise ValueError(
                    f"Patch count ({num_patches}) does not form a perfect square spatial grid."
                )

            # Reshape tokens to 2D spatial feature map: (batch, hidden_dim, grid_h, grid_w)
            spatial_act = patch_act.permute(0, 2, 1).reshape(batch_size, hidden_dim, grid_h, grid_w)
            spatial_grad = patch_grad.permute(0, 2, 1).reshape(batch_size, hidden_dim, grid_h, grid_w)

            # Global average pooling over spatial dimensions: alpha_k in R^(1, 768, 1, 1)
            weights = torch.mean(spatial_grad, dim=(2, 3), keepdim=True)

            # Linear combination: CAM = sum_k (alpha_k * A_k)
            cam = torch.sum(weights * spatial_act, dim=1, keepdim=True)
            cam = torch.relu(cam)[0, 0].numpy()

            raw_min = float(cam.min())
            raw_max = float(cam.max())
            diff = raw_max - raw_min

            # Handle constant or zero activation maps
            if diff < 1e-7:
                logger.warning("Grad-CAM produced a constant/flat heatmap (diff=%.8f).", diff)
                norm_cam = np.zeros_like(cam, dtype=np.float32)
            else:
                norm_cam = ((cam - raw_min) / diff).astype(np.float32)

            # Interpolate 14x14 heatmap to match the original face crop dimensions (orig_w, orig_h)
            resized_cam = cv2.resize(norm_cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            resized_cam = np.clip(resized_cam, 0.0, 1.0)

            # Apply JET colormap (convert BGR to RGB)
            cam_uint8 = np.uint8(255.0 * resized_cam)
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Composite alpha overlay onto original face crop
            face_np = np.array(rgb_crop)
            overlay = cv2.addWeighted(face_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

            # Encode blended overlay to PNG Base64
            pil_overlay = Image.fromarray(overlay)
            buffer = io.BytesIO()
            pil_overlay.save(buffer, format="PNG")
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            explanation_text = generate_gradcam_explanation(verdict, target_label)

            return GradCAMResult(
                heatmap_available=True,
                heatmap_base64=b64_str,
                heatmap_dimensions={"width": orig_w, "height": orig_h},
                target_class=target_class,
                target_label=target_label,
                explanation=explanation_text,
                grid_size=(grid_h, grid_w),
                raw_cam_min=raw_min,
                raw_cam_max=raw_max,
            )

        except Exception as exc:
            logger.exception("Grad-CAM generation failed: %s", exc)
            return GradCAMResult(
                heatmap_available=False,
                heatmap_base64=None,
                heatmap_dimensions=None,
                target_class=target_class,
                target_label=None,
                explanation="Grad-CAM visualization generation failed for this facial region.",
                error=str(exc),
            )

        finally:
            # Guarantee hook removal and zero_grad to prevent memory retention
            h_forward.remove()
            h_backward.remove()
            if model is not None:
                model.zero_grad()
            hook_storage["activations"] = None
            hook_storage["gradients"] = None


# Module singleton instance
gradcam_service = GradCAMService()
