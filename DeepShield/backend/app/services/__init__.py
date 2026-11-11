"""
DeepShield Core Services Package.

Modular service layer isolating domain responsibilities:
- model_service: Hugging Face transformer model management & inference
- face_service: MediaPipe BlazeFace detection and alignment
- image_service: Image validation, format conversion, and boundary padding
- gradcam_service: Vision Transformer attention gradient saliency maps
- explanation_service: Narrative explainability generation
"""

__all__ = [
    "model_service",
    "face_service",
    "image_service",
    "gradcam_service",
    "explanation_service",
]
