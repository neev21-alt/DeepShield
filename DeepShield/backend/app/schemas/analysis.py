"""
Pydantic schemas for DeepShield deepfake analysis and health endpoints.

Provides validated models for face bounding boxes, classification scores,
Grad-CAM heatmap representations, and explainability summaries.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates of a detected face."""

    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized minimum X coordinate (left)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized minimum Y coordinate (top)")
    width: float = Field(..., ge=0.0, le=1.0, description="Normalized bounding box width")
    height: float = Field(..., ge=0.0, le=1.0, description="Normalized bounding box height")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Face detector confidence score")


class FaceAnalysisResult(BaseModel):
    """
    Detailed deepfake analysis outcome for an isolated face region.
    
    IMPORTANT: Model scores represent raw transformer output confidence values
    derived from classification logits and MUST NOT be interpreted as empirically
    calibrated posterior probabilities.
    """

    face_id: int = Field(..., description="Unique integer index of the detected face within the image")
    bounding_box: BoundingBox = Field(..., description="Spatial coordinates of the detected face region")
    verdict: str = Field(
        ...,
        description="Categorical assessment: 'likely_real', 'likely_fake', or 'uncertain'",
    )
    predicted_label: Optional[str] = Field(None, description="Model predicted label string: 'Real' or 'Fake'")
    model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Primary model confidence score for the predicted class. Not a calibrated probability.",
    )
    real_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Comparative score assigned to the genuine/real class hypothesis.",
    )
    fake_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Comparative score assigned to the synthetic/manipulated class hypothesis.",
    )
    detection_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score from face detector",
    )
    inference_time_ms: Optional[float] = Field(
        None,
        description="Model inference execution latency in milliseconds for this face crop",
    )
    heatmap_available: bool = Field(
        False,
        description="Whether a Grad-CAM heatmap was successfully generated for this face region.",
    )
    heatmap_dimensions: Optional[Dict[str, int]] = Field(
        None,
        description="Spatial dimensions (width and height in pixels) of the generated heatmap overlay.",
    )
    heatmap_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG image of the Grad-CAM saliency activation heatmap overlay.",
    )
    explanation: Optional[str] = Field(
        None,
        description="Plain-text summary explaining observed spatial cues and anomalous features.",
    )


class AnalysisResponse(BaseModel):
    """Unified payload returned upon completion of facial deepfake analysis."""

    status: str = Field(
        ...,
        description="Status of analysis: 'success', 'no_faces_detected', 'partial', or 'error'",
    )
    message: str = Field(..., description="Human-readable status description or diagnostic detail")
    faces_detected: int = Field(
        ...,
        ge=0,
        description="Total count of facial regions localized and evaluated in the image",
    )
    faces: List[FaceAnalysisResult] = Field(
        default_factory=list,
        description="List of per-face classification and explainability results",
    )
    overall_verdict: Optional[str] = Field(
        None,
        description="Aggregated verdict across all faces: 'likely_real', 'likely_fake', 'mixed', or 'uncertain'",
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="End-to-end server execution time in milliseconds",
    )
    model_identifier: str = Field(
        ...,
        description="Hugging Face checkpoint identifier used for inference",
    )


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field("ok", description="Overall service status")
    backend: str = Field("DeepShield API", description="Service name")
    version: str = Field(..., description="Application semantic version")
    model_loaded: bool = Field(
        False,
        description="Current memory status of the AI deepfake classification model. Not falsely reported as loaded.",
    )
    model_identifier: str = Field(..., description="Configured candidate model checkpoint name")
    model_architecture: Optional[str] = Field(
        None,
        description="Vision Transformer model architecture class if loaded (e.g. ViTForImageClassification)",
    )
    device: Optional[str] = Field(
        None,
        description="Compute device used for neural inference ('cpu' or 'cuda')",
    )
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp")


class ErrorDetail(BaseModel):
    """Specific error descriptor."""

    code: str = Field(..., description="Machine-readable error classification code")
    message: str = Field(..., description="Human-readable description of the error")


class ErrorResponse(BaseModel):
    """Standardized error envelope for API exceptions."""

    status: str = Field("error", description="Indicates an erroneous response")
    error: ErrorDetail = Field(..., description="Detailed diagnostic error information")
