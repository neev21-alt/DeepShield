"""
Facial deepfake analysis endpoints for DeepShield (Phase 3 Pipeline).

Provides POST /api/v1/analyze and POST /analyze:
- Safe image upload ingestion (max 10MB limit, stream validation).
- MediaPipe BlazeFace detection for single and multiple subjects.
- Adaptive face cropping with boundary padding and clamping.
- Independent Vision Transformer deepfake inference per face.
- Structured response payloads with uncalibrated softmax scores and verdicts.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas.analysis import (
    AnalysisResponse,
    BoundingBox,
    ErrorResponse,
    FaceAnalysisResult,
)
from app.services.face_service import face_service
from app.services.image_service import (
    CorruptedImageError,
    ImageTooLargeError,
    InvalidCropError,
    UnsupportedImageFormatError,
    image_service,
)
from app.services.gradcam_service import gradcam_service
from app.services.model_service import ModelNotLoadedError, model_service

logger = logging.getLogger("deepshield.analyze")

router = APIRouter(tags=["Analysis"])


def _aggregate_verdicts(faces: List[FaceAnalysisResult]) -> str:
    """Determine the aggregated overall verdict across all detected faces."""
    if not faces:
        return "no_faces_detected"

    verdicts = {f.verdict for f in faces}

    if verdicts == {"likely_real"}:
        return "likely_real"
    elif verdicts == {"likely_fake"}:
        return "likely_fake"
    elif "likely_fake" in verdicts and "likely_real" in verdicts:
        return "mixed"
    elif "likely_fake" in verdicts:
        return "likely_fake"
    elif "likely_real" in verdicts and "uncertain" in verdicts:
        return "uncertain"
    else:
        return "uncertain"


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Image for Facial Deepfakes",
    description="Evaluates single or multiple faces in an uploaded image for synthetic manipulation.",
)
@router.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="API v1 Analyze Image for Facial Deepfakes",
    description="Evaluates single or multiple faces in an uploaded image for synthetic manipulation.",
)
async def analyze_image(
    file: UploadFile = File(..., description="Facial image file (JPEG, PNG, or WEBP, max 10MB)"),
) -> AnalysisResponse:
    """
    Ingest image and evaluate all localized facial regions for synthetic manipulation.

    Workflow:
    1. Read and validate upload byte size (max 10MB).
    2. Verify image integrity and decode to RGB PIL Image.
    3. Run MediaPipe face detection.
    4. If no faces are found, return 'no_faces_detected' response.
    5. For each detected face, crop with contextual boundary padding.
    6. Execute ViT classification inference on each crop independently.
    7. Return unified dossier with per-face bounding boxes, scores, and verdicts.
    """
    start_total_time = time.perf_counter()

    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_FILE",
                "message": "No file was uploaded. Please provide a valid JPEG, PNG, or WEBP image.",
            },
        )

    # 1. Read file bytes safely with size check
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Error reading uploaded stream: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "READ_ERROR",
                "message": f"Failed to read file payload: {e}",
            },
        )

    # 2. Validate and decode into RGB PIL Image
    try:
        pil_image = image_service.validate_and_decode(content)
    except ImageTooLargeError as e:
        logger.warning("Upload size rejected: %s", e)
        raise HTTPException(
            status_code=(
                status.HTTP_413_CONTENT_TOO_LARGE
                if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE")
                else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail={"code": "FILE_TOO_LARGE", "message": str(e)},
        )
    except UnsupportedImageFormatError as e:
        logger.warning("Unsupported format rejected: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "UNSUPPORTED_FORMAT", "message": str(e)},
        )
    except CorruptedImageError as e:
        logger.warning("Corrupted image rejected: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CORRUPTED_IMAGE", "message": str(e)},
        )

    # 3. Ensure deepfake model is loaded in memory
    if not model_service.is_loaded:
        model_ready = model_service.load_model()
        if not model_ready:
            logger.error("AI Model failed to load: %s", model_service.load_error)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "MODEL_UNAVAILABLE",
                    "message": f"Deepfake detection model unavailable: {model_service.load_error}",
                },
            )

    # 4. MediaPipe Face Localization
    try:
        detected_faces = face_service.detect_faces(pil_image)
    except Exception as e:
        logger.exception("Face detection graph failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "FACE_DETECTION_ERROR",
                "message": f"Facial detection error: {e}",
            },
        )

    # 5. Handle images without detectable faces
    if not detected_faces:
        elapsed_ms = round((time.perf_counter() - start_total_time) * 1000.0, 2)
        return AnalysisResponse(
            status="no_faces_detected",
            message="No human faces were localized in the image. Please upload a clear portrait.",
            faces_detected=0,
            faces=[],
            overall_verdict=None,
            processing_time_ms=elapsed_ms,
            model_identifier=model_service.model_name,
        )

    # 6. Adaptive Face Cropping and Independent ViT Inference
    face_results: List[FaceAnalysisResult] = []

    for face_data in detected_faces:
        try:
            face_crop = image_service.crop_face(
                image=pil_image,
                bounding_box=face_data.bounding_box,
                padding_factor=settings.face_padding_factor,
            )
        except InvalidCropError as e:
            logger.warning("Face #%d crop failed: %s", face_data.face_id, e)
            continue

        # Run model inference for this facial crop
        try:
            pred = model_service.predict(
                image_input=face_crop,
                real_threshold=settings.real_threshold,
                fake_threshold=settings.fake_threshold,
            )
        except Exception as e:
            logger.error("Inference failed for face #%d: %s", face_data.face_id, e)
            continue

        # Generate Grad-CAM explainability visualization for this face crop
        target_idx = 1 if pred["predicted_label"] == "Fake" else 0
        try:
            cam_res = gradcam_service.generate_heatmap(
                face_image=face_crop,
                target_class=target_idx,
                verdict=pred["verdict"],
            )
        except Exception as e:
            logger.warning("Grad-CAM generation failed for face #%d: %s", face_data.face_id, e)
            cam_res = None

        face_result = FaceAnalysisResult(
            face_id=face_data.face_id,
            bounding_box=face_data.bounding_box,
            verdict=pred["verdict"],
            predicted_label=pred["predicted_label"],
            model_score=pred["model_score"],
            real_score=pred["real_score"],
            fake_score=pred["fake_score"],
            detection_confidence=face_data.confidence,
            inference_time_ms=pred["inference_time_ms"],
            heatmap_available=cam_res.heatmap_available if cam_res else False,
            heatmap_dimensions=cam_res.heatmap_dimensions if cam_res else None,
            heatmap_base64=cam_res.heatmap_base64 if cam_res else None,
            explanation=cam_res.explanation if cam_res else None,
        )
        face_results.append(face_result)

    # 7. Aggregate results
    overall = _aggregate_verdicts(face_results)
    elapsed_ms = round((time.perf_counter() - start_total_time) * 1000.0, 2)

    return AnalysisResponse(
        status="success",
        message=f"Successfully analyzed {len(face_results)} face(s).",
        faces_detected=len(detected_faces),
        faces=face_results,
        overall_verdict=overall,
        processing_time_ms=elapsed_ms,
        model_identifier=model_service.model_name,
    )
