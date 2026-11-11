"""
DeepShield MediaPipe Facial Localization Service (Phase 3).

Integrates MediaPipe BlazeFace detection graph for multi-face localization,
bounding box extraction, and spatial validation.

Responsibilities:
- Detect all human faces in single or group imagery.
- Extract normalized bounding boxes [0.0, 1.0] and pixel coordinates.
- Filter detections by confidence threshold.
- Safely handle images with zero detectable faces.
- Re-use MediaPipe detection graph across requests without per-image reinitialization.
"""

import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

from app.config import settings
from app.schemas.analysis import BoundingBox

logger = logging.getLogger("deepshield.face_service")


@dataclass
class DetectedFaceData:
    """Internal container for a detected facial region."""
    face_id: int
    bounding_box: BoundingBox
    pixel_box: Dict[str, int]  # {"x_min": int, "y_min": int, "width": int, "height": int}
    confidence: float


class FaceServiceError(Exception):
    """Base exception for face detection failures."""
    pass


class FaceService:
    """Manages MediaPipe BlazeFace detection pipeline."""

    def __init__(
        self,
        min_detection_confidence: Optional[float] = None,
        model_selection: int = 1,  # 1 for full-range (<= 5m), 0 for short-range
    ) -> None:
        self.min_detection_confidence: float = (
            min_detection_confidence
            if min_detection_confidence is not None
            else settings.min_face_detection_confidence
        )
        self.model_selection: int = model_selection
        self._detector: Optional[Any] = None
        self._lock = threading.Lock()
        self._is_initialized: bool = False

    def _get_detector(self) -> Any:
        """Thread-safe lazy initialization of MediaPipe FaceDetection instance."""
        global mp
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    if mp is None:
                        try:
                            import mediapipe as mp
                        except ImportError as err:
                            raise FaceServiceError(f"MediaPipe is not installed: {err}")
                    logger.info(
                        "Initializing MediaPipe FaceDetection (model_selection=%d, min_confidence=%.2f)...",
                        self.model_selection,
                        self.min_detection_confidence,
                    )
                    mp_face_detection = mp.solutions.face_detection
                    self._detector = mp_face_detection.FaceDetection(
                        model_selection=self.model_selection,
                        min_detection_confidence=self.min_detection_confidence,
                    )
                    self._is_initialized = True
        return self._detector

    def close(self) -> None:
        """Release MediaPipe resources."""
        with self._lock:
            if self._detector is not None:
                self._detector.close()
                self._detector = None
                self._is_initialized = False
                logger.info("MediaPipe FaceDetection graph closed.")

    @staticmethod
    def _to_rgb_numpy(
        image_input: Union[Image.Image, np.ndarray, bytes]
    ) -> Tuple[np.ndarray, int, int]:
        """Convert input image into RGB uint8 numpy array and return (array, width, height)."""
        if isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            np_arr = np.array(rgb_img)
            return np_arr, rgb_img.width, rgb_img.height

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale to RGB
                rgb_arr = np.stack([image_input] * 3, axis=-1)
                return rgb_arr, image_input.shape[1], image_input.shape[0]
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    # RGBA to RGB
                    pil_img = Image.fromarray(image_input, mode="RGBA").convert("RGB")
                    return np.array(pil_img), pil_img.width, pil_img.height
                elif image_input.shape[2] == 3:
                    return image_input.astype(np.uint8), image_input.shape[1], image_input.shape[0]
            raise FaceServiceError(f"Unsupported numpy image dimensions: {image_input.shape}")

        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise FaceServiceError("Empty image bytes payload provided.")
            try:
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                return np.array(pil_img), pil_img.width, pil_img.height
            except Exception as e:
                raise FaceServiceError(f"Failed to decode image bytes: {e}") from e

        else:
            raise FaceServiceError(f"Unsupported image input type: {type(image_input)}")

    def detect_faces(
        self, image_input: Union[Image.Image, np.ndarray, bytes]
    ) -> List[DetectedFaceData]:
        """
        Detect all human faces in the input image.

        Args:
            image_input: PIL Image, numpy array (RGB), or raw image bytes.

        Returns:
            List[DetectedFaceData]: Ordered list of detected faces with validated bounding boxes.
                                    Returns empty list if no faces are detected.
        """
        np_arr, img_width, img_height = self._to_rgb_numpy(image_input)

        if img_width < 10 or img_height < 10:
            logger.warning("Image dimensions (%dx%d) too small for face detection.", img_width, img_height)
            return []

        detector = self._get_detector()
        with self._lock:
            results = detector.process(np_arr)

        if not results.detections:
            logger.info("MediaPipe detected 0 faces in image (%dx%d).", img_width, img_height)
            return []

        detected_faces: List[DetectedFaceData] = []

        for idx, detection in enumerate(results.detections):
            score = float(detection.score[0]) if detection.score else 0.0
            loc_data = detection.location_data
            if not loc_data or not loc_data.relative_bounding_box:
                continue

            r_box = loc_data.relative_bounding_box

            # Validate and clamp normalized coordinates strictly to [0.0, 1.0]
            x_min = max(0.0, min(1.0, float(r_box.xmin)))
            y_min = max(0.0, min(1.0, float(r_box.ymin)))
            width = max(0.0, min(1.0 - x_min, float(r_box.width)))
            height = max(0.0, min(1.0 - y_min, float(r_box.height)))

            # Reject degenerate or microscopic detections
            if width <= 0.005 or height <= 0.005:
                continue

            # Compute pixel bounding box
            px_x = int(round(x_min * img_width))
            px_y = int(round(y_min * img_height))
            px_w = int(round(width * img_width))
            px_h = int(round(height * img_height))

            bounding_box = BoundingBox(
                x_min=round(x_min, 4),
                y_min=round(y_min, 4),
                width=round(width, 4),
                height=round(height, 4),
                confidence=round(score, 4),
            )

            pixel_box = {
                "x_min": px_x,
                "y_min": px_y,
                "width": px_w,
                "height": px_h,
            }

            detected_faces.append(
                DetectedFaceData(
                    face_id=idx + 1,
                    bounding_box=bounding_box,
                    pixel_box=pixel_box,
                    confidence=score,
                )
            )

        # Sort faces spatially (left to right, top to bottom) and re-index
        detected_faces.sort(key=lambda f: (f.bounding_box.x_min, f.bounding_box.y_min))
        for idx, f in enumerate(detected_faces):
            f.face_id = idx + 1

        logger.info(
            "MediaPipe localized %d valid face(s) in image (%dx%d).",
            len(detected_faces),
            img_width,
            img_height,
        )
        return detected_faces


# Module singleton instance
face_service = FaceService()
