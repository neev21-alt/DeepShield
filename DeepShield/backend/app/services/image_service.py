"""
DeepShield Image Validation & Preprocessing Service (Phase 3).

Handles:
- Upload size and MIME/format validation (JPEG, PNG, WEBP).
- Structural integrity checking via PIL stream verification.
- Consistent RGB color space conversion.
- Bounding-box adaptive face cropping with contextual padding (e.g. 20% margin
  preserving jawline, hairline, and facial perimeter artifacts).
- Coordinate clamping to prevent out-of-bounds crops.
- Empty or microscopic crop rejection.
"""

import io
import logging
from typing import List, Optional, Tuple, Union

from PIL import Image

from app.config import settings
from app.schemas.analysis import BoundingBox
from app.services.face_service import DetectedFaceData
from app.utils.image_utils import calculate_aspect_ratio_padding

logger = logging.getLogger("deepshield.image_service")

SUPPORTED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageProcessingError(Exception):
    """Base exception for image processing and validation failures."""
    pass


class ImageTooLargeError(ImageProcessingError):
    """Raised when uploaded image payload exceeds byte threshold."""
    pass


class UnsupportedImageFormatError(ImageProcessingError):
    """Raised when an uploaded file format is not supported."""
    pass


class CorruptedImageError(ImageProcessingError):
    """Raised when an image payload cannot be verified or decoded."""
    pass


class InvalidCropError(ImageProcessingError):
    """Raised when bounding box crop produces a degenerate or empty region."""
    pass


class ImageService:
    """Provides validated image decoding, RGB normalization, and boundary-padded face cropping."""

    def __init__(self, max_size_bytes: Optional[int] = None) -> None:
        self.max_size_bytes: int = (
            max_size_bytes if max_size_bytes is not None else settings.max_upload_size_bytes
        )

    def validate_and_decode(self, file_bytes: bytes) -> Image.Image:
        """
        Validate file size, verify header integrity, and decode to RGB PIL Image.

        Args:
            file_bytes: Raw uploaded bytes.

        Returns:
            Image.Image: Decoded image in RGB color space.

        Raises:
            ImageTooLargeError: If payload exceeds max size.
            CorruptedImageError: If stream is empty or invalid.
            UnsupportedImageFormatError: If format is not JPEG, PNG, or WEBP.
        """
        if not file_bytes:
            raise CorruptedImageError("Uploaded file is empty (0 bytes).")

        payload_size = len(file_bytes)
        if payload_size > self.max_size_bytes:
            limit_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = payload_size / (1024 * 1024)
            raise ImageTooLargeError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.1f} MB."
            )

        # 1. Structural verification pass
        try:
            stream = io.BytesIO(file_bytes)
            with Image.open(stream) as img:
                img.verify()
                format_name = (img.format or "").upper()
        except Exception as exc:
            raise CorruptedImageError(f"Failed to verify image integrity: {exc}") from exc

        # 2. Format validation
        if format_name not in SUPPORTED_PIL_FORMATS:
            raise UnsupportedImageFormatError(
                f"Unsupported image format: '{format_name}'. Allowed formats: JPEG, PNG, WEBP."
            )

        # 3. Decode into memory and convert to RGB
        try:
            stream.seek(0)
            img = Image.open(stream)
            img.load()  # Force load pixel data
            rgb_img = img.convert("RGB")
            return rgb_img
        except Exception as exc:
            raise CorruptedImageError(f"Failed to decode image data into RGB: {exc}") from exc

    def crop_face(
        self,
        image: Image.Image,
        bounding_box: BoundingBox,
        padding_factor: Optional[float] = None,
    ) -> Image.Image:
        """
        Crop an individual face from an image with aspect ratio padding and boundary clamping.

        Args:
            image: Full PIL Image in RGB format.
            bounding_box: Normalized facial bounding box.
            padding_factor: Contextual expansion factor (default from settings: 0.20 = 20%).

        Returns:
            Image.Image: Cropped facial crop in RGB mode.

        Raises:
            InvalidCropError: If crop dimensions are degenerate or out of bounds.
        """
        if padding_factor is None:
            padding_factor = settings.face_padding_factor

        img_w, img_h = image.size
        if img_w <= 0 or img_h <= 0:
            raise InvalidCropError("Cannot crop from empty image.")

        # Expand normalized box by padding factor and clamp to [0.0, 1.0]
        pad_x, pad_y, pad_w, pad_h = calculate_aspect_ratio_padding(
            x_min=bounding_box.x_min,
            y_min=bounding_box.y_min,
            width=bounding_box.width,
            height=bounding_box.height,
            padding_factor=padding_factor,
        )

        # Convert to integer pixel coordinates clamped strictly within image boundaries
        left = max(0, min(img_w, int(round(pad_x * img_w))))
        top = max(0, min(img_h, int(round(pad_y * img_h))))
        right = max(0, min(img_w, int(round((pad_x + pad_w) * img_w))))
        bottom = max(0, min(img_h, int(round((pad_y + pad_h) * img_h))))

        crop_w = right - left
        crop_h = bottom - top

        # Reject degenerate crops
        if crop_w < 10 or crop_h < 10:
            raise InvalidCropError(
                f"Degenerate face crop coordinates: ({left}, {top}, {right}, {bottom}) with size {crop_w}x{crop_h}."
            )

        cropped = image.crop((left, top, right, bottom))
        return cropped.convert("RGB")

    def crop_all_faces(
        self,
        image: Image.Image,
        detected_faces: List[DetectedFaceData],
        padding_factor: Optional[float] = None,
    ) -> List[Tuple[DetectedFaceData, Image.Image]]:
        """
        Crop all detected faces from the image.

        Returns:
            List of (DetectedFaceData, cropped_PIL_Image) tuples.
        """
        crops: List[Tuple[DetectedFaceData, Image.Image]] = []
        for face_data in detected_faces:
            try:
                crop = self.crop_face(image, face_data.bounding_box, padding_factor)
                crops.append((face_data, crop))
            except InvalidCropError as e:
                logger.warning(
                    "Skipping face #%d due to invalid crop coordinates: %s",
                    face_data.face_id,
                    e,
                )
        return crops


# Module singleton instance
image_service = ImageService()
