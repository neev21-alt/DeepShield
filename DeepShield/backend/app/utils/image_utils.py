"""
Image utility routines for DeepShield.

Provides format checking, geometric aspect ratio calculations,
and coordinate boundary clamping.
"""

from typing import Tuple
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_valid_image_extension(filename: str) -> bool:
    """
    Check if a filename ends with a supported image extension.
    
    Args:
        filename: Filename or path string to inspect.
        
    Returns:
        True if the file extension is supported, False otherwise.
    """
    suffix = Path(filename).suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def calculate_aspect_ratio_padding(
    x_min: float,
    y_min: float,
    width: float,
    height: float,
    padding_factor: float = 0.20,
) -> Tuple[float, float, float, float]:
    """
    Expand a normalized bounding box by a contextual padding margin and clamp to [0.0, 1.0].
    
    Preserving the facial perimeter (chin, hairline, ears) is critical for identifying
    blending seams and warping artifacts characteristic of deepfakes.

    Args:
        x_min: Normalized left boundary (0.0 to 1.0).
        y_min: Normalized top boundary (0.0 to 1.0).
        width: Normalized width.
        height: Normalized height.
        padding_factor: Expansion ratio (default 0.20 = 20% expansion).

    Returns:
        Clamped tuple of (new_x_min, new_y_min, new_width, new_height).
    """
    pad_x = width * padding_factor
    pad_y = height * padding_factor

    new_x = max(0.0, x_min - pad_x)
    new_y = max(0.0, y_min - pad_y)
    
    new_right = min(1.0, (x_min + width) + pad_x)
    new_bottom = min(1.0, (y_min + height) + pad_y)

    return (new_x, new_y, new_right - new_x, new_bottom - new_y)
