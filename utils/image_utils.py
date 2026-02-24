"""
image_utils.py — Shared image/frame processing helpers.

Responsibilities:
  - Frame resize and normalisation
  - BGR ↔ RGB conversion (OpenCV ↔ PIL/numpy)
  - Face region-of-interest (ROI) crop
"""

from __future__ import annotations

import numpy as np


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB."""
    return frame[:, :, ::-1].copy()


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a frame to (width, height) using bilinear interpolation.

    Args:
        frame:  Input image array (H, W, C).
        width:  Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized image array.
    """
    raise NotImplementedError


def crop_face_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> np.ndarray:
    """
    Crop a face bounding box from a frame with optional padding.

    Args:
        frame:   Full image array (H, W, C).
        bbox:    (x, y, w, h) bounding box.
        padding: Extra pixels to include around the bounding box.

    Returns:
        Cropped face array.
    """
    raise NotImplementedError
