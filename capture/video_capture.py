"""
video_capture.py — Webcam frame capture.

Responsibilities:
  - Open the configured webcam device
  - Capture a single representative frame during the PTT window
  - Return a numpy array (H, W, 3) in RGB format
"""

from __future__ import annotations

import numpy as np
from config import Config


class VideoCapture:
    """Captures a single webcam frame for facial emotion analysis."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.video
        # TODO: initialise cv2.VideoCapture here

    def capture_frame(self) -> np.ndarray:
        """
        Capture one frame from the webcam.

        Returns:
            np.ndarray: RGB image array of shape (H, W, 3).
        """
        raise NotImplementedError

    def release(self) -> None:
        """Release the webcam device."""
        raise NotImplementedError
