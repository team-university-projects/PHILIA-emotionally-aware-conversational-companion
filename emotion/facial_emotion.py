"""
facial_emotion.py — Facial expression emotion recognition.

Responsibilities:
  - Load a CNN/ResNet-based facial emotion model (e.g. FER, DeepFace)
  - Accept a webcam frame (numpy array)
  - Detect face region, run inference
  - Return a probability distribution over emotion labels
"""

from __future__ import annotations

import numpy as np
from config import Config
from emotion.audio_emotion import EMOTION_LABELS


class FacialEmotionRecognizer:
    """Predicts emotion probabilities from a webcam frame."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        # TODO: load model from self._cfg.facial_emotion_model_path

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """
        Run inference on an RGB frame.

        Args:
            frame: np.ndarray of shape (H, W, 3) in RGB.

        Returns:
            dict mapping emotion label → probability (values sum to 1.0).
            Returns uniform distribution if no face is detected.
        """
        raise NotImplementedError

    def _detect_face(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Detect and crop the primary face region from the frame.

        Returns:
            Cropped face array, or None if no face found.
        """
        raise NotImplementedError

    def _preprocess(self, face_crop: np.ndarray):
        """Resize and normalise face crop for model input."""
        raise NotImplementedError
