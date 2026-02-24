"""
fusion.py — Confidence-weighted late fusion of multimodal emotion signals.

Responsibilities:
  - Accept three emotion probability dicts (audio, facial, text)
  - Apply configurable confidence weights
  - Produce a single FusedEmotion result with final label and confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from config import Config
from emotion.audio_emotion import EMOTION_LABELS


@dataclass
class FusedEmotion:
    """Result of multimodal emotion fusion."""
    label: str                          # Winning emotion label
    confidence: float                   # Confidence of winning label (0–1)
    distribution: dict[str, float]      # Full fused probability distribution
    breakdown: dict[str, dict[str, float]]  # Per-modality input distributions


class EmotionFuser:
    """
    Combines audio, facial, and text emotion distributions via
    confidence-weighted late fusion.
    """

    def __init__(self, config: Config) -> None:
        self._weights = {
            "audio":  config.fusion.audio_weight,
            "facial": config.fusion.facial_weight,
            "text":   config.fusion.text_weight,
        }

    def fuse(
        self,
        audio_probs: dict[str, float],
        facial_probs: dict[str, float],
        text_probs: dict[str, float],
    ) -> FusedEmotion:
        """
        Fuse three probability distributions into a single emotion estimate.

        Args:
            audio_probs:  Emotion probabilities from AudioEmotionRecognizer.
            facial_probs: Emotion probabilities from FacialEmotionRecognizer.
            text_probs:   Emotion probabilities from TextEmotionRecognizer.

        Returns:
            FusedEmotion with winning label, confidence, and full breakdown.
        """
        raise NotImplementedError

    def _weighted_sum(
        self,
        distributions: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Compute weighted sum across modality distributions.

        Args:
            distributions: {"audio": {...}, "facial": {...}, "text": {...}}

        Returns:
            Merged dict of emotion → weighted probability.
        """
        raise NotImplementedError
