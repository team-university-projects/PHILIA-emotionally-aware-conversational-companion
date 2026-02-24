"""
text_emotion.py — Text-based emotion recognition.

Responsibilities:
  - Load a Transformer-based text emotion classifier
  - Accept a transcript string
  - Return a probability distribution over emotion labels

Suggested model: j-hartmann/emotion-english-distilroberta-base
"""

from __future__ import annotations

from config import Config
from emotion.audio_emotion import EMOTION_LABELS


class TextEmotionRecognizer:
    """Predicts emotion probabilities from a text transcript."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        # TODO: load pipeline from self._cfg.text_emotion_model_name
        # e.g. transformers.pipeline("text-classification", model=..., top_k=None)

    def predict(self, transcript: str) -> dict[str, float]:
        """
        Run inference on a text string.

        Args:
            transcript: Transcribed user speech as a plain string.

        Returns:
            dict mapping emotion label → probability (values sum to 1.0).
            Returns neutral distribution for empty/very short transcripts.
        """
        raise NotImplementedError
