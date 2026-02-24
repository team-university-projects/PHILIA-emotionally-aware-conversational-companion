"""
audio_emotion.py — Audio-based emotion recognition.

Responsibilities:
  - Load a Wav2Vec / CNN-based speech emotion model
  - Accept raw audio bytes
  - Return a probability distribution over emotion labels

Suggested model: superb/wav2vec2-base-superb-er (HuggingFace)
"""

from __future__ import annotations

from config import Config

# Canonical emotion label set shared across all three modalities
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class AudioEmotionRecognizer:
    """Predicts emotion probabilities from raw speech audio."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        # TODO: load model and feature extractor from self._cfg.audio_emotion_model_path

    def predict(self, audio_bytes: bytes) -> dict[str, float]:
        """
        Run inference on audio bytes.

        Args:
            audio_bytes: Raw PCM/WAV bytes at 16 kHz mono.

        Returns:
            dict mapping emotion label → probability (values sum to 1.0).
            Example: {"happy": 0.6, "neutral": 0.3, "sad": 0.1, ...}
        """
        raise NotImplementedError

    def _preprocess(self, audio_bytes: bytes):
        """Convert raw bytes to model input tensor."""
        raise NotImplementedError
