"""
tone_mapper.py — Maps fused emotion to response tone and TTS parameters.

Responsibilities:
  - Accept a FusedEmotion result
  - Map the emotion label to a human-readable tone descriptor
  - Derive concrete TTSParams (pitch, rate, style) for TTS synthesis
"""

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from speech.tts import TTSParams


# Tone mapping table: emotion label → (tone descriptor, pitch, rate, style)
_TONE_MAP: dict[str, tuple[str, float, int, str]] = {
    #           tone         pitch  rate  style
    "angry":   ("firm",      0.95,  160,  "assertive"),
    "disgust": ("measured",  1.00,  155,  "neutral"),
    "fear":    ("reassuring",1.05,  150,  "calm"),
    "happy":   ("energetic", 1.10,  190,  "cheerful"),
    "neutral": ("balanced",  1.00,  175,  "neutral"),
    "sad":     ("gentle",    0.90,  145,  "soft"),
    "surprise":("engaged",   1.08,  180,  "curious"),
}


class ToneMapper:
    """Converts a fused emotion label into a tone descriptor and TTSParams."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.tts

    def map(self, fused: FusedEmotion) -> tuple[str, TTSParams]:
        """
        Map fused emotion to tone + TTS parameters.

        Args:
            fused: FusedEmotion from the emotion fusion step.

        Returns:
            Tuple of (tone_descriptor: str, TTSParams).
        """
        tone_desc, pitch, rate, style = _TONE_MAP.get(
            fused.label,
            ("balanced", 1.00, 175, "neutral"),  # fallback
        )
        params = TTSParams(
            rate=rate,
            pitch=pitch,
            volume=self._cfg.default_volume,
            style=style,
        )
        return tone_desc, params
