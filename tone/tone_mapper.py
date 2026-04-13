# Maps fused emotion label to a tone descriptor and TTS prosody parameters.

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from speech.tts import TTSParams

_TONE_MAP: dict[str, tuple[str, float, int, str]] = {
    "angry":   ("firm",       0.95, 160, "assertive"),
    "disgust": ("measured",   1.00, 155, "neutral"),
    "fear":    ("reassuring", 1.05, 150, "calm"),
    "happy":   ("energetic",  1.10, 190, "cheerful"),
    "neutral": ("balanced",   1.00, 175, "neutral"),
    "sad":     ("gentle",     0.90, 145, "soft"),
    "surprise":("engaged",    1.08, 180, "curious"),
}


class ToneMapper:

    def __init__(self, config: Config) -> None:
        self._cfg = config.tts

    def map(self, fused: FusedEmotion) -> tuple[str, TTSParams]:
        tone_desc, pitch, rate, style = _TONE_MAP.get(
            fused.label, ("balanced", 1.00, 175, "neutral")
        )
        return tone_desc, TTSParams(
            rate=rate,
            pitch=pitch,
            volume=self._cfg.default_volume,
            style=style,
        )
