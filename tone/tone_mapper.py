# Maps fused emotion label to a tone descriptor and TTS prosody parameters.

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from speech.tts import TTSParams

_TONE_MAP: dict[str, tuple[str, float, int, str]] = {
    # (tone_desc, pitch_mult, rate_wpm, style)
    # Pitch: 0.97–1.04 (±3–4 Hz)  |  Rate: 162–182 wpm (±~5%)
    # Keeps the same voice identity while still conveying emotional colouring.
    "angry":   ("firm",       0.98, 168, "assertive"),
    "disgust": ("measured",   0.98, 164, "neutral"),
    "fear":    ("reassuring", 0.98, 162, "calm"),
    "happy":   ("energetic",  1.03, 182, "cheerful"),
    "neutral": ("balanced",   1.00, 172, "neutral"),
    "sad":     ("gentle",     0.97, 162, "empathetic"),
    "surprise":("engaged",    1.02, 178, "curious"),
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
