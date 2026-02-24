"""
tts.py — Emotion-aware text-to-speech synthesis.

Responsibilities:
  - Accept response text and TTSParams from the tone mapper
  - Synthesise speech audio applying pitch, rate, and style
  - Return audio bytes or play directly to output device
"""

from __future__ import annotations

from dataclasses import dataclass
from config import Config
from bot.bot_profile import BotProfile


@dataclass
class TTSParams:
    """Prosody parameters derived from the tone mapper."""
    rate: int       # Words per minute
    pitch: float    # Pitch multiplier (1.0 = neutral)
    volume: float   # Volume multiplier (1.0 = full)
    style: str      # Voice style descriptor e.g. "calm", "energetic", "gentle"


class TextToSpeech:
    """Synthesises speech with emotion-conditioned prosody."""

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._cfg = config.tts
        self._profile = profile
        # TODO: initialise TTS engine (pyttsx3 / ElevenLabs / gTTS)

    def synthesise(self, text: str, params: TTSParams) -> bytes:
        """
        Synthesise speech from text with prosody from TTSParams.

        Args:
            text:   Response text from the LLM.
            params: Prosody parameters (rate, pitch, volume, style).

        Returns:
            Raw WAV audio bytes of the synthesised speech.
        """
        raise NotImplementedError
