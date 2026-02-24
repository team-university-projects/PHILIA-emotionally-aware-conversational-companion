"""
transcriber.py — Speech-to-text transcription.

Responsibilities:
  - Load a Whisper (or compatible STT) model
  - Accept raw audio bytes
  - Return a clean transcript string
"""

from __future__ import annotations

from config import Config


class Transcriber:
    """Transcribes speech audio to text using OpenAI Whisper."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        # TODO: load whisper.load_model(self._cfg.whisper_model)

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Convert audio bytes to a text transcript.

        Args:
            audio_bytes: Raw WAV bytes at 16 kHz mono.

        Returns:
            Transcribed text string. Returns empty string on failure.
        """
        raise NotImplementedError
