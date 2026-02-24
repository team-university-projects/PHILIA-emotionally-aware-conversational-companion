"""
audio_capture.py — Push-to-Talk microphone recording.

Responsibilities:
  - Open the default (or configured) audio input device
  - Record audio while the PTT signal is active
  - Return raw PCM bytes ready for downstream processing
"""

from __future__ import annotations

import wave
import io
from config import Config


class AudioCapture:
    """Records microphone audio during a push-to-talk window."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.audio
        # TODO: initialise PyAudio stream here

    def record(self) -> bytes:
        """
        Block until PTT is released, then return recorded PCM bytes.

        Returns:
            bytes: Raw WAV-format audio bytes at config.audio.sample_rate.
        """
        raise NotImplementedError

    def _open_stream(self):
        """Open the PyAudio input stream."""
        raise NotImplementedError

    def _close_stream(self) -> None:
        """Release the PyAudio stream and resources."""
        raise NotImplementedError
