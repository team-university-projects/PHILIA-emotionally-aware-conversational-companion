"""
audio_utils.py — Shared audio processing helpers.

Responsibilities:
  - WAV buffer read / write
  - Sample-rate conversion
  - Silence trimming
  - Byte ↔ numpy array conversion
"""

from __future__ import annotations

import io
import wave
import numpy as np


def bytes_to_numpy(audio_bytes: bytes, sample_rate: int = 16_000) -> np.ndarray:
    """
    Convert raw WAV bytes to a normalised float32 numpy array.

    Args:
        audio_bytes: Raw WAV-format bytes.
        sample_rate: Expected sample rate for validation.

    Returns:
        1-D float32 array in [-1.0, 1.0].
    """
    raise NotImplementedError


def numpy_to_bytes(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """
    Convert a float32 numpy array back to WAV bytes.

    Args:
        audio:       1-D float32 array in [-1.0, 1.0].
        sample_rate: Sample rate for the WAV header.

    Returns:
        WAV-format bytes.
    """
    raise NotImplementedError


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove leading and trailing silence from an audio array.

    Args:
        audio:     1-D float32 audio array.
        threshold: RMS threshold below which a frame is considered silent.

    Returns:
        Trimmed audio array.
    """
    raise NotImplementedError
