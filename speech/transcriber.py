"""
transcriber.py — Speech-to-text via faster-whisper.

Responsibilities:
  - Load the faster-whisper "small" model once at startup
  - Transcribe a WAV file path to a plain text string
  - Return an empty string (never raise) when audio is silent or too short
  - Preserve silence in timing by leaving VAD filter OFF

Design choices:
  - faster-whisper (CTranslate2 backend) over openai-whisper: 2-4× faster,
    lower memory, identical accuracy on the small model.
  - compute_type="int8" on CPU: best throughput without accuracy loss.
  - beam_size=1 (greedy): fastest decode; adequate for conversational audio.
  - No VAD filter: silence/hesitation is meaningful for emotion inference;
    suppressing it would lose that signal.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Model — distil-large-v3 is the best accuracy/speed balance for accented speech.
# It is a distilled version of large-v3: ~6x faster, ~97% the accuracy.
# Overridable via config.models.whisper_model.
_DEFAULT_MODEL = "distil-large-v3"

# CTranslate2 compute type.  "int8" is fastest on CPU; "float16" on CUDA.
_COMPUTE_TYPE_CPU = "int8"
_COMPUTE_TYPE_CUDA = "float16"


def _cuda_available() -> bool:
    """
    Probe whether CUDA 12 runtime DLLs are accessible on this machine.

    Checks the system PATH first (fast path), then falls back to searching
    common NVIDIA installation directories in case the Toolkit was just
    installed and the PATH hasn't been refreshed in this terminal session.

    Returns:
        ``True`` if the CUDA 12 runtime is present and loadable.
    """
    import glob

    if platform.system() == "Windows":
        dll_name = "cublas64_12.dll"
        # 1. Try PATH first (works after a terminal restart post-install)
        try:
            ctypes.windll.LoadLibrary(dll_name)
            return True
        except OSError:
            pass
        # 2. Search common NVIDIA install locations (works before PATH refresh)
        search_patterns = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin\x64",
        ]
        for pattern in search_patterns:
            for cuda_bin in glob.glob(pattern):
                dll_path = Path(cuda_bin) / dll_name
                if dll_path.exists():
                    try:
                        ctypes.windll.LoadLibrary(str(dll_path))
                        return True
                    except OSError:
                        continue
        # 3. Detect CUDA 13 to give a clearer version-mismatch warning
        cuda13_present = bool(glob.glob(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13*\bin*\cublas64_13.dll"
        ))
        if cuda13_present:
            logger.warning(
                "CUDA 13 detected, but CTranslate2 requires CUDA 12. "
                "Install CUDA 12.x from https://developer.nvidia.com/cuda-12-4-0-download-archive "
                "to enable GPU acceleration."
            )
        return False
    else:
        for lib in ("libcublas.so.12", "libcublas.so"):
            try:
                ctypes.CDLL(lib)
                return True
            except OSError:
                continue
        return False


class Transcriber:
    """
    Transcribes a WAV file to text using faster-whisper.

    The model is loaded once in ``__init__`` and reused across calls.
    Transcription is synchronous and runs in the caller's thread.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(config.models, "whisper_model", _DEFAULT_MODEL)
        requested_device: str = getattr(config.models, "device", "cpu")

        # Resolve actual device: honour the config preference but fall back
        # to CPU if the CUDA runtime libraries are not present.
        if requested_device == "cuda":
            if _cuda_available():
                device, compute_type = "cuda", _COMPUTE_TYPE_CUDA
                logger.info("CUDA 12 runtime detected — using GPU.")
            else:
                device, compute_type = "cpu", _COMPUTE_TYPE_CPU
                logger.warning(
                    "CUDA requested but CUDA 12 runtime not found — "
                    "falling back to CPU. Install the CUDA 12 Toolkit from "
                    "https://developer.nvidia.com/cuda-downloads to enable GPU."
                )
        else:
            device, compute_type = "cpu", _COMPUTE_TYPE_CPU

        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) ...",
            model_name, device, compute_type,
        )
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, wav_path: Path) -> str:
        """
        Transcribe a WAV file to a plain text string.

        Silence handling
        ----------------
        faster-whisper's VAD filter is intentionally **off** so that
        pauses and hesitations are kept (they carry emotion signal).
        If the audio contains no detectable speech at all, Whisper may
        return an empty or whitespace-only result — in that case this
        method logs a warning and returns ``""``.  The caller should
        treat ``""`` as "the user was silent" rather than an error.

        Args:
            wav_path: Path to a 16 kHz (or any sample rate) WAV file.

        Returns:
            Transcribed text, or ``""`` if no speech was detected.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            logger.error("WAV file not found: %s", wav_path)
            return ""

        logger.info("Transcribing: %s", wav_path.name)

        transcript = self._run_inference(wav_path)

        if not transcript:
            logger.warning(
                "No speech detected in '%s' — treating as silence.", wav_path.name
            )
        else:
            # Log a short preview, not the full text, to keep logs readable
            preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
            logger.info("Transcript: %s", preview)

        return transcript

    # ── Private ────────────────────────────────────────────────────────────────

    def _run_inference(self, wav_path: Path) -> str:
        """
        Run faster-whisper inference and join all segment texts.

        Args:
            wav_path: Validated path to the WAV file.

        Returns:
            Joined transcript string, stripped of leading/trailing whitespace.
        """
        segments, info = self._model.transcribe(
            str(wav_path),
            beam_size=5,             # Wider beam — notably better with accents
            language="en",           # Lock to English — faster and more accurate than auto-detect
            initial_prompt="The following is a natural conversation with an empathetic AI companion.",
            vad_filter=True,         # Skip non-speech frames — avoids hallucinations from mic noise
            vad_parameters=dict(
                threshold=0.3,           # Fairly sensitive — catch soft speech
                min_speech_duration_ms=100,
                max_speech_duration_s=30,
                min_silence_duration_ms=300,
                speech_pad_ms=200,       # Pad edges so word boundaries aren't clipped
            ),
            word_timestamps=False,
            condition_on_previous_text=False,  # Avoid hallucination chaining on short turns
        )

        logger.info(
            "Detected language '%s' (confidence %.2f)",
            info.language, info.language_probability,
        )

        # `segments` is a lazy generator — consume it fully
        parts = [seg.text for seg in segments]
        return " ".join(parts).strip()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import Config

    if len(sys.argv) < 2:
        print("Usage: python -m speech.transcriber <path/to/recording.wav>")
        sys.exit(1)

    wav = Path(sys.argv[1])
    cfg = Config.load()
    t = Transcriber(cfg)

    print("=" * 50)
    print("  PHILIA -- Transcriber Test")
    print(f"  File: {wav.name}")
    print("=" * 50)

    result = t.transcribe(wav)
    print(f"\nTranscript:\n  '{result}'\n")
    print(f"Length: {len(result)} chars")
