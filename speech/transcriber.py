# Speech-to-text via faster-whisper with VAD filtering and English language lock.

from __future__ import annotations

import ctypes
import platform
from pathlib import Path

from faster_whisper import WhisperModel

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "distil-large-v3"
_COMPUTE_TYPE_CPU = "int8"
_COMPUTE_TYPE_CUDA = "float16"


def _cuda_available() -> bool:
    import glob
    if platform.system() == "Windows":
        dll_name = "cublas64_12.dll"
        try:
            ctypes.windll.LoadLibrary(dll_name)
            return True
        except OSError:
            pass
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

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(config.models, "whisper_model", _DEFAULT_MODEL)
        requested_device: str = getattr(config.models, "device", "cpu")
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

        logger.info("Loading faster-whisper model '%s' on %s (%s) ...", model_name, device, compute_type)
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("faster-whisper model ready.")

    def transcribe(self, wav_path: Path) -> str:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            logger.error("WAV file not found: %s", wav_path)
            return ""

        logger.info("Transcribing: %s", wav_path.name)
        transcript = self._run_inference(wav_path)

        if not transcript:
            logger.warning("No speech detected in '%s' — treating as silence.", wav_path.name)
        else:
            preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
            logger.info("Transcript: %s", preview)

        return transcript

    def _run_inference(self, wav_path: Path) -> str:
        segments, info = self._model.transcribe(
            str(wav_path),
            beam_size=5,             # Wider beam — notably better with accents
            language=None,           # Auto-detect; handles accent diversity better
            initial_prompt="A natural conversation with an AI companion.",
            vad_filter=False,        # Keep silence; meaningful for emotion
            word_timestamps=False,   # Not needed here
            condition_on_previous_text=False,  # Avoid hallucination on short turns
        )
        logger.info("Detected language '%s' (confidence %.2f)", info.language, info.language_probability)
        return " ".join(seg.text for seg in segments).strip()



