"""
audio_emotion.py — Speech-based emotion recognition via Wav2Vec2.

Responsibilities:
  - Load a pretrained Wav2Vec2 emotion model once at startup
  - Accept a WAV file path and return a structured emotion result
  - Return a neutral result (not raise) for silent / too-short audio
  - Expose the full score distribution for downstream emotion fusion

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
  - Fine-tuned on the RAVDESS dataset (7k+ labelled utterances)
  - 7 emotion labels: angry, calm, disgust, fear, happy, neutral, sad,
    surprise (mapped to the canonical EMOTION_LABELS set)
  - Based on jonatasgrosman/wav2vec2-large-xlsr-53-english (~1.2 GB,
    cached after first use by transformers)

Design:
  - Uses transformers audio-classification pipeline with top_k=None to
    get all label scores, which the fusion module needs.
  - Scores are raw softmax outputs summing to ~1.0.
  - Audio is loaded with soundfile at its native sample rate, then
    resampled to 16 kHz mono (required by Wav2Vec2) via torchaudio.
  - For audio < _MIN_AUDIO_SECONDS, returns neutral with is_silent=True
    to avoid biasing fusion toward any real emotion.
"""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_audio
from transformers import pipeline

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Hugging Face model identifier
_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Wav2Vec2 requires 16 kHz mono input
_TARGET_SAMPLE_RATE = 16_000

# Minimum audio duration to attempt real inference.
# Clips shorter than this are treated as silence / noise.
_MIN_AUDIO_SECONDS = 0.5

# Canonical 7-label emotion set (RAVDESS labels used by this model).
# Must stay in sync with EMOTION_LABELS in fusion.py and the model's id2label.
EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


@dataclass
class AudioEmotionResult:
    """
    Structured output from :class:`AudioEmotionRecognizer`.

    Attributes:
        emotion:    Top predicted emotion label (e.g. ``"happy"``).
        confidence: Model's confidence for the top emotion (0.0 – 1.0).
        all_scores: Full label score dict for fusion weighting.
        is_silent:  ``True`` when the input was too short to infer.
    """

    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    is_silent: bool = False

    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%})"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _neutral_result() -> AudioEmotionResult:
    """Return a full-confidence neutral result for silent/too-short input."""
    scores = {label: 0.0 for label in EMOTION_LABELS}
    scores["neutral"] = 1.0
    return AudioEmotionResult(
        emotion="neutral",
        confidence=1.0,
        all_scores=scores,
        is_silent=True,
    )


def _resolve_device(requested: str) -> str:
    """
    Return 'cuda' only if:
      1. 'cuda' was requested in config
      2. CUDA 12 runtime is present (cublas)
      3. PyTorch's cuDNN is available AND version >= 8.9

    Uses torch.backends.cudnn.version() which is authoritative regardless of
    cuDNN DLL filename changes between versions 8 and 9 on Windows.
    """
    if requested != "cuda":
        return "cpu"

    # -- 1. CUDA 12 runtime ---------------------------------------------------
    import platform as _platform
    import ctypes as _ctypes
    _dll = "cublas64_12.dll" if _platform.system() == "Windows" else "libcublas.so.12"
    try:
        if _platform.system() == "Windows":
            _ctypes.windll.LoadLibrary(_dll)
        else:
            _ctypes.CDLL(_dll)
    except OSError:
        logger.warning("CUDA 12 runtime not found -- audio emotion will run on CPU.")
        return "cpu"

    # -- 2. cuDNN >= 8.9 via PyTorch ------------------------------------------
    import torch as _torch
    if not _torch.backends.cudnn.is_available():
        logger.warning("cuDNN not available in PyTorch -- audio emotion will run on CPU.")
        return "cpu"

    _ver = _torch.backends.cudnn.version()  # e.g. 90100 = cuDNN 9.1.0
    if _ver < 8900:
        logger.warning(
            "cuDNN %d < 8900 -- audio emotion will run on CPU. Upgrade to cuDNN 9.x.",
            _ver,
        )
        return "cpu"

    logger.info("cuDNN %d OK -- audio emotion will use GPU.", _ver)
    return "cuda"



def _load_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (float32 mono array, sample_rate).

    Multi-channel audio is averaged to mono. soundfile is used instead
    of librosa to keep the dependency footprint small.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # average channels → mono
    mono = audio.mean(axis=1)
    return mono, sr


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resample *audio* from *orig_sr* to _TARGET_SAMPLE_RATE using
    torchaudio's high-quality Kaiser-windowed sinc resampler.
    Returns a float32 numpy array.
    """
    if orig_sr == _TARGET_SAMPLE_RATE:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    resampled = F_audio.resample(tensor, orig_freq=orig_sr, new_freq=_TARGET_SAMPLE_RATE)
    return resampled.squeeze(0).numpy()


# ── Main class ─────────────────────────────────────────────────────────────────

class AudioEmotionRecognizer:
    """
    Predicts fine-grained emotion from a WAV file using Wav2Vec2.

    The model is loaded once in ``__init__`` and reused across calls.
    Inference is synchronous and runs in the caller's thread.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(
            config.models, "audio_emotion_model_name", _MODEL_NAME
        )
        requested_device: str = getattr(config.models, "emotion_device", "cpu")
        device = _resolve_device(requested_device)

        # transformers pipeline device: int (GPU index) or -1 (CPU)
        pipeline_device = 0 if device == "cuda" else -1

        logger.info(
            "Loading audio emotion model '%s' on %s ...", model_name, device
        )
        self._pipe = pipeline(
            task="audio-classification",
            model=model_name,
            top_k=None,          # return all label scores
            device=pipeline_device,
        )
        logger.info("Audio emotion model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, wav_path: str | Path) -> AudioEmotionResult:
        """
        Run Wav2Vec2 inference on a WAV file.

        Silent / too-short audio
        ------------------------
        If the WAV is shorter than :data:`_MIN_AUDIO_SECONDS`, inference is
        skipped and a neutral result is returned with ``is_silent=True``.
        The fusion module should weight this channel lower when silent.

        Args:
            wav_path: Path to a 16 kHz mono WAV file (other formats are
                      resampled automatically).

        Returns:
            :class:`AudioEmotionResult` with the top emotion, its confidence,
            and the full label score dict for fusion.
        """
        path = Path(wav_path)
        if not path.exists():
            logger.error("WAV file not found: %s — returning neutral.", path)
            return _neutral_result()

        audio, sr = _load_wav(path)
        duration = len(audio) / sr

        if duration < _MIN_AUDIO_SECONDS:
            logger.warning(
                "Audio too short (%.2fs) — returning neutral.", duration
            )
            return _neutral_result()

        # Resample to 16 kHz if needed
        if sr != _TARGET_SAMPLE_RATE:
            logger.debug(
                "Resampling from %d Hz to %d Hz.", sr, _TARGET_SAMPLE_RATE
            )
            audio = _resample(audio, sr)

        logger.info(
            "Running audio emotion inference on '%s' (%.1fs) ...",
            path.name, duration,
        )

        # Pipeline accepts a numpy array at the target sample rate
        raw: list[dict] = self._pipe(
            {"raw": audio, "sampling_rate": _TARGET_SAMPLE_RATE}
        )

        # Build score dict and find top prediction
        all_scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4)
            for item in raw
        }
        top = max(raw, key=lambda x: x["score"])

        result = AudioEmotionResult(
            emotion=top["label"],
            confidence=round(float(top["score"]), 4),
            all_scores=all_scores,
        )

        logger.info(
            "Audio emotion: %s (%.1f%%) | runner-up: %s (%.1f%%)",
            result.emotion,
            result.confidence * 100,
            *_runner_up(raw, top["label"]),
        )
        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _runner_up(label_scores: list[dict], top_label: str) -> tuple[str, float]:
    """Return the second-highest scoring label and its percentage."""
    others = [s for s in label_scores if s["label"] != top_label]
    if not others:
        return "—", 0.0
    second = max(others, key=lambda x: x["score"])
    return second["label"], second["score"] * 100


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import Config

    cfg = Config.load()
    recognizer = AudioEmotionRecognizer(cfg)

    # Place test WAVs in d:\PHILIA\tmp\ or pass paths as CLI args.
    test_paths: list[str] = sys.argv[1:] if len(sys.argv) > 1 else []

    print("=" * 55)
    print("  PHILIA -- Audio Emotion Test (Wav2Vec2)")
    print("=" * 55)

    if not test_paths:
        print("\nUsage: python -m emotion.audio_emotion path/to/clip.wav ...")
        print("\nNo WAV files provided — running synthetic silence test.")
        # Create a 0.2-second silence WAV in tmp/ for the silent-input test
        import tempfile, soundfile as sf_tmp, numpy as np_tmp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="tmp") as f:
            silence = np_tmp.zeros(int(0.2 * 16000), dtype="float32")
            sf_tmp.write(f.name, silence, 16000)
            test_paths = [f.name]
            print(f"  Generated silence WAV: {f.name}")

    for wav_path in test_paths:
        result = recognizer.predict(wav_path)
        print(f"\nFile:   {wav_path}")
        print(f"Result: {result}")
        if result.is_silent:
            print("        (silent — neutral fallback)")
        else:
            print("  Scores:")
            for label, score in sorted(
                result.all_scores.items(), key=lambda kv: -kv[1]
            ):
                bar = "#" * int(score * 20)
                print(f"    {label:<10} {score:.3f}  {bar}")
