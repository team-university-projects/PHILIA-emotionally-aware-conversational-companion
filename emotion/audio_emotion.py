# Audio emotion recognition via Wav2Vec2 fine-tuned on MELD.

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

_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_TARGET_SAMPLE_RATE = 16_000
_MIN_AUDIO_SECONDS = 0.5

EMOTION_LABELS: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)

_LABEL_NORM: dict[str, str] = {
    "fearful":   "fear",
    "surprised": "surprise",
    "calm":      "neutral",
    "anger":     "angry",
    "sadness":   "sad",
    "joy":       "happy",
}


@dataclass
class AudioEmotionResult:
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    is_silent: bool = False

    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%})"


def _neutral_result() -> AudioEmotionResult:
    scores = {label: 0.0 for label in EMOTION_LABELS}
    scores["neutral"] = 1.0
    return AudioEmotionResult(emotion="neutral", confidence=1.0, all_scores=scores, is_silent=True)


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return "cpu"
    import platform as _platform, ctypes as _ctypes
    _dll = "cublas64_12.dll" if _platform.system() == "Windows" else "libcublas.so.12"
    try:
        if _platform.system() == "Windows":
            _ctypes.windll.LoadLibrary(_dll)
        else:
            _ctypes.CDLL(_dll)
    except OSError:
        logger.warning("CUDA 12 runtime not found -- audio emotion will run on CPU.")
        return "cpu"
    import torch as _torch
    if not _torch.backends.cudnn.is_available():
        logger.warning("cuDNN not available in PyTorch -- audio emotion will run on CPU.")
        return "cpu"
    _ver = _torch.backends.cudnn.version()
    if _ver < 8900:
        logger.warning("cuDNN %d < 8900 -- audio emotion will run on CPU.", _ver)
        return "cpu"
    logger.info("cuDNN %d OK -- audio emotion will use GPU.", _ver)
    return "cuda"


def _load_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == _TARGET_SAMPLE_RATE:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampled = F_audio.resample(tensor, orig_freq=orig_sr, new_freq=_TARGET_SAMPLE_RATE)
    return resampled.squeeze(0).numpy()


class AudioEmotionRecognizer:

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(config.models, "audio_emotion_model_name", _MODEL_NAME)
        device = _resolve_device(getattr(config.models, "emotion_device", "cpu"))
        pipeline_device = 0 if device == "cuda" else -1
        logger.info("Loading audio emotion model '%s' on %s ...", model_name, device)
        self._pipe = pipeline(
            task="audio-classification",
            model=model_name,
            top_k=None,
            device=pipeline_device,
        )
        logger.info("Audio emotion model ready.")

    def predict(self, wav_path: str | Path) -> AudioEmotionResult:
        path = Path(wav_path)
        if not path.exists():
            logger.error("WAV file not found: %s — returning neutral.", path)
            return _neutral_result()

        audio, sr = _load_wav(path)
        duration = len(audio) / sr
        if duration < _MIN_AUDIO_SECONDS:
            logger.warning("Audio too short (%.2fs) — returning neutral.", duration)
            return _neutral_result()

        if sr != _TARGET_SAMPLE_RATE:
            audio = _resample(audio, sr)

        logger.info("Running audio emotion inference on '%s' (%.1fs) ...", path.name, duration)
        raw: list[dict] = self._pipe({"raw": audio, "sampling_rate": _TARGET_SAMPLE_RATE})

        merged: Dict[str, float] = {}
        for item in raw:
            canonical = _LABEL_NORM.get(item["label"], item["label"])
            score = round(float(item["score"]), 4)
            merged[canonical] = merged.get(canonical, 0.0) + score

        all_scores: Dict[str, float] = {
            label: round(merged.get(label, 0.0), 4) for label in EMOTION_LABELS
        }
        top_label = max(all_scores, key=lambda k: all_scores[k])
        result = AudioEmotionResult(
            emotion=top_label,
            confidence=round(all_scores[top_label], 4),
            all_scores=all_scores,
        )
        logger.info(
            "Audio emotion: %s (%.1f%%) | runner-up: %s (%.1f%%)",
            result.emotion, result.confidence * 100,
            *_runner_up_from_scores(all_scores, top_label),
        )
        return result


def _runner_up_from_scores(scores: Dict[str, float], top_label: str) -> tuple[str, float]:
    others = {k: v for k, v in scores.items() if k != top_label}
    if not others:
        return "—", 0.0
    second = max(others, key=lambda k: others[k])
    return second, others[second] * 100
