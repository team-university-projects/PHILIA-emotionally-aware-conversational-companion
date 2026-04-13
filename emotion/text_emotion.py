# Text emotion recognition via GoEmotions BERT (28 labels → canonical 7).

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from typing import Dict

from transformers import pipeline

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "monologg/bert-base-cased-goemotions-original"
_MIN_TEXT_LENGTH = 3

GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
)


@dataclass
class TextEmotionResult:
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    is_silent: bool = False

    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%})"


def _neutral_result() -> TextEmotionResult:
    scores = {label: 0.0 for label in GOEMOTIONS_LABELS}
    scores["neutral"] = 1.0
    return TextEmotionResult(emotion="neutral", confidence=1.0, all_scores=scores, is_silent=True)


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return "cpu"
    dll = "cublas64_12.dll" if platform.system() == "Windows" else "libcublas.so.12"
    try:
        if platform.system() == "Windows":
            ctypes.windll.LoadLibrary(dll)
        else:
            ctypes.CDLL(dll)
        return "cuda"
    except OSError:
        logger.warning("CUDA requested but CUDA 12 runtime not found — text emotion will run on CPU.")
        return "cpu"


class TextEmotionRecognizer:

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(config.models, "text_emotion_model_name", _MODEL_NAME)
        device = _resolve_device(getattr(config.models, "device", "cpu"))
        pipeline_device = 0 if device == "cuda" else -1
        logger.info("Loading text emotion model '%s' on %s ...", model_name, device)
        self._pipe = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
            device=pipeline_device,
            truncation=True,
            max_length=512,
        )
        logger.info("Text emotion model ready.")

    def predict(self, transcript: str) -> TextEmotionResult:
        text = transcript.strip()
        if len(text) < _MIN_TEXT_LENGTH:
            logger.warning("Transcript too short (%d chars) — returning neutral.", len(text))
            return _neutral_result()

        logger.info("Running text emotion inference on: '%s...'", text[:60])
        raw: list[list[dict]] = self._pipe(text)
        label_scores: list[dict] = raw[0] if isinstance(raw[0], list) else raw

        all_scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4) for item in label_scores
        }
        top = max(label_scores, key=lambda x: x["score"])
        result = TextEmotionResult(
            emotion=top["label"],
            confidence=round(float(top["score"]), 4),
            all_scores=all_scores,
        )
        logger.info(
            "Text emotion: %s (%.1f%%) | runner-up: %s (%.1f%%)",
            result.emotion, result.confidence * 100,
            *_runner_up(label_scores, top["label"]),
        )
        return result


def _runner_up(label_scores: list[dict], top_label: str) -> tuple[str, float]:
    others = [s for s in label_scores if s["label"] != top_label]
    if not others:
        return "—", 0.0
    second = max(others, key=lambda x: x["score"])
    return second["label"], second["score"] * 100
