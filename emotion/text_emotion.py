"""
text_emotion.py — Text-based emotion recognition via GoEmotions BERT.

Responsibilities:
  - Load the GoEmotions BERT model (28-class) once at startup
  - Accept a transcript string and return a structured emotion result
  - Return a neutral result (not raise) for empty / silent input
  - Expose the full score distribution for downstream emotion fusion

Model: monologg/bert-base-cased-goemotions-original
  - Trained on the Google GoEmotions dataset (58k Reddit comments)
  - 28 fine-grained emotion labels + neutral
  - ~110MB download, cached after first use

Design:
  - Uses transformers text-classification pipeline with top_k=None to get
    all 28 scores (not just the top-1), which the fusion module needs.
  - Scores are raw sigmoid outputs; not guaranteed to sum to 1.0.
  - For empty input, returns confidence=1.0 for "neutral" to avoid
    biasing the emotion fusion toward any real emotion.
"""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass, field
from typing import Dict

from transformers import pipeline

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Hugging Face model identifier
_MODEL_NAME = "monologg/bert-base-cased-goemotions-original"

# The 28 GoEmotions labels in the order the model outputs them.
# Used to build neutral fallback distributions.
GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
)

# Minimum transcript length (chars) to attempt real inference.
# Anything shorter is treated as silence / noise.
_MIN_TEXT_LENGTH = 3


@dataclass
class TextEmotionResult:
    """
    Structured output from :class:`TextEmotionRecognizer`.

    Attributes:
        emotion:    Top predicted emotion label (e.g. ``"joy"``).
        confidence: Model's confidence for the top emotion (0.0 – 1.0).
        all_scores: Full 28-label score dict for fusion weighting.
        is_silent:  ``True`` when the input was empty / too short to infer.
    """
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    is_silent: bool = False

    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%})"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _neutral_result() -> TextEmotionResult:
    """Return a full-confidence neutral result for silent/empty input."""
    scores = {label: 0.0 for label in GOEMOTIONS_LABELS}
    scores["neutral"] = 1.0
    return TextEmotionResult(
        emotion="neutral",
        confidence=1.0,
        all_scores=scores,
        is_silent=True,
    )


def _resolve_device(requested: str) -> str:
    """
    Return ``"cuda"`` if requested and the CUDA 12 runtime is available,
    otherwise ``"cpu"``. Mirrors the same probe used in the transcriber.
    """
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
        logger.warning(
            "CUDA requested but CUDA 12 runtime not found — "
            "text emotion will run on CPU."
        )
        return "cpu"


# ── Main class ─────────────────────────────────────────────────────────────────

class TextEmotionRecognizer:
    """
    Predicts fine-grained emotion from a text transcript using GoEmotions BERT.

    The model is loaded once in ``__init__`` and reused across calls.
    Inference is synchronous and runs in the caller's thread.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(
            config.models, "text_emotion_model_name", _MODEL_NAME
        )
        requested_device: str = getattr(config.models, "device", "cpu")
        device = _resolve_device(requested_device)

        # transformers pipeline device: int (GPU index) or -1 (CPU)
        pipeline_device = 0 if device == "cuda" else -1

        logger.info(
            "Loading text emotion model '%s' on %s ...", model_name, device
        )
        self._pipe = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,            # Return scores for all 28 labels
            device=pipeline_device,
            truncation=True,       # BERT max 512 tokens — truncate silently
            max_length=512,
        )
        logger.info("Text emotion model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, transcript: str) -> TextEmotionResult:
        """
        Run GoEmotions inference on a transcript string.

        Silence / empty input
        ---------------------
        If ``transcript`` is empty or shorter than :data:`_MIN_TEXT_LENGTH`
        characters, inference is skipped and a neutral result is returned
        with ``is_silent=True``. The fusion module should weight this channel
        lower when it sees a silent result.

        Args:
            transcript: Transcribed user speech (plain string, no formatting).

        Returns:
            :class:`TextEmotionResult` with the top emotion, its confidence,
            and the full 28-label score dictionary for fusion.
        """
        text = transcript.strip()

        if len(text) < _MIN_TEXT_LENGTH:
            logger.warning(
                "Transcript too short (%d chars) — returning neutral.", len(text)
            )
            return _neutral_result()

        logger.info("Running text emotion inference on: '%s...'", text[:60])

        raw: list[list[dict]] = self._pipe(text)
        # Pipeline returns [[{label, score}, ...]] for a single input
        label_scores: list[dict] = raw[0] if isinstance(raw[0], list) else raw

        # Build score dict and find top prediction
        all_scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4)
            for item in label_scores
        }
        top = max(label_scores, key=lambda x: x["score"])

        result = TextEmotionResult(
            emotion=top["label"],
            confidence=round(float(top["score"]), 4),
            all_scores=all_scores,
        )

        logger.info(
            "Text emotion: %s (%.1f%%) | runner-up: %s (%.1f%%)",
            result.emotion,
            result.confidence * 100,
            *_runner_up(label_scores, top["label"]),
        )
        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _runner_up(
    label_scores: list[dict], top_label: str
) -> tuple[str, float]:
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
    recognizer = TextEmotionRecognizer(cfg)

    test_inputs = [
        "I'm so happy today, everything went perfectly!",
        "I'm really frustrated with how this turned out.",
        "I don't know... I just feel empty right now.",
        "Oh my goodness, I can't believe that happened!",
        "",                        # silence / empty
        "Hmm.",                    # very short
    ]

    print("=" * 55)
    print("  PHILIA -- Text Emotion Test (GoEmotions BERT)")
    print("=" * 55)

    for text in test_inputs:
        result = recognizer.predict(text)
        label = f'"{text[:45]}{"..." if len(text) > 45 else ""}"'
        print(f"\nInput:  {label}")
        print(f"Result: {result}")
        if result.is_silent:
            print("        (silent — neutral fallback)")
