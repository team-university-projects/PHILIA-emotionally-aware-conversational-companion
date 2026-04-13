# Confidence-weighted late fusion of audio, facial, and text emotion signals.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from config import Config
from emotion.audio_emotion import AudioEmotionResult
from emotion.facial_emotion import FacialEmotionResult
from emotion.text_emotion import TextEmotionResult
from utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_LABELS: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)

_AUDIO_TO_CANONICAL: dict[str, str] = {
    "angry": "angry", "calm": "neutral", "disgust": "disgust",
    "fear": "fear", "happy": "happy", "neutral": "neutral",
    "sad": "sad", "surprise": "surprise",
}

_TEXT_TO_CANONICAL: dict[str, str] = {
    "admiration": "happy", "amusement": "happy", "anger": "angry",
    "annoyance": "angry", "approval": "happy", "caring": "happy",
    "confusion": "neutral", "curiosity": "surprise", "desire": "happy",
    "disappointment": "sad", "disapproval": "angry", "disgust": "disgust",
    "embarrassment": "fear", "excitement": "happy", "fear": "fear",
    "gratitude": "happy", "grief": "sad", "joy": "happy", "love": "happy",
    "nervousness": "fear", "optimism": "happy", "pride": "happy",
    "realization": "surprise", "relief": "neutral", "remorse": "sad",
    "sadness": "sad", "surprise": "surprise", "neutral": "neutral",
}


@dataclass
class FusedEmotion:
    label: str
    confidence: float
    distribution: Dict[str, float]
    breakdown: Dict[str, Dict[str, float]]
    weights_used: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        w = self.weights_used
        return (
            f"{self.label} ({self.confidence:.1%}) "
            f"[a={w.get('audio', 0):.0%} "
            f"f={w.get('facial', 0):.0%} "
            f"t={w.get('text', 0):.0%}]"
        )


class EmotionFuser:

    def __init__(self, config: Config) -> None:
        self._base_weights: Dict[str, float] = {
            "audio":  config.fusion.audio_weight,
            "facial": config.fusion.facial_weight,
            "text":   config.fusion.text_weight,
        }
        total = sum(self._base_weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning("Fusion weights sum to %.4f (expected 1.0) — normalising.", total)
            self._base_weights = {k: v / total for k, v in self._base_weights.items()}

    def fuse(
        self,
        audio_result:  AudioEmotionResult,
        facial_result: FacialEmotionResult,
        text_result:   TextEmotionResult,
    ) -> FusedEmotion:
        normalised = {
            "audio":  self._normalise_audio(audio_result.all_scores),
            "facial": self._normalise_facial(facial_result.all_scores),
            "text":   self._normalise_text(text_result.all_scores),
        }
        is_inactive = {
            "audio":  audio_result.is_silent,
            "facial": facial_result.is_no_face,
            "text":   text_result.is_silent,
        }
        effective_weights = self._adjust_weights(is_inactive)
        fused_dist = self._aggregate(normalised, effective_weights)
        fused_dist = {k: round(v, 4) for k, v in fused_dist.items()}
        top_label = max(fused_dist, key=lambda k: fused_dist[k])
        result = FusedEmotion(
            label=top_label,
            confidence=fused_dist[top_label],
            distribution=fused_dist,
            breakdown=normalised,
            weights_used=effective_weights,
        )
        logger.info(
            "Fusion >> %s (%.1f%%) | weights: audio=%.0f%% facial=%.0f%% text=%.0f%%",
            result.label, result.confidence * 100,
            effective_weights["audio"]  * 100,
            effective_weights["facial"] * 100,
            effective_weights["text"]   * 100,
        )
        return result

    def _aggregate(self, normalised: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
        result: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for modality, dist in normalised.items():
            w = weights[modality]
            if w == 0.0:
                continue
            for label, score in dist.items():
                result[label] += w * score
        return result

    def _adjust_weights(self, is_inactive: Dict[str, bool]) -> Dict[str, float]:
        active = {m: w for m, w in self._base_weights.items() if not is_inactive.get(m, False)}
        if not active:
            logger.warning("All modalities are silent — using base weights.")
            return dict(self._base_weights)
        total_active = sum(active.values())
        adjusted: Dict[str, float] = {}
        for modality in self._base_weights:
            if is_inactive.get(modality, False):
                adjusted[modality] = 0.0
            else:
                adjusted[modality] = round(self._base_weights[modality] / total_active, 4)
        if any(is_inactive.values()):
            inactive_names = [m for m, v in is_inactive.items() if v]
            logger.info("Inactive modalities %s — redistributed weights: %s",
                        inactive_names, {k: f"{v:.0%}" for k, v in adjusted.items()})
        return adjusted

    @staticmethod
    def _normalise_audio(scores: Dict[str, float]) -> Dict[str, float]:
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            target = _AUDIO_TO_CANONICAL.get(label)
            if target:
                canonical[target] += score
        return _renormalise(canonical)

    @staticmethod
    def _normalise_facial(scores: Dict[str, float]) -> Dict[str, float]:
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            if label in canonical:
                canonical[label] = score
        return _renormalise(canonical)

    @staticmethod
    def _normalise_text(scores: Dict[str, float]) -> Dict[str, float]:
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            target = _TEXT_TO_CANONICAL.get(label)
            if target:
                canonical[target] += score
        return _renormalise(canonical)


def _renormalise(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    if total < 1e-9:
        n = len(dist)
        return {k: round(1.0 / n, 4) for k in dist}
    return {k: round(v / total, 4) for k, v in dist.items()}
