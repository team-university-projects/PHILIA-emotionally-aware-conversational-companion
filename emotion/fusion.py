"""
fusion.py — Confidence-weighted late fusion of multimodal emotion signals.

Responsibilities:
  - Accept AudioEmotionResult, FacialEmotionResult, TextEmotionResult
  - Normalise each to a shared canonical 7-label space
  - Apply per-modality weights (configurable in config.py)
  - Auto-reduce weight for silent/no-face modalities and redistribute
  - Return a FusedEmotion with final label, confidence, and per-modality breakdown

Canonical label set (7 basic emotions — intersection of FER + RAVDESS):
  angry · disgust · fear · happy · neutral · sad · surprise

Label mappings
--------------
RAVDESS (8 → 7):  calm → neutral  (all others are 1-to-1)
GoEmotions (28 → 7): see _TEXT_TO_CANONICAL below

Future extensibility
--------------------
  To swap in attention-based or learned fusion, subclass EmotionFuser and
  override ``_aggregate()``. The rest of the pipeline (normalisation, weight
  adjustment, result packing) stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from config import Config
from emotion.audio_emotion import AudioEmotionResult
from emotion.facial_emotion import FacialEmotionResult
from emotion.text_emotion import TextEmotionResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Canonical label set ────────────────────────────────────────────────────────

CANONICAL_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# ── Label mapping tables ────────────────────────────────────────────────────────

# RAVDESS (audio) → canonical.  "calm" is the only non-1-to-1 label.
_AUDIO_TO_CANONICAL: dict[str, str] = {
    "angry":    "angry",
    "calm":     "neutral",   # RAVDESS-specific; maps to neutral
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "surprise": "surprise",
}

# GoEmotions (text, 28 labels) → canonical.
# Groups semantically similar fine-grained labels into the closest basic emotion.
_TEXT_TO_CANONICAL: dict[str, str] = {
    "admiration":    "happy",
    "amusement":     "happy",
    "anger":         "angry",
    "annoyance":     "angry",
    "approval":      "happy",
    "caring":        "happy",
    "confusion":     "neutral",
    "curiosity":     "surprise",
    "desire":        "happy",
    "disappointment":"sad",
    "disapproval":   "angry",
    "disgust":       "disgust",
    "embarrassment": "fear",
    "excitement":    "happy",
    "fear":          "fear",
    "gratitude":     "happy",
    "grief":         "sad",
    "joy":           "happy",
    "love":          "happy",
    "nervousness":   "fear",
    "optimism":      "happy",
    "pride":         "happy",
    "realization":   "surprise",
    "relief":        "neutral",
    "remorse":       "sad",
    "sadness":       "sad",
    "surprise":      "surprise",
    "neutral":       "neutral",
}


# ── Output dataclass ────────────────────────────────────────────────────────────

@dataclass
class FusedEmotion:
    """
    Result of multimodal late fusion.

    Attributes:
        label:        Winning canonical emotion label.
        confidence:   Fused score for the winning label (0–1).
        distribution: Full canonical fused probability distribution.
        breakdown:    Per-modality normalised distributions used in fusion,
                      keyed by ``"audio"``, ``"facial"``, ``"text"``.
        weights_used: Effective per-modality weights after silent/no-face
                      adjustment (sum to 1.0).
    """
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


# ── Fuser ───────────────────────────────────────────────────────────────────────

class EmotionFuser:
    """
    Combines audio, facial, and text emotion signals via weighted late fusion.

    Weights are loaded from ``config.fusion`` and adjusted at runtime if a
    modality signals silence or no-face — redistributing the removed weight
    proportionally to the remaining active modalities.

    To replace with attention-based fusion: subclass and override
    :meth:`_aggregate`. Everything else (normalisation, weight adjustment,
    result packing) is inherited unchanged.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self._base_weights: Dict[str, float] = {
            "audio":  config.fusion.audio_weight,
            "facial": config.fusion.facial_weight,
            "text":   config.fusion.text_weight,
        }
        total = sum(self._base_weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Fusion weights sum to %.4f (expected 1.0) — normalising.", total
            )
            self._base_weights = {
                k: v / total for k, v in self._base_weights.items()
            }

    # ── Public API ─────────────────────────────────────────────────────────────

    def fuse(
        self,
        audio_result:  AudioEmotionResult,
        facial_result: FacialEmotionResult,
        text_result:   TextEmotionResult,
    ) -> FusedEmotion:
        """
        Fuse three emotion recognition results into a single prediction.

        Silent / no-face handling
        -------------------------
        If a modality flagged ``is_silent`` or ``is_no_face``, its effective
        weight is set to 0 and the surplus is redistributed proportionally
        among the remaining active modalities.  If ALL modalities are silent,
        the base weights are preserved and neutral will dominate naturally.

        Args:
            audio_result:  Output of :class:`~emotion.audio_emotion.AudioEmotionRecognizer`.
            facial_result: Output of :class:`~emotion.facial_emotion.FacialEmotionRecognizer`.
            text_result:   Output of :class:`~emotion.text_emotion.TextEmotionRecognizer`.

        Returns:
            :class:`FusedEmotion` with the winning label, full fused
            distribution, per-modality breakdown, and effective weights.
        """
        # 1. Normalise each modality to the canonical label space
        normalised = {
            "audio":  self._normalise_audio(audio_result.all_scores),
            "facial": self._normalise_facial(facial_result.all_scores),
            "text":   self._normalise_text(text_result.all_scores),
        }

        # 2. Compute effective weights (zero-out silent/no-face modalities)
        is_inactive = {
            "audio":  audio_result.is_silent,
            "facial": facial_result.is_no_face,
            "text":   text_result.is_silent,
        }
        effective_weights = self._adjust_weights(is_inactive)

        # 3. Aggregate (weighted sum — override for attention-based fusion)
        fused_dist = self._aggregate(normalised, effective_weights)

        # 4. Round and find the winner
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
            result.label,
            result.confidence * 100,
            effective_weights["audio"]  * 100,
            effective_weights["facial"] * 100,
            effective_weights["text"]   * 100,
        )
        return result

    # ── Aggregation (override to swap fusion strategy) ─────────────────────────

    def _aggregate(
        self,
        normalised: Dict[str, Dict[str, float]],
        weights:    Dict[str, float],
    ) -> Dict[str, float]:
        """
        Weighted sum over normalised per-modality distributions.

        **Swap this method to use attention-based or learned fusion.**

        Args:
            normalised: ``{"audio": {...}, "facial": {...}, "text": {...}}``
                        each mapping canonical label → score (0–1).
            weights:    Effective weight per modality, summing to 1.0.

        Returns:
            Dict mapping canonical label → weighted-sum score.
        """
        result: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for modality, dist in normalised.items():
            w = weights[modality]
            if w == 0.0:
                continue
            for label, score in dist.items():
                result[label] += w * score
        return result

    # ── Weight adjustment ──────────────────────────────────────────────────────

    def _adjust_weights(
        self, is_inactive: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Zero out inactive modalities and redistribute their weight
        proportionally to the active ones.

        If all modalities are inactive, returns the base weights unchanged
        (neutral scores will dominate naturally).
        """
        active = {
            m: w for m, w in self._base_weights.items()
            if not is_inactive.get(m, False)
        }

        if not active:
            logger.warning("All modalities are silent — using base weights.")
            return dict(self._base_weights)

        total_active = sum(active.values())
        adjusted: Dict[str, float] = {}
        for modality in self._base_weights:
            if is_inactive.get(modality, False):
                adjusted[modality] = 0.0
            else:
                # Rescale so active weights sum to 1.0
                adjusted[modality] = round(
                    self._base_weights[modality] / total_active, 4
                )

        if any(is_inactive.values()):
            inactive_names = [m for m, v in is_inactive.items() if v]
            logger.info(
                "Inactive modalities %s — redistributed weights: %s",
                inactive_names,
                {k: f"{v:.0%}" for k, v in adjusted.items()},
            )

        return adjusted

    # ── Label normalisation ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_audio(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Map RAVDESS 8-label scores → canonical 7-label distribution.
        "calm" is added to "neutral"; all others are 1-to-1.
        Renormalises so values sum to 1.0.
        """
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            target = _AUDIO_TO_CANONICAL.get(label)
            if target:
                canonical[target] += score
        return _renormalise(canonical)

    @staticmethod
    def _normalise_facial(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Map FER 7-label scores → canonical 7-label distribution.
        Labels are identical so this is a direct copy + renormalisation.
        """
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            if label in canonical:
                canonical[label] = score
        return _renormalise(canonical)

    @staticmethod
    def _normalise_text(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Map GoEmotions 28-label scores → canonical 7-label distribution.
        Semantically similar fine-grained labels are grouped and summed,
        then renormalised so the distribution sums to 1.0.
        """
        canonical: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        for label, score in scores.items():
            target = _TEXT_TO_CANONICAL.get(label)
            if target:
                canonical[target] += score
        return _renormalise(canonical)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _renormalise(dist: Dict[str, float]) -> Dict[str, float]:
    """Rescale a distribution so values sum to 1.0. Returns dict unchanged if sum is 0."""
    total = sum(dist.values())
    if total < 1e-9:
        n = len(dist)
        return {k: round(1.0 / n, 4) for k in dist}
    return {k: round(v / total, 4) for k, v in dist.items()}


# ── Standalone test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import Config
    from emotion.audio_emotion import AudioEmotionResult
    from emotion.facial_emotion import FacialEmotionResult
    from emotion.text_emotion import TextEmotionResult

    cfg = Config.load()
    fuser = EmotionFuser(cfg)

    print("=" * 60)
    print("  PHILIA — Fusion Test (Weighted Late Fusion)")
    print(f"  Weights: audio={cfg.fusion.audio_weight:.0%}  "
          f"facial={cfg.fusion.facial_weight:.0%}  "
          f"text={cfg.fusion.text_weight:.0%}")
    print("=" * 60)

    # ── Test 1: all modalities active, clear happy signal ──────────────────────
    print("\n[Test 1] Happy signal across all modalities")
    audio = AudioEmotionResult(
        emotion="happy", confidence=0.75,
        all_scores={"angry":0.05,"calm":0.05,"disgust":0.03,"fear":0.04,
                    "happy":0.75,"neutral":0.05,"sad":0.02,"surprise":0.01},
    )
    facial = FacialEmotionResult(
        emotion="happy", confidence=0.80, frames_used=4,
        all_scores={"angry":0.04,"disgust":0.02,"fear":0.03,
                    "happy":0.80,"sad":0.03,"surprise":0.04,"neutral":0.04},
    )
    text = TextEmotionResult(
        emotion="joy", confidence=0.65,
        all_scores={label: 0.01 for label in [
            "admiration","amusement","anger","annoyance","approval","caring",
            "confusion","curiosity","desire","disappointment","disapproval",
            "disgust","embarrassment","excitement","fear","gratitude","grief",
            "joy","love","nervousness","optimism","pride","realization",
            "relief","remorse","sadness","surprise","neutral",
        ]} | {"joy": 0.65, "excitement": 0.15, "amusement": 0.10},
    )
    result = fuser.fuse(audio, facial, text)
    print(f"  Result: {result}")
    print(f"  Distribution: {result.distribution}")

    # ── Test 2: silent text channel ────────────────────────────────────────────
    print("\n[Test 2] Silent text (is_silent=True) — audio+facial only")
    text_silent = TextEmotionResult(
        emotion="neutral", confidence=1.0,
        all_scores={"neutral": 1.0, **{l: 0.0 for l in [
            "admiration","amusement","anger","annoyance","approval","caring",
            "confusion","curiosity","desire","disappointment","disapproval",
            "disgust","embarrassment","excitement","fear","gratitude","grief",
            "joy","love","nervousness","optimism","pride","realization",
            "relief","remorse","sadness","surprise",
        ]}},
        is_silent=True,
    )
    result2 = fuser.fuse(audio, facial, text_silent)
    print(f"  Result: {result2}")

    # ── Test 3: conflicting signals ────────────────────────────────────────────
    print("\n[Test 3] Conflicting signals — text=sad, audio+facial=happy")
    text_sad = TextEmotionResult(
        emotion="sadness", confidence=0.70,
        all_scores={label: 0.02 for label in [
            "admiration","amusement","anger","annoyance","approval","caring",
            "confusion","curiosity","desire","disappointment","disapproval",
            "disgust","embarrassment","excitement","fear","gratitude","grief",
            "joy","love","nervousness","optimism","pride","realization",
            "relief","remorse","sadness","surprise","neutral",
        ]} | {"sadness": 0.70, "grief": 0.10, "disappointment": 0.08},
    )
    result3 = fuser.fuse(audio, facial, text_sad)
    print(f"  Result: {result3}")
    print(f"  Distribution: {result3.distribution}")
