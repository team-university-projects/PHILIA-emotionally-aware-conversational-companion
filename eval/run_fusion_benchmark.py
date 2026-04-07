"""
run_fusion_benchmark.py — Benchmark EmotionFuser on synthetic multimodal samples.

Usage:
    python -m eval.run_fusion_benchmark
    python -m eval.run_fusion_benchmark --max-samples 100
    python -m eval.run_fusion_benchmark --results-dir path/to/results

Phase 1 rationale
-----------------
A proper multimodal ground-truth dataset (CMU-MOSI, MELD, IEMOCAP) requires
significant alignment work and will be used in Phase 3.  For now this script
constructs synthetic fusion samples by drawing one sample from each modality
loader (with matching canonical label) and runs them through the full
EmotionFuser pipeline.  This validates that:
  1. All three emotion recognisers are loaded and functional together.
  2. The fusion pipeline routes, weights, and collapses correctly.
  3. The benchmark framework (runner → JSON + CM) works end-to-end.

The "ground truth" for each sample is the label that all three modality loaders
agree on.  Only samples where all three loaders return the same canonical label
are used — this avoids injecting synthetic noise as ground truth.

Output:
  - eval_results/fusion_<timestamp>.json
  - eval_results/fusion_<timestamp>_cm.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os as _os
import pathlib as _pathlib
for _sp in sys.path:
    _torch_lib = _pathlib.Path(_sp) / "torch" / "lib"
    if _torch_lib.is_dir():
        if hasattr(_os, "add_dll_directory"):
            _os.add_dll_directory(str(_torch_lib))
        break
del _pathlib, _sp, _torch_lib

import random
from dataclasses import dataclass
from typing import Any

from config import Config
from emotion.audio_emotion import AudioEmotionRecognizer
from emotion.facial_emotion import FacialEmotionRecognizer, FACIAL_EMOTION_LABELS
from emotion.text_emotion import TextEmotionRecognizer
from emotion.fusion import EmotionFuser
from eval.benchmark_runner import BenchmarkRunner
from eval.dataset_loader import (
    RAVDESSLoader, GoEmotionsLoader, FER2013Loader,
    CANONICAL_LABELS, _FER_TO_CANONICAL,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Same image-direct predictor as in run_facial_benchmark ────────────────────
class _ImageFacialPredictor:
    def __init__(self, recognizer: FacialEmotionRecognizer) -> None:
        self._pipe = recognizer._pipe

    def predict(self, image: Any) -> Any:
        from emotion.facial_emotion import FacialEmotionResult
        raw: list[dict] = self._pipe(image)
        all_scores = {item["label"]: round(float(item["score"]), 4) for item in raw}
        top = max(raw, key=lambda x: x["score"])
        return FacialEmotionResult(
            emotion=top["label"],
            confidence=round(float(top["score"]), 4),
            all_scores=all_scores,
            frames_used=1,
        )


@dataclass
class _FusionInput:
    audio_path: Path
    text: str
    image: Any      # PIL.Image
    true_label: str


def _build_synthetic_samples(
    audio_samples: list,
    text_samples:  list,
    image_samples: list,
    max_samples:   int | None,
) -> list[_FusionInput]:
    """
    Pair samples from each modality that share the same canonical label.
    This avoids injecting label noise into the fusion ground truth.
    """
    # Group by canonical label
    from collections import defaultdict
    audio_by_label: dict[str, list] = defaultdict(list)
    text_by_label:  dict[str, list] = defaultdict(list)
    image_by_label: dict[str, list] = defaultdict(list)

    for wav, lbl in audio_samples:
        audio_by_label[lbl].append(wav)
    for txt, lbl in text_samples:
        text_by_label[lbl].append(txt)
    for img, lbl in image_samples:
        image_by_label[lbl].append(img)

    samples: list[_FusionInput] = []
    common_labels = (
        set(audio_by_label) & set(text_by_label) & set(image_by_label)
    )

    logger.info("Labels with data in all three modalities: %s", sorted(common_labels))

    for label in sorted(common_labels):
        a_list = audio_by_label[label]
        t_list = text_by_label[label]
        i_list = image_by_label[label]
        n = min(len(a_list), len(t_list), len(i_list))
        for idx in range(n):
            samples.append(_FusionInput(
                audio_path  = a_list[idx],
                text        = t_list[idx],
                image       = i_list[idx],
                true_label  = label,
            ))

    random.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info("Synthetic fusion samples built: %d", len(samples))
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fusion benchmark on synthetic multimodal samples."
    )
    p.add_argument("--dataset", choices=["synthetic", "meld-synthetic"], default="meld-synthetic",
                   help="Which dataset to evaluate on. 'meld-synthetic' uses aligned MELD audio+text and synthetic FER2013 images.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit the number of fusion samples (default: all).")
    p.add_argument("--max-per-modality", type=int, default=300,
                   help="Max samples to load per modality loader (default: 300).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory (default: eval_results/).")
    return p.parse_args()


def main() -> None:
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args  = parse_args()
    config = Config.load()

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.dataset == "meld-synthetic":
        from eval.dataset_loader import MELDFusionLoader
        loader = MELDFusionLoader()
        raw_samples = loader.load(max_samples=args.max_samples)
        if not raw_samples:
             print("[ERROR] No MELDFusion samples loaded. Check dataset loaders.")
             sys.exit(1)
        fusion_inputs = [
            _FusionInput(audio_path=w, text=t, image=i, true_label=lbl)
            for w, t, i, lbl in raw_samples
        ]
        dataset_name = "MELD-Synthetic (Aligned Audio+Text, Synthetic FER Image)"
    else:
        print("\nLoading RAVDESS (audio)...")
        audio_samples = RAVDESSLoader().load(max_samples=args.max_per_modality)

        print("Loading GoEmotions (text)...")
        text_samples  = GoEmotionsLoader().load(max_samples=args.max_per_modality)

        print("Loading FER2013 (facial)...")
        image_samples = FER2013Loader().load(max_samples=args.max_per_modality)

        fusion_inputs = _build_synthetic_samples(
            audio_samples, text_samples, image_samples, args.max_samples
        )

        if not fusion_inputs:
            print("[ERROR] No fusion samples could be constructed. Check dataset loaders.")
            sys.exit(1)
        dataset_name = "Synthetic (RAVDESS + GoEmotions + FER2013)"

    # ── Load all models ───────────────────────────────────────────────────────
    print(f"\nLoading AudioEmotionRecognizer ({config.models.audio_emotion_model_name})...")
    audio_emo  = AudioEmotionRecognizer(config)

    print(f"Loading FacialEmotionRecognizer ({config.models.facial_emotion_model_name})...")
    facial_emo = FacialEmotionRecognizer(config)
    facial_img = _ImageFacialPredictor(facial_emo)

    print(f"Loading TextEmotionRecognizer ({config.models.text_emotion_model_name})...")
    text_emo   = TextEmotionRecognizer(config)

    fuser = EmotionFuser(config)

    # ── Single predictor_fn for the runner ───────────────────────────────────
    def fuse_sample(inp: _FusionInput):
        audio_result  = audio_emo.predict(inp.audio_path)
        text_result   = text_emo.predict(inp.text)
        facial_result = facial_img.predict(inp.image)
        return fuser.fuse(audio_result, facial_result, text_result)

    # ── Wrap fusion inputs so runner sees (input, label) tuples ───────────────
    runner_samples = [(fi, fi.true_label) for fi in fusion_inputs]

    # ── Run benchmark ─────────────────────────────────────────────────────────
    runner = BenchmarkRunner(config, results_dir=args.results_dir)
    result = runner.run(
        samples        = runner_samples,
        predictor_fn   = fuse_sample,
        modality       = "fusion",
        model_name     = (
            f"audio={config.models.audio_emotion_model_name} | "
            f"text={config.models.text_emotion_model_name} | "
            f"facial={config.models.facial_emotion_model_name}"
        ),
        dataset_name   = dataset_name,
        result_to_label= lambda r: r.label,
    )

    runner.print_summary(result)
    json_path, png_path = runner.save(result)

    print(f"  Results  -> {json_path}")
    print(f"  CM plot  -> {png_path}\n")


if __name__ == "__main__":
    main()
