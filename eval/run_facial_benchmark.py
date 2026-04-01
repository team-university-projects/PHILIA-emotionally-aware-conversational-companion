"""
run_facial_benchmark.py — Benchmark FacialEmotionRecognizer on FER2013.

Usage:
    python -m eval.run_facial_benchmark
    python -m eval.run_facial_benchmark --max-samples 100
    python -m eval.run_facial_benchmark --results-dir path/to/results

FER2013 is an image dataset (no video). This script bypasses
VideoCapture and passes the pre-cropped PIL images from the dataset
directly to the ViT pipeline inside FacialEmotionRecognizer — using
a thin adapter class so the standard BenchmarkRunner interface is
preserved (single predictor_fn call per sample).

Saves:
  - eval_results/facial_<timestamp>.json
  - eval_results/facial_<timestamp>_cm.png
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

from dataclasses import dataclass
from typing import Dict

from PIL import Image

from config import Config
from emotion.facial_emotion import FacialEmotionRecognizer, FACIAL_EMOTION_LABELS
from eval.benchmark_runner import BenchmarkRunner
from eval.dataset_loader import FER2013Loader, CANONICAL_LABELS, _FER_TO_CANONICAL
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Image-direct adapter ──────────────────────────────────────────────────────
# FacialEmotionRecognizer.predict() normally takes a video path.
# For the FER2013 benchmark we need to call the internal ViT pipeline
# directly with a PIL image instead.

@dataclass
class _ImageEmotionResult:
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    is_no_face: bool = False


class _ImageFacialPredictor:
    """
    Thin wrapper around FacialEmotionRecognizer that accepts a PIL Image
    instead of a video path.  Calls the internal ``_pipe`` directly.

    This is valid because FER2013 images are already face-cropped.
    """

    def __init__(self, recognizer: FacialEmotionRecognizer) -> None:
        self._pipe = recognizer._pipe

    def predict(self, image: Image.Image) -> _ImageEmotionResult:
        raw: list[dict] = self._pipe(image)
        all_scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4)
            for item in raw
        }
        top = max(raw, key=lambda x: x["score"])
        return _ImageEmotionResult(
            emotion=top["label"],
            confidence=round(float(top["score"]), 4),
            all_scores=all_scores,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark facial emotion recognition on FER2013."
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit the number of test samples (default: all).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory (default: eval_results/).")
    return p.parse_args()


def _map_to_canonical(label: str) -> str:
    """Map the model's FER label back to the canonical 7-label space.
    Falls back to the lowercased label itself (not neutral) so unknown
    labels surface as wrong predictions rather than inflating neutral's score.
    """
    return _FER_TO_CANONICAL.get(label.lower(), label.lower())


def main() -> None:
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args = parse_args()
    config = Config.load()

    # ── Load dataset ─────────────────────────────────────────────────────────
    loader = FER2013Loader()
    samples = loader.load(max_samples=args.max_samples)

    if not samples:
        print("[ERROR] No FER2013 samples loaded.")
        sys.exit(1)

    # ── Load model and create image-direct predictor ───────────────────────────
    print(f"\nLoading FacialEmotionRecognizer ({config.models.facial_emotion_model_name})...")
    recognizer = FacialEmotionRecognizer(config)
    predictor  = _ImageFacialPredictor(recognizer)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    runner = BenchmarkRunner(config, results_dir=args.results_dir)
    result = runner.run(
        samples        = samples,
        predictor_fn   = predictor.predict,
        modality       = "facial",
        model_name     = config.models.facial_emotion_model_name,
        dataset_name   = FER2013Loader.DATASET_NAME,
        result_to_label= lambda r: _map_to_canonical(r.emotion),
    )

    runner.print_summary(result)
    json_path, png_path = runner.save(result)

    print(f"  Results  → {json_path}")
    print(f"  CM plot  → {png_path}\n")


if __name__ == "__main__":
    main()
