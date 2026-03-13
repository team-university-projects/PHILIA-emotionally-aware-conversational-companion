"""
run_text_benchmark.py — Benchmark TextEmotionRecognizer on GoEmotions.

Usage:
    python -m eval.run_text_benchmark
    python -m eval.run_text_benchmark --max-samples 200
    python -m eval.run_text_benchmark --results-dir path/to/results

Loads the GoEmotions (simplified) test split, collapses 28 labels → 7
canonical labels (same mapping as fusion.py), then runs each text sample
through TextEmotionRecognizer. Saves:
  - eval_results/text_<timestamp>.json
  - eval_results/text_<timestamp>_cm.png
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

from config import Config
from emotion.text_emotion import TextEmotionRecognizer
from eval.benchmark_runner import BenchmarkRunner
from eval.dataset_loader import GoEmotionsLoader, _GOEMOTIONS_TO_CANONICAL
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark TextEmotionRecognizer on GoEmotions test split."
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit the number of test samples (default: all).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory (default: eval_results/).")
    return p.parse_args()


def main() -> None:
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args = parse_args()
    config = Config.load()

    # ── Load dataset ─────────────────────────────────────────────────────────
    loader = GoEmotionsLoader()
    samples = loader.load(max_samples=args.max_samples)

    if not samples:
        print("[ERROR] No GoEmotions samples loaded.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading TextEmotionRecognizer ({config.models.text_emotion_model_name})...")
    recognizer = TextEmotionRecognizer(config)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    runner = BenchmarkRunner(config, results_dir=args.results_dir)
    result = runner.run(
        samples        = samples,
        predictor_fn   = recognizer.predict,
        modality       = "text",
        model_name     = config.models.text_emotion_model_name,
        dataset_name   = GoEmotionsLoader.DATASET_NAME,
        # The model outputs raw GoEmotions fine-grained labels (e.g. "sadness",
        # "joy") — collapse to canonical 7-label space to match y_true.
        result_to_label= lambda r: _GOEMOTIONS_TO_CANONICAL.get(r.emotion, "neutral"),
    )

    runner.print_summary(result)
    json_path, png_path = runner.save(result)

    print(f"  Results  → {json_path}")
    print(f"  CM plot  → {png_path}\n")


if __name__ == "__main__":
    main()
