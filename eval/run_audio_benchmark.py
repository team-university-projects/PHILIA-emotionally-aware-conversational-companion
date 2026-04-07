"""
run_audio_benchmark.py — Benchmark AudioEmotionRecognizer on RAVDESS or MELD.

Usage:
    python -m eval.run_audio_benchmark                  # evaluates on MELD test split (default)
    python -m eval.run_audio_benchmark --dataset ravdess
    python -m eval.run_audio_benchmark --max-samples 200
    python -m eval.run_audio_benchmark --results-dir path/to/results

Saves:
  - eval_results/audio_<timestamp>.json   (metrics + per-class breakdown)
  - eval_results/audio_<timestamp>_cm.png (normalised confusion matrix)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as __main__
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── cuDNN / CUDA DLL registration (mirrors main.py) ──────────────────────────
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
from emotion.audio_emotion import AudioEmotionRecognizer
from eval.benchmark_runner import BenchmarkRunner
from eval.dataset_loader import RAVDESSLoader, MELDLoader, _RAVDESS_STR_TO_CANONICAL
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark AudioEmotionRecognizer on RAVDESS or MELD."
    )
    p.add_argument(
        "--dataset", choices=["ravdess", "meld"], default="meld",
        help="Which dataset to evaluate on (default: meld).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of test samples (default: all).",
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to save JSON + PNG results (default: eval_results/).",
    )
    return p.parse_args()


def main() -> None:
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    config = Config.load()

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.dataset == "meld":
        loader = MELDLoader()
        dataset_name = MELDLoader.DATASET_NAME
        # MELD labels are already canonical — pass emotion directly
        result_to_label = lambda r: r.emotion
    else:
        loader = RAVDESSLoader()
        dataset_name = RAVDESSLoader.DATASET_NAME
        # RAVDESS models may output 'calm', 'fearful', 'surprised' — normalise
        result_to_label = lambda r: _RAVDESS_STR_TO_CANONICAL.get(r.emotion, r.emotion)

    samples = loader.load(max_samples=args.max_samples)

    if not samples:
        print(f"[ERROR] No {args.dataset.upper()} samples loaded.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading AudioEmotionRecognizer ({config.models.audio_emotion_model_name})...")
    recognizer = AudioEmotionRecognizer(config)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    runner = BenchmarkRunner(config, results_dir=args.results_dir)
    result = runner.run(
        samples        = samples,
        predictor_fn   = recognizer.predict,
        modality       = "audio",
        model_name     = config.models.audio_emotion_model_name,
        dataset_name   = dataset_name,
        result_to_label= result_to_label,
    )

    # ── Print + save ──────────────────────────────────────────────────────────
    runner.print_summary(result)
    json_path, png_path = runner.save(result)

    print(f"  Results  -> {json_path}")
    print(f"  CM plot  -> {png_path}\n")


if __name__ == "__main__":
    main()
