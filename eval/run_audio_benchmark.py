"""
run_audio_benchmark.py — Benchmark AudioEmotionRecognizer on RAVDESS.

Usage:
    python -m eval.run_audio_benchmark
    python -m eval.run_audio_benchmark --max-samples 50
    python -m eval.run_audio_benchmark --results-dir path/to/results

The script loads the RAVDESS test split (or 10% of train if no test split
is available), runs each WAV clip through AudioEmotionRecognizer, and saves:
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
from eval.dataset_loader import RAVDESSLoader, _RAVDESS_STR_TO_CANONICAL
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark AudioEmotionRecognizer on RAVDESS."
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
    loader = RAVDESSLoader()
    samples = loader.load(max_samples=args.max_samples)

    if not samples:
        print("[ERROR] No RAVDESS samples loaded. Check your internet connection.")
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
        dataset_name   = RAVDESSLoader.DATASET_NAME,
        # The RAVDESS model may output "calm" — collapse to canonical 7-label
        # space (calm -> neutral) to match y_true from the loader.
        result_to_label= lambda r: _RAVDESS_STR_TO_CANONICAL.get(r.emotion, r.emotion),
    )

    # ── Print + save ──────────────────────────────────────────────────────────
    runner.print_summary(result)
    json_path, png_path = runner.save(result)

    print(f"  Results  → {json_path}")
    print(f"  CM plot  → {png_path}\n")


if __name__ == "__main__":
    main()
