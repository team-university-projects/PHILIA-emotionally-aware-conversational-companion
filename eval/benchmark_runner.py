"""
benchmark_runner.py — Generic benchmark runner for PHILIA evaluation.

Orchestrates the predict → collect → score → save pipeline.
Modality-specific scripts call ``BenchmarkRunner.run()`` with a loader
and a predictor callable — the runner handles all the boilerplate.

Usage (from a benchmark script):
    runner = BenchmarkRunner(config, results_dir=Path("eval_results"))
    result = runner.run(
        samples        = loader.load(max_samples=args.max_samples),
        predictor_fn   = recognizer.predict,
        modality       = "audio",
        model_name     = config.models.audio_emotion_model_name,
        dataset_name   = RAVDESSLoader.DATASET_NAME,
        result_to_label= lambda r: r.emotion,     # extract str label from result object
    )
    runner.save(result)

Output files (in results_dir):
    <modality>_<timestamp>.json       — full metrics + metadata
    <modality>_<timestamp>_cm.png     — normalised confusion matrix
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from config import Config
from eval.confusion_matrix import save_confusion_matrix
from eval.metrics import compute_metrics
from eval.dataset_loader import CANONICAL_LABELS
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Full output of a single benchmark run.

    Attributes:
        modality:       ``"audio"`` | ``"text"`` | ``"facial"`` | ``"fusion"``.
        model_name:     HuggingFace model ID (used for swap tracking).
        dataset_name:   HuggingFace dataset ID used for evaluation.
        timestamp:      ISO-8601 UTC timestamp when the run completed.
        n_samples:      Number of samples evaluated.
        metrics:        Dict from ``eval.metrics.compute_metrics``.
        latency_ms_mean: Mean per-sample inference time in milliseconds.
        latency_ms_p95: 95th-percentile per-sample inference time.
        y_true:         Ground-truth label list (for regenerating CM later).
        y_pred:         Predicted label list.
    """
    modality:         str
    model_name:       str
    dataset_name:     str
    timestamp:        str
    n_samples:        int
    metrics:          dict
    latency_ms_mean:  float
    latency_ms_p95:   float
    y_true:           list[str] = field(default_factory=list)
    y_pred:           list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Runner ───────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs a benchmark and saves results to disk.

    Args:
        config:      Global :class:`~config.Config` instance.
        results_dir: Directory to write JSON + PNG outputs.
                     Overrides ``config.eval_results_dir`` if provided.
    """

    def __init__(
        self,
        config: Config,
        results_dir: Path | str | None = None,
    ) -> None:
        if results_dir is not None:
            self._results_dir = Path(results_dir)
        else:
            rd = getattr(config, "eval_results_dir", "eval_results")
            self._results_dir = Path(__file__).parent.parent / rd
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        samples:         Sequence[tuple[Any, str]],
        predictor_fn:    Callable[[Any], Any],
        modality:        str,
        model_name:      str,
        dataset_name:    str,
        result_to_label: Callable[[Any], str] = lambda r: r.emotion,
    ) -> BenchmarkResult:
        """
        Run the full predict loop and compute metrics.

        Args:
            samples:          ``[(input, canonical_label), ...]`` from a loader.
            predictor_fn:     Callable that takes one input item and returns a
                              result object with an ``.emotion`` attribute (or
                              whatever ``result_to_label`` extracts).
            modality:         Short name for the modality being tested.
            model_name:       HuggingFace model ID.
            dataset_name:     HuggingFace dataset ID.
            result_to_label:  Lambda converting the predictor output to a
                              canonical label string. Default: ``r.emotion``.

        Returns:
            :class:`BenchmarkResult` (not yet written to disk — call
            :meth:`save` afterwards).
        """
        logger.info(
            "Starting %s benchmark | model=%s | dataset=%s | n=%d",
            modality, model_name, dataset_name, len(samples),
        )

        y_true: list[str] = []
        y_pred: list[str] = []
        latencies_ms: list[float] = []
        errors = 0

        for idx, (inp, true_label) in enumerate(samples):
            t0 = time.perf_counter()
            try:
                result = predictor_fn(inp)
                pred_label = result_to_label(result)
            except Exception as exc:
                logger.warning("Sample %d failed: %s", idx, exc)
                errors += 1
                pred_label = "neutral"   # safe fallback so metrics still run

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            y_true.append(true_label)
            y_pred.append(pred_label)

            if (idx + 1) % 50 == 0:
                logger.info(
                    "  Progress: %d/%d (errors=%d)", idx + 1, len(samples), errors
                )

        logger.info("Predict loop done. Errors: %d/%d", errors, len(samples))

        # ── Metrics ──────────────────────────────────────────────────────────
        metrics = compute_metrics(y_true, y_pred, labels=CANONICAL_LABELS)

        # ── Latency stats ─────────────────────────────────────────────────────
        import statistics
        sorted_lat = sorted(latencies_ms)
        mean_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
        p95_idx  = int(len(sorted_lat) * 0.95)
        p95_ms   = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

        logger.info(
            "Latency: mean=%.1f ms  p95=%.1f ms", mean_ms, p95_ms
        )

        return BenchmarkResult(
            modality=modality,
            model_name=model_name,
            dataset_name=dataset_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_samples=len(samples),
            metrics=metrics,
            latency_ms_mean=round(mean_ms, 2),
            latency_ms_p95=round(p95_ms, 2),
            y_true=y_true,
            y_pred=y_pred,
        )

    def save(self, result: BenchmarkResult) -> tuple[Path, Path]:
        """
        Write the benchmark result to disk.

        Creates two files in ``results_dir``:
          - ``<modality>_<ts>.json``     — full metrics + metadata
          - ``<modality>_<ts>_cm.png``   — confusion matrix PNG

        Args:
            result: The :class:`BenchmarkResult` returned by :meth:`run`.

        Returns:
            ``(json_path, png_path)`` tuple.
        """
        # Sanitise timestamp for filenames (colons → dashes)
        ts_str = result.timestamp.replace(":", "-").replace("+", "Z")[:19]
        stem = f"{result.modality}_{ts_str}"

        json_path = self._results_dir / f"{stem}.json"
        png_path  = self._results_dir / f"{stem}_cm.png"

        # ── JSON ─────────────────────────────────────────────────────────────
        data = result.to_dict()
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Results saved → %s", json_path)

        # ── Confusion matrix PNG ───────────────────────────────────────────────
        title = (
            f"PHILIA — {result.modality.capitalize()} Emotion\n"
            f"{result.model_name}\n"
            f"acc={result.metrics['accuracy']:.1%}  "
            f"macro-F1={result.metrics['macro_f1']:.3f}  "
            f"n={result.n_samples}"
        )
        save_confusion_matrix(
            y_true=result.y_true,
            y_pred=result.y_pred,
            labels=CANONICAL_LABELS,
            output_path=png_path,
            title=title,
        )

        return json_path, png_path

    def print_summary(self, result: BenchmarkResult) -> None:
        """Print a formatted summary table to stdout."""
        import sys, io
        # Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

        m = result.metrics
        print()
        print("=" * 58)
        print(f"  PHILIA Benchmark -- {result.modality.upper()}")
        print(f"  Model:   {result.model_name}")
        print(f"  Dataset: {result.dataset_name}")
        print(f"  Samples: {result.n_samples}")
        print("=" * 58)
        print(f"  Accuracy      : {m['accuracy']:.1%}")
        print(f"  Macro F1      : {m['macro_f1']:.4f}")
        print(f"  Weighted F1   : {m['weighted_f1']:.4f}")
        print(f"  Macro Prec.   : {m['macro_precision']:.4f}")
        print(f"  Macro Recall  : {m['macro_recall']:.4f}")
        print(f"  Latency       : mean={result.latency_ms_mean:.0f}ms  "
              f"p95={result.latency_ms_p95:.0f}ms")
        print()
        print("  Per-class F1:")
        for lbl, s in sorted(m["per_class"].items()):
            bar = "#" * int(s["f1"] * 20)
            print(f"    {lbl:<10} {s['f1']:.3f}  {bar}")
        print("=" * 58)
        print()
