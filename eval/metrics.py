"""
metrics.py — Classification metrics for PHILIA benchmarking.

Wraps scikit-learn to produce a consistent result dict used by
BenchmarkRunner and serialised to JSON.

Usage:
    from eval.metrics import compute_metrics
    result = compute_metrics(y_true, y_pred, labels=CANONICAL_LABELS)
"""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict:
    """
    Compute classification metrics for a set of predictions.

    Args:
        y_true: Ground-truth label strings (canonical 7-label space).
        y_pred: Predicted label strings (same space).
        labels: Ordered list of canonical label names to report on.

    Returns:
        dict with keys:
            - ``accuracy``         (float)
            - ``macro_precision``  (float)
            - ``macro_recall``     (float)
            - ``macro_f1``         (float)
            - ``weighted_f1``      (float)
            - ``per_class``        (dict label → {precision, recall, f1, support})
            - ``n_samples``        (int)
            - ``n_correct``        (int)
    """
    labels_list = list(labels)
    n = len(y_true)

    accuracy        = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, labels=labels_list,
                                            average="macro", zero_division=0))
    macro_recall    = float(recall_score(y_true, y_pred, labels=labels_list,
                                         average="macro", zero_division=0))
    macro_f1        = float(f1_score(y_true, y_pred, labels=labels_list,
                                     average="macro", zero_division=0))
    weighted_f1     = float(f1_score(y_true, y_pred, labels=labels_list,
                                     average="weighted", zero_division=0))

    # Per-class breakdown via classification_report (dict output)
    report = classification_report(
        y_true, y_pred,
        labels=labels_list,
        output_dict=True,
        zero_division=0,
    )
    per_class: dict[str, dict] = {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"],    4),
            "f1":        round(report[lbl]["f1-score"],  4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in labels_list
        if lbl in report
    }

    n_correct = int(round(accuracy * n))

    metrics = {
        "accuracy":         round(accuracy,        4),
        "macro_precision":  round(macro_precision,  4),
        "macro_recall":     round(macro_recall,     4),
        "macro_f1":         round(macro_f1,         4),
        "weighted_f1":      round(weighted_f1,      4),
        "per_class":        per_class,
        "n_samples":        n,
        "n_correct":        n_correct,
    }

    logger.info(
        "Metrics: acc=%.1f%%  macro_f1=%.3f  weighted_f1=%.3f  (n=%d)",
        accuracy * 100, macro_f1, weighted_f1, n,
    )
    return metrics
