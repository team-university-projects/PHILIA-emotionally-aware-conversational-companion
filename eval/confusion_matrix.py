"""
confusion_matrix.py — Confusion matrix visualisation for PHILIA benchmarks.

Generates and saves a publication-quality confusion matrix PNG using
matplotlib / sklearn.metrics.ConfusionMatrixDisplay.

Usage:
    from eval.confusion_matrix import save_confusion_matrix
    save_confusion_matrix(y_true, y_pred, labels, Path("out.png"), "Audio — Wav2Vec2")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.logger import get_logger

logger = get_logger(__name__)

# PHILIA colour palette (dark background, teal accent)
_BG_COLOUR    = "#0f1117"
_TEXT_COLOUR  = "#e8eaf6"
_CMAP         = "Blues"


def save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    output_path: Path | str,
    title: str = "",
) -> None:
    """
    Render a normalised confusion matrix and save it as a PNG.

    The matrix is row-normalised (recall proportions) so that class-imbalance
    does not hide per-class performance — each row sums to 1.0.

    Args:
        y_true:      Ground-truth label strings.
        y_pred:      Predicted label strings.
        labels:      Ordered canonical label names (rows/columns).
        output_path: Where to write the PNG file.
        title:       Plot title (model name / modality shown at the top).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_list = list(labels)

    # Compute raw matrix then normalise by row (true label totals)
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # rows with 0 samples → 0

    # ── Figure setup ──────────────────────────────────────────────────────────
    n = len(labels_list)
    fig_size = max(6, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    fig.patch.set_facecolor(_BG_COLOUR)
    ax.set_facecolor(_BG_COLOUR)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=labels_list,
    )
    disp.plot(
        ax=ax,
        cmap=_CMAP,
        colorbar=False,
        values_format=".2f",
        xticks_rotation=45,
    )

    # ── Colour the text in each cell for contrast ─────────────────────────────
    for text in disp.text_.ravel():
        val = float(text.get_text()) if text.get_text() else 0.0
        text.set_color("white" if val < 0.55 else "black")
        text.set_fontsize(8)

    # ── Labels & styling ──────────────────────────────────────────────────────
    ax.set_xlabel("Predicted label", color=_TEXT_COLOUR, fontsize=11)
    ax.set_ylabel("True label",      color=_TEXT_COLOUR, fontsize=11)
    ax.tick_params(colors=_TEXT_COLOUR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_TEXT_COLOUR)

    # Colour matrix grid lines
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title:
        ax.set_title(title, color=_TEXT_COLOUR, fontsize=13, pad=14, fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOUR)
    plt.close(fig)

    logger.info("Confusion matrix saved → %s", output_path)
