"""
dataset_loader.py — HuggingFace dataset loaders for PHILIA benchmarking.

Each loader streams the test split of its native dataset and returns
(input, canonical_label) pairs ready for the benchmark runner.

All labels are collapsed to the 7-label canonical set used by fusion.py:
    angry · disgust · fear · happy · neutral · sad · surprise

Loaders
-------
RAVDESSLoader    — zenodo:ravdess (direct download) → (wav_path: Path, label: str)
GoEmotionsLoader — google-research-datasets/go_emotions (HF)
                 → (text: str, label: str)
FER2013Loader    — 3una/Fer2013 (HF)
                 → (image: PIL.Image, label: str)

Design
------
- All loaders accept ``max_samples: int | None`` for quick smoke tests.
- WAV files for RAVDESS are cached to a local tmp/ directory so the model
  can read them as paths (AudioEmotionRecognizer.predict expects a file path).
- Each loader exposes a single ``load()`` method returning a list of tuples.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Canonical label space ────────────────────────────────────────────────────

CANONICAL_LABELS: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)


# ── RAVDESS → canonical ──────────────────────────────────────────────────────
# RAVDESS emotion codes (from the filename convention):
#   01 neutral  02 calm  03 happy  04 sad  05 angry
#   06 fearful  07 disgust  08 surprised

_RAVDESS_INT_TO_CANONICAL: dict[int, str] = {
    0: "neutral",   # neutral
    1: "neutral",   # calm → neutral
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "surprise",
}

_RAVDESS_STR_TO_CANONICAL: dict[str, str] = {
    "neutral":   "neutral",
    "calm":      "neutral",
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "fearful":   "fear",
    "disgust":   "disgust",
    "surprised": "surprise",
    "fear":      "fear",
    "surprise":  "surprise",
}


# ── GoEmotions (28 classes) → canonical ─────────────────────────────────────
_GOEMOTIONS_TO_CANONICAL: dict[str, str] = {
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


# ── FER2013 → canonical ──────────────────────────────────────────────────────
_FER_TO_CANONICAL: dict[str, str] = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "sad":      "sad",
    "surprise": "surprise",
    "neutral":  "neutral",
}


# ── Helper ───────────────────────────────────────────────────────────────────

def _datasets_import():
    """Import 'datasets' lazily so the module loads even if not installed."""
    try:
        import datasets as hf_datasets
        return hf_datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for benchmarking.\n"
            "Install it with:  pip install datasets"
        ) from exc


# ── RAVDESS Loader ────────────────────────────────────────────────────────────

class RAVDESSLoader:
    """
    Loads RAVDESS speech WAV files from the official Zenodo archive.

    RAVDESS is freely available at https://zenodo.org/record/1188976
    A single ZIP contains all 24 actors' speech files (208 MB).
    Labels are derived from the RAVDESS filename encoding:

        03-01-<emotion>-...-<actor>.wav
          ^^ modality 03 = speech (02 = singing, skipped)
                ^^^^^^^ emotion code (01-08)

    Args:
        cache_dir:  Directory to cache the downloaded ZIP and extracted WAV files.
                    Defaults to ``tmp/ravdess_cache`` under the project root.
        n_actors:   Number of actors (1-24) whose clips to include. Default: 4
                    (gives ~120 WAV clips).
    """

    DATASET_NAME = "zenodo:ravdess"
    _ZENODO_URL = (
        "https://zenodo.org/record/1188976/files/"
        "Audio_Speech_Actors_01-24.zip"
    )
    _ZIP_NAME = "Audio_Speech_Actors_01-24.zip"

    _CODE_TO_CANONICAL: dict[int, str] = {
        1: "neutral", 2: "neutral",
        3: "happy",   4: "sad",
        5: "angry",   6: "fear",
        7: "disgust", 8: "surprise",
    }

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        n_actors: int = 4,
    ) -> None:
        project_root = Path(__file__).parent.parent
        self._cache_dir = (
            Path(cache_dir) if cache_dir
            else project_root / "tmp" / "ravdess_cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._n_actors = max(1, min(n_actors, 24))

    def load(
        self, max_samples: int | None = None
    ) -> list[tuple[Path, str]]:
        """Download RAVDESS ZIPs from Zenodo, extract WAV files, return labelled tuples."""
        import urllib.request
        import zipfile

        zip_path = self._cache_dir / self._ZIP_NAME

        if not zip_path.exists():
            logger.info("Downloading %s (208 MB) from Zenodo...", self._ZIP_NAME)
            try:
                urllib.request.urlretrieve(self._ZENODO_URL, zip_path)
                logger.info(
                    "Saved %s (%.1f MB)",
                    self._ZIP_NAME, zip_path.stat().st_size / 1e6,
                )
            except Exception as exc:
                logger.error("Failed to download RAVDESS ZIP: %s", exc)
                return []

        samples: list[tuple[Path, str]] = []

        try:
            with zipfile.ZipFile(zip_path) as zf:
                wav_names = [n for n in zf.namelist() if n.endswith(".wav")]
                for wname in wav_names:
                    base = Path(wname).name
                    parts = base.split("-")
                    if len(parts) < 7:
                        continue
                    try:
                        modality     = int(parts[0])
                        emotion_code = int(parts[2])
                        actor_id     = int(parts[6].replace(".wav", ""))
                    except ValueError:
                        continue

                    if modality != 3:               # skip singing (02)
                        continue
                    if actor_id > self._n_actors:   # respect n_actors limit
                        continue

                    canonical = self._CODE_TO_CANONICAL.get(emotion_code)
                    if canonical is None:
                        continue

                    out_path = self._cache_dir / base
                    if not out_path.exists():
                        out_path.write_bytes(zf.read(wname))

                    samples.append((out_path, canonical))

        except zipfile.BadZipFile as exc:
            logger.error("Bad ZIP file — deleting for re-download: %s", exc)
            zip_path.unlink(missing_ok=True)
            return []

        import random as _rnd
        _rnd.shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(
            "RAVDESS (Zenodo): %d samples loaded (n_actors<=%d).",
            len(samples), self._n_actors,
        )
        return samples


# ── GoEmotions Loader ─────────────────────────────────────────────────────────

class GoEmotionsLoader:
    """
    Loads the GoEmotions test split from HuggingFace
    (``google-research-datasets/go_emotions``, simplified subset).

    Multi-label samples are included using only the first listed label,
    consistent with how the model was trained (single-label output).

    Returns:
        List of ``(text: str, canonical_label: str)`` tuples.
    """

    DATASET_NAME = "google-research-datasets/go_emotions"

    def load(
        self, max_samples: int | None = None
    ) -> list[tuple[str, str]]:
        hf = _datasets_import()
        logger.info("Loading GoEmotions test split from HuggingFace...")
        ds = hf.load_dataset(self.DATASET_NAME, "simplified", split="test",
                             trust_remote_code=True)

        label_names: list[str] = ds.features["labels"].feature.names

        samples: list[tuple[str, str]] = []
        skipped = 0

        for row in ds:
            if max_samples is not None and len(samples) >= max_samples:
                break

            text: str = row["text"]
            label_ids: list[int] = row["labels"]

            if not label_ids:
                skipped += 1
                continue

            raw_label = label_names[label_ids[0]]
            canonical = _GOEMOTIONS_TO_CANONICAL.get(raw_label)
            if canonical is None:
                skipped += 1
                continue

            if not text or not text.strip():
                skipped += 1
                continue

            samples.append((text, canonical))

        logger.info(
            "GoEmotions: %d samples loaded, %d skipped.", len(samples), skipped
        )
        return samples


# ── FER2013 Loader ────────────────────────────────────────────────────────────

class FER2013Loader:
    """
    Loads the FER2013 test split from HuggingFace (``3una/Fer2013``).

    Returns PIL Images (already face-cropped in the dataset) paired with
    canonical labels.  The facial benchmark script passes these images
    directly to the ViT pipeline, bypassing VideoCapture.

    Returns:
        List of ``(image: PIL.Image.Image, canonical_label: str)`` tuples.
    """

    DATASET_NAME = "3una/Fer2013"

    def load(
        self, max_samples: int | None = None
    ) -> list[tuple[Image.Image, str]]:
        hf = _datasets_import()
        logger.info("Loading FER2013 test split from HuggingFace (3una/Fer2013)...")
        ds = hf.load_dataset(self.DATASET_NAME, split="test")

        label_feature = ds.features.get("label")
        if label_feature is not None and hasattr(label_feature, "names"):
            label_names: list[str] = label_feature.names
        else:
            label_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Load ALL samples, then shuffle + trim.
        # The test set is sorted by label — taking first N without shuffling
        # yields a heavily biased (all-one-class) subset.
        all_samples: list[tuple[Image.Image, str]] = []
        skipped = 0

        for row in ds:
            label_id: int = row["label"]
            if label_id < 0 or label_id >= len(label_names):
                skipped += 1
                continue

            raw_label = label_names[label_id].lower()
            canonical = _FER_TO_CANONICAL.get(raw_label)
            if canonical is None:
                skipped += 1
                continue

            img = row.get("image", None)
            if img is None:
                skipped += 1
                continue

            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    skipped += 1
                    continue

            if img.mode != "RGB":
                img = img.convert("RGB")

            all_samples.append((img, canonical))

        import random as _rnd
        _rnd.seed(42)
        _rnd.shuffle(all_samples)

        samples = all_samples[:max_samples] if max_samples is not None else all_samples

        logger.info(
            "FER2013: %d samples loaded, %d skipped.", len(samples), skipped
        )
        return samples
