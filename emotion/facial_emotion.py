"""
facial_emotion.py — Facial expression emotion recognition via ViT.

Responsibilities:
  - Load a pretrained ViT facial emotion model once at startup
  - Accept an MP4 video path (the PTT recording)
  - Sample N evenly-spaced frames from the video
  - Detect faces with MediaPipe BlazeFace (Tasks API, 0.10+)
  - Run a single batched ViT inference pass over all face crops at once
  - Average per-label scores across frames and return a structured result

Model: mo-thecreator/vit-Facial-Expression-Recognition
  - Vision Transformer fine-tuned on FER2013 + MMI + AffectNet
  - 84.34% accuracy, 7 emotion labels
  - ~330 MB download, cached after first use

Face detection: MediaPipe BlazeFace short-range (Tasks API, mediapipe>=0.10)
  - TFLite model auto-downloaded to models/blaze_face_short_range.tflite
  - < 5ms per frame, handles off-angle faces Haar cascade misses

Batched inference:
  - All N face crops are passed to the ViT pipeline in a single GPU
    forward pass — ~3-4x faster than calling the pipeline in a loop.
"""

from __future__ import annotations

import ctypes
import platform
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from transformers import pipeline

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Hugging Face model identifier
_MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

# Number of evenly-spaced frames to sample from the video
_NUM_FRAMES = 5

# BlazeFace TFLite model (short-range, optimised for webcam conversations)
_BLAZE_FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_BLAZE_FACE_LOCAL = Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite"

# MediaPipe face detection confidence threshold
_MIN_DETECTION_CONFIDENCE = 0.5

# The 7 FER emotion labels this model outputs
FACIAL_EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


@dataclass
class FacialEmotionResult:
    """
    Structured output from :class:`FacialEmotionRecognizer`.

    Attributes:
        emotion:     Dominant emotion label averaged across sampled frames.
        confidence:  Score for the top emotion (0.0 – 1.0).
        all_scores:  Full 7-label score dict (averaged across frames) for fusion.
        frames_used: Number of frames where a face was successfully detected.
        is_no_face:  ``True`` when no face was detected in any sampled frame.
    """

    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    frames_used: int = 0
    is_no_face: bool = False

    def __str__(self) -> str:
        return (
            f"{self.emotion} ({self.confidence:.1%})"
            f" [faces in {self.frames_used}/{_NUM_FRAMES} frames]"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _no_face_result() -> FacialEmotionResult:
    """Return a full-confidence neutral result when no face is detected."""
    scores = {label: 0.0 for label in FACIAL_EMOTION_LABELS}
    scores["neutral"] = 1.0
    return FacialEmotionResult(
        emotion="neutral",
        confidence=1.0,
        all_scores=scores,
        frames_used=0,
        is_no_face=True,
    )


def _resolve_device(requested: str) -> str:
    """
    Return ``"cuda"`` if requested and the CUDA 12 runtime is available,
    otherwise ``"cpu"``. Mirrors the same probe used in all emotion modules.
    """
    if requested != "cuda":
        return "cpu"

    dll = "cublas64_12.dll" if platform.system() == "Windows" else "libcublas.so.12"
    try:
        if platform.system() == "Windows":
            ctypes.windll.LoadLibrary(dll)
        else:
            ctypes.CDLL(dll)
        return "cuda"
    except OSError:
        logger.warning(
            "CUDA requested but CUDA 12 runtime not found — "
            "facial emotion will run on CPU."
        )
        return "cpu"


def _sample_frame_indices(total_frames: int, n: int) -> list[int]:
    """Return *n* evenly-spaced frame indices within [0, total_frames)."""
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(step * i + step / 2) for i in range(n)]


def _ensure_blaze_face_model() -> Path:
    """
    Ensure the BlazeFace TFLite model file exists locally.
    Downloads from Google's model storage on first use (~800 KB).
    """
    if _BLAZE_FACE_LOCAL.exists():
        return _BLAZE_FACE_LOCAL

    _BLAZE_FACE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading BlazeFace model to %s ...", _BLAZE_FACE_LOCAL
    )
    urllib.request.urlretrieve(_BLAZE_FACE_URL, _BLAZE_FACE_LOCAL)
    logger.info("BlazeFace model downloaded (%.0f KB).",
                _BLAZE_FACE_LOCAL.stat().st_size / 1024)
    return _BLAZE_FACE_LOCAL


# ── Main class ─────────────────────────────────────────────────────────────────

class FacialEmotionRecognizer:
    """
    Predicts facial emotion by sampling frames from an MP4 video.

    Optimisations vs. Haar cascade + loop:
    - MediaPipe BlazeFace (Tasks API): < 5ms/frame, better angle handling
    - Single batched ViT pass: all face crops inferred together on the GPU

    All heavy objects are loaded once in ``__init__`` and reused.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(
            config.models, "facial_emotion_model_name", _MODEL_NAME
        )
        requested_device: str = getattr(config.models, "device", "cpu")
        device = _resolve_device(requested_device)
        pipeline_device = 0 if device == "cuda" else -1

        logger.info(
            "Loading facial emotion model '%s' on %s ...", model_name, device
        )
        self._pipe = pipeline(
            task="image-classification",
            model=model_name,
            top_k=None,           # Return all 7 label scores
            device=pipeline_device,
        )
        logger.info("Facial emotion ViT model ready.")

        # MediaPipe BlazeFace detector (Tasks API — mediapipe >= 0.10)
        tflite_path = _ensure_blaze_face_model()
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(tflite_path)
        )
        detector_options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
        )
        self._face_detector = mp_vision.FaceDetector.create_from_options(
            detector_options
        )
        logger.info("MediaPipe BlazeFace detector ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, video_path: str | Path) -> FacialEmotionResult:
        """
        Run facial emotion inference on an MP4 video file.

        Samples :data:`_NUM_FRAMES` evenly-spaced frames, detects faces
        with MediaPipe BlazeFace, then runs a single batched ViT inference
        over all face crops. Scores are averaged across all faces found.

        Args:
            video_path: Path to the ``.mp4`` file produced by VideoCapture.

        Returns:
            :class:`FacialEmotionResult` with dominant emotion, confidence,
            and full 7-label average score dict for fusion.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error("Video file not found: %s — returning no-face.", path)
            return _no_face_result()

        frames = self._sample_frames(path)
        if not frames:
            logger.warning("Could not read any frames from %s.", path.name)
            return _no_face_result()

        # Detect faces across all sampled frames → collect PIL crops
        face_crops: list[Image.Image] = []
        for frame_bgr in frames:
            crop = self._detect_face(frame_bgr)
            if crop is not None:
                face_crops.append(crop)

        frames_used = len(face_crops)

        if frames_used == 0:
            logger.warning(
                "No face detected in any of %d sampled frames of '%s'.",
                len(frames), path.name,
            )
            return _no_face_result()

        logger.info(
            "Running batched ViT inference on %d face crop(s) ...",
            frames_used,
        )

        # ── Single batched pipeline call ──────────────────────────────────────
        # Passing a list → List[List[Dict]], one per input image.
        batch_results: list[list[dict]] = self._pipe(
            face_crops, batch_size=frames_used
        )

        # Accumulate + average per-label scores across crops
        score_accumulator: Dict[str, float] = {
            label: 0.0 for label in FACIAL_EMOTION_LABELS
        }
        for per_image in batch_results:
            for item in per_image:
                label = item["label"]
                if label in score_accumulator:
                    score_accumulator[label] += float(item["score"])

        all_scores: Dict[str, float] = {
            label: round(total / frames_used, 4)
            for label, total in score_accumulator.items()
        }
        top_label = max(all_scores, key=lambda k: all_scores[k])

        result = FacialEmotionResult(
            emotion=top_label,
            confidence=all_scores[top_label],
            all_scores=all_scores,
            frames_used=frames_used,
        )

        logger.info(
            "Facial emotion: %s (%.1f%%) over %d/%d frames | runner-up: %s (%.1f%%)",
            result.emotion,
            result.confidence * 100,
            frames_used,
            len(frames),
            *_runner_up(all_scores, top_label),
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
        """Read :data:`_NUM_FRAMES` evenly-spaced BGR frames from the MP4."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("cv2.VideoCapture could not open: %s", video_path)
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logger.warning("Video has 0 frames: %s", video_path.name)
            cap.release()
            return []

        frames: list[np.ndarray] = []
        for idx in _sample_frame_indices(total, _NUM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        logger.debug(
            "Sampled %d/%d frames from '%s' (total=%d).",
            len(frames), _NUM_FRAMES, video_path.name, total,
        )
        return frames

    def _detect_face(self, frame_bgr: np.ndarray) -> Image.Image | None:
        """
        Detect the highest-confidence face in a BGR frame using MediaPipe
        BlazeFace and return a PIL RGB crop for the ViT pipeline.

        Returns ``None`` if no face is found.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Tasks API expects an mp.Image
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame_rgb
        )
        result = self._face_detector.detect(mp_img)

        if not result.detections:
            return None

        # Pick the highest-confidence detection
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box  # mediapipe.tasks.components.containers.BoundingBox

        # Bounding box is in pixels
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w, bbox.origin_x + bbox.width)
        y2 = min(h, bbox.origin_y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            return None

        face_rgb = frame_rgb[y1:y2, x1:x2]
        return Image.fromarray(face_rgb)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _runner_up(all_scores: Dict[str, float], top_label: str) -> tuple[str, float]:
    """Return the second-highest scoring label and its percentage."""
    others = {k: v for k, v in all_scores.items() if k != top_label}
    if not others:
        return "—", 0.0
    second_label = max(others, key=lambda k: others[k])
    return second_label, others[second_label] * 100


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import Config

    cfg = Config.load()
    recognizer = FacialEmotionRecognizer(cfg)

    video_paths: list[str] = sys.argv[1:] if len(sys.argv) > 1 else []

    print("=" * 55)
    print("  PHILIA -- Facial Emotion Test (ViT + MediaPipe)")
    print("=" * 55)

    if not video_paths:
        print("\nUsage: python -m emotion.facial_emotion path/to/clip.mp4 ...")
        sys.exit(0)

    for video_path in video_paths:
        result = recognizer.predict(video_path)
        print(f"\nFile:   {video_path}")
        print(f"Result: {result}")
        if result.is_no_face:
            print("        (no face detected — neutral fallback)")
        else:
            print("  Scores (averaged across frames):")
            for label, score in sorted(
                result.all_scores.items(), key=lambda kv: -kv[1]
            ):
                bar = "#" * int(score * 20)
                print(f"    {label:<10} {score:.3f}  {bar}")
