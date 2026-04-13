# Facial emotion recognition via ViT on sampled webcam frames using MediaPipe BlazeFace.

from __future__ import annotations

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

_MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
_NUM_FRAMES = 5
_BLAZE_FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_BLAZE_FACE_LOCAL = Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite"
_MIN_DETECTION_CONFIDENCE = 0.5

FACIAL_EMOTION_LABELS: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
)


@dataclass
class FacialEmotionResult:
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    frames_used: int = 0
    is_no_face: bool = False

    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%}) [faces in {self.frames_used}/{_NUM_FRAMES} frames]"


def _no_face_result() -> FacialEmotionResult:
    scores = {label: 0.0 for label in FACIAL_EMOTION_LABELS}
    scores["neutral"] = 1.0
    return FacialEmotionResult(emotion="neutral", confidence=1.0, all_scores=scores, frames_used=0, is_no_face=True)


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return "cpu"
    import platform as _platform, ctypes as _ctypes
    _dll = "cublas64_12.dll" if _platform.system() == "Windows" else "libcublas.so.12"
    try:
        if _platform.system() == "Windows":
            _ctypes.windll.LoadLibrary(_dll)
        else:
            _ctypes.CDLL(_dll)
    except OSError:
        logger.warning("CUDA 12 runtime not found -- facial emotion will run on CPU.")
        return "cpu"
    import torch as _torch
    if not _torch.backends.cudnn.is_available():
        logger.warning("cuDNN not available in PyTorch -- facial emotion will run on CPU.")
        return "cpu"
    _ver = _torch.backends.cudnn.version()
    if _ver < 8900:
        logger.warning("cuDNN %d < 8900 -- facial emotion will run on CPU.", _ver)
        return "cpu"
    logger.info("cuDNN %d OK -- facial emotion will use GPU.", _ver)
    return "cuda"


def _sample_frame_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(step * i + step / 2) for i in range(n)]


def _ensure_blaze_face_model() -> Path:
    if _BLAZE_FACE_LOCAL.exists():
        return _BLAZE_FACE_LOCAL
    _BLAZE_FACE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading BlazeFace model to %s ...", _BLAZE_FACE_LOCAL)
    urllib.request.urlretrieve(_BLAZE_FACE_URL, _BLAZE_FACE_LOCAL)
    logger.info("BlazeFace model downloaded (%.0f KB).", _BLAZE_FACE_LOCAL.stat().st_size / 1024)
    return _BLAZE_FACE_LOCAL


class FacialEmotionRecognizer:

    def __init__(self, config: Config) -> None:
        model_name: str = getattr(config.models, "facial_emotion_model_name", _MODEL_NAME)
        device = _resolve_device(getattr(config.models, "emotion_device", "cpu"))
        pipeline_device = 0 if device == "cuda" else -1
        logger.info("Loading facial emotion model '%s' on %s ...", model_name, device)
        self._pipe = pipeline(
            task="image-classification",
            model=model_name,
            top_k=None,
            device=pipeline_device,
        )
        logger.info("Facial emotion ViT model ready.")
        tflite_path = _ensure_blaze_face_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=str(tflite_path))
        self._face_detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
            )
        )
        logger.info("MediaPipe BlazeFace detector ready.")

    def predict(self, video_path: str | Path) -> FacialEmotionResult:
        path = Path(video_path)
        if not path.exists():
            logger.error("Video file not found: %s — returning no-face.", path)
            return _no_face_result()

        frames = self._sample_frames(path)
        if not frames:
            logger.warning("Could not read any frames from %s.", path.name)
            return _no_face_result()

        face_crops: list[Image.Image] = []
        for frame_bgr in frames:
            crop = self._detect_face(frame_bgr)
            if crop is not None:
                face_crops.append(crop)

        frames_used = len(face_crops)
        if frames_used == 0:
            logger.warning("No face detected in any of %d sampled frames of '%s'.", len(frames), path.name)
            return _no_face_result()

        logger.info("Running batched ViT inference on %d face crop(s) ...", frames_used)
        batch_results: list[list[dict]] = self._pipe(face_crops, batch_size=frames_used)

        score_accumulator: Dict[str, float] = {label: 0.0 for label in FACIAL_EMOTION_LABELS}
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
            result.emotion, result.confidence * 100,
            frames_used, len(frames),
            *_runner_up(all_scores, top_label),
        )
        return result

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
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
        return frames

    def _detect_face(self, frame_bgr: np.ndarray) -> Image.Image | None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._face_detector.detect(mp_img)
        if not result.detections:
            return None
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w, bbox.origin_x + bbox.width)
        y2 = min(h, bbox.origin_y + bbox.height)
        if x2 <= x1 or y2 <= y1:
            return None
        return Image.fromarray(frame_rgb[y1:y2, x1:x2])


def _runner_up(all_scores: Dict[str, float], top_label: str) -> tuple[str, float]:
    others = {k: v for k, v in all_scores.items() if k != top_label}
    if not others:
        return "—", 0.0
    second_label = max(others, key=lambda k: others[k])
    return second_label, others[second_label] * 100
