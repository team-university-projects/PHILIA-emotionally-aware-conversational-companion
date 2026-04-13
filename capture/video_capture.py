# Webcam recording — keeps camera warm between PTT turns and writes MP4 during recording.

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_RECORDINGS_DIR = Path(__file__).parent.parent / "tmp" / "recordings"
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

try:
    cv2.setLogLevel(2)
except AttributeError:
    pass


class VideoCapture:

    def __init__(self, config: Config) -> None:
        self._cfg = config.video
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_video_path: Optional[Path] = None
        self._actual_fps: float = float(config.video.capture_fps) or 15.0
        self._actual_size: tuple[int, int] = (config.video.frame_width, config.video.frame_height)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._record_event = threading.Event()
        self._lock = threading.Lock()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam (device index={self._cfg.device_index}). "
                "Ensure the camera is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.capture_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self._cfg.capture_fps
        self._actual_fps = actual_fps
        self._actual_size = (actual_w, actual_h)

        logger.info("Webcam opened — device=%d, resolution=%dx%d, fps=%.1f",
                    self._cfg.device_index, actual_w, actual_h, actual_fps)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._frame_loop, daemon=True, name="VideoCapture")
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._record_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("VideoCapture thread did not exit cleanly.")
        self._release_writer()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam released.")

    def start_recording(self, recording_id: str) -> None:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Webcam is not open. Call VideoCapture.open() first.")
        video_path = _RECORDINGS_DIR / f"recording_{recording_id}.mp4"
        with self._lock:
            self._writer = cv2.VideoWriter(
                str(video_path), _FOURCC,
                float(self._cfg.capture_fps), self._actual_size,
            )
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(f"cv2.VideoWriter failed to open: {video_path}")
            self._current_video_path = video_path
        self._record_event.set()
        logger.info("Video recording started -> %s", video_path.name)

    def stop_recording(self) -> Optional[Path]:
        self._record_event.clear()
        time.sleep(1.0 / max(self._actual_fps, 1))
        with self._lock:
            self._release_writer()
            saved_path = self._current_video_path
            self._current_video_path = None
        if saved_path and saved_path.exists():
            size_kb = saved_path.stat().st_size / 1024
            logger.info("Video saved -> %s  (%.1f KB)", saved_path.name, size_kb)
        else:
            logger.warning("Video file not found after stop_recording().")
            return None
        return saved_path

    def _frame_loop(self) -> None:
        interval = 1.0 / max(self._cfg.capture_fps, 1)
        _MAX_FAILURES = 30
        consecutive_failures = 0

        while not self._stop_event.is_set():
            t_start = time.monotonic()
            if self._cap is None or not self._cap.isOpened():
                logger.error("Webcam handle lost in frame loop.")
                break

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    logger.warning("Webcam frame read failed (consecutive: %d).", consecutive_failures)
                if consecutive_failures >= _MAX_FAILURES:
                    logger.warning("Webcam stream stalled — attempting recovery...")
                    if self._try_reopen():
                        consecutive_failures = 0
                    else:
                        logger.error("Webcam recovery failed — stopping frame loop.")
                        break
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            if self._record_event.is_set():
                with self._lock:
                    if self._writer is not None and self._writer.isOpened():
                        self._writer.write(frame)

            elapsed = time.monotonic() - t_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _try_reopen(self) -> bool:
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            time.sleep(0.5)
            self._cap = cv2.VideoCapture(self._cfg.device_index)
            if not self._cap.isOpened():
                logger.error("Webcam could not be reopened on device %d.", self._cfg.device_index)
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS,          self._cfg.capture_fps)
            actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self._cfg.capture_fps
            self._actual_fps  = actual_fps
            self._actual_size = (actual_w, actual_h)
            logger.info("Webcam recovery successful — device=%d %dx%d@%.0ffps.",
                        self._cfg.device_index, actual_w, actual_h, actual_fps)
            return True
        except Exception as exc:
            logger.error("Webcam recovery error: %s", exc)
            return False
