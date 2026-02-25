"""
video_capture.py — PTT-triggered webcam video recording.

Responsibilities:
  - Keep the webcam warmed up and ready (avoids cold-start lag on PTT press)
  - Record raw video frames to an MP4 file only during the PTT window
  - Save the video alongside the corresponding audio (shared timestamp ID)
  - Return the Path to the saved video file for downstream use

No face detection or cropping here — that is deferred to the emotion module.

Typical usage in the pipeline::

    # Startup
    video_cap.open()

    # Per PTT turn (audio_cap drives the PTT; video wraps it)
    video_cap.start_recording(recording_id)
    audio_path = audio_cap.record()          # blocks during PTT
    video_path = video_cap.stop_recording()  # finishes and saves MP4

    # Shutdown
    video_cap.close()
"""

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

# Output directory for saved videos (mirrors audio recordings location)
_RECORDINGS_DIR = Path(__file__).parent.parent / "tmp" / "recordings"

# Four-character codec code for the VideoWriter.
# 'mp4v' produces .mp4 files on all platforms without extra codec installs.
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


class VideoCapture:
    """
    Webcam recorder that keeps the camera warm and records to MP4 on demand.

    The camera is opened once (``open()``) and stays open for the duration
    of the session, eliminating the cold-start delay that would otherwise
    cause missed frames at the beginning of each PTT recording.

    During a PTT window, ``start_recording()`` / ``stop_recording()`` control
    a :class:`cv2.VideoWriter` that writes raw frames to a timestamped MP4.
    The timestamp ID is shared with the audio recorder so both files can be
    linked by name (e.g. ``recording_20260225_120000_000000.wav/.mp4``).

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.video
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Camera handle — opened in open(), released in close()
        self._cap: Optional[cv2.VideoCapture] = None

        # Writer handle — opened in start_recording(), closed in stop_recording()
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_video_path: Optional[Path] = None

        # Background thread for continuously reading + writing frames
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._record_event = threading.Event()   # Set only during PTT window
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the webcam and start the background frame loop.

        Call once at application startup. The camera stays open and warm
        until :meth:`close` is called.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam (device index={self._cfg.device_index}). "
                "Ensure the camera is connected and not in use by another app."
            )

        # Apply requested resolution / FPS (best-effort — device may cap these)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.capture_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self._cfg.capture_fps
        self._actual_fps: float = actual_fps
        self._actual_size: tuple[int, int] = (actual_w, actual_h)

        logger.info(
            "Webcam opened — device=%d, resolution=%dx%d, fps=%.1f",
            self._cfg.device_index, actual_w, actual_h, actual_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._frame_loop,
            daemon=True,
            name="VideoCapture",
        )
        self._thread.start()

    def close(self) -> None:
        """
        Stop the background thread and release the webcam.

        Call once at application shutdown.
        """
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

    # ── PTT recording ──────────────────────────────────────────────────────────

    def start_recording(self, recording_id: str) -> None:
        """
        Begin writing webcam frames to a new MP4 file.

        Call immediately before ``audio_cap.record()`` so the video window
        aligns with the audio PTT window.

        Args:
            recording_id: Timestamp string shared with the audio file, e.g.
                          ``"20260225_120000_000000"``.  Creates
                          ``tmp/recordings/recording_{recording_id}.mp4``.

        Raises:
            RuntimeError: If the webcam is not open (``open()`` not called).
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                "Webcam is not open. Call VideoCapture.open() first."
            )

        video_path = _RECORDINGS_DIR / f"recording_{recording_id}.mp4"

        with self._lock:
            self._writer = cv2.VideoWriter(
                str(video_path),
                _FOURCC,
                float(self._cfg.capture_fps),   # Must match our write rate, not camera native fps
                self._actual_size,
            )
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(
                    f"cv2.VideoWriter failed to open: {video_path}\n"
                    "Ensure the mp4v codec is available."
                )
            self._current_video_path = video_path

        self._record_event.set()    # Signal frame_loop to start writing
        logger.info("Video recording started -> %s", video_path.name)

    def stop_recording(self) -> Optional[Path]:
        """
        Stop writing frames, finalise and close the MP4 file.

        Call immediately after ``audio_cap.record()`` returns.

        Returns:
            Path to the saved ``.mp4`` file, or ``None`` if
            ``start_recording()`` was never called.
        """
        self._record_event.clear()   # Signal frame_loop to stop writing

        # Give the frame loop one tick to finish the current frame
        time.sleep(1.0 / max(self._actual_fps, 1))

        with self._lock:
            self._release_writer()
            saved_path = self._current_video_path
            self._current_video_path = None

        if saved_path and saved_path.exists():
            size_kb = saved_path.stat().st_size / 1024
            logger.info(
                "Video saved -> %s  (%.1f KB)", saved_path.name, size_kb
            )
        else:
            logger.warning("Video file not found after stop_recording().")
            return None

        return saved_path

    # ── Background frame loop ──────────────────────────────────────────────────

    def _frame_loop(self) -> None:
        """
        Background thread: continuously read frames.

        Writes to the VideoWriter only while ``_record_event`` is set
        (i.e., during a PTT window). Otherwise frames are read and discarded
        to keep the camera pipeline warm.
        """
        interval = 1.0 / max(self._cfg.capture_fps, 1)

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                logger.error("Webcam handle lost in frame loop.")
                break

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read webcam frame — skipping.")
                time.sleep(0.05)
                continue

            # Write frame only during active PTT recording
            if self._record_event.is_set():
                with self._lock:
                    if self._writer is not None and self._writer.isOpened():
                        self._writer.write(frame)   # frame is BGR — correct for VideoWriter

            # Throttle to target FPS
            elapsed = time.monotonic() - t_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _release_writer(self) -> None:
        """Release and nullify the VideoWriter (call with self._lock held or at shutdown)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import keyboard
    from config import Config

    cfg = Config.load()
    cap = VideoCapture(cfg)

    print("=" * 50)
    print("  PHILIA -- Video Capture Test")
    print("  Hold SPACE to record. Release to save.")
    print("  Ctrl+C to quit.")
    print("=" * 50)

    print("\nOpening webcam (warming up)...")
    cap.open()
    print("Camera ready.\n")

    try:
        while True:
            print("Waiting for SPACE bar...")

            pressed = threading.Event()
            h = keyboard.on_press_key("space", lambda _: pressed.set())
            pressed.wait()
            keyboard.unhook(h)

            rec_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cap.start_recording(rec_id)
            print(f"  Recording... (recording_id={rec_id})")

            released = threading.Event()
            h2 = keyboard.on_release_key("space", lambda _: released.set())
            released.wait()
            keyboard.unhook(h2)

            video_path = cap.stop_recording()
            if video_path:
                print(f"\nSaved -> {video_path}")
                print(f"Pair audio:  recording_{rec_id}.wav\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
        cap.close()
        sys.exit(0)


