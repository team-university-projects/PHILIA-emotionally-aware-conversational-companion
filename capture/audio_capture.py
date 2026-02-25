"""
audio_capture.py — Push-to-Talk microphone recording.

Responsibilities:
  - Wait for the spacebar to be pressed (PTT start)
  - Record microphone audio via a non-blocking sounddevice callback
    while the spacebar remains held down
  - Stop recording on spacebar release
  - Save the captured audio to a timestamped WAV file
  - Return the Path to the saved file

Usage (standalone test)::

    from config import Config
    from capture.audio_capture import AudioCapture

    capture = AudioCapture(Config.load())
    print("Hold SPACE to record...")
    path = capture.record()
    print(f"Saved to: {path}")
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import List

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Directory where WAV recordings are saved (relative to project root)
_RECORDINGS_DIR = Path(__file__).parent.parent / "tmp" / "recordings"


class AudioCapture:
    """
    Records microphone audio during a push-to-talk (PTT) window.

    The spacebar acts as the PTT trigger:
      - Press and hold  → recording starts
      - Release         → recording stops and WAV file is saved

    Audio collection is driven by a ``sounddevice`` callback running on
    PortAudio's internal thread, making the capture fully non-blocking from
    the Python main thread's perspective.

    Args:
        config: Global :class:`~config.Config` instance supplying audio
                device settings (sample rate, channels, block size).
    """

    PTT_KEY: str = "space"

    def __init__(self, config: Config) -> None:
        self._sample_rate: int = config.audio.sample_rate
        self._channels: int = config.audio.channels
        self._blocksize: int = config.audio.chunk_size
        self._device: int | None = config.audio.device_index
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self) -> tuple[Path, str]:
        """
        Block until the spacebar is pressed, record while it is held, then
        save the audio and return the file path and shared recording ID.

        The recording loop:

        1. Wait for ``SPACE`` press (blocking).
        2. Register a release hook *before* opening the stream so a fast
           release is never missed.
        3. Open a :class:`sounddevice.InputStream`; its callback appends
           audio chunks to an in-memory buffer.
        4. Wait for ``SPACE`` release (``threading.Event``).
        5. Stop the stream and flush the buffer.
        6. Write the buffer to a timestamped WAV file.

        Returns:
            Tuple of:
            - ``Path``: Absolute path to the saved ``.wav`` file.
            - ``str``: Recording ID (timestamp string) shared with the
              video recorder for matched file naming, e.g.
              ``"20260225_120000_000000"``.

        Raises:
            RuntimeError: If the audio device cannot be opened.
        """
        self._await_key_press()

        # Generate the shared recording ID at press-time — before the audio
        # is captured — so the video recorder can use the same ID immediately.
        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        audio_chunks: List[np.ndarray] = []
        stop_event = threading.Event()

        # Register the release hook BEFORE opening the stream so we cannot
        # miss a release that happens during stream initialisation.
        released = threading.Event()
        release_hook = keyboard.on_release_key(
            self.PTT_KEY, lambda _: released.set()
        )

        def _callback(
            indata: np.ndarray,
            frames: int,
            time,           # noqa: ANN001 — sounddevice CData type
            status: sd.CallbackFlags,
        ) -> None:
            """PortAudio callback — appends a copy of each incoming block."""
            if status:
                logger.warning("Audio callback status: %s", status)
            if not stop_event.is_set():
                audio_chunks.append(indata.copy())

        logger.info("Recording started — release SPACE to stop.")

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._blocksize,
                device=self._device,
                dtype="float32",
                callback=_callback,
            ):
                released.wait()        # Block until spacebar is released
                stop_event.set()       # Signal callback to stop appending
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Could not open audio device (index={self._device}): {exc}"
            ) from exc
        finally:
            keyboard.unhook(release_hook)   # Always clean up the hook

        logger.info("Recording stopped.")

        return self._save_wav(audio_chunks, recording_id)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _await_key_press(self) -> None:
        """
        Block until the PTT key is pressed.

        Uses ``on_press_key`` + ``threading.Event`` rather than
        ``keyboard.wait()`` for reliable single-event detection.
        """
        logger.info("Waiting for SPACE bar to start recording...")
        pressed = threading.Event()
        hook = keyboard.on_press_key(self.PTT_KEY, lambda _: pressed.set())
        pressed.wait()
        keyboard.unhook(hook)

    def _save_wav(self, chunks: List[np.ndarray], recording_id: str) -> tuple[Path, str]:
        """
        Concatenate audio chunks and write them to a WAV file named with
        the shared recording ID.

        Args:
            chunks:       List of float32 numpy arrays collected during recording.
            recording_id: Timestamp string shared with the video file.

        Returns:
            Tuple of (Path to WAV file, recording_id).
        """
        if not chunks:
            logger.warning("No audio data captured; writing silent file.")
            audio = np.zeros((0, self._channels), dtype="float32")
        else:
            audio = np.concatenate(chunks, axis=0)

        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = _RECORDINGS_DIR / f"recording_{recording_id}.wav"

        sf.write(
            file=str(output_path),
            data=audio,
            samplerate=self._sample_rate,
            subtype="PCM_16",
        )

        duration = len(audio) / self._sample_rate
        logger.info(
            "WAV saved -> %s  (%.2fs, %d samples)",
            output_path,
            duration,
            len(audio),
        )
        return output_path, recording_id


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = Config.load()
    capture = AudioCapture(cfg)

    print("=" * 50)
    print("  PHILIA — Audio Capture Test")
    print("  Hold SPACE to record. Release to save.")
    print("  Ctrl+C to quit.")
    print("=" * 50)

    try:
        while True:
            saved_path = capture.record()
            print(f"\n✅  Saved: {saved_path}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
