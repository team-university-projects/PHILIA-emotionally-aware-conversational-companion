# Push-to-talk audio recording — press/release events driven by the UI, saves to WAV.

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_RECORDINGS_DIR = Path(__file__).parent.parent / "tmp" / "recordings"


class AudioCapture:

    def __init__(self, config: Config, profile=None) -> None:
        self._sample_rate: int = config.audio.sample_rate
        self._channels: int = config.audio.channels
        self._blocksize: int = config.audio.chunk_size
        # Always use the system default device — Windows routes to the
        # correct active endpoint (A2DP for BT headsets, not HFP 8kHz).
        self._device: int | None = None
        logger.info("Microphone: system default")
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # These events are set externally by the UI (ChatWindow key bindings)
        self.ptt_press_event   = threading.Event()
        self.ptt_release_event = threading.Event()

    def record(self, on_start=None) -> tuple[Path, str]:
        """Record audio until ptt_release_event is set. Call this from a background thread."""
        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Ensure release event is clear before we start listening
        self.ptt_release_event.clear()

        if on_start is not None:
            try:
                on_start(recording_id)
            except Exception as exc:
                logger.warning("on_start callback raised: %s", exc)

        audio_chunks: List[np.ndarray] = []
        stop_event = threading.Event()

        def _callback(indata, frames, time, status) -> None:
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
                self.ptt_release_event.wait()
                stop_event.set()
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Could not open audio device (index={self._device}): {exc}") from exc

        logger.info("Recording stopped.")
        return self._save_wav(audio_chunks, recording_id)

    def _save_wav(self, chunks: List[np.ndarray], recording_id: str) -> tuple[Path, str]:
        if not chunks:
            logger.warning("No audio data captured; writing silent file.")
            audio = np.zeros((0, self._channels), dtype="float32")
        else:
            audio = np.concatenate(chunks, axis=0)

        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = _RECORDINGS_DIR / f"recording_{recording_id}.wav"
        sf.write(file=str(output_path), data=audio, samplerate=self._sample_rate, subtype="PCM_16")

        duration = len(audio) / self._sample_rate
        logger.info("WAV saved -> %s  (%.2fs, %d samples)", output_path, duration, len(audio))
        return output_path, recording_id


if __name__ == "__main__":
    import sys
    cfg = Config.load()
    capture = AudioCapture(cfg)
    print("Hold SPACE to record. Release to save. Ctrl+C to quit.")
    # Simulate UI events for standalone test
    import keyboard as _kb
    try:
        while True:
            _kb.wait("space")
            capture.ptt_press_event.set()
            _kb.wait("space", suppress=False)
            capture.ptt_release_event.set()
            saved_path, _ = capture.record()
            print(f"Saved: {saved_path}")
    except KeyboardInterrupt:
        sys.exit(0)
