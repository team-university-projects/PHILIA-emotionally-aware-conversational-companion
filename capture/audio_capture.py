# Push-to-talk audio recording — waits for SPACE, records while held, saves to WAV.

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

_RECORDINGS_DIR = Path(__file__).parent.parent / "tmp" / "recordings"


class AudioCapture:

    PTT_KEY: str = "space"

    def __init__(self, config: Config, profile=None) -> None:
        self._sample_rate: int = config.audio.sample_rate
        self._channels: int = config.audio.channels
        self._blocksize: int = config.audio.chunk_size
        mic_idx = getattr(profile, "mic_device_index", -1) if profile else -1
        self._device: int | None = mic_idx if mic_idx >= 0 else config.audio.device_index
        if self._device is not None:
            try:
                dev_info = sd.query_devices(self._device)
                logger.info("Microphone: [%d] %s", self._device, dev_info["name"])
            except Exception:
                logger.warning("Mic device index %d not found — using system default.", self._device)
                self._device = None
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def record(self, on_start=None) -> tuple[Path, str]:
        self._await_key_press()
        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if on_start is not None:
            try:
                on_start(recording_id)
            except Exception as exc:
                logger.warning("on_start callback raised: %s", exc)

        audio_chunks: List[np.ndarray] = []
        stop_event = threading.Event()
        released = threading.Event()
        release_hook = keyboard.on_release_key(self.PTT_KEY, lambda _: released.set())

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
                released.wait()
                stop_event.set()
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Could not open audio device (index={self._device}): {exc}") from exc
        finally:
            keyboard.unhook(release_hook)

        logger.info("Recording stopped.")
        return self._save_wav(audio_chunks, recording_id)

    def _await_key_press(self) -> None:
        logger.info("Waiting for SPACE bar to start recording...")
        pressed = threading.Event()
        hook = keyboard.on_press_key(self.PTT_KEY, lambda _: pressed.set())
        pressed.wait()
        keyboard.unhook(hook)

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
    try:
        while True:
            saved_path = capture.record()
            print(f"\n✅  Saved: {saved_path}\n")
    except KeyboardInterrupt:
        sys.exit(0)
