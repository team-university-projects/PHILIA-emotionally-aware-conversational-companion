"""
tts.py — Emotion-aware text-to-speech synthesis via edge-tts.

Responsibilities:
  - Accept response text and TTSParams from the tone mapper
  - Convert prosody params to edge-tts SSML rate/pitch/volume tags
  - Synthesise to MP3 bytes using edge-tts (Microsoft neural voices)
  - Play audio directly via sounddevice, return raw bytes for pipeline
  - Provide voice catalog listing and interactive preview for setup UI
  - Fall back to pyttsx3 if edge-tts is unavailable (offline mode)

Engine: edge-tts (Microsoft Edge TTS)
  - Free, no API key, high-quality neural voices
  - Requires internet (streams from Microsoft's TTS service)
  - SSML prosody: rate (%/s), pitch (Hz offset), volume (%)

Offline fallback: pyttsx3
  - Fully offline, lower quality, but functional without network
  - Auto-activated if edge-tts raises an SSL/network error
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import Config
from bot.bot_profile import BotProfile
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_VOICE      = "en-US-JennyNeural"
_NEUTRAL_WPM        = 175
_NEUTRAL_PITCH      = 1.0
_VOICE_CATALOG_PATH = Path(__file__).parent.parent / "assets" / "voices" / "voice_catalog.json"

# Dedicated asyncio loop for TTS — avoids conflicts with tkinter's event loop
_TTS_LOOP: asyncio.AbstractEventLoop | None = None
_TTS_LOOP_LOCK = threading.Lock()


def _get_tts_loop() -> asyncio.AbstractEventLoop:
    """Get (or lazily create) a persistent background asyncio loop for TTS."""
    global _TTS_LOOP
    with _TTS_LOOP_LOCK:
        if _TTS_LOOP is None or _TTS_LOOP.is_closed():
            _TTS_LOOP = asyncio.new_event_loop()
            t = threading.Thread(target=_TTS_LOOP.run_forever, daemon=True)
            t.start()
        return _TTS_LOOP


def _run_async(coro) -> object:
    """Submit a coroutine to the persistent TTS event loop and block for result."""
    loop = _get_tts_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


# ── TTSParams ──────────────────────────────────────────────────────────────────

@dataclass
class TTSParams:
    """
    Prosody parameters derived from the tone_mapper.

    Attributes:
        rate:   Speech rate in words per minute (neutral = 175).
        pitch:  Pitch multiplier relative to neutral (1.0 = no change).
        volume: Volume multiplier (1.0 = full).
        style:  Voice style label from tone_mapper (informational).
    """
    rate:   int    # words per minute
    pitch:  float  # multiplier around 1.0
    volume: float  # multiplier around 1.0
    style:  str    # e.g. "calm", "energetic", "gentle"


# ── Prosody conversion helpers ─────────────────────────────────────────────────

def _wpm_to_ssml_rate(wpm: int) -> str:
    pct = round(((wpm - _NEUTRAL_WPM) / _NEUTRAL_WPM) * 100)
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


def _pitch_multiplier_to_ssml(multiplier: float) -> str:
    hz = round((multiplier - 1.0) * 100)
    hz = max(-20, min(20, hz))
    return f"{hz:+d}Hz"


def _volume_to_ssml(volume: float) -> str:
    pct = round((volume - 1.0) * 100)
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


# ── Voice catalog ──────────────────────────────────────────────────────────────

def list_voices() -> list[dict]:
    """
    Return the curated voice catalog from assets/voices/voice_catalog.json.
    Falls back to a single default entry if the file is missing.
    """
    if _VOICE_CATALOG_PATH.exists():
        try:
            return json.loads(_VOICE_CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [{"id": _DEFAULT_VOICE, "label": "Jenny (US, Female)", "gender": "female"}]


# ── TTS engine ─────────────────────────────────────────────────────────────────

class TextToSpeech:
    """
    Synthesises speech with emotion-conditioned prosody using edge-tts,
    with automatic pyttsx3 fallback for offline environments.

    Args:
        config:  Global Config instance.
        profile: Bot personality profile (drives voice selection).
    """

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._cfg     = config.tts
        self._profile = profile
        self._voice   = getattr(profile, "tts_voice", _DEFAULT_VOICE) or _DEFAULT_VOICE
        self._offline = False   # switches to True if edge-tts network fails

        logger.info("TTS engine ready — voice=%s", self._voice)

    # ── Public API ──────────────────────────────────────────────────────────────

    def synthesise(self, text: str, params: TTSParams) -> bytes:
        """
        Synthesise text to speech and play through the default audio device.

        Falls back to pyttsx3 if edge-tts encounters a network error.

        Returns:
            Raw MP3 bytes (or empty bytes on pyttsx3 fallback).
        """
        logger.info(
            "Synthesising — rate=%dwpm pitch=x%.2f style=%s voice=%s",
            params.rate, params.pitch, params.style, self._voice,
        )
        if self._offline:
            self._speak_pyttsx3(text, params)
            return b""

        try:
            audio_bytes = _run_async(self._synthesise_async(text, params))
            self._play(audio_bytes)
            return audio_bytes
        except Exception as exc:
            logger.warning("edge-tts failed (%s) — switching to pyttsx3 fallback", exc)
            self._offline = True
            self._speak_pyttsx3(text, params)
            return b""

    def preview(self, text: str, voice_id: str) -> None:
        """
        Play a quick TTS sample in the given voice — used by the setup UI.
        Runs in the background so it doesn't block the UI thread.
        """
        def _play():
            try:
                params = TTSParams(rate=175, pitch=1.0, volume=1.0, style="neutral")
                audio_bytes = _run_async(self._synthesise_async(text, params, voice_override=voice_id))
                self._play(audio_bytes)
            except Exception as exc:
                logger.warning("Voice preview failed: %s", exc)

        threading.Thread(target=_play, daemon=True).start()

    def update_voice(self, voice_id: str) -> None:
        """Hot-swap the active voice (called after profile save)."""
        self._voice = voice_id
        self._offline = False   # reset offline flag on voice change
        logger.info("TTS voice updated to %s", voice_id)

    # ── Async edge-tts ─────────────────────────────────────────────────────────

    async def _synthesise_async(
        self,
        text: str,
        params: TTSParams,
        voice_override: str | None = None,
    ) -> bytes:
        """Stream edge-tts output into a memory buffer and return MP3 bytes."""
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts") from exc

        voice  = voice_override or self._voice
        rate   = _wpm_to_ssml_rate(params.rate)
        pitch  = _pitch_multiplier_to_ssml(params.pitch)
        volume = _volume_to_ssml(params.volume)

        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio_bytes = buf.getvalue()
        logger.debug("Synthesised %d bytes of MP3 audio", len(audio_bytes))
        return audio_bytes

    # ── Playback ───────────────────────────────────────────────────────────────

    def _play(self, mp3_bytes: bytes) -> None:
        """Decode MP3 bytes and play through the default output device."""
        try:
            import sounddevice as sd
            import soundfile as sf
            buf = io.BytesIO(mp3_bytes)
            data, samplerate = sf.read(buf)
            sd.play(data, samplerate=samplerate)
            sd.wait()
        except Exception as exc:
            logger.warning("sounddevice playback failed (%s) — saving to tmp/response.mp3", exc)
            _tmp = Path("tmp") / "response.mp3"
            _tmp.parent.mkdir(parents=True, exist_ok=True)
            _tmp.write_bytes(mp3_bytes)

    # ── pyttsx3 fallback ───────────────────────────────────────────────────────

    def _speak_pyttsx3(self, text: str, params: TTSParams) -> None:
        """Offline TTS via pyttsx3. Lower quality but no network needed."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", params.rate)
            engine.setProperty("volume", params.volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.error("pyttsx3 fallback also failed: %s", exc)
