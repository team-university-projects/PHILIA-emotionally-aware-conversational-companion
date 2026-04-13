# Emotion-aware TTS via edge-tts (Microsoft neural voices) with pyttsx3 offline fallback.

from __future__ import annotations

import asyncio
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from config import Config
from bot.bot_profile import BotProfile
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_VOICE      = "en-US-JennyNeural"
_NEUTRAL_WPM        = 175
_NEUTRAL_PITCH      = 1.0
_VOICE_CATALOG_PATH = Path(__file__).parent.parent / "assets" / "voices" / "voice_catalog.json"

_TTS_LOOP: asyncio.AbstractEventLoop | None = None
_TTS_LOOP_LOCK = threading.Lock()


def _get_tts_loop() -> asyncio.AbstractEventLoop:
    global _TTS_LOOP
    with _TTS_LOOP_LOCK:
        if _TTS_LOOP is None or _TTS_LOOP.is_closed():
            _TTS_LOOP = asyncio.new_event_loop()
            threading.Thread(target=_TTS_LOOP.run_forever, daemon=True).start()
        return _TTS_LOOP


def _run_async(coro) -> object:
    return asyncio.run_coroutine_threadsafe(coro, _get_tts_loop()).result(timeout=30)


@dataclass
class TTSParams:
    rate:   int
    pitch:  float
    volume: float
    style:  str


def _wpm_to_ssml_rate(wpm: int) -> str:
    pct = round(((wpm - _NEUTRAL_WPM) / _NEUTRAL_WPM) * 100)
    return f"{max(-50, min(50, pct)):+d}%"


def _pitch_multiplier_to_ssml(multiplier: float) -> str:
    hz = round((multiplier - 1.0) * 100)
    return f"{max(-20, min(20, hz)):+d}Hz"


def _volume_to_ssml(volume: float) -> str:
    pct = round((volume - 1.0) * 100)
    return f"{max(-50, min(50, pct)):+d}%"


def list_voices() -> list[dict]:
    if _VOICE_CATALOG_PATH.exists():
        try:
            return json.loads(_VOICE_CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [{"id": _DEFAULT_VOICE, "label": "Jenny (US, Female)", "gender": "female"}]


class TextToSpeech:

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._cfg     = config.tts
        self._profile = profile
        self._voice   = getattr(profile, "tts_voice", _DEFAULT_VOICE) or _DEFAULT_VOICE
        self._offline = False
        logger.info("TTS engine ready — voice=%s", self._voice)

    def synthesise(self, text: str, params: TTSParams) -> bytes:
        logger.info("Synthesising — rate=%dwpm pitch=x%.2f style=%s voice=%s",
                    params.rate, params.pitch, params.style, self._voice)
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
        def _play():
            try:
                params = TTSParams(rate=175, pitch=1.0, volume=1.0, style="neutral")
                audio_bytes = _run_async(self._synthesise_async(text, params, voice_override=voice_id))
                self._play(audio_bytes)
            except Exception as exc:
                logger.warning("Voice preview failed: %s", exc)
        threading.Thread(target=_play, daemon=True).start()

    def update_voice(self, voice_id: str) -> None:
        self._voice = voice_id
        self._offline = False
        logger.info("TTS voice updated to %s", voice_id)

    async def _synthesise_async(self, text: str, params: TTSParams, voice_override: str | None = None) -> bytes:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts") from exc
        voice  = voice_override or self._voice
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(
            text, voice,
            rate=_wpm_to_ssml_rate(params.rate),
            pitch=_pitch_multiplier_to_ssml(params.pitch),
            volume=_volume_to_ssml(params.volume),
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio_bytes = buf.getvalue()
        logger.debug("Synthesised %d bytes of MP3 audio", len(audio_bytes))
        return audio_bytes

    def _play(self, mp3_bytes: bytes) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
            data, samplerate = sf.read(io.BytesIO(mp3_bytes))
            sd.play(data, samplerate=samplerate)
            sd.wait()
        except Exception as exc:
            logger.warning("sounddevice playback failed (%s) — saving to tmp/response.mp3", exc)
            _tmp = Path("tmp") / "response.mp3"
            _tmp.parent.mkdir(parents=True, exist_ok=True)
            _tmp.write_bytes(mp3_bytes)

    def _speak_pyttsx3(self, text: str, params: TTSParams) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", params.rate)
            engine.setProperty("volume", params.volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.error("pyttsx3 fallback also failed: %s", exc)
