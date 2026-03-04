"""
tts.py — Emotion-aware text-to-speech synthesis via edge-tts.

Responsibilities:
  - Accept response text and TTSParams from the tone mapper
  - Convert prosody params to edge-tts SSML rate/pitch/volume tags
  - Synthesise to MP3 bytes using edge-tts (Microsoft neural voices)
  - Play audio directly via sounddevice, return raw bytes for pipeline

Engine: edge-tts (Microsoft Edge TTS)
  - Free, no API key, high-quality neural voices
  - Requires internet (streams from Microsoft's TTS service)
  - No training, no model downloads
  - SSML prosody: rate (%/s), pitch (Hz offset), volume (%)

Emotion-to-prosody mapping
--------------------------
The tone_mapper already converts emotion → TTSParams. This module handles
the unit conversion from our internal format to SSML strings:

  | Emotion  | Tone        | Rate wpm | Pitch | SSML rate | SSML pitch |
  |----------|-------------|----------|-------|-----------|------------|
  | happy    | energetic   | 190      | 1.10  | +15%      | +8Hz       |
  | sad      | gentle      | 145      | 0.90  | -15%      | -5Hz       |
  | angry    | firm        | 160      | 0.95  | -5%       | -2Hz       |
  | fear     | reassuring  | 150      | 1.05  | -10%      | +3Hz       |
  | surprise | engaged     | 180      | 1.08  | +10%      | +6Hz       |
  | disgust  | measured    | 155      | 1.00  | -8%       | 0Hz        |
  | neutral  | balanced    | 175      | 1.00  | 0%        | 0Hz        |

Backend extensibility
---------------------
  To swap to a different TTS engine, subclass TextToSpeech and override
  ``synthesise()``. TTSParams and the calling interface stay unchanged.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path

from config import Config
from bot.bot_profile import BotProfile
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Default edge-tts voice. Full list: `edge-tts --list-voices`
# Jenny is a high-quality US female neural voice.
_DEFAULT_VOICE = "en-US-JennyNeural"

# Neutral baseline: tone_mapper uses 175 wpm, multiplier 1.0
_NEUTRAL_WPM   = 175
_NEUTRAL_PITCH = 1.0


# ── TTSParams ──────────────────────────────────────────────────────────────────

@dataclass
class TTSParams:
    """
    Prosody parameters derived from the tone_mapper.

    Attributes:
        rate:   Speech rate in words per minute (neutral = 175).
        pitch:  Pitch multiplier relative to neutral (1.0 = no change).
        volume: Volume multiplier (1.0 = full).
        style:  Voice style label from tone_mapper (informational, not used
                by edge-tts which derives prosody from rate/pitch/volume).
    """
    rate:   int    # words per minute
    pitch:  float  # multiplier around 1.0
    volume: float  # multiplier around 1.0
    style:  str    # e.g. "calm", "energetic", "gentle"


# ── Prosody conversion helpers ─────────────────────────────────────────────────

def _wpm_to_ssml_rate(wpm: int) -> str:
    """
    Convert words-per-minute to an SSML ``prosody rate`` percentage string.

    edge-tts prosody rate is relative to the voice default (~175 wpm).
    Formula: ((wpm - neutral) / neutral) * 100  → clipped to [-50%, +50%]

    Examples:
        190 wpm → +9% → "+9%"
        145 wpm → -17% → "-17%"
        175 wpm → 0%  → "+0%"
    """
    pct = round(((wpm - _NEUTRAL_WPM) / _NEUTRAL_WPM) * 100)
    pct = max(-50, min(50, pct))   # clamp to safe range
    return f"{pct:+d}%"


def _pitch_multiplier_to_ssml(multiplier: float) -> str:
    """
    Convert a pitch multiplier to an SSML ``prosody pitch`` Hz-offset string.

    Maps multiplier linearly to ±15 Hz range:
        1.0 → +0Hz    (neutral)
        1.1 → +10Hz   (cheerful)
        0.9 → -10Hz   (gentle/sad)

    edge-tts Hz offsets are clamped to [-20Hz, +20Hz] by the service.
    """
    hz = round((multiplier - 1.0) * 100)   # 0.1 → 10Hz
    hz = max(-20, min(20, hz))
    return f"{hz:+d}Hz"


def _volume_to_ssml(volume: float) -> str:
    """
    Convert a volume multiplier (0.0–1.0) to SSML ``prosody volume`` percent.

    1.0 → "+0%"  (full, no change)
    0.8 → "-20%"
    """
    pct = round((volume - 1.0) * 100)
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


def _wrap_ssml(text: str, params: TTSParams, voice: str) -> str:
    """Wrap text in SSML speak/voice/prosody tags for edge-tts."""
    rate   = _wpm_to_ssml_rate(params.rate)
    pitch  = _pitch_multiplier_to_ssml(params.pitch)
    volume = _volume_to_ssml(params.volume)

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
        f"{text}"
        "</prosody>"
        "</voice>"
        "</speak>"
    )


# ── TTS engine ─────────────────────────────────────────────────────────────────

class TextToSpeech:
    """
    Synthesises speech with emotion-conditioned prosody using edge-tts.

    ``synthesise()`` is synchronous — it wraps the async edge-tts API
    internally so the rest of the pipeline stays blocking.

    Args:
        config:  Global :class:`~config.Config` instance.
        profile: Bot personality profile (used for voice selection).
    """

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._cfg     = config.tts
        self._profile = profile
        self._voice   = getattr(profile, "tts_voice", _DEFAULT_VOICE)

        logger.info(
            "TTS engine ready — voice=%s provider=edge-tts", self._voice
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def synthesise(self, text: str, params: TTSParams) -> bytes:
        """
        Synthesise text to speech and play it through the default audio device.

        Uses edge-tts SSML ``<prosody>`` to apply the emotion-derived rate,
        pitch, and volume from ``params``. Audio is synthesised to an in-memory
        MP3 buffer, played via sounddevice, and returned as raw bytes.

        Args:
            text:   LLM response text to speak.
            params: Prosody parameters from :class:`~tone.tone_mapper.ToneMapper`.

        Returns:
            Raw MP3 bytes of the synthesised audio.
        """
        logger.info(
            "Synthesising speech — rate=%dwpm pitch=x%.2f volume=x%.2f style=%s",
            params.rate, params.pitch, params.volume, params.style,
        )
        audio_bytes = asyncio.run(self._synthesise_async(text, params))
        self._play(audio_bytes)
        return audio_bytes

    # ── Async edge-tts ─────────────────────────────────────────────────────────

    async def _synthesise_async(self, text: str, params: TTSParams) -> bytes:
        """Stream edge-tts output into a memory buffer and return MP3 bytes."""
        try:
            import edge_tts   # lazy import — only needed for this backend
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts not installed. Run: pip install edge-tts"
            ) from exc

        rate   = _wpm_to_ssml_rate(params.rate)
        pitch  = _pitch_multiplier_to_ssml(params.pitch)
        volume = _volume_to_ssml(params.volume)

        buf = io.BytesIO()
        communicate = edge_tts.Communicate(
            text,
            self._voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio_bytes = buf.getvalue()
        logger.debug("Synthesised %d bytes of MP3 audio.", len(audio_bytes))
        return audio_bytes

    # ── Playback ───────────────────────────────────────────────────────────────

    def _play(self, mp3_bytes: bytes) -> None:
        """
        Decode MP3 bytes and play through the default output device.

        Uses ``soundfile`` + ``sounddevice`` (already in requirements).
        Falls back to writing a temp file if runtime decoding is unavailable.
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            buf = io.BytesIO(mp3_bytes)
            data, samplerate = sf.read(buf)
            sd.play(data, samplerate=samplerate)
            sd.wait()
        except Exception as exc:
            logger.warning(
                "sounddevice playback failed (%s) — saving to tmp/response.mp3", exc
            )
            _tmp = Path("tmp") / "response.mp3"
            _tmp.parent.mkdir(parents=True, exist_ok=True)
            _tmp.write_bytes(mp3_bytes)
            logger.info("Audio saved to %s", _tmp)


# ── Standalone test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import Config
    from bot.bot_profile import BotProfile

    cfg     = Config.load()
    profile = BotProfile()
    tts     = TextToSpeech(cfg, profile)

    # Emotion-to-prosody mapping demonstration table
    from tone.tone_mapper import _TONE_MAP, ToneMapper
    from emotion.fusion import FusedEmotion

    fuser_cfg = cfg
    tone_mapper = ToneMapper(fuser_cfg)

    print("=" * 55)
    print("  PHILIA -- TTS Engine Test (edge-tts)")
    print("=" * 55)
    print(f"  Voice: {tts._voice}")
    print()
    print(f"  {'Emotion':<10} {'Tone':<12} {'Rate wpm':>8} {'SSML rate':>10} {'SSML pitch':>11}")
    print("  " + "-" * 55)
    for emotion, (tone_desc, pitch, rate, style) in _TONE_MAP.items():
        ssml_rate  = _wpm_to_ssml_rate(rate)
        ssml_pitch = _pitch_multiplier_to_ssml(pitch)
        print(f"  {emotion:<10} {tone_desc:<12} {rate:>8} {ssml_rate:>10} {ssml_pitch:>11}")

    test_text = sys.argv[1] if len(sys.argv) > 1 else (
        "I'm sorry to hear you had such a difficult day. "
        "It sounds incredibly frustrating. I'm here for you."
    )

    # Default: sad prosody
    emotion_label = sys.argv[2] if len(sys.argv) > 2 else "sad"
    dummy_fused = FusedEmotion(
        label=emotion_label,
        confidence=0.8,
        distribution={},
        breakdown={},
    )
    tone_desc, params = tone_mapper.map(dummy_fused)

    print(f"\nSpeaking as '{emotion_label}' (tone: {tone_desc})...")
    print(f"Text: {test_text}")
    tts.synthesise(test_text, params)
    print("Done.")
