"""
main.py — PHILIA Desktop Application Entry Point

Architecture:
  1. Launch CustomTkinter loading screen (main thread)
  2. Background thread: run health checks + load all models
  3. If first run: show setup wizard → collect BotProfile
  4. Open main chat window
  5. Each PTT event runs run_turn() in a new background thread,
     posting UI updates via the Interface API

Keyboard shortcut: SPACE = push-to-talk
"""

from __future__ import annotations

import os
import sys
from typing import Callable

# Suppress OpenCV MSMF/DSHOW verbose stderr before cv2 is imported anywhere
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# ── PyTorch CUDA DLL search path fix (Windows) ────────────────────────────────
import pathlib as _pathlib
for _sp in sys.path:
    _torch_lib = _pathlib.Path(_sp) / "torch" / "lib"
    if _torch_lib.is_dir():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_torch_lib))
        break
del _pathlib, _sp, _torch_lib

# ── Standard imports ──────────────────────────────────────────────────────────
from config import Config
from bot.bot_profile import BotProfile
from utils.logger import get_logger
import utils.health_check as hc

logger = get_logger(__name__)


# ── Component type alias ───────────────────────────────────────────────────────

def _initialise_components(
    config: Config,
    profile: BotProfile,
    update_status: Callable,
    set_check: Callable,
) -> tuple:
    """
    Load all heavy ML models with progress reporting.

    Returns the components tuple consumed by run_turn().
    """
    from capture.audio_capture import AudioCapture
    from capture.video_capture import VideoCapture
    from speech.transcriber import Transcriber
    from emotion.audio_emotion import AudioEmotionRecognizer
    from emotion.facial_emotion import FacialEmotionRecognizer
    from emotion.text_emotion import TextEmotionRecognizer
    from emotion.fusion import EmotionFuser
    from tone.tone_mapper import ToneMapper
    from llm.prompt_builder import PromptBuilder
    from llm.response_generator import ResponseGenerator
    from speech.tts import TextToSpeech

    # ── Health checks ──────────────────────────────────────────────────────────
    update_status("Running health checks…", 0.05)
    results = hc.run_all(config)

    for r in results:
        set_check(r.name, r.ok, r.critical)

    failed_critical = [r for r in results if r.critical and not r.ok]
    if failed_critical:
        msgs = "\n".join(f"• {r.name}: {r.message}" for r in failed_critical)
        raise RuntimeError(f"System check failed:\n{msgs}")

    # ── Capture devices ───────────────────────────────────────────────────────
    update_status("Initialising capture devices…", 0.15)
    audio_cap = AudioCapture(config)
    video_cap = VideoCapture(config)

    # ── Speech ────────────────────────────────────────────────────────────────
    update_status("Loading Whisper ASR…", 0.25)
    asr = Transcriber(config)

    # ── Emotion models ────────────────────────────────────────────────────────
    update_status("Loading audio emotion model…", 0.40)
    audio_emo = AudioEmotionRecognizer(config)

    update_status("Loading facial emotion model…", 0.55)
    facial_emo = FacialEmotionRecognizer(config)

    update_status("Loading text emotion model…", 0.70)
    text_emo = TextEmotionRecognizer(config)
    fuser = EmotionFuser(config)

    # ── Response generation ────────────────────────────────────────────────────
    update_status("Initialising LLM & TTS…", 0.85)
    tone_map   = ToneMapper(config)
    prompt_bld = PromptBuilder(config, profile)
    llm        = ResponseGenerator(config)
    tts        = TextToSpeech(config, profile)

    # ── Webcam warm-up ────────────────────────────────────────────────────────
    update_status("Opening webcam…", 0.95)
    try:
        video_cap.open()
    except Exception as exc:
        logger.warning("Webcam open failed (non-critical): %s", exc)

    update_status("Ready", 1.0)
    logger.info("All modules ready. %s is online.", profile.name)

    return (
        config, profile,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    )


def _no_face_result():
    """Return a neutral facial result when video capture is unavailable."""
    from emotion.facial_emotion import FacialEmotionResult, FACIAL_EMOTION_LABELS
    scores = {lbl: 0.0 for lbl in FACIAL_EMOTION_LABELS}
    scores["neutral"] = 1.0
    return FacialEmotionResult(
        emotion="neutral", confidence=1.0,
        all_scores=scores, is_no_face=True,
    )


# ── Per-turn pipeline ──────────────────────────────────────────────────────────

def run_turn(components: tuple, ui: "Interface", history: list[tuple[str, str]]) -> None:
    """
    Execute one full push-to-talk conversation turn.
    All UI updates are routed through the Interface thread-safe API.
    History is updated in-place on success.
    """
    from ui.interface import Interface  # avoid circular at module level

    (
        config, profile,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    ) = components

    try:
        # ── Step 1: Record ──────────────────────────────────────────────────────
        ui.set_stage("Listening")
        audio_path, recording_id = audio_cap.record(
            on_start=lambda rec_id: video_cap.start_recording(rec_id)
        )
        video_path = video_cap.stop_recording()

        # ── Step 2: Transcribe ──────────────────────────────────────────────────
        ui.set_stage("Transcribing")
        transcript = asr.transcribe(audio_path)
        logger.info("Transcript: %r", transcript)

        if not transcript or not transcript.strip():
            logger.warning("Empty transcript — skipping turn.")
            ui.set_stage("Ready")
            return

        ui.add_message("user", transcript)

        # ── Step 3: Emotion detection ───────────────────────────────────────────
        ui.set_stage("Analysing")
        text_result   = text_emo.predict(transcript)
        audio_result  = audio_emo.predict(audio_path)
        facial_result = facial_emo.predict(video_path) if video_path else _no_face_result()

        logger.info(
            "Emotions raw  text=%s(%.0f%%)  audio=%s(%.0f%%)  face=%s(%.0f%%)",
            text_result.emotion,   text_result.confidence   * 100,
            audio_result.emotion,  audio_result.confidence  * 100,
            facial_result.emotion, facial_result.confidence * 100,
        )

        # ── Step 4: Fuse emotions ───────────────────────────────────────────────
        fused = fuser.fuse(audio_result, facial_result, text_result)
        logger.info("Fused emotion: %s (%.0f%%)", fused.label, fused.confidence * 100)

        ui.update_emotion(
            label=fused.label,
            confidence=fused.confidence,
            audio_label=f"{audio_result.emotion} ({audio_result.confidence:.0%})",
            facial_label=f"{facial_result.emotion} ({facial_result.confidence:.0%})",
            text_label=f"{text_result.emotion} ({text_result.confidence:.0%})",
        )

        # ── Step 5: Tone + prompt ───────────────────────────────────────────────
        tone, tts_params = tone_map.map(fused)
        prompt = prompt_bld.build(transcript, fused, tone, history=history)

        # ── Step 6: LLM response ────────────────────────────────────────────────
        ui.set_stage("Thinking")
        try:
            response_text = llm.generate(prompt)
        except RuntimeError as exc:
            logger.error("LLM failed: %s", exc)
            response_text = "I'm here for you. Could you tell me more?"

        logger.info("[%s] %s", profile.name, response_text)
        ui.add_message("bot", response_text)

        # Update history
        history.append((transcript, response_text))
        # Keep only last 12 turns in memory (6 pairs)
        if len(history) > 12:
            history[:] = history[-12:]

        # ── Step 7: TTS ─────────────────────────────────────────────────────────
        ui.set_stage("Speaking")
        try:
            tts.synthesise(response_text, tts_params)
        except Exception as exc:
            logger.error("TTS failed: %s", exc)

    except Exception as exc:
        logger.error("Turn failed: %s", exc, exc_info=True)
        ui.add_message("bot", f"⚠ Something went wrong: {exc}")
    finally:
        ui.set_stage("Ready")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    from ui.interface import Interface

    config = Config.load()
    ui     = Interface(config)

    # These will be populated by the initialise flow
    _components: list = []
    _history: list[tuple[str, str]] = []

    def _initialise(update_status: Callable, set_check: Callable) -> BotProfile:
        """Called from loading screen background thread."""
        # Load or default profile first (needed by TTS + PromptBuilder)
        profile = BotProfile.load(config)
        comps = _initialise_components(config, profile, update_status, set_check)
        _components.clear()
        _components.extend(comps)
        return profile

    def _on_ptt() -> None:
        """Called from UI thread (runs pipeline in its own thread)."""
        if not _components:
            return
        run_turn(tuple(_components), ui, _history)

    ui.set_ptt_callback(_on_ptt)
    ui.run_loading(_initialise)


if __name__ == "__main__":
    main()
