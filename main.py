# PHILIA desktop application entry point — manages GUI lifecycle and per-turn pipeline.

from __future__ import annotations

import os
import sys
from typing import Callable

os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import pathlib as _pathlib
for _sp in sys.path:
    _torch_lib = _pathlib.Path(_sp) / "torch" / "lib"
    if _torch_lib.is_dir():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_torch_lib))
        break
del _pathlib, _sp, _torch_lib

from config import Config
from bot.bot_profile import BotProfile
from utils.logger import get_logger
import utils.health_check as hc

logger = get_logger(__name__)


def _initialise_components(
    config: Config,
    profile: BotProfile,
    update_status: Callable,
    set_check: Callable,
) -> tuple:
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

    update_status("Running health checks…", 0.05)
    results = hc.run_all(config)
    for r in results:
        set_check(r.name, r.ok, r.critical)
    failed_critical = [r for r in results if r.critical and not r.ok]
    if failed_critical:
        msgs = "\n".join(f"• {r.name}: {r.message}" for r in failed_critical)
        raise RuntimeError(f"System check failed:\n{msgs}")

    update_status("Initialising capture devices…", 0.15)
    audio_cap = AudioCapture(config, profile)
    video_cap = VideoCapture(config)

    update_status("Loading Whisper ASR…", 0.25)
    asr = Transcriber(config)

    update_status("Loading audio emotion model…", 0.40)
    audio_emo = AudioEmotionRecognizer(config)

    update_status("Loading facial emotion model…", 0.55)
    facial_emo = FacialEmotionRecognizer(config)

    update_status("Loading text emotion model…", 0.70)
    text_emo = TextEmotionRecognizer(config)
    fuser = EmotionFuser(config)

    update_status("Initialising LLM & TTS…", 0.85)
    tone_map   = ToneMapper(config)
    prompt_bld = PromptBuilder(config, profile)
    llm        = ResponseGenerator(config)
    tts        = TextToSpeech(config, profile)

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
    from emotion.facial_emotion import FacialEmotionResult, FACIAL_EMOTION_LABELS
    scores = {lbl: 0.0 for lbl in FACIAL_EMOTION_LABELS}
    scores["neutral"] = 1.0
    return FacialEmotionResult(
        emotion="neutral", confidence=1.0,
        all_scores=scores, is_no_face=True,
    )


def run_turn(components: tuple, ui: "Interface", history: list[tuple[str, str]]) -> None:
    from ui.interface import Interface

    (
        config, profile,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    ) = components

    try:
        ui.set_stage("Listening")
        audio_path, recording_id = audio_cap.record(
            on_start=lambda rec_id: video_cap.start_recording(rec_id)
        )
        video_path = video_cap.stop_recording()

        ui.set_stage("Transcribing")
        transcript = asr.transcribe(audio_path)
        logger.info("Transcript: %r", transcript)

        if not transcript or not transcript.strip():
            logger.warning("Empty transcript — skipping turn.")
            ui.set_stage("Ready")
            return

        ui.add_message("user", transcript)

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

        fused = fuser.fuse(audio_result, facial_result, text_result)
        logger.info("Fused emotion: %s (%.0f%%)", fused.label, fused.confidence * 100)

        ui.update_emotion(
            label=fused.label,
            confidence=fused.confidence,
            audio_label=f"{audio_result.emotion} ({audio_result.confidence:.0%})",
            facial_label=f"{facial_result.emotion} ({facial_result.confidence:.0%})",
            text_label=f"{text_result.emotion} ({text_result.confidence:.0%})",
        )

        tone, tts_params = tone_map.map(fused)
        prompt = prompt_bld.build(transcript, fused, tone, history=history)

        ui.set_stage("Thinking")
        try:
            response_text = llm.generate(prompt)
        except RuntimeError as exc:
            logger.error("LLM failed: %s", exc)
            response_text = "I'm here for you. Could you tell me more?"

        logger.info("[%s] %s", profile.name, response_text)
        ui.add_message("bot", response_text)

        history.append((transcript, response_text))
        if len(history) > 12:
            history[:] = history[-12:]

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


def main() -> None:
    from ui.interface import Interface

    config = Config.load()
    ui     = Interface(config)

    _components: list = []
    _history: list[tuple[str, str]] = []

    def _initialise(profile: BotProfile, update_status: Callable, set_check: Callable) -> BotProfile:
        comps = _initialise_components(config, profile, update_status, set_check)
        _components.clear()
        _components.extend(comps)
        # Wire AudioCapture's PTT events into the UI
        audio_cap = comps[2]  # index 2 = audio_cap in the tuple
        ui.set_ptt_events(audio_cap.ptt_press_event, audio_cap.ptt_release_event)
        return profile

    def _on_profile_final(final_profile: BotProfile) -> None:
        """Called after the new-profile wizard completes — hot-swap voice + persona."""
        if _components:
            tts        = _components[12]   # TextToSpeech
            prompt_bld = _components[10]   # PromptBuilder
            tts.update_voice(final_profile.tts_voice)
            prompt_bld.update_profile(final_profile)

    def _on_ptt() -> None:
        if not _components:
            return
        run_turn(tuple(_components), ui, _history)

    ui.set_ptt_callback(_on_ptt)
    ui.run_loading(_initialise, on_profile_final=_on_profile_final)


if __name__ == "__main__":
    main()
