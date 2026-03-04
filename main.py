
from __future__ import annotations

import os
import sys
from typing import TypeAlias

# ── cuDNN PATH fix ─────────────────────────────────────────────────────────────
# Windows DLL resolution is first-come-first-served. If an older cuDNN version
# (e.g. shipped with the CUDA Toolkit) appears in PATH before the standalone
# cuDNN 9.x install, PyTorch picks it up and crashes with:
#   "Could not load symbol cudnnGetLibConfig. Error code 127"
# Prepend the cuDNN 9.x bin directory so Windows finds the correct DLL first.
# This must happen BEFORE any module that triggers a PyTorch/CUDA import.
_CUDNN_PATH = r"C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.3.1\x64"
if os.path.isdir(_CUDNN_PATH) and _CUDNN_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDNN_PATH + os.pathsep + os.environ.get("PATH", "")

from config import Config
from bot.bot_profile import BotProfile
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
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Component bundle type ──────────────────────────────────────────────────────

Components: TypeAlias = tuple[
    Config, BotProfile,
    AudioCapture, VideoCapture, Transcriber,
    AudioEmotionRecognizer, FacialEmotionRecognizer, TextEmotionRecognizer,
    EmotionFuser, ToneMapper, PromptBuilder, ResponseGenerator, TextToSpeech,
]


# ── Initialisation ─────────────────────────────────────────────────────────────

def initialise() -> Components:
    """
    Load config and instantiate all module singletons.

    Heavy models (Whisper, Wav2Vec2, ViT, BERT) are loaded here once so
    every subsequent turn runs at inference speed without reload overhead.
    """
    logger.info("Initialising PHILIA...")

    config  = Config.load()
    profile = BotProfile.load(config)

    # ── Capture ────────────────────────────────────────────────────────────────
    audio_cap = AudioCapture(config)
    video_cap = VideoCapture(config)

    # ── Speech ─────────────────────────────────────────────────────────────────
    asr = Transcriber(config)

    # ── Emotion recognition (load all models upfront) ──────────────────────────
    audio_emo  = AudioEmotionRecognizer(config)
    facial_emo = FacialEmotionRecognizer(config)
    text_emo   = TextEmotionRecognizer(config)
    fuser      = EmotionFuser(config)

    # ── Response generation ────────────────────────────────────────────────────
    tone_map   = ToneMapper(config)
    prompt_bld = PromptBuilder(config, profile)
    llm        = ResponseGenerator(config)
    tts        = TextToSpeech(config, profile)

    # Warm up the webcam once — stays open for the whole session
    logger.info("Opening webcam...")
    video_cap.open()

    logger.info(
        "All modules ready. %s is online. Press SPACE to speak, Ctrl+C to quit.",
        profile.name,
    )
    return (
        config, profile,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    )


# ── Single conversation turn ───────────────────────────────────────────────────

def run_turn(components: Components) -> None:
    """
    Execute one full push-to-talk conversation turn.

    Returns normally on success. Raises exceptions only for unrecoverable
    hardware failures (camera lost, audio device gone). All soft errors
    (empty transcript, no face detected, LLM timeout) are handled gracefully
    with logged warnings and sensible fallbacks.
    """
    (
        config, profile,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    ) = components

    # ── Step 1: Capture audio + video simultaneously ───────────────────────────
    # Pass an on_start callback into record() so the video writer starts at
    # the exact moment the spacebar is detected — using a single key hook
    # (inside AudioCapture) avoids the KeyError from two competing hooks.
    print("\n[PHILIA]  Hold SPACE to speak...")
    audio_path, recording_id = audio_cap.record(
        on_start=lambda rec_id: video_cap.start_recording(rec_id)
    )
    video_path = video_cap.stop_recording()         # Finalise the MP4

    logger.info(
        "Captured: audio=%s  video=%s",
        audio_path.name if audio_path else "None",
        video_path.name if video_path else "None",
    )

    # ── Step 2: Transcribe speech ──────────────────────────────────────────────
    logger.info("Transcribing speech...")
    transcript = asr.transcribe(audio_path)
    logger.info("Transcript: %r", transcript)

    if not transcript or not transcript.strip():
        logger.warning("Empty transcript — skipping turn.")
        return

    # ── Step 3: Detect emotions (text + audio + face) ─────────────────────────
    logger.info("Running emotion recognition...")

    text_result   = text_emo.predict(transcript)
    audio_result  = audio_emo.predict(audio_path)
    facial_result = facial_emo.predict(video_path) if video_path else _no_face_result()

    logger.info(
        "Emotions raw  text=%s(%.0f%%)  audio=%s(%.0f%%)  face=%s(%.0f%%)",
        text_result.emotion,   text_result.confidence   * 100,
        audio_result.emotion,  audio_result.confidence  * 100,
        facial_result.emotion, facial_result.confidence * 100,
    )

    # ── Step 4: Fuse emotions ──────────────────────────────────────────────────
    logger.info("Fusing emotion signals...")
    fused = fuser.fuse(audio_result, facial_result, text_result)
    logger.info(
        "Fused emotion: %s (%.0f%%) | weights used: %s",
        fused.label,
        fused.confidence * 100,
        {k: f"{v:.0%}" for k, v in fused.weights_used.items()},
    )

    # ── Step 5: Map emotion to tone + TTS params ───────────────────────────────
    tone, tts_params = tone_map.map(fused)
    logger.debug("Tone: %s  |  TTS: rate=%dwpm pitch=x%.2f", tone, tts_params.rate, tts_params.pitch)

    # ── Step 6: Build LLM prompt ───────────────────────────────────────────────
    prompt = prompt_bld.build(transcript, fused, tone)

    # ── Step 7: Generate response ──────────────────────────────────────────────
    logger.info("Generating response...")
    try:
        response_text = llm.generate(prompt)
    except RuntimeError as exc:
        logger.error("LLM failed: %s — using fallback response.", exc)
        response_text = "I'm here for you. Could you tell me more?"

    logger.info("[%s] %s", profile.name, response_text)
    print(f"\n[{profile.name}]  {response_text}\n")

    # ── Step 8: Synthesise and play speech ─────────────────────────────────────
    logger.info("Synthesising and playing response...")
    try:
        tts.synthesise(response_text, tts_params)
    except Exception as exc:
        logger.error("TTS failed: %s — response printed to terminal only.", exc)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _no_face_result():
    """Return a neutral facial result when video capture is unavailable."""
    from emotion.facial_emotion import FacialEmotionResult, FACIAL_EMOTION_LABELS
    scores = {lbl: 0.0 for lbl in FACIAL_EMOTION_LABELS}
    scores["neutral"] = 1.0
    return FacialEmotionResult(
        emotion="neutral", confidence=1.0,
        all_scores=scores, is_no_face=True,
    )


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point. Runs the push-to-talk loop until Ctrl+C."""
    components = initialise()
    _, _, _, video_cap, *_ = components

    print()
    print("=" * 55)
    print("  PHILIA — Emotionally Aware Conversational Companion")
    print("  Hold SPACE to speak  |  Ctrl+C to quit")
    print("=" * 55)

    turn_number = 0
    try:
        while True:
            turn_number += 1
            logger.debug("--- Turn %d ---", turn_number)
            try:
                run_turn(components)
            except KeyboardInterrupt:
                raise   # Propagate to outer handler
            except Exception as exc:
                # Per-turn error: log and offer to continue
                logger.error("Turn %d failed: %s", turn_number, exc, exc_info=True)
                print(f"\n[PHILIA]  An error occurred: {exc}")
                print("[PHILIA]  Press SPACE to try again, or Ctrl+C to quit.\n")

    except (KeyboardInterrupt, SystemExit):
        print("\n\n[PHILIA]  Goodbye.\n")
        logger.info("PHILIA shutting down after %d turn(s).", turn_number)
        video_cap.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
