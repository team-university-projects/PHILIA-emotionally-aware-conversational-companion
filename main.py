"""
main.py — PHILIA Orchestration Pipeline

Entry point for the PHILIA multimodal emotion-aware conversational agent.

Pipeline per push-to-talk turn:
  1.  Capture audio + webcam frame
  2.  Transcribe audio → text
  3.  Run three parallel emotion recognisers (audio / facial / text)
  4.  Fuse emotion signals (confidence-weighted late fusion)
  5.  Map fused emotion → tone descriptor + TTS prosody params
  6.  Build emotion-conditioned LLM prompt
  7.  Generate LLM response
  8.  Synthesise speech with prosody
  9.  Play response via avatar interface
"""

from __future__ import annotations

from config import Config
from bot.bot_profile import BotProfile
from ui.interface import Interface
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


# ── Type alias for the components bundle ─────────────────────────────────────
type Components = tuple[
    Config, BotProfile, Interface,
    AudioCapture, VideoCapture, Transcriber,
    AudioEmotionRecognizer, FacialEmotionRecognizer, TextEmotionRecognizer,
    EmotionFuser, ToneMapper, PromptBuilder, ResponseGenerator, TextToSpeech,
]


# ── Initialisation ────────────────────────────────────────────────────────────
def initialise() -> Components:
    """
    Load config and instantiate all module singletons.

    Order matters: BotProfile before Interface (profile drives avatar path),
    Config before all models (provides paths + weights).
    """
    config = Config.load()
    profile = BotProfile.load(config)
    interface = Interface(profile)

    audio_cap = AudioCapture(config)
    video_cap = VideoCapture(config)
    asr = Transcriber(config)

    audio_emo = AudioEmotionRecognizer(config)
    facial_emo = FacialEmotionRecognizer(config)
    text_emo = TextEmotionRecognizer(config)
    fuser = EmotionFuser(config)

    tone_map = ToneMapper(config)
    prompt_bld = PromptBuilder(config, profile)
    llm = ResponseGenerator(config)
    tts = TextToSpeech(config, profile)

    logger.info("All PHILIA modules initialised — opening webcam...")
    video_cap.open()   # Warm up camera once; stays open for the session
    logger.info("PHILIA ready.")
    return (
        config, profile, interface,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    )


# ── Single Conversation Turn ───────────────────────────────────────────────────
def run_turn(components: Components) -> None:
    """
    Execute one full push-to-talk conversation turn.

    All heavy inference is sequential within this function but completely
    decoupled from other turns. Parallelism within the emotion recognition
    step (audio / facial / text) can be added here with threading/asyncio
    without changing any module contracts.
    """
    (
        config, profile, interface,
        audio_cap, video_cap, asr,
        audio_emo, facial_emo, text_emo,
        fuser, tone_map, prompt_bld, llm, tts,
    ) = components

    # ── Step 1: Capture (audio + video, duration-matched) ────────────────────
    logger.info("Starting capture — hold SPACE to record...")
    # audio_cap.record() generates recording_id at PTT press-time and video
    # uses the same ID so .wav and .mp4 files share the same filename stem.
    audio_path, recording_id = audio_cap.record()  # Blocks for PTT duration
    video_path = video_cap.stop_recording()        # Finalise and close the MP4

    # ── Step 2: Transcribe ───────────────────────────────────────────────────
    logger.info("Transcribing speech...")
    transcript = asr.transcribe(audio_bytes)
    logger.debug("Transcript: %s", transcript)

    # ── Step 3: Emotion Recognition (3 modalities) ───────────────────────────
    logger.info("Running emotion recognition...")
    audio_probs = audio_emo.predict(audio_bytes)
    facial_probs = facial_emo.predict(frame)
    text_probs = text_emo.predict(transcript)

    # ── Step 4: Fuse Emotions ────────────────────────────────────────────────
    logger.info("Fusing emotion signals...")
    fused = fuser.fuse(audio_probs, facial_probs, text_probs)
    logger.info("Fused emotion: %s (%.0f%% confidence)", fused.label, fused.confidence * 100)

    # ── Step 5: Tone Mapping ─────────────────────────────────────────────────
    tone, tts_params = tone_map.map(fused)
    logger.debug("Tone: %s | TTS params: %s", tone, tts_params)

    # ── Step 6: Build LLM Prompt ─────────────────────────────────────────────
    prompt = prompt_bld.build(transcript, fused, tone)

    # ── Step 7: Generate LLM Response ────────────────────────────────────────
    logger.info("Generating LLM response...")
    response_text = llm.generate(prompt)
    logger.debug("Response: %s", response_text)

    # ── Step 8: Text-to-Speech ───────────────────────────────────────────────
    logger.info("Synthesising speech...")
    audio_response = tts.synthesise(response_text, tts_params)

    # ── Step 9: Avatar + Playback ────────────────────────────────────────────
    interface.play_response(audio_response, response_text)


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point. Runs the push-to-talk conversation loop."""
    logger.info("Starting PHILIA...")
    components = initialise()
    _, _, interface, *_ = components

    logger.info("PHILIA is ready. Press PTT to speak. Ctrl+C to quit.")

    try:
        while True:
            # Pre-arm video: register a one-shot hook so video starts the
            # instant the spacebar is pressed (same moment audio_cap.record()
            # begins capturing), then call record() which will block.
            def _on_ptt_press(_event) -> None:
                keyboard.unhook(_on_ptt_press)
                video_cap.start_recording(
                    datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
            keyboard.on_press_key("space", _on_ptt_press)

            interface.wait_for_ptt()    # Blocking; raises SystemExit on quit
            run_turn(components)
    except (KeyboardInterrupt, SystemExit):
        logger.info("PHILIA shutting down. Goodbye.")
        video_cap.close()


if __name__ == "__main__":
    main()
