"""
latency_benchmark.py — Measures per-component inference latency for PHILIA.

Runs each pipeline stage on a short real audio/video clip and reports
min / mean / max over N trials for inclusion in the research paper.

Usage:
    python -m eval.latency_benchmark
    python -m eval.latency_benchmark --trials 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Windows DLL fix (same as other eval scripts) ──────────────────────────────
import os as _os, pathlib as _pathlib
for _sp in sys.path:
    _torch_lib = _pathlib.Path(_sp) / "torch" / "lib"
    if _torch_lib.is_dir():
        if hasattr(_os, "add_dll_directory"):
            _os.add_dll_directory(str(_torch_lib))
        break
del _pathlib, _sp, _torch_lib

import sys, io
# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import Config
from eval.dataset_loader import RAVDESSLoader, FER2013Loader, MELDFusionLoader, CANONICAL_LABELS


def timed(fn, *args, **kwargs):
    """Run fn(*args) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def mean_ms(times: list[float]) -> str:
    """Format list of seconds as 'mean ± std ms'."""
    ms = [t * 1000 for t in times]
    if len(ms) == 1:
        return f"{ms[0]:.0f} ms"
    return f"{mean(ms):.0f} ± {stdev(ms):.0f} ms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHILIA per-component latency benchmark.")
    p.add_argument("--trials", type=int, default=5,
                   help="Number of timed trials per component (default: 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    N = args.trials
    config = Config.load()

    print(f"\n{'='*60}")
    print(f"  PHILIA Latency Benchmark  ({N} trials per component)")
    print(f"  Hardware: RTX 3060 — emotion_device={config.models.emotion_device}")
    print(f"{'='*60}\n")

    results: dict[str, list[float]] = {}

    # ── Load sample data ──────────────────────────────────────────────────────
    print("Loading sample data...")
    meld_samples = MELDFusionLoader().load(max_samples=N * 2)
    if not meld_samples:
        print("[ERROR] Could not load MELD samples. Check dataset paths.")
        sys.exit(1)

    fer_samples = FER2013Loader().load(max_samples=N * 2)
    if not fer_samples:
        print("[ERROR] Could not load FER2013 samples. Check dataset paths.")
        sys.exit(1)

    audio_paths = [s[0] for s in meld_samples[:N]]
    # Sanitise text: replace non-ASCII smart quotes / curly apostrophes that
    # crash Windows cp1252 logging. The models handle ASCII just as well.
    texts       = [s[1].encode('ascii', errors='replace').decode('ascii') for s in meld_samples[:N]]
    images      = [s[0] for s in fer_samples[:N]]  # FER2013Loader returns (image, label) 2-tuples

    # ── COMPONENT 1: Whisper ASR ──────────────────────────────────────────────
    print("\n[1/6] Benchmarking Whisper ASR (Faster-Whisper distil-large-v3)...")
    from speech.transcriber import Transcriber
    asr = Transcriber(config)
    times_asr = []
    for ap in audio_paths:
        _, t = timed(asr.transcribe, ap)
        times_asr.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results["Whisper ASR"] = times_asr

    # ── COMPONENT 2: Audio Emotion ────────────────────────────────────────────
    print("\n[2/6] Benchmarking Audio Emotion (Wav2Vec2)...")
    from emotion.audio_emotion import AudioEmotionRecognizer
    audio_emo = AudioEmotionRecognizer(config)
    times_audio = []
    for ap in audio_paths:
        _, t = timed(audio_emo.predict, ap)
        times_audio.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results["Audio Emotion (Wav2Vec2)"] = times_audio

    # ── COMPONENT 3: Facial Emotion ───────────────────────────────────────────
    print("\n[3/6] Benchmarking Facial Emotion (ViT)...")
    from emotion.facial_emotion import FacialEmotionRecognizer, FacialEmotionResult
    facial_emo = FacialEmotionRecognizer(config)
    # Direct image predict (same as in fusion benchmark)
    pipe = facial_emo._pipe
    times_facial = []
    for img in images[:N]:
        _, t = timed(pipe, img)
        times_facial.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results["Facial Emotion (ViT)"] = times_facial

    # ── COMPONENT 4: Text Emotion ─────────────────────────────────────────────
    print("\n[4/6] Benchmarking Text Emotion (RoBERTa)...")
    from emotion.text_emotion import TextEmotionRecognizer
    text_emo = TextEmotionRecognizer(config)
    times_text = []
    for txt in texts:
        _, t = timed(text_emo.predict, txt)
        times_text.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results["Text Emotion (RoBERTa)"] = times_text

    # ── COMPONENT 5: LLM (Gemini / Ollama) ───────────────────────────────────
    print(f"\n[5/6] Benchmarking LLM response generation ({config.models.llm_provider})...")
    from llm.prompt_builder import PromptBuilder
    from llm.response_generator import ResponseGenerator
    from bot.bot_profile import BotProfile
    from emotion.fusion import FusedEmotion

    profile = BotProfile.load(config)
    prompt_bld = PromptBuilder(config, profile)
    llm_gen = ResponseGenerator(config)

    # Build a dummy fusion result for prompting
    dummy_fused = FusedEmotion(
        label="neutral",
        confidence=0.75,
        distribution={l: (1.0 if l == "neutral" else 0.0) for l in CANONICAL_LABELS},
        breakdown={},
        weights_used={"audio": 0.2, "facial": 0.5, "text": 0.3},
    )
    from tone.tone_mapper import ToneMapper
    tone_map = ToneMapper(config)
    tone, _ = tone_map.map(dummy_fused)

    times_llm = []
    test_text = "I've been feeling a bit off today."
    for _ in range(N):
        prompt = prompt_bld.build(test_text, dummy_fused, tone)
        _, t = timed(llm_gen.generate, prompt)
        times_llm.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results[f"LLM ({config.models.llm_provider} / {config.models.llm_model})"] = times_llm

    # ── COMPONENT 6: TTS ──────────────────────────────────────────────────────
    print("\n[6/6] Benchmarking TTS synthesis (pyttsx3)...")
    from speech.tts import TextToSpeech, TTSParams
    tts = TextToSpeech(config, profile)
    tts_params = TTSParams(rate=175, pitch=1.0, volume=1.0, style="neutral")
    test_response = "I hear you. It sounds like today has been tough. I'm here for you."

    times_tts = []
    for _ in range(N):
        _, t = timed(tts.synthesise, test_response, tts_params)
        times_tts.append(t)
        print(f"      trial: {t*1000:.0f} ms")
    results["TTS (pyttsx3)"] = times_tts

    # ── FUSION (negligible, but measure anyway) ───────────────────────────────
    from emotion.fusion import EmotionFuser
    from emotion.audio_emotion import AudioEmotionResult
    from emotion.facial_emotion import FacialEmotionResult
    from emotion.text_emotion import TextEmotionResult

    fuser = EmotionFuser(config)
    dummy_audio  = AudioEmotionResult(emotion="neutral", confidence=0.6, all_scores={l: 1/7 for l in CANONICAL_LABELS})
    dummy_facial = FacialEmotionResult(emotion="neutral", confidence=0.8, all_scores={l: 1/7 for l in CANONICAL_LABELS}, frames_used=5)
    dummy_text   = TextEmotionResult(emotion="neutral",  confidence=0.7, all_scores={l: 1/7 for l in CANONICAL_LABELS})

    times_fuse = []
    for _ in range(N):
        _, t = timed(fuser.fuse, dummy_audio, dummy_facial, dummy_text)
        times_fuse.append(t)
    results["Fusion (weighted avg)"] = times_fuse

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LATENCY SUMMARY ({N} trials, RTX 3060 consumer hardware)")
    print(f"{'='*60}")
    print(f"  {'Component':<42} {'Latency':>15}")
    print(f"  {'-'*42} {'-'*15}")

    total_means = []
    for component, times in results.items():
        label = mean_ms(times)
        print(f"  {component:<42} {label:>15}")
        total_means.append(mean(times))

    total_ms = sum(total_means) * 1000
    print(f"  {'-'*42} {'-'*15}")
    print(f"  {'TOTAL (sequential)':<42} {total_ms:.0f} ms")
    print(f"{'='*60}")
    print()
    print("NOTE: Audio emotion, facial emotion, and text emotion run")
    print("      sequentially on CPU. Parallelisation would reduce")
    print(f"      the three emotion steps to ~{max(mean(results['Audio Emotion (Wav2Vec2)']), mean(results['Facial Emotion (ViT)']), mean(results['Text Emotion (RoBERTa)']))*1000:.0f} ms.")
    print()
    print("Paste the table above into your paper (Implementation Details section).")
    print()


if __name__ == "__main__":
    main()
