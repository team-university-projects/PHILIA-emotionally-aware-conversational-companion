# Pre-launch system readiness checks for Ollama, emotion models, webcam, and microphone.

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from config import Config


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    critical: bool = True


def check_ollama(config: Config) -> CheckResult:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        model_name = config.models.llm_model
        available = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        if model_name not in available and model_name.split(":")[0] not in available:
            return CheckResult(
                name="Ollama Model", ok=False, critical=True,
                message=f"Model '{model_name}' not found in Ollama.\nRun: ollama pull {model_name}",
            )
        return CheckResult(name="Ollama", ok=True, message=f"Ollama running • model '{model_name}' available")
    except (urllib.error.URLError, OSError):
        return CheckResult(
            name="Ollama", ok=False, critical=True,
            message="Cannot reach Ollama at localhost:11434.\nStart it with: ollama serve",
        )


def check_models(config: Config) -> CheckResult:
    root = Path(__file__).parent.parent
    required = [
        ("Audio Emotion",  root / config.models.audio_emotion_model_name),
        ("Facial Emotion", root / config.models.facial_emotion_model_name),
        ("Text Emotion",   root / config.models.text_emotion_model_name),
    ]
    missing = [name for name, path in required if not path.exists()]
    if missing:
        return CheckResult(
            name="Emotion Models", ok=False, critical=True,
            message=f"Missing fine-tuned model(s): {', '.join(missing)}",
        )
    return CheckResult(name="Emotion Models", ok=True, message="All fine-tuned emotion models found")


def check_webcam() -> CheckResult:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return CheckResult(name="Webcam", ok=True, message="Webcam accessible (index 0)")
        return CheckResult(
            name="Webcam", ok=False, critical=False,
            message="No webcam found at index 0. Facial emotion will use neutral fallback.",
        )
    except Exception as exc:
        return CheckResult(name="Webcam", ok=False, critical=False, message=f"Webcam check failed: {exc}")


def check_microphone() -> CheckResult:
    try:
        import sounddevice as sd
        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if inputs:
            return CheckResult(name="Microphone", ok=True, message=f"{len(inputs)} audio input device(s) available")
        return CheckResult(
            name="Microphone", ok=False, critical=True,
            message="No microphone detected. Voice input will not work.",
        )
    except Exception as exc:
        return CheckResult(name="Microphone", ok=False, critical=True, message=f"Audio device check failed: {exc}")


def run_all(config: Config) -> list[CheckResult]:
    return [check_ollama(config), check_models(config), check_webcam(), check_microphone()]


def all_critical_passed(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results if r.critical)
