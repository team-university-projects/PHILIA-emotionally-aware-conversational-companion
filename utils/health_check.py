"""
health_check.py — Pre-launch system readiness checks for PHILIA.

Checks:
  1. Ollama server reachability (localhost:11434)
  2. Fine-tuned model directories present
  3. Webcam accessible
  4. Microphone accessible

Returns a list of CheckResult objects for display in a pre-launch dialog.
"""

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
    critical: bool = True   # If True and not ok, PHILIA cannot start


def check_ollama(config: Config) -> CheckResult:
    """Verify Ollama HTTP API is reachable at localhost:11434."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        # Check if the required model is available
        model_name = config.models.llm_model
        available = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        if model_name not in available and model_name.split(":")[0] not in available:
            return CheckResult(
                name="Ollama Model",
                ok=False,
                message=(
                    f"Model '{model_name}' not found in Ollama.\n"
                    f"Run: ollama pull {model_name}"
                ),
                critical=True,
            )
        return CheckResult(
            name="Ollama",
            ok=True,
            message=f"Ollama running • model '{model_name}' available",
        )
    except (urllib.error.URLError, OSError):
        return CheckResult(
            name="Ollama",
            ok=False,
            message=(
                "Cannot reach Ollama at localhost:11434.\n"
                "Start it with: ollama serve"
            ),
            critical=True,
        )


def check_models(config: Config) -> CheckResult:
    """Verify fine-tuned model directories exist on disk."""
    root = Path(__file__).parent.parent
    required = [
        ("Audio Emotion",   root / config.models.audio_emotion_model_name),
        ("Facial Emotion",  root / config.models.facial_emotion_model_name),
        ("Text Emotion",    root / config.models.text_emotion_model_name),
    ]
    missing = [name for name, path in required if not path.exists()]
    if missing:
        return CheckResult(
            name="Emotion Models",
            ok=False,
            message=f"Missing fine-tuned model(s): {', '.join(missing)}",
            critical=True,
        )
    return CheckResult(
        name="Emotion Models",
        ok=True,
        message="All fine-tuned emotion models found",
    )


def check_webcam() -> CheckResult:
    """Check if at least one webcam is accessible via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return CheckResult(
                name="Webcam",
                ok=True,
                message="Webcam accessible (index 0)",
            )
        return CheckResult(
            name="Webcam",
            ok=False,
            message="No webcam found at index 0. Facial emotion will use neutral fallback.",
            critical=False,   # Non-critical: system runs without webcam
        )
    except Exception as exc:
        return CheckResult(
            name="Webcam",
            ok=False,
            message=f"Webcam check failed: {exc}",
            critical=False,
        )


def check_microphone() -> CheckResult:
    """Check if at least one audio input device is available."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if inputs:
            return CheckResult(
                name="Microphone",
                ok=True,
                message=f"{len(inputs)} audio input device(s) available",
            )
        return CheckResult(
            name="Microphone",
            ok=False,
            message="No microphone detected. Voice input will not work.",
            critical=True,
        )
    except Exception as exc:
        return CheckResult(
            name="Microphone",
            ok=False,
            message=f"Audio device check failed: {exc}",
            critical=True,
        )


def run_all(config: Config) -> list[CheckResult]:
    """Run all health checks and return results."""
    return [
        check_ollama(config),
        check_models(config),
        check_webcam(),
        check_microphone(),
    ]


def all_critical_passed(results: list[CheckResult]) -> bool:
    """True if no critical check has failed."""
    return all(r.ok for r in results if r.critical)
