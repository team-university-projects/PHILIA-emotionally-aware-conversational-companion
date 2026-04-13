"""
bot_profile.py — Bot configuration and persona.

Responsibilities:
  - Load and persist bot name, selected voice, and avatar image path
  - Expose a BotProfile dataclass consumed by UI, TTS, and PromptBuilder
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from config import Config

_PROFILE_PATH = Path(__file__).parent.parent / "assets" / "bot_profile.json"

# Mapping from avatar filename (stem) → display name
AVATAR_CATALOG: list[dict[str, str]] = [
    {"file": "luna.png",  "name": "Luna",  "desc": "Warm & Empathetic"},
    {"file": "orion.png", "name": "Orion", "desc": "Calm & Wise"},
    {"file": "sol.png",   "name": "Sol",   "desc": "Cheerful & Energetic"},
    {"file": "sage.png",  "name": "Sage",  "desc": "Thoughtful & Serene"},
    {"file": "nova.png",  "name": "Nova",  "desc": "Mysterious & Elegant"},
    {"file": "kai.png",   "name": "Kai",   "desc": "Friendly & Approachable"},
]

PERSONALITY_PRESETS: list[dict[str, str]] = [
    {"key": "empathetic",    "label": "Empathetic",    "desc": "Warm, caring, and deeply supportive"},
    {"key": "cheerful",      "label": "Cheerful",      "desc": "Upbeat, positive, and encouraging"},
    {"key": "calm",          "label": "Calm",          "desc": "Steady, grounding, and reassuring"},
    {"key": "professional",  "label": "Professional",  "desc": "Focused, clear, and constructive"},
]


@dataclass
class BotProfile:
    """Holds the user-configured bot persona."""
    name: str = "PHILIA"
    tts_voice: str = "en-US-JennyNeural"   # edge-tts voice ID
    avatar_path: str = "assets/avatars/luna.png"
    avatar_style: str = "luna"              # avatar key (filename stem)
    personality: str = "empathetic"         # controls system prompt style

    @classmethod
    def load(cls, config: Config) -> "BotProfile":
        """
        Load bot profile from JSON file, or return defaults.

        Args:
            config: Global Config (reserved for future path overrides).

        Returns:
            BotProfile instance.
        """
        if _PROFILE_PATH.exists():
            data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
            # Filter to known fields to handle old profiles gracefully
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        return cls()

    def save(self) -> None:
        """Persist the current profile to JSON."""
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @property
    def is_configured(self) -> bool:
        """True if the user has completed the setup wizard at least once."""
        return _PROFILE_PATH.exists()
