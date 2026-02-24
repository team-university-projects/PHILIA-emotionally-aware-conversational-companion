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


@dataclass
class BotProfile:
    """Holds the user-configured bot persona."""
    name: str = "PHILIA"
    voice: str = "default"              # Voice identifier for TTS
    avatar_path: str = "assets/avatars/default.png"

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
            return cls(**data)
        return cls()

    def save(self) -> None:
        """Persist the current profile to JSON."""
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
