# Bot persona configuration — per-profile JSON files in assets/profiles/.

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from config import Config

_PROFILES_DIR = Path(__file__).parent.parent / "assets" / "profiles"
_AVATARS_DIR  = _PROFILES_DIR / "avatars"

# Preset avatar filenames in assets/avatars/ — shown without names in the wizard.
PRESET_AVATARS: list[str] = [
    "luna.png", "orion.png", "sol.png",
    "sage.png", "nova.png",  "kai.png",
]

PERSONALITY_PRESETS: list[dict[str, str]] = [
    {"key": "empathetic",   "label": "Empathetic",   "desc": "Warm, caring, and deeply supportive"},
    {"key": "cheerful",     "label": "Cheerful",     "desc": "Upbeat, positive, and encouraging"},
    {"key": "calm",         "label": "Calm",         "desc": "Steady, grounding, and reassuring"},
    {"key": "professional", "label": "Professional", "desc": "Focused, clear, and constructive"},
]


@dataclass
class BotProfile:
    name:        str = "PHILIA"
    tts_voice:   str = "en-US-JennyNeural"
    avatar_path: str = "assets/avatars/luna.png"
    personality: str = "empathetic"
    last_used:   str = ""

    # ── Persistence ────────────────────────────────────────────────────────────

    @classmethod
    def list_all(cls) -> list["BotProfile"]:
        """Return all saved profiles sorted by most recently used (newest first)."""
        _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profiles: list["BotProfile"] = []
        for path in sorted(_PROFILES_DIR.glob("*.json")):
            try:
                data  = json.loads(path.read_text(encoding="utf-8"))
                known = set(cls.__dataclass_fields__)
                profiles.append(cls(**{k: v for k, v in data.items() if k in known}))
            except Exception:
                pass
        profiles.sort(key=lambda p: p.last_used, reverse=True)
        return profiles

    @classmethod
    def load(cls, config: Config) -> "BotProfile":
        """Return the most recently used profile, or a fresh default."""
        all_profiles = cls.list_all()
        return all_profiles[0] if all_profiles else cls()

    def save(self) -> None:
        """Write this profile to assets/profiles/<name>.json."""
        _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        _AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        self.last_used = datetime.now().isoformat()
        path = _PROFILES_DIR / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def touch(self) -> None:
        """Update last_used timestamp without altering other fields."""
        self.last_used = datetime.now().isoformat()
        path = _PROFILES_DIR / f"{self.name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["last_used"] = self.last_used
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def delete(self) -> None:
        """Remove this profile's JSON file from disk."""
        path = _PROFILES_DIR / f"{self.name}.json"
        if path.exists():
            path.unlink()

    # ── Computed ───────────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.list_all())

    def resolve_avatar_path(self) -> Path:
        """Return the absolute Path to this profile's avatar image."""
        return Path(__file__).parent.parent / self.avatar_path
