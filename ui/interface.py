"""
interface.py — Push-to-Talk UI and avatar response playback.

Responsibilities:
  - Display the bot avatar
  - Listen for PTT key/button events (blocking wait)
  - Play synthesised audio response back to the user
  - Provide a quit/exit mechanism
"""

from __future__ import annotations

from bot.bot_profile import BotProfile


class Interface:
    """Handles PTT input and avatar + audio output."""

    def __init__(self, profile: BotProfile) -> None:
        self._profile = profile
        # TODO: initialise display / GUI framework (tkinter, pygame, or CLI)

    def wait_for_ptt(self) -> None:
        """
        Block until the user presses the push-to-talk key.
        Raises SystemExit on quit command.
        """
        raise NotImplementedError

    def play_response(self, audio_bytes: bytes, text: str) -> None:
        """
        Play synthesised audio and optionally display the transcript.

        Args:
            audio_bytes: WAV bytes of the TTS response.
            text:        Response text (for subtitle / log display).
        """
        raise NotImplementedError

    def show_avatar(self) -> None:
        """Render the bot avatar image in the UI."""
        raise NotImplementedError
