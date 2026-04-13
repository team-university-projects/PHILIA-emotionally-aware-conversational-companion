"""
interface.py — PHILIA UI orchestrator.

Responsibilities:
  - Manage the lifecycle of all UI windows (loading → setup → chat)
  - Expose a clean public API for the pipeline to update the UI
  - Bridge push-to-talk events from the UI to the pipeline
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import customtkinter as ctk

from bot.bot_profile import BotProfile
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Global CTk theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class Interface:
    """
    Manages all UI windows and acts as the bridge between the pipeline and GUI.

    Lifecycle:
        1. __init__() — create a hidden root window
        2. run_loading(initialise_fn) — show loading screen, run model init
        3. After init, if first run → show_setup_wizard()
        4. show_chat(profile) — open main chat window
        5. mainloop() — enter tkinter event loop (call once at the end of main)

    Pipeline callbacks (thread-safe, callable from any thread):
        - set_stage(stage)
        - update_emotion(label, confidence, ...)
        - add_message(role, text)
    """

    def __init__(self, config: Config) -> None:
        self._config   = config
        self._profile: BotProfile | None = None

        # Hidden root — keeps tkinter alive across toplevel windows
        self._root = ctk.CTk()
        self._root.withdraw()

        self._loading: "LoadingScreen | None" = None
        self._chat:    "ChatWindow | None"    = None

        self._on_ptt_callback: Callable | None = None

    # ── Public pipeline API ─────────────────────────────────────────────────────

    def set_ptt_callback(self, fn: Callable) -> None:
        """Register the function called when PTT is triggered from the UI."""
        self._on_ptt_callback = fn

    def set_stage(self, stage: str) -> None:
        if self._chat:
            self._chat.set_stage(stage)

    def update_emotion(
        self,
        label: str,
        confidence: float,
        audio_label: str = "—",
        facial_label: str = "—",
        text_label: str = "—",
    ) -> None:
        if self._chat:
            self._chat.update_emotion(label, confidence, audio_label, facial_label, text_label)

    def add_message(self, role: str, text: str) -> None:
        if self._chat:
            self._chat.add_message(role, text)

    # ── Window lifecycle ────────────────────────────────────────────────────────

    def run_loading(self, initialise_fn: Callable) -> None:
        """
        Show the loading/splash screen and run initialise_fn in a background
        thread. On success, opens chat window. On failure, shows error.

        Args:
            initialise_fn: Function(update_status, set_check) → BotProfile
                           that loads all heavy models and runs health checks.
                           Must call update_status(text, progress) and
                           set_check(name, ok, critical) as it progresses.
        """
        from ui.loading_screen import LoadingScreen

        self._loading = LoadingScreen(
            on_ready=self._on_loading_complete,
            on_error=self._on_loading_error,
        )

        def _run():
            try:
                profile = initialise_fn(
                    update_status=self._loading.set_status,
                    set_check=self._loading.set_check,
                )
                self._profile = profile
                self._loading.close_and_continue()
                self._root.after(0, self._post_loading)
            except Exception as exc:
                logger.exception("Initialisation failed")
                self._loading.show_error(str(exc))

        threading.Thread(target=_run, daemon=True).start()
        self._loading.mainloop()

    def _post_loading(self) -> None:
        """Called on main thread after loading screen closes."""
        from bot.bot_profile import _PROFILE_PATH
        profile = self._profile or BotProfile()
        if not _PROFILE_PATH.exists():
            self._show_setup_wizard(on_complete=self._open_chat)
        else:
            self._open_chat(profile)

    def _on_loading_complete(self) -> None:
        pass   # handled via close_and_continue

    def _on_loading_error(self, msg: str) -> None:
        logger.error("Loading error: %s", msg)

    def _show_setup_wizard(self, on_complete: Callable[[BotProfile], None]) -> None:
        from ui.setup_screen import SetupScreen
        wizard = SetupScreen(
            parent=self._root,
            config=self._config,
            on_complete=on_complete,
        )
        self._root.wait_window(wizard)

    def _open_chat(self, profile: BotProfile) -> None:
        self._profile = profile
        from ui.chat_window import ChatWindow
        self._chat = ChatWindow(
            profile=profile,
            on_ptt=self._on_ptt_callback or (lambda: None),
            on_settings=self._show_settings,
        )
        # Run the chat window as the main event loop
        self._chat.mainloop()

    # ── Settings ─────────────────────────────────────────────────────────────

    def _show_settings(self) -> None:
        """Open a simple settings panel over the chat window."""
        if not self._chat:
            return

        from ui.setup_screen import SetupScreen

        def _on_save(new_profile: BotProfile):
            self._profile = new_profile
            logger.info("Profile updated: name=%s voice=%s", new_profile.name, new_profile.tts_voice)
            # Notify the chat window (it won't auto-reload avatar without restart, so show a tip)
            if self._chat:
                self._chat.add_message(
                    "bot",
                    f"Profile saved! Changes to name, avatar, and voice will fully apply on next start.",
                )

        wizard = SetupScreen(
            parent=self._chat,
            config=self._config,
            on_complete=_on_save,
        )

    def mainloop(self) -> None:
        """Enter the tkinter event loop (only if root is still alive)."""
        try:
            self._root.mainloop()
        except Exception:
            pass
