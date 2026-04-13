"""
interface.py — PHILIA UI orchestrator.

Responsibilities:
  - Manage the lifecycle of all UI windows (loading → setup → chat)
  - Expose a clean public API for the pipeline to update the UI
  - Bridge push-to-talk events from the UI to the pipeline

Thread model:
  - main thread:       tkinter event loop (one window at a time)
  - background thread: model initialisation (reports progress via after())

Window sequence:
  LoadingScreen.mainloop()  →  (closes itself)
  SetupScreen (first run)   →  (wait_window)
  ChatWindow.mainloop()     →  app exits
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import customtkinter as ctk

from bot.bot_profile import BotProfile, _PROFILE_PATH
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Global CTk theme — set before any window is created
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class Interface:
    """
    Manages all UI windows and bridges the pipeline ↔ GUI.

    Usage in main():
        ui = Interface(config)
        ui.set_ptt_callback(fn)
        ui.run_loading(initialise_fn)   # blocks until chat window closes
    """

    def __init__(self, config: Config) -> None:
        self._config   = config
        self._profile: BotProfile | None = None
        self._chat:    "ChatWindow | None" = None
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
        Phase 1: Show loading splash + run model init in background thread.
        Blocks until the entire app lifecycle completes (chat window closed).

        initialise_fn signature:
            fn(update_status: Callable, set_check: Callable) -> BotProfile
        """
        from ui.loading_screen import LoadingScreen

        loading = LoadingScreen(
            on_ready=lambda: None,   # unused — transition is driven below
            on_error=lambda msg: None,
        )

        def _run_init():
            try:
                profile = initialise_fn(
                    update_status=loading.set_status,
                    set_check=loading.set_check,
                )
                self._profile = profile
                # Schedule transition on the main thread (loading is still alive here)
                loading.after(0, lambda: self._transition_from_loading(loading))
            except Exception as exc:
                logger.exception("Initialisation failed")
                loading.show_error(str(exc))

        threading.Thread(target=_run_init, daemon=True).start()
        loading.mainloop()   # ← main thread blocks here while loading screen is open

        # ── After loading.mainloop() returns, run setup + chat ──────────────────
        self._run_post_loading()

    def _transition_from_loading(self, loading: "LoadingScreen") -> None:
        """Called on main thread: close loading screen so mainloop() can return."""
        loading._pulse_active = False
        loading.after(300, loading.destroy)   # small delay so final status is visible

    def _run_post_loading(self) -> None:
        """Called on main thread after loading screen closes."""
        profile = self._profile or BotProfile()

        # First-run: show wizard
        if not _PROFILE_PATH.exists():
            profile = self._run_setup_wizard()

        self._open_chat(profile)

    def _run_setup_wizard(self) -> BotProfile:
        """Run the setup wizard modally and return the saved profile."""
        from ui.setup_screen import SetupScreen

        result_holder: list[BotProfile] = []

        # SetupScreen needs a parent — create a temporary hidden root for it
        tmp_root = ctk.CTk()
        tmp_root.withdraw()

        def _on_complete(p: BotProfile):
            result_holder.append(p)
            tmp_root.destroy()

        wizard = SetupScreen(
            parent=tmp_root,
            config=self._config,
            on_complete=_on_complete,
        )
        wizard.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable X button
        tmp_root.mainloop()

        return result_holder[0] if result_holder else BotProfile()

    def _open_chat(self, profile: BotProfile) -> None:
        """Open the main chat window (blocks until it's closed)."""
        self._profile = profile
        from ui.chat_window import ChatWindow
        self._chat = ChatWindow(
            profile=profile,
            on_ptt=self._on_ptt_callback or (lambda: None),
            on_settings=self._show_settings,
        )
        self._chat.mainloop()

    # ── Settings ─────────────────────────────────────────────────────────────

    def _show_settings(self) -> None:
        """Re-open the setup wizard as a settings panel over the chat window."""
        if not self._chat:
            return

        from ui.setup_screen import SetupScreen

        def _on_save(new_profile: BotProfile):
            self._profile = new_profile
            logger.info("Profile updated: name=%s voice=%s", new_profile.name, new_profile.tts_voice)
            if self._chat:
                self._chat.add_message(
                    "bot",
                    "Profile saved! New avatar and voice will apply on the next start.",
                )

        SetupScreen(
            parent=self._chat,
            config=self._config,
            on_complete=_on_save,
        )
