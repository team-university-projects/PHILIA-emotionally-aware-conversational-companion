"""
loading_screen.py — Splash/loading screen shown while PHILIA initialises models.

Displays:
  - PHILIA branding with animated pulse
  - A health check status list (Ollama, Models, Webcam, Mic)
  - A progress bar that advances as each component loads
  - Error callout if any critical check fails
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

# ── Theme tokens ───────────────────────────────────────────────────────────────

_BG           = "#0F0D1A"
_SURFACE      = "#1A1730"
_ACCENT       = "#7C4DFF"
_ACCENT_DIM   = "#4A2FA0"
_TEXT_PRIMARY = "#F0EEFF"
_TEXT_DIM     = "#8B80B0"
_GREEN        = "#4CAF82"
_RED          = "#FF5370"
_YELLOW       = "#FFD166"

_ASSETS = Path(__file__).parent.parent / "assets"


class LoadingScreen(ctk.CTk):
    """
    Splash screen shown during model initialisation.

    Usage:
        screen = LoadingScreen(on_ready_callback)
        screen.mainloop()          # blocks until models are loaded
        # on_ready_callback is called on the main thread after success
    """

    def __init__(self, on_ready: Callable, on_error: Callable[[str], None]) -> None:
        super().__init__()
        self._on_ready = on_ready
        self._on_error = on_error
        self._icon_img = None         # populated in _build_ui after window exists

        self._setup_window()
        self._build_ui()
        self._pulse_active = True
        self._animate_pulse()

    # ── Window setup ────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title("PHILIA — Starting")
        self.configure(fg_color=_BG)
        self.geometry("520x560")
        self.resizable(False, False)
        self.overrideredirect(True)   # borderless
        # Centre on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 520) // 2
        y = (self.winfo_screenheight() - 560) // 2
        self.geometry(f"+{x}+{y}")

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Icon / logo — must be created after the Tk window is fully alive
        icon_path = _ASSETS / "icon.png"
        if icon_path.exists():
            try:
                img = Image.open(icon_path).resize((96, 96), Image.LANCZOS)
                self._icon_img = ctk.CTkImage(light_image=img, dark_image=img, size=(96, 96))
                icon_lbl = ctk.CTkLabel(self, image=self._icon_img, text="")
                icon_lbl.pack(pady=(40, 0))
            except Exception:
                ctk.CTkLabel(self, text="◈", font=("Segoe UI", 52), text_color=_ACCENT).pack(pady=(40, 0))
        else:
            ctk.CTkLabel(self, text="◈", font=("Segoe UI", 52), text_color=_ACCENT).pack(pady=(40, 0))

        # Title
        ctk.CTkLabel(
            self, text="PHILIA", font=("Segoe UI", 28, "bold"),
            text_color=_TEXT_PRIMARY,
        ).pack(pady=(10, 0))
        ctk.CTkLabel(
            self, text="Emotionally Aware Conversational Companion",
            font=("Segoe UI", 11), text_color=_TEXT_DIM,
        ).pack(pady=(2, 20))

        # Animated pulse ring (canvas)
        self._pulse_canvas = tk.Canvas(self, width=120, height=120, bg=_BG, highlightthickness=0)
        self._pulse_canvas.pack()
        self._pulse_r = 40
        self._pulse_alpha = 1.0
        self._pulse_canvas.create_oval(10, 10, 110, 110, outline=_ACCENT, width=2, tags="ring0")
        self._pulse_canvas.create_oval(25, 25, 95, 95, outline=_ACCENT_DIM, width=1, tags="ring1")
        self._pulse_canvas.create_oval(38, 38, 82, 82, fill=_ACCENT, outline="", tags="core")

        # Status label
        self._status_var = tk.StringVar(value="Initialising…")
        self._status_lbl = ctk.CTkLabel(
            self, textvariable=self._status_var,
            font=("Segoe UI", 11), text_color=_TEXT_DIM,
        )
        self._status_lbl.pack(pady=(12, 6))

        # Progress bar
        self._progress = ctk.CTkProgressBar(
            self, width=340, height=6,
            fg_color=_SURFACE, progress_color=_ACCENT,
        )
        self._progress.pack(pady=(0, 16))
        self._progress.set(0)

        # Check list
        self._checks_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._checks_frame.pack(fill="x", padx=60)
        self._check_labels: dict[str, ctk.CTkLabel] = {}

        for name in ["Ollama", "Emotion Models", "Webcam", "Microphone"]:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            dot = ctk.CTkLabel(row, text="◌", font=("Segoe UI", 13), text_color=_TEXT_DIM, width=28)
            dot.pack(side="left")
            ctk.CTkLabel(row, text=name, font=("Segoe UI", 10), text_color=_TEXT_DIM, anchor="w").pack(side="left")
            self._check_labels[name] = dot

        # Error callout (hidden initially)
        self._error_var = tk.StringVar(value="")
        self._error_lbl = ctk.CTkLabel(
            self, textvariable=self._error_var,
            font=("Segoe UI", 10), text_color=_RED,
            wraplength=440, justify="center",
        )
        self._error_lbl.pack(pady=(12, 0))

    # ── Animation ───────────────────────────────────────────────────────────────

    def _animate_pulse(self) -> None:
        if not self._pulse_active:
            return
        try:
            # Grow / shrink outer ring
            self._pulse_r = self._pulse_r + 0.5
            if self._pulse_r > 55:
                self._pulse_r = 40
            r = self._pulse_r
            self._pulse_canvas.coords("ring0", 60-r, 60-r, 60+r, 60+r)
            # Fade alpha — just cycle colour between accent shades
            alpha = int(255 * (1.0 - (r - 40) / 15))
            hex_col = f"#{_ACCENT[1:3]}{int(alpha/255 * int(_ACCENT[3:5], 16)):02x}{_ACCENT[5:]}"
        except Exception:
            pass
        self.after(40, self._animate_pulse)

    # ── Public control API (called from worker thread via .after) ───────────────

    def set_status(self, text: str, progress: float) -> None:
        """Thread-safe status update."""
        self.after(0, lambda: self._status_var.set(text))
        self.after(0, lambda: self._progress.set(progress))

    def set_check(self, name: str, ok: bool, critical: bool = True) -> None:
        """Thread-safe check item update."""
        def _update():
            dot = self._check_labels.get(name)
            if dot:
                if ok:
                    dot.configure(text="✓", text_color=_GREEN)
                elif critical:
                    dot.configure(text="✗", text_color=_RED)
                else:
                    dot.configure(text="⚠", text_color=_YELLOW)
        self.after(0, _update)

    def show_error(self, message: str) -> None:
        """Display a critical error and a quit button."""
        def _update():
            self._pulse_active = False
            self._status_var.set("Cannot start PHILIA")
            self._error_var.set(message)
            ctk.CTkButton(
                self, text="Quit", width=120,
                fg_color=_RED, hover_color="#C0392B",
                command=self.quit,
            ).pack(pady=(16, 0))
        self.after(0, _update)

    def close_and_continue(self) -> None:
        """Destroy the loading screen; caller shows main window."""
        self._pulse_active = False
        self.after(0, self.destroy)
