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
        # Icon / logo
        icon_path = _ASSETS / "icon.png"
        if icon_path.exists():
            img = Image.open(icon_path).resize((96, 96), Image.LANCZOS)
            self._icon_img = ImageTk.PhotoImage(img)
            tk.Label(self, image=self._icon_img, bg=_BG).pack(pady=(40, 0))
        else:
            tk.Label(self, text="◈", font=("Segoe UI", 52), fg=_ACCENT, bg=_BG).pack(pady=(40, 0))

        # Title
        tk.Label(
            self, text="PHILIA", font=("Segoe UI", 28, "bold"),
            fg=_TEXT_PRIMARY, bg=_BG,
        ).pack(pady=(10, 0))
        tk.Label(
            self, text="Emotionally Aware Conversational Companion",
            font=("Segoe UI", 11), fg=_TEXT_DIM, bg=_BG,
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
        tk.Label(
            self, textvariable=self._status_var,
            font=("Segoe UI", 11), fg=_TEXT_DIM, bg=_BG,
        ).pack(pady=(12, 6))

        # Progress bar
        self._progress = ctk.CTkProgressBar(
            self, width=340, height=6,
            fg_color=_SURFACE, progress_color=_ACCENT,
        )
        self._progress.pack(pady=(0, 16))
        self._progress.set(0)

        # Check list
        self._checks_frame = tk.Frame(self, bg=_BG)
        self._checks_frame.pack(fill="x", padx=60)
        self._check_labels: dict[str, tk.Label] = {}

        for name in ["Ollama", "Emotion Models", "Webcam", "Microphone"]:
            row = tk.Frame(self._checks_frame, bg=_BG)
            row.pack(fill="x", pady=1)
            dot = tk.Label(row, text="◌", font=("Segoe UI", 13), fg=_TEXT_DIM, bg=_BG, width=2)
            dot.pack(side="left")
            lbl = tk.Label(row, text=name, font=("Segoe UI", 10), fg=_TEXT_DIM, bg=_BG, anchor="w")
            lbl.pack(side="left")
            self._check_labels[name] = dot

        # Error callout (hidden initially)
        self._error_var = tk.StringVar(value="")
        self._error_label = tk.Label(
            self, textvariable=self._error_var,
            font=("Segoe UI", 10), fg=_RED, bg=_BG,
            wraplength=440, justify="center",
        )
        self._error_label.pack(pady=(12, 0))

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
                    dot.config(text="✓", fg=_GREEN)
                elif critical:
                    dot.config(text="✗", fg=_RED)
                else:
                    dot.config(text="⚠", fg=_YELLOW)
        self.after(0, _update)

    def show_error(self, message: str) -> None:
        """Display a critical error and a retry/quit button."""
        def _update():
            self._pulse_active = False
            self._status_var.set("Cannot start PHILIA")
            self._error_var.set(message)
            ctk.CTkButton(
                self, text="Quit", width=120,
                fg_color=_RED, hover_color="#C0392B",
                command=self.destroy,
            ).pack(pady=(16, 0))
        self.after(0, _update)

    def close_and_continue(self) -> None:
        """Destroy the loading screen; caller shows main window."""
        self._pulse_active = False
        self.after(0, self.destroy)
