"""
setup_screen.py — First-run bot creation wizard (3 steps).

Step 1 — Name & Personality
Step 2 — Choose Avatar
Step 3 — Choose Voice

Saves a BotProfile on completion and calls the on_complete callback.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from bot.bot_profile import (
    BotProfile, AVATAR_CATALOG, PERSONALITY_PRESETS,
)
from speech.tts import TextToSpeech, TTSParams, list_voices
from config import Config

# ── Theme ──────────────────────────────────────────────────────────────────────

_BG           = "#0F0D1A"
_SURFACE      = "#1A1730"
_SURFACE2     = "#231F3A"
_ACCENT       = "#7C4DFF"
_ACCENT_HOVER = "#9D71FF"
_BORDER       = "#3A3060"
_TEXT_PRIMARY = "#F0EEFF"
_TEXT_DIM     = "#8B80B0"
_GREEN        = "#4CAF82"
_STEP_TOTAL   = 3

_ASSETS = Path(__file__).parent.parent / "assets"

_PREVIEW_PHRASE = "Hello! I'm here to listen and support you."


class SetupScreen(ctk.CTkToplevel):
    """
    Three-step onboarding wizard.

    Args:
        parent:      The root tk window (may be hidden).
        config:      Global Config instance.
        on_complete: Callback(profile: BotProfile) called after wizard finishes.
    """

    def __init__(self, parent: tk.Misc, config: Config, on_complete: Callable[[BotProfile], None]) -> None:
        super().__init__(parent)
        self._config = config
        self._on_complete = on_complete
        self._step = 0

        # Working profile state
        self._name_var        = tk.StringVar(value="PHILIA")
        self._personality_var = tk.StringVar(value="empathetic")
        self._avatar_var      = tk.StringVar(value="luna.png")
        self._voice_var       = tk.StringVar(value="en-US-JennyNeural")
        self._mic_device_idx  = tk.IntVar(value=-1)  # -1 = system default

        # Build mic device list once
        self._mic_options: list[tuple[int, str]] = self._enumerate_mics()

        self._avatar_images: dict[str, ImageTk.PhotoImage] = {}
        self._avatar_buttons: dict[str, ctk.CTkButton] = {}

        self._tts = TextToSpeech(config, BotProfile())

        self._setup_window()
        self._build_shell()
        self._show_step(0)

    # ── Window ──────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title("PHILIA — Create Your Companion")
        self.configure(fg_color=_BG)
        self.geometry("760x640")
        self.resizable(False, False)
        self.grab_set()
        self.focus_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 760) // 2
        y = (self.winfo_screenheight() - 640) // 2
        self.geometry(f"+{x}+{y}")

    # ── Shell layout (header + body + footer always present) ───────────────────

    def _build_shell(self) -> None:
        # ── Header ────────────────────────────────────────────────────────────
        self._header = tk.Frame(self, bg=_BG)
        self._header.pack(fill="x", padx=40, pady=(28, 0))

        self._title_lbl = tk.Label(
            self._header, text="", font=("Segoe UI", 22, "bold"),
            fg=_TEXT_PRIMARY, bg=_BG,
        )
        self._title_lbl.pack(anchor="w")
        self._subtitle_lbl = tk.Label(
            self._header, text="", font=("Segoe UI", 12),
            fg=_TEXT_DIM, bg=_BG,
        )
        self._subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Step indicator dots
        dots_frame = tk.Frame(self._header, bg=_BG)
        dots_frame.pack(anchor="w", pady=(10, 0))
        self._dot_labels = []
        for i in range(_STEP_TOTAL):
            d = tk.Label(dots_frame, text="●", font=("Segoe UI", 10), bg=_BG, fg=_TEXT_DIM)
            d.pack(side="left", padx=3)
            self._dot_labels.append(d)

        # Divider
        tk.Frame(self, bg=_BORDER, height=1).pack(fill="x", padx=40, pady=(14, 0))

        # ── Body (swapped per step) ────────────────────────────────────────────
        self._body = tk.Frame(self, bg=_BG)
        self._body.pack(fill="both", expand=True, padx=40, pady=20)

        # ── Footer ────────────────────────────────────────────────────────────
        tk.Frame(self, bg=_BORDER, height=1).pack(fill="x", padx=40)
        footer = tk.Frame(self, bg=_BG)
        footer.pack(fill="x", padx=40, pady=16)

        self._back_btn = ctk.CTkButton(
            footer, text="← Back", width=110,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, command=self._go_back,
        )
        self._back_btn.pack(side="left")

        self._next_btn = ctk.CTkButton(
            footer, text="Next →", width=130,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            font=("Segoe UI", 13, "bold"), command=self._go_next,
        )
        self._next_btn.pack(side="right")

    # ── Step navigation ─────────────────────────────────────────────────────────

    def _show_step(self, step: int) -> None:
        self._step = step
        for widget in self._body.winfo_children():
            widget.destroy()

        # Dots
        for i, d in enumerate(self._dot_labels):
            d.config(fg=_ACCENT if i == step else _TEXT_DIM)

        # Back button visibility
        if step == 0:
            self._back_btn.configure(state="disabled", fg_color=_SURFACE, text_color=_BORDER)
        else:
            self._back_btn.configure(state="normal", fg_color=_SURFACE2, text_color=_TEXT_DIM)

        # Next/Finish button label
        if step == _STEP_TOTAL - 1:
            self._next_btn.configure(text="Start Chatting ✓")
        else:
            self._next_btn.configure(text="Next →")

        _BUILDERS = [self._build_step_name, self._build_step_avatar, self._build_step_voice]
        _TITLES   = ["Name Your Companion", "Choose an Avatar", "Pick a Voice"]
        _SUBS     = [
            "Give your AI companion a name and personality.",
            "Select the face that feels right for your companion.",
            "Choose how your companion will speak to you.",
        ]
        self._title_lbl.config(text=_TITLES[step])
        self._subtitle_lbl.config(text=_SUBS[step])
        _BUILDERS[step]()

    def _go_next(self) -> None:
        if self._step < _STEP_TOTAL - 1:
            self._show_step(self._step + 1)
        else:
            self._finish()

    def _go_back(self) -> None:
        if self._step > 0:
            self._show_step(self._step - 1)

    # ── Step 1: Name & Personality ─────────────────────────────────────────────

    def _build_step_name(self) -> None:
        body = self._body

        tk.Label(body, text="Companion Name", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(10, 4))
        entry = ctk.CTkEntry(
            body, textvariable=self._name_var, width=320, height=40,
            fg_color=_SURFACE, border_color=_BORDER,
            text_color=_TEXT_PRIMARY, font=("Segoe UI", 13),
            placeholder_text="e.g.  PHILIA, Nova, Kai…",
        )
        entry.pack(anchor="w")

        tk.Label(body, text="Personality", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(24, 8))

        grid = tk.Frame(body, bg=_BG)
        grid.pack(anchor="w")

        for i, preset in enumerate(PERSONALITY_PRESETS):
            col = i % 2
            row = i // 2
            card = tk.Frame(
                grid, bg=_SURFACE, bd=0,
                width=320, height=80,
                cursor="hand2",
            )
            card.grid(row=row, column=col, padx=(0, 12) if col == 0 else 0, pady=6, sticky="w")
            card.pack_propagate(False)

            key   = preset["key"]
            label = preset["label"]
            desc  = preset["desc"]

            def _select(k=key, c=card):
                self._personality_var.set(k)
                self._refresh_personality_cards(grid)

            card.bind("<Button-1>", lambda e, k=key, c=card: _select(k, c))

            tk.Label(card, text=label, font=("Segoe UI", 12, "bold"),
                     fg=_TEXT_PRIMARY, bg=_SURFACE, cursor="hand2").pack(anchor="w", padx=14, pady=(14, 2))
            tk.Label(card, text=desc, font=("Segoe UI", 10),
                     fg=_TEXT_DIM, bg=_SURFACE, cursor="hand2").pack(anchor="w", padx=14)

        self._refresh_personality_cards(grid)

        # ── Microphone selector ───────────────────────────────────────────────
        tk.Label(body, text="Microphone", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(20, 6))
        tk.Label(
            body,
            text="Choose the mic you speak into. Using the right device greatly improves transcription.",
            font=("Segoe UI", 10), fg=_TEXT_DIM, bg=_BG, wraplength=650, justify="left",
        ).pack(anchor="w", pady=(0, 6))

        mic_frame = ctk.CTkScrollableFrame(body, width=640, height=110, fg_color=_SURFACE)
        mic_frame.pack(anchor="w", fill="x")

        # System default option
        ctk.CTkRadioButton(
            mic_frame, text="System Default (let Windows decide)",
            variable=self._mic_device_idx, value=-1,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            text_color=_TEXT_PRIMARY, font=("Segoe UI", 11),
        ).pack(anchor="w", padx=12, pady=(8, 4))

        for idx, name in self._mic_options:
            # Flag recommended devices
            lower = name.lower()
            is_rec = any(k in lower for k in ("headset", "headphone", "usb", "jabra", "sony", "wh-", "bose", "airpods"))
            label = f"{name}  ✓ Recommended" if is_rec else name
            color = _GREEN if is_rec else _TEXT_PRIMARY
            ctk.CTkRadioButton(
                mic_frame, text=label,
                variable=self._mic_device_idx, value=idx,
                fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                text_color=color, font=("Segoe UI", 11),
            ).pack(anchor="w", padx=12, pady=2)

    def _refresh_personality_cards(self, grid: tk.Frame) -> None:
        selected = self._personality_var.get()
        for widget in grid.winfo_children():
            if isinstance(widget, tk.Frame):
                key_lbl = widget.winfo_children()[0] if widget.winfo_children() else None
                if key_lbl:
                    # Find which preset this card represents by its label text
                    for p in PERSONALITY_PRESETS:
                        if key_lbl.cget("text") == p["label"]:
                            is_sel = p["key"] == selected
                            widget.config(bg=_ACCENT if is_sel else _SURFACE)
                            for child in widget.winfo_children():
                                child.config(bg=_ACCENT if is_sel else _SURFACE)

    # ── Step 2: Avatar ─────────────────────────────────────────────────────────

    def _build_step_avatar(self) -> None:
        body = self._body
        avatar_dir = _ASSETS / "avatars"
        grid = tk.Frame(body, bg=_BG)
        grid.pack(fill="both", expand=True)

        for i, av in enumerate(AVATAR_CATALOG):
            col = i % 3
            row = i // 3
            filename = av["file"]
            name     = av["name"]
            desc     = av["desc"]

            card = tk.Frame(grid, bg=_SURFACE, cursor="hand2", width=200, height=220)
            card.grid(row=row, column=col, padx=8, pady=8)
            card.pack_propagate(False)

            # Avatar image
            img_path = avatar_dir / filename
            if img_path.exists():
                img = Image.open(img_path).resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._avatar_images[filename] = photo
                tk.Label(card, image=photo, bg=_SURFACE, cursor="hand2").pack(pady=(16, 4))
            else:
                tk.Label(card, text="◈", font=("Segoe UI", 36), fg=_ACCENT, bg=_SURFACE).pack(pady=(20, 4))

            tk.Label(card, text=name, font=("Segoe UI", 12, "bold"),
                     fg=_TEXT_PRIMARY, bg=_SURFACE, cursor="hand2").pack()
            tk.Label(card, text=desc, font=("Segoe UI", 9),
                     fg=_TEXT_DIM, bg=_SURFACE, cursor="hand2").pack()

            def _on_select(f=filename, c=card):
                self._avatar_var.set(f)
                self._refresh_avatar_selection(grid)

            card.bind("<Button-1>", lambda e, f=filename, c=card: _on_select(f, c))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, f=filename, c=card: _on_select(f, c))

        self._refresh_avatar_selection(grid)

    def _refresh_avatar_selection(self, grid: tk.Frame) -> None:
        selected = self._avatar_var.get()
        for i, av in enumerate(AVATAR_CATALOG):
            card = grid.grid_slaves(row=i // 3, column=i % 3)
            if card:
                c = card[0]
                is_sel = av["file"] == selected
                bg = _ACCENT if is_sel else _SURFACE
                c.config(bg=bg, highlightthickness=2 if is_sel else 0,
                          highlightbackground=_ACCENT if is_sel else _SURFACE)
                for child in c.winfo_children():
                    child.config(bg=bg)

    # ── Step 3: Voice ──────────────────────────────────────────────────────────

    def _build_step_voice(self) -> None:
        body = self._body
        voices = list_voices()

        tk.Label(body, text="Voice", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(10, 8))

        # Scrollable list of radio buttons
        scroll_frame = ctk.CTkScrollableFrame(
            body, width=640, height=320, fg_color=_SURFACE,
        )
        scroll_frame.pack(fill="both", expand=True)

        for v in voices:
            row = tk.Frame(scroll_frame, bg=_SURFACE, cursor="hand2")
            row.pack(fill="x", pady=2, padx=4)

            rb = ctk.CTkRadioButton(
                row,
                text=v["label"],
                variable=self._voice_var,
                value=v["id"],
                fg_color=_ACCENT,
                hover_color=_ACCENT_HOVER,
                text_color=_TEXT_PRIMARY,
                font=("Segoe UI", 12),
            )
            rb.pack(side="left", pady=8, padx=12)

            preview_btn = ctk.CTkButton(
                row, text="▶ Preview", width=90, height=28,
                fg_color=_SURFACE2, hover_color=_BORDER,
                text_color=_TEXT_DIM, font=("Segoe UI", 10),
                command=lambda vid=v["id"]: self._preview_voice(vid),
            )
            preview_btn.pack(side="right", padx=12)

        # Preview indicator
        self._preview_status = tk.Label(
            body, text="", font=("Segoe UI", 10),
            fg=_GREEN, bg=_BG,
        )
        self._preview_status.pack(anchor="w", pady=(8, 0))

    def _preview_voice(self, voice_id: str) -> None:
        self._preview_status.config(text=f"▶ Playing preview for {voice_id.split('-')[1]}…")
        params = TTSParams(rate=175, pitch=1.0, volume=1.0, style="neutral")
        self._tts.preview(_PREVIEW_PHRASE, voice_id)
        self.after(3000, lambda: self._preview_status.config(text=""))

    # ── Finish ────────────────────────────────────────────────────────────────

    def _finish(self) -> None:
        name   = self._name_var.get().strip() or "PHILIA"
        avatar = self._avatar_var.get()
        stem   = Path(avatar).stem  # "luna.png" -> "luna"

        profile = BotProfile(
            name=name,
            tts_voice=self._voice_var.get(),
            avatar_path=f"assets/avatars/{avatar}",
            avatar_style=stem,
            personality=self._personality_var.get(),
            mic_device_index=self._mic_device_idx.get(),
        )
        profile.save()
        self.destroy()
        self._on_complete(profile)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _enumerate_mics() -> list[tuple[int, str]]:
        """Return list of (device_index, name) for real input devices only."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            seen_names: set[str] = set()
            result = []
            skip_keywords = ("steam", "pc speaker", "stereo mix", "mapper", "primary sound")
            for i, d in enumerate(devices):
                if d["max_input_channels"] < 1:
                    continue
                name = d["name"]
                # Deduplicate near-identical names (WASAPI exposes devices multiple times)
                short = name[:28].lower()
                if short in seen_names:
                    continue
                if any(k in name.lower() for k in skip_keywords):
                    continue
                seen_names.add(short)
                result.append((i, name))
            return result
        except Exception:
            return []
