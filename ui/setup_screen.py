# First-run bot creation wizard: Welcome → Name & Personality → Avatar → Voice → Microphone.

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from bot.bot_profile import BotProfile, AVATAR_CATALOG, PERSONALITY_PRESETS
from speech.tts import TextToSpeech, TTSParams, list_voices
from config import Config

_BG           = "#0F0D1A"
_SURFACE      = "#1A1730"
_SURFACE2     = "#231F3A"
_ACCENT       = "#7C4DFF"
_ACCENT_HOVER = "#9D71FF"
_BORDER       = "#3A3060"
_TEXT_PRIMARY = "#F0EEFF"
_TEXT_DIM     = "#8B80B0"
_GREEN        = "#4CAF82"

_ASSETS = Path(__file__).parent.parent / "assets"
_PREVIEW_PHRASE = "Hello! I'm here to listen and support you."

# Steps: 0=Welcome, 1=Name+Personality, 2=Avatar, 3=Voice, 4=Microphone
_STEPS = [
    ("",              ""),
    ("Name Your Companion",  "Give your AI companion a name and personality."),
    ("Choose an Avatar",     "Select the face that feels right."),
    ("Pick a Voice",         "Choose how your companion will speak to you."),
    ("Select Microphone",    "Choose the microphone you speak into."),
]
_WIZARD_STEPS = 4  # steps 1–4 show dots


class SetupScreen(ctk.CTkToplevel):

    def __init__(self, parent: tk.Misc, config: Config, on_complete: Callable[[BotProfile], None]) -> None:
        super().__init__(parent)
        self._config      = config
        self._on_complete = on_complete
        self._step        = 0

        self._name_var        = tk.StringVar(value="PHILIA")
        self._personality_var = tk.StringVar(value="empathetic")
        self._avatar_var      = tk.StringVar(value="luna.png")
        self._voice_var       = tk.StringVar(value="en-US-JennyNeural")
        self._mic_device_idx  = tk.IntVar(value=-1)

        self._mic_options: list[tuple[int, str]] = self._enumerate_mics()
        self._avatar_images: dict[str, ImageTk.PhotoImage] = {}
        self._tts = TextToSpeech(config, BotProfile())

        self._setup_window()
        self._build_shell()
        self._show_step(0)

    # ── Window ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title("PHILIA — Create Your Companion")
        self.configure(fg_color=_BG)
        w, h = 800, 680
        self.geometry(f"{w}x{h}")
        self.minsize(640, 540)
        self.resizable(True, True)
        self.grab_set()
        self.focus_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    # ── Shell ─────────────────────────────────────────────────────────────────

    def _build_shell(self) -> None:
        # Header (hidden on welcome step)
        self._header = tk.Frame(self, bg=_BG)
        self._header.pack(fill="x", padx=48, pady=(28, 0))

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

        # Step dots (1–4 only)
        dots_frame = tk.Frame(self._header, bg=_BG)
        dots_frame.pack(anchor="w", pady=(10, 0))
        self._dot_labels: list[tk.Label] = []
        for _ in range(_WIZARD_STEPS):
            d = tk.Label(dots_frame, text="●", font=("Segoe UI", 10), bg=_BG, fg=_TEXT_DIM)
            d.pack(side="left", padx=3)
            self._dot_labels.append(d)

        # Divider
        self._divider = tk.Frame(self, bg=_BORDER, height=1)
        self._divider.pack(fill="x", padx=48, pady=(14, 0))

        # Body — expands to fill all available space between header and footer
        self._body = tk.Frame(self, bg=_BG)
        self._body.pack(fill="both", expand=True, padx=48, pady=20)

        # Footer
        self._footer_divider = tk.Frame(self, bg=_BORDER, height=1)
        self._footer_divider.pack(fill="x", padx=48)

        footer = tk.Frame(self, bg=_BG)
        footer.pack(fill="x", padx=48, pady=16)

        self._back_btn = ctk.CTkButton(
            footer, text="Back", width=110,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, command=self._go_back,
        )
        self._back_btn.pack(side="left")

        self._next_btn = ctk.CTkButton(
            footer, text="Next", width=140,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            font=("Segoe UI", 13, "bold"), command=self._go_next,
        )
        self._next_btn.pack(side="right")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_step(self, step: int) -> None:
        self._step = step
        for w in self._body.winfo_children():
            w.destroy()

        is_welcome = step == 0
        last_step  = len(_STEPS) - 1

        # Header / dots visibility
        if is_welcome:
            self._header.pack_forget()
            self._divider.pack_forget()
        else:
            self._header.pack(fill="x", padx=48, pady=(28, 0), before=self._divider)
            self._divider.pack(fill="x", padx=48, pady=(14, 0), before=self._body)
            self._title_lbl.config(text=_STEPS[step][0])
            self._subtitle_lbl.config(text=_STEPS[step][1])
            dot_idx = step - 1
            for i, d in enumerate(self._dot_labels):
                d.config(fg=_ACCENT if i == dot_idx else _TEXT_DIM)

        # Back button
        if step == 0:
            self._back_btn.configure(state="disabled", fg_color=_SURFACE, text_color=_BORDER)
        else:
            self._back_btn.configure(state="normal", fg_color=_SURFACE2, text_color=_TEXT_DIM)

        # Next button label
        if step == last_step:
            self._next_btn.configure(text="Start Chatting")
        elif step == 0:
            self._next_btn.configure(text="Get Started")
        else:
            self._next_btn.configure(text="Next")

        _builders = [
            self._build_welcome,
            self._build_step_name,
            self._build_step_avatar,
            self._build_step_voice,
            self._build_step_mic,
        ]
        _builders[step]()

    def _go_next(self) -> None:
        if self._step < len(_STEPS) - 1:
            self._show_step(self._step + 1)
        else:
            self._finish()

    def _go_back(self) -> None:
        if self._step > 0:
            self._show_step(self._step - 1)

    # ── Step 0: Welcome ───────────────────────────────────────────────────────

    def _build_welcome(self) -> None:
        body = self._body
        body.columnconfigure(0, weight=1)

        # Centre content vertically
        spacer = tk.Frame(body, bg=_BG)
        spacer.pack(expand=True, fill="both")

        inner = tk.Frame(body, bg=_BG)
        inner.pack()

        tk.Label(
            inner, text="PHILIA",
            font=("Segoe UI", 52, "bold"),
            fg=_ACCENT, bg=_BG,
        ).pack()

        tk.Label(
            inner, text="Emotionally Aware Conversational Companion",
            font=("Segoe UI", 14),
            fg=_TEXT_DIM, bg=_BG,
        ).pack(pady=(4, 0))

        tk.Frame(inner, bg=_BORDER, height=1).pack(fill="x", pady=28)

        tk.Label(
            inner,
            text=(
                "PHILIA listens to your voice, reads your expressions,\n"
                "and responds with genuine emotional awareness.\n\n"
                "This wizard will help you set up your companion.\n"
                "It only takes a minute."
            ),
            font=("Segoe UI", 12),
            fg=_TEXT_PRIMARY, bg=_BG,
            justify="center",
        ).pack()

        tk.Frame(body, bg=_BG).pack(expand=True, fill="both")

    # ── Step 1: Name & Personality ────────────────────────────────────────────

    def _build_step_name(self) -> None:
        body = self._body

        tk.Label(body, text="Companion Name", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(8, 4))

        ctk.CTkEntry(
            body, textvariable=self._name_var, width=340, height=42,
            fg_color=_SURFACE, border_color=_BORDER,
            text_color=_TEXT_PRIMARY, font=("Segoe UI", 13),
            placeholder_text="e.g.  PHILIA, Nova, Kai...",
        ).pack(anchor="w")

        tk.Label(body, text="Personality", font=("Segoe UI", 12, "bold"),
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(24, 8))

        grid = tk.Frame(body, bg=_BG)
        grid.pack(anchor="w")

        for i, preset in enumerate(PERSONALITY_PRESETS):
            col = i % 2
            row = i // 2
            card = tk.Frame(grid, bg=_SURFACE, cursor="hand2", width=330, height=76)
            card.grid(row=row, column=col, padx=(0, 12) if col == 0 else 0, pady=6)
            card.pack_propagate(False)

            key   = preset["key"]
            label = preset["label"]
            desc  = preset["desc"]

            def _select(k=key):
                self._personality_var.set(k)
                self._refresh_personality_cards(grid)

            card.bind("<Button-1>", lambda e, k=key: _select(k))
            name_lbl = tk.Label(card, text=label, font=("Segoe UI", 12, "bold"),
                                fg=_TEXT_PRIMARY, bg=_SURFACE, cursor="hand2")
            name_lbl.pack(anchor="w", padx=14, pady=(12, 2))
            desc_lbl = tk.Label(card, text=desc, font=("Segoe UI", 10),
                                fg=_TEXT_DIM, bg=_SURFACE, cursor="hand2")
            desc_lbl.pack(anchor="w", padx=14)
            for child in (name_lbl, desc_lbl):
                child.bind("<Button-1>", lambda e, k=key: _select(k))

        self._refresh_personality_cards(grid)

    def _refresh_personality_cards(self, grid: tk.Frame) -> None:
        selected = self._personality_var.get()
        for i, preset in enumerate(PERSONALITY_PRESETS):
            cards = grid.grid_slaves(row=i // 2, column=i % 2)
            if not cards:
                continue
            card = cards[0]
            is_sel = preset["key"] == selected
            bg = _ACCENT if is_sel else _SURFACE
            card.config(bg=bg)
            for child in card.winfo_children():
                child.config(bg=bg)

    # ── Step 2: Avatar ────────────────────────────────────────────────────────

    def _build_step_avatar(self) -> None:
        body = self._body
        avatar_dir = _ASSETS / "avatars"

        outer = tk.Frame(body, bg=_BG)
        outer.pack(fill="both", expand=True)

        grid = tk.Frame(outer, bg=_BG)
        grid.pack(anchor="center", expand=True)

        for i, av in enumerate(AVATAR_CATALOG):
            col = i % 3
            row = i // 3
            filename = av["file"]

            card = tk.Frame(grid, bg=_SURFACE, cursor="hand2", width=190, height=210)
            card.grid(row=row, column=col, padx=8, pady=8)
            card.pack_propagate(False)

            img_path = avatar_dir / filename
            if img_path.exists():
                img = Image.open(img_path).resize((96, 96), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._avatar_images[filename] = photo
                img_lbl = tk.Label(card, image=photo, bg=_SURFACE, cursor="hand2")
            else:
                img_lbl = tk.Label(card, text="◈", font=("Segoe UI", 36), fg=_ACCENT, bg=_SURFACE)
            img_lbl.pack(pady=(14, 4))

            name_lbl = tk.Label(card, text=av["name"], font=("Segoe UI", 12, "bold"),
                                fg=_TEXT_PRIMARY, bg=_SURFACE, cursor="hand2")
            name_lbl.pack()
            desc_lbl = tk.Label(card, text=av["desc"], font=("Segoe UI", 9),
                                fg=_TEXT_DIM, bg=_SURFACE, cursor="hand2")
            desc_lbl.pack()

            def _select(f=filename):
                self._avatar_var.set(f)
                self._refresh_avatar_cards(grid)

            for w in (card, img_lbl, name_lbl, desc_lbl):
                w.bind("<Button-1>", lambda e, f=filename: _select(f))

        self._refresh_avatar_cards(grid)

    def _refresh_avatar_cards(self, grid: tk.Frame) -> None:
        selected = self._avatar_var.get()
        for i, av in enumerate(AVATAR_CATALOG):
            cards = grid.grid_slaves(row=i // 3, column=i % 3)
            if not cards:
                continue
            card = cards[0]
            is_sel = av["file"] == selected
            bg = _ACCENT if is_sel else _SURFACE
            card.config(bg=bg)
            for child in card.winfo_children():
                child.config(bg=bg)

    # ── Step 3: Voice ─────────────────────────────────────────────────────────

    def _build_step_voice(self) -> None:
        body = self._body
        voices = list_voices()

        scroll = ctk.CTkScrollableFrame(body, fg_color=_SURFACE)
        scroll.pack(fill="both", expand=True)

        for v in voices:
            row = tk.Frame(scroll, bg=_SURFACE, cursor="hand2")
            row.pack(fill="x", padx=4, pady=2)

            ctk.CTkRadioButton(
                row, text=v["label"],
                variable=self._voice_var, value=v["id"],
                fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                text_color=_TEXT_PRIMARY, font=("Segoe UI", 12),
            ).pack(side="left", pady=8, padx=12)

            ctk.CTkButton(
                row, text="Preview", width=84, height=28,
                fg_color=_SURFACE2, hover_color=_BORDER,
                text_color=_TEXT_DIM, font=("Segoe UI", 10),
                command=lambda vid=v["id"]: self._preview_voice(vid),
            ).pack(side="right", padx=12)

        self._preview_status = tk.Label(body, text="", font=("Segoe UI", 10), fg=_GREEN, bg=_BG)
        self._preview_status.pack(anchor="w", pady=(8, 0))

    def _preview_voice(self, voice_id: str) -> None:
        label = voice_id.split("-")[1] if "-" in voice_id else voice_id
        self._preview_status.config(text=f"Playing preview for {label}...")
        self._tts.preview(_PREVIEW_PHRASE, voice_id)
        self.after(3500, lambda: self._preview_status.config(text=""))

    # ── Step 4: Microphone ────────────────────────────────────────────────────

    def _build_step_mic(self) -> None:
        body = self._body

        tk.Label(
            body,
            text="Using the right microphone significantly improves transcription accuracy.",
            font=("Segoe UI", 11), fg=_TEXT_DIM, bg=_BG, wraplength=680, justify="left",
        ).pack(anchor="w", pady=(0, 12))

        scroll = ctk.CTkScrollableFrame(body, fg_color=_SURFACE)
        scroll.pack(fill="both", expand=True)

        ctk.CTkRadioButton(
            scroll, text="System Default (let Windows decide)",
            variable=self._mic_device_idx, value=-1,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            text_color=_TEXT_PRIMARY, font=("Segoe UI", 11),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        for idx, name in self._mic_options:
            lower = name.lower()
            is_rec = any(k in lower for k in ("headset", "headphone", "usb", "jabra", "sony", "wh-", "bose", "airpods"))
            label = f"{name}   (Recommended)" if is_rec else name
            color = _GREEN if is_rec else _TEXT_PRIMARY
            ctk.CTkRadioButton(
                scroll, text=label,
                variable=self._mic_device_idx, value=idx,
                fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                text_color=color, font=("Segoe UI", 11),
            ).pack(anchor="w", padx=12, pady=3)

    # ── Finish ────────────────────────────────────────────────────────────────

    def _finish(self) -> None:
        name  = self._name_var.get().strip() or "PHILIA"
        avatar = self._avatar_var.get()
        profile = BotProfile(
            name=name,
            tts_voice=self._voice_var.get(),
            avatar_path=f"assets/avatars/{avatar}",
            avatar_style=Path(avatar).stem,
            personality=self._personality_var.get(),
            mic_device_index=self._mic_device_idx.get(),
        )
        profile.save()
        self.destroy()
        self._on_complete(profile)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _enumerate_mics() -> list[tuple[int, str]]:
        try:
            import sounddevice as sd
            seen: set[str] = set()
            result = []
            skip = ("steam", "pc speaker", "stereo mix", "mapper", "primary sound")
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] < 1:
                    continue
                name = d["name"]
                short = name[:28].lower()
                if short in seen or any(k in name.lower() for k in skip):
                    continue
                seen.add(short)
                result.append((i, name))
            return result
        except Exception:
            return []
