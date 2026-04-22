"""
chat_window.py — Main PHILIA chat interface.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  Left Panel (300px)         │  Right Panel (fills)       │
  │  ┌──────────────────────┐   │  ┌──────────────────────┐  │
  │  │  Avatar (animated)   │   │  │  Chat history        │  │
  │  │  Bot name            │   │  │  (scrollable bubbles)│  │
  │  │  Emotion HUD badge   │   │  └──────────────────────┘  │
  │  │  Pipeline status     │   │  ┌──────────────────────┐  │
  │  └──────────────────────┘   │  │  Mic button + status │  │
  │                             │  └──────────────────────┘  │
  └──────────────────────────────────────────────────────────┘

Key features:
  - Smooth idle-pulse animation on avatar
  - Animated speaking glow when bot is talking
  - Emotion badge updates each turn
  - Chat bubbles with timestamps
  - Pipeline stage indicator (Listening → Transcribing → Thinking → Speaking)
  - Settings button → inline panel (voice, model, re-run wizard)
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk, ImageFilter

from bot.bot_profile import BotProfile

# ── Theme ──────────────────────────────────────────────────────────────────────

_BG           = "#0F0D1A"
_SURFACE      = "#1A1730"
_SURFACE2     = "#231F3A"
_ACCENT       = "#7C4DFF"
_ACCENT_HOVER = "#9D71FF"
_ACCENT_DIM   = "#3D2880"
_BORDER       = "#3A3060"
_TEXT_PRIMARY = "#F0EEFF"
_TEXT_DIM     = "#8B80B0"
_GREEN        = "#4CAF82"
_RED          = "#FF5370"
_YELLOW       = "#FFD166"
_BUBBLE_USER  = "#2A2450"
_BUBBLE_BOT   = "#1E1B3A"

# Emotion → colour mapping for the HUD badge
_EMOTION_COLORS: dict[str, str] = {
    "happy":   "#FFD166",
    "sad":     "#74B9FF",
    "angry":   "#FF5370",
    "fear":    "#A29BFE",
    "surprise":"#55EFC4",
    "disgust": "#B2BEC3",
    "neutral": "#8B80B0",
}

_EMOTION_ICONS: dict[str, str] = {
    "happy":   "😊",
    "sad":     "😢",
    "angry":   "😠",
    "fear":    "😨",
    "surprise":"😲",
    "disgust": "😒",
    "neutral": "😐",
}

_ASSETS = Path(__file__).parent.parent / "assets"


class ChatWindow(ctk.CTk):
    """
    Main application window.

    Args:
        profile:     Loaded BotProfile.
        on_ptt:      Callback() invoked when the user triggers push-to-talk.
        on_settings: Callback() for the settings gear button.
    """

    def __init__(
        self,
        profile: BotProfile,
        on_ptt: Callable,
        on_settings: Callable | None = None,
        ptt_press_event=None,
        ptt_release_event=None,
    ) -> None:
        super().__init__()
        self._profile          = profile
        self._on_ptt           = on_ptt
        self._on_settings      = on_settings
        self._ptt_press_event  = ptt_press_event
        self._ptt_release_event= ptt_release_event

        # Animation state
        self._ptt_active       = False  # synchronous guard against key-repeat
        self._pulse_step    = 0
        self._pulse_growing = True
        self._is_speaking   = False
        self._is_listening  = False
        self._is_processing = False

        # Conversation history: list of (role, text, timestamp)
        self._messages: list[tuple[str, str, str]] = []

        self._setup_window()
        self._build_ui()
        self._bind_keys()
        self._animate_avatar()

    # ── Window setup ────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title(f"PHILIA — {self._profile.name}")
        self.configure(fg_color=_BG)
        self.geometry("1100x700")
        self.minsize(900, 600)

        icon_path = _ASSETS / "icon.png"
        if icon_path.exists():
            try:
                icon = tk.PhotoImage(file=str(icon_path))
                self.iconphoto(True, icon)
            except Exception:
                pass

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Root paned layout
        root_frame = tk.Frame(self, bg=_BG)
        root_frame.pack(fill="both", expand=True)

        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(root_frame, bg=_SURFACE, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_left_panel(left)

        # ── Divider ────────────────────────────────────────────────────────────
        tk.Frame(root_frame, bg=_BORDER, width=1).pack(side="left", fill="y")

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(root_frame, bg=_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

    # ── Left panel ─────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent: tk.Frame) -> None:
        # ── Avatar canvas ─────────────────────────────────────────────────────
        self._avatar_canvas = tk.Canvas(
            parent, width=230, height=230, bg=_SURFACE, highlightthickness=0,
        )
        self._avatar_canvas.pack(pady=(30, 10))

        # Glow rings (drawn behind avatar)
        self._ring1 = self._avatar_canvas.create_oval(5, 5, 225, 225, outline=_ACCENT_DIM, width=2)
        self._ring2 = self._avatar_canvas.create_oval(18, 18, 212, 212, outline=_ACCENT, width=1)

        # Load and display avatar image
        av_path = _ASSETS.parent / self._profile.avatar_path
        if not av_path.exists():
            av_path = _ASSETS / "avatars" / "luna.png"
        self._load_avatar(av_path)

        # Bot name
        tk.Label(
            parent, text=self._profile.name,
            font=("Segoe UI", 18, "bold"), fg=_TEXT_PRIMARY, bg=_SURFACE,
        ).pack(pady=(4, 0))

        # Personality chip
        tk.Label(
            parent, text=self._profile.personality.capitalize(),
            font=("Segoe UI", 10), fg=_TEXT_DIM, bg=_SURFACE,
        ).pack()

        # Divider
        tk.Frame(parent, bg=_BORDER, height=1).pack(fill="x", padx=24, pady=16)

        # ── Emotion HUD ───────────────────────────────────────────────────────
        hud = tk.Frame(parent, bg=_SURFACE)
        hud.pack(fill="x", padx=24)

        tk.Label(hud, text="Detected Emotion", font=("Segoe UI", 9),
                 fg=_TEXT_DIM, bg=_SURFACE).pack(anchor="w")

        badge_row = tk.Frame(hud, bg=_SURFACE)
        badge_row.pack(anchor="w", pady=(4, 0))

        self._emotion_icon_lbl = tk.Label(
            badge_row, text="😐", font=("Segoe UI", 22), bg=_SURFACE,
        )
        self._emotion_icon_lbl.pack(side="left", padx=(0, 8))

        emo_text_col = tk.Frame(badge_row, bg=_SURFACE)
        emo_text_col.pack(side="left")

        self._emotion_lbl = tk.Label(
            emo_text_col, text="—", font=("Segoe UI", 13, "bold"),
            fg=_TEXT_PRIMARY, bg=_SURFACE,
        )
        self._emotion_lbl.pack(anchor="w")
        self._confidence_lbl = tk.Label(
            emo_text_col, text="Waiting for input",
            font=("Segoe UI", 9), fg=_TEXT_DIM, bg=_SURFACE,
        )
        self._confidence_lbl.pack(anchor="w")

        # Emotion bar
        self._emo_bar = ctk.CTkProgressBar(
            hud, width=240, height=5,
            fg_color=_SURFACE2, progress_color=_ACCENT,
        )
        self._emo_bar.pack(anchor="w", pady=(8, 0))
        self._emo_bar.set(0)

        # Modality breakdown
        tk.Label(hud, text="Modality Signals", font=("Segoe UI", 9),
                 fg=_TEXT_DIM, bg=_SURFACE).pack(anchor="w", pady=(14, 4))

        self._modality_frame = tk.Frame(hud, bg=_SURFACE)
        self._modality_frame.pack(anchor="w")
        self._modality_labels: dict[str, tk.Label] = {}
        for mod in ["🎤 Audio", "📹 Facial", "💬 Text"]:
            row = tk.Frame(self._modality_frame, bg=_SURFACE)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=mod, font=("Segoe UI", 9), fg=_TEXT_DIM,
                     bg=_SURFACE, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="—", font=("Segoe UI", 9, "bold"),
                           fg=_TEXT_PRIMARY, bg=_SURFACE)
            lbl.pack(side="left")
            self._modality_labels[mod] = lbl

        # Divider
        tk.Frame(parent, bg=_BORDER, height=1).pack(fill="x", padx=24, pady=16)

        # ── Pipeline status ───────────────────────────────────────────────────
        self._pipeline_lbl = tk.Label(
            parent, text="Ready", font=("Segoe UI", 10),
            fg=_GREEN, bg=_SURFACE,
        )
        self._pipeline_lbl.pack()

        # Settings button
        ctk.CTkButton(
            parent, text="⚙ Settings", width=120, height=28,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, font=("Segoe UI", 10),
            command=self._on_settings if self._on_settings else lambda: None,
        ).pack(pady=(18, 0))

    def _load_avatar(self, path: Path) -> None:
        """Load and display the avatar image on the canvas."""
        try:
            raw = Image.open(path).convert("RGBA")
            # Circular crop
            size = 180
            raw = raw.resize((size, size), Image.LANCZOS)
            mask = Image.new("L", (size, size), 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size, size), fill=255)
            raw.putalpha(mask)
            self._avatar_photo = ImageTk.PhotoImage(raw)
            offset = (230 - size) // 2
            self._avatar_img_id = self._avatar_canvas.create_image(
                offset, offset, anchor="nw", image=self._avatar_photo,
            )
        except Exception:
            self._avatar_canvas.create_text(115, 115, text="◈", font=("Segoe UI", 52), fill=_ACCENT)

    # ── Right panel ─────────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: tk.Frame) -> None:
        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = tk.Frame(parent, bg=_BG)
        topbar.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(
            topbar, text=f"Chat with {self._profile.name}",
            font=("Segoe UI", 14, "bold"), fg=_TEXT_PRIMARY, bg=_BG,
        ).pack(side="left")

        ctk.CTkButton(
            topbar, text="Clear", width=70, height=28,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, font=("Segoe UI", 10),
            command=self._clear_chat,
        ).pack(side="right")

        # ── Chat history ──────────────────────────────────────────────────────
        self._chat_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color=_SURFACE,
            scrollbar_button_color=_BORDER,
            scrollbar_button_hover_color=_ACCENT,
        )
        self._chat_frame.pack(fill="both", expand=True, padx=20, pady=(12, 0))

        # Placeholder text
        self._placeholder = tk.Label(
            self._chat_frame,
            text=f"Hold  SPACE  to start talking with {self._profile.name}",
            font=("Segoe UI", 13), fg=_TEXT_DIM, bg=_SURFACE,
        )
        self._placeholder.pack(expand=True, pady=60)

        # ── Bottom input bar ──────────────────────────────────────────────────
        bottom = tk.Frame(parent, bg=_BG)
        bottom.pack(fill="x", padx=20, pady=16)

        # Mic button
        self._mic_btn = ctk.CTkButton(
            bottom,
            text="🎤  Hold SPACE to Speak",
            width=340, height=52,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            font=("Segoe UI", 14, "bold"),
            corner_radius=26,
            command=lambda: self._on_space_press(None),
        )
        self._mic_btn.pack(side="left", expand=True)

        # Stage label right of button
        self._stage_lbl = tk.Label(
            bottom, text="",
            font=("Segoe UI", 10), fg=_TEXT_DIM, bg=_BG,
        )
        self._stage_lbl.pack(side="left", padx=(14, 0))

    # ── Avatar animation ────────────────────────────────────────────────────────

    def _animate_avatar(self) -> None:
        """Idle pulse + speaking glow animation on avatar canvas."""
        try:
            if self._is_speaking:
                # Rapid glow — alternate between two shades
                t = time.monotonic()
                phase = (t % 0.6) / 0.6   # 0 → 1 over 600ms
                r1_w = 3 + int(phase * 4)
                r2_w = 2
                col1  = _ACCENT
                col2  = _ACCENT_HOVER
            else:
                # Gentle idle pulse
                self._pulse_step += 1 if self._pulse_growing else -1
                if self._pulse_step >= 30:
                    self._pulse_growing = False
                elif self._pulse_step <= 0:
                    self._pulse_growing = True

                pulse = self._pulse_step / 30.0  # 0.0 → 1.0
                r1_w = 1 + int(pulse * 2)
                r2_w = 1
                # Blend colour between dim and bright
                col1  = _ACCENT_DIM if pulse < 0.5 else _ACCENT
                col2  = _BORDER

            self._avatar_canvas.itemconfig(self._ring1, outline=col1, width=r1_w)
            self._avatar_canvas.itemconfig(self._ring2, outline=col2, width=r2_w)
        except Exception:
            pass

        self.after(50, self._animate_avatar)

    # ── Key bindings ────────────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        self._release_job = None
        self.bind("<space>", self._on_space_press)
        self.bind("<KeyRelease-space>", self._on_space_release)

    def _on_space_press(self, event) -> None:
        if self._release_job is not None:
            self.after_cancel(self._release_job)
            self._release_job = None

        if self._is_listening or self._is_processing or self._ptt_active:
            return
        self._ptt_active = True
        if self._ptt_release_event:
            self._ptt_release_event.clear()
        threading.Thread(target=self._on_ptt, daemon=True).start()

    def _on_space_release(self, event) -> None:
        if self._release_job is not None:
            self.after_cancel(self._release_job)
        self._release_job = self.after(50, self._do_release)

    def _do_release(self) -> None:
        self._release_job = None
        self._ptt_active = False
        if self._ptt_release_event:
            self._ptt_release_event.set()

    # ── Public update API (called from pipeline thread via after()) ─────────────

    def set_stage(self, stage: str) -> None:
        """Update the pipeline stage label. Thread-safe."""
        stage_icons = {
            "Listening":     "🔴 Listening…",
            "Transcribing":  "📝 Transcribing…",
            "Analysing":     "🧠 Analysing emotion…",
            "Thinking":      "💭 Thinking…",
            "Speaking":      "🔊 Speaking…",
            "Ready":         "✓  Ready",
            "Error":         "⚠  Error — try again",
        }
        label = stage_icons.get(stage, stage)

        def _update():
            self._stage_lbl.config(text=label)
            self._pipeline_lbl.config(text=stage)
            self._is_listening   = stage == "Listening"
            self._is_processing  = stage in ("Transcribing", "Analysing", "Thinking")
            self._is_speaking    = stage == "Speaking"

            if stage == "Listening":
                self._mic_btn.configure(
                    fg_color=_RED, text="🔴  Recording…", state="disabled",
                )
            elif stage == "Speaking":
                self._mic_btn.configure(
                    fg_color=_GREEN, text="🔊  Playing response…", state="disabled",
                )
            elif stage in ("Transcribing", "Analysing", "Thinking"):
                self._mic_btn.configure(
                    fg_color=_ACCENT_DIM, text="⏳  Processing…", state="disabled",
                )
            else:
                self._mic_btn.configure(
                    fg_color=_ACCENT, text="🎤  Hold SPACE to Speak", state="normal",
                )

        self.after(0, _update)

    def update_emotion(
        self,
        label: str,
        confidence: float,
        audio_label: str = "—",
        facial_label: str = "—",
        text_label: str = "—",
    ) -> None:
        """Update the emotion HUD. Thread-safe."""
        def _update():
            color = _EMOTION_COLORS.get(label, _TEXT_DIM)
            icon  = _EMOTION_ICONS.get(label, "🤔")
            self._emotion_icon_lbl.config(text=icon)
            self._emotion_lbl.config(text=label.capitalize(), fg=color)
            self._confidence_lbl.config(text=f"{confidence:.0%} confidence")
            self._emo_bar.set(confidence)
            self._emo_bar.configure(progress_color=color)

            self._modality_labels.get("🎤 Audio", tk.Label()).config(text=audio_label)
            self._modality_labels.get("📹 Facial", tk.Label()).config(text=facial_label)
            self._modality_labels.get("💬 Text", tk.Label()).config(text=text_label)

        self.after(0, _update)

    def add_message(self, role: str, text: str) -> None:
        """
        Add a chat bubble to the history. Thread-safe.

        Args:
            role: "user" or "bot"
            text: Message text
        """
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M")
        self._messages.append((role, text, ts))

        def _add():
            # Remove placeholder on first message
            if self._placeholder and self._placeholder.winfo_exists():
                self._placeholder.destroy()

            is_user = role == "user"
            bubble_bg   = _BUBBLE_USER if is_user else _BUBBLE_BOT
            anchor      = "e" if is_user else "w"
            side        = "right" if is_user else "left"
            name_text   = "You" if is_user else self._profile.name
            name_color  = _ACCENT_HOVER if is_user else _GREEN

            outer = tk.Frame(self._chat_frame, bg=_SURFACE)
            outer.pack(fill="x", pady=4, padx=8)

            bubble = tk.Frame(outer, bg=bubble_bg)
            bubble.pack(anchor=anchor, padx=6)

            # Name + timestamp row
            meta = tk.Frame(bubble, bg=bubble_bg)
            meta.pack(fill="x", padx=12, pady=(8, 0))
            tk.Label(meta, text=name_text, font=("Segoe UI", 9, "bold"),
                     fg=name_color, bg=bubble_bg).pack(side="left")
            tk.Label(meta, text=ts, font=("Segoe UI", 8),
                     fg=_TEXT_DIM, bg=bubble_bg).pack(side="right")

            # Message text
            tk.Label(
                bubble, text=text,
                font=("Segoe UI", 11), fg=_TEXT_PRIMARY, bg=bubble_bg,
                wraplength=540, justify="left", anchor="w",
                padx=12, pady=8,
            ).pack(anchor="w")

            # Scroll to bottom
            self._chat_frame._parent_canvas.yview_moveto(1.0)

        self.after(0, _add)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _clear_chat(self) -> None:
        for w in self._chat_frame.winfo_children():
            w.destroy()
        self._messages.clear()
        self._placeholder = tk.Label(
            self._chat_frame,
            text=f"Hold  SPACE  to start talking with {self._profile.name}",
            font=("Segoe UI", 13), fg=_TEXT_DIM, bg=_SURFACE,
        )
        self._placeholder.pack(expand=True, pady=60)
