# Profile creation / edit wizard: Name+Personality → Avatar → Voice (3 steps).

from __future__ import annotations

import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from bot.bot_profile import BotProfile, PRESET_AVATARS, PERSONALITY_PRESETS
from speech.tts import TextToSpeech, list_voices
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

_ASSETS      = Path(__file__).parent.parent / "assets"
_AVATARS_DIR = _ASSETS / "avatars"
_PREVIEW_PHRASE = "Hello! I'm here to listen and support you."

# Steps 1-3 (no welcome, no mic)
_STEPS = [
    ("Name Your Companion",  "Give your companion a name and choose a personality."),
    ("Choose an Avatar",     "Pick an image or upload your own."),
    ("Pick a Voice",         "Choose how your companion will speak to you."),
]
_WIZARD_STEPS = len(_STEPS)


class SetupScreen(ctk.CTkToplevel):

    def __init__(
        self,
        parent: tk.Misc,
        config: Config,
        on_complete: Callable[[BotProfile], None],
        existing: BotProfile | None = None,
    ) -> None:
        super().__init__(parent)
        self._config      = config
        self._on_complete = on_complete
        self._existing    = existing          # None = new, BotProfile = edit mode
        self._step        = 0

        # State
        initial_name        = existing.name        if existing else ""
        initial_personality = existing.personality if existing else "empathetic"
        initial_avatar      = Path(existing.avatar_path).name if existing else "luna.png"
        initial_voice       = existing.tts_voice   if existing else "en-US-JennyNeural"

        self._name_var        = tk.StringVar(value=initial_name)
        self._personality_var = tk.StringVar(value=initial_personality)
        self._avatar_var      = tk.StringVar(value=initial_avatar)  # filename only
        self._uploaded_src    = [None]   # [str | None] — source path of uploaded image
        self._voice_var       = tk.StringVar(value=initial_voice)
        self._avatar_images:  dict[str, ImageTk.PhotoImage] = {}
        self._tts = TextToSpeech(config, BotProfile())

        self._setup_window()
        self._build_shell()
        self._show_step(0)

    # ── Window ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        mode = "Edit Companion" if self._existing else "Create Companion"
        self.title(f"PHILIA — {mode}")
        self.configure(fg_color=_BG)
        w, h = 800, 640
        self.geometry(f"{w}x{h}")
        self.minsize(640, 520)
        self.resizable(True, True)
        self.grab_set()
        self.focus_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    # ── Shell ──────────────────────────────────────────────────────────────────

    def _build_shell(self) -> None:
        # Header
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

        # Step dots
        dots_frame = tk.Frame(self._header, bg=_BG)
        dots_frame.pack(anchor="w", pady=(10, 0))
        self._dot_labels: list[tk.Label] = []
        for _ in range(_WIZARD_STEPS):
            d = tk.Label(dots_frame, text="●", font=("Segoe UI", 10), bg=_BG, fg=_TEXT_DIM)
            d.pack(side="left", padx=3)
            self._dot_labels.append(d)

        tk.Frame(self, bg=_BORDER, height=1).pack(fill="x", padx=48, pady=(14, 0))

        # Body
        self._body = tk.Frame(self, bg=_BG)
        self._body.pack(fill="both", expand=True, padx=48, pady=20)

        # Footer
        tk.Frame(self, bg=_BORDER, height=1).pack(fill="x", padx=48)
        footer = tk.Frame(self, bg=_BG)
        footer.pack(fill="x", padx=48, pady=16)

        self._back_btn = ctk.CTkButton(
            footer, text="Back", width=110,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, command=self._go_back,
        )
        self._back_btn.pack(side="left")

        self._next_btn = ctk.CTkButton(
            footer, text="Next", width=150,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
            font=("Segoe UI", 13, "bold"), command=self._go_next,
        )
        self._next_btn.pack(side="right")

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _show_step(self, step: int) -> None:
        self._step = step
        for w in self._body.winfo_children():
            w.destroy()

        self._title_lbl.config(text=_STEPS[step][0])
        self._subtitle_lbl.config(text=_STEPS[step][1])

        for i, d in enumerate(self._dot_labels):
            d.config(fg=_ACCENT if i == step else _TEXT_DIM)

        if step == 0:
            self._back_btn.configure(state="disabled", fg_color=_SURFACE, text_color=_BORDER)
        else:
            self._back_btn.configure(state="normal", fg_color=_SURFACE2, text_color=_TEXT_DIM)

        label = "Save" if self._existing else "Start Chatting"
        self._next_btn.configure(text=label if step == _WIZARD_STEPS - 1 else "Next")

        [self._build_step_name, self._build_step_avatar, self._build_step_voice][step]()

    def _go_next(self) -> None:
        if self._step == 0:
            name = self._name_var.get().strip()
            if not name:
                messagebox.showwarning("Name required", "Please enter a name for your companion.", parent=self)
                return
            # Check uniqueness (skip if editing same name)
            existing_names = {p.name for p in BotProfile.list_all()}
            if name in existing_names and (self._existing is None or name != self._existing.name):
                messagebox.showwarning("Name taken", f"A companion named '{name}' already exists. Choose a different name.", parent=self)
                return
        if self._step < _WIZARD_STEPS - 1:
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
                 fg=_TEXT_PRIMARY, bg=_BG).pack(anchor="w", pady=(8, 4))

        ctk.CTkEntry(
            body, textvariable=self._name_var, width=340, height=42,
            fg_color=_SURFACE, border_color=_BORDER,
            text_color=_TEXT_PRIMARY, font=("Segoe UI", 13),
            placeholder_text="e.g. Luna, Justin, Nova...",
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

            key, label, desc = preset["key"], preset["label"], preset["desc"]

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
            bg = _ACCENT if preset["key"] == selected else _SURFACE
            card.config(bg=bg)
            for child in card.winfo_children():
                child.config(bg=bg)

    # ── Step 2: Avatar ─────────────────────────────────────────────────────────

    def _build_step_avatar(self) -> None:
        body = self._body

        # Upload preview area (shown when an image is uploaded)
        self._upload_preview_frame = tk.Frame(body, bg=_BG)
        self._upload_preview_frame.pack(anchor="w", pady=(0, 8))
        self._upload_preview_lbl: tk.Label | None = None
        if self._uploaded_src[0]:
            self._show_upload_preview(self._uploaded_src[0])

        # Preset grid — 6 images, 3 per row, no names
        grid = tk.Frame(body, bg=_BG)
        grid.pack(anchor="center")

        for i, filename in enumerate(PRESET_AVATARS):
            col = i % 3
            row = i // 3
            card = tk.Frame(grid, bg=_SURFACE, cursor="hand2", width=165, height=165)
            card.grid(row=row, column=col, padx=8, pady=8)
            card.pack_propagate(False)

            img_path = _AVATARS_DIR / filename
            if img_path.exists():
                img = Image.open(img_path).resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._avatar_images[filename] = photo
                lbl = tk.Label(card, image=photo, bg=_SURFACE, cursor="hand2")
            else:
                lbl = tk.Label(card, text="◈", font=("Segoe UI", 36), fg=_ACCENT, bg=_SURFACE)
            lbl.pack(expand=True)

            def _sel(f=filename):
                self._avatar_var.set(f)
                self._uploaded_src[0] = None       # clear any upload
                self._clear_upload_preview()
                self._refresh_avatar_cards(grid)

            for w in (card, lbl):
                w.bind("<Button-1>", lambda e, f=filename: _sel(f))

        self._refresh_avatar_cards(grid)

        # Upload button
        upload_row = tk.Frame(body, bg=_BG)
        upload_row.pack(anchor="w", pady=(12, 0))

        ctk.CTkButton(
            upload_row, text="⬆  Upload your own image", width=220, height=36,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_TEXT_DIM, font=("Segoe UI", 11),
            command=lambda: self._pick_upload(grid),
        ).pack(side="left")

        self._upload_status = tk.Label(upload_row, text="", font=("Segoe UI", 10),
                                       fg=_GREEN, bg=_BG)
        self._upload_status.pack(side="left", padx=10)

    def _pick_upload(self, grid: tk.Frame) -> None:
        path = filedialog.askopenfilename(
            title="Choose an avatar image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.bmp")],
            parent=self,
        )
        if not path:
            return
        self._uploaded_src[0] = path
        self._avatar_var.set("__uploaded__")
        self._show_upload_preview(path)
        self._refresh_avatar_cards(grid)   # deselect presets
        self._upload_status.config(text=f"✓  {Path(path).name}")

    def _show_upload_preview(self, src: str) -> None:
        self._clear_upload_preview()
        try:
            img = Image.open(src).resize((72, 72), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._avatar_images["__uploaded__"] = photo
            self._upload_preview_lbl = tk.Label(
                self._upload_preview_frame, image=photo,
                bg=_BG, relief="flat",
            )
            self._upload_preview_lbl.pack(side="left")
            tk.Label(
                self._upload_preview_frame,
                text="  Custom image selected", font=("Segoe UI", 10),
                fg=_GREEN, bg=_BG,
            ).pack(side="left")
        except Exception:
            pass

    def _clear_upload_preview(self) -> None:
        for w in self._upload_preview_frame.winfo_children():
            w.destroy()
        self._upload_preview_lbl = None

    def _refresh_avatar_cards(self, grid: tk.Frame) -> None:
        selected = self._avatar_var.get()
        for i, filename in enumerate(PRESET_AVATARS):
            cards = grid.grid_slaves(row=i // 3, column=i % 3)
            if not cards:
                continue
            card = cards[0]
            bg = _ACCENT if filename == selected else _SURFACE
            card.config(bg=bg)
            for child in card.winfo_children():
                child.config(bg=bg)

    # ── Step 3: Voice ──────────────────────────────────────────────────────────

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
        self._preview_status.config(text="Playing preview...")
        self._tts.preview(_PREVIEW_PHRASE, voice_id)
        self.after(3500, lambda: self._preview_status.config(text=""))

    # ── Finish ─────────────────────────────────────────────────────────────────

    def _finish(self) -> None:
        from bot.bot_profile import _AVATARS_DIR as _PROF_AVATARS
        name    = self._name_var.get().strip() or "PHILIA"
        src     = self._uploaded_src[0]
        avatar_filename = self._avatar_var.get()

        if src:
            # Copy uploaded image to assets/profiles/avatars/<name>.<ext>
            ext  = Path(src).suffix or ".png"
            dest = _PROF_AVATARS / f"{name}{ext}"
            _PROF_AVATARS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            avatar_path = f"assets/profiles/avatars/{name}{ext}"
        else:
            avatar_path = f"assets/avatars/{avatar_filename}"

        # If editing and the name changed, delete the old JSON
        if self._existing and self._existing.name != name:
            self._existing.delete()

        profile = BotProfile(
            name=name,
            tts_voice=self._voice_var.get(),
            avatar_path=avatar_path,
            personality=self._personality_var.get(),
        )
        profile.save()
        self.destroy()
        self._on_complete(profile)
