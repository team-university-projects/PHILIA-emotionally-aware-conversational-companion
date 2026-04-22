# UI orchestrator — welcome → loading → profile list / wizard → chat window.

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from bot.bot_profile import BotProfile
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

_BG      = "#0F0D1A"
_SURFACE = "#1A1730"
_SURFACE2= "#231F3A"
_ACCENT  = "#7C4DFF"
_AHOVER  = "#9D71FF"
_BORDER  = "#3A3060"
_FG      = "#F0EEFF"
_DIM     = "#8B80B0"
_RED     = "#E05252"
_GREEN   = "#4CAF82"


class Interface:

    def __init__(self, config: Config) -> None:
        self._config  = config
        self._profile: BotProfile | None = None
        self._chat:   "ChatWindow | None" = None
        self._on_ptt_callback: Callable | None = None
        self._ptt_press_event   = None
        self._ptt_release_event = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_ptt_callback(self, fn: Callable) -> None:
        self._on_ptt_callback = fn

    def set_ptt_events(self, press_event, release_event) -> None:
        self._ptt_press_event   = press_event
        self._ptt_release_event = release_event

    def set_stage(self, stage: str) -> None:
        if self._chat:
            self._chat.set_stage(stage)

    def update_emotion(self, label, confidence, audio_label="—", facial_label="—", text_label="—") -> None:
        if self._chat:
            self._chat.update_emotion(label, confidence, audio_label, facial_label, text_label)

    def add_message(self, role: str, text: str) -> None:
        if self._chat:
            self._chat.add_message(role, text)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run_loading(self, initialise_fn: Callable, on_profile_final: Callable | None = None) -> None:
        # 1. Welcome / profile-selection screen
        choice, selected_profile = self._run_welcome()

        # 2. Loading screen + model init (uses selected profile, or default for "new")
        from ui.loading_screen import LoadingScreen
        loading = LoadingScreen(on_ready=lambda: None, on_error=lambda msg: None)
        init_profile = selected_profile if selected_profile else BotProfile()

        def _run_init():
            try:
                profile = initialise_fn(
                    profile=init_profile,
                    update_status=loading.set_status,
                    set_check=loading.set_check,
                )
                self._profile = profile
                loading.after(0, lambda: self._transition_from_loading(loading))
            except Exception as exc:
                logger.exception("Initialisation failed")
                loading.show_error(str(exc))

        threading.Thread(target=_run_init, daemon=True).start()
        loading.mainloop()

        # 3. After loading: wizard for new, or go directly to chat
        if choice == "new":
            profile = self._run_setup_wizard()
            if on_profile_final:
                on_profile_final(profile)
        else:
            profile = selected_profile or self._profile or BotProfile()
            if on_profile_final:
                on_profile_final(profile)

        profile.touch()
        self._open_chat(profile)

    def _transition_from_loading(self, loading) -> None:
        loading._pulse_active = False
        loading.after(300, loading.destroy)

    # ── Welcome + Profile List screen ──────────────────────────────────────────

    def _run_welcome(self) -> tuple[str, BotProfile | None]:
        """
        Shows welcome branding, then a profile-list page (if profiles exist).
        Returns ("continue", profile) or ("new", None).
        """
        result: list = ["new", None]

        root = ctk.CTk()
        root.title("PHILIA")
        root.configure(fg_color=_BG)
        w, h = 860, 700
        root.geometry(f"{w}x{h}")
        root.minsize(680, 560)
        root.resizable(True, True)
        root.update_idletasks()
        x = (root.winfo_screenwidth()  - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"+{x}+{y}")

        content = tk.Frame(root, bg=_BG)
        content.pack(fill="both", expand=True)

        footer = tk.Frame(root, bg=_BG)
        footer.pack(fill="x", padx=48, pady=16)

        back_btn = ctk.CTkButton(
            footer, text="Back", width=110,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_DIM, state="disabled",
        )
        back_btn.pack(side="left")

        new_btn = ctk.CTkButton(
            footer, text="+ New Companion", width=170,
            fg_color=_SURFACE2, hover_color=_BORDER,
            text_color=_DIM, font=("Segoe UI", 12),
        )
        # new_btn hidden on welcome page, shown on profile list

        next_btn = ctk.CTkButton(
            footer, text="Get Started", width=150,
            fg_color=_ACCENT, hover_color=_AHOVER,
            font=("Segoe UI", 13, "bold"),
        )
        next_btn.pack(side="right")

        # Shared image cache so PhotoImages stay alive
        _img_cache: dict[str, ImageTk.PhotoImage] = {}

        def _clear():
            for w in content.winfo_children():
                w.destroy()

        def _commit(profile: BotProfile):
            result[0] = "continue"
            result[1] = profile
            root.quit()

        def _pick_new():
            result[0] = "new"
            result[1] = None
            root.quit()

        # ── Page 1: Welcome ────────────────────────────────────────────────────
        def _show_welcome():
            _clear()
            back_btn.configure(state="disabled", fg_color=_SURFACE, text_color=_BORDER)
            new_btn.pack_forget()
            next_btn.pack(side="right")
            profiles = BotProfile.list_all()
            next_btn.configure(
                text="Get Started",
                command=_show_profiles if profiles else _pick_new,
            )

            tk.Frame(content, bg=_BG).pack(expand=True, fill="both")
            inner = tk.Frame(content, bg=_BG)
            inner.pack()

            tk.Label(inner, text="PHILIA", font=("Segoe UI", 52, "bold"),
                     fg=_ACCENT, bg=_BG).pack()
            tk.Label(inner, text="Emotionally Aware Conversational Companion",
                     font=("Segoe UI", 14), fg=_DIM, bg=_BG).pack(pady=(4, 0))
            tk.Frame(inner, bg=_BORDER, height=1).pack(fill="x", pady=28)
            tk.Label(
                inner,
                text=(
                    "PHILIA listens to your voice, reads your expressions,\n"
                    "and responds with genuine emotional awareness.\n\n"
                    "Set up your companion in under a minute."
                ),
                font=("Segoe UI", 12), fg=_FG, bg=_BG, justify="center",
            ).pack()
            tk.Frame(content, bg=_BG).pack(expand=True, fill="both")

        # ── Page 2: Profile List ───────────────────────────────────────────────
        def _show_profiles():
            _clear()
            back_btn.configure(
                state="normal", fg_color=_SURFACE2, text_color=_DIM,
                command=_show_welcome,
            )
            next_btn.pack_forget()
            new_btn.pack(side="right")
            new_btn.configure(command=_pick_new)

            profiles = BotProfile.list_all()

            outer = tk.Frame(content, bg=_BG)
            outer.pack(fill="both", expand=True, padx=48, pady=(24, 8))

            tk.Label(outer, text="Your Companions",
                     font=("Segoe UI", 20, "bold"), fg=_FG, bg=_BG).pack(anchor="w")
            tk.Label(outer, text="Select a companion to start chatting.",
                     font=("Segoe UI", 11), fg=_DIM, bg=_BG).pack(anchor="w", pady=(2, 16))

            scroll = ctk.CTkScrollableFrame(outer, fg_color=_BG)
            scroll.pack(fill="both", expand=True)

            def _build_card(profile: BotProfile):
                card = tk.Frame(scroll, bg=_SURFACE, cursor="hand2")
                card.pack(fill="x", pady=6)

                # Avatar thumbnail
                av_path = Path(__file__).parent.parent / profile.avatar_path
                if av_path.exists():
                    try:
                        img = Image.open(av_path).resize((56, 56), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        _img_cache[profile.name] = photo
                        av_lbl = tk.Label(card, image=photo, bg=_SURFACE, cursor="hand2")
                    except Exception:
                        av_lbl = tk.Label(card, text="◈", font=("Segoe UI", 28),
                                          fg=_ACCENT, bg=_SURFACE, cursor="hand2")
                else:
                    av_lbl = tk.Label(card, text="◈", font=("Segoe UI", 28),
                                      fg=_ACCENT, bg=_SURFACE, cursor="hand2")
                av_lbl.pack(side="left", padx=(14, 10), pady=10)

                # Name + voice info
                info = tk.Frame(card, bg=_SURFACE, cursor="hand2")
                info.pack(side="left", fill="both", expand=True, pady=10)

                tk.Label(info, text=profile.name, font=("Segoe UI", 14, "bold"),
                         fg=_FG, bg=_SURFACE, cursor="hand2").pack(anchor="w")

                # Resolve a friendly voice label
                from speech.tts import list_voices
                voices = {v["id"]: v["label"] for v in list_voices()}
                voice_label = voices.get(profile.tts_voice, profile.tts_voice)
                tk.Label(info, text=voice_label, font=("Segoe UI", 10),
                         fg=_DIM, bg=_SURFACE, cursor="hand2").pack(anchor="w")

                # Edit / Delete buttons
                btns = tk.Frame(card, bg=_SURFACE)
                btns.pack(side="right", padx=12)

                ctk.CTkButton(
                    btns, text="✏ Edit", width=72, height=30,
                    fg_color=_SURFACE2, hover_color=_BORDER,
                    text_color=_DIM, font=("Segoe UI", 10),
                    command=lambda p=profile: _edit_profile(p),
                ).pack(side="left", padx=(0, 6))

                ctk.CTkButton(
                    btns, text="🗑 Delete", width=80, height=30,
                    fg_color=_SURFACE2, hover_color="#5A1A1A",
                    text_color=_RED, font=("Segoe UI", 10),
                    command=lambda p=profile: _delete_profile(p),
                ).pack(side="left")

                # Clicking the card (excluding buttons) launches the profile
                for w in [card, av_lbl, info] + list(info.winfo_children()):
                    w.bind("<Button-1>", lambda e, p=profile: _commit(p))

            for p in profiles:
                _build_card(p)

        def _edit_profile(profile: BotProfile):
            from ui.setup_screen import SetupScreen
            result_holder: list[BotProfile] = []

            def _on_done(p: BotProfile):
                result_holder.append(p)

            wizard = SetupScreen(
                parent=root,
                config=self._config,
                on_complete=_on_done,
                existing=profile,
            )
            root.wait_window(wizard)
            if result_holder:
                _show_profiles()   # refresh list

        def _delete_profile(profile: BotProfile):
            ok = messagebox.askyesno(
                "Delete Companion",
                f"Delete '{profile.name}'? This cannot be undone.",
                parent=root,
            )
            if ok:
                profile.delete()
                _show_profiles()   # refresh list

        root.protocol("WM_DELETE_WINDOW", root.quit)
        _show_welcome()
        root.mainloop()
        try:
            root.destroy()
        except Exception:
            pass
        return result[0], result[1]

    # ── Setup wizard ───────────────────────────────────────────────────────────

    def _run_setup_wizard(self, existing: BotProfile | None = None) -> BotProfile:
        from ui.setup_screen import SetupScreen
        result_holder: list[BotProfile] = []
        tmp_root = ctk.CTk()
        tmp_root.withdraw()

        def _on_complete(p: BotProfile):
            result_holder.append(p)
            tmp_root.destroy()

        wizard = SetupScreen(
            parent=tmp_root,
            config=self._config,
            on_complete=_on_complete,
            existing=existing,
        )
        wizard.protocol("WM_DELETE_WINDOW", lambda: None)
        tmp_root.mainloop()
        return result_holder[0] if result_holder else BotProfile()

    def _open_chat(self, profile: BotProfile) -> None:
        self._profile = profile
        from ui.chat_window import ChatWindow
        self._chat = ChatWindow(
            profile=profile,
            on_ptt=self._on_ptt_callback or (lambda: None),
            on_settings=self._show_settings,
            ptt_press_event=self._ptt_press_event,
            ptt_release_event=self._ptt_release_event,
        )
        self._chat.mainloop()

    # ── Settings ───────────────────────────────────────────────────────────────

    def _show_settings(self) -> None:
        if not self._chat:
            return
        from ui.setup_screen import SetupScreen

        def _on_save(new_profile: BotProfile):
            self._profile = new_profile
            logger.info("Profile updated: name=%s voice=%s", new_profile.name, new_profile.tts_voice)
            if self._chat:
                self._chat.add_message("bot", "Profile saved. Changes apply on the next start.")

        SetupScreen(
            parent=self._chat,
            config=self._config,
            on_complete=_on_save,
            existing=self._profile,
        )
