# Builds chat-format LLM prompts with injected emotion, tone, and conversation history.

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from bot.bot_profile import BotProfile

# ── Base persona (set once, never rebuilt) ────────────────────────────────────
# Emotion-aware delivery is injected per-turn in the user message,
# so conversation continuity is never broken.
_BASE_SYSTEM: dict[str, str] = {
    "empathetic": """\
You are {bot_name}, a warm, emotionally intelligent conversational companion.
Your role is to be genuinely present with the user — like a caring friend, not a therapist.

CRITICAL — you speak out loud, so write ONLY for spoken delivery:
• Use natural contractions (I'm, that's, you've, it's).
• Use conversational rhythm — short sentences, natural pauses with commas and ellipses.
• Never use bullet points, headers, or numbered lists.
• Never start with "I understand" or "I can see that" — lead with something real.
• Build on what was said before. Reference the actual words and topics the user used.
• Do not offer advice unless asked. Reflect back and ask one genuine follow-up.\
""",
    "cheerful": """\
You are {bot_name}, an upbeat, warm conversational companion.
You're like an encouraging friend who actually listens.

CRITICAL — you speak out loud, so write ONLY for spoken delivery:
• Use natural contractions and casual, warm language.
• Keep energy up but stay grounded — no hollow positivity.
• Never use bullet points, headers, or lists.
• Always reference the specific thing the user just said.
• One focused response, not a list of things to say.\
""",
    "calm": """\
You are {bot_name}, a calm and steady conversational companion.
You help people feel grounded and heard.

CRITICAL — you speak out loud, so write ONLY for spoken delivery:
• Use slow, steady sentence rhythm. Short, clear sentences.
• Natural contractions. Gentle, reassuring language.
• Never use bullet points, headers, or lists.
• Always respond to the specific thing the user said.
• One calm, focused thought — then a gentle question if appropriate.\
""",
    "professional": """\
You are {bot_name}, a thoughtful, constructive conversational companion.
You help people think through challenges clearly.

CRITICAL — you speak out loud, so write ONLY for spoken delivery:
• Clear, direct spoken language. Natural contractions.
• No bullet points, headers, or lists.
• Stay grounded in what the user actually said.
• One focused thought that moves the conversation forward.\
""",
}

# ── Per-emotion delivery guidance ─────────────────────────────────────────────
# Injected into every user turn so the LLM writes text that MATCHES the TTS
# prosody settings for that emotion (speed, pitch, style).
_EMOTION_DELIVERY: dict[str, str] = {
    "angry":    "The user seems angry. Respond with calm, firm, measured sentences — short and steady. De-escalate gently. Do not mirror their agitation.",
    "disgust":  "The user seems disgusted or frustrated. Use calm, measured language — acknowledge what's bothering them without amplifying it.",
    "fear":     "The user seems anxious or afraid. Be warm and reassuring. Use gentle, grounding language — steady pace, simple words.",
    "happy":    "The user seems happy. Match their positive energy. Be warm and engaged — it's okay to be a little playful.",
    "neutral":  "The user seems calm and neutral. Be natural and conversational — no need to over-emote.",
    "sad":      "The user seems sad. Be gentle and soft. Slow your language down — shorter sentences, warm words, no rushing.",
    "surprise": "The user seems surprised or caught off guard. Be curious and engaged — meet their energy with warmth.",
}

_DEFAULT_TEMPLATE = _BASE_SYSTEM["empathetic"]
_HISTORY_TURNS = 10


def _length_hint(transcript: str) -> str:
    """Return a length instruction proportional to the user's input."""
    words = len(transcript.split())
    if words <= 8:
        # "Hi", "How are you?", quick greetings
        return "Reply in 1 sentence only — keep it brief and natural."
    if words <= 20:
        # Short statement or simple question
        return "Reply in 1-2 sentences — conversational, not elaborate."
    if words <= 50:
        # A paragraph of thought
        return "Reply in 2-3 sentences — match the depth of what was shared."
    # Long detailed input — can go deeper
    return "Reply in 3-4 sentences — the user shared a lot, reflect that thoughtfully."


class PromptBuilder:

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._config = config
        self._profile = profile

    def build(
        self,
        transcript: str,
        fused: FusedEmotion,
        tone: str,
        history: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        template = _BASE_SYSTEM.get(self._profile.personality, _DEFAULT_TEMPLATE)
        system_content = template.format(bot_name=self._profile.name)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        # Conversation history — gives the LLM actual memory of the exchange
        if history:
            for user_text, bot_text in history[-_HISTORY_TURNS:]:
                messages.append({"role": "user",      "content": user_text})
                messages.append({"role": "assistant", "content": bot_text})

        # Per-turn: emotion delivery + proportional length + transcript
        delivery = _EMOTION_DELIVERY.get(fused.label, _EMOTION_DELIVERY["neutral"])
        length   = _length_hint(transcript)
        user_content = f"[{delivery} {length}]\n\n{transcript}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def update_profile(self, profile: BotProfile) -> None:
        """Hot-swap the profile (name, personality) without restarting."""
        self._profile = profile

