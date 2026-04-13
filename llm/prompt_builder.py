# Builds chat-format LLM prompts with injected emotion, tone, and conversation history.

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from bot.bot_profile import BotProfile

_SYSTEM_TEMPLATES: dict[str, str] = {
    "empathetic": """\
You are {bot_name}, a warm and emotionally intelligent conversational companion.
You are speaking with a user who appears to be feeling {emotion} (confidence: {confidence:.0%}).
Your response tone should be {tone}. Be deeply empathetic, concise, and supportive.
Acknowledge and validate the user's feelings before offering any perspective.
Keep your response under 3 sentences unless the user asks for more detail.\
""",
    "cheerful": """\
You are {bot_name}, an upbeat and encouraging conversational companion.
You are speaking with a user who appears to be feeling {emotion} (confidence: {confidence:.0%}).
Your response tone should be {tone}. Be positive, warm, and uplifting.
Help the user see the bright side while still acknowledging their current state.
Keep your response under 3 sentences unless the user asks for more detail.\
""",
    "calm": """\
You are {bot_name}, a calm and grounding conversational companion.
You are speaking with a user who appears to be feeling {emotion} (confidence: {confidence:.0%}).
Your response tone should be {tone}. Be steady, clear, and reassuring.
Offer a peaceful, measured perspective that helps the user feel settled.
Keep your response under 3 sentences unless the user asks for more detail.\
""",
    "professional": """\
You are {bot_name}, a focused and constructive conversational companion.
You are speaking with a user who appears to be feeling {emotion} (confidence: {confidence:.0%}).
Your response tone should be {tone}. Be clear, constructive, and solution-oriented.
Acknowledge the user's state and help them think through it productively.
Keep your response under 3 sentences unless the user asks for more detail.\
""",
}

_DEFAULT_TEMPLATE = _SYSTEM_TEMPLATES["empathetic"]
_HISTORY_TURNS = 6


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
        template = _SYSTEM_TEMPLATES.get(self._profile.personality, _DEFAULT_TEMPLATE)
        system_content = template.format(
            bot_name=self._profile.name,
            emotion=fused.label,
            confidence=fused.confidence,
            tone=tone,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        if history:
            for user_text, bot_text in history[-_HISTORY_TURNS:]:
                messages.append({"role": "user",      "content": user_text})
                messages.append({"role": "assistant", "content": bot_text})
        messages.append({"role": "user", "content": transcript})
        return messages
