"""
prompt_builder.py — Constructs LLM prompts with emotion and tone context.

Responsibilities:
  - Maintain the PHILIA system persona prompt
  - Inject fused emotion label and tone description into the prompt
  - Inject the user's transcript as the human turn
  - Return a fully-formed prompt dict for the LLM
"""

from __future__ import annotations

from config import Config
from emotion.fusion import FusedEmotion
from bot.bot_profile import BotProfile


_SYSTEM_TEMPLATE = """\
You are {bot_name}, an emotionally intelligent conversational companion.
You are speaking with a user who appears to be feeling {emotion} (confidence: {confidence:.0%}).
Your response tone should be {tone}. Be empathetic, concise, and supportive.
Keep your response under 3 sentences unless the user asks for more detail.
"""


class PromptBuilder:
    """Assembles the LLM prompt with injected emotion and tone context."""

    def __init__(self, config: Config, profile: BotProfile) -> None:
        self._config = config
        self._profile = profile

    def build(
        self,
        transcript: str,
        fused: FusedEmotion,
        tone: str,
    ) -> list[dict[str, str]]:
        """
        Build a chat-format prompt for the LLM.

        Args:
            transcript: User's transcribed speech.
            fused:      FusedEmotion containing label and confidence.
            tone:       Tone descriptor from ToneMapper.

        Returns:
            List of message dicts: [{"role": ..., "content": ...}, ...]
        """
        system_content = _SYSTEM_TEMPLATE.format(
            bot_name=self._profile.name,
            emotion=fused.label,
            confidence=fused.confidence,
            tone=tone,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": transcript},
        ]
