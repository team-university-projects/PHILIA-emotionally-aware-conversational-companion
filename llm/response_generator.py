"""
response_generator.py — LLM inference for emotion-aware response generation.

Responsibilities:
  - Accept a chat-format prompt list
  - Call the configured LLM provider (OpenAI / Gemini / Ollama)
  - Return the raw text response
"""

from __future__ import annotations

from config import Config


class ResponseGenerator:
    """Calls the configured LLM and returns a text response."""

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        # TODO: initialise LLM client based on self._cfg.llm_provider

    def generate(self, prompt: list[dict[str, str]]) -> str:
        """
        Send prompt to the LLM and return the response text.

        Args:
            prompt: List of {"role": ..., "content": ...} dicts.

        Returns:
            LLM-generated response string.
        """
        raise NotImplementedError

    def _call_openai(self, prompt: list[dict[str, str]]) -> str:
        """Call the OpenAI Chat Completions API."""
        raise NotImplementedError

    def _call_gemini(self, prompt: list[dict[str, str]]) -> str:
        """Call the Google Gemini API."""
        raise NotImplementedError

    def _call_ollama(self, prompt: list[dict[str, str]]) -> str:
        """Call a local Ollama model via HTTP."""
        raise NotImplementedError
