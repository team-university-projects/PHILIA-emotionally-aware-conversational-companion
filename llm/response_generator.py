"""
response_generator.py — LLM inference for emotion-aware response generation.

Responsibilities:
  - Accept a chat-format prompt (list[dict]) from PromptBuilder
  - Route to the configured LLM backend (OpenAI / Gemini / Ollama)
  - Return the generated response text

Supported backends (set via config.models.llm_provider):
  - "openai"  — OpenAI Chat Completions (gpt-4o-mini, gpt-4o, etc.)
  - "gemini"  — Google Gemini (gemini-2.0-flash, gemini-1.5-pro, etc.)
  - "ollama"  — Local Ollama server (llama3, mistral, etc.)

Backend extensibility
---------------------
  To add a new backend, implement a method matching the signature:
      def _call_<name>(self, messages: list[dict[str, str]]) -> str
  and add its provider key to _ROUTER in __init__.
  No changes to generate() or calling code are needed.

Prompt template (injected by PromptBuilder)
-------------------------------------------
  System:
    You are {bot_name}, an emotionally intelligent conversational companion.
    You are speaking with a user who appears to be feeling {emotion}
    (confidence: {confidence:.0%}). Your response tone should be {tone}.
    Be empathetic, concise, and supportive. Keep your response under 3
    sentences unless the user asks for more detail.

  User:
    {transcript}
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Routes LLM inference to the backend configured in ``config.models``.

    The active backend is determined once at ``__init__`` time and stored
    as a callable — ``generate()`` is a thin wrapper around it, allowing
    drop-in backend swaps without modifying calling code.

    Args:
        config: Global :class:`~config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.models

        # ── Backend routing ────────────────────────────────────────────────────
        # To add a new backend: implement _call_<name>() and add it here.
        _router: dict[str, Callable[[list[dict[str, str]]], str]] = {
            "openai": self._call_openai,
            "gemini": self._call_gemini,
            "ollama": self._call_ollama,
        }

        provider = self._cfg.llm_provider.lower()
        if provider not in _router:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Supported: {list(_router.keys())}"
            )

        self._backend: Callable[[list[dict[str, str]]], str] = _router[provider]
        logger.info(
            "ResponseGenerator initialised — provider=%s model=%s",
            provider, self._cfg.llm_model,
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Send a chat-format prompt to the active LLM backend.

        Args:
            messages: Prompt as returned by :class:`~llm.prompt_builder.PromptBuilder`
                      — a list of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            The LLM-generated response string.

        Raises:
            RuntimeError: If the backend call fails (API error, timeout, etc.).
        """
        try:
            response = self._backend(messages)
            logger.debug("LLM response (%d chars): %s...", len(response), response[:80])
            return response
        except Exception as exc:
            logger.error("LLM backend error: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    # ── Backends ────────────────────────────────────────────────────────────────

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        """
        Call the OpenAI Chat Completions API.

        Requires the ``openai`` package and a valid API key in the
        environment variable specified by ``config.models.llm_api_key_env``
        (default: ``OPENAI_API_KEY``).
        """
        try:
            from openai import OpenAI  # lazy import — only needed for this backend
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai>=1.12"
            ) from exc

        api_key = os.environ.get(self._cfg.llm_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set the '{self._cfg.llm_api_key_env}' "
                "environment variable."
            )

        client = OpenAI(api_key=api_key)
        logger.debug("Calling OpenAI model='%s' ...", self._cfg.llm_model)

        completion = client.chat.completions.create(
            model=self._cfg.llm_model,
            messages=messages,        # type: ignore[arg-type]
            max_tokens=self._cfg.llm_max_tokens,
            temperature=self._cfg.llm_temperature,
        )
        return completion.choices[0].message.content or ""

    def _call_gemini(self, messages: list[dict[str, str]]) -> str:
        """
        Call the Google Gemini API via the ``google-genai`` SDK.

        Requires ``google-genai`` package and a valid API key in the
        environment variable specified by ``config.models.llm_api_key_env``
        (default: ``OPENAI_API_KEY``; set ``llm_api_key_env = GEMINI_API_KEY``).

        Converts OpenAI-style message dicts to Gemini's Content format,
        treating the last ``user`` role message as the prompt and any prior
        messages as conversation history.
        """
        try:
            from google import genai                      # lazy import
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from exc

        api_key = os.environ.get(self._cfg.llm_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set the '{self._cfg.llm_api_key_env}' "
                "environment variable (e.g. GEMINI_API_KEY)."
            )

        # Extract system prompt and conversation turns
        system_parts: list[str] = []
        contents: list[genai_types.Content] = []

        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_parts.append(content)
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append(
                    genai_types.Content(
                        role=gemini_role,
                        parts=[genai_types.Part(text=content)],
                    )
                )

        system_instruction = "\n".join(system_parts) if system_parts else None
        logger.debug("Calling Gemini model='%s' ...", self._cfg.llm_model)

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self._cfg.llm_model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=self._cfg.llm_max_tokens,
                temperature=self._cfg.llm_temperature,
            ),
        )
        return response.text or ""

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        """
        Call a local Ollama server via its HTTP API (no SDK required).

        Ollama must be running locally (``ollama serve`` or the desktop app).
        The endpoint defaults to ``http://localhost:11434/api/chat``.
        Override by setting ``OLLAMA_HOST`` in the environment.

        Requires no additional packages beyond the standard library.
        """
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        endpoint = f"{host}/api/chat"

        payload = json.dumps({
            "model":    self._cfg.llm_model,
            "messages": messages,
            "stream":   False,
            "options": {
                "num_predict": self._cfg.llm_max_tokens,
                "temperature": self._cfg.llm_temperature,
            },
        }).encode("utf-8")

        logger.debug(
            "Calling Ollama model='%s' at %s ...", self._cfg.llm_model, endpoint
        )

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {endpoint}. "
                "Ensure Ollama is running ('ollama serve')."
            ) from exc

        return result.get("message", {}).get("content", "")


# ── Standalone test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import Config

    cfg = Config.load()
    provider = cfg.models.llm_provider
    gen = ResponseGenerator(cfg)

    # Example prompt simulating a user who is feeling sad
    test_messages = [
        {
            "role": "system",
            "content": (
                "You are PHILIA, an emotionally intelligent conversational companion.\n"
                "You are speaking with a user who appears to be feeling sad "
                "(confidence: 72%).\n"
                "Your response tone should be warm and gentle. Be empathetic, "
                "concise, and supportive.\n"
                "Keep your response under 3 sentences unless the user asks for more detail."
            ),
        },
        {
            "role": "user",
            "content": "I had a really rough day at work. Nothing went right.",
        },
    ]

    print("=" * 60)
    print(f"  PHILIA -- LLM Response Test ({provider})")
    print("=" * 60)
    print("\nSystem prompt:")
    print(f"  {test_messages[0]['content'][:120]}...")
    print(f"\nUser: {test_messages[1]['content']}")
    print("\nGenerating response...")

    try:
        response = gen.generate(test_messages)
        print(f"\nPHILIA: {response}")
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
