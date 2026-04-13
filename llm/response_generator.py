# Routes LLM inference to the configured backend (OpenAI / Gemini / Ollama).

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:

    def __init__(self, config: Config) -> None:
        self._cfg = config.models
        _router: dict[str, Callable[[list[dict[str, str]]], str]] = {
            "openai": self._call_openai,
            "gemini": self._call_gemini,
            "ollama": self._call_ollama,
        }
        provider = self._cfg.llm_provider.lower()
        if provider not in _router:
            raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {list(_router.keys())}")
        self._backend: Callable[[list[dict[str, str]]], str] = _router[provider]
        logger.info("ResponseGenerator initialised — provider=%s model=%s", provider, self._cfg.llm_model)

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._backend(messages)
            logger.debug("LLM response (%d chars): %s...", len(response), response[:80])
            return response
        except Exception as exc:
            logger.error("LLM backend error: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed. Run: pip install openai>=1.12") from exc
        api_key = os.environ.get(self._cfg.llm_api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not found. Set the '{self._cfg.llm_api_key_env}' environment variable.")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self._cfg.llm_model,
            messages=messages,
            max_tokens=self._cfg.llm_max_tokens,
            temperature=self._cfg.llm_temperature,
        )
        return completion.choices[0].message.content or ""

    def _call_gemini(self, messages: list[dict[str, str]]) -> str:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError("google-genai package not installed. Run: pip install google-genai") from exc
        api_key = os.environ.get(self._cfg.llm_api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not found. Set the '{self._cfg.llm_api_key_env}' environment variable.")
        system_parts: list[str] = []
        contents = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_parts.append(content)
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append(genai_types.Content(role=gemini_role, parts=[genai_types.Part(text=content)]))
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self._cfg.llm_model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction="\n".join(system_parts) if system_parts else None,
                max_output_tokens=self._cfg.llm_max_tokens,
                temperature=self._cfg.llm_temperature,
            ),
        )
        return response.text or ""

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
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
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {endpoint}. Ensure Ollama is running ('ollama serve')."
            ) from exc
        return result.get("message", {}).get("content", "")
