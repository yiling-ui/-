"""llm_provider.py — Pluggable LLM backends.

The ai_engine layer used to hard-code DeepSeek's OpenAI-compatible endpoint.
This module factors that out so any of the following can be swapped in via
the ``LLM_PROVIDER`` env var:

    * deepseek   (default)        OpenAI-compatible, response_format=json_object
    * openai                      same shape, openai.com/v1
    * openrouter                  same shape, openrouter.ai/api/v1
    * moonshot                    same shape, api.moonshot.cn/v1
    * qwen                        same shape, dashscope-compatible
    * anthropic                   Claude /v1/messages — translated for us
    * generic                     any OpenAI-compatible base_url + model

Rules every provider obeys:

    * single async ``chat_json(messages) -> (raw_text, total_tokens)``
    * raises httpx.HTTPError / httpx.TimeoutException on transport failure
    * ai_engine handles JSON parsing; ``response_format`` differences are
      absorbed inside the provider (Anthropic does not support json_object,
      we instead append a strict instruction to the system message).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Single async call returning (raw_content, total_tokens_used)."""

    name: str
    model: str

    async def chat_json(
        self, messages: list[dict[str, str]], *, timeout: float,
    ) -> tuple[str, int]: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------- #
# OpenAI-compatible (DeepSeek / OpenAI / OpenRouter / generic)
# --------------------------------------------------------------------- #


@dataclass
class OpenAICompatibleProvider:
    """Works for any backend that implements OpenAI's /v1/chat/completions
    endpoint contract. ``response_format=json_object`` is sent only when the
    backend is known to support it (DeepSeek, OpenAI). For unknown backends
    we soft-instruct the model in the system message."""

    name: str
    api_key: str
    api_base: str
    model: str
    supports_json_format: bool = True
    temperature: float = 0.2
    extra_headers: dict[str, str] | None = None
    _client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": "altcoin-momentum-agent/1.0"}
            if self.extra_headers:
                headers.update(self.extra_headers)
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=headers,
            )
        return self._client

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float,
    ) -> tuple[str, int]:
        client = await self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.supports_json_format:
            body["response_format"] = {"type": "json_object"}
        else:
            body["messages"] = [
                {"role": "system",
                 "content": "Respond with ONE valid JSON object and nothing else."},
                *messages,
            ]
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        used = int(usage.get("total_tokens") or 0)
        return content, used

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------- #
# Anthropic (Claude /v1/messages)
# --------------------------------------------------------------------- #


@dataclass
class AnthropicProvider:
    """Wraps Anthropic's /v1/messages endpoint."""

    api_key: str
    name: str = "anthropic"
    api_base: str = "https://api.anthropic.com"
    model: str = "claude-3-5-sonnet-20240620"
    temperature: float = 0.2
    max_tokens: int = 1024
    _client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "User-Agent": "altcoin-momentum-agent/1.0",
                    "anthropic-version": "2023-06-01",
                },
            )
        return self._client

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float,
    ) -> tuple[str, int]:
        client = await self._get_client()
        # Translate: collapse all 'system' messages into the top-level
        # `system` field; everything else goes into `messages`.
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                system_parts.append(m.get("content", ""))
            else:
                chat.append({"role": role, "content": m.get("content", "")})
        system_msg = "\n\n".join(system_parts)
        if system_msg:
            system_msg += (
                "\n\nIMPORTANT: respond with ONE valid JSON object and "
                "nothing else."
            )
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_msg,
            "messages": chat or [{"role": "user", "content": "Reply now."}],
        }
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": self.api_key},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        blocks = data.get("content") or []
        text = ""
        for blk in blocks:
            if blk.get("type") == "text":
                text += blk.get("text", "")
        usage = data.get("usage") or {}
        used = int((usage.get("input_tokens") or 0) +
                    (usage.get("output_tokens") or 0))
        return text, used

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------- #
# Factory: env-driven default
# --------------------------------------------------------------------- #


_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "supports_json_format": True,
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "api_base": "https://api.openai.com",
        "model": "gpt-4o-mini",
        "supports_json_format": True,
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api",
        "model": "anthropic/claude-3.5-sonnet",
        "supports_json_format": False,
        "key_env": "OPENROUTER_API_KEY",
    },
    "moonshot": {
        "api_base": "https://api.moonshot.cn",
        "model": "moonshot-v1-8k",
        "supports_json_format": True,
        "key_env": "MOONSHOT_API_KEY",
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode",
        "model": "qwen-plus",
        "supports_json_format": True,
        "key_env": "DASHSCOPE_API_KEY",
    },
    "anthropic": {
        "api_base": "https://api.anthropic.com",
        "model": "claude-3-5-sonnet-20240620",
        "supports_json_format": False,
        "key_env": "ANTHROPIC_API_KEY",
    },
    "generic": {
        "api_base": None,
        "model": None,
        "supports_json_format": True,
        "key_env": "LLM_API_KEY",
    },
}


def build_default_provider(
    *, name: str | None = None, api_key: str | None = None,
) -> LLMProvider | None:
    """Construct the active LLM provider from environment variables.

    Resolution:
        * LLM_PROVIDER env (default "deepseek") selects the backend.
        * The matching <BACKEND>_API_KEY env supplies the key.
        * For "generic", LLM_API_BASE + LLM_MODEL are required.

    Returns None if no key is available — the caller should treat that as
    "rule-only mode".
    """
    chosen = (name or os.getenv("LLM_PROVIDER") or "deepseek").lower()
    if chosen not in _PROVIDER_DEFAULTS:
        logger.warning("Unknown LLM_PROVIDER=%s, falling back to deepseek", chosen)
        chosen = "deepseek"

    cfg = _PROVIDER_DEFAULTS[chosen]
    key = api_key or os.getenv(cfg["key_env"]) or os.getenv("LLM_API_KEY")
    if not key:
        return None

    api_base = os.getenv("LLM_API_BASE") or cfg["api_base"]
    model = os.getenv("LLM_MODEL") or cfg["model"]
    if not api_base or not model:
        logger.error("LLM_API_BASE and LLM_MODEL are required for provider=%s",
                     chosen)
        return None

    if chosen == "anthropic":
        return AnthropicProvider(
            api_key=key, api_base=api_base, model=model,
        )
    return OpenAICompatibleProvider(
        name=chosen, api_key=key, api_base=api_base, model=model,
        supports_json_format=bool(cfg["supports_json_format"]),
    )


def parse_chat_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON parser shared by all providers."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
