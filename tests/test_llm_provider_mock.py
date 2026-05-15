"""Tests for the LLM provider abstraction."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from altcoin_agent.llm_provider import (
    AnthropicProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    build_default_provider,
    parse_chat_json,
)

# --------------------------------------------------------------------- #
# Protocol satisfaction
# --------------------------------------------------------------------- #


def test_openai_compat_satisfies_protocol() -> None:
    p = OpenAICompatibleProvider(name="x", api_key="k", api_base="https://x",
                                  model="m")
    assert isinstance(p, LLMProvider)


def test_anthropic_satisfies_protocol() -> None:
    p = AnthropicProvider(api_key="k")
    assert isinstance(p, LLMProvider)


# --------------------------------------------------------------------- #
# parse_chat_json
# --------------------------------------------------------------------- #


def test_parse_chat_json_strips_fences() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    assert parse_chat_json(raw) == {"a": 1}


def test_parse_chat_json_plain() -> None:
    assert parse_chat_json('{"a":1}') == {"a": 1}


# --------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_sends_response_format_when_supported() -> None:
    route = respx.post("https://example.com/v1/chat/completions").mock(
        return_value=Response(200, json={
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"total_tokens": 42},
        })
    )
    p = OpenAICompatibleProvider(
        name="x", api_key="k", api_base="https://example.com",
        model="m", supports_json_format=True,
    )
    raw, used = await p.chat_json([{"role": "user", "content": "hi"}], timeout=5)
    assert raw == '{"ok": true}'
    assert used == 42
    body = route.calls[0].request.content
    assert b"response_format" in body
    await p.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_softprompt_when_json_format_unsupported() -> None:
    respx.post("https://example.com/v1/chat/completions").mock(
        return_value=Response(200, json={
            "choices": [{"message": {"content": '{"x": 1}'}}],
            "usage": {"total_tokens": 7},
        })
    )
    p = OpenAICompatibleProvider(
        name="x", api_key="k", api_base="https://example.com",
        model="m", supports_json_format=False,
    )
    raw, used = await p.chat_json([{"role": "user", "content": "hi"}], timeout=5)
    assert raw == '{"x": 1}'
    assert used == 7
    await p.aclose()


# --------------------------------------------------------------------- #
# Anthropic provider
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_translates_messages_and_extracts_text() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json={
            "content": [{"type": "text", "text": '{"intent":"pump"}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
    )
    p = AnthropicProvider(api_key="k", model="claude-mini")
    raw, used = await p.chat_json(
        [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "judge"},
        ],
        timeout=5,
    )
    assert raw == '{"intent":"pump"}'
    assert used == 15
    sent_body = route.calls[0].request.content.decode()
    # Anthropic body should contain `system` and the user message,
    # but NOT a 'system' role inside `messages`.
    assert '"system"' in sent_body
    assert "be precise" in sent_body
    await p.aclose()


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #


def test_factory_returns_none_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in [
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    p = build_default_provider()
    assert p is None


def test_factory_picks_deepseek_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    p = build_default_provider()
    assert p is not None
    assert p.name == "deepseek"
    assert "deepseek" in p.api_base  # type: ignore[attr-defined]


def test_factory_switches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    p = build_default_provider()
    assert p is not None
    assert p.name == "openai"
    assert p.model == "gpt-4o-mini"


def test_factory_switches_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anth")
    p = build_default_provider()
    assert p is not None
    assert p.name == "anthropic"
    assert isinstance(p, AnthropicProvider)


def test_factory_generic_requires_base_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "generic")
    monkeypatch.setenv("LLM_API_KEY", "sk-gen")
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert build_default_provider() is None

    monkeypatch.setenv("LLM_API_BASE", "https://my-internal-llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "internal-7b")
    p = build_default_provider()
    assert p is not None
    assert p.api_base == "https://my-internal-llm.example/v1"  # type: ignore[attr-defined]
    assert p.model == "internal-7b"


def test_factory_unknown_provider_falls_back_to_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "no-such-thing")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    p = build_default_provider()
    assert p is not None
    assert p.name == "deepseek"
