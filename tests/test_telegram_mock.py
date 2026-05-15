"""Tests for the Telegram notifier."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from altcoin_agent.notifier import (
    NullNotifier,
    TelegramNotifier,
    build_default_notifier,
)

# --------------------------------------------------------------------- #
# NullNotifier (default when TG_ENABLED is unset)
# --------------------------------------------------------------------- #


def test_default_notifier_is_null_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TG_ENABLED", raising=False)
    n = build_default_notifier()
    assert isinstance(n, NullNotifier)


def test_default_notifier_is_null_when_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_ENABLED", "true")
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    n = build_default_notifier()
    assert isinstance(n, NullNotifier)


def test_default_notifier_returns_telegram_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_ENABLED", "true")
    monkeypatch.setenv("TG_BOT_TOKEN", "1234:ABCDE")
    monkeypatch.setenv("TG_CHAT_ID", "-100123")
    n = build_default_notifier()
    assert isinstance(n, TelegramNotifier)


# --------------------------------------------------------------------- #
# Notifier protocol — every method is async + non-raising
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_null_notifier_methods_are_no_ops() -> None:
    n = NullNotifier()
    await n.signal({"x": 1})
    await n.opened({})
    await n.closed({})
    await n.rejected({})
    await n.error("boom")
    await n.aclose()


# --------------------------------------------------------------------- #
# TelegramNotifier sends and survives errors
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_telegram_signal_sends_to_bot_api() -> None:
    route = respx.post(
        "https://api.telegram.org/bot1234:ABCDE/sendMessage",
    ).mock(return_value=Response(200, json={"ok": True}))
    n = TelegramNotifier(bot_token="1234:ABCDE", chat_id="-100123")
    await n.signal({
        "symbol": "RAVEUSDT", "direction": "long", "final_score": 95,
        "trigger_price": 1.0, "rule_signal_kinds": ["volume_spike"],
        "notes": ["LLM agree x1.2"],
    })
    assert route.called
    body = route.calls[0].request.content.decode()
    assert "RAVEUSDT" in body
    assert "LONG" in body
    assert "📡" in body
    await n.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_telegram_swallows_http_errors() -> None:
    respx.post("https://api.telegram.org/bot1234:ABCDE/sendMessage").mock(
        return_value=Response(500, text="bot failure"),
    )
    n = TelegramNotifier(bot_token="1234:ABCDE", chat_id="-100123")
    # Must NOT raise even though Telegram returned 500.
    await n.signal({"symbol": "X", "direction": "long"})
    await n.opened({"symbol": "X", "side": "long"})
    await n.error("something exploded")
    await n.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_telegram_swallows_network_errors() -> None:
    import httpx
    respx.post("https://api.telegram.org/bot1234:ABCDE/sendMessage").mock(
        side_effect=httpx.ConnectError("network down"),
    )
    n = TelegramNotifier(bot_token="1234:ABCDE", chat_id="-100123")
    await n.signal({"symbol": "X"})
    await n.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_telegram_html_escapes_user_content() -> None:
    route = respx.post(
        "https://api.telegram.org/bot1234:ABCDE/sendMessage",
    ).mock(return_value=Response(200, json={"ok": True}))
    n = TelegramNotifier(bot_token="1234:ABCDE", chat_id="-100123")
    await n.error("<script>alert(1)</script>")
    body = route.calls[0].request.content.decode()
    assert "&lt;script&gt;" in body
    assert "<script>" not in body
    await n.aclose()
