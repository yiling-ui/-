"""Tests for the two-way Telegram command poller."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from altcoin_agent.notifier.telegram_commands import (
    CommandHandlers,
    TelegramCommandPoller,
    TelegramCommandPollerConfig,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


class _FakeTransport(httpx.MockTransport):
    """Wrap ``httpx.MockTransport`` with a programmable response queue.

    Each ``getUpdates`` call pops the next entry from ``responses`` and
    returns it as a 200 JSON. When the queue is empty we return
    ``{"ok": true, "result": []}`` so long-polls behave like an idle
    Telegram server.
    """

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[httpx.Request] = []
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self._responses:
            body = self._responses.pop(0)
        else:
            body = {"ok": True, "result": []}
        return httpx.Response(
            200, json=body,
            headers={"Content-Type": "application/json"},
        )


def _poller(
    *,
    handlers: CommandHandlers,
    responses: list[dict[str, Any]],
    allow_write: bool = False,
    chat_ids: set[int] = None,  # type: ignore[assignment]
) -> tuple[TelegramCommandPoller, _FakeTransport, list[str]]:
    """Build a poller wired to a fake transport. Returns (poller,
    transport, captured_replies)."""
    transport = _FakeTransport(responses)
    cfg = TelegramCommandPollerConfig(
        enabled=True,
        bot_token="123456:abc",
        allowed_chat_ids=chat_ids if chat_ids is not None else {42},
        allow_write_commands=allow_write,
        long_poll_timeout_sec=1,
        backoff_sec=0.01,
    )

    captured: list[str] = []

    async def reply_sender(text: str) -> None:
        captured.append(text)

    poller = TelegramCommandPoller(
        cfg=cfg, handlers=handlers, reply_sender=reply_sender,
    )
    # Inject the fake transport into the lazy client.
    poller._client = httpx.AsyncClient(
        base_url=cfg.api_base, transport=transport,
        timeout=cfg.long_poll_timeout_sec + 5.0,
    )
    return poller, transport, captured


def _update(update_id: int, chat_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "text": text,
        },
    }


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_authorized_status_command_dispatches() -> None:
    """A whitelisted chat sending /status invokes the handler and the
    handler's return text is sent as a reply."""
    captured_calls: list[tuple[str, list[str], int]] = []

    async def status(cmd: str, args: list[str], chat_id: int) -> str:
        captured_calls.append((cmd, args, chat_id))
        return "all systems nominal"

    handlers = CommandHandlers(status=status)
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/status")],
        }],
    )
    n = await poller.poll_once()
    assert n == 1
    assert captured_calls == [("status", [], 42)]
    assert replies == ["all systems nominal"]
    await poller.aclose()


@pytest.mark.asyncio
async def test_unauthorized_chat_id_silently_rejected() -> None:
    """A command from a non-whitelisted chat must NOT invoke the
    handler and must NOT reply (silent drop = no info leak)."""
    handler_calls: list[Any] = []

    async def status(*a: Any) -> str:
        handler_calls.append(a)
        return "should not run"

    handlers = CommandHandlers(status=status)
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 99999, "/status")],  # unauthorised
        }],
        chat_ids={42},
    )
    n = await poller.poll_once()
    assert n == 0
    assert handler_calls == []
    assert replies == []
    assert poller.stats["commands_rejected"] == 1
    await poller.aclose()


@pytest.mark.asyncio
async def test_write_command_blocked_when_allow_write_is_false() -> None:
    """``/halt`` from a whitelisted chat is refused when the poller
    is in read-only mode. Refusal is replied (so the operator knows
    why) but the handler is NOT invoked."""
    invoked: list[Any] = []

    async def halt(*a: Any) -> str:
        invoked.append(a)
        return "halted"

    handlers = CommandHandlers(halt=halt)
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/halt")],
        }],
        allow_write=False,
    )
    await poller.poll_once()
    assert invoked == []
    assert any("refused" in r.lower() for r in replies)
    await poller.aclose()


@pytest.mark.asyncio
async def test_write_command_allowed_when_flag_is_set() -> None:
    """When ``allow_write_commands=True`` the same /halt invokes the
    handler."""
    invoked: list[Any] = []

    async def halt(cmd: str, args: list[str], chat_id: int) -> str:
        invoked.append((cmd, args, chat_id))
        return "halted by operator"

    handlers = CommandHandlers(halt=halt)
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/halt manual ops")],
        }],
        allow_write=True,
    )
    await poller.poll_once()
    assert invoked == [("halt", ["manual", "ops"], 42)]
    assert replies == ["halted by operator"]
    await poller.aclose()


@pytest.mark.asyncio
async def test_unknown_command_replies_with_help_pointer() -> None:
    handlers = CommandHandlers()
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/banana")],
        }],
    )
    await poller.poll_once()
    assert any("/help" in r for r in replies)
    await poller.aclose()


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_poller() -> None:
    """A buggy handler must not break the loop; the poller logs and
    sends an error reply."""
    async def status(*a: Any) -> str:
        raise RuntimeError("boom")

    handlers = CommandHandlers(status=status)
    poller, _, replies = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/status")],
        }],
    )
    await poller.poll_once()
    assert any("error" in r.lower() for r in replies)
    assert poller.stats["errors"] == 1
    await poller.aclose()


@pytest.mark.asyncio
async def test_offset_advances_past_unauthorized_updates() -> None:
    """Even rejected updates must advance the offset; otherwise an
    attacker can fill the queue with junk and we'd reprocess forever."""
    handlers = CommandHandlers(status=_passthrough)
    poller, transport, _ = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [
                _update(10, 99999, "/status"),  # unauth
                _update(11, 99999, "/status"),  # unauth
                _update(12, 42, "/status"),     # auth
            ],
        }],
        chat_ids={42},
    )
    n = await poller.poll_once()
    # Only the authorised one was dispatched.
    assert n == 1
    # Offset advanced past update_id=12 -> next call uses 13.
    assert poller.stats["next_offset"] == 13
    await poller.aclose()


@pytest.mark.asyncio
async def test_botname_suffix_stripped_from_command() -> None:
    """``/status@MyBotName`` must dispatch to ``status``."""
    seen: list[str] = []

    async def status(cmd: str, args: list[str], chat_id: int) -> str:
        seen.append(cmd)
        return "ok"

    handlers = CommandHandlers(status=status)
    poller, _, _ = _poller(
        handlers=handlers,
        responses=[{
            "ok": True,
            "result": [_update(1, 42, "/status@AltcoinBot")],
        }],
    )
    await poller.poll_once()
    assert seen == ["status"]
    await poller.aclose()


@pytest.mark.asyncio
async def test_disabled_run_returns_immediately() -> None:
    """``cfg.enabled=False`` must short-circuit ``run`` without any
    polling."""
    import asyncio as _asyncio
    cfg = TelegramCommandPollerConfig(enabled=False, bot_token="x")
    poller = TelegramCommandPoller(cfg=cfg, handlers=CommandHandlers())
    stop = _asyncio.Event()
    await poller.run(stop)
    assert poller.stats["polls"] == 0


@pytest.mark.asyncio
async def test_empty_chat_id_whitelist_short_circuits_run() -> None:
    """Empty allowed_chat_ids would mean every command is rejected;
    instead of silently polling forever we refuse to start."""
    import asyncio as _asyncio
    cfg = TelegramCommandPollerConfig(
        enabled=True, bot_token="x", allowed_chat_ids=set(),
    )
    poller = TelegramCommandPoller(cfg=cfg, handlers=CommandHandlers())
    stop = _asyncio.Event()
    await poller.run(stop)
    assert poller.stats["polls"] == 0


async def _passthrough(cmd: str, args: list[str], chat_id: int) -> str:
    return "ok"
