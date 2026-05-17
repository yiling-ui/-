"""telegram_commands.py — Two-way Telegram control plane.

Operational patch
=================

V1.0's ``TelegramNotifier`` is **send-only**: the operator gets
push notifications on signals / opens / closes / errors but cannot
issue commands (e.g. "halt now", "what's my equity?") from the
phone. The audit and operator both flagged this as a missing piece
of "production-grade observability".

This module adds a lightweight long-polling command receiver that
lives alongside the existing ``TelegramNotifier`` (push) without
interfering with it. They share the bot token but use independent
request paths:

    Notifier  →  POST /bot<token>/sendMessage    (push)
    Commands  →  GET  /bot<token>/getUpdates     (pull)

Design constraints
------------------

* **Authorisation by chat_id whitelist.** Only messages from the
  configured ``allowed_chat_ids`` are processed; everything else is
  silently discarded. Without this, anyone who guesses or leaks the
  bot token could issue ``/halt`` and grief the operator.

* **Read-only by default.** Commands that modify state (``/halt``,
  ``/resume``) require ``allow_write_commands=True`` in the config.
  Read commands (``/status``, ``/equity``, ``/positions``) are
  available with the chat-id whitelist alone. This double-gates the
  destructive ones.

* **Idempotent halt/resume.** ``/halt`` while already halted is a
  no-op; ``/resume`` while not halted is a no-op. The handler logs
  what it did either way so the operator gets a confirmation
  message.

* **Stop-event aware.** The poller respects an ``asyncio.Event``
  for shutdown so the daemon shuts down cleanly even mid-poll. The
  long-poll timeout is 25s by default — well under the typical
  Telegram 30s server side limit, so even a SIGTERM mid-poll
  resolves within ~25s.

* **Network failures are non-fatal.** Any httpx error is swallowed
  with a redacted log line; the poller sleeps ``backoff_sec`` and
  retries. A persistently broken Telegram does NOT halt trading.

* **Token redaction.** Same redaction as ``TelegramNotifier``:
  every error log scrubs the bot token before it touches the logger
  so token leaks via stdout / Sentry are impossible.

Public surface
--------------

    TelegramCommandPoller(token, allowed_chat_ids, command_handlers,
                          allow_write_commands=False).run(stop_event)

Tests construct the poller with a fake httpx client and assert the
handler dispatch is correct without hitting the network.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Command handler protocol — async (cmd_name, args, chat_id) -> reply text
# ---------------------------------------------------------------------- #

CommandHandler = Callable[[str, list[str], int], Awaitable[str]]


@dataclass
class CommandHandlers:
    """Container for the operator-facing command implementations.

    Each handler returns the **reply text** to be sent back to the
    user. Returning an empty string means "send no reply" (rare; the
    operator usually wants confirmation even for noops).

    Handlers are bound to the daemon's live state in ``main.py``; this
    module only routes the dispatch.
    """

    status: CommandHandler | None = None
    equity: CommandHandler | None = None
    positions: CommandHandler | None = None
    halt: CommandHandler | None = None      # write-gated
    resume: CommandHandler | None = None    # write-gated
    pnl: CommandHandler | None = None
    help: CommandHandler | None = None

    @property
    def write_commands(self) -> set[str]:
        """Commands that are gated behind ``allow_write_commands``."""
        return {"halt", "resume"}

    def get(self, name: str) -> CommandHandler | None:
        return getattr(self, name, None)


@dataclass
class TelegramCommandPollerConfig:
    enabled: bool = False
    bot_token: str = ""
    api_base: str = "https://api.telegram.org"
    # Whitelist of chat IDs allowed to issue commands. The same
    # ``TG_CHAT_ID`` used for push is the obvious default but operators
    # can broaden this to e.g. a small ops team.
    allowed_chat_ids: set[int] = field(default_factory=set)
    # When False, ``/halt`` and ``/resume`` reply with a refusal even
    # if the chat is whitelisted. Defence in depth: phone-loss + chat
    # whitelist still doesn't hand the attacker a halt.
    allow_write_commands: bool = False
    # Long-poll timeout (server-side). Telegram's hard cap is 50s; we
    # default to 25s so SIGTERM resolves within half a minute.
    long_poll_timeout_sec: int = 25
    # Backoff after a failed poll cycle (network error, 5xx, etc.).
    backoff_sec: float = 5.0
    # ``getUpdates`` offset starting point (set on first reply).
    initial_offset: int = 0


# ---------------------------------------------------------------------- #
# Poller
# ---------------------------------------------------------------------- #


@dataclass
class TelegramCommandPoller:
    cfg: TelegramCommandPollerConfig
    handlers: CommandHandlers
    # Optional sender for replies. When None we no-op the reply (tests
    # exercise the dispatch without needing a live client). In
    # production the daemon passes ``TelegramNotifier`` so command
    # replies flow through the same rate-limited bucket as push.
    reply_sender: Callable[[str], Awaitable[None]] | None = None

    _client: httpx.AsyncClient | None = None
    _next_offset: int = 0
    _polls: int = 0
    _commands_seen: int = 0
    _commands_rejected: int = 0
    _errors: int = 0

    def __post_init__(self) -> None:
        self._next_offset = self.cfg.initial_offset

    # --------------------------- lifecycle --------------------------- #

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Larger HTTP timeout than Notifier: long-polls run for up
            # to ``long_poll_timeout_sec`` so the request-side timeout
            # has to exceed that.
            self._client = httpx.AsyncClient(
                base_url=self.cfg.api_base,
                timeout=self.cfg.long_poll_timeout_sec + 5.0,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _redact(self, s: str) -> str:
        if not self.cfg.bot_token:
            return s
        return (s
                .replace(f"/bot{self.cfg.bot_token}", "/bot***REDACTED***")
                .replace(self.cfg.bot_token, "***REDACTED***"))

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self.cfg.enabled:
            logger.info("TelegramCommandPoller disabled by config")
            return
        if not self.cfg.bot_token:
            logger.warning(
                "TelegramCommandPoller enabled but bot_token is empty; "
                "skipping",
            )
            return
        if not self.cfg.allowed_chat_ids:
            logger.warning(
                "TelegramCommandPoller enabled but allowed_chat_ids is "
                "empty -> every command would be rejected; skipping",
            )
            return
        logger.info(
            "TelegramCommandPoller started: %d allowed chat(s), "
            "write_commands=%s",
            len(self.cfg.allowed_chat_ids), self.cfg.allow_write_commands,
        )
        try:
            while not stop_event.is_set():
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._errors += 1
                    logger.warning(
                        "TelegramCommandPoller.poll_once failed "
                        "(swallowed): %s", self._redact(str(e)),
                    )
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(), timeout=self.cfg.backoff_sec,
                        )
                        return
                    except asyncio.TimeoutError:
                        continue
        finally:
            await self.aclose()

    # --------------------------- one poll cycle --------------------------- #

    async def poll_once(self) -> int:
        """Run a single ``getUpdates`` long-poll round. Returns the
        number of commands dispatched this round (0 means nothing new
        from Telegram). Never raises in normal operation; transient
        errors propagate to ``run`` for backoff."""
        self._polls += 1
        client = await self._get_client()
        params: dict[str, Any] = {
            "timeout": self.cfg.long_poll_timeout_sec,
        }
        if self._next_offset:
            params["offset"] = self._next_offset

        r = await client.get(
            f"/bot{self.cfg.bot_token}/getUpdates", params=params,
        )
        if r.status_code >= 400:
            logger.warning(
                "Telegram getUpdates non-2xx: %s %s",
                r.status_code, self._redact(r.text[:200]),
            )
            return 0

        try:
            payload = r.json()
        except Exception as e:
            logger.warning(
                "Telegram getUpdates JSON decode failed: %s",
                self._redact(str(e)),
            )
            return 0

        if not payload.get("ok"):
            logger.warning(
                "Telegram getUpdates not ok: %s",
                payload.get("description", ""),
            )
            return 0

        updates = payload.get("result", []) or []
        dispatched = 0
        for upd in updates:
            update_id = int(upd.get("update_id", 0))
            # Advance the offset past EVERY update we've now seen,
            # regardless of whether we acted on it. Otherwise an
            # unauthorised user could fill the queue with junk and
            # we'd re-process forever.
            if update_id >= self._next_offset:
                self._next_offset = update_id + 1
            try:
                if await self._dispatch(upd):
                    dispatched += 1
            except Exception as e:
                self._errors += 1
                logger.warning(
                    "TelegramCommandPoller dispatch failed: %s",
                    self._redact(str(e)),
                )
        return dispatched

    # --------------------------- dispatch --------------------------- #

    async def _dispatch(self, update: dict[str, Any]) -> bool:
        """Authorise, parse, and dispatch one command. Returns True iff
        a command handler ran (vs ignored/unauthorised)."""
        msg = update.get("message") or update.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = int(chat.get("id", 0))
        text = str(msg.get("text") or "").strip()

        if not text or not text.startswith("/"):
            return False
        if chat_id not in self.cfg.allowed_chat_ids:
            self._commands_rejected += 1
            logger.info(
                "TelegramCommandPoller: rejected command from "
                "unauthorised chat_id=%d", chat_id,
            )
            return False

        # Parse "/cmd@botname arg1 arg2" -> ("cmd", ["arg1", "arg2"]).
        parts = text.split()
        head = parts[0][1:]  # drop leading slash
        if "@" in head:
            head = head.split("@", 1)[0]
        cmd = head.lower()
        args = parts[1:]

        handler = self.handlers.get(cmd)
        if handler is None:
            await self._reply(
                f"unknown command: /{cmd}\n"
                f"try /help for the list",
            )
            return False

        if cmd in self.handlers.write_commands and not self.cfg.allow_write_commands:
            self._commands_rejected += 1
            await self._reply(
                f"refused: /{cmd} requires allow_write_commands=true "
                f"on this poller",
            )
            return False

        self._commands_seen += 1
        try:
            reply = await handler(cmd, args, chat_id)
        except Exception as e:
            self._errors += 1
            logger.exception(
                "command handler /%s raised: %s", cmd, e,
            )
            await self._reply(f"/{cmd} error: {type(e).__name__}")
            return True
        if reply:
            await self._reply(reply)
        return True

    async def _reply(self, text: str) -> None:
        if self.reply_sender is None:
            return
        try:
            await self.reply_sender(text)
        except Exception as e:
            logger.warning(
                "TelegramCommandPoller reply send failed: %s",
                self._redact(str(e)),
            )

    # --------------------------- diagnostics --------------------------- #

    @property
    def stats(self) -> dict[str, int]:
        return {
            "polls": self._polls,
            "commands_seen": self._commands_seen,
            "commands_rejected": self._commands_rejected,
            "errors": self._errors,
            "next_offset": self._next_offset,
        }
