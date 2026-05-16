"""telegram.py — Telegram bot notifier.

Sends structured cards for:

  * SIGNAL    — fused high-priority signal observed
  * OPENED    — order placed, position opened
  * CLOSED    — position closed (will be wired in a later iteration)
  * REJECTED  — risk gate rejected a signal
  * ERROR     — a critical error in the daemon

Failure-mode contract: every notifier method swallows all errors after
logging them. A flaky Telegram MUST NOT crash the trading loop.

Configuration:
    TG_ENABLED        = "true" to opt in
    TG_BOT_TOKEN      = bot token from @BotFather
    TG_CHAT_ID        = target chat id (channel, group, or DM)
    TG_API_BASE       = optional, defaults to https://api.telegram.org
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class Notifier(Protocol):
    """Out-of-band notification channel. All methods are async and never raise."""

    name: str

    async def signal(self, payload: dict[str, Any]) -> None: ...
    async def opened(self, payload: dict[str, Any]) -> None: ...
    async def closed(self, payload: dict[str, Any]) -> None: ...
    async def rejected(self, payload: dict[str, Any]) -> None: ...
    async def error(self, message: str, payload: dict[str, Any] | None = None) -> None: ...
    async def aclose(self) -> None: ...


class NullNotifier:
    """Default no-op notifier when Telegram is not configured."""

    name = "null"

    async def signal(self, payload: dict[str, Any]) -> None:
        return None

    async def opened(self, payload: dict[str, Any]) -> None:
        return None

    async def closed(self, payload: dict[str, Any]) -> None:
        return None

    async def rejected(self, payload: dict[str, Any]) -> None:
        return None

    async def error(self, message: str, payload: dict[str, Any] | None = None) -> None:
        return None

    async def aclose(self) -> None:
        return None


@dataclass
class TelegramNotifier:
    """Telegram bot notifier using sendMessage with HTML parse mode.

    Bug C2 fix: the bot token used to be baked into ``client.base_url``
    as ``/bot{token}``. That meant any httpx exception (timeout, 5xx,
    DNS failure) would carry the full URL in its ``repr`` and end up
    in stdout / Sentry / log shippers — leaking the token to anyone
    with log access. We now keep ``base_url=api_base`` and put the
    token only in the request *path* (and request-time header), and
    every error path runs ``_redact`` over the rendered exception
    before it touches the logger.
    """

    bot_token: str
    chat_id: str
    api_base: str = "https://api.telegram.org"
    name: str = "telegram"
    timeout: float = 5.0
    _client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=self.timeout,
            )
        return self._client

    def _redact(self, s: str) -> str:
        """Strip the bot token from a string before logging.

        We replace both the bare token and the ``/bot<token>`` URL
        substring so anything httpx might emit (URL, repr of
        Request/Response, traceback frames) is safe to log.
        """
        if not self.bot_token:
            return s
        return (s
                .replace(f"/bot{self.bot_token}", "/bot***REDACTED***")
                .replace(self.bot_token, "***REDACTED***"))

    async def _send(self, html: str) -> None:
        try:
            client = await self._get_client()
            r = await client.post(
                f"/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": html,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if r.status_code >= 400:
                logger.warning("Telegram non-2xx: %s %s",
                               r.status_code, self._redact(r.text[:200]))
        except Exception as e:
            logger.warning("Telegram send failed (swallowed): %s",
                           self._redact(str(e)))

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ---------------- card formatting ---------------- #

    @staticmethod
    def _esc(s: str) -> str:
        return (str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    async def signal(self, p: dict[str, Any]) -> None:
        sym = self._esc(p.get("symbol", "?"))
        direction = self._esc(p.get("direction", "?")).upper()
        score = p.get("final_score", "?")
        reason = self._esc(", ".join(p.get("rule_signal_kinds", []))[:120])
        notes = self._esc(" | ".join(p.get("notes") or [])[:200])
        msg = (
            f"<b>📡 SIGNAL {direction}</b>  <code>{sym}</code>\n"
            f"score: <b>{score}</b>   trigger: {p.get('trigger_price', '?')}\n"
            f"rules: {reason}\n"
            f"<i>{notes}</i>"
        )
        await self._send(msg)

    async def opened(self, p: dict[str, Any]) -> None:
        sym = self._esc(p.get("symbol", "?"))
        side = self._esc(p.get("side", "?")).upper()
        size = p.get("size", "?")
        lev = p.get("leverage", "?")
        entry = p.get("entry_price", "?")
        stop = p.get("initial_stop", "?")
        msg = (
            f"<b>🟢 OPENED {side}</b>  <code>{sym}</code>\n"
            f"size: <b>{size}</b>  lev: <b>{lev}x</b>\n"
            f"entry: {entry}  stop: {stop}"
        )
        await self._send(msg)

    async def closed(self, p: dict[str, Any]) -> None:
        sym = self._esc(p.get("symbol", "?"))
        side = self._esc(p.get("side", "?")).upper()
        pnl = p.get("realized_pnl_usdt", "?")
        r_mult = p.get("realized_r", "?")
        reason = self._esc(p.get("reason", ""))
        emoji = "🔴"
        try:
            if isinstance(pnl, (int, float)) and pnl >= 0:
                emoji = "✅"
        except Exception:
            pass
        msg = (
            f"<b>{emoji} CLOSED {side}</b>  <code>{sym}</code>\n"
            f"pnl: <b>${pnl}</b>  R: <b>{r_mult}</b>\n"
            f"reason: {reason}"
        )
        await self._send(msg)

    async def rejected(self, p: dict[str, Any]) -> None:
        sym = self._esc(p.get("symbol", "?"))
        reason = self._esc(p.get("reason", "?"))
        msg = (
            f"<b>⚠️ REJECTED</b>  <code>{sym}</code>\n"
            f"reason: {reason}"
        )
        await self._send(msg)

    async def error(self, message: str, payload: dict[str, Any] | None = None) -> None:
        msg = f"<b>🚨 ERROR</b>\n<pre>{self._esc(message)}</pre>"
        if payload:
            msg += "\n" + self._esc(str(payload)[:300])
        await self._send(msg)


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #


def build_default_notifier() -> Notifier:
    """Construct the configured notifier from env vars; NullNotifier if off."""
    enabled = os.getenv("TG_ENABLED", "false").lower() in ("1", "true", "yes")
    if not enabled:
        return NullNotifier()
    token = os.getenv("TG_BOT_TOKEN", "")
    chat_id = os.getenv("TG_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning(
            "TG_ENABLED=true but TG_BOT_TOKEN/TG_CHAT_ID is missing; "
            "using NullNotifier",
        )
        return NullNotifier()
    api_base = os.getenv("TG_API_BASE", "https://api.telegram.org")
    return TelegramNotifier(
        bot_token=token, chat_id=chat_id, api_base=api_base,
    )
