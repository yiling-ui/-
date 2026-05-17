"""telegram.py — Telegram bot notifier.

Sends structured cards for:

  * SIGNAL    — fused high-priority signal observed
  * OPENED    — order placed, position opened
  * CLOSED    — position closed (will be wired in a later iteration)
  * REJECTED  — risk gate rejected a signal
  * ERROR     — a critical error in the daemon

Failure-mode contract: every notifier method swallows all errors after
logging them. A flaky Telegram MUST NOT crash the trading loop.

Rate limiting (audit #24):
    Telegram's bot API caps a single bot at 30 msg/sec across the
    whole API surface. During a fast pump-and-dump we may emit dozens
    of SIGNAL / OPENED / CLOSED in a single second; the 31st request
    onwards gets a 429 (and on the worst case the bot is throttled
    for 60+ seconds). We use a token bucket sized to 28 msg/sec
    (leaving 2 messages of headroom for ad-hoc /status replies if
    a future maintenance bot shares the token) and *await* on the
    bucket — failed sends still fail-open, but rate-limit-induced
    sleeps protect the bot's reputation with the Telegram backend.

Configuration:
    TG_ENABLED        = "true" to opt in
    TG_BOT_TOKEN      = bot token from @BotFather
    TG_CHAT_ID        = target chat id (channel, group, or DM)
    TG_API_BASE       = optional, defaults to https://api.telegram.org
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------- #


@dataclass
class _TokenBucket:
    """Async-aware token bucket for outbound Telegram messages.

    Capacity = ``rate`` * 1 second of burst tolerance. ``acquire``
    awaits until at least one token is available. Single-bucket per
    notifier is fine: bot tokens are 1:1 with bots and the API limit
    is per-bot.
    """

    rate_per_sec: float = 28.0   # 30 - 2 headroom
    capacity: float = 28.0
    _tokens: float = field(default=28.0)
    _last_refill: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    async def acquire(self) -> None:
        # Audit (third pass) #5: previously we held the lock across
        # ``await asyncio.sleep(...)``, which serialised every
        # concurrent acquire — 30 racing senders would each wait for
        # the previous one's sleep to finish, turning a 30-msg burst
        # into ~1.1s of latency instead of the bucket's design intent
        # (28 msgs immediate, 29th waits ~36ms). The fix is the
        # standard "compute under lock, sleep outside" pattern: each
        # iteration acquires the lock just long enough to refill +
        # check + (on miss) compute the sleep horizon, then releases
        # before sleeping. Concurrent coroutines therefore observe
        # tokens being added by wall-clock during their own sleep
        # rather than queuing behind one another.
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last_refill)
                self._last_refill = now
                self._tokens = min(
                    self.capacity, self._tokens + elapsed * self.rate_per_sec,
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Compute exact sleep until 1 full token will be there.
                deficit = 1.0 - self._tokens
                sleep_for = max(0.001, deficit / max(self.rate_per_sec, 1e-6))
            # Lock released here; another coroutine may now refill /
            # decrement under the same wall clock we're about to sleep
            # over, which is exactly what we want — no serialisation.
            await asyncio.sleep(sleep_for)


@runtime_checkable
class Notifier(Protocol):
    """Out-of-band notification channel. All methods are async and never raise."""

    name: str

    async def signal(self, payload: dict[str, Any]) -> None: ...
    async def opened(self, payload: dict[str, Any]) -> None: ...
    async def closed(self, payload: dict[str, Any]) -> None: ...
    async def rejected(self, payload: dict[str, Any]) -> None: ...
    async def error(self, message: str, payload: dict[str, Any] | None = None) -> None: ...
    # Operational patch (post-review): dedicated channel for legitimate
    # bank flow events (deposit / withdrawal). Previously routed through
    # ``error()`` -> 🚨 ERROR icon, which trains the operator to ignore
    # the message they most need to read. ``flow()`` renders with a
    # neutral 🏦 BANK icon. Existing callers that don't implement this
    # method are not broken: ``NullNotifier.flow`` is a no-op and the
    # Protocol is ``runtime_checkable`` (``hasattr`` works).
    async def flow(self, message: str, payload: dict[str, Any] | None = None) -> None: ...
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

    async def flow(self, message: str, payload: dict[str, Any] | None = None) -> None:
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
    rate_per_sec: float = 28.0
    _client: httpx.AsyncClient | None = None
    _bucket: _TokenBucket | None = None

    def _get_bucket(self) -> _TokenBucket:
        if self._bucket is None:
            self._bucket = _TokenBucket(
                rate_per_sec=self.rate_per_sec,
                capacity=self.rate_per_sec,
            )
        return self._bucket

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
        # Audit #24: enforce 28 msg/s ceiling so the bot never gets
        # 429'd by Telegram during a high-priority burst.
        try:
            await self._get_bucket().acquire()
        except Exception as e:  # pragma: no cover — async cancellation
            logger.warning("Telegram rate-limit acquire failed: %s", e)
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

    async def flow(self, message: str, payload: dict[str, Any] | None = None) -> None:
        """Operator deposits/withdrawals — neutral bank icon, NOT error."""
        msg = f"<b>🏦 BANK FLOW</b>\n<pre>{self._esc(message)}</pre>"
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
