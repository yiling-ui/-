"""retry.py — unified RetryPolicy for ccxt-backed adapter calls (TICKET-005).

Background
----------
Before this module landed:

  * ``CCXTExecutor.place_stop_order`` had a naive ``await asyncio.sleep(0.5
    * 2 ** attempt)`` retry that did not distinguish ``InvalidOrder``
    (operator error -- never retry) from ``NetworkError`` (transient).
  * ``market_order`` / ``cancel_order`` / ``set_leverage`` had ZERO retry.
    A single 502 from Binance during a tighten-stop chain would null the
    ``stop_order_id`` and trigger a market emergency-close on a winning
    position.

Design
------
``RetryPolicy.execute(coro_factory, *, op_name, ...)`` runs the coroutine
returned by ``coro_factory()`` and classifies any exception into one of
three buckets:

  * **TRANSIENT** -- retry with exponential backoff:
        ccxt: NetworkError, RequestTimeout, DDoSProtection,
              RateLimitExceeded, ExchangeNotAvailable
        Plus aiohttp/httpx timeout types as a string-name fallback so we
        keep working when ccxt isn't importable (tests).
        ``RateLimitExceeded`` parses the optional ``Retry-After`` hint
        from the exception args and uses that as the next sleep.

  * **FATAL_IMMEDIATE** -- never retry, re-raise on first hit:
        ccxt: InvalidOrder, InsufficientFunds, BadSymbol, NotSupported,
              AuthenticationError, PermissionDenied, ArgumentsRequired,
              OrderNotFound (idempotent caller decides what to do)

  * **ONE_SHOT** -- generic ``ExchangeError`` / ``RuntimeError`` ->
    retry exactly once, then give up. Catches the long tail of
    venue-specific errors that don't map cleanly to ccxt's taxonomy.

The factory pattern (``coro_factory: Callable[[], Awaitable]``) is on
purpose: it lets each retry build a *fresh* coroutine. Coroutines in
Python can only be awaited once; a list of awaitables would not work.

Idempotency
-----------
Retry on a ``create_order`` call is safe ONLY because TICKET-001 makes
every order carry a ``clientOrderId``: a duplicate create with the same
cid is rejected (or returned as the original) by every venue we ship.
The executor calls ``adapter.fetch_order(client_order_id=...)`` BEFORE
the retry to short-circuit "did the prior call actually succeed even
though we got a network error?". This module is therefore safe to use
for idempotent and non-idempotent calls alike, as long as the caller
follows the cid contract.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ------------------------------------------------------------------ #
# Error classification
# ------------------------------------------------------------------ #
#
# We import ccxt's exception classes lazily so the test environment
# (which sometimes runs without ccxt installed) can still import this
# module. ``_classify`` falls back to a string-name comparison when the
# class objects aren't available.


def _ccxt_class_names() -> tuple[set[str], set[str], set[str]]:
    """Return (transient, fatal, one_shot) class-name sets.

    Lazily computed once; tolerant of partial ccxt installs.
    """
    transient = {
        "NetworkError",
        "RequestTimeout",
        "DDoSProtection",
        "RateLimitExceeded",
        "ExchangeNotAvailable",
        "OnMaintenance",
        # stdlib / aiohttp / httpx
        "TimeoutError",
        "ConnectionError",
        "ClientConnectorError",
        "ServerDisconnectedError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
    fatal = {
        "InvalidOrder",
        "InsufficientFunds",
        "BadSymbol",
        "BadRequest",
        "NotSupported",
        "AuthenticationError",
        "PermissionDenied",
        "ArgumentsRequired",
        "OrderNotFound",
        "OrderNotFillable",
        "MarginModeAlreadySet",
    }
    one_shot = {
        "ExchangeError",
        "RuntimeError",
    }
    return transient, fatal, one_shot


_TRANSIENT_NAMES, _FATAL_NAMES, _ONE_SHOT_NAMES = _ccxt_class_names()


class _Verdict:
    TRANSIENT = "transient"
    FATAL = "fatal"
    ONE_SHOT = "one_shot"


def classify(exc: BaseException) -> str:
    """Map an exception to one of the three retry verdicts.

    We walk the MRO so subclasses (e.g. a venue-specific
    ``BinanceRateLimitExceeded`` subclassing ``RateLimitExceeded``) are
    classified correctly without needing to know every subclass.
    """
    for klass in type(exc).__mro__:
        name = klass.__name__
        if name in _TRANSIENT_NAMES:
            return _Verdict.TRANSIENT
        if name in _FATAL_NAMES:
            return _Verdict.FATAL
        if name in _ONE_SHOT_NAMES:
            return _Verdict.ONE_SHOT
    return _Verdict.ONE_SHOT


def parse_retry_after(exc: BaseException) -> float | None:
    """Best-effort extraction of a Retry-After hint from a RateLimitExceeded.

    ccxt sometimes embeds the retry-after in the exception message
    (``"... retry after 2.5s"``) or in ``args[1]`` as a number of
    milliseconds. Returns the suggested sleep in *seconds*, or None
    when nothing usable is found.
    """
    msg = str(exc)
    # Pattern: "retry after 2500ms" / "retry after 2.5s"
    import re
    m = re.search(r"retry[\s_-]*after[\s:=]*(\d+(?:\.\d+)?)\s*(ms|s)?", msg, re.I)
    if m:
        value = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit == "ms":
            return value / 1000.0
        return value
    # Fallback: numeric args[1] interpreted as ms.
    args = getattr(exc, "args", ())
    if len(args) >= 2 and isinstance(args[1], (int, float)):
        return float(args[1]) / 1000.0
    return None


# ------------------------------------------------------------------ #
# RetryPolicy
# ------------------------------------------------------------------ #


@dataclass
class RetryPolicy:
    """Run an async callable with classification-aware retry semantics.

    Args:
        max_attempts: hard ceiling on retries for transient errors
            (default 3). Total wall-clock time is bounded by
            ``base_backoff_sec * (2^max_attempts - 1) + max_attempts*0.5``.
        base_backoff_sec: first-retry sleep before exponential growth.
        max_backoff_sec: cap on any single sleep so a misbehaving
            ``Retry-After`` (e.g. "60s") cannot pin the trader for a
            full minute.
        jitter_sec: uniform [0, jitter_sec) random additive jitter to
            spread retries when many calls collide on a rate-limit
            window.
        respect_retry_after: when True (default), a RateLimitExceeded
            with a parseable Retry-After overrides the exponential
            backoff for that attempt only.

    The policy is stateless across calls — it's safe to share one
    instance across the whole adapter.
    """

    max_attempts: int = 3
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 8.0
    jitter_sec: float = 0.25
    respect_retry_after: bool = True
    # Test seam: tests can substitute a deterministic sleeper.
    sleeper: Callable[[float], Awaitable[None]] = field(
        default=asyncio.sleep,
    )

    async def execute(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        *,
        op_name: str,
        idempotency_check: Callable[[], Awaitable[T | None]] | None = None,
    ) -> T:
        """Run ``coro_factory()`` with retry-on-transient semantics.

        Args:
            coro_factory: zero-arg callable that returns a fresh
                awaitable each time it's called. Required because
                Python coroutines are single-shot.
            op_name: short label used in log messages (e.g.
                "market_order:RAVEUSDT").
            idempotency_check: optional async callable invoked AFTER a
                transient error and BEFORE retrying. If it returns a
                non-None value, that value is returned to the caller
                as if the original call had succeeded — this is how
                TICKET-001 short-circuits "did the prior request
                actually land on the venue?". When None or absent,
                ``execute`` simply retries without the pre-check.

        Raises:
            * the originating exception when classification is FATAL.
            * the originating exception when ONE_SHOT retry also fails.
            * the LAST originating exception when TRANSIENT retries
              exhaust ``max_attempts``.
        """
        attempts = 0
        one_shot_used = False
        last_exc: BaseException | None = None

        while True:
            attempts += 1
            try:
                return await coro_factory()
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001 — classification re-raises
                last_exc = e
                verdict = classify(e)

                if verdict == _Verdict.FATAL:
                    logger.warning(
                        "RetryPolicy[%s] FATAL %s: %s — giving up immediately",
                        op_name, type(e).__name__, e,
                    )
                    raise

                if verdict == _Verdict.ONE_SHOT:
                    if one_shot_used:
                        logger.warning(
                            "RetryPolicy[%s] ONE_SHOT %s exhausted: %s",
                            op_name, type(e).__name__, e,
                        )
                        raise
                    one_shot_used = True
                    sleep_for = self._compute_sleep(attempts, e)
                    logger.warning(
                        "RetryPolicy[%s] one-shot retry on %s after %.2fs: %s",
                        op_name, type(e).__name__, sleep_for, e,
                    )
                    await self.sleeper(sleep_for)
                    if idempotency_check is not None:
                        out = await self._run_idempotency_check(
                            idempotency_check, op_name,
                        )
                        if out is not None:
                            return out
                    continue

                # TRANSIENT
                if attempts >= self.max_attempts:
                    logger.warning(
                        "RetryPolicy[%s] transient %s exhausted after %d attempts: %s",
                        op_name, type(e).__name__, attempts, e,
                    )
                    raise

                sleep_for = self._compute_sleep(attempts, e)
                logger.info(
                    "RetryPolicy[%s] transient %s, attempt %d/%d, sleeping %.2fs: %s",
                    op_name, type(e).__name__, attempts,
                    self.max_attempts, sleep_for, e,
                )
                await self.sleeper(sleep_for)
                if idempotency_check is not None:
                    out = await self._run_idempotency_check(
                        idempotency_check, op_name,
                    )
                    if out is not None:
                        return out
                continue

        # unreachable; keeps mypy happy.
        assert last_exc is not None  # pragma: no cover
        raise last_exc  # pragma: no cover

    # ------------------- internals ------------------- #

    def _compute_sleep(self, attempt: int, exc: BaseException) -> float:
        """Exponential backoff with jitter; honours Retry-After when
        present and ``respect_retry_after`` is True."""
        if self.respect_retry_after:
            hint = parse_retry_after(exc)
            if hint is not None:
                return min(max(hint, 0.0), self.max_backoff_sec)
        # 2^(attempt-1) * base + uniform[0, jitter)
        base = self.base_backoff_sec * (2 ** max(0, attempt - 1))
        jitter = random.uniform(0.0, self.jitter_sec)
        return min(base + jitter, self.max_backoff_sec)

    @staticmethod
    async def _run_idempotency_check(
        check: Callable[[], Awaitable[Any]], op_name: str,
    ) -> Any:
        """Run the post-retry idempotency probe; swallow its own errors
        so a flaky probe never escalates a recoverable transient into a
        crashed call."""
        try:
            return await check()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "RetryPolicy[%s] idempotency_check raised (ignored): %s",
                op_name, e,
            )
            return None
