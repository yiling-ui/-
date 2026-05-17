"""ccxt_adapter.py — Live ExchangeAdapter backed by ccxt.pro.

Wraps a ccxt.pro client (Binance/OKX/Gate.io USDT-M perpetuals) into the
``ExchangeAdapter`` Protocol consumed by ``CCXTExecutor``. The adapter:

  * normalizes ccxt's heterogeneous return shapes into the small dict we use,
  * sets ``positionSide`` correctly for hedge-mode-enabled accounts,
  * handles testnet via the ``sandbox`` flag,
  * routes STOP_MARKET orders through the right ``params`` per venue,
  * **enforces exchange-level idempotency** (audit-fix #E1) by attaching
    a venue-specific ``clientOrderId`` to every entry, stop, and reduce-
    only close order. If a network blip / 502 / 504 makes us retry the
    same logical order, the venue dedupes on the ID instead of creating
    a phantom second position.
  * **retries transient failures** (audit-fix #E2) — 502 / 504 / 429 /
    DDoSProtection / NetworkError / RequestTimeout are full-jitter
    exponential-backoff retried up to ``transient_retries`` times. Logical
    errors (``InvalidOrder`` / ``InsufficientFunds`` / ``BadSymbol``) are
    re-raised immediately. The whole network call is also wrapped in
    ``asyncio.wait_for(..., total_timeout_sec)`` so a hung TCP socket
    can't deadlock the executor coroutine.

Usage:

    import ccxt.pro as ccxtpro
    client = ccxtpro.binance({
        "apiKey": ..., "secret": ...,
        "options": {"defaultType": "swap"},
    })
    if testnet:
        client.set_sandbox_mode(True)
    adapter = CCXTExchangeAdapter(client=client, exchange_name="binance")

The adapter is intentionally side-effect-free at construction so it can be
unit-tested without a network connection (we mock ``client`` itself).
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from altcoin_agent.risk.state import Side

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --------------------------------------------------------------------- #
# Transient error classification (audit-fix #E2)
# --------------------------------------------------------------------- #
#
# ccxt is imported lazily so the test suite (which mocks the client) does
# not require ccxt at import time. We classify exceptions by class name +
# stringified message rather than ``isinstance`` so test fakes can raise
# ordinary ``RuntimeError("rate limit ...")`` and still exercise the
# retry path.

_TRANSIENT_CCXT_NAMES: tuple[str, ...] = (
    "NetworkError",
    "ExchangeNotAvailable",
    "RequestTimeout",
    "DDoSProtection",
    "RateLimitExceeded",
    "OperationFailed",  # ccxt umbrella for 5xx-class venue errors
)

_LOGICAL_CCXT_NAMES: tuple[str, ...] = (
    "InvalidOrder",
    "InsufficientFunds",
    "BadSymbol",
    "BadRequest",
    "AuthenticationError",
    "PermissionDenied",
    "AccountSuspended",
    "MarginModeAlreadySet",
)

_TRANSIENT_MSG_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        # HTTP infrastructure errors (CDN/edge/upstream)
        r"\b50[234]\b",          # 502 / 503 / 504
        r"\b429\b",
        r"bad gateway",
        r"gateway timeout",
        r"service unavailable",
        # Connection & timeout
        r"connection (reset|aborted|refused)",
        r"timed? ?out",
        r"timeout",
        # Generic rate limit phrasing across venues
        r"rate.?limit",
        r"too many requests",
        r"ddos",
        r"temporarily unavailable",
    )
)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` looks like a transient infrastructure failure
    that is safe to retry idempotently (i.e. with the same clientOrderId)."""
    name = type(exc).__name__
    if name in _LOGICAL_CCXT_NAMES:
        return False
    if name in _TRANSIENT_CCXT_NAMES:
        return True
    msg = str(exc)
    if not msg:
        return False
    for pat in _TRANSIENT_MSG_PATTERNS:
        if pat.search(msg):
            return True
    return False


# --------------------------------------------------------------------- #
# clientOrderId derivation (audit-fix #E1)
# --------------------------------------------------------------------- #
#
# Every venue we ship has a slightly different field name and character-
# set rule. We standardise on a short, alphanumeric-only string so the
# same ID round-trips through binance / okx / bybit / gateio without
# hitting "invalid characters" rejections.
#
#   Binance USDT-M futures: ``newClientOrderId``,
#       ``^[\.A-Z\:/a-z0-9_-]{1,36}$``
#   OKX:                    ``clOrdId``,
#       1-32 chars, ``[A-Za-z0-9]``
#   Gate.io:                ``text``,
#       must start with ``t-`` and be 28 chars or fewer total
#   Bybit:                  ``orderLinkId``, 36 chars max alnum
#   ccxt unified:           ``clientOrderId``
#
# Our derivation produces 18-char base IDs (prefix ``e``/``s``/``c`` +
# 16 hex chars from secrets.token_hex). The Gate "t-" prefix adds 2 more
# chars (20 total), well under all venue limits.

_COID_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")


def _sanitize_for_coid(raw: str | None) -> str:
    if not raw:
        return ""
    return _COID_ALNUM_RE.sub("", raw)[:16]


def _derive_client_oid(
    *,
    prefix: str,
    trace_id: str | None,
    suffix: str | None = None,
) -> str:
    """Build a deterministic, venue-safe clientOrderId.

    The same ``trace_id`` always yields the same ``base`` portion, which
    is what makes retries idempotent. ``suffix`` lets ``add_leg`` /
    stop-replacement emit unique-but-related IDs (the venue rejects two
    orders with the same ID in the same 24h window on most exchanges,
    so legs and stop-resets must each get their own).
    """
    base = _sanitize_for_coid(trace_id)
    if not base:
        base = secrets.token_hex(8)  # 16 alnum chars
    if suffix:
        base = f"{base}{_sanitize_for_coid(suffix)}"
    # Final shape: <prefix-1ch><base-up-to-16ch>[<suffix-up-to-Nch>]
    return f"{prefix}{base}"[:30]


def _inject_client_oid(
    params: dict[str, Any],
    exchange_name: str,
    coid: str,
) -> None:
    """Set the venue-specific clientOrderId field IN-PLACE on ``params``."""
    name = (exchange_name or "").lower()
    if name == "binance":
        params["newClientOrderId"] = coid
    elif name == "okx":
        params["clOrdId"] = coid
    elif name == "gateio":
        # Gate.io requires a 't-' prefix and 28-char total cap.
        params["text"] = f"t-{coid}"[:28]
    elif name == "bybit":
        params["orderLinkId"] = coid
    else:
        # ccxt unified field — supported by every modern venue ccxt knows.
        params["clientOrderId"] = coid


@runtime_checkable
class _CCXTLike(Protocol):
    """The subset of ccxt.pro we depend on. Lets tests use a fake."""

    async def create_market_order(
        self, symbol: str, side: str, amount: float, *,
        params: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...

    async def create_order(
        self, symbol: str, type: str, side: str, amount: float, *,
        price: float | None = ..., params: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...

    async def cancel_order(
        self, id: str, symbol: str | None = ...,
        params: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...

    async def set_leverage(
        self, leverage: float, symbol: str,
        params: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...

    async def fetch_positions(
        self, symbols: list[str] | None = ...,
        params: dict[str, Any] | None = ...,
    ) -> list[dict[str, Any]]: ...

    async def fetch_open_orders(
        self, symbol: str | None = ...,
        params: dict[str, Any] | None = ...,
    ) -> list[dict[str, Any]]: ...


@dataclass
class CCXTExchangeAdapter:
    """ccxt.pro-backed ``ExchangeAdapter``.

    Args:
        client: a constructed ccxt.pro client.
        exchange_name: "binance" / "okx" / "gateio" / "bybit".
        hedge_mode: when True, sets ``positionSide`` on each order so the
            two directions don't share a single net position. Required for
            the SHORT path on Binance USDT-M futures unless the account is
            in one-way mode.
        transient_retries: max retry attempts for transient errors
            (502/504/429/NetworkError). 3 by default. Idempotency is
            guaranteed by the venue-attached ``clientOrderId``.
        retry_base_delay_sec: full-jitter exponential backoff base; the
            actual sleep is ``random.uniform(0, base * 2**attempt)``,
            capped at ``retry_max_delay_sec``.
        retry_max_delay_sec: ceiling on each retry sleep.
        total_timeout_sec: hard ceiling per network call (including all
            retries). Above this we give up and re-raise the last
            transient error so the executor can fail-closed.
    """

    client: _CCXTLike
    exchange_name: str = "binance"
    hedge_mode: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)
    transient_retries: int = 3
    retry_base_delay_sec: float = 0.4
    retry_max_delay_sec: float = 4.0
    total_timeout_sec: float = 12.0

    # ---------------- helpers ---------------- #

    def _stop_params(self, side: Side, stop_price: float, reduce_only: bool) -> dict[str, Any]:
        params: dict[str, Any] = {"reduceOnly": reduce_only,
                                   "stopPrice": stop_price,
                                   "workingType": "MARK_PRICE"}
        params.update(self.extra_params)
        if self.hedge_mode:
            # On Binance hedge mode, a sell-stop closes a LONG so positionSide=LONG.
            params["positionSide"] = "LONG" if side == Side.SHORT else "SHORT"
        # OKX uses different params shape; ccxt normalizes most of it but
        # `tdMode` (cross/isolated) must be passed if not set globally.
        if self.exchange_name == "okx":
            params.setdefault("tdMode", "cross")
        return params

    def _entry_params(self, side: Side, reduce_only: bool) -> dict[str, Any]:
        params: dict[str, Any] = {"reduceOnly": reduce_only}
        params.update(self.extra_params)
        if self.hedge_mode and not reduce_only:
            params["positionSide"] = "LONG" if side == Side.LONG else "SHORT"
        if self.exchange_name == "okx":
            params.setdefault("tdMode", "cross")
        return params

    @staticmethod
    def _normalize_order(o: dict[str, Any]) -> dict[str, Any]:
        """ccxt returns a unified dict; we project the fields downstream uses."""
        return {
            "id": str(o.get("id") or o.get("info", {}).get("orderId") or ""),
            "symbol": o.get("symbol"),
            "side": (o.get("side") or "").lower(),
            "size": float(o.get("amount") or o.get("filled") or 0.0),
            "price": float(o.get("price") or 0.0),
            "average": float(o.get("average") or o.get("price") or 0.0),
            "stop_price": float(o.get("stopPrice") or
                                 (o.get("info") or {}).get("stopPrice") or 0.0),
            "status": (o.get("status") or "").lower(),
            "reduce_only": bool(o.get("reduceOnly") or
                                 (o.get("info") or {}).get("reduceOnly") or False),
            "filled": float(o.get("filled") or 0.0),
            "amount": float(o.get("amount") or 0.0),
            # Surface the clientOrderId we (or the venue) attached so the
            # executor can log it and reconcile against retried calls.
            "client_order_id": str(
                o.get("clientOrderId")
                or (o.get("info") or {}).get("clientOrderId")
                or (o.get("info") or {}).get("clOrdId")
                or (o.get("info") or {}).get("orderLinkId")
                or "",
            ),
        }

    # ---------------- transient retry wrapper ---------------- #

    async def _with_retry(
        self,
        op_name: str,
        coro_factory: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Run ``coro_factory()`` with full-jitter exponential backoff on
        transient failures, bounded by ``total_timeout_sec``.

        The contract this provides:

          * Logical errors (InvalidOrder, InsufficientFunds, BadSymbol,
            AuthenticationError, MarginModeAlreadySet) are re-raised
            immediately — retrying them won't help and would just delay
            failure visibility.
          * Transient errors (NetworkError, RateLimitExceeded, 502/504,
            connection reset, timed out) are retried up to
            ``transient_retries`` times with full-jitter backoff in
            ``[0, retry_base_delay_sec * 2**attempt]``.
          * After the per-call ``total_timeout_sec`` deadline, the last
            transient error is re-raised. Hot path catches it and goes
            through the SR-2 fail-closed branch (orders_rejected += 1
            on entry; emergency-close on stop placement).
          * The whole sequence runs inside ``asyncio.wait_for`` so a
            hung TCP socket inside ccxt cannot deadlock the executor.
        """
        deadline = asyncio.get_event_loop().time() + self.total_timeout_sec

        async def _runner() -> _T:
            attempt = 0
            last_exc: BaseException | None = None
            while True:
                try:
                    return await coro_factory()
                except BaseException as e:  # noqa: BLE001 — narrowed below
                    if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt,
                                       SystemExit)):
                        raise
                    if not _is_transient_error(e):
                        raise
                    last_exc = e
                    attempt += 1
                    if attempt > self.transient_retries:
                        logger.error(
                            "%s exhausted %d retries — last error: %s",
                            op_name, self.transient_retries, e,
                        )
                        raise
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        logger.error(
                            "%s deadline reached before retry %d — last: %s",
                            op_name, attempt, e,
                        )
                        raise
                    cap = min(
                        self.retry_max_delay_sec,
                        self.retry_base_delay_sec * (2 ** (attempt - 1)),
                    )
                    sleep_s = min(remaining, random.uniform(0.0, cap))
                    logger.warning(
                        "%s transient %s (attempt %d/%d) — sleeping %.3fs",
                        op_name, type(e).__name__, attempt,
                        self.transient_retries, sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
            # Unreachable — exists for type-checkers.
            assert last_exc is not None
            raise last_exc

        try:
            return await asyncio.wait_for(_runner(), timeout=self.total_timeout_sec)
        except asyncio.TimeoutError as e:
            logger.error("%s exceeded total_timeout_sec=%.2f", op_name,
                         self.total_timeout_sec)
            raise RuntimeError(
                f"{op_name} total_timeout_exceeded:{self.total_timeout_sec}",
            ) from e

    # ---------------- ExchangeAdapter Protocol ---------------- #

    async def market_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params = self._entry_params(side, reduce_only)
        # Audit-fix #E1: every market order — entry, reduce-only close,
        # emergency close — gets a clientOrderId. Caller-supplied IDs win
        # so the executor can derive deterministic IDs from trace_id.
        coid = client_order_id or _derive_client_oid(
            prefix="r" if reduce_only else "e", trace_id=None,
        )
        _inject_client_oid(params, self.exchange_name, coid)

        async def _call() -> dict[str, Any]:
            return await self.client.create_market_order(
                symbol, side.value, size, params=params,
            )

        op = f"market_order({symbol},{side.value},{size},reduce={reduce_only})"
        resp = await self._with_retry(op, _call)
        normalized = self._normalize_order(resp)
        # Surface the *requested* coid even if the venue stripped it from
        # the response so the executor can log it.
        normalized["client_order_id"] = (
            normalized.get("client_order_id") or coid
        )
        logger.info("market %s %s %s ccxt_id=%s coid=%s avg=%s",
                    side.value, size, symbol,
                    resp.get("id"), coid,
                    resp.get("average") or resp.get("price"))
        return normalized

    async def place_stop_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params = self._stop_params(side, stop_price, reduce_only)
        coid = client_order_id or _derive_client_oid(
            prefix="s", trace_id=None,
        )
        _inject_client_oid(params, self.exchange_name, coid)
        # Order type "stop_market" is unified across most ccxt venues, but
        # binance accepts "STOP_MARKET" via params; we prefer the unified
        # form when the venue supports it.
        order_type = "stop_market" if self.exchange_name != "binance" else "STOP_MARKET"

        async def _call() -> dict[str, Any]:
            return await self.client.create_order(
                symbol, order_type, side.value, size, price=None, params=params,
            )

        op = f"place_stop_order({symbol},{side.value},{size},@{stop_price})"
        resp = await self._with_retry(op, _call)
        normalized = self._normalize_order(resp)
        normalized["client_order_id"] = (
            normalized.get("client_order_id") or coid
        )
        logger.info("stop %s %s %s @ %s ccxt_id=%s coid=%s",
                    side.value, size, symbol, stop_price,
                    resp.get("id"), coid)
        return normalized

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        async def _call() -> dict[str, Any]:
            return await self.client.cancel_order(order_id, symbol)

        op = f"cancel_order({order_id},{symbol})"
        resp = await self._with_retry(op, _call)
        return self._normalize_order(resp)

    async def set_leverage(self, symbol: str, leverage: float) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.exchange_name == "okx":
            params = {"mgnMode": "cross"}
        try:
            resp = await self.client.set_leverage(leverage, symbol, params=params)
            return {"symbol": symbol, "leverage": leverage, "raw": resp}
        except Exception as e:
            # Many venues no-op when leverage is already at the requested
            # value but raise; treat that path as success.
            msg = str(e)
            if "no need to" in msg.lower() or "no change" in msg.lower():
                logger.info("set_leverage no-op (%s @ %sx already): %s",
                            symbol, leverage, e)
                return {"symbol": symbol, "leverage": leverage, "raw": "no-op"}
            raise

    async def fetch_positions(self) -> list[dict[str, Any]]:
        raw = await self.client.fetch_positions()
        out: list[dict[str, Any]] = []
        for p in raw or []:
            size = abs(float(p.get("contracts") or p.get("contractSize") or 0.0))
            if size <= 0:
                continue
            out.append({
                "symbol": p.get("symbol"),
                "side": (p.get("side") or "").lower(),
                "contracts": size,
                "entryPrice": float(p.get("entryPrice") or 0.0),
                "info": p.get("info") or {},
            })
        return out

    async def fetch_ticker_price(self, symbol: str) -> float:
        """Best-effort live price for the dynamic-slippage check (SR-1).

        Bug #2 fix: ``RiskGate`` previously received the ``trigger_price``
        as both the trigger AND the ``current_price``, so the slippage
        comparison was always identity-zero. We now query the venue for a
        fresh mark/last price and let the gate compare against it.

        Preference order: ``last`` -> ``markPrice`` -> ``info.markPrice``
        -> ``close``. ``ccxt`` populates ``last`` on every venue we ship.

        Raises ``RuntimeError`` if no usable price is available -- the
        caller is expected to fail-closed (skip the order) rather than
        fall back silently to the stale trigger.
        """
        if not hasattr(self.client, "fetch_ticker"):
            raise RuntimeError(
                f"{self.exchange_name} client has no fetch_ticker",
            )
        t = await self.client.fetch_ticker(symbol)  # type: ignore[attr-defined]
        if not isinstance(t, dict):
            raise RuntimeError(f"unexpected ticker shape for {symbol}: {t!r}")
        for key in ("last", "markPrice"):
            v = t.get(key)
            if v is not None:
                price = float(v)
                if price > 0:
                    return price
        info = t.get("info")
        if isinstance(info, dict):
            v = info.get("markPrice")
            if v is not None:
                price = float(v)
                if price > 0:
                    return price
        v = t.get("close")
        if v is not None:
            price = float(v)
            if price > 0:
                return price
        raise RuntimeError(f"no usable price in ticker for {symbol}: {t!r}")

    async def fetch_top_depth_usdt(
        self, symbol: str, *, levels: int = 5,
    ) -> float:
        """Estimate top-N order-book depth in USDT (sum of both sides).

        Bug C4 fix: ``RiskGate.evaluate`` previously got this value
        hard-coded to ``cfg.min_liquidity_usdt`` from the hot path,
        which equates "we don't know" with "exactly the floor" —
        SR-2's liquidity gate then approves every signal at the floor.

        We sum ``price * size`` over the first ``levels`` rows of bids
        and asks and return the total. Raises ``RuntimeError`` if the
        venue doesn't return a usable book; the caller MUST fail-closed.
        """
        if not hasattr(self.client, "fetch_order_book"):
            raise RuntimeError(
                f"{self.exchange_name} client has no fetch_order_book",
            )
        ob = await self.client.fetch_order_book(  # type: ignore[attr-defined]
            symbol, levels,
        )
        if not isinstance(ob, dict):
            raise RuntimeError(f"unexpected order-book shape for {symbol}: {ob!r}")
        total = 0.0
        for side_key in ("bids", "asks"):
            rows = ob.get(side_key) or []
            for row in rows[:levels]:
                # ccxt rows: [price, size, ...]
                if not row or len(row) < 2:
                    continue
                try:
                    price = float(row[0])
                    size = float(row[1])
                except (TypeError, ValueError):
                    continue
                if price > 0 and size > 0:
                    total += price * size
        if total <= 0:
            raise RuntimeError(f"empty order book for {symbol}")
        return total

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        raw = await self.client.fetch_open_orders()
        out: list[dict[str, Any]] = []
        for o in raw or []:
            out.append({
                "id": str(o.get("id") or ""),
                "symbol": o.get("symbol"),
                "type": (o.get("type") or "").lower(),
                "reduceOnly": bool(o.get("reduceOnly") or
                                    (o.get("info") or {}).get("reduceOnly") or
                                    False),
                "side": (o.get("side") or "").lower(),
            })
        return out


# ------------------------------------------------------------------ #
# Convenience constructor with safety belts
# ------------------------------------------------------------------ #


def build_ccxt_adapter(
    *,
    exchange_name: str,
    api_key: str,
    api_secret: str,
    api_passphrase: str | None = None,
    testnet: bool = True,
    hedge_mode: bool = False,
) -> CCXTExchangeAdapter:
    """Construct a ccxt.pro client + adapter with sane defaults.

    Raises ImportError if ccxt is not installed (so tests don't need it).
    """
    try:
        import ccxt.pro as ccxtpro  # noqa: WPS433 — lazy on purpose
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ccxt is required for live trading. Install with `pip install ccxt`."
        ) from e

    klass = getattr(ccxtpro, exchange_name, None)
    if klass is None:
        raise ValueError(f"ccxt.pro does not support exchange {exchange_name!r}")

    config: dict[str, Any] = {
        "enableRateLimit": True,
        "apiKey": api_key,
        "secret": api_secret,
        "options": {
            "defaultType": "swap" if exchange_name in ("binance", "okx", "gateio") else None,
            "fetchMarkets": ["swap"],
        },
    }
    if api_passphrase:
        config["password"] = api_passphrase

    client = klass(config)
    if testnet:
        client.set_sandbox_mode(True)

    return CCXTExchangeAdapter(
        client=client, exchange_name=exchange_name, hedge_mode=hedge_mode,
    )
