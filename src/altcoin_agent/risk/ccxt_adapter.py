"""ccxt_adapter.py — Live ExchangeAdapter backed by ccxt.pro.

Wraps a ccxt.pro client (Binance/OKX/Gate.io USDT-M perpetuals) into the
``ExchangeAdapter`` Protocol consumed by ``CCXTExecutor``. The adapter:

  * normalizes ccxt's heterogeneous return shapes into the small dict we use,
  * sets ``positionSide`` correctly for hedge-mode-enabled accounts,
  * handles testnet via the ``sandbox`` flag,
  * routes STOP_MARKET orders through the right ``params`` per venue,
  * **TICKET-001**: injects a venue-specific ``clientOrderId`` on every
    write call so retries are idempotent. The cid is a 26-char
    alphanumeric string starting with a letter (the strictest common
    prefix across Binance/OKX/Gate.io). Gate.io requires the additional
    ``"t-"`` prefix on its ``text`` field; we add it inside
    ``_inject_client_order_id`` so the executor doesn't need to know.
  * **TICKET-002**: ``_normalize_order`` returns the full set of fields
    the executor's partial-fill detection needs (``amount`` /
    ``filled`` / ``remaining`` / ``status`` / ``client_order_id``).
    The legacy ``size`` key is preserved for back-compat with callers
    that haven't been migrated.
  * **TICKET-005**: every public network method is wrapped in a
    ``RetryPolicy`` that classifies ccxt errors into transient
    (retry with backoff), fatal (re-raise immediately) and one-shot
    (retry once). See ``risk/retry.py`` for the classification table.
  * never silently swallows errors from the exchange — Risk Gate / Executor
    handle every error explicitly.

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

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from altcoin_agent.risk.retry import RetryPolicy
from altcoin_agent.risk.state import Side

logger = logging.getLogger(__name__)


# TICKET-001: per-venue client-order-id parameter names. The names are
# what the venue's REST API expects on the WRITE side. ``_extract`` is
# how we read it back from the heterogeneous response shapes ccxt
# returns. Keeping the two together so adding a new venue is one
# self-contained edit.
_CID_WRITE_PARAM = {
    "binance": "newClientOrderId",
    "binanceusdm": "newClientOrderId",
    "binancecoinm": "newClientOrderId",
    "okx": "clOrdId",
    "gateio": "text",         # gate.io requires "t-" prefix
    "bybit": "orderLinkId",
    "bitget": "clientOid",
}

_CID_READ_KEYS = (
    "clientOrderId",
    "clientOid",
    "orderLinkId",
    "newClientOrderId",
    "clOrdId",
    "text",
    "c",                       # binance some endpoints use "c"
)


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
        exchange_name: "binance" / "okx" / "gateio".
        hedge_mode: when True, sets ``positionSide`` on each order so the
            two directions don't share a single net position. Required for
            the SHORT path on Binance USDT-M futures unless the account is
            in one-way mode.
        retry_policy: TICKET-005. ``None`` -> default policy; tests can
            inject a deterministic policy with a stub sleeper. Wraps
            every public network method.
    """

    client: _CCXTLike
    exchange_name: str = "binance"
    hedge_mode: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)
    retry_policy: RetryPolicy | None = None

    def __post_init__(self) -> None:
        if self.retry_policy is None:
            self.retry_policy = RetryPolicy()

    # ---------------- helpers ---------------- #

    def _stop_params(
        self,
        side: Side,
        stop_price: float,
        reduce_only: bool,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
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
        if client_order_id:
            self._inject_client_order_id(params, client_order_id)
        return params

    def _entry_params(
        self,
        side: Side,
        reduce_only: bool,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"reduceOnly": reduce_only}
        params.update(self.extra_params)
        if self.hedge_mode and not reduce_only:
            params["positionSide"] = "LONG" if side == Side.LONG else "SHORT"
        if self.exchange_name == "okx":
            params.setdefault("tdMode", "cross")
        if client_order_id:
            self._inject_client_order_id(params, client_order_id)
        return params

    def _inject_client_order_id(
        self, params: dict[str, Any], cid: str,
    ) -> None:
        """TICKET-001: write the cid into the venue-specific params slot.

        Gate.io's ``text`` field has special-case rules: it must start
        with ``t-`` followed by 1..28 alphanumeric chars (or ``_``,
        ``-``, ``.``). We add the prefix here so the executor can pass
        the same cid format to every venue.
        """
        param_name = _CID_WRITE_PARAM.get(
            self.exchange_name.lower(), "clientOrderId",
        )
        value = cid
        if self.exchange_name.lower() == "gateio":
            # Strip t- if the caller already prefixed (idempotent), then
            # re-add. Total length stays <= 28.
            if value.startswith("t-"):
                value = value[2:]
            value = f"t-{value[:28]}"
        params[param_name] = value
        # Also mirror to the unified ccxt key so adapters in
        # rest-fallback mode pick it up too. Harmless on venues that
        # ignore unrecognised params.
        params.setdefault("clientOrderId", cid)

    @staticmethod
    def _extract_client_order_id(o: dict[str, Any]) -> str | None:
        """Read the cid back from any of the known top-level / info.* slots."""
        for key in _CID_READ_KEYS:
            v = o.get(key)
            if v:
                return str(v)
        info = o.get("info") or {}
        if isinstance(info, dict):
            for key in _CID_READ_KEYS:
                v = info.get(key)
                if v:
                    return str(v)
        return None

    @classmethod
    def _normalize_order(cls, o: dict[str, Any]) -> dict[str, Any]:
        """ccxt returns a unified dict; we project the fields downstream uses.

        TICKET-002: the projection now exposes the full set of fields
        the executor's partial-fill detection relies on. Pre-fix this
        method merged ``amount`` and ``filled`` into a single ``size``
        key, which made the executor's ``fill_ratio`` calculation
        always equal 1.0 in production (the ``get("filled")`` /
        ``get("amount")`` chain inside ``open()`` always missed and
        fell through to ``decision.size``). The output schema is now:

            id                — venue order id
            client_order_id   — TICKET-001 cid (None for legacy adapters)
            symbol, side, status, reduce_only — unchanged
            amount            — requested size (from ccxt's "amount" or
                                 falling back to "filled" / "size" /
                                 the request size when the venue echoes
                                 nothing)
            filled            — actually-filled qty (0.0 when missing)
            remaining         — amount - filled (clamped >= 0)
            average           — VWAP fill price; ``price`` retained
                                 separately for caller convenience
            stop_price        — for STOP_MARKET orders
            size              — LEGACY: equals ``amount`` for back-compat
                                 with code paths that haven't migrated
        """
        amount = _safe_float(
            o.get("amount"),
            o.get("filled"),
            o.get("remaining"),
            (o.get("info") or {}).get("origQty"),
            (o.get("info") or {}).get("size"),
            default=0.0,
        )
        filled = _safe_float(
            o.get("filled"),
            (o.get("info") or {}).get("executedQty"),
            (o.get("info") or {}).get("filled_size"),
            (o.get("info") or {}).get("accFillSz"),
            default=0.0,
        )
        remaining_raw = o.get("remaining")
        if remaining_raw is None:
            remaining = max(0.0, amount - filled)
        else:
            remaining = _safe_float(remaining_raw, default=max(0.0, amount - filled))
        status = (o.get("status") or "").lower()
        # Heuristic to flag closed orders that ccxt left as "" (some
        # adapters omit the field but populate fully-filled qty).
        if not status and amount > 0 and filled >= amount - 1e-12:
            status = "closed"
        return {
            "id": str(o.get("id") or o.get("info", {}).get("orderId") or ""),
            "client_order_id": cls._extract_client_order_id(o),
            "symbol": o.get("symbol"),
            "side": (o.get("side") or "").lower(),
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "size": amount,                          # LEGACY back-compat
            "price": _safe_float(o.get("price"), default=0.0),
            "average": _safe_float(
                o.get("average"), o.get("price"), default=0.0,
            ),
            "stop_price": _safe_float(
                o.get("stopPrice"),
                (o.get("info") or {}).get("stopPrice"),
                default=0.0,
            ),
            "status": status,
            "reduce_only": bool(
                o.get("reduceOnly")
                or (o.get("info") or {}).get("reduceOnly")
                or False,
            ),
        }

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
        params = self._entry_params(
            side, reduce_only, client_order_id=client_order_id,
        )

        async def _call() -> dict[str, Any]:
            return await self.client.create_market_order(
                symbol, side.value, size, params=params,
            )

        async def _idem_check() -> dict[str, Any] | None:
            if not client_order_id:
                return None
            return await self._fetch_order_safely(
                symbol=symbol, client_order_id=client_order_id,
            )

        resp = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call,
            op_name=f"market_order:{symbol}",
            idempotency_check=_idem_check,
        )
        normalised = self._normalize_order(resp)
        logger.info(
            "market %s %s %s ccxt_id=%s cid=%s avg=%s status=%s filled=%s",
            side.value, size, symbol,
            normalised["id"], normalised["client_order_id"],
            normalised["average"], normalised["status"],
            normalised["filled"],
        )
        return normalised

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
        params = self._stop_params(
            side, stop_price, reduce_only, client_order_id=client_order_id,
        )
        # Order type "stop_market" is unified across most ccxt venues, but
        # binance accepts "STOP_MARKET" via params; we prefer the unified
        # form when the venue supports it.
        order_type = (
            "stop_market" if self.exchange_name != "binance" else "STOP_MARKET"
        )

        async def _call() -> dict[str, Any]:
            return await self.client.create_order(
                symbol, order_type, side.value, size,
                price=None, params=params,
            )

        async def _idem_check() -> dict[str, Any] | None:
            if not client_order_id:
                return None
            return await self._fetch_order_safely(
                symbol=symbol, client_order_id=client_order_id,
            )

        resp = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call,
            op_name=f"place_stop_order:{symbol}",
            idempotency_check=_idem_check,
        )
        normalised = self._normalize_order(resp)
        logger.info(
            "stop %s %s %s @ %s ccxt_id=%s cid=%s",
            side.value, size, symbol, stop_price,
            normalised["id"], normalised["client_order_id"],
        )
        return normalised

    async def cancel_order(
        self, order_id: str, symbol: str,
    ) -> dict[str, Any]:
        async def _call() -> dict[str, Any]:
            return await self.client.cancel_order(order_id, symbol)

        resp = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name=f"cancel_order:{symbol}",
        )
        return self._normalize_order(resp)

    async def set_leverage(
        self, symbol: str, leverage: float,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.exchange_name == "okx":
            params = {"mgnMode": "cross"}

        async def _call() -> dict[str, Any]:
            try:
                resp = await self.client.set_leverage(
                    leverage, symbol, params=params,
                )
                return {"symbol": symbol, "leverage": leverage, "raw": resp}
            except Exception as e:
                # Many venues no-op when leverage is already at the
                # requested value but raise; treat that path as success.
                # We do this BEFORE the retry policy sees it because
                # "no need to change" is not a transient error -- it
                # is success in disguise.
                msg = str(e)
                if (
                    "no need to" in msg.lower()
                    or "no change" in msg.lower()
                ):
                    logger.info(
                        "set_leverage no-op (%s @ %sx already): %s",
                        symbol, leverage, e,
                    )
                    return {
                        "symbol": symbol, "leverage": leverage, "raw": "no-op",
                    }
                raise

        return await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name=f"set_leverage:{symbol}",
        )

    async def fetch_positions(self) -> list[dict[str, Any]]:
        async def _call() -> list[dict[str, Any]]:
            return await self.client.fetch_positions()

        raw = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name="fetch_positions",
        )
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

        async def _call() -> dict[str, Any]:
            return await self.client.fetch_ticker(symbol)  # type: ignore[attr-defined]

        t = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name=f"fetch_ticker:{symbol}",
        )
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

        async def _call() -> dict[str, Any]:
            return await self.client.fetch_order_book(  # type: ignore[attr-defined]
                symbol, levels,
            )

        ob = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name=f"fetch_order_book:{symbol}",
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
        async def _call() -> list[dict[str, Any]]:
            return await self.client.fetch_open_orders()

        raw = await self.retry_policy.execute(  # type: ignore[union-attr]
            _call, op_name="fetch_open_orders",
        )
        out: list[dict[str, Any]] = []
        for o in raw or []:
            out.append({
                "id": str(o.get("id") or ""),
                "client_order_id": self._extract_client_order_id(o),
                "symbol": o.get("symbol"),
                "type": (o.get("type") or "").lower(),
                "reduceOnly": bool(o.get("reduceOnly") or
                                    (o.get("info") or {}).get("reduceOnly") or
                                    False),
                "side": (o.get("side") or "").lower(),
            })
        return out

    # ---------------- TICKET-001 / 004 idempotency probes ---------------- #

    async def fetch_order(
        self,
        *,
        symbol: str,
        client_order_id: str | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Look up an order by cid or venue id. Returns None when not found.

        TICKET-001: this is the post-retry idempotency probe. After a
        ``NetworkError`` on ``create_order`` we don't know whether the
        request reached the venue; calling ``fetch_order`` with the
        same ``client_order_id`` resolves the ambiguity. ccxt's
        ``fetch_order`` accepts ``params={"clientOrderId": ...}`` on
        most venues; some require a separate ``fetch_order_by_client_id``
        method. We try the unified call first, then fall back.

        Returns the same normalized shape as ``_normalize_order``, or
        None when the order does not exist on the venue.
        """
        if not (client_order_id or order_id):
            return None
        if not hasattr(self.client, "fetch_order"):
            return None

        params: dict[str, Any] = {}
        if client_order_id and not order_id:
            # Different venues use different params slots; set them all.
            cid_params = self._cid_query_params(client_order_id)
            params.update(cid_params)

        try:
            resp = await self.client.fetch_order(  # type: ignore[attr-defined]
                order_id or client_order_id, symbol, params=params,
            )
        except Exception as e:
            name = type(e).__name__
            # OrderNotFound is the expected "really wasn't placed" case.
            if name in ("OrderNotFound", "InvalidOrder"):
                return None
            # On any other failure surface None — the caller will retry
            # the original write and live with potential duplication
            # (which is also short-circuited by the venue's own cid
            # uniqueness check at create time).
            logger.debug(
                "fetch_order(symbol=%s, cid=%s) failed: %s — treating as not-found",
                symbol, client_order_id, e,
            )
            return None
        if not isinstance(resp, dict):
            return None
        return self._normalize_order(resp)

    async def _fetch_order_safely(
        self,
        *,
        symbol: str,
        client_order_id: str,
    ) -> dict[str, Any] | None:
        """Inner wrapper used as ``RetryPolicy.idempotency_check``.

        We only return a non-None value when the prior call clearly
        succeeded (status indicates the order exists on the venue).
        ``fetch_order`` returning a record with status=="closed" /
        "open" / "filled" / "partially_filled" is "yes, the prior write
        landed". Anything else -> None (let the retry happen).
        """
        rec = await self.fetch_order(
            symbol=symbol, client_order_id=client_order_id,
        )
        if rec is None:
            return None
        status = (rec.get("status") or "").lower()
        # "canceled" / "rejected" / "expired" mean the prior call
        # reached the venue but the order is dead — re-issuing would
        # not help (a fresh cid would be needed). Treat as terminal.
        if status in ("closed", "filled", "open", "partially_filled",
                       "canceled", "rejected", "expired"):
            return rec
        return None

    def _cid_query_params(self, cid: str) -> dict[str, Any]:
        """Build a params dict that asks for the order by cid across the
        venue-specific param-name flavours ccxt's ``fetch_order`` accepts."""
        out: dict[str, Any] = {"clientOrderId": cid}
        name = self.exchange_name.lower()
        if name in ("binance", "binanceusdm", "binancecoinm"):
            out["origClientOrderId"] = cid
        elif name == "okx":
            out["clOrdId"] = cid
            # Strip the "t-" prefix that gateio adds, in case caller
            # passed in the prefixed form.
        elif name == "gateio":
            stripped = cid[2:] if cid.startswith("t-") else cid
            out["text"] = f"t-{stripped[:28]}"
        elif name == "bybit":
            out["orderLinkId"] = cid
        elif name == "bitget":
            out["clientOid"] = cid
        return out

    async def fetch_my_trades(
        self,
        *,
        symbol: str,
        since_ms: int,
        client_order_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """TICKET-004: trades for ``symbol`` since ``since_ms``.

        Used by ``main.App._on_position_close`` to compute the real
        VWAP fill price (instead of substituting ``position.current_stop``
        which is the *expected* stop fill, not the realised one).

        When ``client_order_id`` is supplied, only trades belonging to
        that order are returned. Otherwise all trades since ``since_ms``
        are returned and the caller filters by ``order_id`` /
        ``client_order_id`` on its side.

        ccxt unifies ``fetch_my_trades`` across all our venues. When the
        client does not have the method (older ccxt, mock adapter), we
        return an empty list so the caller can fall back to the
        legacy stop-price proxy.

        Each returned trade has the canonical ccxt shape:

            {
                "id": "...",                  # trade id
                "order": "...",               # parent order id
                "client_order_id": "...",     # parent cid (when known)
                "symbol": "BTC/USDT:USDT",
                "side": "buy"|"sell",
                "amount": float,              # base qty filled
                "price": float,               # fill price
                "cost": float,                # quote qty (price*amount)
                "timestamp": int,             # ms
                "fee": {...},                 # optional
            }
        """
        if not hasattr(self.client, "fetch_my_trades"):
            return []

        params: dict[str, Any] = {}
        if client_order_id:
            params.update(self._cid_query_params(client_order_id))

        async def _call() -> list[dict[str, Any]]:
            return await self.client.fetch_my_trades(  # type: ignore[attr-defined]
                symbol, since_ms, limit, params=params,
            )

        try:
            raw = await self.retry_policy.execute(  # type: ignore[union-attr]
                _call, op_name=f"fetch_my_trades:{symbol}",
            )
        except Exception as e:
            logger.warning(
                "fetch_my_trades(%s) failed: %s — caller will fall back",
                symbol, e,
            )
            return []
        out: list[dict[str, Any]] = []
        for t in raw or []:
            try:
                out.append({
                    "id": str(t.get("id") or ""),
                    "order": str(t.get("order") or ""),
                    "client_order_id": self._extract_client_order_id(t),
                    "symbol": t.get("symbol") or symbol,
                    "side": (t.get("side") or "").lower(),
                    "amount": float(t.get("amount") or 0.0),
                    "price": float(t.get("price") or 0.0),
                    "cost": float(t.get("cost") or 0.0),
                    "timestamp": int(t.get("timestamp") or 0),
                    "fee": t.get("fee"),
                    "info": t.get("info") or {},
                })
            except (TypeError, ValueError) as e:
                logger.debug("skip malformed trade: %s (%s)", t, e)
                continue
        # ccxt orders trades from oldest to newest; preserve.
        return out


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #


def _safe_float(*candidates: Any, default: float = 0.0) -> float:
    """First candidate that converts to a finite float wins.

    Treats None / "" / non-numeric as miss. Returns ``default`` when
    every candidate misses. Used by ``_normalize_order`` to gracefully
    handle the heterogeneous shapes ccxt returns across venues.
    """
    for c in candidates:
        if c is None:
            continue
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        # NaN / inf -> treat as miss too.
        if v != v or v in (float("inf"), float("-inf")):
            continue
        return v
    return default


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
