"""ccxt_adapter.py — Live ExchangeAdapter backed by ccxt.pro.

Wraps a ccxt.pro client (Binance/OKX/Gate.io USDT-M perpetuals) into the
``ExchangeAdapter`` Protocol consumed by ``CCXTExecutor``. The adapter:

  * normalizes ccxt's heterogeneous return shapes into the small dict we use,
  * sets ``positionSide`` correctly for hedge-mode-enabled accounts,
  * handles testnet via the ``sandbox`` flag,
  * routes STOP_MARKET orders through the right ``params`` per venue,
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

from altcoin_agent.risk.state import Side

logger = logging.getLogger(__name__)


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
    """

    client: _CCXTLike
    exchange_name: str = "binance"
    hedge_mode: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)

    # ---------------- helpers ---------------- #

    def _idempotency_param_key(self) -> str:
        """Return the venue-specific parameter name ccxt expects for
        client-side idempotency tokens.

        Phase B.1.1 — ccxt unifies most Binance / OKX / Gate.io
        params but leaves the idempotency key venue-specific:
            binance USDT-M futures: ``newClientOrderId``
            okx swap:               ``clOrdId``
            gateio swap:            ``text``  (Gate.io's client_oid
                                                 must start with ``t-`` and
                                                 the param is named ``text``;
                                                 the value is normalised
                                                 below to satisfy that).
        For unknown venues we default to ``newClientOrderId`` because
        most ccxt-supported futures venues now accept it, and the
        worst case (a venue that ignores unknown params) is the same
        as not setting it: no idempotency, but no crash.
        """
        return {
            "binance": "newClientOrderId",
            "binanceusdm": "newClientOrderId",
            "okx":     "clOrdId",
            "gateio":  "text",
            "gate":    "text",
        }.get(self.exchange_name, "newClientOrderId")

    def _normalize_client_order_id(self, coid: str) -> str:
        """Sanitize and (where required) prefix the token for the venue.

        Gate.io requires user-supplied IDs to start with ``t-`` —
        we add it if absent so callers don't have to track venue-
        specific quirks. Binance and OKX accept the raw token.
        """
        if self.exchange_name in ("gateio", "gate"):
            if not coid.startswith("t-"):
                return f"t-{coid}"[:30]
        return coid

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
        params = self._entry_params(side, reduce_only)
        # Phase B.1.1: stamp the venue-specific idempotency key.
        # ccxt's create_market_order forwards unknown params verbatim
        # to the venue, so this works even if a future ccxt version
        # adds native support for ``client_order_id`` (which would
        # then take precedence and we'd just be sending it twice
        # — harmless because the values match).
        if client_order_id is not None:
            key = self._idempotency_param_key()
            params[key] = self._normalize_client_order_id(client_order_id)
        resp = await self.client.create_market_order(
            symbol, side.value, size, params=params,
        )
        logger.info("market %s %s %s ccxt_id=%s avg=%s coid=%s",
                    side.value, size, symbol,
                    resp.get("id"), resp.get("average") or resp.get("price"),
                    client_order_id)
        return self._normalize_order(resp)

    async def place_stop_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params = self._stop_params(side, stop_price, reduce_only)
        if client_order_id is not None:
            key = self._idempotency_param_key()
            params[key] = self._normalize_client_order_id(client_order_id)
        # Order type "stop_market" is unified across most ccxt venues, but
        # binance accepts "STOP_MARKET" via params; we prefer the unified
        # form when the venue supports it.
        order_type = "stop_market" if self.exchange_name != "binance" else "STOP_MARKET"
        resp = await self.client.create_order(
            symbol, order_type, side.value, size, price=None, params=params,
        )
        logger.info("stop %s %s %s @ %s ccxt_id=%s coid=%s",
                    side.value, size, symbol, stop_price, resp.get("id"),
                    client_order_id)
        return self._normalize_order(resp)

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        resp = await self.client.cancel_order(order_id, symbol)
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

    async def fetch_total_usdt_balance(self) -> float:
        """Total USDT-equivalent balance for ``WithdrawalDetector``.

        For a USDT-margined perpetuals account we want a number that:
          * tracks the operator's bank-flow (deposits / withdrawals
            move it 1:1),
          * already includes unrealised PnL on open positions (so the
            detector's "unexplained delta" calculus only needs to net
            against *realised* PnL flowing through ``account``).

        Both Binance USDT-M futures and Gate.io USDT futures return a
        wallet-and-margin object via ``fetch_balance()``. The shape
        differs a little but ccxt unifies enough of it that we can
        prefer ``info.totalWalletBalance + info.totalUnrealizedProfit``
        on Binance, ``info.total_initial_margin + info.cross_wallet_balance``
        on Gate, and a generic ``USDT.total`` fallback elsewhere.

        Raises ``RuntimeError`` on transient errors so the
        ``WithdrawalDetector`` swallows the round and retries; never
        returns 0 on error (0 would look like a full withdrawal).
        """
        if not hasattr(self.client, "fetch_balance"):
            raise RuntimeError(
                f"{self.exchange_name} client has no fetch_balance",
            )
        bal = await self.client.fetch_balance()  # type: ignore[attr-defined]
        if not isinstance(bal, dict):
            raise RuntimeError(f"unexpected balance shape: {bal!r}")

        # Venue-specific extraction (defensively, with fallback).
        info = bal.get("info") or {}
        # Binance USDT-M futures: ``totalWalletBalance`` is the wallet
        # in USDT; ``totalUnrealizedProfit`` is unrealised PnL on
        # all open positions. Their sum is the "marginBalance" the
        # account-holder sees in the UI.
        if self.exchange_name == "binance":
            try:
                wallet = float(info.get("totalWalletBalance") or 0.0)
                upnl = float(info.get("totalUnrealizedProfit") or 0.0)
                total = wallet + upnl
                if total > 0:
                    return total
            except (TypeError, ValueError):
                pass
        if self.exchange_name == "gateio" or self.exchange_name == "gate":
            # Gate.io futures returns ``total`` and ``available`` per
            # currency; for cross-margin the wallet equity is in
            # ``info.cross_wallet_balance`` plus ``info.unrealised_pnl``.
            try:
                cross_wb = float(info.get("cross_wallet_balance") or 0.0)
                upnl = float(info.get("unrealised_pnl") or 0.0)
                total = cross_wb + upnl
                if total > 0:
                    return total
            except (TypeError, ValueError):
                pass

        # Generic fallback: ccxt unifies ``USDT.total`` to the
        # net-of-margin total balance for most venues.
        usdt = bal.get("USDT") or bal.get("usdt") or {}
        if isinstance(usdt, dict):
            for key in ("total", "free"):
                v = usdt.get(key)
                if v is not None:
                    try:
                        out = float(v)
                    except (TypeError, ValueError):
                        continue
                    if out > 0:
                        return out

        # ccxt sometimes puts the unified totals under ``total[USDT]``.
        totals = bal.get("total") or {}
        if isinstance(totals, dict):
            v = totals.get("USDT")
            if v is not None:
                try:
                    out = float(v)
                except (TypeError, ValueError):
                    out = 0.0
                if out > 0:
                    return out

        raise RuntimeError(
            f"no usable USDT balance in fetch_balance response on "
            f"{self.exchange_name}: keys={list(bal.keys())}",
        )


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
