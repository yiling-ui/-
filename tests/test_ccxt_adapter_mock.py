"""Mock tests for the live CCXTExchangeAdapter."""

from __future__ import annotations

import pytest

from altcoin_agent.risk.ccxt_adapter import CCXTExchangeAdapter, _CCXTLike
from altcoin_agent.risk.state import Side


class FakeCCXT:
    """In-memory fake satisfying the _CCXTLike Protocol."""

    def __init__(self) -> None:
        self.market_calls: list[dict] = []
        self.order_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self.leverage_calls: list[dict] = []
        self.next_id = 0

    def _id(self) -> str:
        self.next_id += 1
        return f"ccxt-{self.next_id}"

    async def create_market_order(self, symbol, side, amount, *, params=None):  # noqa: ANN001
        self.market_calls.append({"symbol": symbol, "side": side,
                                    "amount": amount, "params": params or {}})
        return {"id": self._id(), "symbol": symbol, "side": side,
                "amount": amount, "average": 1.5, "filled": amount,
                "status": "closed"}

    async def create_order(self, symbol, type, side, amount, *,  # noqa: A002
                            price=None, params=None):  # noqa: ANN001
        self.order_calls.append({
            "symbol": symbol, "type": type, "side": side,
            "amount": amount, "price": price, "params": params or {},
        })
        return {"id": self._id(), "symbol": symbol, "side": side,
                "amount": amount, "stopPrice": (params or {}).get("stopPrice"),
                "info": {"reduceOnly": (params or {}).get("reduceOnly", False)},
                "status": "open"}

    async def cancel_order(self, id, symbol=None, params=None):  # noqa: A002, ANN001
        self.cancel_calls.append({"id": id, "symbol": symbol})
        return {"id": id, "status": "cancelled"}

    async def set_leverage(self, leverage, symbol, params=None):  # noqa: ANN001
        self.leverage_calls.append({"leverage": leverage, "symbol": symbol,
                                     "params": params or {}})
        return {"leverage": leverage, "symbol": symbol}

    async def fetch_positions(self, symbols=None, params=None):  # noqa: ANN001
        return [{"symbol": "RAVEUSDT", "side": "long", "contracts": 100.0,
                 "entryPrice": 1.0}]

    async def fetch_open_orders(self, symbol=None, params=None):  # noqa: ANN001
        return [{"id": "x1", "symbol": "RAVEUSDT", "type": "stop_market",
                 "reduceOnly": True, "side": "sell"}]


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeCCXT(), _CCXTLike)


@pytest.mark.asyncio
async def test_market_order_long_passes_reduce_only_flag() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    out = await a.market_order("RAVEUSDT", Side.LONG, 10.0)
    assert out["id"].startswith("ccxt-")
    assert cli.market_calls[0]["side"] == "long"
    assert cli.market_calls[0]["params"]["reduceOnly"] is False


@pytest.mark.asyncio
async def test_place_stop_order_uses_binance_native_type() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.place_stop_order("RAVEUSDT", Side.SHORT, 10.0, stop_price=0.95)
    call = cli.order_calls[0]
    assert call["type"] == "STOP_MARKET"
    assert call["params"]["reduceOnly"] is True
    assert call["params"]["stopPrice"] == 0.95


@pytest.mark.asyncio
async def test_place_stop_order_uses_unified_type_on_okx() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="okx")
    await a.place_stop_order("RAVE-USDT-SWAP", Side.LONG, 10.0, stop_price=0.95)
    call = cli.order_calls[0]
    assert call["type"] == "stop_market"
    # OKX requires tdMode default
    assert call["params"]["tdMode"] == "cross"


@pytest.mark.asyncio
async def test_hedge_mode_sets_position_side() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance", hedge_mode=True)
    await a.market_order("RAVEUSDT", Side.LONG, 10.0)
    assert cli.market_calls[0]["params"]["positionSide"] == "LONG"
    # And on the closing stop side, positionSide should be the OPPOSITE of the
    # stop's order side (because the stop closes a LONG -> sell).
    await a.place_stop_order("RAVEUSDT", Side.SHORT, 10.0, stop_price=0.95)
    assert cli.order_calls[0]["params"]["positionSide"] == "LONG"


@pytest.mark.asyncio
async def test_set_leverage_swallows_no_op_error() -> None:
    class FlakyCli(FakeCCXT):
        async def set_leverage(self, leverage, symbol, params=None):  # noqa: ANN001
            raise RuntimeError("no need to change leverage")
    a = CCXTExchangeAdapter(client=FlakyCli(), exchange_name="binance")
    out = await a.set_leverage("RAVEUSDT", 10)
    assert out["leverage"] == 10
    assert out["raw"] == "no-op"


@pytest.mark.asyncio
async def test_set_leverage_propagates_real_errors() -> None:
    class FlakyCli(FakeCCXT):
        async def set_leverage(self, leverage, symbol, params=None):  # noqa: ANN001
            raise RuntimeError("rate limit")
    a = CCXTExchangeAdapter(client=FlakyCli(), exchange_name="binance")
    with pytest.raises(RuntimeError, match="rate limit"):
        await a.set_leverage("RAVEUSDT", 10)


@pytest.mark.asyncio
async def test_fetch_positions_filters_zero_size() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    positions = await a.fetch_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "RAVEUSDT"
    assert positions[0]["contracts"] == 100.0


@pytest.mark.asyncio
async def test_fetch_open_orders_normalizes_reduce_only() -> None:
    cli = FakeCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    orders = await a.fetch_open_orders()
    assert orders[0]["reduceOnly"] is True
    assert orders[0]["type"] == "stop_market"
