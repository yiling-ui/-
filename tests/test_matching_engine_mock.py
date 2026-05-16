"""tests/test_matching_engine_mock.py — Phase B.4 in-memory exchange simulator."""

from __future__ import annotations

import pytest

from altcoin_agent.backtest.data_adapter import BacktestDataAdapter
from altcoin_agent.backtest.historical_loader import TIMEFRAME_MS, HistoricalDataLoader
from altcoin_agent.backtest.matching_engine import (
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_OPEN,
    MatchingEngine,
    MatchingEngineConfig,
)
from altcoin_agent.backtest.slippage_model import SlippageModel, SlippageParams
from altcoin_agent.risk.executor import ExchangeAdapter
from altcoin_agent.risk.state import Side

# ----------------------- fixture builders ----------------------- #


class _StubFetcher:
    def __init__(self, bars):
        self.bars = sorted(bars, key=lambda b: b[0])

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return [list(b) for b in self.bars if b[0] >= since][:limit]


def _build_engine(tmp_path, bars, *, slip_params: SlippageParams | None = None,
                  starting_balance: float = 10_000.0) -> tuple[MatchingEngine, BacktestDataAdapter]:
    loader = HistoricalDataLoader(
        fetcher=_StubFetcher(bars),
        cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0,
        sleep_fn=lambda *_: None,
    )
    loader.download(
        symbol="X", timeframe="1m",
        start_ms=int(bars[0][0]),
        end_ms=int(bars[-1][0]) + TIMEFRAME_MS["1m"],
    )
    adapter = BacktestDataAdapter(loader=loader, timeframe="1m")
    adapter.preload(
        symbols=["X"],
        start_ms=int(bars[0][0]),
        end_ms=int(bars[-1][0]) + TIMEFRAME_MS["1m"],
    )
    engine = MatchingEngine(
        data=adapter,
        slippage=SlippageModel(slip_params or SlippageParams(
            base_spread=0.0, impact_coeff=0.0, vol_premium_coeff=0.0,
            taker_fee=0.0,
        )),
        cfg=MatchingEngineConfig(
            default_top_depth_usdt=100_000.0, default_realized_vol=0.0,
        ),
        starting_balance_usdt=starting_balance,
    )
    return engine, adapter


def _flat_bars(n: int = 10, price: float = 100.0) -> list[list[float]]:
    step = TIMEFRAME_MS["1m"]
    return [[i * step, price, price, price, price, 100.0] for i in range(n)]


# ----------------------- protocol conformance ----------------------- #


def test_matching_engine_satisfies_exchange_adapter_protocol(tmp_path):
    engine, _ = _build_engine(tmp_path, _flat_bars())
    assert isinstance(engine, ExchangeAdapter)


# ----------------------- market orders + position ----------------------- #


@pytest.mark.asyncio
async def test_market_buy_then_sell_zero_slippage_pnl_zero(tmp_path):
    engine, adapter = _build_engine(tmp_path, _flat_bars(price=100.0))
    bars_iter = adapter.iter_bars("X")
    # Advance one bar then BUY.
    next(bars_iter)
    await engine.market_order("X", Side.LONG, size=1.0)
    # Advance another bar, sell flat.
    next(bars_iter)
    await engine.market_order("X", Side.SHORT, size=1.0, reduce_only=True)
    pos = await engine.fetch_positions()
    assert pos == []
    assert engine.realized_pnl_usdt == pytest.approx(0.0)
    assert engine.balance_usdt == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_long_then_close_in_profit_realises_correctly(tmp_path):
    bars = [
        [0,        100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000,   100.0, 100.0, 100.0, 100.0, 100.0],
        [120_000,  150.0, 150.0, 150.0, 150.0, 100.0],
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    next(it)
    await engine.market_order("X", Side.LONG, size=2.0)  # at 100
    next(it)
    await engine.market_order("X", Side.SHORT, size=2.0, reduce_only=True)  # at 150
    # Realised PnL = (150 - 100) * 2 = 100
    assert engine.realized_pnl_usdt == pytest.approx(100.0)
    assert engine.balance_usdt == pytest.approx(10_100.0)
    pos = await engine.fetch_positions()
    assert pos == []


@pytest.mark.asyncio
async def test_short_then_close_in_profit(tmp_path):
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000,   80.0,  80.0,  80.0,  80.0, 100.0],
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.SHORT, size=1.0)  # short at 100
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0, reduce_only=True)  # cover at 80
    # PnL = (100 - 80) * 1 = 20
    assert engine.realized_pnl_usdt == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_partial_close_preserves_remaining_size(tmp_path):
    bars = [[i * 60_000, 100.0, 100.0, 100.0, 100.0, 100.0] for i in range(3)]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=4.0)
    next(it)
    await engine.market_order("X", Side.SHORT, size=1.0, reduce_only=True)
    pos = await engine.fetch_positions()
    assert len(pos) == 1
    assert pos[0]["size"] == pytest.approx(3.0)
    assert pos[0]["side"] == "long"


@pytest.mark.asyncio
async def test_oversized_close_flips_position_when_not_reduce_only(tmp_path):
    bars = [[i * 60_000, 100.0, 100.0, 100.0, 100.0, 100.0] for i in range(3)]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=2.0)
    next(it)
    # Sell 3.0 — closes the 2.0 long, opens 1.0 short.
    await engine.market_order("X", Side.SHORT, size=3.0)  # not reduce_only
    pos = await engine.fetch_positions()
    assert len(pos) == 1
    assert pos[0]["side"] == "short"
    assert pos[0]["size"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_reduce_only_on_flat_position_is_noop(tmp_path):
    bars = _flat_bars(3)
    engine, adapter = _build_engine(tmp_path, bars)
    next(adapter.iter_bars("X"))
    fill = await engine.market_order(
        "X", Side.LONG, size=1.0, reduce_only=True,
    )
    assert fill["status"] == STATUS_FILLED
    pos = await engine.fetch_positions()
    assert pos == []


# ----------------------- fees + slippage ----------------------- #


@pytest.mark.asyncio
async def test_taker_fee_and_slippage_reduce_balance(tmp_path):
    engine, adapter = _build_engine(
        tmp_path, _flat_bars(price=100.0),
        slip_params=SlippageParams(
            base_spread=0.0010, impact_coeff=0.0, vol_premium_coeff=0.0,
            taker_fee=0.0004,
        ),
    )
    next(adapter.iter_bars("X"))
    await engine.market_order("X", Side.LONG, size=1.0)
    # Just opened: realised pnl 0, but fee paid 0.04 USDT.
    assert engine.fees_paid_usdt == pytest.approx(0.04, abs=1e-6)
    assert engine.balance_usdt < 10_000.0


# ----------------------- stop orders ----------------------- #


@pytest.mark.asyncio
async def test_long_protective_stop_triggers_on_low_breach(tmp_path):
    bars = [
        [0,       100.0, 101.0,  99.0, 100.0, 100.0],
        [60_000, 100.0, 100.0,  90.0, 95.0, 100.0],   # bar.low=90 crosses 95
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)  # filled at 100
    # Place a sell-stop at 95 (closes long position).
    stop = await engine.place_stop_order(
        "X", Side.SHORT, size=1.0, stop_price=95.0, reduce_only=True,
    )
    assert stop["status"] == STATUS_OPEN

    bar1 = next(it)
    triggered = engine.on_bar("X", bar1)
    assert len(triggered) == 1
    assert triggered[0]["fill_price"] == pytest.approx(95.0)
    pos = await engine.fetch_positions()
    assert pos == []  # closed
    # Realised PnL = (95 - 100) * 1 = -5.
    assert engine.realized_pnl_usdt == pytest.approx(-5.0)


@pytest.mark.asyncio
async def test_short_protective_stop_triggers_on_high_breach(tmp_path):
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000, 100.0, 110.0, 100.0, 105.0, 100.0],  # bar.high=110 crosses 105
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.SHORT, size=1.0)
    await engine.place_stop_order(
        "X", Side.LONG, size=1.0, stop_price=105.0, reduce_only=True,
    )
    bar1 = next(it)
    triggered = engine.on_bar("X", bar1)
    assert len(triggered) == 1
    # PnL = (100 - 105) * 1 = -5
    assert engine.realized_pnl_usdt == pytest.approx(-5.0)


@pytest.mark.asyncio
async def test_gap_through_stop_uses_open_price_not_stop_price(tmp_path):
    """If bar.open is already below the LONG-protective stop, the stop
    fills at the worse-of-two price (open) — no free ride for the gap."""
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000,  90.0,  91.0,  85.0,  88.0, 100.0],  # opens at 90, below stop=95
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)
    await engine.place_stop_order(
        "X", Side.SHORT, size=1.0, stop_price=95.0, reduce_only=True,
    )
    bar1 = next(it)
    triggered = engine.on_bar("X", bar1)
    assert len(triggered) == 1
    assert triggered[0]["fill_price"] == pytest.approx(90.0)


@pytest.mark.asyncio
async def test_stop_does_not_trigger_when_range_misses(tmp_path):
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000, 100.0, 102.0,  98.0,  99.0, 100.0],  # range never reaches 90
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)
    await engine.place_stop_order(
        "X", Side.SHORT, size=1.0, stop_price=90.0, reduce_only=True,
    )
    bar1 = next(it)
    triggered = engine.on_bar("X", bar1)
    assert triggered == []
    open_orders = await engine.fetch_open_orders()
    assert len(open_orders) == 1


@pytest.mark.asyncio
async def test_cancel_stop_removes_it(tmp_path):
    bars = _flat_bars(3)
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    stop = await engine.place_stop_order(
        "X", Side.SHORT, size=1.0, stop_price=95.0, reduce_only=True,
    )
    cancel_result = await engine.cancel_order(stop["id"], "X")
    assert cancel_result["status"] == STATUS_CANCELLED
    open_orders = await engine.fetch_open_orders()
    assert open_orders == []


@pytest.mark.asyncio
async def test_cancel_unknown_order_is_noop(tmp_path):
    bars = _flat_bars(3)
    engine, _ = _build_engine(tmp_path, bars)
    res = await engine.cancel_order("nonexistent", "X")
    assert res["status"] == "not_found"


# ----------------------- idempotency ----------------------- #


@pytest.mark.asyncio
async def test_duplicate_client_order_id_returns_cached_market_fill(tmp_path):
    engine, adapter = _build_engine(tmp_path, _flat_bars(price=100.0))
    next(adapter.iter_bars("X"))
    coid = "live-test-client-id"
    f1 = await engine.market_order(
        "X", Side.LONG, size=1.0, client_order_id=coid,
    )
    # Same client_order_id again: same response, no new fill recorded.
    f2 = await engine.market_order(
        "X", Side.LONG, size=99.0, client_order_id=coid,
    )
    assert f1["id"] == f2["id"]
    assert f2["size"] == 1.0  # ignored the new size; deduped
    assert len(engine.fills) == 1


@pytest.mark.asyncio
async def test_duplicate_client_order_id_dedupes_stops(tmp_path):
    engine, adapter = _build_engine(tmp_path, _flat_bars(3))
    next(adapter.iter_bars("X"))
    coid = "stop-coid"
    s1 = await engine.place_stop_order(
        "X", Side.SHORT, size=1.0, stop_price=95.0,
        reduce_only=True, client_order_id=coid,
    )
    s2 = await engine.place_stop_order(
        "X", Side.SHORT, size=99.0, stop_price=80.0,
        reduce_only=True, client_order_id=coid,
    )
    assert s1["id"] == s2["id"]
    open_orders = await engine.fetch_open_orders()
    assert len(open_orders) == 1


# ----------------------- callbacks ----------------------- #


@pytest.mark.asyncio
async def test_on_fill_and_on_stop_trigger_callbacks_fire(tmp_path):
    fills_seen = []
    stops_seen = []
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000, 100.0, 100.0,  90.0,  95.0, 100.0],
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    engine.cfg = MatchingEngineConfig(
        default_top_depth_usdt=100_000.0, default_realized_vol=0.0,
        on_fill=lambda f: fills_seen.append(f),
        on_stop_trigger=lambda f: stops_seen.append(f),
    )
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)
    await engine.place_stop_order("X", Side.SHORT, size=1.0,
                                  stop_price=95.0, reduce_only=True)
    bar1 = next(it)
    engine.on_bar("X", bar1)
    # fills_seen should contain entry + stop close (2 records).
    assert len(fills_seen) == 2
    assert len(stops_seen) == 1


# ----------------------- summary + equity ----------------------- #


@pytest.mark.asyncio
async def test_equity_includes_open_position_mtm(tmp_path):
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000, 110.0, 110.0, 110.0, 110.0, 100.0],
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)  # at 100
    next(it)
    # Position is still open; mark = 110.
    eq = engine.equity()
    assert eq == pytest.approx(10_010.0)


@pytest.mark.asyncio
async def test_summary_reports_counts(tmp_path):
    bars = [
        [0,       100.0, 100.0, 100.0, 100.0, 100.0],
        [60_000, 100.0, 100.0,  90.0,  95.0, 100.0],
    ]
    engine, adapter = _build_engine(tmp_path, bars)
    it = adapter.iter_bars("X")
    next(it)
    await engine.market_order("X", Side.LONG, size=1.0)
    await engine.place_stop_order("X", Side.SHORT, size=1.0,
                                  stop_price=95.0, reduce_only=True)
    bar1 = next(it)
    engine.on_bar("X", bar1)
    summary = engine.summary()
    assert summary["fills"] == 2  # market + stop
    assert summary["triggered_stops"] == 1
    assert summary["open_positions"] == []


# ----------------------- error paths ----------------------- #


@pytest.mark.asyncio
async def test_market_order_without_current_bar_raises(tmp_path):
    engine, _ = _build_engine(tmp_path, _flat_bars())
    with pytest.raises(RuntimeError, match="no current bar"):
        await engine.market_order("X", Side.LONG, size=1.0)
