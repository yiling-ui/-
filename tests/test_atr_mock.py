"""Tests for ATRCalculator + TradeFlowAggregator."""

from __future__ import annotations

import pytest

from altcoin_agent.risk.atr import ATRCalculator
from altcoin_agent.screener import Kline
from altcoin_agent.screener_extras import TradeFlowAggregator


def _bar(ts: int, *, h: float, low: float, c: float, o: float | None = None) -> Kline:
    return Kline(ts=ts, open=o if o is not None else c, high=h, low=low,
                 close=c, volume=100.0, timeframe="1m")


def test_atr_returns_first_tr_when_warming_up() -> None:
    a = ATRCalculator(period=14)
    out = a.update("binance", "X", _bar(0, h=1.05, low=0.95, c=1.0))
    # Only one bar -> TR = high - low
    assert out == pytest.approx(0.10)


def test_atr_averages_after_window_fills() -> None:
    a = ATRCalculator(period=4)
    a.update("binance", "X", _bar(0, h=1.05, low=0.95, c=1.0))
    a.update("binance", "X", _bar(60_000, h=1.06, low=0.96, c=1.0))   # TR = 0.10
    a.update("binance", "X", _bar(120_000, h=1.07, low=0.97, c=1.0))  # TR = 0.10
    a.update("binance", "X", _bar(180_000, h=1.20, low=1.05, c=1.10)) # TR = 0.20 (vs prev close 1.0)
    final = a.update("binance", "X", _bar(240_000, h=1.15, low=1.05, c=1.10))
    # avg(0.10, 0.10, 0.10, 0.20, 0.10) = 0.12
    assert final == pytest.approx(0.12, rel=0.05)


def test_atr_segregates_by_symbol_and_tf() -> None:
    a = ATRCalculator(period=4)
    a.update("binance", "X", _bar(0, h=1.05, low=0.95, c=1.0))
    a.update("binance", "Y", _bar(0, h=2.0, low=1.0, c=1.5))
    assert a.get("binance", "X") != a.get("binance", "Y")


def test_aggregator_emits_closed_bar_when_minute_advances() -> None:
    agg = TradeFlowAggregator(timeframes=("1m",))
    agg.ingest_trade("binance", "X", 60_000, price=1.00, amount=10)
    agg.ingest_trade("binance", "X", 60_500, price=1.01, amount=5)
    # Same minute, no bar yet
    out_now = agg.ingest_trade("binance", "X", 70_000, price=1.02, amount=3)
    assert out_now == []
    # Cross into next minute -> previous bar closes
    out = agg.ingest_trade("binance", "X", 121_000, price=1.05, amount=7)
    assert len(out) == 1
    tf, bar = out[0]
    assert tf == "1m"
    assert bar.timeframe == "1m"
    assert bar.ts == 60_000
    assert bar.open == pytest.approx(1.00)
    assert bar.close == pytest.approx(1.02)
    assert bar.high == pytest.approx(1.02)
    assert bar.volume == pytest.approx(18)
    assert bar.trade_count == 3


def test_aggregator_segregates_timeframes() -> None:
    agg = TradeFlowAggregator(timeframes=("1m", "5m"))
    # Several trades across multiple 1m buckets within the same 5m bucket.
    agg.ingest_trade("binance", "X", 60_000, price=1.0, amount=1)
    agg.ingest_trade("binance", "X", 121_000, price=1.1, amount=1)  # closes 1m
    agg.ingest_trade("binance", "X", 181_000, price=1.2, amount=1)  # closes 1m
    out = agg.ingest_trade("binance", "X", 360_000, price=1.3, amount=1)
    # Crossing into a new 5m bucket closes both timeframes' buckets.
    tfs = sorted(tf for tf, _ in out)
    assert tfs == ["1m", "5m"]
