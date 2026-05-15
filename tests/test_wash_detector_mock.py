"""Mock tests for WashTradingDetector (SR-3 / TA-07)."""

from __future__ import annotations

import pytest

from altcoin_agent.screener import Kline, SignalKind, WashTradingDetector


def _bar(ts: int, vol: float, count: int, *, bullish: bool = True) -> Kline:
    o, c = (1.0, 1.005) if bullish else (1.005, 1.0)
    return Kline(
        ts=ts, open=o, high=max(o, c) + 0.001, low=min(o, c) - 0.001,
        close=c, volume=vol, trade_count=count,
    )


def test_wash_skips_bars_without_trade_count() -> None:
    """If trade_count == 0 (legacy / unknown), the detector is a no-op."""
    det = WashTradingDetector(window=60, min_samples=30)
    for i in range(40):
        ev = det.feed("binance", "X", Kline(
            ts=i * 60_000, open=1.0, high=1.01, low=0.99, close=1.0, volume=1000.0,
            # trade_count omitted -> default 0
        ))
        assert ev is None


def test_wash_does_not_fire_on_calm_baseline() -> None:
    det = WashTradingDetector(window=60, min_samples=30, volume_z_min=3.0)
    # 35 calm bars with very different volume / count noise patterns,
    # ensuring std > 0 on both series and avg_size has variance too.
    for i in range(35):
        vol = 1000.0 + (i * 7) % 23
        cnt = 100 + (i * 13) % 31
        ev = det.feed("binance", "X", _bar(i * 60_000, vol, cnt))
        assert ev is None
    # An only-mildly-elevated bar (z ~ much less than 3) should not fire.
    ev = det.feed("binance", "X", _bar(36 * 60_000, 1010.0, 105))
    assert ev is None


def test_wash_ghost_volume_pattern_fires() -> None:
    """Volume z-score huge but trade_count z-score tiny -> ghost_volume."""
    det = WashTradingDetector(window=60, min_samples=30,
                              volume_z_min=3.0, ratio_threshold=2.0)
    # 40 calm bars: volume noisy 800-1200, trade_count noisy 100-130.
    for i in range(40):
        vol = 1000.0 + (i * 11) % 200 - 100
        cnt = 110 + (i * 7) % 21 - 10
        det.feed("binance", "X", _bar(i * 60_000, vol, cnt))

    # Spike: volume 25x average; trade_count same as baseline -> ghost.
    ev = det.feed("binance", "X", _bar(41 * 60_000, vol=25_000.0, count=115))
    assert ev is not None
    assert ev.kind == SignalKind.WASH_TRADING_DETECTED
    assert "ghost_volume" in ev.payload["patterns"]
    assert ev.payload["volume_zscore"] >= 3.0


def test_wash_whale_single_print_pattern_fires() -> None:
    """Volume up AND avg trade size massively up -> whale_single_print."""
    det = WashTradingDetector(window=60, min_samples=30,
                              volume_z_min=3.0, avg_size_z_min=4.0)
    for i in range(40):
        vol = 1000.0 + (i * 11) % 200 - 100
        cnt = 100 + (i * 5) % 17  # trade_count tied to nothing here
        det.feed("binance", "X", _bar(i * 60_000, vol, cnt))

    # 50x volume, only 2x trades -> avg trade size shoots up.
    ev = det.feed("binance", "X", _bar(41 * 60_000, vol=50_000.0, count=200))
    assert ev is not None
    assert ev.kind == SignalKind.WASH_TRADING_DETECTED
    # ghost_volume is also expected here (vol_z huge vs count_z modest).
    assert "whale_single_print" in ev.payload["patterns"]


def test_wash_dedupes_per_bar_ts() -> None:
    det = WashTradingDetector(window=60, min_samples=30)
    for i in range(40):
        det.feed("binance", "X", _bar(i * 60_000, 1000.0 + (i * 11) % 200 - 100,
                                       110 + (i * 7) % 21 - 10))
    spike = _bar(41 * 60_000, 30_000.0, 115)
    assert det.feed("binance", "X", spike) is not None
    assert det.feed("binance", "X", spike) is None


def test_wash_window_must_be_geq_min_samples() -> None:
    with pytest.raises(ValueError):
        WashTradingDetector(window=10, min_samples=30)
