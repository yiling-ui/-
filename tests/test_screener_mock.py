"""
Mock tests for screener.py.

Goal: prove that detectors fire correctly when fed synthetic OI surge,
extreme funding, volume spike, and SMC sweep sequences — with NO network.
"""

from __future__ import annotations

import asyncio

import pytest

from altcoin_agent.screener import (
    FundingAnomalyDetector,
    FundingSnapshot,
    Kline,
    LiquidityPoolAnalyzer,
    OISnapshot,
    OISurgeDetector,
    Screener,
    SignalEvent,
    SignalKind,
    VolumeSpikeDetector,
)

# --------------------------------------------------------------------------- #
# OI Surge
# --------------------------------------------------------------------------- #


def test_oi_surge_silent_build_fires_when_oi_jumps_with_flat_price() -> None:
    det = OISurgeDetector(window=5, surge_pct=0.15, silent_max_price_move=0.01)
    base_oi = 1_000_000.0
    base_price = 1.000

    # 5 stable samples (flat OI, flat price)
    for i in range(5):
        ev = det.feed("binance", OISnapshot(ts=i * 60_000, symbol="RAVEUSDT",
                                            open_interest=base_oi, price=base_price))
        assert ev is None

    # +20% OI, price barely moves -> SILENT_BUILD
    ev = det.feed(
        "binance",
        OISnapshot(ts=6 * 60_000, symbol="RAVEUSDT", open_interest=base_oi * 1.20, price=base_price * 1.002),
    )
    assert ev is not None
    assert ev.kind == SignalKind.OI_SILENT_BUILD
    assert ev.payload["oi_delta_pct"] == pytest.approx(0.20, rel=1e-6)
    assert ev.payload["price_move_pct"] < 0.01


def test_oi_surge_breakout_when_price_also_moves() -> None:
    det = OISurgeDetector(window=5, surge_pct=0.15, silent_max_price_move=0.01)
    for i in range(5):
        det.feed("okx", OISnapshot(ts=i * 60_000, symbol="MYXUSDT", open_interest=500_000.0, price=2.0))
    ev = det.feed(
        "okx",
        OISnapshot(ts=6 * 60_000, symbol="MYXUSDT", open_interest=500_000 * 1.30, price=2.0 * 1.05),
    )
    assert ev is not None
    assert ev.kind == SignalKind.OI_SURGE
    assert ev.payload["oi_delta_pct"] == pytest.approx(0.30, rel=1e-6)


def test_oi_surge_does_not_fire_below_threshold() -> None:
    det = OISurgeDetector(window=5, surge_pct=0.15)
    for i in range(5):
        det.feed("binance", OISnapshot(ts=i * 60_000, symbol="X", open_interest=1_000_000, price=1.0))
    ev = det.feed("binance", OISnapshot(ts=6 * 60_000, symbol="X", open_interest=1_100_000, price=1.0))
    assert ev is None  # only +10%


def test_oi_surge_dedupes_same_ts() -> None:
    det = OISurgeDetector(window=5, surge_pct=0.15)
    for i in range(5):
        det.feed("binance", OISnapshot(ts=i * 60_000, symbol="X", open_interest=1_000_000, price=1.0))
    snap = OISnapshot(ts=6 * 60_000, symbol="X", open_interest=1_300_000, price=1.0)
    ev1 = det.feed("binance", snap)
    ev2 = det.feed("binance", snap)
    assert ev1 is not None and ev2 is None


# --------------------------------------------------------------------------- #
# Funding rate anomalies
# --------------------------------------------------------------------------- #


def test_funding_extreme_short_squeeze_after_consecutive_samples() -> None:
    det = FundingAnomalyDetector(extreme_low=-0.001, extreme_high=0.0015, consecutive=2)

    # one extreme reading is not enough
    evs = det.feed("binance", FundingSnapshot(ts=1, symbol="RAVEUSDT", rate=-0.002))
    assert all(e.kind != SignalKind.FUNDING_EXTREME for e in evs)

    # second consecutive extreme reading -> fire
    evs = det.feed("binance", FundingSnapshot(ts=2, symbol="RAVEUSDT", rate=-0.0025))
    extremes = [e for e in evs if e.kind == SignalKind.FUNDING_EXTREME]
    assert len(extremes) == 1
    assert extremes[0].payload["direction"] == "short_squeeze"


def test_funding_extreme_long_fragile() -> None:
    det = FundingAnomalyDetector(consecutive=2)
    det.feed("binance", FundingSnapshot(ts=1, symbol="X", rate=0.002))
    evs = det.feed("binance", FundingSnapshot(ts=2, symbol="X", rate=0.003))
    fired = [e for e in evs if e.kind == SignalKind.FUNDING_EXTREME]
    assert fired and fired[0].payload["direction"] == "long_fragile"


def test_funding_streak_resets_on_normalization() -> None:
    det = FundingAnomalyDetector(consecutive=2)
    det.feed("binance", FundingSnapshot(ts=1, symbol="X", rate=-0.002))   # streak=1
    det.feed("binance", FundingSnapshot(ts=2, symbol="X", rate=-0.0001))  # streak=0
    evs = det.feed("binance", FundingSnapshot(ts=3, symbol="X", rate=-0.002))  # streak=1
    assert all(e.kind != SignalKind.FUNDING_EXTREME for e in evs)


def test_funding_short_window_deviation_fires_before_extreme() -> None:
    """
    Build a long calm baseline at ~0, then a sharp dive in the short window.
    The deviation detector should flag this even if absolute level is not yet
    at the extreme threshold.
    """
    det = FundingAnomalyDetector(
        extreme_low=-0.005,    # raise the bar so absolute trigger does NOT fire
        extreme_high=0.005,
        consecutive=2,
        short_window=3,
        long_window=24,
        deviation_sigma=2.5,
    )
    # 21 calm samples around 0 with tiny noise to avoid std=0
    noise = [-1e-5, 1e-5] * 11
    for i, n in enumerate(noise[:21]):
        det.feed("binance", FundingSnapshot(ts=i, symbol="X", rate=n))
    # 3 sharp negatives — well within (greater than) extreme_low so EXTREME does not trip.
    evs_collected: list[SignalEvent] = []
    for i, r in enumerate([-0.0008, -0.0009, -0.0011], start=21):
        evs_collected.extend(det.feed("binance", FundingSnapshot(ts=i, symbol="X", rate=r)))

    deviations = [e for e in evs_collected if e.kind == SignalKind.FUNDING_DEVIATION]
    extremes = [e for e in evs_collected if e.kind == SignalKind.FUNDING_EXTREME]
    assert len(deviations) >= 1, "deviation should fire on the sharp short-window dive"
    assert not extremes, "absolute extreme threshold should not have triggered in this scenario"


# --------------------------------------------------------------------------- #
# Volume spike
# --------------------------------------------------------------------------- #


def _bar(ts: int, vol: float, *, bullish: bool = True, base: float = 1.0) -> Kline:
    o, c = (base, base * 1.01) if bullish else (base * 1.01, base)
    return Kline(ts=ts, open=o, high=max(o, c), low=min(o, c), close=c, volume=vol)


def test_volume_spike_fires_on_4_sigma_jump() -> None:
    det = VolumeSpikeDetector(window=60, k_sigma=4.0, min_samples=20)
    # 30 quiet bars
    for i in range(30):
        ev = det.feed("binance", "RAVEUSDT", _bar(i * 60_000, vol=100.0 + (i % 3)))
        assert ev is None
    # one bar with 50x average volume
    spike = _bar(31 * 60_000, vol=5_000.0, bullish=True)
    ev = det.feed("binance", "RAVEUSDT", spike)
    assert ev is not None
    assert ev.kind == SignalKind.VOLUME_SPIKE
    assert ev.payload["side"] == "buy"
    assert ev.payload["zscore"] > 4.0


def test_volume_spike_dedupes_same_bar() -> None:
    det = VolumeSpikeDetector(window=60, k_sigma=4.0, min_samples=20)
    for i in range(30):
        det.feed("binance", "X", _bar(i * 60_000, vol=100.0 + (i % 3)))
    spike = _bar(31 * 60_000, vol=5_000.0)
    assert det.feed("binance", "X", spike) is not None
    assert det.feed("binance", "X", spike) is None


# --------------------------------------------------------------------------- #
# Liquidity pool sweep (SMC)
# --------------------------------------------------------------------------- #


def test_liquidity_pool_forms_then_gets_swept_above_equal_highs() -> None:
    """
    Build two equal swing highs at ~1.10 to form a sell-side liquidity pool.
    Then send a wick-up bar that pierces the level and closes back below —
    classic stop-hunt above equal highs.
    """
    ana = LiquidityPoolAnalyzer(
        fractal=2,
        cluster_pct=0.002,
        sweep_pierce_bps=5.0,
        wick_to_body_min=1.5,
    )

    # Two well-separated pivot-high patterns (low, low, HIGH, low, low) where
    # only the pivot bar reaches the equal-high level. Surrounding bars stay
    # well below so the fractal detector picks the pivot uniquely.
    def pivot_high_block(t0: int, level: float) -> list[Kline]:
        return [
            Kline(t0,         1.00, 1.02, 1.00, 1.01, 100.0),
            Kline(t0 + 60,    1.01, 1.03, 1.00, 1.02, 100.0),
            Kline(t0 + 120,   1.02, level, 1.02, level - 0.005, 100.0),  # pivot
            Kline(t0 + 180,   1.03, 1.04, 1.02, 1.03, 100.0),
            Kline(t0 + 240,   1.02, 1.03, 1.01, 1.02, 100.0),
        ]

    bars: list[Kline] = []
    bars += pivot_high_block(t0=0,           level=1.1000)
    # Drift back to baseline before the second pivot so the second pivot's
    # surrounding bars don't accidentally outrank it.
    bars += [
        Kline(300, 1.02, 1.03, 1.01, 1.02, 100.0),
        Kline(360, 1.02, 1.03, 1.01, 1.02, 100.0),
    ]
    bars += pivot_high_block(t0=420,         level=1.1015)

    events: list[SignalEvent] = []
    for b in bars:
        events.extend(ana.feed("binance", "RAVEUSDT", b))

    pool_events = [e for e in events if e.kind == SignalKind.LIQUIDITY_POOL_FORMED]
    assert len(pool_events) == 1, f"expected pool to form, got events={events}"
    assert pool_events[0].payload["side"] == "sell_side"
    assert pool_events[0].payload["touch_count"] == 2

    # Now a sweep bar AFTER the pool is fully confirmed.
    pool_level = pool_events[0].payload["level"]
    sweep_bar = Kline(
        ts=20 * 60,
        open=pool_level - 0.010,
        high=pool_level + 0.005,    # pierces by ~45 bps
        low=pool_level - 0.011,
        close=pool_level - 0.010,   # closes back below
        volume=500.0,
    )
    sweep_events = ana.feed("binance", "RAVEUSDT", sweep_bar)
    sweeps = [e for e in sweep_events if e.kind == SignalKind.LIQUIDITY_SWEEP]
    assert len(sweeps) == 1, f"expected 1 sweep event, got {sweep_events}"
    payload = sweeps[0].payload
    assert payload["side"] == "sell_side"
    assert payload["wick_to_body"] >= 1.5
    assert payload["touch_count_before_sweep"] == 2


# --------------------------------------------------------------------------- #
# Screener orchestration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_screener_dispatches_oi_and_funding_events_to_sink() -> None:
    """
    End-to-end: feed mock OI and funding snapshots through the Screener's
    public on_* hooks (the same hooks the live ws loops call) and verify
    the sink receives the expected events. No ccxt, no network.
    """
    received: list[SignalEvent] = []

    async def sink(ev: SignalEvent) -> None:
        received.append(ev)

    sc = Screener(
        exchanges=["binance"],
        symbols=["RAVEUSDT"],
        sink=sink,
        oi_detector=OISurgeDetector(window=3, surge_pct=0.10, silent_max_price_move=0.01),
        funding_detector=FundingAnomalyDetector(extreme_low=-0.001, extreme_high=0.0015, consecutive=2),
    )

    # OI: 3 calm samples, then +25% with flat price -> silent build
    base = OISnapshot(ts=0, symbol="RAVEUSDT", open_interest=1_000_000, price=1.0)
    for i in range(3):
        await sc.on_oi("binance", OISnapshot(ts=i * 60_000, symbol="RAVEUSDT",
                                             open_interest=base.open_interest, price=base.price))
    await sc.on_oi(
        "binance",
        OISnapshot(ts=4 * 60_000, symbol="RAVEUSDT",
                   open_interest=base.open_interest * 1.25, price=base.price * 1.001),
    )

    # Funding: two consecutive extreme negatives
    await sc.on_funding("binance", FundingSnapshot(ts=1, symbol="RAVEUSDT", rate=-0.002))
    await sc.on_funding("binance", FundingSnapshot(ts=2, symbol="RAVEUSDT", rate=-0.0022))

    # let any pending callbacks drain
    await asyncio.sleep(0)

    kinds = [ev.kind for ev in received]
    assert SignalKind.OI_SILENT_BUILD in kinds
    assert SignalKind.FUNDING_EXTREME in kinds

    # And sanity-check fields on the OI event
    oi_ev = next(e for e in received if e.kind == SignalKind.OI_SILENT_BUILD)
    assert oi_ev.symbol == "RAVEUSDT"
    assert oi_ev.exchange == "binance"
    assert 0.20 <= oi_ev.payload["oi_delta_pct"] <= 0.30
