"""
Mock tests for WashTradingDetector (TA-07 / SR-3) and its integration
into the fuser as a hard long-side veto.
"""

from __future__ import annotations

from collections import deque

import pytest

from altcoin_agent.fuser import Direction, FuserConfig, ScoreFuser
from altcoin_agent.screener import (
    Kline,
    SignalEvent,
    SignalKind,
    WashTradingDetector,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bar(
    ts: int,
    *,
    vol: float = 100.0,
    trade_count: int = 100,
    bullish: bool = True,
    base: float = 1.0,
) -> Kline:
    o, c = (base, base * 1.01) if bullish else (base * 1.01, base)
    return Kline(
        ts=ts, open=o, high=max(o, c), low=min(o, c), close=c,
        volume=vol, trade_count=trade_count,
    )


def _ev(
    kind: SignalKind,
    *,
    ts: int = 1_000_000,
    symbol: str = "RAVEUSDT",
    exchange: str = "binance",
    payload: dict | None = None,
) -> SignalEvent:
    return SignalEvent(
        kind=kind, symbol=symbol, ts=ts, exchange=exchange, payload=payload or {},
    )


# --------------------------------------------------------------------------- #
# Detector unit tests
# --------------------------------------------------------------------------- #


def test_detector_skips_bars_without_trade_count() -> None:
    """trade_count == 0 means the data source didn't provide it. The
    detector must NEVER fire — false positives in this regime would
    be unconscionable."""
    det = WashTradingDetector(window=30, min_samples=20)
    for i in range(25):
        bar = _bar(i * 60_000, vol=100.0, trade_count=0)
        ev = det.feed("binance", "X", bar)
        assert ev is None
    # Even an extreme volume spike with no trade count gives no signal.
    spike = _bar(26 * 60_000, vol=10_000.0, trade_count=0)
    assert det.feed("binance", "X", spike) is None


def test_detector_fires_on_ghost_volume_pattern() -> None:
    """Pattern 1: volume z-score huge, trade-count z-score flat. Wash."""
    det = WashTradingDetector(
        window=60, min_samples=20, z_ratio_max=2.0,
        trade_count_z_floor=1.5, volume_z_gate=4.0,
    )
    # 25 calm bars with INDEPENDENT variation in volume vs trade_count
    # (so the rolling stds are non-zero — required for z-scores).
    for i in range(25):
        det.feed("binance", "X", _bar(
            i * 60_000, vol=100.0 + (i % 5), trade_count=100 + (i % 3),
        ))
    # Spike: volume 50x, but trade_count basically unchanged (a few extra
    # large self-deals). Classic ghost volume.
    spike = _bar(26 * 60_000, vol=5_000.0, trade_count=102)
    ev = det.feed("binance", "X", spike)
    assert ev is not None
    assert ev.kind == SignalKind.WASH_TRADING_DETECTED
    assert "ghost_volume" in ev.payload["patterns"]
    assert ev.payload["volume_zscore"] > 4.0
    # ratio sanity
    assert ev.payload["volume_to_count_z_ratio"] > 2.0


def test_detector_fires_on_whale_single_print_pattern() -> None:
    """Pattern 2: trade_count rises a bit but avg_trade_size shoots up
    way more than the baseline. A handful of whale-sized self-deals."""
    det = WashTradingDetector(
        window=60, min_samples=20, z_ratio_max=2.0,
        trade_count_z_floor=1.5, avg_size_sigma_threshold=4.0,
        volume_z_gate=4.0,
    )
    # baseline: vol ~ 100 with small noise, count ~ 100 with DIFFERENT noise
    # so avg_size = vol/count has a meaningful (non-zero) std around ~1.0.
    for i in range(25):
        det.feed("binance", "X", _bar(
            i * 60_000,
            vol=100.0 + (i % 5) * 0.5,        # varies 100, 100.5, 101, 101.5, 102
            trade_count=100 + (i % 3),         # varies 100, 101, 102
        ))
    # spike: vol=5000 (avg ~50.0 per trade vs baseline ~1.0)
    spike = _bar(26 * 60_000, vol=5_000.0, trade_count=100)
    ev = det.feed("binance", "X", spike)
    assert ev is not None
    assert "whale_single_print" in ev.payload["patterns"]
    assert ev.payload["avg_trade_size_zscore"] is not None
    assert ev.payload["avg_trade_size_zscore"] >= 4.0


def test_detector_does_NOT_fire_on_organic_pump() -> None:
    """A real retail-driven pump: BOTH volume AND trade_count explode,
    avg_trade_size stays roughly stable or falls (lots of small orders)."""
    det = WashTradingDetector(
        window=60, min_samples=20, z_ratio_max=2.0,
        trade_count_z_floor=1.5, avg_size_sigma_threshold=4.0,
        volume_z_gate=4.0,
    )
    for i in range(25):
        det.feed("binance", "X", _bar(
            i * 60_000, vol=100.0 + (i % 5), trade_count=100 + (i % 3),
        ))
    # Organic spike: vol 50x AND trade_count 50x. avg_size unchanged.
    spike = _bar(26 * 60_000, vol=5_000.0, trade_count=5_000)
    ev = det.feed("binance", "X", spike)
    assert ev is None


def test_detector_dedupes_same_bar() -> None:
    det = WashTradingDetector(window=60, min_samples=20)
    # Independent baseline variation so volume std > 0.
    for i in range(25):
        det.feed("binance", "X", _bar(
            i * 60_000, vol=100 + (i % 5), trade_count=100 + (i % 3),
        ))
    spike = _bar(26 * 60_000, vol=5_000, trade_count=102)
    assert det.feed("binance", "X", spike) is not None
    # Feeding the same bar (same ts) again must NOT re-fire
    assert det.feed("binance", "X", spike) is None


def test_detector_ignores_calm_bars() -> None:
    """Without a concurrent volume spike, no wash flag. We never claim
    'this looks weird' on quiet activity."""
    det = WashTradingDetector(window=60, min_samples=20, volume_z_gate=4.0)
    for i in range(25):
        det.feed("binance", "X", _bar(
            i * 60_000, vol=100 + (i % 5), trade_count=100 + (i % 3),
        ))
    # A bar similar to the baseline (vol_z well below the 4σ gate)
    bar = _bar(26 * 60_000, vol=104, trade_count=101)
    assert det.feed("binance", "X", bar) is None


# --------------------------------------------------------------------------- #
# Fuser integration: long-side veto + short still allowed
# --------------------------------------------------------------------------- #


def test_fuser_vetoes_long_when_wash_event_present() -> None:
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault(
        "binance:RAVEUSDT", deque(maxlen=64),
    )
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-5_000,
            payload={"side": "buy", "zscore": 6.0}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-3_000,
            payload={"oi_delta_pct": 0.22, "from_price": 1.000, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now-1_000,
            payload={"side": "buy_side", "wick_to_body": 2.1}),
        # Wash event arrives concurrently
        _ev(SignalKind.WASH_TRADING_DETECTED, ts=now,
            payload={
                "patterns": ["ghost_volume"],
                "volume_zscore": 7.5,
                "trade_count_zscore": 0.4,
                "volume_to_count_z_ratio": 18.7,
            }),
    ])
    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.is_high_priority is False
    assert sig.blocked is True
    assert sig.block_reason == "wash_trading_detected"
    assert sig.direction == Direction.NEUTRAL
    assert sig.final_score <= FuserConfig().wash_trading_veto_score
    assert any("wash trading" in n.lower() for n in sig.notes)


def test_fuser_does_NOT_veto_short_when_wash_event_present() -> None:
    """Wash trading typically signals a fake pump that's about to dump.
    Shorting alongside it is the right play; the veto is long-only."""
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault(
        "binance:MYXUSDT", deque(maxlen=64),
    )
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-5_000, symbol="MYXUSDT",
            payload={"side": "sell", "zscore": 6.5}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-3_000, symbol="MYXUSDT",
            payload={"oi_delta_pct": 0.20, "from_price": 2.000, "to_price": 1.985}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now-1_000, symbol="MYXUSDT",
            payload={"side": "sell_side", "wick_to_body": 2.4}),
        _ev(SignalKind.WASH_TRADING_DETECTED, ts=now, symbol="MYXUSDT",
            payload={"patterns": ["whale_single_print"]}),
    ])
    sig = fuser.evaluate("MYXUSDT", "binance", now)
    assert sig.direction == Direction.SHORT
    assert sig.blocked is False
    assert sig.is_high_priority is True


def test_fuser_ignores_stale_wash_event_outside_window() -> None:
    fuser = ScoreFuser(config=FuserConfig(window_sec=90))
    now = 10_000_000
    bucket = fuser._rules.setdefault("binance:X", deque(maxlen=64))
    # A wash event from 200s ago (well outside the 90s window) must NOT veto
    # current rule signals.
    bucket.append(_ev(SignalKind.WASH_TRADING_DETECTED, ts=now - 200_000,
                      symbol="X", payload={"patterns": ["ghost_volume"]}))
    bucket.append(_ev(SignalKind.VOLUME_SPIKE, ts=now-5_000, symbol="X",
                      payload={"side": "buy"}))
    bucket.append(_ev(SignalKind.OI_SILENT_BUILD, ts=now-3_000, symbol="X",
                      payload={"oi_delta_pct": 0.20,
                               "from_price": 1.0, "to_price": 1.005}))
    bucket.append(_ev(SignalKind.LIQUIDITY_SWEEP, ts=now, symbol="X",
                      payload={"side": "buy_side", "wick_to_body": 2.0}))
    sig = fuser.evaluate("X", "binance", now)
    assert sig.direction == Direction.LONG
    assert sig.is_high_priority is True  # not vetoed
    assert sig.blocked is False


def test_fuser_wash_veto_runs_in_pure_rule_mode_too() -> None:
    """Wash veto must apply even when no LLM verdict is present."""
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault("binance:Y", deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-5_000, symbol="Y",
            payload={"side": "buy"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-3_000, symbol="Y",
            payload={"oi_delta_pct": 0.20, "from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now-1_000, symbol="Y",
            payload={"side": "buy_side", "wick_to_body": 2.0}),
        _ev(SignalKind.WASH_TRADING_DETECTED, ts=now, symbol="Y",
            payload={"patterns": ["ghost_volume"]}),
    ])
    sig = fuser.evaluate("Y", "binance", now)
    # No LLM was injected; veto still applies
    assert sig.llm_verdict is None
    assert sig.blocked is True
    assert sig.block_reason == "wash_trading_detected"


def test_wash_event_alone_does_not_promote() -> None:
    """A wash event by itself contributes 0 to rule_score and direction
    is NEUTRAL — it must never accidentally PROMOTE anything."""
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault("binance:Z", deque(maxlen=64))
    bucket.append(_ev(SignalKind.WASH_TRADING_DETECTED, ts=now, symbol="Z",
                      payload={"patterns": ["ghost_volume"]}))
    sig = fuser.evaluate("Z", "binance", now)
    assert sig.is_high_priority is False
    assert sig.rule_score == 0.0
    assert sig.direction == Direction.NEUTRAL


# --------------------------------------------------------------------------- #
# End-to-end: Screener emits the wash event into the sink
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_screener_dispatches_wash_event_to_sink() -> None:
    from altcoin_agent.screener import Screener

    received: list[SignalEvent] = []

    async def sink(ev: SignalEvent) -> None:
        received.append(ev)

    sc = Screener(
        exchanges=["binance"], symbols=["X"], sink=sink,
        wash_trading_detector=WashTradingDetector(
            window=30, min_samples=20, volume_z_gate=4.0,
        ),
    )
    # 25 calm bars
    for i in range(25):
        await sc.on_kline("binance", "X", _bar(
            i * 60_000, vol=100.0 + (i % 5), trade_count=100 + (i % 3),
        ))
    # Wash spike: 50x volume, trade_count basically flat
    await sc.on_kline("binance", "X",
                      _bar(26 * 60_000, vol=5_000.0, trade_count=102))

    kinds = [e.kind for e in received]
    assert SignalKind.WASH_TRADING_DETECTED in kinds
