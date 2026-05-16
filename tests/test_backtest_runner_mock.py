"""tests/test_backtest_runner_mock.py — QUADRANT Phase 3 coverage.

Smoke-tests the backtest runner end-to-end by feeding a synthetic
1-minute bar stream that walks through the canonical pump cycle and
verifying the FSM's phase trajectory.
"""

from __future__ import annotations

import math

from altcoin_agent.backtest.runner import (
    BARS_PER_DAY_1M,
    BacktestRunner,
    PhaseTaggedBar,
    compute_phase_inputs,
)
from altcoin_agent.risk.pump_phase import KlineBar, PhaseThresholds, PumpPhase

# ----------------------- compute_phase_inputs ----------------------- #


def _kbar(ts: int, close: float, *, hi: float | None = None,
          lo: float | None = None, o: float | None = None,
          vol: float = 1.0, vol_z: float = 0.0) -> KlineBar:
    return KlineBar(
        ts_ms=ts,
        open=o if o is not None else close,
        high=hi if hi is not None else close,
        low=lo if lo is not None else close,
        close=close,
        volume=vol,
        vol_z_score=vol_z,
    )


def test_compute_inputs_empty_history_returns_safe_defaults():
    x = compute_phase_inputs([])
    assert x.pct_change_24h == 0.0
    assert x.days_since_last_pump == 9999
    assert 0.0 <= x.daily_close_pos_in_range <= 1.0


def test_compute_inputs_pct_change_lookbacks():
    # 26h of bars: 0..1559 minutes; first 1.0, last 2.0 -> +100% over 24h.
    history = [_kbar(i * 60_000, 1.0 + 1.0 * i / 1559) for i in range(1560)]
    x = compute_phase_inputs(history)
    assert math.isclose(x.pct_change_24h, (history[-1].close - history[-BARS_PER_DAY_1M - 1].close) / history[-BARS_PER_DAY_1M - 1].close, abs_tol=1e-9)
    assert x.pct_change_1h > 0
    # 6h delta exists too.
    assert x.pct_change_6h > 0


def test_compute_inputs_fewer_bars_uses_oldest_as_reference():
    """If history is shorter than the 24h lookback, pct_change_24h is
    computed against the very first bar — clamping to available data
    rather than NaN'ing out."""
    history = [_kbar(0, 1.0), _kbar(60_000, 1.5)]
    x = compute_phase_inputs(history)
    assert math.isclose(x.pct_change_24h, 0.5)


def test_compute_inputs_idle_days_count():
    # Synthetic 4-day stream with last_pump_idx 1 day before end.
    history = [_kbar(i * 60_000, 1.0) for i in range(BARS_PER_DAY_1M * 4)]
    last_pump_idx = len(history) - BARS_PER_DAY_1M - 1
    x = compute_phase_inputs(history, last_pump_idx=last_pump_idx)
    assert x.days_since_last_pump == 1


# ----------------------- runner ----------------------- #


def _build_pump_stream() -> list[KlineBar]:
    """Synthesize a stream that walks ACC -> RAMP -> PARABOLIC -> BLOWOFF -> CRASH.

    All ts in 1m increments. Volumes inject a clean z-score spike at the
    pump. Prices follow a rough sigmoid + crash.
    """
    bars: list[KlineBar] = []
    ts = 0
    step = 60_000

    # 200 bars of calm @ vol=1.0, price ~ 1.0
    for _ in range(200):
        bars.append(_kbar(ts, 1.0, vol=1.0))
        ts += step

    # 60 bars of ramp: vol explodes 30x, price climbs from 1.0 to 1.6.
    for i in range(60):
        p = 1.0 + 0.6 * (i / 60)
        bars.append(_kbar(ts, p, vol=30.0))
        ts += step

    # 60 bars of parabolic: vol stays high, price goes to 3.0.
    for i in range(60):
        p = 1.6 + 1.4 * (i / 60)
        bars.append(_kbar(ts, p, vol=40.0))
        ts += step

    # 60 bars of blow-off: long upper wicks; close drifts down.
    for i in range(60):
        c = 3.0 - 1.0 * (i / 60)
        bars.append(_kbar(ts, c, hi=c + 1.5, lo=c - 0.05, vol=20.0))
        ts += step

    # 60 bars of crash: each bar drops 1%.
    p = 2.0
    for _ in range(60):
        p *= 0.99
        bars.append(_kbar(ts, p, hi=p * 1.005, lo=p * 0.95, vol=10.0))
        ts += step

    return bars


def test_runner_produces_expected_phase_sequence():
    bars = _build_pump_stream()
    runner = BacktestRunner(thresholds=PhaseThresholds(
        # Synthetic stream uses gentler vol/price moves than real meme
        # coins; loosen thresholds so the test exercises the full FSM.
        # Real-world calibration is the trainer's job (Phase 4).
        ramp_min_vol_z=2.0,
        ramp_min_24h_pct=0.10,
        ramp_max_24h_pct=5.0,
        parabolic_min_vol_z=2.0,
        parabolic_min_6h_pct=0.20,
        blowoff_min_upper_wick_to_body=1.0,
        blowoff_max_close_pos_in_range=0.95,
        crash_min_1h_drop_pct=0.05,
    ))
    tagged: list[PhaseTaggedBar] = runner.run(bars)
    assert len(tagged) == len(bars)

    phases_seen = {pb.phase for pb in tagged}
    # We must walk through at least RAMP, PARABOLIC, and CRASH.
    assert PumpPhase.RAMP in phases_seen
    assert PumpPhase.PARABOLIC in phases_seen
    assert PumpPhase.CRASH in phases_seen


def test_runner_summary_reports_phase_counts_and_transitions():
    bars = _build_pump_stream()
    runner = BacktestRunner(thresholds=PhaseThresholds(
        ramp_min_vol_z=2.0,
        ramp_min_24h_pct=0.10,
        ramp_max_24h_pct=5.0,
        parabolic_min_vol_z=2.0,
        parabolic_min_6h_pct=0.20,
        blowoff_min_upper_wick_to_body=1.0,
        blowoff_max_close_pos_in_range=0.95,
        crash_min_1h_drop_pct=0.05,
    ))
    runner.run(bars)
    summary = runner.summary()
    assert "phase_counts" in summary
    assert "transitions" in summary
    # Phase counts add up to bar count (every bar is tagged).
    total = sum(summary["phase_counts"].values())
    assert total == len(bars)
    # We must have observed at least one transition off ACCUMULATION.
    assert any(t["from"] == "accumulation" for t in summary["transitions"])


def test_runner_accepts_raw_ohlcv_lists():
    """ccxt returns [ts, o, h, l, c, v] arrays — the runner must coerce."""
    raw = [[i * 60_000, 1.0, 1.01, 0.99, 1.0, 1.0] for i in range(50)]
    runner = BacktestRunner()
    tagged = runner.run(raw)
    assert len(tagged) == 50
    assert tagged[0].close == 1.0


def test_runner_skips_malformed_bars():
    raw = [[0, 1, 1, 1, 1, 1], "garbage", [1, 2], [60_000, 1, 1, 1, 1, 1]]
    runner = BacktestRunner()
    tagged = runner.run(raw)
    assert len(tagged) == 2  # only the two well-formed entries
