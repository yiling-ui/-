"""Mock tests for the millisecond price tape and its anti-chase /
vol-kill gates.

The contract is:

  * ``observe`` is cheap O(1) and bounded in memory.
  * ``anti_chase_breach`` returns True iff the price moved past the
    cap *in the trade direction* during the last window.
  * ``vol_kill_breach`` returns True iff hi-lo range over the last
    window exceeded the cap.
  * Cold start (< 2 samples) fails OPEN — we don't want a freshly
    booted daemon to refuse every signal.
  * gc_stale evicts symbols whose last sample is too old.
"""

from __future__ import annotations

import pytest

from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk.state import Side

# --------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------- #


def test_config_rejects_zero_or_negative_windows() -> None:
    with pytest.raises(ValueError, match="anti_chase_window_ms"):
        PriceTapeConfig(anti_chase_window_ms=0)
    with pytest.raises(ValueError, match="anti_chase_window_ms"):
        PriceTapeConfig(anti_chase_window_ms=-1)
    with pytest.raises(ValueError, match="vol_kill_window_ms"):
        PriceTapeConfig(vol_kill_window_ms=0)


def test_config_rejects_zero_or_negative_caps() -> None:
    with pytest.raises(ValueError, match="anti_chase_max_move_pct"):
        PriceTapeConfig(anti_chase_max_move_pct=0)
    with pytest.raises(ValueError, match="vol_kill_range_pct"):
        PriceTapeConfig(vol_kill_range_pct=-0.01)


def test_config_accepts_inf_to_disable() -> None:
    cfg = PriceTapeConfig(
        anti_chase_max_move_pct=float("inf"),
        vol_kill_range_pct=float("inf"),
    )
    assert cfg.anti_chase_max_move_pct == float("inf")


def test_config_rejects_tiny_buffer() -> None:
    with pytest.raises(ValueError, match="max_samples_per_symbol"):
        PriceTapeConfig(max_samples_per_symbol=4)


# --------------------------------------------------------------------- #
# observe / latest / reset
# --------------------------------------------------------------------- #


def test_observe_drops_zero_or_negative_prices() -> None:
    tape = PriceTape()
    tape.observe("RAVEUSDT", 0.0, ts_ms=1_000)
    tape.observe("RAVEUSDT", -1.0, ts_ms=1_001)
    assert tape.latest("RAVEUSDT") is None


def test_observe_keeps_only_max_samples() -> None:
    cfg = PriceTapeConfig(max_samples_per_symbol=8)
    tape = PriceTape(cfg=cfg)
    for i in range(20):
        tape.observe("RAVEUSDT", 1.0 + i * 0.001, ts_ms=1_000 + i)
    # Underlying buffer is bounded.
    assert len(tape._samples["RAVEUSDT"]) == 8


def test_reset_clears_one_symbol_only() -> None:
    tape = PriceTape()
    tape.observe("A", 1.0, ts_ms=1_000)
    tape.observe("B", 2.0, ts_ms=1_000)
    tape.reset("A")
    assert tape.latest("A") is None
    assert tape.latest("B") == (1_000, 2.0)


# --------------------------------------------------------------------- #
# anti-chase
# --------------------------------------------------------------------- #


def test_anti_chase_cold_start_fails_open() -> None:
    """No samples at all -> False, 0.0. A fresh boot must not refuse
    every signal."""
    tape = PriceTape()
    breached, move = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_000,
    )
    assert breached is False
    assert move == 0.0


def test_anti_chase_single_sample_fails_open() -> None:
    """One sample alone cannot describe a move; fail open."""
    tape = PriceTape()
    tape.observe("RAVEUSDT", 1.0, ts_ms=1_000_000)
    breached, _ = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_010,
    )
    assert breached is False


def test_anti_chase_breach_long_when_price_ran_up() -> None:
    """+5% in the last 30 s with cap=2.5% -> reject."""
    cfg = PriceTapeConfig(anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025)
    tape = PriceTape(cfg=cfg)
    # 30s ago: 1.000; now: 1.050  -> +5%
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 1.020, ts_ms=985_000)
    tape.observe("RAVEUSDT", 1.050, ts_ms=1_000_000)
    breached, move = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_000,
    )
    assert breached is True
    assert move == pytest.approx(0.05, rel=1e-3)


def test_anti_chase_long_does_not_breach_when_price_dropped() -> None:
    """Favourable move (we're entering on a dip) is never a chase."""
    tape = PriceTape()
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 0.950, ts_ms=1_000_000)    # -5%
    breached, move = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_000,
    )
    assert breached is False
    assert move < 0


def test_anti_chase_short_breach_when_price_dropped() -> None:
    """SHORT after a dump that already happened -> reject."""
    cfg = PriceTapeConfig(anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025)
    tape = PriceTape(cfg=cfg)
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 0.940, ts_ms=1_000_000)    # -6%
    breached, move = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.SHORT, now_ms=1_000_000,
    )
    assert breached is True
    assert move == pytest.approx(0.06, rel=1e-3)


def test_anti_chase_ignores_samples_outside_window() -> None:
    """A move that happened 2 minutes ago doesn't count if window=30s."""
    cfg = PriceTapeConfig(anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025)
    tape = PriceTape(cfg=cfg)
    # 2 min ago: 1.000 (old, outside window)
    tape.observe("RAVEUSDT", 1.000, ts_ms=880_000)
    # Inside window: stable.
    tape.observe("RAVEUSDT", 1.040, ts_ms=970_000)
    tape.observe("RAVEUSDT", 1.045, ts_ms=1_000_000)    # +0.5% in 30s
    breached, _ = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_000,
    )
    assert breached is False


def test_anti_chase_disabled_via_inf_cap() -> None:
    """Operator opts out by setting cap=inf."""
    cfg = PriceTapeConfig(anti_chase_max_move_pct=float("inf"))
    tape = PriceTape(cfg=cfg)
    tape.observe("RAVEUSDT", 1.0, ts_ms=970_000)
    tape.observe("RAVEUSDT", 100.0, ts_ms=1_000_000)    # +9900%
    breached, _ = tape.anti_chase_breach(
        symbol="RAVEUSDT", side=Side.LONG, now_ms=1_000_000,
    )
    assert breached is False


# --------------------------------------------------------------------- #
# vol-kill
# --------------------------------------------------------------------- #


def test_vol_kill_breach_when_range_exceeds_cap() -> None:
    """Hi=1.10, lo=1.00 over 60s with cap=8% -> 10% > 8% -> reject."""
    cfg = PriceTapeConfig(vol_kill_window_ms=60_000, vol_kill_range_pct=0.08)
    tape = PriceTape(cfg=cfg)
    for i, p in enumerate([1.00, 1.10, 1.05, 1.08]):
        tape.observe("RAVEUSDT", p, ts_ms=940_000 + i * 15_000)
    breached, rng = tape.vol_kill_breach(symbol="RAVEUSDT", now_ms=1_000_000)
    assert breached is True
    assert rng == pytest.approx(0.10, rel=1e-3)


def test_vol_kill_no_breach_in_calm_tape() -> None:
    cfg = PriceTapeConfig(vol_kill_window_ms=60_000, vol_kill_range_pct=0.08)
    tape = PriceTape(cfg=cfg)
    for i, p in enumerate([1.00, 1.01, 1.02, 1.015]):    # 2% range
        tape.observe("RAVEUSDT", p, ts_ms=940_000 + i * 15_000)
    breached, rng = tape.vol_kill_breach(symbol="RAVEUSDT", now_ms=1_000_000)
    assert breached is False
    assert rng < 0.08


def test_vol_kill_cold_start_fails_open() -> None:
    tape = PriceTape()
    breached, _ = tape.vol_kill_breach(symbol="RAVEUSDT", now_ms=1_000_000)
    assert breached is False


# --------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------- #


def test_gc_stale_drops_dormant_symbols() -> None:
    tape = PriceTape()
    tape.observe("ACTIVE", 1.0, ts_ms=1_000_000)
    tape.observe("STALE", 2.0, ts_ms=200_000)
    evicted = tape.gc_stale(now_ms=1_000_000, ttl_ms=600_000)
    assert evicted == 1
    assert tape.latest("ACTIVE") == (1_000_000, 1.0)
    assert tape.latest("STALE") is None
