"""Integration tests for RiskGate's anti-chase / vol-kill plumbing.

The price-tape gates are O(1) and run BEFORE the networked checks so a
rejection saves a venue round-trip. These tests pin two contracts:

  * When the tape says "the move already happened", the gate rejects
    with reason ``chase_too_late:...`` and ALL downstream checks
    (live-quote fetch, sizing, leverage) are skipped.
  * When the tape is in chop, ``vol_kill_active:...`` is returned the
    same way.

Negative cases prove that a calm tape lets the original 9 checks run
unchanged.
"""

from __future__ import annotations

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk import (
    AccountState,
    PositionSizer,
    RiskGate,
    RiskGateConfig,
)


def _signal(direction: Direction = Direction.LONG, score: float = 95.0) -> FusedSignal:
    return FusedSignal(
        symbol="RAVEUSDT", exchange="binance", ts=1,
        direction=direction, rule_score=80.0, llm_score=85.0,
        final_score=score, is_high_priority=True, blocked=False,
        block_reason=None, trigger_price=1.0,
    )


def _account() -> AccountState:
    a = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
    a.reconciliation_complete = True
    return a


def _gate() -> RiskGate:
    return RiskGate(
        PositionSizer(),
        RiskGateConfig(min_liquidity_usdt=100_000.0),
    )


# --------------------------------------------------------------------- #
# Anti-chase
# --------------------------------------------------------------------- #


def test_gate_rejects_long_after_runaway_pump() -> None:
    """+5% in 30 s, cap=2.5% -> reject ``chase_too_late:...`` BEFORE
    any other check. Note we feed an obviously approvable signal so
    the only thing that can make the gate reject is anti-chase."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025,
    ))
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 1.050, ts_ms=1_000_000)

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.050,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert not decision.approved
    assert decision.reason.startswith("chase_too_late:")
    # Anti-chase fires BEFORE sizing -> sizing fields are None.
    assert decision.size is None


def test_gate_lets_calm_tape_through() -> None:
    """A 0.4% move in 30 s is well below the 2.5% cap; the gate goes
    through the original 9-check flow and approves."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025,
    ))
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 1.004, ts_ms=1_000_000)

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.004,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert decision.approved, decision.reason
    assert decision.size is not None and decision.size > 0


def test_gate_short_anti_chase_after_runaway_dump() -> None:
    """Mirror of LONG case: -5% in 30s on SHORT -> reject."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025,
    ))
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 0.940, ts_ms=1_000_000)

    decision = _gate().evaluate(
        signal=_signal(direction=Direction.SHORT),
        account=_account(),
        current_price=0.940,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=1.05,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert not decision.approved
    assert decision.reason.startswith("chase_too_late:")


# --------------------------------------------------------------------- #
# Vol-kill
# --------------------------------------------------------------------- #


def test_gate_vol_kill_blocks_chop() -> None:
    """8.x% range over 60 s -> reject ``vol_kill_active:...``. We
    deliberately keep the latest price near the start so anti-chase
    is NOT triggered; only vol-kill fires."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.10,   # disabled
        vol_kill_window_ms=60_000, vol_kill_range_pct=0.08,
    ))
    # Whipsaw: 1.00 -> 1.05 -> 0.96 -> 1.005
    tape.observe("RAVEUSDT", 1.000, ts_ms=940_000)
    tape.observe("RAVEUSDT", 1.050, ts_ms=960_000)
    tape.observe("RAVEUSDT", 0.960, ts_ms=980_000)
    tape.observe("RAVEUSDT", 1.005, ts_ms=1_000_000)

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.005,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert not decision.approved
    assert decision.reason.startswith("vol_kill_active:")


def test_gate_vol_kill_stays_quiet_in_calm_tape() -> None:
    """Range 0.4% << 8% cap -> approve normally."""
    tape = PriceTape(cfg=PriceTapeConfig(
        vol_kill_window_ms=60_000, vol_kill_range_pct=0.08,
    ))
    for i, p in enumerate([1.00, 1.001, 1.003, 1.002]):
        tape.observe("RAVEUSDT", p, ts_ms=940_000 + i * 15_000)

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.002,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert decision.approved, decision.reason


# --------------------------------------------------------------------- #
# Backwards compatibility
# --------------------------------------------------------------------- #


def test_gate_without_tape_keeps_legacy_behaviour() -> None:
    """Old call sites (no price_tape kwarg) must work unchanged.

    Important: the existing 238-test suite exercises this path
    extensively; this test simply pins it so nobody removes the
    Optional default."""
    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.0,
        top5_depth_usdt=400_000.0,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
    )
    assert decision.approved


def test_gate_anti_chase_evaluates_before_live_quote() -> None:
    """Sanity: anti-chase rejection happens with no networked side
    effects. We use a tiny depth that would otherwise reject as
    insufficient_liquidity, and verify we get chase_too_late instead
    -- proving the order of checks."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000, anti_chase_max_move_pct=0.025,
    ))
    tape.observe("RAVEUSDT", 1.000, ts_ms=970_000)
    tape.observe("RAVEUSDT", 1.080, ts_ms=1_000_000)    # +8%

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.080,
        top5_depth_usdt=10.0,    # would normally trigger insufficient_liquidity
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=1_000_000,
        price_tape=tape,
    )
    assert not decision.approved
    # The first check that fires wins.
    assert decision.reason.startswith("chase_too_late:")
    assert "insufficient_liquidity" not in decision.reason
