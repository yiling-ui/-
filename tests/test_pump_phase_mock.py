"""tests/test_pump_phase_mock.py — QUADRANT Phase 1.C / 3 coverage."""

from __future__ import annotations

from altcoin_agent.risk.pump_phase import (
    KlineBar,
    PhaseInputs,
    PhaseThresholds,
    PumpPhase,
    PumpPhaseFSM,
)

# ----------------------- helpers ----------------------- #


def _bar(ts_ms: int, close: float, vol_z: float = 0.0,
         hi: float | None = None, lo: float | None = None,
         o: float | None = None, vol: float = 1.0) -> KlineBar:
    return KlineBar(
        ts_ms=ts_ms,
        open=o if o is not None else close,
        high=hi if hi is not None else close,
        low=lo if lo is not None else close,
        close=close,
        volume=vol,
        vol_z_score=vol_z,
    )


def _calm_inputs(**overrides) -> PhaseInputs:
    base = PhaseInputs(
        pct_change_24h=0.0,
        pct_change_6h=0.0,
        pct_change_1h=0.0,
        daily_upper_wick_to_body_ratio=0.0,
        daily_close_pos_in_range=0.5,
        intrabar_lower_wick_pct=0.0,
        realized_vol_pct_30d=0.0,
        days_since_last_pump=0,
    )
    return PhaseInputs(**{**base.__dict__, **overrides})


# ----------------------- cold-start ----------------------- #


def test_cold_start_stays_accumulation():
    """First N bars must not transition out, even if inputs scream RAMP."""
    fsm = PumpPhaseFSM(min_history_bars=5)
    ramp_inputs = _calm_inputs(pct_change_24h=0.40)
    for i in range(4):
        s = fsm.advance(_bar(i, 1.0, vol_z=10.0), ramp_inputs)
        assert s is PumpPhase.ACCUMULATION
    # 5th bar — now we have enough history to fire.
    s = fsm.advance(_bar(5, 1.4, vol_z=10.0), ramp_inputs)
    assert s is PumpPhase.RAMP


def test_late_or_duplicate_bars_ignored():
    """ccxt.pro reconnect replays must not advance the FSM out of order."""
    fsm = PumpPhaseFSM(min_history_bars=2)
    fsm.advance(_bar(100, 1.0), _calm_inputs())
    fsm.advance(_bar(200, 1.0), _calm_inputs())
    state_before = fsm.state
    # Replay the older bar — should be a no-op.
    s = fsm.advance(_bar(150, 99.0, vol_z=99.0), _calm_inputs(pct_change_24h=10))
    assert s is state_before


# ----------------------- canonical sequence ----------------------- #


def test_full_pump_cycle_sequence():
    """Walk through ACC -> RAMP -> PARABOLIC -> BLOWOFF_TOP -> CRASH -> BLEED -> DEAD."""
    fsm = PumpPhaseFSM(min_history_bars=2)

    # warm up
    fsm.advance(_bar(0, 1.0), _calm_inputs())
    fsm.advance(_bar(1, 1.0), _calm_inputs())

    # ACC -> RAMP (vol_z=4, +50% 24h)
    s = fsm.advance(
        _bar(2, 1.5, vol_z=4.0),
        _calm_inputs(pct_change_24h=0.50),
    )
    assert s is PumpPhase.RAMP

    # RAMP -> PARABOLIC (vol_z=7, +120% 6h)
    s = fsm.advance(
        _bar(3, 3.0, vol_z=7.0),
        _calm_inputs(pct_change_24h=2.0, pct_change_6h=1.20),
    )
    assert s is PumpPhase.PARABOLIC

    # PARABOLIC -> BLOWOFF_TOP (long upper wick + low close-in-range)
    s = fsm.advance(
        _bar(4, 2.5, vol_z=5.0),
        _calm_inputs(
            pct_change_24h=2.0, pct_change_6h=1.20,
            daily_upper_wick_to_body_ratio=2.5,
            daily_close_pos_in_range=0.30,
        ),
    )
    assert s is PumpPhase.BLOWOFF_TOP

    # BLOWOFF_TOP -> CRASH (-35% in 1h)
    s = fsm.advance(
        _bar(5, 1.6, vol_z=4.0),
        _calm_inputs(pct_change_1h=-0.35),
    )
    assert s is PumpPhase.CRASH

    # CRASH -> BLEED (idle days >= bleed threshold)
    s = fsm.advance(
        _bar(6, 1.5),
        _calm_inputs(days_since_last_pump=5),
    )
    assert s is PumpPhase.BLEED

    # BLEED -> DEAD (idle >= 14 days)
    s = fsm.advance(
        _bar(7, 1.5),
        _calm_inputs(days_since_last_pump=20),
    )
    assert s is PumpPhase.DEAD

    # Transitions log captured every hop.
    transitions = [(t.from_phase, t.to_phase) for t in fsm.transitions]
    assert transitions == [
        (PumpPhase.ACCUMULATION, PumpPhase.RAMP),
        (PumpPhase.RAMP, PumpPhase.PARABOLIC),
        (PumpPhase.PARABOLIC, PumpPhase.BLOWOFF_TOP),
        (PumpPhase.BLOWOFF_TOP, PumpPhase.CRASH),
        (PumpPhase.CRASH, PumpPhase.BLEED),
        (PumpPhase.BLEED, PumpPhase.DEAD),
    ]


def test_crash_short_circuits_from_any_phase():
    """A sudden 1h drop should jump straight to CRASH from PARABOLIC."""
    fsm = PumpPhaseFSM(min_history_bars=2)
    fsm.advance(_bar(0, 1.0), _calm_inputs())
    fsm.advance(_bar(1, 1.0), _calm_inputs())
    # Push to PARABOLIC.
    fsm.advance(_bar(2, 1.5, vol_z=4.0), _calm_inputs(pct_change_24h=0.5))
    fsm.advance(
        _bar(3, 3.0, vol_z=7.0),
        _calm_inputs(pct_change_24h=2.0, pct_change_6h=1.2),
    )
    assert fsm.state is PumpPhase.PARABOLIC
    s = fsm.advance(_bar(4, 1.0), _calm_inputs(pct_change_1h=-0.40))
    assert s is PumpPhase.CRASH


def test_dead_can_reawaken_via_fresh_ramp():
    """After DEAD, a new vol spike resets to ACCUMULATION (per plan).

    The multi-hop chase will keep going DEAD -> ACCUMULATION -> RAMP in
    the same bar if the inputs warrant it (vol_z + 24h pct both pass).
    What we care about for the plan is that the *first* transition is
    out of DEAD into ACCUMULATION (the reawaken edge); the FSM may
    then immediately fire RAMP off the same bar.
    """
    fsm = PumpPhaseFSM(min_history_bars=1)
    fsm.advance(_bar(0, 1.0), _calm_inputs())
    fsm.state = PumpPhase.DEAD
    s = fsm.advance(
        _bar(1, 1.5, vol_z=4.0),
        _calm_inputs(pct_change_24h=0.50),
    )
    # Final state may be RAMP (multi-hop), but the canonical reawaken
    # edge DEAD -> ACCUMULATION must have been recorded.
    assert s in (PumpPhase.ACCUMULATION, PumpPhase.RAMP)
    edges = [(t.from_phase, t.to_phase) for t in fsm.transitions]
    assert (PumpPhase.DEAD, PumpPhase.ACCUMULATION) in edges


# ----------------------- threshold sensitivity ----------------------- #


def test_custom_thresholds_overridable():
    custom = PhaseThresholds(ramp_min_vol_z=2.0, ramp_min_24h_pct=0.10)
    fsm = PumpPhaseFSM(thresholds=custom, min_history_bars=2)
    fsm.advance(_bar(0, 1.0), _calm_inputs())
    fsm.advance(_bar(1, 1.0), _calm_inputs())
    s = fsm.advance(_bar(2, 1.2, vol_z=2.0), _calm_inputs(pct_change_24h=0.15))
    assert s is PumpPhase.RAMP


def test_24h_pct_outside_band_does_not_fire_ramp():
    """+200%+ on the 24h is too high — that's already PARABOLIC territory.

    The plan caps the RAMP band at +200% so a hyper-spike on a fresh
    bar can't be mislabeled as RAMP.
    """
    fsm = PumpPhaseFSM(min_history_bars=2)
    fsm.advance(_bar(0, 1.0), _calm_inputs())
    fsm.advance(_bar(1, 1.0), _calm_inputs())
    s = fsm.advance(_bar(2, 5.0, vol_z=5.0), _calm_inputs(pct_change_24h=4.0))
    assert s is not PumpPhase.RAMP


# ----------------------- reset ----------------------- #


def test_reset_clears_state_and_history():
    fsm = PumpPhaseFSM(min_history_bars=2)
    for i in range(10):
        fsm.advance(_bar(i, 1.0 + i * 0.1, vol_z=5.0),
                    _calm_inputs(pct_change_24h=0.5))
    assert fsm.state is not PumpPhase.ACCUMULATION
    fsm.reset()
    assert fsm.state is PumpPhase.ACCUMULATION
    assert fsm.transitions == []
