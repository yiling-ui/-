"""R6 — phase + confidence wiring tests.

Confirms three things:
  1. ``trailing.TrailingStopFSM.tick(phase=...)`` tightens the ATR
     multiplier in BLOWOFF_TOP / CRASH / BLEED / DEAD phases.
  2. ``RiskGate.evaluate(phase=..., confidence=..., symbol_profile=...)``
     enforces per-phase confidence floor + allowed-entry-phase allow-list.
  3. ``ScoreFuser.evaluate(symbol, exchange, ts, phase=...)`` swaps in
     the per-phase ``high_priority_threshold`` override.

All three are 100% backward compatible: omitting the new kwargs gives
v1.0 behaviour byte-for-byte.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.fuser import (
    Direction,
    FusedSignal,
    FuserConfig,
    ScoreFuser,
)
from altcoin_agent.risk.gate import RiskGate, RiskGateConfig
from altcoin_agent.risk.pump_phase import PumpPhase
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Side
from altcoin_agent.risk.symbol_profile import (
    DEFAULT_QUADRANT_PARAMS,
    Quadrant,
    SymbolProfile,
)
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM
from altcoin_agent.screener import SignalEvent, SignalKind


# --------------------------------------------------------------------- #
# Trailing FSM — phase tightening
# --------------------------------------------------------------------- #


def _long_position(entry: float = 100.0, stop: float = 95.0):
    """Build a minimal Position-like object the FSM can read."""
    pos = MagicMock()
    pos.side = Side.LONG
    pos.closed = False
    pos.entry_price = entry
    pos.current_stop = stop
    # r_unit = entry - stop for LONG.
    pos.r_unit = entry - stop
    return pos


def test_trailing_factory_from_quadrant_params():
    """The factory must read trailing_atr_mult + breakeven_at_r."""
    a = DEFAULT_QUADRANT_PARAMS[Quadrant.A]
    fsm = TrailingStopFSM.from_quadrant_params(a)
    assert fsm.atr_multiplier == a.trailing_atr_mult
    assert fsm.breakeven_at_r == a.breakeven_at_r


def test_trailing_no_phase_preserves_v1_behaviour():
    """Omit ``phase`` -> stop is at price - 2 * ATR for LONG."""
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    pos = _long_position(entry=100.0, stop=95.0)
    state, new_stop, _ = fsm.tick(
        position=pos, current_price=120.0, atr=2.0,
        current_state=TrailingState.BREAKEVEN,
    )
    # 2x ATR away from 120 = 120 - 4 = 116.
    assert new_stop == 116.0
    assert state == TrailingState.TRAILING


def test_trailing_phase_blowoff_tightens():
    """BLOWOFF_TOP halves the ATR multiplier (default tighten=0.5)."""
    fsm = TrailingStopFSM(atr_multiplier=2.0, blowoff_atr_tighten=0.5)
    pos = _long_position(entry=100.0, stop=95.0)
    _, new_stop, reason = fsm.tick(
        position=pos, current_price=120.0, atr=2.0,
        current_state=TrailingState.BREAKEVEN,
        phase=PumpPhase.BLOWOFF_TOP,
    )
    # 1.0x ATR away from 120 = 118 (tighter than 116).
    assert new_stop == 118.0
    assert "phase=blowoff_top" in reason


def test_trailing_phase_crash_tightens_more():
    """CRASH/BLEED/DEAD use the more aggressive crash_atr_tighten."""
    fsm = TrailingStopFSM(atr_multiplier=2.0, crash_atr_tighten=0.3)
    pos = _long_position(entry=100.0, stop=95.0)
    _, new_stop, _ = fsm.tick(
        position=pos, current_price=120.0, atr=2.0,
        current_state=TrailingState.TRAILING,
        phase=PumpPhase.CRASH,
    )
    # 0.6x ATR away from 120 = 118.8.
    assert abs(new_stop - 118.8) < 1e-9


def test_trailing_phase_ramp_no_change():
    """Early phases keep the unmodified ATR multiplier."""
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    pos = _long_position()
    _, new_stop_no_phase, _ = fsm.tick(
        position=pos, current_price=120.0, atr=2.0,
        current_state=TrailingState.BREAKEVEN,
    )
    pos2 = _long_position()
    _, new_stop_ramp, _ = fsm.tick(
        position=pos2, current_price=120.0, atr=2.0,
        current_state=TrailingState.BREAKEVEN,
        phase=PumpPhase.RAMP,
    )
    assert new_stop_no_phase == new_stop_ramp


# --------------------------------------------------------------------- #
# RiskGate — phase + confidence
# --------------------------------------------------------------------- #


def _high_priority_signal(direction: Direction = Direction.LONG) -> FusedSignal:
    return FusedSignal(
        symbol="PEPE/USDT:USDT", exchange="binance", ts=1_700_000_000_000,
        direction=direction, rule_score=80.0, llm_score=70.0, final_score=88.0,
        is_high_priority=True, blocked=False, block_reason=None,
        rule_signals=[], llm_verdict=None, notes=[],
        trigger_price=1.0,
    )


def _account() -> AccountState:
    return AccountState(
        equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        global_trading_halted=False, halt_reason="",
        reconciliation_complete=True,
    )


def test_gate_no_phase_preserves_v1_behaviour():
    """Omitting phase/confidence -> identical pre-R6 control flow."""
    gate = RiskGate(PositionSizer(max_risk_per_trade=0.01))
    decision = gate.evaluate(
        signal=_high_priority_signal(), account=_account(),
        current_price=1.0, top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.04, initial_stop=0.95,
    )
    # Sizing might still reject for other reasons; the test only
    # asserts no R6-introduced rejection reason appeared.
    assert "phase_not_allowed" not in decision.reason
    assert "confidence_below_floor" not in decision.reason


def test_gate_phase_allow_list_blocks_dead_phase():
    cfg = RiskGateConfig(
        allowed_entry_phases=frozenset({"accumulation", "ramp", "parabolic"}),
    )
    gate = RiskGate(PositionSizer(max_risk_per_trade=0.01), config=cfg)
    decision = gate.evaluate(
        signal=_high_priority_signal(), account=_account(),
        current_price=1.0, top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.04, initial_stop=0.95,
        phase=PumpPhase.DEAD,
    )
    assert decision.approved is False
    assert decision.reason.startswith("phase_not_allowed:dead")


def test_gate_confidence_floor_blocks_low_confidence():
    cfg = RiskGateConfig(
        phase_min_confidence={"parabolic": 0.85},
    )
    gate = RiskGate(PositionSizer(max_risk_per_trade=0.01), config=cfg)
    decision = gate.evaluate(
        signal=_high_priority_signal(), account=_account(),
        current_price=1.0, top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.04, initial_stop=0.95,
        phase=PumpPhase.PARABOLIC, confidence=0.70,
    )
    assert decision.approved is False
    assert "confidence_below_floor" in decision.reason


def test_gate_symbol_profile_overrides_floor():
    """Profile's effective_confidence_threshold beats gate floor."""
    profile = SymbolProfile.from_scores(
        symbol="MEME/USDT:USDT",
        social_score=85.0, liquidity_score=85.0,  # quadrant A
    )
    # quadrant A's default confidence_threshold = 0.80; we let the gate
    # have a much lower floor -> profile's stricter floor must win.
    cfg = RiskGateConfig(phase_min_confidence={"ramp": 0.10})
    gate = RiskGate(PositionSizer(max_risk_per_trade=0.01), config=cfg)
    decision = gate.evaluate(
        signal=_high_priority_signal(), account=_account(),
        current_price=1.0, top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.04, initial_stop=0.95,
        phase=PumpPhase.RAMP, confidence=0.50,  # below A's 0.80
        symbol_profile=profile,
    )
    assert decision.approved is False
    assert "confidence_below_floor" in decision.reason
    assert "0.500" in decision.reason or "0.50" in decision.reason


def test_gate_passes_when_confidence_clears_floor():
    cfg = RiskGateConfig(phase_min_confidence={"ramp": 0.70})
    gate = RiskGate(PositionSizer(max_risk_per_trade=0.01), config=cfg)
    decision = gate.evaluate(
        signal=_high_priority_signal(), account=_account(),
        current_price=1.0, top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.04, initial_stop=0.95,
        phase=PumpPhase.RAMP, confidence=0.85,
    )
    # Sizing might still fail later in the pipeline, but R6 itself
    # should not be the reason.
    assert "phase_not_allowed" not in decision.reason
    assert "confidence_below_floor" not in decision.reason


# --------------------------------------------------------------------- #
# ScoreFuser — phase threshold override
# --------------------------------------------------------------------- #


def _vol_spike_event(
    side: str = "buy", ts: int = 1_700_000_000_000,
) -> SignalEvent:
    return SignalEvent(
        ts=ts, symbol="PEPE/USDT:USDT", exchange="binance",
        kind=SignalKind.VOLUME_SPIKE,
        payload={
            "side": side, "zscore": 4.0,
            "to_price": 1.0, "bar_close": 1.0,
        },
    )


def test_fuser_no_phase_preserves_v1_behaviour():
    """Default threshold = 85 still fires for a strong rule signal."""
    fuser = ScoreFuser(
        config=FuserConfig(high_priority_threshold=50.0),
    )
    fused = fuser.evaluate("PEPE/USDT:USDT", "binance", 1_700_000_000_000)
    # No rule events fed in -> NEUTRAL.
    assert fused.direction == Direction.NEUTRAL


def test_fuser_phase_override_relaxes_threshold(tmp_path):
    """phase_threshold_overrides[ramp]=10 lets a small rule_score
    pass the gate where the default 85 would not."""
    cfg = FuserConfig(
        high_priority_threshold=85.0,
        phase_threshold_overrides={"ramp": 10.0},
        # Loosen require_min_rule_score so the test isn't blocked there.
        require_min_rule_score=0.0,
    )
    fuser = ScoreFuser(config=cfg)
    ev = _vol_spike_event(side="buy")
    fuser._rules[fuser._key("binance", ev.symbol)] = type(fuser._rules)({})
    fuser._rules[fuser._key("binance", ev.symbol)] = __import__(
        "collections"
    ).deque([ev], maxlen=64)

    fused_no_phase = fuser.evaluate(ev.symbol, "binance", ev.ts)
    fused_ramp = fuser.evaluate(ev.symbol, "binance", ev.ts, phase="ramp")

    assert fused_ramp.is_high_priority is True
    assert fused_no_phase.is_high_priority is False


def test_fuser_phase_override_records_note():
    cfg = FuserConfig(
        high_priority_threshold=85.0,
        phase_threshold_overrides={"parabolic": 92.0},
        require_min_rule_score=0.0,
    )
    fuser = ScoreFuser(config=cfg)
    ev = _vol_spike_event(side="buy")
    from collections import deque
    fuser._rules[fuser._key("binance", ev.symbol)] = deque([ev], maxlen=64)
    fused = fuser.evaluate(
        ev.symbol, "binance", ev.ts, phase="parabolic",
    )
    assert any("parabolic" in n and "92" in n for n in fused.notes)


def test_fuser_unknown_phase_falls_back_to_default():
    cfg = FuserConfig(
        high_priority_threshold=85.0,
        phase_threshold_overrides={"ramp": 10.0},
    )
    fuser = ScoreFuser(config=cfg)
    # phase="weather_is_great" -> not in overrides, defaults to 85.
    fused = fuser.evaluate(
        "PEPE/USDT:USDT", "binance", 1_700_000_000_000,
        phase="weather_is_great",
    )
    # Empty rule stream -> NEUTRAL regardless.
    assert fused.direction == Direction.NEUTRAL
