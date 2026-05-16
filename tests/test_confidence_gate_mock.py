"""tests/test_confidence_gate_mock.py — QUADRANT Phase 1.D coverage."""

from __future__ import annotations

import pytest

from altcoin_agent.risk.confidence_gate import (
    DEFAULT_PHASE_MULTIPLIER,
    ConfidenceGate,
    ConfidenceWeights,
)
from altcoin_agent.risk.pump_phase import PumpPhase
from altcoin_agent.risk.symbol_profile import Quadrant, SymbolProfile


def _profile(
    quadrant: Quadrant = Quadrant.A,
    *,
    historical_win_rate: float = 0.7,
    scam_score: float = 0.0,
    confidence_threshold: float | None = None,
) -> SymbolProfile:
    return SymbolProfile(
        symbol="X",
        quadrant=quadrant,
        social_score=80,
        liquidity_score=80,
        historical_win_rate=historical_win_rate,
        scam_score=scam_score,
        confidence_threshold=confidence_threshold,
    )


# ----------------------- weights validation ----------------------- #


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConfidenceWeights(rule=0.5, llm=0.3, profile_history=0.1, learned=0.0)


# ----------------------- approval / rejection ----------------------- #


def test_high_inputs_in_a_quadrant_approve():
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=95.0,
        llm_score=0.92,
        learned_rules_score=0.85,
        profile=_profile(Quadrant.A),
        phase=PumpPhase.RAMP,
    )
    assert v.should_open is True
    assert v.observer_mode is False
    assert v.confidence >= 0.80
    assert v.threshold == 0.80


def test_low_inputs_reject_into_observer_mode():
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=30.0,
        llm_score=0.30,
        learned_rules_score=0.20,
        profile=_profile(Quadrant.A),
        phase=PumpPhase.RAMP,
    )
    assert v.should_open is False
    assert v.observer_mode is True
    assert v.confidence < v.threshold
    assert "below_threshold" in v.reason


def test_d_quadrant_threshold_is_0_90():
    """Even a strong D-quadrant signal needs >= 0.90 confidence."""
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=82.0,
        llm_score=0.80,
        learned_rules_score=0.75,
        profile=_profile(Quadrant.D),
        phase=PumpPhase.RAMP,
    )
    # D's threshold is 0.90; the linear blend stays <= 0.90 at these inputs.
    assert v.threshold == 0.90
    assert v.should_open is False


def test_per_symbol_threshold_override_is_respected():
    gate = ConfidenceGate()
    p = _profile(Quadrant.A, confidence_threshold=0.99)
    v = gate.evaluate(
        rule_score=95, llm_score=0.95, learned_rules_score=0.95,
        profile=p, phase=PumpPhase.PARABOLIC,
    )
    assert v.threshold == 0.99


def test_hard_min_threshold_floor_applied():
    gate = ConfidenceGate(hard_min_threshold=0.75)
    p = _profile(Quadrant.A, confidence_threshold=0.30)  # too lax
    v = gate.evaluate(
        rule_score=70, llm_score=0.70, learned_rules_score=0.5,
        profile=p, phase=PumpPhase.RAMP,
    )
    assert v.threshold == 0.75  # floor wins over the lax override


# ----------------------- phase + scam adjustments ----------------------- #


def test_phase_multiplier_lowers_dead_phase_confidence():
    gate = ConfidenceGate()
    p = _profile(Quadrant.A)
    ramp = gate.evaluate(
        rule_score=80, llm_score=0.80, learned_rules_score=0.80,
        profile=p, phase=PumpPhase.RAMP,
    )
    dead = gate.evaluate(
        rule_score=80, llm_score=0.80, learned_rules_score=0.80,
        profile=p, phase=PumpPhase.DEAD,
    )
    assert dead.confidence < ramp.confidence
    assert dead.components["phase_mult"] == DEFAULT_PHASE_MULTIPLIER[PumpPhase.DEAD]


def test_scam_score_penalises_confidence():
    gate = ConfidenceGate()
    clean = gate.evaluate(
        rule_score=80, llm_score=0.80, learned_rules_score=0.80,
        profile=_profile(Quadrant.A, scam_score=0.0), phase=PumpPhase.RAMP,
    )
    dirty = gate.evaluate(
        rule_score=80, llm_score=0.80, learned_rules_score=0.80,
        profile=_profile(Quadrant.A, scam_score=1.0), phase=PumpPhase.RAMP,
    )
    assert dirty.confidence < clean.confidence
    assert dirty.components["scam_penalty"] == pytest.approx(0.5)


# ----------------------- input clamping ----------------------- #


def test_nan_inputs_dont_crash_or_leak():
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=float("nan"),
        llm_score=float("nan"),
        learned_rules_score=float("nan"),
        profile=_profile(Quadrant.A,
                         historical_win_rate=float("nan")),
        phase=PumpPhase.RAMP,
    )
    assert 0.0 <= v.confidence <= 1.0


def test_overflow_inputs_clamped_to_unit_interval():
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=10_000.0, llm_score=5.0, learned_rules_score=5.0,
        profile=_profile(Quadrant.A, historical_win_rate=10.0),
        phase=PumpPhase.PARABOLIC,
    )
    assert v.confidence <= 1.0
    assert v.components["rule"] == 1.0
    assert v.components["llm"] == 1.0


# ----------------------- as_dict serialisation ----------------------- #


def test_verdict_as_dict_roundtrips():
    gate = ConfidenceGate()
    v = gate.evaluate(
        rule_score=70, llm_score=0.7, learned_rules_score=0.5,
        profile=_profile(Quadrant.A), phase=PumpPhase.RAMP,
    )
    d = v.as_dict()
    assert d["confidence"] == v.confidence
    assert d["threshold"] == v.threshold
    assert d["should_open"] == v.should_open
    assert "components" in d
