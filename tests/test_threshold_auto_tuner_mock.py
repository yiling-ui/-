"""Unit tests for risk/threshold_auto_tuner.py (Phase A.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altcoin_agent.risk.reject_reason_scorer import (
    RejectReasonScore,
    RejectReasonScorer,
    RejectReasonScorerConfig,
)
from altcoin_agent.risk.threshold_auto_tuner import (
    DEFAULT_TUNING_RULES,
    ThresholdAutoTuner,
    ThresholdAutoTunerConfig,
    TuneableKnob,
)

# --------------------------------------------------------------------- #
# TuneableKnob — pure math
# --------------------------------------------------------------------- #


def test_tuneable_knob_loosen_steps_toward_hard_max() -> None:
    k = TuneableKnob(
        reason="x", param_path="x", default=0.025,
        hard_min=0.025, hard_max=0.06, direction=+1, step_pct=0.20,
    )
    # 25% step toward 6.0%: 0.025 + 0.20 * (0.06 - 0.025) = 0.032
    assert k.loosen(0.025) == pytest.approx(0.032)


def test_tuneable_knob_loosen_steps_toward_hard_min() -> None:
    k = TuneableKnob(
        reason="x", param_path="x", default=200_000,
        hard_min=100_000, hard_max=200_000, direction=-1, step_pct=0.20,
    )
    # Step toward hard_min: 200_000 + 0.20 * (100_000 - 200_000) = 180_000
    assert k.loosen(200_000) == pytest.approx(180_000)


def test_tuneable_knob_clamp_respects_both_bounds() -> None:
    k = TuneableKnob(
        reason="x", param_path="x", default=1.0,
        hard_min=0.5, hard_max=2.0, direction=+1,
    )
    assert k.clamp(0.1) == 0.5
    assert k.clamp(3.0) == 2.0
    assert k.clamp(1.0) == 1.0


def test_tuneable_knob_loosen_caps_at_hard_max() -> None:
    """Repeated loosen calls converge to the hard limit, not past it."""
    k = TuneableKnob(
        reason="x", param_path="x", default=0.025,
        hard_min=0.025, hard_max=0.06, direction=+1, step_pct=0.20,
    )
    v = 0.025
    for _ in range(50):
        v = k.clamp(k.loosen(v))
    assert v == pytest.approx(0.06, abs=1e-3)
    assert v <= 0.06


# --------------------------------------------------------------------- #
# Tuner — behaviour
# --------------------------------------------------------------------- #


def _scorer(tmp_path: Path) -> RejectReasonScorer:
    return RejectReasonScorer(
        decisions_log_path=tmp_path / "decisions.jsonl",
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "scores.json"),
            samples_required=10,
            loosen_when_confidence_below=-5.0,
            loosen_when_missed_pumps_at_least=3,
        ),
    )


def _tuner(
    tmp_path: Path,
    *,
    scorer: RejectReasonScorer | None = None,
    clock: float = 1_000_000.0,
    rules: dict | None = None,
    cooldown_sec: int = 6 * 24 * 3600,
) -> ThresholdAutoTuner:
    cfg = ThresholdAutoTunerConfig(
        overrides_state_path=str(tmp_path / "overrides.json"),
        min_seconds_between_loosens=cooldown_sec,
        rules=rules if rules is not None else dict(DEFAULT_TUNING_RULES),
    )
    sc = scorer or _scorer(tmp_path)
    return ThresholdAutoTuner(scorer=sc, config=cfg, clock=lambda: clock)


def test_tune_holds_when_no_score_data(tmp_path: Path) -> None:
    tuner = _tuner(tmp_path)
    out = tuner.tune(scores={})
    # Three default rules; all should report "hold" with default values.
    assert set(out.keys()) == set(DEFAULT_TUNING_RULES.keys())
    for ov in out.values():
        assert ov.direction_label == "hold"
        assert ov.previous == ov.new


def test_tune_holds_when_score_above_loosen_threshold(tmp_path: Path) -> None:
    tuner = _tuner(tmp_path)
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=20, missed_pumps=2,  # confidence = 14
        ),
    }
    out = tuner.tune(scores=scores)
    assert out["chase_too_late"].direction_label == "hold"


def test_tune_loosens_anti_chase_when_score_qualifies(tmp_path: Path) -> None:
    tuner = _tuner(tmp_path)
    # samples=20, confidence = 5 - 45 = -40 (well below -5),
    # missed_pumps=15 (>= 3) -> qualified
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    out = tuner.tune(scores=scores)
    ov = out["chase_too_late"]
    assert ov.direction_label == "loosen"
    assert ov.previous == pytest.approx(0.025)
    assert ov.new > ov.previous
    assert ov.new <= 0.06  # never exceeds hard_max


def test_tune_persists_overrides_to_disk(tmp_path: Path) -> None:
    tuner = _tuner(tmp_path)
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    tuner.tune(scores=scores)

    state = tmp_path / "overrides.json"
    assert state.exists()
    data = json.loads(state.read_text())
    by_reason = {ov["reason"]: ov for ov in data["overrides"]}
    assert by_reason["chase_too_late"]["direction_label"] == "loosen"
    assert by_reason["chase_too_late"]["new"] > 0.025


def test_tune_respects_min_cooldown_between_runs(tmp_path: Path) -> None:
    """A second invocation within the cooldown window must NOT
    advance the override -- prevents double-stepping when the cron
    fires twice in a week."""
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    sc = _scorer(tmp_path)

    tuner_a = _tuner(tmp_path, scorer=sc, clock=1_000_000.0)
    first = tuner_a.tune(scores=scores)
    first_value = first["chase_too_late"].new

    # Second run only 1 hour later.
    tuner_b = _tuner(tmp_path, scorer=sc, clock=1_000_000.0 + 3600)
    second = tuner_b.tune(scores=scores)
    assert second["chase_too_late"].new == pytest.approx(first_value)
    assert second["chase_too_late"].direction_label == "hold"


def test_tune_advances_after_cooldown(tmp_path: Path) -> None:
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    sc = _scorer(tmp_path)

    tuner_a = _tuner(tmp_path, scorer=sc, clock=1_000_000.0)
    first = tuner_a.tune(scores=scores)

    # Move clock 7 days forward.
    tuner_b = _tuner(tmp_path, scorer=sc, clock=1_000_000.0 + 7 * 24 * 3600)
    second = tuner_b.tune(scores=scores)
    assert second["chase_too_late"].new > first["chase_too_late"].new


def test_tune_clamps_to_hard_max_after_many_runs(tmp_path: Path) -> None:
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    sc = _scorer(tmp_path)

    clock = 1_000_000.0
    final_value = 0.025
    for _i in range(20):
        clock += 8 * 24 * 3600  # advance past cooldown each iteration
        tuner = _tuner(tmp_path, scorer=sc, clock=clock)
        out = tuner.tune(scores=scores)
        final_value = out["chase_too_late"].new
    assert final_value == pytest.approx(0.06, abs=1e-3)
    assert final_value <= 0.06


def test_tune_handles_min_liquidity_with_negative_direction(tmp_path: Path) -> None:
    scores = {
        "insufficient_liquidity": RejectReasonScore(
            reason="insufficient_liquidity",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    tuner = _tuner(tmp_path)
    out = tuner.tune(scores=scores)
    ov = out["insufficient_liquidity"]
    assert ov.direction_label == "loosen"
    assert ov.new < 200_000
    assert ov.new >= 100_000  # respects hard_min


def test_emit_param_overrides_returns_flat_dict(tmp_path: Path) -> None:
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    tuner = _tuner(tmp_path)
    overrides = tuner.tune(scores=scores)
    flat = tuner.emit_param_overrides(overrides)
    assert "price_tape.anti_chase_max_move_pct" in flat
    assert flat["price_tape.anti_chase_max_move_pct"] > 0.025


def test_emit_param_overrides_uses_default_when_no_data(tmp_path: Path) -> None:
    tuner = _tuner(tmp_path)
    out = tuner.tune(scores={})
    flat = tuner.emit_param_overrides(out)
    assert flat["price_tape.anti_chase_max_move_pct"] == pytest.approx(0.025)
    assert flat["risk_gate.min_liquidity_usdt"] == pytest.approx(200_000)
    assert flat["risk_gate.consecutive_loss_cooldown_sec"] == 4 * 3600


def test_load_overrides_round_trips(tmp_path: Path) -> None:
    scores = {
        "chase_too_late": RejectReasonScore(
            reason="chase_too_late",
            correct_rejects=5, missed_pumps=15,
        ),
    }
    tuner = _tuner(tmp_path)
    tuner.tune(scores=scores)

    fresh_tuner = _tuner(tmp_path, clock=1_000_000.0)
    loaded = fresh_tuner.load_overrides()
    assert "chase_too_late" in loaded
    assert loaded["chase_too_late"].new > 0.025


def test_tune_skips_loosening_for_untuneable_reasons(tmp_path: Path) -> None:
    """Even if we cooked a fake 'reconciliation_pending' score with
    catastrophic confidence, the scorer's untuneable-reason filter
    must keep the tuner from emitting an override for it."""
    sc = RejectReasonScorer(
        decisions_log_path=tmp_path / "d.jsonl",
        missed_opportunities_path=tmp_path / "m.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "s.json"),
            samples_required=10,
            loosen_when_confidence_below=-5.0,
            loosen_when_missed_pumps_at_least=3,
        ),
    )
    # Custom rules table that maps an untuneable reason to a knob,
    # to prove the tuner respects ``should_loosen`` even when a custom
    # rule would otherwise apply.
    custom_rules = {
        "reconciliation_pending": TuneableKnob(
            reason="reconciliation_pending",
            param_path="risk_gate.reconciliation_timeout_sec",
            default=60.0, hard_min=10.0, hard_max=60.0, direction=-1,
        ),
    }
    tuner = _tuner(tmp_path, scorer=sc, rules=custom_rules)
    out = tuner.tune(scores={
        "reconciliation_pending": RejectReasonScore(
            reason="reconciliation_pending",
            correct_rejects=5, missed_pumps=20,
        ),
    })
    assert out["reconciliation_pending"].direction_label == "hold"
    assert out["reconciliation_pending"].new == pytest.approx(60.0)
