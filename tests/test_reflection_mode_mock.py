"""Unit tests for risk/reflection_mode.py (Phase A.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from altcoin_agent.risk.miss_penalty_engine import MissedOpportunity
from altcoin_agent.risk.reflection_mode import (
    ReflectionConfig,
    ReflectionModeController,
    ReflectionState,
    stub_llm_caller,
)
from altcoin_agent.risk.reject_reason_scorer import RejectReasonScore

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _miss(
    *, ts_ms: int, severity: float = 0.8,
    reason: str = "anti_chase", direction: str = "long",
    symbol: str = "PEPE",
) -> MissedOpportunity:
    return MissedOpportunity(
        trace_id=f"t-{ts_ms}-{symbol}",
        symbol=symbol,
        rejected_at_ts_ms=ts_ms,
        rejected_reason=f"{reason}:0.04",
        rejected_reason_bucket=reason,
        rejected_score=78.0,
        direction=direction,
        entry_price_if_taken=1.0,
        realized_max_favorable_pct=2.0,
        realized_max_adverse_pct=-0.05,
        is_missed_pump=True,
        miss_severity=severity,
        bars_observed=1440,
    )


def _make_controller(
    tmp_path: Path,
    *,
    clock_s: float = 1_000_000.0,
    llm_caller=None,
    notifier=None,
    cfg_overrides=None,
) -> ReflectionModeController:
    cfg = ReflectionConfig(
        state_path=str(tmp_path / "reflection_state.json"),
        reports_dir=str(tmp_path / "reports"),
        miss_threshold=3,
        trade_threshold=2,
        window_sec=7 * 24 * 3600,
        suspension_sec=24 * 3600,
        min_seconds_between_reports=24 * 3600,
        a_quadrant_bypass_score=95.0,
    )
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    return ReflectionModeController(
        config=cfg,
        llm_caller=llm_caller,
        notifier=notifier,
        clock=lambda: clock_s,
    )


# --------------------------------------------------------------------- #
# maybe_trigger: pure decision logic
# --------------------------------------------------------------------- #


def test_maybe_trigger_fires_on_three_misses_one_trade(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)

    misses = [
        _miss(ts_ms=now_ms - 1 * 86_400_000),
        _miss(ts_ms=now_ms - 2 * 86_400_000),
        _miss(ts_ms=now_ms - 3 * 86_400_000),
    ]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=1)
    assert decision.triggered is True
    assert decision.missed_pumps_in_window == 3
    assert decision.trades_in_window == 1


def test_maybe_trigger_holds_when_trades_above_threshold(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)

    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in range(1, 6)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=5)
    assert decision.triggered is False
    assert "ok_active_strategy" in decision.reason


def test_maybe_trigger_holds_when_misses_below_threshold(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)

    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in range(1, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    assert decision.triggered is False


def test_maybe_trigger_excludes_stale_misses_outside_window(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)

    misses = [
        _miss(ts_ms=now_ms - 30 * 86_400_000),  # 30d -> outside 7d window
        _miss(ts_ms=now_ms - 31 * 86_400_000),
        _miss(ts_ms=now_ms - 32 * 86_400_000),
    ]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    assert decision.triggered is False
    assert decision.missed_pumps_in_window == 0


def test_maybe_trigger_excludes_non_missed_pumps(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)

    correct_reject = _miss(ts_ms=now_ms - 86_400_000)
    correct_reject.is_missed_pump = False
    decision = ctrl.maybe_trigger(
        missed=[correct_reject], actual_trades_in_window=0,
    )
    assert decision.triggered is False
    assert decision.missed_pumps_in_window == 0


def test_maybe_trigger_respects_report_cooldown(tmp_path: Path) -> None:
    """If a report was generated within the cooldown window, the
    next maybe_trigger must NOT re-trigger -- the operator is still
    digesting the previous one."""
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)
    # Manually bake in a recent last_report_ts_ms on disk.
    ctrl._update_state(ReflectionState(
        last_report_ts_ms=now_ms - 2 * 3600 * 1000,  # 2 hours ago
        last_report_path="x", acknowledged=False,
        pending_report_id="prior",
    ))

    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in range(1, 6)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    assert decision.triggered is False
    assert decision.reason == "report_cooldown_active"


# --------------------------------------------------------------------- #
# is_suspended / can_bypass_suspension
# --------------------------------------------------------------------- #


def test_is_suspended_returns_false_by_default(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    assert ctrl.is_suspended() is False


def test_is_suspended_true_during_window(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)
    ctrl._update_state(ReflectionState(
        suspended_until_ts_ms=now_ms + 3600 * 1000,
        acknowledged=False, pending_report_id="x",
    ))
    assert ctrl.is_suspended() is True


def test_is_suspended_auto_expires(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)
    ctrl._update_state(ReflectionState(
        suspended_until_ts_ms=now_ms - 1,
        acknowledged=False, pending_report_id="x",
    ))
    assert ctrl.is_suspended() is False


def test_can_bypass_suspension_at_score_floor(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    assert ctrl.can_bypass_suspension(final_score=95.0) is True
    assert ctrl.can_bypass_suspension(final_score=94.9) is False


# --------------------------------------------------------------------- #
# generate_report — async path
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_report_writes_markdown_and_flips_state(
    tmp_path: Path,
) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    notifier_calls = []

    async def _notify(payload: dict) -> None:
        notifier_calls.append(payload)

    ctrl = _make_controller(
        tmp_path,
        clock_s=now_s,
        llm_caller=stub_llm_caller("LLM analysis text"),
        notifier=_notify,
    )

    misses = [
        _miss(ts_ms=now_ms - 86_400_000, symbol="PEPE", severity=0.95),
        _miss(ts_ms=now_ms - 2 * 86_400_000, symbol="WIF", severity=0.80),
        _miss(ts_ms=now_ms - 3 * 86_400_000, symbol="DOGE", severity=0.55),
    ]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=1)
    assert decision.triggered

    report = await ctrl.generate_report(
        decision=decision, missed=misses,
        scores={"anti_chase": RejectReasonScore(
            reason="anti_chase", correct_rejects=2, missed_pumps=3,
        )},
        suggested_overrides=[
            {
                "param_path": "price_tape.anti_chase_max_move_pct",
                "previous": 0.025, "new": 0.032, "reason": "anti_chase",
            },
        ],
    )

    # Markdown on disk?
    md_path = Path(report.markdown_path)
    assert md_path.exists()
    text = md_path.read_text()
    assert "策略反思报告" in text
    assert "PEPE" in text
    assert "anti_chase" in text
    assert "LLM analysis text" in text
    assert "0.025" in text  # current value from suggested overrides
    assert "0.032" in text  # proposed value

    # State flipped to suspended + pending ack.
    assert ctrl.is_suspended() is True
    assert ctrl.has_pending_review() is True

    # Telegram notifier called once.
    assert len(notifier_calls) == 1
    payload = notifier_calls[0]
    assert payload["kind"] == "reflection_report"
    assert payload["missed_pumps"] == 3


@pytest.mark.asyncio
async def test_generate_report_uses_fallback_when_no_llm(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s, llm_caller=None)

    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    report = await ctrl.generate_report(decision=decision, missed=misses)

    assert "未配置 LLM" in report.llm_analysis
    assert "anti_chase" in Path(report.markdown_path).read_text()


@pytest.mark.asyncio
async def test_generate_report_swallows_llm_failure(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)

    async def _bad_llm(_prompt: str) -> str:
        raise RuntimeError("upstream LLM is down")

    ctrl = _make_controller(tmp_path, clock_s=now_s, llm_caller=_bad_llm)
    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    report = await ctrl.generate_report(decision=decision, missed=misses)
    # Falls back without raising.
    assert "未配置 LLM" in report.llm_analysis


@pytest.mark.asyncio
async def test_generate_report_swallows_notifier_failure(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)

    async def _bad_notifier(_payload: dict) -> None:
        raise RuntimeError("telegram down")

    ctrl = _make_controller(tmp_path, clock_s=now_s, notifier=_bad_notifier)
    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    # Should not raise even though notifier did.
    await ctrl.generate_report(decision=decision, missed=misses)
    assert ctrl.is_suspended() is True


@pytest.mark.asyncio
async def test_generate_report_rejects_non_triggered_decision(
    tmp_path: Path,
) -> None:
    ctrl = _make_controller(tmp_path)
    decision = ctrl.maybe_trigger(missed=[], actual_trades_in_window=0)
    with pytest.raises(ValueError):
        await ctrl.generate_report(decision=decision, missed=[])


# --------------------------------------------------------------------- #
# acknowledge
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_acknowledge_clears_suspension_and_pending(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)
    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    report = await ctrl.generate_report(decision=decision, missed=misses)
    assert ctrl.is_suspended() is True

    assert ctrl.acknowledge(report_id=report.report_id) is True
    assert ctrl.is_suspended() is False
    assert ctrl.has_pending_review() is False


@pytest.mark.asyncio
async def test_acknowledge_rejects_stale_report_id(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl = _make_controller(tmp_path, clock_s=now_s)
    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl.maybe_trigger(missed=misses, actual_trades_in_window=0)
    await ctrl.generate_report(decision=decision, missed=misses)

    assert ctrl.acknowledge(report_id="some-old-id") is False
    assert ctrl.is_suspended() is True


# --------------------------------------------------------------------- #
# State persistence across instances
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_state_persists_across_controller_instances(tmp_path: Path) -> None:
    now_s = 10_000_000.0
    now_ms = int(now_s * 1000)
    ctrl_a = _make_controller(tmp_path, clock_s=now_s)
    misses = [_miss(ts_ms=now_ms - i * 86_400_000) for i in (1, 2, 3)]
    decision = ctrl_a.maybe_trigger(missed=misses, actual_trades_in_window=0)
    await ctrl_a.generate_report(decision=decision, missed=misses)

    # Fresh controller in the same workspace -> sees the suspension.
    ctrl_b = _make_controller(tmp_path, clock_s=now_s + 60)
    assert ctrl_b.is_suspended() is True
    assert ctrl_b.has_pending_review() is True
