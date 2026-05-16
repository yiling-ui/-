"""tests/test_token_budget_mock.py — QUADRANT 六 coverage."""

from __future__ import annotations

import json

from altcoin_agent.llm.token_budget import (
    BudgetMode,
    TokenBudgetManager,
    TokenBudgetState,
)

# ----------------------- mode tiers ----------------------- #


def test_default_mode_is_free():
    mgr = TokenBudgetManager(monthly_budget=1000)
    assert mgr.mode() is BudgetMode.FREE
    ok, _ = mgr.can_call_llm(quadrant="D", signal_score=10.0)
    assert ok is True


def test_economy_blocks_c_and_d():
    mgr = TokenBudgetManager(monthly_budget=1000)
    mgr.record_usage(600)  # 60% used -> economy
    assert mgr.mode() is BudgetMode.ECONOMY

    ok, _ = mgr.can_call_llm(quadrant="A", signal_score=10.0)
    assert ok is True
    ok, _ = mgr.can_call_llm(quadrant="B", signal_score=10.0)
    assert ok is True
    ok, reason = mgr.can_call_llm(quadrant="C", signal_score=10.0)
    assert ok is False and "economy_block" in reason
    ok, reason = mgr.can_call_llm(quadrant="D", signal_score=10.0)
    assert ok is False


def test_emergency_only_a_with_high_score():
    mgr = TokenBudgetManager(monthly_budget=1000)
    mgr.record_usage(850)  # 85% -> emergency
    assert mgr.mode() is BudgetMode.EMERGENCY

    ok, _ = mgr.can_call_llm(quadrant="A", signal_score=80.0)
    assert ok is True
    ok, reason = mgr.can_call_llm(quadrant="A", signal_score=50.0)
    assert ok is False and "score=" in reason
    ok, _ = mgr.can_call_llm(quadrant="B", signal_score=99.0)
    assert ok is False


def test_freeze_rejects_everything():
    mgr = TokenBudgetManager(monthly_budget=100)
    mgr.record_usage(100)
    assert mgr.mode() is BudgetMode.FREEZE
    ok, reason = mgr.can_call_llm(quadrant="A", signal_score=99.0)
    assert ok is False and "freeze" in reason


# ----------------------- usage accounting ----------------------- #


def test_record_usage_accumulates():
    mgr = TokenBudgetManager(monthly_budget=10_000)
    mgr.record_usage(500)
    mgr.record_usage(750)
    assert mgr.state.used == 1250
    assert mgr.state.calls == 2


def test_used_pct_computation():
    mgr = TokenBudgetManager(monthly_budget=4_000_000)
    mgr.record_usage(1_000_000)
    assert mgr.used_pct() == 0.25


def test_estimate_tokens_handles_empty():
    assert TokenBudgetManager.estimate_tokens("") == 0
    assert TokenBudgetManager.estimate_tokens("hello world!") >= 1


# ----------------------- month rollover ----------------------- #


def test_month_rollover_archives_previous_bucket(tmp_path):
    # Use mutable [now] container to drive time forward.
    now = [_dt_to_ts(2026, 5, 15)]

    def now_fn():
        return now[0]

    mgr = TokenBudgetManager(
        monthly_budget=1000,
        state_path=str(tmp_path / "u.json"),
        now_fn=now_fn,
    )
    mgr.record_usage(400)
    assert mgr.state.used == 400
    assert mgr.state.month_key == "2026-05"

    # Advance into June.
    now[0] = _dt_to_ts(2026, 6, 2)
    mgr.record_usage(100)
    assert mgr.state.month_key == "2026-06"
    assert mgr.state.used == 100
    assert len(mgr.history) == 1
    assert mgr.history[0].month_key == "2026-05"
    assert mgr.history[0].used == 400


def test_history_capped_at_12_months():
    mgr = TokenBudgetManager(monthly_budget=1)
    for y in range(2020, 2025):
        for m in range(1, 13):
            mgr.history.append(TokenBudgetState(month_key=f"{y}-{m:02d}"))
    # Trigger trim by simulating a rollover via direct call.
    mgr._maybe_rollover()  # noqa: SLF001
    # Only the cap matters here; check that cap is enforced after a save+load.
    if len(mgr.history) > 12:
        # The trim happens on rollover; force rollover by changing month.
        mgr.now_fn = lambda: _dt_to_ts(2099, 1, 15)
        mgr._maybe_rollover()  # noqa: SLF001
    assert len(mgr.history) <= 12


# ----------------------- persistence ----------------------- #


def test_persist_roundtrip(tmp_path):
    path = str(tmp_path / "tokens.json")
    now = [_dt_to_ts(2026, 5, 15)]
    mgr = TokenBudgetManager(
        monthly_budget=10_000,
        state_path=path,
        now_fn=lambda: now[0],
    )
    mgr.record_usage(2_000)
    mgr.record_usage(500)

    # Reload in a new manager instance.
    mgr2 = TokenBudgetManager(
        monthly_budget=10_000,
        state_path=path,
        now_fn=lambda: now[0],
    )
    assert mgr2.state.month_key == "2026-05"
    assert mgr2.state.used == 2_500
    assert mgr2.state.calls == 2


def test_stale_persisted_file_archives_to_history(tmp_path):
    path = str(tmp_path / "tokens.json")
    # Manually write a "May 2026" payload, then construct manager in June.
    payload = {
        "version": 1,
        "active": {"month_key": "2026-05", "used": 999, "calls": 7,
                    "budget": 5_000_000},
        "history": [],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    mgr = TokenBudgetManager(
        monthly_budget=5_000_000,
        state_path=path,
        now_fn=lambda: _dt_to_ts(2026, 6, 1),
    )
    assert mgr.state.month_key == "2026-06"
    assert mgr.state.used == 0
    archived_keys = [b.month_key for b in mgr.history]
    assert "2026-05" in archived_keys


def test_load_skips_corrupted_file(tmp_path):
    path = str(tmp_path / "tokens.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json {{{")
    mgr = TokenBudgetManager(monthly_budget=1000, state_path=path)
    # Falls back to a fresh bucket without raising.
    assert mgr.state.used == 0


# ----------------------- helpers ----------------------- #


def _dt_to_ts(y: int, m: int, d: int) -> float:
    import calendar
    return float(calendar.timegm((y, m, d, 0, 0, 0, 0, 0, 0)))
