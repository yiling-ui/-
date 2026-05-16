"""Unit tests for risk/reject_reason_scorer.py (Phase A.2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from altcoin_agent.risk.miss_penalty_engine import MissedOpportunity
from altcoin_agent.risk.reject_reason_scorer import (
    RejectReasonScore,
    RejectReasonScorer,
    RejectReasonScorerConfig,
)

# --------------------------------------------------------------------- #
# RejectReasonScore — math
# --------------------------------------------------------------------- #


def test_confidence_score_uses_one_to_three_weighting() -> None:
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=10, missed_pumps=5,
    )
    # +1 * 10 - 3 * 5 == -5
    assert s.confidence_score == pytest.approx(-5.0)


def test_win_rate_is_laplace_smoothed() -> None:
    # 0/0 -> 1/2 == 0.5 (neutral)
    fresh = RejectReasonScore(reason="x")
    assert fresh.win_rate == pytest.approx(0.5)
    # 10 correct + 0 misses -> (10+1)/(10+2) ~= 0.9166
    perfect = RejectReasonScore(
        reason="x", correct_rejects=10, missed_pumps=0,
    )
    assert perfect.win_rate == pytest.approx(11 / 12)


def test_total_audited_excludes_insufficient_data() -> None:
    s = RejectReasonScore(
        reason="x", correct_rejects=4, missed_pumps=2,
        insufficient_data=99,
    )
    assert s.total_audited == 6


def test_to_dict_includes_derived_fields() -> None:
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=5, missed_pumps=2,
    )
    d = s.to_dict()
    assert "confidence_score" in d
    assert "win_rate" in d
    assert "total_audited" in d
    assert d["confidence_score"] == pytest.approx(-1.0)


# --------------------------------------------------------------------- #
# RejectReasonScorer — should_loosen rule
# --------------------------------------------------------------------- #


def _scorer(tmp_path: Path, **cfg_kwargs) -> RejectReasonScorer:
    cfg = RejectReasonScorerConfig(
        state_path=str(tmp_path / "scores.json"),
        **cfg_kwargs,
    )
    return RejectReasonScorer(
        decisions_log_path=tmp_path / "decisions.jsonl",
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=cfg,
    )


def test_should_loosen_blocks_until_min_samples(tmp_path: Path) -> None:
    sc = _scorer(tmp_path, samples_required=50)
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=5, missed_pumps=10,
    )
    # confidence = 5 - 30 = -25 (very bad), but only 15 samples.
    assert s.confidence_score == pytest.approx(-25.0)
    assert sc.should_loosen(s) is False


def test_should_loosen_fires_when_all_conditions_met(tmp_path: Path) -> None:
    sc = _scorer(
        tmp_path,
        samples_required=50,
        loosen_when_confidence_below=-10.0,
        loosen_when_missed_pumps_at_least=5,
    )
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=20, missed_pumps=15,
    )
    # 20 - 45 = -25, missed_pumps >= 5, samples 35... wait, 35 < 50.
    # Bump samples.
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=40, missed_pumps=15,
    )
    # confidence = 40 - 45 = -5 (above -10) -> blocked.
    assert sc.should_loosen(s) is False
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=30, missed_pumps=20,
    )
    # confidence = 30 - 60 = -30 <= -10, samples = 50, missed >= 5.
    assert sc.should_loosen(s) is True


def test_should_loosen_skips_untuneable_reasons(tmp_path: Path) -> None:
    sc = _scorer(tmp_path, samples_required=10)
    s = RejectReasonScore(
        reason="reconciliation_pending",
        correct_rejects=5, missed_pumps=20,
    )
    # Confidence is awful but the reason is non-tunable.
    assert sc.should_loosen(s) is False


def test_should_loosen_requires_minimum_missed_pumps(tmp_path: Path) -> None:
    sc = _scorer(
        tmp_path,
        samples_required=10,
        loosen_when_confidence_below=-1.0,
        loosen_when_missed_pumps_at_least=5,
    )
    # Lots of correct rejects, only 2 misses -> confidence is HIGH
    # so this reason wouldn't fire anyway, but the test pins the rule.
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=2, missed_pumps=2,
    )
    # confidence = 2 - 6 = -4 (below -1), samples 4... need samples 10.
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=8, missed_pumps=2,
    )
    # samples=10, confidence = 8-6 = +2 (NOT below -1) -> blocked.
    assert sc.should_loosen(s) is False
    s = RejectReasonScore(
        reason="anti_chase", correct_rejects=2, missed_pumps=8,
    )
    # samples=10, confidence = 2-24 = -22, missed=8 >= 5 -> fires.
    assert sc.should_loosen(s) is True


# --------------------------------------------------------------------- #
# Recompute — full integration with synthetic logs
# --------------------------------------------------------------------- #


def _write_decision_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_missed(path: Path, rows: list[MissedOpportunity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def test_recompute_classifies_correct_rejects_and_missed_pumps(
    tmp_path: Path,
) -> None:
    now = time.time()
    decisions = tmp_path / "decisions.jsonl"
    missed = tmp_path / "missed.jsonl"

    # Three rejected entries:
    #   t1 -> labelled missed pump
    #   t2 -> NOT in missed file, old enough -> correct reject
    #   t3 -> NOT in missed file, RECENT -> insufficient_data
    _write_decision_lines(decisions, [
        {
            "ts": now - 48 * 3600, "trace_id": "t1", "approved": False,
            "symbol": "PEPE", "direction": "long", "current_price": 1.0,
            "reason": "anti_chase:0.04",
        },
        {
            "ts": now - 48 * 3600, "trace_id": "t2", "approved": False,
            "symbol": "WIF", "direction": "long", "current_price": 1.0,
            "reason": "anti_chase:0.05",
        },
        {
            "ts": now - 60 * 60, "trace_id": "t3", "approved": False,
            "symbol": "DOGE", "direction": "long", "current_price": 1.0,
            "reason": "anti_chase:0.05",
        },
        # An approved trade -- should be ignored entirely.
        {
            "ts": now - 48 * 3600, "trace_id": "t-approved", "approved": True,
            "symbol": "BTC", "direction": "long", "current_price": 1.0,
            "reason": "ok",
        },
    ])
    _write_missed(missed, [
        MissedOpportunity(
            trace_id="t1", symbol="PEPE",
            rejected_at_ts_ms=int((now - 48 * 3600) * 1000),
            rejected_reason="anti_chase:0.04",
            rejected_reason_bucket="anti_chase",
            rejected_score=78.0, direction="long",
            entry_price_if_taken=1.0,
            realized_max_favorable_pct=2.0,
            realized_max_adverse_pct=-0.05,
            is_missed_pump=True, miss_severity=0.85,
            bars_observed=1440,
        ),
    ])

    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=missed,
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "scores.json"),
        ),
    )
    table = sc.recompute()
    anti = table["anti_chase"]
    assert anti.correct_rejects == 1   # t2
    assert anti.missed_pumps == 1      # t1
    assert anti.insufficient_data == 1 # t3
    assert anti.confidence_score == pytest.approx(-2.0)
    assert anti.weighted_miss_severity == pytest.approx(0.85)


def test_recompute_buckets_reasons_with_param_suffixes(tmp_path: Path) -> None:
    """Different ``slippage_too_high:0.034>0.021@lev=10`` variants must
    bucket together as ``slippage_too_high``."""
    now = time.time()
    decisions = tmp_path / "decisions.jsonl"
    _write_decision_lines(decisions, [
        {
            "ts": now - 48 * 3600, "trace_id": f"t{i}", "approved": False,
            "symbol": "X", "direction": "long", "current_price": 1.0,
            "reason": f"slippage_too_high:{0.03 + i * 0.001}>0.02@lev=10",
        }
        for i in range(5)
    ])
    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "scores.json"),
        ),
    )
    table = sc.recompute()
    assert "slippage_too_high" in table
    # 5 distinct trace_ids, all old enough, none in missed -> 5 correct
    assert table["slippage_too_high"].correct_rejects == 5


def test_recompute_persists_to_state_file(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    state = tmp_path / "scores.json"
    _write_decision_lines(decisions, [
        {
            "ts": time.time() - 48 * 3600, "trace_id": "t1",
            "approved": False, "symbol": "X", "direction": "long",
            "current_price": 1.0, "reason": "anti_chase:0.04",
        },
    ])
    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(state_path=str(state)),
    )
    sc.recompute()
    assert state.exists()
    payload = json.loads(state.read_text())
    assert "scores" in payload
    reasons = [r["reason"] for r in payload["scores"]]
    assert "anti_chase" in reasons


def test_recompute_filters_out_rows_outside_lookback(tmp_path: Path) -> None:
    now = time.time()
    decisions = tmp_path / "decisions.jsonl"
    _write_decision_lines(decisions, [
        {  # 60 days ago -> outside default 30d lookback
            "ts": now - 60 * 24 * 3600, "trace_id": "old",
            "approved": False, "symbol": "X", "direction": "long",
            "current_price": 1.0, "reason": "anti_chase:0.04",
        },
        {  # 2 days ago -> inside lookback
            "ts": now - 2 * 24 * 3600, "trace_id": "fresh",
            "approved": False, "symbol": "X", "direction": "long",
            "current_price": 1.0, "reason": "anti_chase:0.04",
        },
    ])
    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "scores.json"),
        ),
    )
    table = sc.recompute()
    assert table["anti_chase"].correct_rejects == 1


def test_load_round_trips_through_disk(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    state = tmp_path / "scores.json"
    _write_decision_lines(decisions, [
        {
            "ts": time.time() - 48 * 3600, "trace_id": f"t{i}",
            "approved": False, "symbol": "X", "direction": "long",
            "current_price": 1.0, "reason": "anti_chase:0.04",
        }
        for i in range(3)
    ])
    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(state_path=str(state)),
    )
    sc.recompute()
    loaded = sc.load()
    assert "anti_chase" in loaded
    assert loaded["anti_chase"].correct_rejects == 3


def test_load_returns_empty_when_no_state(tmp_path: Path) -> None:
    sc = RejectReasonScorer(
        decisions_log_path=tmp_path / "decisions.jsonl",
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "missing.json"),
        ),
    )
    assert sc.load() == {}


def test_recompute_recent_rejects_count_as_insufficient_data(
    tmp_path: Path,
) -> None:
    """Reject inside the 24h audit window without an audit row yet."""
    now = time.time()
    decisions = tmp_path / "decisions.jsonl"
    _write_decision_lines(decisions, [
        {
            "ts": now - 60 * 60,  # 1h ago, audit not closed
            "trace_id": "t-fresh", "approved": False, "symbol": "X",
            "direction": "long", "current_price": 1.0,
            "reason": "anti_chase:0.04",
        },
    ])
    sc = RejectReasonScorer(
        decisions_log_path=decisions,
        missed_opportunities_path=tmp_path / "missed.jsonl",
        config=RejectReasonScorerConfig(
            state_path=str(tmp_path / "scores.json"),
        ),
    )
    table = sc.recompute()
    assert table["anti_chase"].correct_rejects == 0
    assert table["anti_chase"].insufficient_data == 1
