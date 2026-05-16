"""tests/test_rule_miner_mock.py — Phase 4 rule miner unit tests."""

from __future__ import annotations

import pytest

from altcoin_agent.training.rule_miner import (
    MinerConfig,
    RuleAccumulator,
    TradeObservation,
    bucketize_pct,
    bucketize_score,
    mine_rules,
)
from altcoin_agent.training.rules_promoter import LearnedRule

# ----------------------- bucketizers ----------------------- #


@pytest.mark.parametrize(
    "score,expected",
    [
        (95.0, "score>=90"),
        (85.0, "score>=80"),
        (75.0, "score>=70"),
        (65.0, "score>=60"),
        (55.0, "score<60"),
        (0.0, "score<60"),
    ],
)
def test_bucketize_score_default_edges(score, expected):
    assert bucketize_score(score) == expected


def test_bucketize_score_custom_edges():
    assert bucketize_score(50.0, edges=(40.0, 60.0)) == "score>=40"
    assert bucketize_score(80.0, edges=(40.0, 60.0)) == "score>=60"
    assert bucketize_score(30.0, edges=(40.0, 60.0)) == "score<40"


def test_bucketize_score_nan_safe():
    assert bucketize_score(float("nan")) == "score=na"


def test_bucketize_pct():
    edges = (0.0, 0.03, 0.06, 0.10)
    assert bucketize_pct(0.05, edges=edges) == "pct>=0.03"
    assert bucketize_pct(0.10, edges=edges) == "pct>=0.1"
    assert bucketize_pct(-0.01, edges=edges) == "pct<0"
    assert bucketize_pct(float("nan"), edges=edges) == "pct=na"


# ----------------------- helpers ----------------------- #


def _obs(
    *,
    ts_ms: int,
    pnl_pct: float,
    quadrant: str = "A",
    phase: str = "ramp",
    score_bucket: str = "score>=80",
    rejected_reason: str = "none",
    symbol: str = "X",
) -> TradeObservation:
    return TradeObservation(
        ts_ms=ts_ms,
        symbol=symbol,
        pnl_pct=pnl_pct,
        features={
            "quadrant": quadrant,
            "phase": phase,
            "score_bucket": score_bucket,
            "rejected_reason": rejected_reason,
        },
    )


# ----------------------- mine_rules ----------------------- #


def test_mine_empty_returns_empty():
    assert mine_rules([]) == []


def test_mine_skips_below_min_samples():
    cfg = MinerConfig(min_samples_per_bucket=5)
    obs = [_obs(ts_ms=i, pnl_pct=0.1) for i in range(3)]
    assert mine_rules(obs, cfg) == []


def test_mine_emits_one_rule_per_bucket():
    cfg = MinerConfig(min_samples_per_bucket=3, feature_combos=(("quadrant",),))
    obs = (
        [_obs(ts_ms=i, pnl_pct=0.1, quadrant="A") for i in range(5)]
        + [_obs(ts_ms=10 + i, pnl_pct=-0.05, quadrant="B") for i in range(5)]
    )
    rules = mine_rules(obs, cfg)
    rids = {r.rule_id for r in rules}
    assert "quadrant=A" in rids
    assert "quadrant=B" in rids


def test_mine_win_rate_and_pnl_correct():
    cfg = MinerConfig(min_samples_per_bucket=3, feature_combos=(("quadrant",),))
    obs = [
        _obs(ts_ms=0, pnl_pct=0.10),
        _obs(ts_ms=1, pnl_pct=0.20),
        _obs(ts_ms=2, pnl_pct=-0.05),
        _obs(ts_ms=3, pnl_pct=0.15),
    ]
    rules = mine_rules(obs, cfg)
    assert len(rules) == 1
    r = rules[0]
    assert r.samples == 4
    assert r.wins == 3
    assert r.losses == 1
    assert r.win_rate == pytest.approx(0.75)
    assert r.avg_pnl_pct == pytest.approx(0.10)
    assert r.sharpe > 0


def test_mine_combo_features_combine_correctly():
    cfg = MinerConfig(
        min_samples_per_bucket=2,
        feature_combos=(("quadrant", "phase"),),
    )
    obs = [
        _obs(ts_ms=0, pnl_pct=0.1, quadrant="A", phase="ramp"),
        _obs(ts_ms=1, pnl_pct=0.2, quadrant="A", phase="ramp"),
        _obs(ts_ms=2, pnl_pct=0.05, quadrant="A", phase="parabolic"),
        _obs(ts_ms=3, pnl_pct=-0.10, quadrant="A", phase="parabolic"),
    ]
    rules = mine_rules(obs, cfg)
    rids = {r.rule_id for r in rules}
    assert "quadrant+phase=A/ramp" in rids
    assert "quadrant+phase=A/parabolic" in rids


def test_mine_skips_observations_missing_combo_key():
    cfg = MinerConfig(
        min_samples_per_bucket=2, feature_combos=(("missing_key",),),
    )
    obs = [_obs(ts_ms=i, pnl_pct=0.1) for i in range(5)]
    rules = mine_rules(obs, cfg)
    assert rules == []


def test_mine_max_rules_per_cycle_truncates():
    # Build many distinct quadrants to overflow the cap.
    cfg = MinerConfig(
        min_samples_per_bucket=1,
        feature_combos=(("quadrant",),),
        max_rules_per_cycle=3,
    )
    obs = [
        _obs(ts_ms=i, pnl_pct=0.1, quadrant=f"Q{i}") for i in range(10)
    ]
    rules = mine_rules(obs, cfg)
    assert len(rules) == 3


def test_mine_carries_extras_for_promoter():
    cfg = MinerConfig(
        min_samples_per_bucket=2, feature_combos=(("quadrant", "phase"),),
    )
    obs = [
        _obs(ts_ms=0, pnl_pct=0.1, quadrant="A", phase="ramp"),
        _obs(ts_ms=1, pnl_pct=0.2, quadrant="A", phase="ramp"),
    ]
    rules = mine_rules(obs, cfg)
    assert rules[0].extras["combo"] == "quadrant+phase"
    assert rules[0].extras["value"] == "A/ramp"
    assert rules[0].extras["feature_keys"] == ["quadrant", "phase"]


def test_mine_zero_volatility_bucket_yields_zero_sharpe():
    cfg = MinerConfig(min_samples_per_bucket=2, feature_combos=(("quadrant",),))
    obs = [_obs(ts_ms=i, pnl_pct=0.05) for i in range(5)]
    rules = mine_rules(obs, cfg)
    assert rules[0].sharpe == 0.0


def test_mine_first_last_observed_ts_track_extremes():
    cfg = MinerConfig(min_samples_per_bucket=2, feature_combos=(("quadrant",),))
    obs = [
        _obs(ts_ms=100, pnl_pct=0.1),
        _obs(ts_ms=300, pnl_pct=0.1),
        _obs(ts_ms=200, pnl_pct=-0.05),
    ]
    rules = mine_rules(obs, cfg)
    assert rules[0].first_observed_ts == 100
    assert rules[0].last_observed_ts == 300


# ----------------------- RuleAccumulator ----------------------- #


def _rule(
    rid: str = "quadrant=A",
    samples: int = 10,
    wins: int = 8,
    win_rate: float = 0.80,
    avg_pnl: float = 0.05,
    first_ts: int = 100,
    last_ts: int = 200,
) -> LearnedRule:
    losses = samples - wins
    return LearnedRule(
        rule_id=rid,
        samples=samples,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl,
        sharpe=0.5,
        first_observed_ts=first_ts,
        last_observed_ts=last_ts,
        extras={"combo": "quadrant", "value": "A", "feature_keys": ["quadrant"]},
    )


def test_accumulator_pools_disjoint_windows():
    acc = RuleAccumulator()
    acc.add_window([_rule(samples=10, wins=8, win_rate=0.8, avg_pnl=0.05,
                           first_ts=100, last_ts=200)])
    acc.add_window([_rule(samples=20, wins=14, win_rate=0.70, avg_pnl=0.03,
                           first_ts=300, last_ts=400)])
    out = acc.flush()
    assert len(out) == 1
    pooled = out[0]
    assert pooled.samples == 30
    assert pooled.wins == 22
    assert pooled.losses == 8
    assert pooled.win_rate == pytest.approx(22 / 30)
    # Sample-weighted mean: (0.05*10 + 0.03*20) / 30
    assert pooled.avg_pnl_pct == pytest.approx((0.05 * 10 + 0.03 * 20) / 30)
    assert pooled.first_observed_ts == 100
    assert pooled.last_observed_ts == 400


def test_accumulator_counts_qualifying_windows():
    """validation_months_passed = number of validate windows where
    the rule's win_rate >= qualifying_win_rate AND samples >= floor."""
    acc = RuleAccumulator(qualifying_win_rate=0.80, qualifying_min_samples=5)
    # Window 1: passes
    acc.add_window([_rule(samples=10, wins=8, win_rate=0.80)])
    # Window 2: passes
    acc.add_window([_rule(samples=10, wins=9, win_rate=0.90)])
    # Window 3: too few samples (under floor) → doesn't count
    acc.add_window([_rule(samples=3, wins=3, win_rate=1.0)])
    # Window 4: low win_rate → doesn't count
    acc.add_window([_rule(samples=10, wins=5, win_rate=0.50)])
    out = acc.flush()
    assert len(out) == 1
    assert out[0].validation_months_passed == 2


def test_accumulator_handles_unique_rules_separately():
    acc = RuleAccumulator()
    acc.add_window([
        _rule(rid="quadrant=A", samples=10, wins=8, win_rate=0.8),
        _rule(rid="quadrant=B", samples=10, wins=4, win_rate=0.4),
    ])
    out = sorted(acc.flush(), key=lambda r: r.rule_id)
    assert [r.rule_id for r in out] == ["quadrant=A", "quadrant=B"]
    assert out[0].win_rate == pytest.approx(0.8)
    assert out[1].win_rate == pytest.approx(0.4)
