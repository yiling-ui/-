"""tests/test_walkforward_trainer_mock.py — Phase 4 walk-forward driver."""

from __future__ import annotations

import json
import os

from altcoin_agent.backtest.walk_forward import DAY_MS
from altcoin_agent.training.rule_miner import MinerConfig, TradeObservation
from altcoin_agent.training.rules_promoter import PromotionConfig
from altcoin_agent.training.walkforward_trainer import (
    WalkforwardTrainer,
    WalkforwardTrainerConfig,
    list_observation_provider,
)

# ----------------------- helpers ----------------------- #


def _obs(ts_ms: int, *, pnl: float, q: str = "A", phase: str = "ramp",
          score: str = "score>=80") -> TradeObservation:
    return TradeObservation(
        ts_ms=ts_ms,
        symbol="X",
        pnl_pct=pnl,
        features={
            "quadrant": q, "phase": phase, "score_bucket": score,
            "rejected_reason": "none",
        },
    )


# ----------------------- empty + edge ----------------------- #


def test_run_with_empty_provider_returns_zero_splits():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider([])
    report = trainer.run(start_ms=0, end_ms=4 * DAY_MS, provider=provider)
    assert report.splits == 3  # 3 (train, validate) splits fit
    assert report.rules_after_pooling == 0
    assert report.rules_promoted_now == 0


def test_zero_length_window_emits_no_splits():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider([])
    report = trainer.run(start_ms=0, end_ms=DAY_MS, provider=provider)
    assert report.splits == 0


# ----------------------- happy path ----------------------- #


def test_run_mines_rules_from_train_regrades_on_validate():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
        miner=MinerConfig(min_samples_per_bucket=3,
                          feature_combos=(("quadrant",),)),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    # Train window 0..1d: 5 winning A-quadrant trades.
    train = [_obs(ts_ms=i * 1000, pnl=0.05) for i in range(5)]
    # Validate window 1..2d: 5 winning A-quadrant trades.
    validate = [_obs(ts_ms=DAY_MS + i * 1000, pnl=0.05) for i in range(5)]
    provider = list_observation_provider(train + validate)
    report = trainer.run(
        start_ms=0, end_ms=2 * DAY_MS, provider=provider, now_ts=1_700_000_000,
    )
    assert report.splits == 1
    window = report.windows[0]
    assert window.train_observations == 5
    assert window.validate_observations == 5
    assert window.rules_mined >= 1
    assert window.win_stats.win_rate == 1.0
    # The rule must survive into the candidate pool (production gate
    # requires 30 samples + 3 months which we won't hit here).
    assert report.rules_after_pooling >= 1


def test_run_promotes_rule_after_three_qualifying_validate_windows():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
        miner=MinerConfig(min_samples_per_bucket=10,
                          feature_combos=(("quadrant",),)),
        promotion=PromotionConfig(
            min_samples=30,
            min_win_rate=0.80,
            min_sharpe=0.0,  # disable for this test — sharpe approximation
                              # is conservative on pooled disjoint windows
            min_validation_months=3,
        ),
    )
    trainer = WalkforwardTrainer(cfg=cfg)

    # 4 splits => 5 days. Each train + validate window has >= 10 winning
    # trades on quadrant A, so the rule "quadrant=A" qualifies in every
    # validate window. After 4 validates, validation_months_passed=4.
    obs = []
    for window_idx in range(5):  # generates train+validate splits 0..3
        for i in range(15):
            obs.append(_obs(
                ts_ms=window_idx * DAY_MS + i * 1000,
                pnl=0.05,
            ))
    provider = list_observation_provider(obs)
    report = trainer.run(
        start_ms=0, end_ms=5 * DAY_MS, provider=provider, now_ts=1_700_000_000,
    )
    assert report.splits == 4
    assert report.rules_promoted_now == 1
    assert report.rules_production_total == 1


def test_run_writes_promoter_state_when_dir_set(tmp_path):
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
        miner=MinerConfig(min_samples_per_bucket=3,
                          feature_combos=(("quadrant",),)),
        state_dir=str(tmp_path),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    obs = [_obs(ts_ms=i * 1000, pnl=0.05) for i in range(15)]
    provider = list_observation_provider(obs)
    trainer.run(start_ms=0, end_ms=2 * DAY_MS, provider=provider,
                now_ts=1_700_000_000)
    cand_path = os.path.join(str(tmp_path), "candidate_rules.json")
    prod_path = os.path.join(str(tmp_path), "production_rules.json")
    # At least one of the two files exists; both should be valid JSON.
    assert os.path.exists(cand_path) or os.path.exists(prod_path)
    if os.path.exists(cand_path):
        with open(cand_path, encoding="utf-8") as fh:
            json.load(fh)


# ----------------------- regrading discipline ----------------------- #


def test_train_only_rule_dropped_if_no_validate_match():
    """Rule mined on train window has zero matching observations on
    validate window → must be dropped (no basis for stats)."""
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
        miner=MinerConfig(min_samples_per_bucket=3,
                          feature_combos=(("quadrant",),)),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    # Train: A-quadrant. Validate: B-quadrant only.
    train = [_obs(ts_ms=i * 1000, pnl=0.05, q="A") for i in range(5)]
    validate = [_obs(ts_ms=DAY_MS + i * 1000, pnl=0.05, q="B")
                for i in range(5)]
    provider = list_observation_provider(train + validate)
    report = trainer.run(
        start_ms=0, end_ms=2 * DAY_MS, provider=provider,
    )
    # The "quadrant=A" rule would be mined on train but has no
    # validate observation → not in pooled output.
    rids = [w for w in report.windows]
    assert rids[0].rules_mined >= 1  # mined on train
    # But the accumulator stays empty for that rule_id.
    assert report.rules_after_pooling == 0


def test_validate_window_pnl_drives_win_rate():
    """Even if the train window had perfect performance, the
    promoter's view is dictated by the validate window's stats."""
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
        miner=MinerConfig(min_samples_per_bucket=3,
                          feature_combos=(("quadrant",),)),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    train = [_obs(ts_ms=i * 1000, pnl=0.05) for i in range(5)]
    # Validate is all losers.
    validate = [_obs(ts_ms=DAY_MS + i * 1000, pnl=-0.10)
                for i in range(5)]
    provider = list_observation_provider(train + validate)
    report = trainer.run(start_ms=0, end_ms=2 * DAY_MS, provider=provider)
    # Pooled rule must reflect validate-window losers.
    assert report.rules_after_pooling == 1
    # Lookup pooled rule via promoter:
    candidates = trainer.promoter.candidate_rules()
    a_rule = next((r for r in candidates if r.rule_id == "quadrant=A"), None)
    assert a_rule is not None
    assert a_rule.win_rate == 0.0
    assert a_rule.avg_pnl_pct == -0.10


# ----------------------- report shape ----------------------- #


def test_report_as_dict_round_trip():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider([])
    report = trainer.run(start_ms=0, end_ms=2 * DAY_MS, provider=provider)
    d = report.as_dict()
    assert "splits" in d
    assert "windows" in d
    assert "rules_promoted_now" in d
    assert isinstance(d["windows"], list)


def test_window_report_carries_indices_and_window_bounds():
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider([])
    report = trainer.run(start_ms=0, end_ms=3 * DAY_MS, provider=provider)
    assert report.splits == 2
    assert report.windows[0].index == 0
    assert report.windows[1].index == 1
    assert report.windows[0].train_start_ms == 0
    assert report.windows[0].train_end_ms == DAY_MS
    assert report.windows[0].validate_start_ms == DAY_MS
    assert report.windows[0].validate_end_ms == 2 * DAY_MS


# ----------------------- token cap visibility ----------------------- #


def test_tokens_used_zero_in_pure_rule_mode():
    """Phase 4 rule miner is pure-rule; the report must show 0 tokens
    so an audit can prove we stayed under the 200K plan cap."""
    cfg = WalkforwardTrainerConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider([
        _obs(ts_ms=i * 1000, pnl=0.05) for i in range(20)
    ])
    report = trainer.run(start_ms=0, end_ms=3 * DAY_MS, provider=provider)
    assert report.tokens_used == 0
