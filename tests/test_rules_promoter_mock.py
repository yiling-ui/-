"""tests/test_rules_promoter_mock.py — QUADRANT Phase 4 stub coverage."""

from __future__ import annotations

import json
import os

from altcoin_agent.training.rules_promoter import (
    LearnedRule,
    PromotionConfig,
    RulesPromoter,
)
from altcoin_agent.training.trainer import Trainer, TrainerConfig

# ----------------------- promotion logic ----------------------- #


def _good_rule(rid: str = "r1") -> LearnedRule:
    return LearnedRule(
        rule_id=rid,
        samples=50,
        wins=40,
        losses=10,
        win_rate=0.80,
        sharpe=1.6,
        validation_months_passed=3,
    )


def test_qualifies_promotes_to_production():
    p = RulesPromoter()
    promoted, demoted = p.promote_all([_good_rule()])
    assert len(promoted) == 1
    assert len(demoted) == 0
    assert promoted[0].production_ready is True
    assert p.production_rules() and not p.candidate_rules()


def test_low_samples_blocks_promotion():
    rule = _good_rule()
    rule.samples = 10
    p = RulesPromoter()
    promoted, _ = p.promote_all([rule])
    assert promoted == []
    assert rule.production_ready is False
    assert p.candidate_rules()


def test_low_win_rate_blocks_promotion():
    rule = _good_rule()
    rule.win_rate = 0.79
    p = RulesPromoter()
    promoted, _ = p.promote_all([rule])
    assert promoted == []
    assert p.candidate_rules()


def test_low_sharpe_blocks_promotion():
    rule = _good_rule()
    rule.sharpe = 1.4
    promoted, _ = RulesPromoter().promote_all([rule])
    assert promoted == []


def test_short_validation_blocks_promotion():
    rule = _good_rule()
    rule.validation_months_passed = 2
    promoted, _ = RulesPromoter().promote_all([rule])
    assert promoted == []


# ----------------------- demotion logic ----------------------- #


def test_demotion_requires_grace_period():
    cfg = PromotionConfig(demotion_grace_days=3, demotion_win_rate=0.50)
    p = RulesPromoter(cfg=cfg)

    # Cycle 1: promote.
    rule = _good_rule()
    p.promote_all([rule], now_ts=1)
    assert p.production_rules()

    # Cycles 2-4: degrade rule, but stay within grace.
    bad = _good_rule()
    bad.win_rate = 0.40
    bad.sharpe = 0.5  # also fails sharpe so qualifies returns False
    promoted, demoted = p.promote_all([bad], now_ts=2)
    assert demoted == []   # 1 bad day so far; grace=3
    promoted, demoted = p.promote_all([bad], now_ts=3)
    assert demoted == []
    promoted, demoted = p.promote_all([bad], now_ts=4)
    # 3rd consecutive bad day -> demotion.
    assert len(demoted) == 1
    assert demoted[0].production_ready is False


def test_demotion_does_not_fire_if_winrate_recovers_above_threshold():
    cfg = PromotionConfig(demotion_grace_days=2, demotion_win_rate=0.50)
    p = RulesPromoter(cfg=cfg)
    p.promote_all([_good_rule()], now_ts=1)
    # Win rate above demotion floor -> stays in production even if it
    # otherwise doesn't qualify (e.g. drop in samples).
    weakened = _good_rule()
    weakened.samples = 28  # disqualifies
    weakened.win_rate = 0.70
    promoted, demoted = p.promote_all([weakened], now_ts=2)
    assert demoted == []


# ----------------------- file persistence ----------------------- #


def test_persist_round_trip(tmp_path):
    state_dir = str(tmp_path)
    p = RulesPromoter(state_dir=state_dir)
    p.promote_all([_good_rule(), LearnedRule(rule_id="cand", samples=5)])

    prod_file = os.path.join(state_dir, "production_rules.json")
    cand_file = os.path.join(state_dir, "candidate_rules.json")
    assert os.path.exists(prod_file)
    assert os.path.exists(cand_file)

    with open(prod_file, encoding="utf-8") as fh:
        payload = json.load(fh)
    rids = {r["rule_id"] for r in payload["rules"]}
    assert "r1" in rids

    # Reload in a fresh promoter — the production list survives.
    p2 = RulesPromoter(state_dir=state_dir)
    prods = {r.rule_id for r in p2.production_rules()}
    assert "r1" in prods


def test_promoting_then_re_evaluating_promoted_rule_is_idempotent():
    p = RulesPromoter()
    promoted_first, _ = p.promote_all([_good_rule()])
    # Same rule on second cycle: should not re-promote (it's already in).
    promoted_second, _ = p.promote_all([_good_rule()])
    assert len(promoted_first) == 1
    assert promoted_second == []
    assert len(p.production_rules()) == 1


# ----------------------- trainer wrapper ----------------------- #


def test_trainer_run_cycle_writes_report(tmp_path):
    cfg = TrainerConfig(report_dir=str(tmp_path))
    t = Trainer(cfg=cfg)
    report = t.run_cycle([_good_rule()], now_ts=1_700_000_000)
    assert report.rules_seen == 1
    assert report.rules_promoted == 1
    out_files = os.listdir(str(tmp_path))
    assert any(f.startswith("daily_report_") for f in out_files)


def test_trainer_run_cycle_works_without_report_dir():
    t = Trainer(cfg=TrainerConfig(report_dir=None))
    report = t.run_cycle([], now_ts=1)
    assert report.rules_seen == 0
    assert report.rules_promoted == 0
