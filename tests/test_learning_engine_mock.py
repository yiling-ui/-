"""Tests for the Learning Engine: feature extraction, RuleStore, post-mortem
heuristic fallback, and the dynamic_rules.json/.md round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altcoin_agent.learning_engine import (
    DynamicRule,
    PostMortemPick,
    RuleStore,
    bucket_funding,
    bucket_pct,
    bucket_zscore,
    compute_event_result,
    extract_candidate_features,
    run_post_mortem,
    synthesize_dump_slice,
)

# --------------------------------------------------------------------- #
# Bucket helpers
# --------------------------------------------------------------------- #


def test_bucket_pct_signs_and_thresholds() -> None:
    assert bucket_pct(0.0) == "flat"
    assert bucket_pct(0.001) == "flat"
    assert bucket_pct(0.01) == "pos_small"
    assert bucket_pct(-0.01) == "neg_small"
    assert bucket_pct(0.07) == "pos_large"
    assert bucket_pct(-0.20) == "neg_xlarge"


def test_bucket_zscore_signs() -> None:
    assert bucket_zscore(0.5) == "calm"
    assert bucket_zscore(-1.5) == "neg_elevated"
    assert bucket_zscore(3.0) == "pos_high"
    assert bucket_zscore(-5.0) == "neg_extreme"


def test_bucket_funding() -> None:
    assert bucket_funding(0) == "neutral"
    assert bucket_funding(-0.0006) == "negative"
    assert bucket_funding(-0.002) == "very_negative"
    assert bucket_funding(0.0008) == "positive"
    assert bucket_funding(0.0020) == "very_positive"


# --------------------------------------------------------------------- #
# Slice -> result + features
# --------------------------------------------------------------------- #


def test_synthetic_slice_recovers_dump_direction() -> None:
    s = synthesize_dump_slice()
    result = compute_event_result(s)
    assert result.direction == "dump"
    assert result.magnitude_pct < -0.10
    assert result.minutes_to_extremum > 0


def test_extract_features_returns_eight_named_candidates() -> None:
    s = synthesize_dump_slice()
    result = compute_event_result(s)
    features = extract_candidate_features(s, result)
    names = {c.name for c in features}
    expected = {
        "volume_zscore_last1h",
        "funding_pre2h_extreme",
        "oi_growth_pre1h",
        "oi_price_decoupling",
        "range_compression_pre1h",
        "upper_wick_dominance_last1h",
        "lower_wick_dominance_last1h",
        "funding_slope_pre1h",
    }
    assert expected == names


# --------------------------------------------------------------------- #
# RuleStore: persistence + Bayesian update + MD regeneration
# --------------------------------------------------------------------- #


def test_rulestore_persists_json_and_md(tmp_path: Path) -> None:
    json_path = tmp_path / "dynamic_rules.json"
    store = RuleStore(json_path=json_path)
    r = store.update(
        feature_name="oi_growth_pre1h", bucket="pos_xlarge", side="dump",
        hit=True, ts_ms=1000,
    )
    assert r.hits == 1 and r.total == 1
    assert r.hit_rate == pytest.approx((1 + 1) / (1 + 2))
    store.save()

    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["rules"][0]["feature_name"] == "oi_growth_pre1h"
    md_path = json_path.with_suffix(".md")
    assert md_path.exists()
    md = md_path.read_text()
    assert "Active Rules" in md
    assert "oi_growth_pre1h" in md


def test_rulestore_repeated_update_accumulates_not_duplicates(tmp_path: Path) -> None:
    store = RuleStore(json_path=tmp_path / "rules.json")
    store.update(feature_name="f", bucket="b", side="pump", hit=True)
    store.update(feature_name="f", bucket="b", side="pump", hit=True)
    store.update(feature_name="f", bucket="b", side="pump", hit=False)
    rules = store.all_rules()
    assert len(rules) == 1
    assert rules[0].hits == 2 and rules[0].total == 3


def test_rulestore_md_regen_not_append(tmp_path: Path) -> None:
    """Saving twice does not duplicate rows in the markdown file."""
    store = RuleStore(json_path=tmp_path / "rules.json")
    store.update(feature_name="f", bucket="b", side="pump", hit=True)
    store.save()
    first = (tmp_path / "rules.md").read_text()
    store.save()
    second = (tmp_path / "rules.md").read_text()
    assert first == second
    # Only one row containing 'f | b | pump' must exist.
    rows = [ln for ln in second.splitlines() if "| f | b | pump |" in ln]
    assert len(rows) == 1


def test_rulestore_archives_low_hit_rate_high_total(tmp_path: Path) -> None:
    store = RuleStore(
        json_path=tmp_path / "rules.json",
        archive_below_hit_rate=0.5, archive_min_samples=5,
    )
    # 1 hit, 9 total -> hit_rate (1+1)/(9+2) = 18% -> archived.
    rule = DynamicRule(feature_name="loser", bucket="b", side="pump",
                        hits=1, total=9)
    store._rules[rule.key] = rule    # type: ignore[attr-defined]
    store.save()
    md = (tmp_path / "rules.md").read_text()
    # active table is empty; archived has the row
    active_block = md.split("## Active Rules", 1)[1].split("## Archived", 1)[0]
    assert "loser" not in active_block
    archived_block = md.split("## Archived", 1)[1]
    assert "loser" in archived_block


# --------------------------------------------------------------------- #
# Closed-loop heuristic post-mortem (no LLM)
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_post_mortem_persists_picks_with_no_engine(tmp_path: Path) -> None:
    store = RuleStore(json_path=tmp_path / "rules.json")
    s = synthesize_dump_slice(symbol="RAVEUSDT")
    report = await run_post_mortem(
        symbol="RAVEUSDT", target_ts_ms=s.target_ts_ms,
        store=store, engine=None, slice_override=s,
    )
    assert report.result.direction == "dump"
    assert 1 <= len(report.picks) <= 2
    # The picks must be a subset of the candidates we extracted.
    names_in_pool = {c.name for c in report.candidates}
    assert all(p.feature_name in names_in_pool for p in report.picks)
    # Persisted rules cover all candidates (one row per candidate, picks have hits=1).
    saved = store.all_rules()
    assert len(saved) == len(report.candidates)
    picked_keys = {(p.feature_name, p.bucket) for p in report.picks}
    for r in saved:
        if (r.feature_name, r.bucket) in picked_keys:
            assert r.hits == 1


@pytest.mark.asyncio
async def test_run_post_mortem_two_runs_climb_hit_rate(tmp_path: Path) -> None:
    """Same dump pattern twice -> same picks -> hit_rate climbs."""
    store = RuleStore(json_path=tmp_path / "rules.json")
    s1 = synthesize_dump_slice(symbol="A", target_ts_ms=1_000_000_000)
    s2 = synthesize_dump_slice(symbol="B", target_ts_ms=2_000_000_000)
    r1 = await run_post_mortem(symbol="A", target_ts_ms=s1.target_ts_ms,
                                store=store, engine=None, slice_override=s1)
    r2 = await run_post_mortem(symbol="B", target_ts_ms=s2.target_ts_ms,
                                store=store, engine=None, slice_override=s2)

    # The picks in the second run should match the first since the slices
    # are deterministic clones.
    p1 = sorted([(p.feature_name, p.bucket) for p in r1.picks])
    p2 = sorted([(p.feature_name, p.bucket) for p in r2.picks])
    assert p1 == p2

    # And those picks should now have hits=2 / total=2 in the store.
    for fname, bucket in p1:
        rule = store.get(fname, bucket, "dump")
        assert rule is not None
        assert rule.total == 2
        assert rule.hits == 2
        assert rule.hit_rate == pytest.approx((2 + 1) / (2 + 2))


def test_post_mortem_pick_dataclass() -> None:
    p = PostMortemPick(feature_name="x", bucket="b", rationale="r")
    assert p.feature_name == "x"
