"""
Mock tests for learning_engine.py.

No DeepSeek calls, no network. We exercise:

    * synthesize_dump_slice → realistic, deterministic fixture
    * compute_event_result → derives DUMP from the slice
    * extract_candidate_features → produces grounded, bucketed features
    * fallback picker → chooses 1-2 features by direction_hint
    * RuleStore Bayesian update + persistence + markdown rendering
    * Archival of low-hit-rate rules
    * dynamic_rules.md is REGENERATED, not appended
    * No-LLM run_post_mortem path uses the fallback
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altcoin_agent.learning_engine import (
    DynamicRule,
    EventSide,
    RuleStore,
    _fallback_pick,
    compute_event_result,
    extract_candidate_features,
    run_post_mortem,
    synthesize_dump_slice,
)

# --------------------------------------------------------------------------- #
# Fixture builder
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_store(tmp_path: Path) -> RuleStore:
    return RuleStore(
        json_path=tmp_path / "dynamic_rules.json",
        md_path=tmp_path / "dynamic_rules.md",
    )


# --------------------------------------------------------------------------- #
# 1. Slice → result → features
# --------------------------------------------------------------------------- #


def test_synthesized_slice_is_complete() -> None:
    slc = synthesize_dump_slice()
    assert slc.is_complete()
    assert len(slc.klines_1m) == 240
    assert len(slc.funding_rates) >= 1
    assert len(slc.open_interest) >= 1


def test_compute_event_result_detects_DUMP() -> None:
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    assert result.side == EventSide.DUMP
    # The synthesized slice drops about 22% in the last 10 minutes,
    # but the OPEN of the slice is at $1.00 which is also near the high,
    # so realized magnitude is close to the dump depth.
    assert result.magnitude_pct >= 0.15
    # The extreme is hit late in the window (last 10m).
    assert result.minutes_to_extreme >= 230


def test_extract_features_returns_grounded_set() -> None:
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)

    # Core sanity: enough features, all named, all bucketed
    names = {f.name for f in features}
    assert {
        "funding_pre2h_extreme",
        "oi_growth_pre1h",
        "oi_price_decoupling",
        "volume_zscore_last1h",
        "range_compression_pre1h",
        "wick_asymmetry_last30m",
        "funding_slope_last1h",
        "upper_wick_dominance_last1h",
    }.issubset(names)
    for f in features:
        assert f.name and f.bucket and f.direction_hint
        assert isinstance(f.value, float)


def test_extracted_features_for_dump_have_supporting_hints() -> None:
    """In our textbook DUMP fixture, at least these features must be flagged
    `supports_dump`:
        * funding_pre2h_extreme — funding has been climbing positive
        * oi_price_decoupling   — OI grew while price compressed
        * upper_wick_dominance  — top-side rejection over last hour
    """
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    by_name = {f.name: f for f in features}

    # funding extreme: in the synthetic data funding starts at +0.02% and
    # climbs to +0.15% over 4h, so the pre-2h value is > +0.05% and gets
    # the "supports_dump" hint
    assert by_name["funding_pre2h_extreme"].direction_hint == "supports_dump"

    # OI grew ~22% while price compressed → decoupling supports a dump
    assert by_name["oi_price_decoupling"].direction_hint == "supports_dump"

    # Upper-wick dominance in last hour
    assert by_name["upper_wick_dominance_last1h"].direction_hint == "supports_dump"


# --------------------------------------------------------------------------- #
# 2. Fallback picker
# --------------------------------------------------------------------------- #


def test_fallback_picker_selects_supportive_extreme_features() -> None:
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    verdict = _fallback_pick(features)

    assert verdict.picks
    assert len(verdict.picks) <= 2
    # Picks must be named features that exist in our candidates
    names_in = {f.name for f in features}
    for p in verdict.picks:
        assert p.feature_name in names_in
        assert 1 <= p.rank <= 2

    # Picked features should have supportive direction (or extreme value).
    # In our setup at least one of them must be supports_dump.
    by_name = {f.name: f for f in features}
    hints = {by_name[p.feature_name].direction_hint for p in verdict.picks}
    assert "supports_dump" in hints


def test_fallback_picker_does_not_invent_features() -> None:
    """The picker must only select from the candidates given."""
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    names = {f.name for f in features}
    verdict = _fallback_pick(features)
    for p in verdict.picks:
        assert p.feature_name in names


# --------------------------------------------------------------------------- #
# 3. RuleStore — Bayesian update, archival, regeneration
# --------------------------------------------------------------------------- #


def test_rule_store_persists_and_increments_on_repeated_event(
    tmp_store: RuleStore,
) -> None:
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    verdict = _fallback_pick(features)

    # First update
    touched1 = tmp_store.update_with_picks(verdict, features, result)
    pick_names = [p.feature_name for p in verdict.picks]
    pick_rules = [r for r in touched1 if r.feature_name in pick_names]
    assert all(r.hits == 1 for r in pick_rules)
    assert all(r.total == 1 for r in pick_rules)

    # Same event again — picked rules accumulate, not duplicate
    touched2 = tmp_store.update_with_picks(verdict, features, result)
    pick_rules_2 = [r for r in touched2 if r.feature_name in pick_names]
    assert all(r.hits == 2 for r in pick_rules_2)
    assert all(r.total == 2 for r in pick_rules_2)

    # And there's only ONE entry per (feature, bucket, side) in the store
    rule_ids = list(tmp_store.rules.keys())
    assert len(rule_ids) == len(set(rule_ids))


def test_rule_store_files_are_written_with_expected_structure(
    tmp_store: RuleStore,
) -> None:
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    verdict = _fallback_pick(features)
    tmp_store.update_with_picks(verdict, features, result)

    # JSON exists, parses, has rules
    assert tmp_store.json_path.exists()
    data = json.loads(tmp_store.json_path.read_text())
    assert "rules" in data
    assert len(data["rules"]) >= 1

    # MD exists with the auto-generated header + tables
    assert tmp_store.md_path.exists()
    md = tmp_store.md_path.read_text()
    assert "inclusion: always" in md
    assert "Dynamic Trading Rules" in md
    assert "Active Rules" in md
    assert "Archived" in md


def test_rule_store_md_is_REGENERATED_not_appended(tmp_store: RuleStore) -> None:
    """Critical: dynamic_rules.md must be regenerated each save, not appended.
    Otherwise the file grows unboundedly."""
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    verdict = _fallback_pick(features)

    # Three rounds with the SAME picks
    for _ in range(3):
        tmp_store.update_with_picks(verdict, features, result)

    md = tmp_store.md_path.read_text()
    # The header should appear EXACTLY ONCE
    assert md.count("# Dynamic Trading Rules") == 1
    assert md.count("inclusion: always") == 1
    assert md.count("## Active Rules") == 1
    # And there should be only one row per picked rule, not three
    for p in verdict.picks:
        # Feature name appears in active table; rough check via backticked name
        assert md.count(f"`{p.feature_name}`") <= 3  # could appear in active + archive + summary


def test_rule_store_archives_low_hit_rate_rules() -> None:
    store = RuleStore(
        json_path=Path("/tmp/test_archive.json"),
        md_path=Path("/tmp/test_archive.md"),
    )
    store.json_path.unlink(missing_ok=True)
    store.md_path.unlink(missing_ok=True)
    store = RuleStore(json_path=store.json_path, md_path=store.md_path)

    # Inject a rule with 1 hit out of 6 total → hit_rate = 2/8 = 0.25
    rule = DynamicRule(
        feature_name="funding_pre2h_extreme",
        bucket="neutral",
        side=EventSide.PUMP,
        hits=1, total=6,
        last_seen_iso="2025-01-01T00:00:00Z",
        last_summary="weak rule",
    )
    store.rules[rule.rule_id] = rule
    active, archived = store.active_and_archived()
    assert rule not in active
    assert rule in archived


def test_rule_store_active_rules_sorted_by_hit_rate(tmp_store: RuleStore) -> None:
    # Inject rules with hit rates 0.7, 0.6, 0.55 directly
    for name, hits, total in [("a", 7, 10), ("b", 5, 10), ("c", 6, 12)]:
        r = DynamicRule(
            feature_name=name, bucket="medium", side=EventSide.DUMP,
            hits=hits, total=total, last_seen_iso="2025-01-01T00:00:00Z",
        )
        tmp_store.rules[r.rule_id] = r
    active, _ = tmp_store.active_and_archived()
    rates = [r.hit_rate for r in active]
    assert rates == sorted(rates, reverse=True)


def test_rule_store_loads_from_disk(tmp_path: Path) -> None:
    json_path = tmp_path / "rules.json"
    md_path = tmp_path / "rules.md"

    store_a = RuleStore(json_path=json_path, md_path=md_path)
    slc = synthesize_dump_slice()
    result = compute_event_result(slc)
    assert result is not None
    features = extract_candidate_features(slc, result)
    verdict = _fallback_pick(features)
    store_a.update_with_picks(verdict, features, result)

    # New instance should pick up the persisted rules
    store_b = RuleStore(json_path=json_path, md_path=md_path)
    assert len(store_b.rules) == len(store_a.rules)
    for rid, r in store_a.rules.items():
        assert rid in store_b.rules
        assert store_b.rules[rid].hits == r.hits
        assert store_b.rules[rid].total == r.total


# --------------------------------------------------------------------------- #
# 4. Top-level orchestrator (no-LLM path)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_post_mortem_end_to_end_no_llm(tmp_path: Path) -> None:
    json_path = tmp_path / "rules.json"
    md_path = tmp_path / "rules.md"
    store = RuleStore(json_path=json_path, md_path=md_path)

    slc = synthesize_dump_slice(symbol="RAVEUSDT", target_ts_ms=1_700_000_000_000)
    run = await run_post_mortem(
        symbol="RAVEUSDT", target_ts_ms=1_700_000_000_000,
        engine=None,                     # no LLM available
        rule_store=store,
        slice_override=slc,
    )

    assert run.used_fallback is True
    assert run.result is not None and run.result.side == EventSide.DUMP
    assert run.features
    assert run.verdict is not None and len(run.verdict.picks) >= 1
    assert run.touched_rules
    # Markdown was actually written and is non-trivial
    md = md_path.read_text()
    assert "Dynamic Trading Rules" in md
    # And the JSON has the picked rules with hits=1
    data = json.loads(json_path.read_text())
    pick_names = {p.feature_name for p in run.verdict.picks}
    rule_for_pick = [r for r in data["rules"]
                     if r["feature_name"] in pick_names and r["side"] == "dump"]
    assert all(r["hits"] >= 1 for r in rule_for_pick)


@pytest.mark.asyncio
async def test_two_runs_accumulate_in_persistent_store(tmp_path: Path) -> None:
    json_path = tmp_path / "rules.json"
    md_path = tmp_path / "rules.md"
    store = RuleStore(json_path=json_path, md_path=md_path)

    base_ts = 1_700_000_000_000
    for i in range(3):
        slc = synthesize_dump_slice(
            symbol=f"TOK{i}USDT",
            target_ts_ms=base_ts + i * 24 * 3600 * 1000,
        )
        await run_post_mortem(
            symbol=slc.symbol, target_ts_ms=slc.target_ts_ms,
            engine=None, rule_store=store, slice_override=slc,
        )

    # Same shape thrice → at least one rule should have hits == 3
    max_hits = max(r.hits for r in store.rules.values())
    assert max_hits >= 3

    # And markdown still exists with one header
    md = md_path.read_text()
    assert md.count("# Dynamic Trading Rules") == 1
