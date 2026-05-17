"""Unit tests for ``social.historical_analyzer``.

The analyzer has three independent surfaces — each gets its own focused
test cluster:

  * ``KOLHistoryStore`` — counter math + atomic persistence + corruption
    resilience.
  * ``HistoricalAnalyzer`` — the fuser-facing ``adjust_kol_confidence``
    hook + ``record_observation`` write path.
  * ``build_observations_from_posts`` — async batch helper that replays
    posts through a kline fetcher into observations.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from altcoin_agent.social.historical_analyzer import (
    HistoricalAnalyzer,
    HistoricalAnalyzerConfig,
    KOLAdjustment,
    KOLHistoryStore,
    KOLObservation,
    KOLScore,
    _classify_post_intent,
    _outcome_from_klines,
    _PostHandle,
    build_observations_from_posts,
    normalize_author,
)

# --------------------------------------------------------------------- #
# normalize_author
# --------------------------------------------------------------------- #


def test_normalize_author_strips_prefix_and_casing():
    assert normalize_author("@KOL_X") == "kol_x"
    assert normalize_author("$Kol_X") == "kol_x"
    assert normalize_author("  Kol_X  ") == "kol_x"
    assert normalize_author("@@kol_x") == "kol_x"


def test_normalize_author_empty_inputs_return_empty_string():
    assert normalize_author("") == ""
    assert normalize_author(None) == ""
    assert normalize_author("   ") == ""


# --------------------------------------------------------------------- #
# KOLObservation.is_correct
# --------------------------------------------------------------------- #


def test_observation_is_correct_for_aligned_intent_direction_pairs():
    pump_call = KOLObservation(
        author="x", symbol="PEPE/USDT:USDT", intent="frontrun_call",
        ts_ms=100, realised_direction="pump", magnitude_pct=0.10,
    )
    assert pump_call.is_correct() is True

    exit_dump = KOLObservation(
        author="x", symbol="PEPE/USDT:USDT", intent="exit_liquidity",
        ts_ms=100, realised_direction="dump", magnitude_pct=-0.08,
    )
    assert exit_dump.is_correct() is True


def test_observation_is_incorrect_when_direction_misses():
    wrong_call = KOLObservation(
        author="x", symbol="PEPE/USDT:USDT", intent="frontrun_call",
        ts_ms=100, realised_direction="dump", magnitude_pct=-0.05,
    )
    assert wrong_call.is_correct() is False

    wrong_exit = KOLObservation(
        author="x", symbol="PEPE/USDT:USDT", intent="exit_liquidity",
        ts_ms=100, realised_direction="pump", magnitude_pct=0.05,
    )
    assert wrong_exit.is_correct() is False


def test_observation_neutral_intent_never_scores_as_correct():
    obs = KOLObservation(
        author="x", symbol="PEPE/USDT:USDT", intent="neutral",
        ts_ms=100, realised_direction="pump", magnitude_pct=0.10,
    )
    assert obs.is_correct() is False


# --------------------------------------------------------------------- #
# KOLHistoryStore — counter math
# --------------------------------------------------------------------- #


def _make_obs(
    *, author="goat", symbol="PEPE/USDT:USDT", intent="frontrun_call",
    direction="pump", mag=0.05, ts_ms=1000,
) -> KOLObservation:
    return KOLObservation(
        author=author, symbol=symbol, intent=intent,
        ts_ms=ts_ms, realised_direction=direction, magnitude_pct=mag,
    )


def test_store_record_increments_total_only_on_first_call(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json")
    score = store.record(_make_obs(direction="pump"))
    assert isinstance(score, KOLScore)
    assert score.hits == 1
    assert score.total == 1
    # Laplace-smoothed: 1 hit, 1 total -> (1+1)/(1+2) = 2/3
    assert pytest.approx(score.hit_rate, rel=1e-9) == 2 / 3


def test_store_record_neutral_intent_returns_none_and_skips_storage(
    tmp_path: Path,
):
    store = KOLHistoryStore(path=tmp_path / "kol.json")
    out = store.record(_make_obs(intent="neutral"))
    assert out is None
    # Neutral observations must not pollute the store.
    assert len(store) == 0


def test_store_record_anonymous_author_returns_none(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json")
    out = store.record(_make_obs(author=""))
    assert out is None
    assert len(store) == 0


def test_store_record_multiple_observations_track_hit_rate(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    # 7 wins, 3 losses on frontrun_call.
    for i in range(7):
        store.record(_make_obs(direction="pump", ts_ms=1000 + i))
    for i in range(3):
        store.record(_make_obs(direction="dump", ts_ms=2000 + i))
    score = store.get_score("goat", "frontrun_call")
    assert score is not None
    assert score.hits == 7
    assert score.total == 10
    # Laplace: 8/12 ≈ 0.667
    assert pytest.approx(score.hit_rate, rel=1e-9) == 8 / 12
    # avg magnitude is the running mean of |magnitude|.
    assert pytest.approx(score.avg_magnitude_pct, rel=1e-9) == 0.05


def test_store_partitions_intents_independently(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    store.record(_make_obs(intent="frontrun_call", direction="pump"))
    store.record(
        _make_obs(intent="exit_liquidity", direction="dump", mag=-0.07),
    )
    fc = store.get_score("goat", "frontrun_call")
    el = store.get_score("goat", "exit_liquidity")
    assert fc is not None and el is not None
    # Both 1/1 hits on their respective intents — counters do NOT mix.
    assert fc.hits == 1 and fc.total == 1
    assert el.hits == 1 and el.total == 1
    # Magnitudes are stored as absolute values.
    assert pytest.approx(el.avg_magnitude_pct, rel=1e-9) == 0.07


def test_store_last_seen_ts_is_max_observed(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    store.record(_make_obs(ts_ms=2000))
    store.record(_make_obs(ts_ms=1000))  # out of order
    store.record(_make_obs(ts_ms=3000))
    score = store.get_score("goat", "frontrun_call")
    assert score is not None
    assert score.last_seen_ts_ms == 3000


def test_store_normalises_author_for_lookup(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    store.record(_make_obs(author="@KOL_X"))
    # Lookups via different surface forms hit the same row.
    a = store.get_score("kol_x", "frontrun_call")
    b = store.get_score("@KOL_X", "frontrun_call")
    c = store.get_score("$KOL_X", "frontrun_call")
    assert a == b == c
    assert a is not None and a.total == 1


def test_store_all_scores_filters_by_intent(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    store.record(_make_obs(author="a", intent="frontrun_call"))
    store.record(
        _make_obs(author="b", intent="exit_liquidity", direction="dump",
                  mag=-0.05),
    )
    fc = store.all_scores(intent="frontrun_call")
    el = store.all_scores(intent="exit_liquidity")
    assert {s.author for s in fc} == {"a"}
    assert {s.author for s in el} == {"b"}
    # No filter -> both.
    everyone = store.all_scores()
    assert {(s.author, s.intent) for s in everyone} == {
        ("a", "frontrun_call"), ("b", "exit_liquidity"),
    }


def test_store_reset_wipes_counters(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    store.record(_make_obs())
    assert len(store) == 1
    store.reset()
    assert len(store) == 0
    assert store.get_score("goat", "frontrun_call") is None


# --------------------------------------------------------------------- #
# KOLHistoryStore — persistence
# --------------------------------------------------------------------- #


def test_store_save_and_reload_round_trip(tmp_path: Path):
    path = tmp_path / "kol.json"
    s1 = KOLHistoryStore(path=path)
    s1.record(_make_obs(author="kol_x", direction="pump"))
    s1.record(_make_obs(author="kol_x", direction="dump"))
    s1.record(_make_obs(author="kol_y", direction="pump"))
    s1.save()

    s2 = KOLHistoryStore(path=path)
    assert len(s2) == 2
    score_x = s2.get_score("kol_x", "frontrun_call")
    assert score_x is not None
    assert score_x.hits == 1
    assert score_x.total == 2


def test_store_persistence_uses_atomic_write(tmp_path: Path):
    # The save() path renders to a sibling .tmp file then os.replace's.
    # We assert that no lingering tmp file is left after save.
    path = tmp_path / "kol.json"
    store = KOLHistoryStore(path=path)
    store.record(_make_obs())
    store.save()
    assert path.exists()
    # tmp suffix lives at path.suffix + ".tmp" -> ".json.tmp"
    siblings = [p.name for p in tmp_path.iterdir()]
    assert "kol.json" in siblings
    assert "kol.json.tmp" not in siblings


def test_store_corrupt_file_falls_back_to_empty(tmp_path: Path):
    path = tmp_path / "kol.json"
    path.write_text("{ this is not valid json")
    store = KOLHistoryStore(path=path)
    assert len(store) == 0
    # Recovery path is silent: a fresh write does not raise.
    store.record(_make_obs())
    store.save()
    # New file is valid JSON.
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1


def test_store_wrong_schema_version_wipes(tmp_path: Path):
    path = tmp_path / "kol.json"
    path.write_text(json.dumps({
        "schema_version": 999,
        "counters": {"kol_x|frontrun_call": {"hits": 1, "total": 1}},
    }))
    store = KOLHistoryStore(path=path)
    assert len(store) == 0


def test_store_malformed_entries_are_skipped(tmp_path: Path):
    path = tmp_path / "kol.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "counters": {
            "good|frontrun_call": {
                "hits": 2, "total": 5,
                "avg_magnitude_pct": 0.1, "last_seen_ts_ms": 1234,
            },
            "no_pipe": {"hits": 1, "total": 1},  # invalid key
            "another|frontrun_call": "not a dict",
            "broken|frontrun_call": {
                "hits": "not_an_int", "total": 5,
            },
        },
    }))
    store = KOLHistoryStore(path=path)
    # Only the "good" entry survives.
    assert len(store) == 1
    score = store.get_score("good", "frontrun_call")
    assert score is not None and score.hits == 2 and score.total == 5


def test_store_autosave_persists_each_record(tmp_path: Path):
    path = tmp_path / "kol.json"
    s1 = KOLHistoryStore(path=path, autosave=True)
    s1.record(_make_obs())
    # Re-open without explicit save — autosave should have flushed.
    s2 = KOLHistoryStore(path=path)
    assert len(s2) == 1


# --------------------------------------------------------------------- #
# HistoricalAnalyzer — record_observation
# --------------------------------------------------------------------- #


def test_analyzer_record_observation_returns_score(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    score = analyzer.record_observation(
        author="kol_x", symbol="PEPE/USDT:USDT",
        intent="frontrun_call", ts_ms=1000,
        realised_direction="pump", magnitude_pct=0.10,
    )
    assert score is not None
    assert score.hits == 1
    assert score.total == 1


def test_analyzer_record_observation_swallows_invalid_input(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    # ts_ms cannot be coerced -> rejected without raising.
    out = analyzer.record_observation(
        author="kol_x", symbol="PEPE/USDT:USDT",
        intent="frontrun_call", ts_ms="not-a-number",  # type: ignore[arg-type]
        realised_direction="pump", magnitude_pct=0.10,
    )
    assert out is None
    assert len(store) == 0


def test_analyzer_lookup_proxy(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    analyzer.record_observation(
        author="kol_x", symbol="PEPE/USDT:USDT",
        intent="exit_liquidity", ts_ms=1000,
        realised_direction="dump", magnitude_pct=-0.07,
    )
    found = analyzer.lookup("KOL_X", "exit_liquidity")
    assert found is not None
    assert found.total == 1


# --------------------------------------------------------------------- #
# HistoricalAnalyzer — adjust_kol_confidence
# --------------------------------------------------------------------- #


def _seed_author(
    analyzer: HistoricalAnalyzer,
    *, author: str, intent: str, hits: int, total: int,
    direction_correct="pump" if True else "dump",
):
    """Helper that rolls the counters until they match (hits, total).

    Aware of intent so the generated direction / magnitude pair flips
    is_correct() the right way.
    """
    # Intent -> the realised direction that scores as a hit.
    correct_dir = "pump" if intent == "frontrun_call" else "dump"
    wrong_dir = "dump" if correct_dir == "pump" else "pump"
    correct_mag = 0.05 if intent == "frontrun_call" else -0.05
    wrong_mag = -0.05 if intent == "frontrun_call" else 0.05
    for i in range(hits):
        analyzer.record_observation(
            author=author, symbol="PEPE/USDT:USDT", intent=intent,  # type: ignore[arg-type]
            ts_ms=1_000 + i,
            realised_direction=correct_dir,  # type: ignore[arg-type]
            magnitude_pct=correct_mag,
        )
    for i in range(total - hits):
        analyzer.record_observation(
            author=author, symbol="PEPE/USDT:USDT", intent=intent,  # type: ignore[arg-type]
            ts_ms=2_000 + i,
            realised_direction=wrong_dir,  # type: ignore[arg-type]
            magnitude_pct=wrong_mag,
        )


def test_adjust_returns_unchanged_when_no_authors_supplied(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
    )
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.6, authors=[],
    )
    assert isinstance(out, KOLAdjustment)
    assert out.adjusted_confidence == 0.6
    assert out.delta == 0.0
    assert out.note == "no_authors"


def test_adjust_returns_unchanged_for_neutral_intent(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
    )
    out = analyzer.adjust_kol_confidence(
        kol_intent="neutral", confidence=0.7, authors=["kol_x"],
    )
    assert out.adjusted_confidence == 0.7
    assert out.delta == 0.0
    assert "neutral_intent_skipped" in out.note


def test_adjust_skips_authors_below_min_samples(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(min_samples=10),
    )
    # Only 5 samples; below the floor.
    _seed_author(analyzer, author="newbie", intent="exit_liquidity",
                 hits=4, total=5)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.65, authors=["newbie"],
    )
    assert out.adjusted_confidence == 0.65
    assert out.delta == 0.0
    assert out.note == "insufficient_samples"
    # The author still surfaces so the operator knows the layer ran.
    assert "newbie" in out.contributing_authors


def test_adjust_lifts_confidence_for_well_calibrated_kol(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(
            min_samples=10, strong_bound=0.65, conf_lift_max=0.20,
        ),
    )
    # 19/20 hits ≈ 95% raw, ~83% Laplace-smoothed.
    _seed_author(analyzer, author="goat", intent="exit_liquidity",
                 hits=19, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.50, authors=["goat"],
    )
    assert out.delta > 0.0
    assert out.adjusted_confidence > 0.50
    # Cap is 0.20 lift on a 0.50 base => never above 0.70.
    assert out.adjusted_confidence <= 0.70 + 1e-9


def test_adjust_drops_confidence_for_unreliable_kol(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(
            min_samples=10, weak_bound=0.40, conf_drop_max=0.20,
        ),
    )
    # 2/20 hits ≈ 10% raw, ~14% Laplace.
    _seed_author(analyzer, author="permabear", intent="exit_liquidity",
                 hits=2, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.70, authors=["permabear"],
    )
    assert out.delta < 0.0
    assert out.adjusted_confidence < 0.70
    assert out.adjusted_confidence >= 0.50 - 1e-9


def test_adjust_no_change_when_rate_within_neutral_band(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(
            min_samples=10, weak_bound=0.40, strong_bound=0.65,
        ),
    )
    # 11/20 hits ≈ 55% raw, ~54% Laplace -> in the neutral band.
    _seed_author(analyzer, author="meh", intent="exit_liquidity",
                 hits=11, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.60, authors=["meh"],
    )
    assert out.delta == 0.0
    assert out.adjusted_confidence == 0.60


def test_adjust_clamps_to_unit_interval(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(min_samples=5, conf_lift_max=0.20),
    )
    _seed_author(analyzer, author="goat", intent="frontrun_call",
                 hits=20, total=20)
    # 0.95 + max-lift 0.20 -> clamps at 1.0.
    out = analyzer.adjust_kol_confidence(
        kol_intent="frontrun_call", confidence=0.95, authors=["goat"],
    )
    assert out.adjusted_confidence <= 1.0
    assert out.adjusted_confidence > 0.95


def test_adjust_handles_oversaturated_input_confidence(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
    )
    # A buggy provider hands us 1.5; we must clamp before applying delta.
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=1.5, authors=[],
    )
    assert out.original_confidence <= 1.0
    assert out.adjusted_confidence <= 1.0


def test_adjust_weights_authors_by_sample_size(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(min_samples=10),
    )
    # Author A: 19/20 calibrated hits.
    _seed_author(analyzer, author="goat", intent="exit_liquidity",
                 hits=19, total=20)
    # Author B: 12/20 — barely above neutral band.
    _seed_author(analyzer, author="meh", intent="exit_liquidity",
                 hits=12, total=20)
    out_only_goat = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.50, authors=["goat"],
    )
    out_both = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.50,
        authors=["goat", "meh"],
    )
    # Adding a meh (neutral-band) caller should pull the lift toward 0;
    # the lift with both authors is still positive but smaller than goat-alone.
    assert out_only_goat.delta > out_both.delta >= 0.0


def test_adjust_returns_serialisable_diagnostic_dict(tmp_path: Path):
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
    )
    _seed_author(analyzer, author="goat", intent="exit_liquidity",
                 hits=15, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.5, authors=["goat"],
    )
    payload = out.as_dict()
    assert payload["intent"] == "exit_liquidity"
    assert payload["original_confidence"] == 0.5
    assert "delta" in payload
    assert "contributing_authors" in payload
    # JSON-round-trippable.
    assert json.loads(json.dumps(payload))["intent"] == "exit_liquidity"


# --------------------------------------------------------------------- #
# _classify_post_intent
# --------------------------------------------------------------------- #


def test_classify_post_intent_picks_exit_liquidity_for_dump_words():
    assert _classify_post_intent(
        "Time to take profit on this name, exit liquidity",
    ) == "exit_liquidity"


def test_classify_post_intent_picks_frontrun_for_buy_words():
    assert _classify_post_intent(
        "BUY now or get rekt, easy 100x send it",
    ) == "frontrun_call"


def test_classify_post_intent_neutral_when_both_or_neither():
    assert _classify_post_intent("") == "neutral"
    assert _classify_post_intent("on chain volume looks normal") == "neutral"
    # Mixed words -> neutral (avoids false positives).
    assert _classify_post_intent(
        "Some buy, some take profit",
    ) == "neutral"


# --------------------------------------------------------------------- #
# _outcome_from_klines
# --------------------------------------------------------------------- #


def test_outcome_classifies_pump_when_high_dominates():
    bars = [
        # (open_ts_ms, high, low, close)
        (1, 1.10, 1.00, 1.05),
        (2, 1.12, 1.05, 1.10),
    ]
    direction, mag = _outcome_from_klines(
        bars, pre_close=1.00,
        pump_threshold=0.05, dump_threshold=0.05,
    )
    assert direction == "pump"
    assert pytest.approx(mag, rel=1e-9) == 0.12


def test_outcome_classifies_dump_when_low_dominates():
    bars = [
        (1, 1.01, 0.95, 0.98),
        (2, 0.99, 0.85, 0.86),
    ]
    direction, mag = _outcome_from_klines(
        bars, pre_close=1.00,
        pump_threshold=0.05, dump_threshold=0.05,
    )
    assert direction == "dump"
    assert mag < 0  # signed
    assert pytest.approx(abs(mag), rel=1e-9) == 0.15


def test_outcome_neutral_when_neither_threshold_breached():
    bars = [
        (1, 1.01, 0.99, 1.00),
        (2, 1.02, 0.98, 1.01),
    ]
    direction, _ = _outcome_from_klines(
        bars, pre_close=1.00,
        pump_threshold=0.05, dump_threshold=0.05,
    )
    assert direction == "neutral"


def test_outcome_neutral_when_inputs_invalid():
    direction, mag = _outcome_from_klines(
        [], pre_close=1.0, pump_threshold=0.05, dump_threshold=0.05,
    )
    assert direction == "neutral"
    assert mag == 0.0
    direction, mag = _outcome_from_klines(
        [(1, 1.0, 0.9, 0.95)], pre_close=0.0,
        pump_threshold=0.05, dump_threshold=0.05,
    )
    assert direction == "neutral"
    assert mag == 0.0


# --------------------------------------------------------------------- #
# build_observations_from_posts
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_observations_classifies_pump_call():
    posts = [
        _PostHandle(
            author="goat", symbol="PEPE/USDT:USDT", ts_ms=10_000,
            intent="frontrun_call", follower_count=15_000,
        )
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        # Pre-bar at 9_500 (close=1.0); post-bars 11_000+ (high climbs to 1.20)
        return [
            (9_500, 1.01, 0.99, 1.00),
            (10_500, 1.10, 1.00, 1.08),
            (11_000, 1.20, 1.05, 1.18),
        ]

    obs = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
        forward_window_sec=3600, pre_window_sec=120,
        pump_threshold=0.05, dump_threshold=0.05,
    )
    assert len(obs) == 1
    o = obs[0]
    assert o.author == "goat"
    assert o.intent == "frontrun_call"
    assert o.realised_direction == "pump"
    assert o.is_correct() is True


@pytest.mark.asyncio
async def test_build_observations_classifies_exit_liquidity_dump():
    posts = [
        _PostHandle(
            author="exitor", symbol="PEPE/USDT:USDT", ts_ms=10_000,
            intent="exit_liquidity",
        )
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        return [
            (9_500, 1.01, 0.99, 1.00),
            (10_500, 0.95, 0.85, 0.86),
        ]

    obs = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
    )
    assert len(obs) == 1
    assert obs[0].realised_direction == "dump"
    assert obs[0].is_correct() is True


@pytest.mark.asyncio
async def test_build_observations_skips_when_no_pre_bars():
    posts = [
        _PostHandle(
            author="goat", symbol="X/USDT:USDT", ts_ms=10_000,
            intent="frontrun_call",
        )
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        # All bars are post-post.
        return [(10_500, 1.10, 1.00, 1.08)]

    obs = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
    )
    assert obs == []


@pytest.mark.asyncio
async def test_build_observations_skips_neutral_intent():
    posts = [
        _PostHandle(
            author="goat", symbol="X/USDT:USDT", ts_ms=10_000,
            intent="neutral",
        )
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        raise AssertionError("should not be called for neutral intent")

    obs = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
    )
    assert obs == []


@pytest.mark.asyncio
async def test_build_observations_swallows_fetcher_failures():
    posts = [
        _PostHandle(author="a", symbol="X", ts_ms=1_000, intent="frontrun_call"),
        _PostHandle(author="b", symbol="X", ts_ms=2_000, intent="frontrun_call"),
    ]
    calls = {"n": 0}

    async def fetch(symbol, since, until):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("API timeout")
        return [
            (1_500, 1.0, 1.0, 1.0),
            (2_500, 1.20, 1.05, 1.18),
        ]

    obs = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
    )
    # First post failed -> skipped; second succeeded.
    assert len(obs) == 1
    assert obs[0].author == "b"


@pytest.mark.asyncio
async def test_build_observations_accepts_dict_posts():
    raw_posts = [
        {
            "author": "goat", "symbol": "PEPE",
            "ts_ms": 10_000, "intent": "frontrun_call",
            "follower_count": 5_000,
        }
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        return [
            (9_900, 1.0, 1.0, 1.0),
            (10_500, 1.20, 1.05, 1.18),
        ]

    obs = await build_observations_from_posts(
        posts=raw_posts, fetch_klines=fetch,
    )
    assert len(obs) == 1
    assert obs[0].follower_count == 5_000


@pytest.mark.asyncio
async def test_build_observations_classifies_text_when_intent_missing():
    raw_posts = [
        {
            "author": "anon", "symbol": "PEPE",
            "ts_ms": 10_000,
            # No 'intent' supplied; classifier should pick exit_liquidity.
            "text": "TIME TO TAKE PROFIT here, careful, exit liquidity ahead",
        }
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        return [
            (9_500, 1.0, 1.0, 1.0),
            (10_500, 0.95, 0.86, 0.87),  # dumps 13%
        ]

    obs = await build_observations_from_posts(
        posts=raw_posts, fetch_klines=fetch,
    )
    assert len(obs) == 1
    assert obs[0].intent == "exit_liquidity"
    assert obs[0].realised_direction == "dump"


@pytest.mark.asyncio
async def test_build_then_record_round_trip(tmp_path: Path):
    """End-to-end: posts -> observations -> store records adjust_kol works."""
    # Use far-apart timestamps so each post's pre-window doesn't overlap
    # with the next post's post-window in the fixture below.
    posts = [
        _PostHandle(
            author="goat", symbol="X",
            ts_ms=10_000 + i * 10_000_000,  # 10 K seconds apart
            intent="exit_liquidity",
        )
        for i in range(12)
    ]

    async def fetch(symbol, since, until):  # noqa: ANN001
        # ``since == handle.ts_ms - pre_window_sec*1000`` (default 300_000ms).
        # ``until == handle.ts_ms + forward_window_sec*1000``.
        # Reconstruct the post timestamp and frame one bar on each side.
        # default pre_window=5min, forward=4h => post_ts = since + 300_000.
        post_ts = since + 300_000
        return [
            (post_ts - 60_000, 1.0, 1.0, 1.0),     # pre-bar (1 min before)
            (post_ts + 60_000, 0.95, 0.85, 0.86),  # post-bar (1 min after)
        ]

    obs_list = await build_observations_from_posts(
        posts=posts, fetch_klines=fetch,
    )
    assert len(obs_list) == 12

    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(min_samples=10),
    )
    for o in obs_list:
        analyzer.record_observation(
            author=o.author, symbol=o.symbol, intent=o.intent,
            ts_ms=o.ts_ms, realised_direction=o.realised_direction,
            magnitude_pct=o.magnitude_pct,
            follower_count=o.follower_count,
        )
    score = analyzer.lookup("goat", "exit_liquidity")
    assert score is not None
    assert score.total == 12
    assert score.hits == 12  # All 12 calls realised as dumps.
    # 13/14 ≈ 92.8% -> well above strong_bound, should LIFT confidence.
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.55, authors=["goat"],
    )
    assert out.delta > 0.0
    assert out.adjusted_confidence > 0.55


# --------------------------------------------------------------------- #
# Concurrency safety: lock prevents lost updates
# --------------------------------------------------------------------- #


def test_store_record_is_thread_safe(tmp_path: Path):
    """The KOLHistoryStore lock must protect against lost-update races
    when multiple workers record observations on the same author."""
    import threading

    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)

    def worker(n_obs: int) -> None:
        for i in range(n_obs):
            store.record(_make_obs(direction="pump", ts_ms=i))

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    score = store.get_score("goat", "frontrun_call")
    assert score is not None
    # Without the lock this would land somewhere below 200 due to lost
    # increments; with the lock it must be exactly 200.
    assert score.total == 200
    assert score.hits == 200


# --------------------------------------------------------------------- #
# Smoke test: fuser-style "exit_liquidity" workflow
# --------------------------------------------------------------------- #


def test_exit_liquidity_workflow_promotes_known_dumper(tmp_path: Path):
    """Simulates the fuser hook: an LLM verdict says ``exit_liquidity``
    with confidence=0.55 (below the 0.70 hard-veto threshold), so the
    fuser would soft-cap. With a known-bad-actor history, the analyzer
    lifts confidence past the 0.70 threshold -- so the fuser would
    instead hard-veto. This is the whole reason the analyzer exists.
    """
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(
            min_samples=10, strong_bound=0.65, conf_lift_max=0.20,
        ),
    )
    # 20-of-20 perfect track record on dumps -> Laplace ~ 21/22 = 0.955
    # well past strong_bound so we get full conf_lift_max.
    _seed_author(analyzer, author="dumper", intent="exit_liquidity",
                 hits=20, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.55,
        authors=["dumper"],
    )
    # Confidence pushed above the 0.70 hard-veto bar.
    assert out.adjusted_confidence >= 0.70


def test_exit_liquidity_workflow_demotes_unreliable_caller(tmp_path: Path):
    """Inverse: a 0.72 confidence verdict (would hard-veto) attributed
    to a known-unreliable caller drops below 0.70 so the fuser only
    soft-caps."""
    analyzer = HistoricalAnalyzer(
        store=KOLHistoryStore(path=tmp_path / "kol.json", autosave=False),
        config=HistoricalAnalyzerConfig(
            min_samples=10, weak_bound=0.40, conf_drop_max=0.20,
        ),
    )
    _seed_author(analyzer, author="permabear", intent="exit_liquidity",
                 hits=2, total=20)
    out = analyzer.adjust_kol_confidence(
        kol_intent="exit_liquidity", confidence=0.72,
        authors=["permabear"],
    )
    assert out.adjusted_confidence < 0.70


# --------------------------------------------------------------------- #
# Reset semantics
# --------------------------------------------------------------------- #


def test_analyzer_lookup_after_reset_returns_none(tmp_path: Path):
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    _seed_author(analyzer, author="x", intent="frontrun_call",
                 hits=5, total=10)
    assert analyzer.lookup("x", "frontrun_call") is not None
    store.reset()
    assert analyzer.lookup("x", "frontrun_call") is None


# --------------------------------------------------------------------- #
# Sanity: importing via the social package re-exports work
# --------------------------------------------------------------------- #


def test_public_imports_via_social_package():
    from altcoin_agent.social import (
        HistoricalAnalyzer as HA,
    )
    from altcoin_agent.social import (
        HistoricalAnalyzerConfig as HAC,
    )
    from altcoin_agent.social import (
        KOLAdjustment as KA,
    )
    from altcoin_agent.social import (
        KOLHistoryStore as KS,
    )
    from altcoin_agent.social import (
        KOLObservation as KO,
    )
    from altcoin_agent.social import (
        KOLScore as KSC,
    )
    from altcoin_agent.social import (
        build_observations_from_posts as build,
    )
    from altcoin_agent.social import (
        normalize_author as norm,
    )
    assert HA is HistoricalAnalyzer
    assert HAC is HistoricalAnalyzerConfig
    assert KA is KOLAdjustment
    assert KS is KOLHistoryStore
    assert KO is KOLObservation
    assert KSC is KOLScore
    assert build is build_observations_from_posts
    assert norm is normalize_author


# Reduce noise from asyncio: confirm asyncio.gather still works.
def test_asyncio_module_still_usable_after_import():
    assert asyncio.iscoroutinefunction(build_observations_from_posts)
