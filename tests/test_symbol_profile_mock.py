"""tests/test_symbol_profile_mock.py — QUADRANT Phase 1.B coverage."""

from __future__ import annotations

import json
import os

import pytest

from altcoin_agent.risk.symbol_profile import (
    DEFAULT_QUADRANT_PARAMS,
    LIQUIDITY_THRESHOLD,
    SOCIAL_THRESHOLD,
    Quadrant,
    SymbolProfile,
    SymbolProfileStore,
    classify,
    quadrant_params,
)

# ----------------------- classification ----------------------- #


@pytest.mark.parametrize(
    "social,liq,expected",
    [
        (90, 90, Quadrant.A),
        (70, 70, Quadrant.A),                  # boundary inclusive
        (75, 60, Quadrant.B),
        (60, 75, Quadrant.C),
        (50, 50, Quadrant.D),
        (0, 0, Quadrant.D),
        (69.99, 100, Quadrant.C),
        (100, 69.99, Quadrant.B),
    ],
)
def test_classify_quadrant(social, liq, expected):
    assert classify(social, liq) is expected


def test_classify_clamps_negative_and_nan():
    assert classify(-10, -10) is Quadrant.D
    assert classify(float("nan"), 100) is Quadrant.C  # social treated as 0
    assert classify(100, float("nan")) is Quadrant.B


def test_thresholds_constants():
    # Plan-locked: do not move these without updating the plan + tests.
    assert SOCIAL_THRESHOLD == 70.0
    assert LIQUIDITY_THRESHOLD == 70.0


# ----------------------- quadrant matrix ----------------------- #


def test_default_quadrant_matrix_matches_plan():
    """Plan section 三 verbatim values. Locked."""
    a = DEFAULT_QUADRANT_PARAMS[Quadrant.A]
    assert a.max_risk_per_trade == 0.025
    assert a.max_leverage_long == 15.0
    assert a.rolling_max_legs == 4
    assert a.short_on_blowoff_top is True
    assert a.confidence_threshold == 0.80

    b = DEFAULT_QUADRANT_PARAMS[Quadrant.B]
    assert b.max_risk_per_trade == 0.015
    assert b.confidence_threshold == 0.85

    c = DEFAULT_QUADRANT_PARAMS[Quadrant.C]
    # C: 庄拉 — short on blowoff is BLOCKED per plan ("庄控盘风险")
    assert c.short_on_blowoff_top is False
    assert c.rolling_enabled is False

    d = DEFAULT_QUADRANT_PARAMS[Quadrant.D]
    assert d.max_risk_per_trade == 0.005
    assert d.confidence_threshold == 0.90
    assert d.rolling_max_legs == 0


def test_quadrant_params_helper():
    assert quadrant_params(Quadrant.A) is DEFAULT_QUADRANT_PARAMS[Quadrant.A]


# ----------------------- profile dataclass ----------------------- #


def test_from_scores_picks_quadrant_and_stamps_ts():
    p = SymbolProfile.from_scores("PEPE/USDT:USDT", 88.0, 80.0, now_ts=1_700_000_000)
    assert p.symbol == "PEPE/USDT:USDT"
    assert p.quadrant is Quadrant.A
    assert p.refreshed_at_ts == 1_700_000_000
    assert p.params() is DEFAULT_QUADRANT_PARAMS[Quadrant.A]


def test_effective_threshold_falls_back_to_quadrant():
    p = SymbolProfile.from_scores("X", 50, 50)
    assert p.effective_confidence_threshold() == 0.90  # D default
    p.confidence_threshold = 0.95
    assert p.effective_confidence_threshold() == 0.95


def test_profile_round_trip_dict():
    p = SymbolProfile(
        symbol="WIF/USDT:USDT",
        quadrant=Quadrant.A,
        social_score=85.0,
        liquidity_score=75.0,
        historical_win_rate=0.6,
        last_pump_ts=1_700_000_000,
        scam_score=0.1,
        confidence_threshold=0.82,
        samples=42,
        refreshed_at_ts=1_700_001_000,
        notes=["seeded"],
    )
    d = p.as_dict()
    p2 = SymbolProfile.from_dict(d)
    assert p2 == p


def test_profile_from_dict_handles_missing_optional_fields():
    p = SymbolProfile.from_dict({"symbol": "X", "quadrant": "D"})
    assert p.symbol == "X"
    assert p.quadrant is Quadrant.D
    assert p.confidence_threshold is None
    assert p.notes == []


# ----------------------- store persistence ----------------------- #


def test_store_persists_and_reloads(tmp_path):
    path = str(tmp_path / "symbol_profiles.json")
    store = SymbolProfileStore(path=path)
    store.upsert(SymbolProfile.from_scores("PEPE/USDT:USDT", 90, 90, now_ts=10))
    store.upsert(SymbolProfile.from_scores("DOGE/USDT:USDT", 60, 60, now_ts=20))
    store.save()
    assert os.path.exists(path)

    store2 = SymbolProfileStore(path=path)
    store2.load()
    assert sorted(store2.all_symbols()) == ["DOGE/USDT:USDT", "PEPE/USDT:USDT"]
    pepe = store2.get("PEPE/USDT:USDT")
    assert pepe is not None and pepe.quadrant is Quadrant.A
    assert len(store2) == 2


def test_store_handles_missing_file(tmp_path):
    path = str(tmp_path / "absent.json")
    store = SymbolProfileStore(path=path)
    store.load()
    assert len(store) == 0
    assert store.get("ANY") is None


def test_store_skips_malformed_entries(tmp_path):
    path = str(tmp_path / "broken.json")
    bad_payload = {
        "profiles": [
            {"symbol": "GOOD", "quadrant": "A"},
            {"quadrant": "A"},                       # missing symbol
            "not-a-dict",                            # type-error
            {"symbol": "BAD", "quadrant": "Z"},      # bad enum value
        ]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bad_payload, fh)
    store = SymbolProfileStore(path=path)
    store.load()
    assert store.all_symbols() == ["GOOD"]


def test_store_atomic_write_no_partial_file_on_crash(tmp_path, monkeypatch):
    """If json.dump raises mid-flush we must not overwrite the existing file."""
    path = str(tmp_path / "p.json")
    store = SymbolProfileStore(path=path)
    store.upsert(SymbolProfile.from_scores("A", 90, 90))
    store.save()
    original = open(path, "rb").read()

    # Inject a failure inside json.dump.
    real_dump = json.dump

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("altcoin_agent.risk.symbol_profile.json.dump", boom)
    store.upsert(SymbolProfile.from_scores("B", 90, 90))
    with pytest.raises(RuntimeError):
        store.save()

    # The original file is intact (atomic rename preserves it).
    assert open(path, "rb").read() == original
    monkeypatch.setattr("altcoin_agent.risk.symbol_profile.json.dump", real_dump)
