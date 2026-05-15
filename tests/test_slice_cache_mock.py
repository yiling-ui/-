"""Tests for the SQLite slice cache."""

from __future__ import annotations

from pathlib import Path

from altcoin_agent.learning_engine import Bar, HistoricalSlice
from altcoin_agent.slice_cache import SliceCache


def _slice(symbol: str = "RAVEUSDT", ts: int = 1_700_000_000_000) -> HistoricalSlice:
    return HistoricalSlice(
        symbol=symbol, target_ts_ms=ts,
        bars=[
            Bar(ts_ms=ts - 60_000, open=1.0, high=1.05, low=0.95,
                close=1.02, volume=100.0),
            Bar(ts_ms=ts, open=1.02, high=1.10, low=1.00,
                close=1.08, volume=200.0),
        ],
        funding_rates=[(ts - 3600_000, -0.001), (ts, -0.002)],
        open_interest=[(ts - 60_000, 1_000_000.0), (ts, 1_220_000.0)],
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    with SliceCache(db) as cache:
        s = _slice()
        cache.put(s)
        loaded = cache.get(s.symbol, s.target_ts_ms)
        assert loaded is not None
        assert loaded.symbol == s.symbol
        assert loaded.target_ts_ms == s.target_ts_ms
        assert len(loaded.bars) == 2
        assert loaded.bars[1].close == 1.08
        assert loaded.funding_rates[0] == (s.funding_rates[0][0], -0.001)


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    with SliceCache(tmp_path / "cache.sqlite") as cache:
        assert cache.get("UNKNOWN", 1) is None


def test_cache_replaces_on_repeated_put(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    with SliceCache(db) as cache:
        cache.put(_slice(symbol="X", ts=100))
        cache.put(_slice(symbol="X", ts=100))   # replace
        assert cache.stats()["total_slices"] == 1


def test_cache_segregates_by_symbol_and_ts(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    with SliceCache(db) as cache:
        cache.put(_slice(symbol="A", ts=100))
        cache.put(_slice(symbol="B", ts=100))
        cache.put(_slice(symbol="A", ts=200))
        assert cache.stats()["total_slices"] == 3
        assert cache.get("A", 100) is not None
        assert cache.get("A", 200) is not None
        assert cache.get("B", 100) is not None
        assert cache.get("B", 200) is None


def test_cache_persists_across_open_close(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    cache = SliceCache(db)
    cache.put(_slice())
    cache.close()
    cache2 = SliceCache(db)
    assert cache2.get("RAVEUSDT", 1_700_000_000_000) is not None
    cache2.close()
