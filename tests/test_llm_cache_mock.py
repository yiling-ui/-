"""tests/test_llm_cache_mock.py — QUADRANT 六.2 measure 1 coverage."""

from __future__ import annotations

import json
import os

from altcoin_agent.llm.cache import LLMCache

# ----------------------- basic put / get ----------------------- #


def test_get_miss_returns_none_and_counts():
    c = LLMCache()
    assert c.get(LLMCache.make_key("X", "ramp", "h1")) is None
    assert c.misses == 1


def test_put_then_get_hit():
    c = LLMCache(default_ttl_sec=3600, now_fn=lambda: 100.0)
    key = c.put(symbol="X", phase="ramp", social_hash="h1",
                verdict={"score": 0.9})
    got = c.get(key)
    assert got == {"score": 0.9}
    assert c.hits == 1
    assert c.misses == 0
    assert c.hit_ratio() == 1.0


def test_get_returns_copy_not_reference():
    """Mutating the returned dict must not poison the cache."""
    c = LLMCache(now_fn=lambda: 100.0)
    key = c.put(symbol="X", phase="ramp", social_hash="h",
                verdict={"score": 0.9})
    got = c.get(key)
    got["score"] = -1
    again = c.get(key)
    assert again == {"score": 0.9}


# ----------------------- TTL ----------------------- #


def test_expired_entries_treated_as_miss():
    now = [100.0]
    c = LLMCache(default_ttl_sec=10, now_fn=lambda: now[0])
    key = c.put(symbol="X", phase="ramp", social_hash="h",
                verdict={"v": 1})
    now[0] = 200.0   # past expiry
    assert c.get(key) is None
    # Expired entry was evicted on access.
    assert len(c) == 0


def test_purge_expired_drops_old_entries():
    now = [100.0]
    c = LLMCache(default_ttl_sec=10, now_fn=lambda: now[0])
    c.put(symbol="A", phase="r", social_hash="h", verdict={"v": 1})
    c.put(symbol="B", phase="r", social_hash="h", verdict={"v": 2})
    now[0] = 1000.0
    purged = c.purge_expired()
    assert purged == 2
    assert len(c) == 0


def test_zero_or_negative_ttl_falls_back_to_default():
    c = LLMCache(default_ttl_sec=999, now_fn=lambda: 100.0)
    key = c.put(symbol="X", phase="r", social_hash="h",
                verdict={"v": 1}, ttl_sec=0)
    entry = c._store[key]  # noqa: SLF001
    assert entry.expires_at == 100.0 + 999


# ----------------------- LRU eviction ----------------------- #


def test_lru_evicts_least_recently_used():
    c = LLMCache(max_entries=2, now_fn=lambda: 100.0)
    k1 = c.put(symbol="A", phase="r", social_hash="h", verdict={"v": 1})
    k2 = c.put(symbol="B", phase="r", social_hash="h", verdict={"v": 2})
    # Touch k1 so k2 becomes the LRU.
    c.get(k1)
    k3 = c.put(symbol="C", phase="r", social_hash="h", verdict={"v": 3})
    assert c.get(k1) is not None
    assert c.get(k2) is None  # evicted
    assert c.get(k3) is not None
    assert c.evictions >= 1


def test_overwrite_existing_key_does_not_grow_store():
    c = LLMCache(max_entries=4, now_fn=lambda: 100.0)
    c.put(symbol="X", phase="r", social_hash="h", verdict={"v": 1})
    c.put(symbol="X", phase="r", social_hash="h", verdict={"v": 2})
    assert len(c) == 1
    got = c.get(LLMCache.make_key("X", "r", "h"))
    assert got == {"v": 2}


# ----------------------- key composition ----------------------- #


def test_make_key_composes_three_parts():
    k = LLMCache.make_key("PEPE/USDT:USDT", "parabolic", "abcdef")
    assert k == "PEPE/USDT:USDT|parabolic|abcdef"


def test_different_phase_gets_different_key():
    c = LLMCache(now_fn=lambda: 100.0)
    c.put(symbol="X", phase="ramp", social_hash="h", verdict={"v": 1})
    c.put(symbol="X", phase="parabolic", social_hash="h", verdict={"v": 2})
    assert c.get(LLMCache.make_key("X", "ramp", "h")) == {"v": 1}
    assert c.get(LLMCache.make_key("X", "parabolic", "h")) == {"v": 2}


# ----------------------- persistence ----------------------- #


def test_persist_roundtrip(tmp_path):
    path = str(tmp_path / "cache.json")
    c = LLMCache(state_path=path, now_fn=lambda: 100.0)
    c.put(symbol="X", phase="r", social_hash="h", verdict={"v": 1},
          tokens_estimate=500)
    c.put(symbol="Y", phase="r", social_hash="h", verdict={"v": 2})
    assert os.path.exists(path)

    c2 = LLMCache(state_path=path, now_fn=lambda: 100.0)
    assert len(c2) == 2
    got = c2.get(LLMCache.make_key("X", "r", "h"))
    assert got == {"v": 1}


def test_persist_skips_corrupted(tmp_path):
    path = str(tmp_path / "cache.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("garbage")
    c = LLMCache(state_path=path)
    assert len(c) == 0


def test_persisted_entries_with_bad_shape_skipped(tmp_path):
    path = str(tmp_path / "cache.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"entries": [{"key": "good|r|h", "verdict": {"v": 1},
                          "inserted_at": 1, "expires_at": 9e9},
                          "not-a-dict",
                          {"verdict": {}}]},  # missing key
            fh,
        )
    c = LLMCache(state_path=path)
    # Only the well-formed entry survives; expired ones (not the case
    # here because expires_at is far future) survive load.
    assert len(c) == 1
