"""cache.py — LLM verdict cache (QUADRANT 六.2 measure 1).

The single highest-leverage token saver: most signals on the same symbol
in the same pump phase get a near-identical LLM verdict within a 12-hour
window. Caching the verdict keyed by ``(symbol, phase, social_hash)``
typically saves 60-80% of all calls per the plan.

This module provides the in-memory + on-disk cache; ai_engine wires it
in front of the provider call.

Design:

* Keys are explicit and human-readable: ``"<symbol>|<phase>|<social_hash>"``
* TTL is per-entry; the default 12h matches the plan section 六.2 row 1.
* In-memory LRU bounded by ``max_entries`` to avoid unbounded growth.
* On-disk JSON for cross-restart persistence; load is best-effort.
* Stored verdicts are opaque dicts — we don't bind to ai_engine's
  ``AIVerdict`` shape so this module stays decoupled.
* No eviction by phase/quadrant in v1; trainer can rebuild from logs if
  it wants tighter control.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------- #


@dataclass
class LLMCacheEntry:
    """One cached verdict + its provenance metadata."""

    key: str
    verdict: dict[str, Any]
    inserted_at: float
    expires_at: float
    hits: int = 0
    tokens_saved_estimate: int = 0
    # Provenance — useful for the dashboard but not used in lookup.
    symbol: str = ""
    phase: str = ""
    social_hash: str = ""

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "verdict": self.verdict,
            "inserted_at": self.inserted_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
            "tokens_saved_estimate": self.tokens_saved_estimate,
            "symbol": self.symbol,
            "phase": self.phase,
            "social_hash": self.social_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMCacheEntry:
        return cls(
            key=str(d["key"]),
            verdict=dict(d.get("verdict") or {}),
            inserted_at=float(d.get("inserted_at", 0.0)),
            expires_at=float(d.get("expires_at", 0.0)),
            hits=int(d.get("hits", 0)),
            tokens_saved_estimate=int(d.get("tokens_saved_estimate", 0)),
            symbol=str(d.get("symbol") or ""),
            phase=str(d.get("phase") or ""),
            social_hash=str(d.get("social_hash") or ""),
        )


# --------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------- #


@dataclass
class LLMCache:
    """LRU + TTL cache for LLM verdicts.

    Hit/miss counters are exposed so the dashboard / Prometheus can
    publish ``llm_cache_hit_ratio`` (already a tracked quantity in the
    plan's Phase B.4 metric set).
    """

    max_entries: int = 1024
    default_ttl_sec: int = 12 * 3600
    state_path: str | None = None
    now_fn: Callable[[], float] = field(default=time.time, repr=False)
    _store: OrderedDict[str, LLMCacheEntry] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def __post_init__(self) -> None:
        if self.state_path and os.path.exists(self.state_path):
            self._load()

    # ---- key ---- #

    @staticmethod
    def make_key(symbol: str, phase: str, social_hash: str) -> str:
        return f"{symbol}|{phase}|{social_hash}"

    # ---- access ---- #

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached verdict (a copy) or None on miss/expiry."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        now = self.now_fn()
        if entry.is_expired(now):
            self._store.pop(key, None)
            self.misses += 1
            return None
        # LRU bump.
        self._store.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return dict(entry.verdict)

    def put(
        self,
        *,
        symbol: str,
        phase: str,
        social_hash: str,
        verdict: dict[str, Any],
        ttl_sec: int | None = None,
        tokens_estimate: int = 0,
    ) -> str:
        """Insert / overwrite an entry. Returns the cache key."""
        ttl = int(ttl_sec) if ttl_sec is not None else self.default_ttl_sec
        if ttl <= 0:
            ttl = self.default_ttl_sec
        now = self.now_fn()
        key = self.make_key(symbol, phase, social_hash)
        entry = LLMCacheEntry(
            key=key,
            verdict=dict(verdict),
            inserted_at=now,
            expires_at=now + ttl,
            tokens_saved_estimate=int(tokens_estimate),
            symbol=symbol,
            phase=phase,
            social_hash=social_hash,
        )
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        self._enforce_capacity()
        if self.state_path:
            self._save()
        return key

    # ---- maintenance ---- #

    def purge_expired(self) -> int:
        """Drop expired entries. Returns count purged."""
        now = self.now_fn()
        before = len(self._store)
        self._store = OrderedDict(
            (k, v) for k, v in self._store.items() if not v.is_expired(now)
        )
        purged = before - len(self._store)
        if purged and self.state_path:
            self._save()
        return purged

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def _enforce_capacity(self) -> None:
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._store)

    # ---- persistence ---- #

    def _save(self) -> None:
        assert self.state_path
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "saved_at_ts": int(self.now_fn()),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": [e.as_dict() for e in self._store.values()],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".llm_cache.", suffix=".json.tmp",
            dir=os.path.dirname(self.state_path) or ".",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2,
                          sort_keys=True)
            os.replace(tmp_path, self.state_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        assert self.state_path
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "LLMCache: failed to load %s (%s); starting empty",
                self.state_path, exc,
            )
            return
        if not isinstance(raw, dict):
            return
        self.hits = int(raw.get("hits", 0))
        self.misses = int(raw.get("misses", 0))
        self.evictions = int(raw.get("evictions", 0))
        for entry_dict in (raw.get("entries") or []):
            if not isinstance(entry_dict, dict):
                continue
            try:
                e = LLMCacheEntry.from_dict(entry_dict)
            except (KeyError, ValueError, TypeError):
                continue
            self._store[e.key] = e
        # Trim if a previous build had a larger capacity.
        self._enforce_capacity()


__all__ = ["LLMCache", "LLMCacheEntry"]
