"""slice_cache.py — SQLite cache for OKX historical slices.

Repeated backtests over the same (symbol, target_ts) combinations are slow
and rude to OKX's rate limits. This module memoizes ``HistoricalSlice``
objects in a small local SQLite database so subsequent runs over the same
event list complete in milliseconds.

The cache key is ``(symbol, target_ts_ms, hours_back)``. Cache values
serialize the slice as JSON; we tolerate schema drift by checking a
version field on read.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from altcoin_agent.learning_engine import Bar, HistoricalSlice

logger = logging.getLogger(__name__)


_SCHEMA_VERSION = 1


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS slices (
    symbol         TEXT NOT NULL,
    target_ts_ms   INTEGER NOT NULL,
    hours_back     INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    payload_json   TEXT NOT NULL,
    saved_at_ms    INTEGER NOT NULL,
    PRIMARY KEY (symbol, target_ts_ms, hours_back)
);
"""


class SliceCache:
    """Tiny SQLite-backed cache. Thread-safe enough for our use (single
    writer per process, no cross-thread reuse)."""

    def __init__(self, path: str | Path = ".kiro/cache/slices.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SliceCache:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------- API ---------------- #

    def get(
        self, symbol: str, target_ts_ms: int, hours_back: int = 4,
    ) -> HistoricalSlice | None:
        cur = self._conn.execute(
            "SELECT payload_json, schema_version FROM slices "
            "WHERE symbol=? AND target_ts_ms=? AND hours_back=?",
            (symbol, target_ts_ms, hours_back),
        )
        row = cur.fetchone()
        if not row:
            return None
        payload_json, schema_version = row
        if int(schema_version) != _SCHEMA_VERSION:
            logger.info("Cache miss: schema mismatch for %s @ %d", symbol, target_ts_ms)
            return None
        try:
            return _deserialize(payload_json)
        except Exception as e:
            logger.warning("Cache deserialize error for %s @ %d: %s",
                           symbol, target_ts_ms, e)
            return None

    def put(self, slice_: HistoricalSlice, *, hours_back: int = 4) -> None:
        payload_json = _serialize(slice_)
        import time
        self._conn.execute(
            "INSERT OR REPLACE INTO slices "
            "(symbol, target_ts_ms, hours_back, schema_version, payload_json, saved_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (slice_.symbol, slice_.target_ts_ms, hours_back, _SCHEMA_VERSION,
             payload_json, int(time.time() * 1000)),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT COUNT(*) FROM slices")
        return {"total_slices": int(cur.fetchone()[0])}


def _serialize(s: HistoricalSlice) -> str:
    return json.dumps({
        "symbol": s.symbol,
        "target_ts_ms": s.target_ts_ms,
        "bars": [asdict(b) for b in s.bars],
        "funding_rates": s.funding_rates,
        "open_interest": s.open_interest,
    })


def _deserialize(payload_json: str) -> HistoricalSlice:
    data = json.loads(payload_json)
    return HistoricalSlice(
        symbol=data["symbol"],
        target_ts_ms=int(data["target_ts_ms"]),
        bars=[Bar(**b) for b in data.get("bars", [])],
        funding_rates=[(int(t), float(r)) for t, r in data.get("funding_rates", [])],
        open_interest=[(int(t), float(v)) for t, v in data.get("open_interest", [])],
    )
