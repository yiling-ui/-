"""funding_history_loader.py — Funding-rate + open-interest archivists.

Companion to :class:`HistoricalDataLoader`. The trainer (Phase 4) needs
both klines AND funding-rate / open-interest history to mine the
"funding extremes precede pump" rules QUADRANT_STRATEGY_PLAN section
五 calls out: spec for ``ramp`` features explicitly references
``funding_pre2h_extreme`` and ``oi_growth_pre1h``.

This module ships two loaders sharing the same disk + rate-limit shape
as ``HistoricalDataLoader`` so callers can hand both the same
``RateLimiter`` and same cache root.

Cache layout::

    .kiro/state/backtest_cache/{exchange}/{symbol_safe}/funding/{YYYY}/{MM}.json
    .kiro/state/backtest_cache/{exchange}/{symbol_safe}/openInterest/{YYYY}/{MM}.json

Each file is a JSON list of records:

    funding:        {"ts": ms, "rate": float, "info": {...?}}
    openInterest:   {"ts": ms, "oi": float, "info": {...?}}

Both loaders are mock-friendly: callers pass any object exposing
``fetch_funding_rate_history`` / ``fetch_open_interest_history``
(matching ccxt's contract). Tests inject a stub list-returning fake
without ever importing ccxt.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from altcoin_agent.backtest.historical_loader import (
    RateLimiter,
    _iter_months,
    _safe_symbol,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Fetcher protocols
# --------------------------------------------------------------------- #


class FundingFetcher(Protocol):
    """ccxt-shape: returns list of dicts with at least ``timestamp`` + ``fundingRate``."""

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]: ...


class OpenInterestFetcher(Protocol):
    """ccxt-shape: returns list of dicts with at least ``timestamp`` + ``openInterestAmount``."""

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]: ...


# --------------------------------------------------------------------- #
# Funding loader
# --------------------------------------------------------------------- #


@dataclass
class FundingHistoryLoader:
    """Pulls + caches funding-rate history per month.

    Binance Futures funding settles every 8h; one month is ~93 records,
    well below the 1000-record per-call cap. We still page in case an
    exchange returns smaller windows.
    """

    fetcher: FundingFetcher
    cache_root: str
    exchange_name: str = "binance"
    chunk_limit: int = 1000
    rate_limiter: RateLimiter | None = None

    # ---- public ---- #

    def download(
        self,
        *,
        symbol: str,
        start_ms: int,
        end_ms: int,
        force: bool = False,
    ) -> int:
        if start_ms >= end_ms:
            return 0
        records_written = 0
        for month_start, month_end in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, month_start)
            if not force and os.path.exists(cache_file):
                logger.debug("funding cache hit, skipping %s", cache_file)
                continue
            window = self._fetch_window(
                symbol=symbol, start_ms=month_start, end_ms=month_end,
            )
            if not window:
                continue
            self._write_cache(cache_file, window)
            records_written += len(window)
        return records_written

    def load(
        self,
        *,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        out: list[dict] = []
        for month_start, _ in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, month_start)
            if not os.path.exists(cache_file):
                continue
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read %s: %s", cache_file, exc)
                continue
            if not isinstance(payload, list):
                continue
            for r in payload:
                ts = r.get("ts")
                if not isinstance(ts, (int, float)):
                    continue
                ts_i = int(ts)
                if start_ms <= ts_i < end_ms:
                    out.append(r)
        out.sort(key=lambda r: int(r.get("ts", 0)))
        return out

    # ---- internals ---- #

    def _fetch_window(
        self, *, symbol: str, start_ms: int, end_ms: int,
    ) -> list[dict]:
        records: list[dict] = []
        cursor = start_ms
        last_seen_ts = -1
        # Hard cap on iterations so a misbehaving exchange that
        # returns the same window forever can't wedge the loader.
        # 1024 funding records covers > 1 year of monthly windows.
        for _ in range(1024):
            if cursor >= end_ms:
                break
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(1.0)
            try:
                chunk = self.fetcher.fetch_funding_rate_history(
                    symbol, since=cursor, limit=self.chunk_limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "funding fetch error symbol=%s since=%s: %s",
                    symbol, cursor, exc,
                )
                break
            if not chunk:
                break

            new_records: list[dict] = []
            chunk_max_ts = -1
            for raw in chunk:
                rec = _normalize_funding(raw)
                if rec is None:
                    continue
                ts = rec["ts"]
                chunk_max_ts = max(chunk_max_ts, ts)
                if start_ms <= ts < end_ms:
                    new_records.append(rec)
                    last_seen_ts = max(last_seen_ts, ts)
            if not new_records:
                break
            records.extend(new_records)

            # Advance past the LAST ts we saw in the chunk (not just
            # the last we kept) so a chunk whose tail spilled past
            # end_ms still moves the cursor forward instead of
            # +1ms-creeping.
            new_cursor = max(chunk_max_ts + 1, cursor + 1)
            if new_cursor <= cursor:
                break
            cursor = new_cursor
        # Dedupe by ts in case paging overlapped.
        dedup: dict[int, dict] = {}
        for r in records:
            dedup[int(r["ts"])] = r
        return [dedup[k] for k in sorted(dedup)]

    def _cache_path(self, symbol: str, month_start_ms: int) -> str:
        gm = time.gmtime(month_start_ms / 1000)
        return os.path.join(
            self.cache_root,
            self.exchange_name,
            _safe_symbol(symbol),
            "funding",
            f"{gm.tm_year:04d}",
            f"{gm.tm_mon:02d}.json",
        )

    @staticmethod
    def _write_cache(path: str, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".funding.", suffix=".json.tmp",
            dir=os.path.dirname(path),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# --------------------------------------------------------------------- #
# Open Interest loader
# --------------------------------------------------------------------- #


@dataclass
class OpenInterestHistoryLoader:
    """Pulls + caches open-interest history per month.

    Default timeframe is 5m: dense enough to feed the OI z-score and
    pre-pump features without overwhelming the cache.
    """

    fetcher: OpenInterestFetcher
    cache_root: str
    exchange_name: str = "binance"
    timeframe: str = "5m"
    chunk_limit: int = 500
    rate_limiter: RateLimiter | None = None

    def download(
        self, *, symbol: str, start_ms: int, end_ms: int, force: bool = False,
    ) -> int:
        if start_ms >= end_ms:
            return 0
        records_written = 0
        for month_start, month_end in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, month_start)
            if not force and os.path.exists(cache_file):
                continue
            window = self._fetch_window(
                symbol=symbol, start_ms=month_start, end_ms=month_end,
            )
            if not window:
                continue
            self._write_cache(cache_file, window)
            records_written += len(window)
        return records_written

    def load(
        self, *, symbol: str, start_ms: int, end_ms: int,
    ) -> list[dict]:
        out: list[dict] = []
        for month_start, _ in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, month_start)
            if not os.path.exists(cache_file):
                continue
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for r in payload:
                ts = r.get("ts")
                if not isinstance(ts, (int, float)):
                    continue
                ts_i = int(ts)
                if start_ms <= ts_i < end_ms:
                    out.append(r)
        out.sort(key=lambda r: int(r.get("ts", 0)))
        return out

    def _fetch_window(
        self, *, symbol: str, start_ms: int, end_ms: int,
    ) -> list[dict]:
        records: list[dict] = []
        cursor = start_ms
        last_seen_ts = -1
        for _ in range(1024):
            if cursor >= end_ms:
                break
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(1.0)
            try:
                chunk = self.fetcher.fetch_open_interest_history(
                    symbol, self.timeframe, since=cursor, limit=self.chunk_limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OI fetch error symbol=%s since=%s: %s",
                    symbol, cursor, exc,
                )
                break
            if not chunk:
                break
            new_records: list[dict] = []
            chunk_max_ts = -1
            for raw in chunk:
                rec = _normalize_oi(raw)
                if rec is None:
                    continue
                ts = rec["ts"]
                chunk_max_ts = max(chunk_max_ts, ts)
                if start_ms <= ts < end_ms:
                    new_records.append(rec)
                    last_seen_ts = max(last_seen_ts, ts)
            if not new_records:
                break
            records.extend(new_records)
            new_cursor = max(chunk_max_ts + 1, cursor + 1)
            if new_cursor <= cursor:
                break
            cursor = new_cursor
        dedup: dict[int, dict] = {}
        for r in records:
            dedup[int(r["ts"])] = r
        return [dedup[k] for k in sorted(dedup)]

    def _cache_path(self, symbol: str, month_start_ms: int) -> str:
        gm = time.gmtime(month_start_ms / 1000)
        return os.path.join(
            self.cache_root,
            self.exchange_name,
            _safe_symbol(symbol),
            "openInterest",
            f"{gm.tm_year:04d}",
            f"{gm.tm_mon:02d}.json",
        )

    @staticmethod
    def _write_cache(path: str, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".oi.", suffix=".json.tmp",
            dir=os.path.dirname(path),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# --------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------- #


def _normalize_funding(raw: object) -> dict | None:
    """Coerce a ccxt funding record (or our own JSON) into ``{"ts","rate"}``."""
    if not isinstance(raw, dict):
        return None
    ts = raw.get("timestamp") or raw.get("ts")
    rate = raw.get("fundingRate")
    if rate is None:
        rate = raw.get("rate")
    if ts is None or rate is None:
        return None
    try:
        return {"ts": int(ts), "rate": float(rate)}
    except (TypeError, ValueError):
        return None


def _normalize_oi(raw: object) -> dict | None:
    """Coerce a ccxt OI record into ``{"ts","oi"}``."""
    if not isinstance(raw, dict):
        return None
    ts = raw.get("timestamp") or raw.get("ts")
    oi = (
        raw.get("openInterestAmount")
        or raw.get("openInterestValue")
        or raw.get("oi")
    )
    if ts is None or oi is None:
        return None
    try:
        return {"ts": int(ts), "oi": float(oi)}
    except (TypeError, ValueError):
        return None


__all__ = [
    "FundingFetcher",
    "FundingHistoryLoader",
    "OpenInterestFetcher",
    "OpenInterestHistoryLoader",
]
