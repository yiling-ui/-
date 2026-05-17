"""historical_loader.py — Bulk OHLCV downloader + on-disk cache.

Pulls historical 1m / 5m / 1d klines for a list of symbols from any
ccxt exchange and caches them to ``.kiro/state/backtest_cache/``.

Design constraints:

* **Mock-friendly.** A ``KlineFetcher`` protocol abstracts the ccxt
  call so unit tests pass in a list of fake bars instead of hitting
  Binance. The ccxt path is exercised only by an opt-in CLI.
* **Resumable.** Already-cached month files are skipped, so killing
  the process mid-download and restarting picks up where it left off.
* **Rate-limit aware.** ccxt's ``enableRateLimit=True`` plus a small
  per-call sleep keeps us well under Binance's 1200 req/min.
* **Safe writes.** Each chunk lands in a ``.tmp`` file then renames into
  place — a crash mid-write never leaves a half-baked cache file.

The cache layout is::

    .kiro/state/backtest_cache/{exchange}/{symbol_safe}/{timeframe}/{YYYY}/{MM}.json

Each file is a JSON list of ``[ts_ms, open, high, low, close, volume]``
arrays — same shape ccxt returns natively, so loading + reuse is zero
transformation.

Phase 1 ships the loader + fetcher protocol + JSON store. Parquet is
mentioned in the plan but adds a pyarrow dependency; we keep JSON for
v1 since the data volumes are modest and the read path uses streaming.
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

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Rate limiter — token bucket
# --------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    """Token-bucket rate limiter for ccxt downloads.

    Binance Futures REST publishes a 1200-weight-per-minute window;
    ``fetch_ohlcv`` weighs 1, so 20 req/s leaves a comfortable margin.
    Rather than hard-code the venue limit we expose ``max_rate`` and
    ``capacity`` so the CLI can tune them per exchange.

    Implementation is the canonical token bucket: each ``acquire``
    refills tokens proportional to elapsed time, blocks (via the
    injected ``sleep_fn``) until at least one token is available, then
    decrements. The ``time_fn`` / ``sleep_fn`` injection points keep
    the limiter trivially testable without monkey-patching the time
    module.
    """

    max_rate: float = 10.0          # tokens per second
    capacity: float = 20.0          # max tokens stored (burst budget)
    _tokens: float = field(default=0.0)
    _last_refill_ts: float = field(default=0.0)
    time_fn: Any = field(default=time.time, repr=False)
    sleep_fn: Any = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        # Start full so the first ``capacity`` calls aren't throttled.
        self._tokens = float(self.capacity)
        self._last_refill_ts = float(self.time_fn())

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available, then consume them.

        Returns the seconds slept (0.0 if no wait was needed). Useful
        for tests + observability metrics.
        """
        if tokens <= 0:
            return 0.0
        if tokens > self.capacity:
            # Asking for more than the bucket can hold; clamp to
            # ``capacity`` so we don't sleep forever on a misuse.
            tokens = self.capacity

        # Tiny epsilon to absorb fp drift on long-running clocks.
        # Without it, callers that drive the limiter via a high-magnitude
        # wall clock (``time.time()`` ~ 1.7e9) accumulate ~1e-7 of
        # arithmetic error per refill, which can leave ``_tokens`` at
        # 0.99999... forever and spin the loop. 1e-9 is well below any
        # rate the limiter is meant to enforce in practice.
        EPS = 1e-9
        slept_total = 0.0
        for _ in range(64):  # bounded so a misconfigured clock can't wedge us
            self._refill()
            if self._tokens + EPS >= tokens:
                self._tokens = max(0.0, self._tokens - tokens)
                return slept_total
            # Sleep just long enough to earn the missing tokens.
            deficit = tokens - self._tokens
            wait = deficit / self.max_rate if self.max_rate > 0 else 0.0
            if wait <= 0:
                # max_rate <= 0 -> limiter disabled; pretend tokens
                # are infinite. This is the "no throttle" knob.
                self._tokens = max(self._tokens, tokens)
                continue
            self.sleep_fn(wait)
            slept_total += wait
        # Fallback: 64 iterations couldn't satisfy the request -- almost
        # certainly a misconfigured ``time_fn``/``sleep_fn`` pair.
        # Consume whatever's there and return; better to under-throttle
        # than to spin forever.
        self._tokens = max(0.0, self._tokens - tokens)
        return slept_total

    def _refill(self) -> None:
        now = float(self.time_fn())
        elapsed = max(0.0, now - self._last_refill_ts)
        self._last_refill_ts = now
        if self.max_rate <= 0:
            return
        self._tokens = min(
            self.capacity, self._tokens + elapsed * self.max_rate,
        )


# Timeframe -> ms per bar. Used to chunk fetch_ohlcv (Binance returns
# at most 1500 bars per call).
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

DEFAULT_LIMIT = 1000


# --------------------------------------------------------------------- #
# Fetcher protocol — abstracts ccxt for testing
# --------------------------------------------------------------------- #


class KlineFetcher(Protocol):
    """Return a list of [ts_ms, o, h, l, c, v] arrays for one window.

    Implementations must obey the ``since`` / ``limit`` semantics of
    ccxt's ``fetch_ohlcv``:

        * ``since`` is the inclusive lower bound, ms.
        * Up to ``limit`` bars are returned, chronologically ascending.
        * Fewer than ``limit`` is allowed (end of available history).
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]: ...


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #


@dataclass
class HistoricalDataLoader:
    """Downloads + caches OHLCV.

    Construct with an injected ``fetcher`` (ccxt exchange instance with
    a ``fetch_ohlcv`` method) and a ``cache_root`` directory. ``download``
    is the high-level entry point; lower-level methods are exposed for
    callers that need finer control.
    """

    fetcher: KlineFetcher
    cache_root: str
    exchange_name: str = "binance"
    inter_call_sleep_sec: float = 0.25
    chunk_limit: int = DEFAULT_LIMIT
    sleep_fn: Any = field(default=time.sleep, repr=False)
    # Optional token-bucket limiter. When set, ``acquire(1)`` runs
    # before every fetch_ohlcv call; this is the canonical knob the
    # ``fetch_history`` CLI uses to stay under exchange rate caps
    # without hand-tuning ``inter_call_sleep_sec``.
    rate_limiter: RateLimiter | None = None

    # ---- public API ---- #

    def download(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        force: bool = False,
    ) -> int:
        """Download ``[start_ms, end_ms)`` and write per-month JSON files.

        Returns the total number of bars written. Skips months whose
        cache file already exists unless ``force=True``.
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        if start_ms >= end_ms:
            return 0

        bars_written = 0
        for month_start, month_end in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, timeframe, month_start)
            if not force and os.path.exists(cache_file):
                logger.debug("cache hit, skipping: %s", cache_file)
                continue

            month_bars = self._fetch_window(
                symbol=symbol,
                timeframe=timeframe,
                start_ms=month_start,
                end_ms=month_end,
            )
            if not month_bars:
                continue
            self._write_cache(cache_file, month_bars)
            bars_written += len(month_bars)
        return bars_written

    def load(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[float]]:
        """Read cached bars in ``[start_ms, end_ms)``, sorted ascending.

        Missing months are silently skipped — callers should run
        ``download`` first if they need a complete window.
        """
        out: list[list[float]] = []
        for month_start, _ in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, timeframe, month_start)
            if not os.path.exists(cache_file):
                continue
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    bars = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read %s: %s", cache_file, exc)
                continue
            if not isinstance(bars, list):
                continue
            for bar in bars:
                if not isinstance(bar, list) or len(bar) < 6:
                    continue
                ts = int(bar[0])
                if start_ms <= ts < end_ms:
                    out.append(bar)
        out.sort(key=lambda b: b[0])
        return out

    def stream(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> Iterator[list[float]]:
        """Yield cached bars one at a time without loading the full list."""
        for month_start, _ in _iter_months(start_ms, end_ms):
            cache_file = self._cache_path(symbol, timeframe, month_start)
            if not os.path.exists(cache_file):
                continue
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    bars = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(bars, list):
                continue
            for bar in sorted(bars, key=lambda b: b[0] if b else 0):
                if not isinstance(bar, list) or len(bar) < 6:
                    continue
                ts = int(bar[0])
                if start_ms <= ts < end_ms:
                    yield bar

    # ---- internals ---- #

    def _fetch_window(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[float]]:
        """Page through ccxt fetch_ohlcv until end_ms is covered.

        Stops if a fetch returns no progress (end of history) so we
        don't loop forever on a delisted symbol.
        """
        step_ms = TIMEFRAME_MS[timeframe]
        bars: list[list[float]] = []
        cursor = start_ms
        while cursor < end_ms:
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(1.0)
            try:
                chunk = self.fetcher.fetch_ohlcv(
                    symbol, timeframe, cursor, self.chunk_limit
                )
            except Exception as exc:  # noqa: BLE001 — ccxt wraps many types
                logger.warning(
                    "fetch_ohlcv error symbol=%s tf=%s since=%s: %s",
                    symbol, timeframe, cursor, exc,
                )
                break
            if not chunk:
                break
            # Trim to window.
            for bar in chunk:
                if not bar:
                    continue
                ts = int(bar[0])
                if start_ms <= ts < end_ms:
                    bars.append(list(bar))
            last_ts = int(chunk[-1][0])
            next_cursor = last_ts + step_ms
            if next_cursor <= cursor:
                # No forward progress -> bail to avoid infinite loop.
                break
            cursor = next_cursor
            if self.inter_call_sleep_sec > 0:
                self.sleep_fn(self.inter_call_sleep_sec)
        return bars

    def _cache_path(
        self, symbol: str, timeframe: str, month_start_ms: int
    ) -> str:
        gm = time.gmtime(month_start_ms / 1000)
        symbol_safe = _safe_symbol(symbol)
        return os.path.join(
            self.cache_root,
            self.exchange_name,
            symbol_safe,
            timeframe,
            f"{gm.tm_year:04d}",
            f"{gm.tm_mon:02d}.json",
        )

    @staticmethod
    def _write_cache(path: str, bars: list[list[float]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".bars.", suffix=".json.tmp",
            dir=os.path.dirname(path),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(bars, fh)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _safe_symbol(symbol: str) -> str:
    """Translate a ccxt symbol like 'PEPE/USDT:USDT' into a path-safe slug."""
    return (
        symbol.replace("/", "_")
              .replace(":", "_")
              .replace("\\", "_")
    )


def _iter_months(start_ms: int, end_ms: int) -> Iterator[tuple[int, int]]:
    """Yield ``(month_start_ms, month_end_ms)`` in UTC covering [start, end).

    Months are emitted in ascending order. ``month_start_ms`` is always
    the first millisecond of that month; ``month_end_ms`` is the first
    ms of the *next* month, so the half-open interval is canonical.
    """
    if start_ms >= end_ms:
        return
    cur = _month_floor_ms(start_ms)
    while cur < end_ms:
        nxt = _month_ceil_ms(cur)
        yield cur, nxt
        if nxt <= cur:
            # Defensive: malformed month math could loop. Bail.
            return
        cur = nxt


def _month_floor_ms(ts_ms: int) -> int:
    gm = time.gmtime(ts_ms / 1000)
    return int(_utc_to_ms(gm.tm_year, gm.tm_mon, 1))


def _month_ceil_ms(ts_ms: int) -> int:
    gm = time.gmtime(ts_ms / 1000)
    y, m = gm.tm_year, gm.tm_mon
    if m == 12:
        return int(_utc_to_ms(y + 1, 1, 1))
    return int(_utc_to_ms(y, m + 1, 1))


def _utc_to_ms(year: int, month: int, day: int) -> int:
    """Convert a (Y, M, D) midnight-UTC date to ms since epoch."""
    # ``calendar.timegm`` mirrors gmtime so the round-trip is exact.
    import calendar
    return calendar.timegm((year, month, day, 0, 0, 0, 0, 0, 0)) * 1000


__all__ = [
    "DEFAULT_LIMIT",
    "HistoricalDataLoader",
    "KlineFetcher",
    "RateLimiter",
    "TIMEFRAME_MS",
]
