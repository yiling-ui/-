"""Mock-only tests for the fetch_history CLI + RateLimiter.

We don't import ccxt; all the CLI's network-touching paths are
``--dry-run`` or stubbed via a fake fetcher passed straight into the
HistoricalDataLoader. The RateLimiter is exercised purely via injected
clocks.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from altcoin_agent.backtest.historical_loader import (
    HistoricalDataLoader,
    RateLimiter,
)
from scripts import fetch_history


# --------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------- #


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.t += secs


def test_rate_limiter_initial_burst_is_free():
    """Capacity=5 means the first 5 calls cost 0 sleep."""
    clock = FakeClock()
    rl = RateLimiter(
        max_rate=10.0, capacity=5.0,
        time_fn=clock.time, sleep_fn=clock.sleep,
    )
    for _ in range(5):
        slept = rl.acquire(1.0)
        assert slept == 0.0
    # bucket is empty now -> next call should wait.
    slept = rl.acquire(1.0)
    assert slept > 0.0
    # rate=10/s, deficit=1 -> wait ~0.1s.
    assert clock.sleeps[-1] == pytest.approx(0.1, rel=1e-6)


def test_rate_limiter_refills_with_elapsed_time():
    clock = FakeClock()
    rl = RateLimiter(
        max_rate=10.0, capacity=5.0,
        time_fn=clock.time, sleep_fn=clock.sleep,
    )
    # Drain bucket.
    for _ in range(5):
        rl.acquire(1.0)
    # Advance the clock by 1 second -> 10 tokens earned, capped to 5.
    clock.t += 1.0
    slept = rl.acquire(1.0)
    assert slept == 0.0


def test_rate_limiter_zero_max_rate_disables_throttle():
    """max_rate <= 0 means "no throttle" — never sleeps."""
    clock = FakeClock()
    rl = RateLimiter(
        max_rate=0.0, capacity=1.0,
        time_fn=clock.time, sleep_fn=clock.sleep,
    )
    for _ in range(100):
        slept = rl.acquire(1.0)
        assert slept == 0.0
    assert clock.sleeps == []


def test_rate_limiter_clamp_oversized_acquire():
    """Asking for tokens > capacity must not loop forever."""
    clock = FakeClock()
    rl = RateLimiter(
        max_rate=100.0, capacity=2.0,
        time_fn=clock.time, sleep_fn=clock.sleep,
    )
    # Ask for 100 (way over capacity) — should clamp to 2 and succeed.
    rl.acquire(100.0)
    # And only one sleep happened (or zero if the initial bucket
    # already covered the clamped request).
    assert len(clock.sleeps) <= 1


# --------------------------------------------------------------------- #
# Loader integration with limiter
# --------------------------------------------------------------------- #


class FakeFetcher:
    """Returns a deterministic page of bars and counts calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append((symbol, timeframe, since, limit))
        # One bar per minute starting at ``since`` for ``limit`` bars,
        # but cap at 5 to make the loop terminate quickly.
        n = min(5, limit)
        out = []
        for i in range(n):
            ts = since + i * 60_000
            out.append([ts, 1.0, 1.1, 0.9, 1.05, 100.0])
        return out


def test_loader_invokes_rate_limiter(tmp_path):
    clock = FakeClock()
    rl = RateLimiter(
        max_rate=1000.0, capacity=2.0,
        time_fn=clock.time, sleep_fn=clock.sleep,
    )
    loader = HistoricalDataLoader(
        fetcher=FakeFetcher(),
        cache_root=str(tmp_path),
        exchange_name="binance",
        inter_call_sleep_sec=0.0,
        rate_limiter=rl,
    )
    # 1 day = 1440 minutes of 1m bars; with chunk_limit=5 the loader
    # makes ~288 calls — well past the bucket capacity, so the limiter
    # MUST kick in.
    start = 1_700_000_000_000  # Nov 2023, far enough back to be a stable month
    end = start + 86_400_000
    loader.download(
        symbol="PEPE/USDT:USDT", timeframe="1m",
        start_ms=start, end_ms=end,
    )
    # At least one sleep was recorded.
    assert len(clock.sleeps) > 0


def test_loader_resumable_skips_existing_months(tmp_path):
    fetcher = FakeFetcher()
    loader = HistoricalDataLoader(
        fetcher=fetcher,
        cache_root=str(tmp_path),
        exchange_name="binance",
        inter_call_sleep_sec=0.0,
    )
    # Use a 1-day window inside a single calendar month.
    start = 1_700_000_000_000
    end = start + 86_400_000
    loader.download(
        symbol="PEPE/USDT:USDT", timeframe="1m",
        start_ms=start, end_ms=end,
    )
    first_calls = len(fetcher.calls)
    assert first_calls > 0

    # Re-run: should skip the cached month entirely (zero new calls).
    loader.download(
        symbol="PEPE/USDT:USDT", timeframe="1m",
        start_ms=start, end_ms=end,
    )
    assert len(fetcher.calls) == first_calls

    # Force re-download.
    loader.download(
        symbol="PEPE/USDT:USDT", timeframe="1m",
        start_ms=start, end_ms=end, force=True,
    )
    assert len(fetcher.calls) > first_calls


# --------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------- #


def test_cli_parses_symbols_and_days():
    args = fetch_history._parse_args([
        "--symbols", "PEPE/USDT:USDT,WIF/USDT:USDT",
        "--timeframe", "5m",
        "--days", "7",
    ])
    assert args.symbols == ["PEPE/USDT:USDT", "WIF/USDT:USDT"]
    assert args.timeframe == "5m"
    assert (args.end_ms - args.start_ms) == 7 * 86_400_000


def test_cli_dry_run_exits_clean(monkeypatch, capsys):
    """--dry-run must not import ccxt or touch disk."""
    # If the dry-run code accidentally calls _build_ccxt_fetcher,
    # the next line makes the test fail loudly.
    monkeypatch.setattr(
        fetch_history, "_build_ccxt_fetcher",
        lambda *a, **kw: pytest.fail("dry-run must not build ccxt"),
    )
    rc = fetch_history.run([
        "--symbols", "PEPE/USDT:USDT",
        "--timeframe", "1m",
        "--days", "1",
        "--dry-run",
    ])
    assert rc == 0


def test_cli_rejects_inverted_window():
    with pytest.raises(SystemExit):
        fetch_history._parse_args([
            "--symbols", "PEPE/USDT:USDT",
            "--start", "2025-01-01T00:00:00Z",
            "--end", "2024-01-01T00:00:00Z",
        ])


def test_cli_rejects_empty_symbols():
    with pytest.raises(SystemExit):
        fetch_history._parse_args([
            "--symbols", " , , ",
            "--days", "1",
        ])


def test_iso_to_ms_handles_z_suffix():
    ms = fetch_history._iso_to_ms("2024-01-01T00:00:00Z")
    # 2024-01-01 UTC = 1704067200000 ms
    assert ms == 1_704_067_200_000
