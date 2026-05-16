"""tests/test_historical_loader_mock.py — QUADRANT Phase 2 coverage."""

from __future__ import annotations

import calendar
import os

from altcoin_agent.backtest.historical_loader import (
    DEFAULT_LIMIT,
    TIMEFRAME_MS,
    HistoricalDataLoader,
)

# ----------------------- helpers ----------------------- #


def _ts(year: int, month: int, day: int = 1) -> int:
    return calendar.timegm((year, month, day, 0, 0, 0, 0, 0, 0)) * 1000


def _make_bars(start_ms: int, n: int, step_ms: int = TIMEFRAME_MS["1m"]) -> list[list[float]]:
    """Synthesize a deterministic kline sequence."""
    out = []
    for i in range(n):
        ts = start_ms + i * step_ms
        price = 1.0 + 0.001 * i
        out.append([ts, price, price + 0.01, price - 0.01, price + 0.005, 100.0])
    return out


class _StubFetcher:
    """In-memory fetch_ohlcv that pages through a fixed bar list."""

    def __init__(self, bars: list[list[float]]):
        self.bars = sorted(bars, key=lambda b: b[0])
        self.calls: list[tuple[str, str, int, int]] = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append((symbol, timeframe, since, limit))
        out: list[list[float]] = []
        for bar in self.bars:
            if bar[0] >= since and len(out) < limit:
                out.append(list(bar))
        return out


# ----------------------- download + cache ----------------------- #


def test_download_writes_per_month_files(tmp_path):
    start = _ts(2026, 5, 1)
    bars = _make_bars(start, 90)  # 90 minutes — all in May
    fetcher = _StubFetcher(bars)

    loader = HistoricalDataLoader(
        fetcher=fetcher,
        cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0,
        sleep_fn=lambda *_: None,
    )
    written = loader.download(
        symbol="PEPE/USDT:USDT",
        timeframe="1m",
        start_ms=start,
        end_ms=start + 90 * TIMEFRAME_MS["1m"],
    )
    assert written == 90

    # File on disk under YYYY/MM.json.
    expected = os.path.join(
        str(tmp_path), "binance", "PEPE_USDT_USDT", "1m", "2026", "05.json",
    )
    assert os.path.exists(expected)


def test_download_skips_existing_month_unless_force(tmp_path):
    start = _ts(2026, 5, 1)
    bars = _make_bars(start, 60)
    fetcher = _StubFetcher(bars)
    loader = HistoricalDataLoader(
        fetcher=fetcher, cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0, sleep_fn=lambda *_: None,
    )
    loader.download(symbol="X", timeframe="1m",
                    start_ms=start, end_ms=start + 60 * TIMEFRAME_MS["1m"])
    first_calls = len(fetcher.calls)

    # Re-run -> all months cached, fetch should NOT be called.
    loader.download(symbol="X", timeframe="1m",
                    start_ms=start, end_ms=start + 60 * TIMEFRAME_MS["1m"])
    assert len(fetcher.calls) == first_calls

    # force=True -> fetch again.
    loader.download(symbol="X", timeframe="1m",
                    start_ms=start, end_ms=start + 60 * TIMEFRAME_MS["1m"],
                    force=True)
    assert len(fetcher.calls) > first_calls


def test_download_spans_month_boundary_into_two_files(tmp_path):
    # 2 days of bars spanning May 31 -> June 1.
    may_31 = _ts(2026, 5, 31)
    bars = _make_bars(may_31, 60 * 24 * 2)
    fetcher = _StubFetcher(bars)
    loader = HistoricalDataLoader(
        fetcher=fetcher, cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0, sleep_fn=lambda *_: None,
    )
    loader.download(symbol="X", timeframe="1m",
                    start_ms=may_31,
                    end_ms=may_31 + 2 * 24 * 60 * TIMEFRAME_MS["1m"])
    may_path = os.path.join(str(tmp_path), "binance", "X", "1m", "2026", "05.json")
    jun_path = os.path.join(str(tmp_path), "binance", "X", "1m", "2026", "06.json")
    assert os.path.exists(may_path)
    assert os.path.exists(jun_path)


# ----------------------- load + stream ----------------------- #


def test_load_returns_only_bars_in_window(tmp_path):
    start = _ts(2026, 5, 1)
    bars = _make_bars(start, 30)
    fetcher = _StubFetcher(bars)
    loader = HistoricalDataLoader(
        fetcher=fetcher, cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0, sleep_fn=lambda *_: None,
    )
    loader.download(symbol="X", timeframe="1m",
                    start_ms=start, end_ms=start + 30 * TIMEFRAME_MS["1m"])

    win_start = start + 5 * TIMEFRAME_MS["1m"]
    win_end = start + 10 * TIMEFRAME_MS["1m"]
    out = loader.load(symbol="X", timeframe="1m",
                      start_ms=win_start, end_ms=win_end)
    assert len(out) == 5
    assert all(win_start <= b[0] < win_end for b in out)
    # Output is ascending by ts.
    assert [b[0] for b in out] == sorted(b[0] for b in out)


def test_stream_yields_bars_lazily(tmp_path):
    start = _ts(2026, 5, 1)
    bars = _make_bars(start, 5)
    fetcher = _StubFetcher(bars)
    loader = HistoricalDataLoader(
        fetcher=fetcher, cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0, sleep_fn=lambda *_: None,
    )
    loader.download(symbol="X", timeframe="1m",
                    start_ms=start, end_ms=start + 5 * TIMEFRAME_MS["1m"])
    seen = list(loader.stream(
        symbol="X", timeframe="1m",
        start_ms=start, end_ms=start + 5 * TIMEFRAME_MS["1m"],
    ))
    assert len(seen) == 5
    assert seen[0][0] == start


# ----------------------- robustness ----------------------- #


def test_unsupported_timeframe_raises(tmp_path):
    loader = HistoricalDataLoader(
        fetcher=_StubFetcher([]), cache_root=str(tmp_path),
    )
    import pytest
    with pytest.raises(ValueError):
        loader.download(symbol="X", timeframe="13m",
                        start_ms=0, end_ms=1)


def test_empty_window_returns_zero(tmp_path):
    loader = HistoricalDataLoader(
        fetcher=_StubFetcher([]), cache_root=str(tmp_path),
    )
    assert loader.download(symbol="X", timeframe="1m",
                            start_ms=100, end_ms=100) == 0


def test_fetch_error_does_not_raise_aborts_window(tmp_path):
    class BoomFetcher:
        def fetch_ohlcv(self, *a, **k):
            raise RuntimeError("rate-limited")

    loader = HistoricalDataLoader(
        fetcher=BoomFetcher(), cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0, sleep_fn=lambda *_: None,
    )
    written = loader.download(symbol="X", timeframe="1m",
                               start_ms=_ts(2026, 5, 1),
                               end_ms=_ts(2026, 5, 2))
    assert written == 0


def test_default_limit_constant():
    # Phase 2 plan calls for chunked fetches; ccxt's default cap is 1500
    # but Binance's is 1000. We default to 1000 for portability.
    assert DEFAULT_LIMIT == 1000
