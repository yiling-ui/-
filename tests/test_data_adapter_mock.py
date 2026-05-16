"""tests/test_data_adapter_mock.py — Phase B.4 read-side adapter."""

from __future__ import annotations

from altcoin_agent.backtest.data_adapter import BacktestDataAdapter
from altcoin_agent.backtest.historical_loader import TIMEFRAME_MS, HistoricalDataLoader

# ----------------------- helpers ----------------------- #


class _StubFetcher:
    def __init__(self, bars):
        self.bars = sorted(bars, key=lambda b: b[0])

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return [list(b) for b in self.bars
                if b[0] >= since][:limit]


def _bars(start_ms: int, n: int) -> list[list[float]]:
    out = []
    step = TIMEFRAME_MS["1m"]
    for i in range(n):
        ts = start_ms + i * step
        p = 1.0 + 0.001 * i
        out.append([ts, p, p + 0.01, p - 0.01, p + 0.005, 100.0])
    return out


def _build_adapter(tmp_path, symbols_bars: dict[str, list[list[float]]]):
    loader = HistoricalDataLoader(
        fetcher=_StubFetcher(sum(symbols_bars.values(), [])),
        cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0,
        sleep_fn=lambda *_: None,
    )
    # Pre-cache each symbol's bars to disk (download path).
    for sym, bars in symbols_bars.items():
        loader.fetcher = _StubFetcher(bars)
        if not bars:
            continue
        loader.download(
            symbol=sym, timeframe="1m",
            start_ms=int(bars[0][0]),
            end_ms=int(bars[-1][0]) + TIMEFRAME_MS["1m"],
        )
    return BacktestDataAdapter(loader=loader, timeframe="1m")


# ----------------------- preload ----------------------- #


def test_preload_returns_total_bar_count(tmp_path):
    bars_a = _bars(0, 10)
    bars_b = _bars(0, 5)
    adapter = _build_adapter(tmp_path, {"A": bars_a, "B": bars_b})
    n = adapter.preload(
        symbols=["A", "B"], start_ms=0, end_ms=15 * TIMEFRAME_MS["1m"],
    )
    assert n == 15


def test_preload_handles_unknown_symbol_gracefully(tmp_path):
    bars = _bars(0, 5)
    adapter = _build_adapter(tmp_path, {"A": bars})
    n = adapter.preload(
        symbols=["A", "MISSING"], start_ms=0, end_ms=5 * TIMEFRAME_MS["1m"],
    )
    assert n == 5
    assert not adapter.has_data("MISSING")
    assert adapter.has_data("A")


def test_preload_rejects_unsupported_timeframe(tmp_path):
    adapter = _build_adapter(tmp_path, {})
    adapter.timeframe = "13s"
    import pytest
    with pytest.raises(ValueError):
        adapter.preload(symbols=["A"], start_ms=0, end_ms=1)


# ----------------------- iteration ----------------------- #


def test_iter_bars_yields_in_order_and_advances_cursor(tmp_path):
    bars = _bars(0, 5)
    adapter = _build_adapter(tmp_path, {"A": bars})
    adapter.preload(symbols=["A"], start_ms=0, end_ms=5 * TIMEFRAME_MS["1m"])
    seen = []
    for bar in adapter.iter_bars("A"):
        seen.append(bar)
        assert adapter.current_bar("A").ts_ms == bar.ts_ms
    assert [b.ts_ms for b in seen] == sorted(b.ts_ms for b in seen)
    assert len(seen) == 5


def test_iter_multi_does_kway_merge(tmp_path):
    # A: bars at t=0,2,4 ; B: at t=1,3,5  → merged should interleave.
    step = TIMEFRAME_MS["1m"]
    bars_a = [[0, 1, 1, 1, 1, 1], [2 * step, 1, 1, 1, 1, 1], [4 * step, 1, 1, 1, 1, 1]]
    bars_b = [[step, 1, 1, 1, 1, 1], [3 * step, 1, 1, 1, 1, 1], [5 * step, 1, 1, 1, 1, 1]]
    adapter = _build_adapter(tmp_path, {"A": bars_a, "B": bars_b})
    adapter.preload(symbols=["A", "B"], start_ms=0, end_ms=6 * step)
    seen = list(adapter.iter_multi(["A", "B"]))
    syms = [s for s, _ in seen]
    ts = [b.ts_ms for _, b in seen]
    assert ts == sorted(ts)
    assert syms == ["A", "B", "A", "B", "A", "B"]


def test_current_bar_none_before_iteration_starts(tmp_path):
    bars = _bars(0, 3)
    adapter = _build_adapter(tmp_path, {"A": bars})
    adapter.preload(symbols=["A"], start_ms=0, end_ms=3 * TIMEFRAME_MS["1m"])
    assert adapter.current_bar("A") is None


def test_bar_at_supports_o1_lookup(tmp_path):
    bars = _bars(0, 5)
    adapter = _build_adapter(tmp_path, {"A": bars})
    adapter.preload(symbols=["A"], start_ms=0, end_ms=5 * TIMEFRAME_MS["1m"])
    target_ts = int(bars[2][0])
    bar = adapter.bar_at("A", target_ts)
    assert bar is not None and bar.ts_ms == target_ts
    assert adapter.bar_at("A", -1) is None
