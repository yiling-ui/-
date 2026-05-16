"""data_adapter.py — Backtest-side replacement for ccxt (Phase B.4 / plan B.5.1).

Two responsibilities, kept in one module so the matching engine has
a single dependency:

1. **Bar reader.** Wraps ``HistoricalDataLoader`` and exposes a streaming
   iterator over a (symbol, timeframe, window) — same shape the
   ``BacktestRunner`` already consumes for phase tagging.

2. **Mark-price source.** During matching we need a way to look up
   "the OHLC bar the order would have hit" without re-walking the
   whole history. We index the loaded bars by ts in a dict so the
   matching engine can do an O(1) lookup per order.

This adapter does **not** implement the ``ExchangeAdapter`` Protocol —
that's the matching engine's job. We keep the read-side concerns
(historical data) separate from the write-side (orders + positions)
so each module stays small and the matching engine can be tested
without disk I/O.

Phase B.4 ships a 1-minute kline reader. Order-book depth and funding
plug into the same adapter shape later when the trainer needs them
(plan B.5.1 mentions parquet for funding; we keep the JSON-on-disk
layout for now and let parquet be a Phase 5 optimisation).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from altcoin_agent.backtest.historical_loader import (
    TIMEFRAME_MS,
    HistoricalDataLoader,
)
from altcoin_agent.risk.pump_phase import KlineBar

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------- #


@dataclass
class BacktestDataAdapter:
    """Read-side of the backtest IO seam.

    Construct with a ``HistoricalDataLoader`` and the symbols + window
    you intend to run. ``preload`` fills in-memory caches so the
    matching engine can look up a bar by ``(symbol, ts_ms)`` in O(1).
    """

    loader: HistoricalDataLoader
    timeframe: str = "1m"
    # symbol -> list of bars in ascending ts order
    _bars_by_symbol: dict[str, list[KlineBar]] = field(default_factory=dict)
    # symbol -> { ts_ms -> bar } for O(1) lookup
    _bar_index: dict[str, dict[int, KlineBar]] = field(default_factory=dict)
    # (symbol, ts_ms) for the *last* bar streamed — used by the matching
    # engine to know which bar an order placed "now" is filling against.
    _cursor_ts_by_symbol: dict[str, int] = field(default_factory=dict)

    # ---- preload ---- #

    def preload(
        self,
        *,
        symbols: list[str],
        start_ms: int,
        end_ms: int,
    ) -> int:
        """Load + index bars for ``symbols`` over ``[start_ms, end_ms)``.

        Returns the total number of bars cached. Symbols with zero bars
        on disk are silently skipped — caller is expected to have run
        ``HistoricalDataLoader.download`` first.
        """
        if start_ms >= end_ms:
            return 0
        if self.timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {self.timeframe!r}")

        total = 0
        for sym in symbols:
            raw_bars = self.loader.load(
                symbol=sym,
                timeframe=self.timeframe,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            if not raw_bars:
                logger.info("data_adapter: no bars cached for %s", sym)
                self._bars_by_symbol[sym] = []
                self._bar_index[sym] = {}
                continue
            bars = [_coerce(b) for b in raw_bars if _coerce(b) is not None]
            bars.sort(key=lambda b: b.ts_ms)
            self._bars_by_symbol[sym] = bars
            self._bar_index[sym] = {b.ts_ms: b for b in bars}
            total += len(bars)
        return total

    # ---- streaming ---- #

    def iter_bars(self, symbol: str) -> Iterator[KlineBar]:
        """Yield bars in time order for ``symbol``.

        Each yield advances the per-symbol cursor so a matching engine
        coupled to this adapter via ``current_bar(symbol)`` can resolve
        an order to "the bar I just emitted".
        """
        bars = self._bars_by_symbol.get(symbol, [])
        for bar in bars:
            self._cursor_ts_by_symbol[symbol] = bar.ts_ms
            yield bar

    def iter_multi(
        self, symbols: list[str]
    ) -> Iterator[tuple[str, KlineBar]]:
        """Merge multiple symbols into one chronological stream.

        Implementation is a k-way merge: at each step we emit the bar
        with the smallest ts across all symbols. Useful for portfolio
        backtests where a single signal can fan out to several open
        positions; the matching engine then sees events in the same
        order the live daemon would.
        """
        # Per-symbol indexes into the sorted list.
        idx = {s: 0 for s in symbols}
        while True:
            best_sym: str | None = None
            best_ts: int | None = None
            for s in symbols:
                bars = self._bars_by_symbol.get(s, [])
                i = idx[s]
                if i >= len(bars):
                    continue
                ts = bars[i].ts_ms
                if best_ts is None or ts < best_ts:
                    best_ts = ts
                    best_sym = s
            if best_sym is None:
                return
            bar = self._bars_by_symbol[best_sym][idx[best_sym]]
            idx[best_sym] += 1
            self._cursor_ts_by_symbol[best_sym] = bar.ts_ms
            yield best_sym, bar

    # ---- lookups ---- #

    def current_bar(self, symbol: str) -> KlineBar | None:
        """Last bar yielded for ``symbol`` (None if iteration hasn't started)."""
        ts = self._cursor_ts_by_symbol.get(symbol)
        if ts is None:
            return None
        return self._bar_index.get(symbol, {}).get(ts)

    def bar_at(self, symbol: str, ts_ms: int) -> KlineBar | None:
        """O(1) lookup of the bar with exact ``ts_ms``."""
        return self._bar_index.get(symbol, {}).get(ts_ms)

    def bars(self, symbol: str) -> list[KlineBar]:
        return list(self._bars_by_symbol.get(symbol, []))

    def has_data(self, symbol: str) -> bool:
        return bool(self._bars_by_symbol.get(symbol))

    def symbols(self) -> list[str]:
        return sorted(self._bars_by_symbol)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _coerce(raw: object) -> KlineBar | None:
    """Translate ccxt-shape ``[ts, o, h, l, c, v]`` to a ``KlineBar``."""
    if isinstance(raw, KlineBar):
        return raw
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        return None
    try:
        return KlineBar(
            ts_ms=int(raw[0]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            vol_z_score=0.0,
        )
    except (TypeError, ValueError):
        return None


__all__ = ["BacktestDataAdapter"]
