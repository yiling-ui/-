"""atr.py — Average True Range calculator for the trailing FSM.

ATR(n) over a rolling window of bars. True Range for bar i:
    TR_i = max(high - low, |high - prev_close|, |low - prev_close|)

We use simple moving average over the last ``period`` bars, which matches
the Wilder smoothing within an order of magnitude for our 14-bar default
and is much simpler / cheaper to update online.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from altcoin_agent.screener import Kline


@dataclass
class ATRCalculator:
    """One ATR per (exchange, symbol, timeframe). Online, O(1) per bar."""

    period: int = 14
    _bars: dict[str, deque[Kline]] = field(default_factory=dict)
    _trs: dict[str, deque[float]] = field(default_factory=dict)

    def update(self, exchange: str, symbol: str, bar: Kline) -> float:
        key = f"{exchange}:{symbol}:{bar.timeframe}"
        history = self._bars.setdefault(key, deque(maxlen=self.period + 1))
        trs = self._trs.setdefault(key, deque(maxlen=self.period))
        if history:
            prev_close = history[-1].close
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
        else:
            tr = bar.high - bar.low
        history.append(bar)
        trs.append(tr)
        if len(trs) < max(2, self.period // 2):
            # Not enough history yet — use a wide ATR equal to the latest TR
            # so the trailing FSM at least has something non-zero to work with.
            return tr
        return sum(trs) / len(trs)

    def get(self, exchange: str, symbol: str, timeframe: str = "1m") -> float:
        key = f"{exchange}:{symbol}:{timeframe}"
        trs = self._trs.get(key)
        if not trs:
            return 0.0
        return sum(trs) / len(trs)
