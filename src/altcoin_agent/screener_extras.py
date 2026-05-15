"""screener_extras.py — Optional trade-stream aggregator.

ccxt.pro's ``watch_ohlcv`` does NOT include ``trade_count``. To make SR-3
(wash trading) work on the live path we instead consume ``watch_trades``
and aggregate trades into our own ``Kline`` objects, which DO carry
``trade_count``. The aggregator emits a closed bar each time a 1-minute
boundary is crossed.

This module is OPTIONAL: ``main.py`` only uses it when configured to. The
core test suite continues to run without it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from altcoin_agent.screener import Kline

logger = logging.getLogger(__name__)


_BAR_MS = {"1m": 60_000, "5m": 300_000}


@dataclass
class _Accum:
    bucket_ms: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    trade_count: int = 0


class TradeFlowAggregator:
    """Aggregates ccxt trade dicts into closed 1m / 5m Klines.

    Each call to ``ingest_trade`` may return one or more closed Klines if
    the trade's timestamp crossed a bar boundary.
    """

    def __init__(self, timeframes: tuple[str, ...] = ("1m", "5m")):
        self.timeframes = timeframes
        self._state: dict[tuple[str, str, str], _Accum] = {}

    def _bucket(self, ts_ms: int, tf: str) -> int:
        size = _BAR_MS.get(tf, 60_000)
        return (ts_ms // size) * size

    def ingest_trade(
        self,
        exchange: str,
        symbol: str,
        ts_ms: int,
        price: float,
        amount: float,
    ) -> list[tuple[str, Kline]]:
        """Returns a list of (timeframe, closed_bar) pairs to emit."""
        emitted: list[tuple[str, Kline]] = []
        for tf in self.timeframes:
            key = (exchange, symbol, tf)
            bucket = self._bucket(ts_ms, tf)
            acc = self._state.get(key)
            if acc is None:
                acc = _Accum(bucket_ms=bucket, open=price, high=price,
                             low=price, close=price, volume=amount, trade_count=1)
                self._state[key] = acc
                continue
            if bucket != acc.bucket_ms:
                # Close the old bucket.
                emitted.append((tf, Kline(
                    ts=acc.bucket_ms, open=acc.open, high=acc.high,
                    low=acc.low, close=acc.close, volume=acc.volume,
                    timeframe=tf, trade_count=acc.trade_count,
                )))
                self._state[key] = _Accum(
                    bucket_ms=bucket, open=price, high=price, low=price,
                    close=price, volume=amount, trade_count=1,
                )
                continue
            # Same bucket — update.
            acc.high = max(acc.high, price)
            acc.low = min(acc.low, price)
            acc.close = price
            acc.volume += amount
            acc.trade_count += 1
        return emitted


KlineSink = Callable[[str, Kline], Awaitable[None]]


async def stream_trades_into_aggregator(
    *,
    client,                                             # noqa: ANN001 — ccxt.pro client
    exchange_name: str,
    symbol: str,
    aggregator: TradeFlowAggregator,
    on_kline: KlineSink,
    stop_event: asyncio.Event,
) -> None:
    """Drive the aggregator from a ccxt.pro ``watch_trades`` loop until stop."""
    backoff = 1.0
    while not stop_event.is_set():
        try:
            trades = await client.watch_trades(symbol)
            if not trades:
                continue
            for t in trades:
                ts = int(t.get("timestamp") or 0)
                price = float(t.get("price") or 0.0)
                amount = float(t.get("amount") or 0.0)
                if ts == 0 or price <= 0 or amount <= 0:
                    continue
                for tf, bar in aggregator.ingest_trade(
                    exchange_name, symbol, ts, price, amount,
                ):
                    await on_kline(tf, bar)
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("watch_trades error %s/%s: %s",
                           exchange_name, symbol, e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
