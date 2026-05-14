"""
screener.py — Task A: Market & On-Chain Screener.

Pure-detector design:
    - All detectors are *IO-free* classes that consume snapshots and emit
      SignalEvent objects. They are unit-testable in milliseconds, no network.
    - Network/asyncio code lives in `Screener`, which uses ccxt.pro to fan in
      WebSocket streams from Binance/OKX/Gate.io and dispatches to detectors.

Detectors implemented:
    1. VolumeSpikeDetector       — z-score on rolling volume
    2. FundingAnomalyDetector    — short-window deviation + extreme threshold
    3. OISurgeDetector           — % change with optional "silent build" tag
    4. LiquidityPoolAnalyzer     — SMC liquidity pool tracking + sweep detection

The detectors are written so they can run on either live ws ticks or replayed
historical fixtures (Task E backtest), with no code change.

NOTE: Per requirements.md FR-A1, ccxt.pro is the chosen client. We import it
lazily inside Screener so unit tests don't need ccxt installed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Domain types
# --------------------------------------------------------------------------- #


class SignalKind(str, Enum):
    VOLUME_SPIKE = "volume_spike"
    FUNDING_EXTREME = "funding_extreme"
    FUNDING_DEVIATION = "funding_deviation"
    OI_SURGE = "oi_surge"
    OI_SILENT_BUILD = "oi_silent_build"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    LIQUIDITY_POOL_FORMED = "liquidity_pool_formed"


@dataclass(frozen=True)
class Kline:
    """OHLCV bar from a single timeframe."""

    ts: int  # ms since epoch, bar OPEN time
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bull(self) -> bool:
        return self.close >= self.open


@dataclass(frozen=True)
class FundingSnapshot:
    ts: int
    symbol: str
    rate: float  # per funding interval (typically 8h on Binance)
    next_funding_ts: int | None = None


@dataclass(frozen=True)
class OISnapshot:
    ts: int
    symbol: str
    open_interest: float  # in contract base units (or USDT, by exchange convention)
    price: float


@dataclass
class SignalEvent:
    """Uniform output of every detector. Sent to the bus."""

    kind: SignalKind
    symbol: str
    ts: int
    exchange: str
    payload: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "symbol": self.symbol,
            "ts": self.ts,
            "exchange": self.exchange,
            "payload": self.payload,
            "trace_id": self.trace_id,
        }


# --------------------------------------------------------------------------- #
# Volume Spike Detector
# --------------------------------------------------------------------------- #


class VolumeSpikeDetector:
    """
    Rolling z-score over the last N kline volumes.

    Emits VOLUME_SPIKE when z >= k_sigma AND price moved in the same direction
    as the dominant volume (bullish bar -> buy spike, bearish bar -> sell spike).
    """

    def __init__(self, window: int = 60, k_sigma: float = 4.0, min_samples: int = 20):
        if window < min_samples:
            raise ValueError("window must be >= min_samples")
        self.window = window
        self.k_sigma = k_sigma
        self.min_samples = min_samples
        self._buf: dict[str, deque[Kline]] = {}
        self._fired_bar: dict[str, int] = {}  # last fired ts (dedupe per bar)

    def feed(self, exchange: str, symbol: str, bar: Kline) -> SignalEvent | None:
        key = f"{exchange}:{symbol}:{bar.timeframe}"
        buf = self._buf.setdefault(key, deque(maxlen=self.window))
        buf.append(bar)

        if len(buf) < self.min_samples:
            return None

        vols = [b.volume for b in buf]
        mean = sum(vols) / len(vols)
        var = sum((v - mean) ** 2 for v in vols) / len(vols)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return None

        z = (bar.volume - mean) / std
        if z < self.k_sigma:
            return None

        # Dedupe: each bar fires at most once.
        if self._fired_bar.get(key) == bar.ts:
            return None
        self._fired_bar[key] = bar.ts

        side = "buy" if bar.is_bull else "sell"
        return SignalEvent(
            kind=SignalKind.VOLUME_SPIKE,
            symbol=symbol,
            ts=bar.ts,
            exchange=exchange,
            payload={
                "zscore": round(z, 3),
                "vol_ratio": round(bar.volume / mean, 3),
                "side": side,
                "timeframe": bar.timeframe,
                "window": self.window,
            },
        )


# --------------------------------------------------------------------------- #
# Funding Rate Anomaly Detector
# --------------------------------------------------------------------------- #


class FundingAnomalyDetector:
    """
    Detects two anomaly types:

      - FUNDING_EXTREME: absolute level breaches threshold (e.g. <= -0.1%/8h)
        for `consecutive` consecutive samples.

      - FUNDING_DEVIATION: short-window deviation from a longer baseline.
        Useful for catching sudden moves before the absolute threshold is hit.
        Emits when the latest rate deviates from the rolling mean by
        `deviation_sigma` standard deviations (computed on the longer baseline).
    """

    def __init__(
        self,
        extreme_low: float = -0.001,   # -0.1% / 8h
        extreme_high: float = 0.0015,  # +0.15% / 8h
        consecutive: int = 2,
        short_window: int = 3,
        long_window: int = 24,
        deviation_sigma: float = 3.0,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.extreme_low = extreme_low
        self.extreme_high = extreme_high
        self.consecutive = consecutive
        self.short_window = short_window
        self.long_window = long_window
        self.deviation_sigma = deviation_sigma

        self._hist: dict[str, deque[FundingSnapshot]] = {}
        self._streak: dict[str, int] = {}

    def feed(self, exchange: str, snap: FundingSnapshot) -> list[SignalEvent]:
        key = f"{exchange}:{snap.symbol}"
        hist = self._hist.setdefault(key, deque(maxlen=self.long_window))
        hist.append(snap)

        events: list[SignalEvent] = []

        # 1) Extreme absolute level (with consecutive-sample debounce)
        is_extreme = snap.rate <= self.extreme_low or snap.rate >= self.extreme_high
        if is_extreme:
            self._streak[key] = self._streak.get(key, 0) + 1
        else:
            self._streak[key] = 0

        if self._streak[key] == self.consecutive:
            direction = "short_squeeze" if snap.rate <= self.extreme_low else "long_fragile"
            events.append(
                SignalEvent(
                    kind=SignalKind.FUNDING_EXTREME,
                    symbol=snap.symbol,
                    ts=snap.ts,
                    exchange=exchange,
                    payload={
                        "rate": snap.rate,
                        "direction": direction,
                        "consecutive": self.consecutive,
                    },
                )
            )

        # 2) Short-window deviation vs long-window baseline
        if len(hist) >= self.long_window:
            recent = list(hist)[-self.short_window:]
            short_avg = sum(s.rate for s in recent) / len(recent)
            baseline = list(hist)[: -self.short_window]
            mean = sum(s.rate for s in baseline) / len(baseline)
            var = sum((s.rate - mean) ** 2 for s in baseline) / len(baseline)
            std = math.sqrt(var) if var > 0 else 0.0
            if std > 0:
                z = (short_avg - mean) / std
                if abs(z) >= self.deviation_sigma:
                    events.append(
                        SignalEvent(
                            kind=SignalKind.FUNDING_DEVIATION,
                            symbol=snap.symbol,
                            ts=snap.ts,
                            exchange=exchange,
                            payload={
                                "short_avg": short_avg,
                                "baseline_mean": mean,
                                "zscore": round(z, 3),
                                "short_window": self.short_window,
                                "long_window": self.long_window,
                            },
                        )
                    )

        return events


# --------------------------------------------------------------------------- #
# OI Surge Detector
# --------------------------------------------------------------------------- #


class OISurgeDetector:
    """
    Detects fast OI growth.

    Inputs: 1m-resolution OISnapshot (or any uniform interval).

    OI_SURGE         — OI grew >= surge_pct within `window` samples.
    OI_SILENT_BUILD  — OI surged AND price moved <= silent_max_price_move.
                       This is the textbook smart-money pre-pump pattern.
    """

    def __init__(
        self,
        window: int = 5,                 # samples (~5 min if 1m cadence)
        surge_pct: float = 0.15,         # +15%
        silent_max_price_move: float = 0.01,  # 1%
    ):
        self.window = window
        self.surge_pct = surge_pct
        self.silent_max_price_move = silent_max_price_move
        self._buf: dict[str, deque[OISnapshot]] = {}
        self._fired_ts: dict[str, int] = {}

    def feed(self, exchange: str, snap: OISnapshot) -> SignalEvent | None:
        key = f"{exchange}:{snap.symbol}"
        buf = self._buf.setdefault(key, deque(maxlen=self.window + 1))
        buf.append(snap)

        if len(buf) < self.window + 1:
            return None

        oldest = buf[0]
        if oldest.open_interest <= 0:
            return None

        oi_delta_pct = (snap.open_interest - oldest.open_interest) / oldest.open_interest
        if oi_delta_pct < self.surge_pct:
            return None

        if self._fired_ts.get(key) == snap.ts:
            return None
        self._fired_ts[key] = snap.ts

        price_move = abs(snap.price - oldest.price) / oldest.price if oldest.price > 0 else 0.0
        kind = (
            SignalKind.OI_SILENT_BUILD
            if price_move <= self.silent_max_price_move
            else SignalKind.OI_SURGE
        )

        return SignalEvent(
            kind=kind,
            symbol=snap.symbol,
            ts=snap.ts,
            exchange=exchange,
            payload={
                "oi_delta_pct": round(oi_delta_pct, 4),
                "price_move_pct": round(price_move, 4),
                "from_oi": oldest.open_interest,
                "to_oi": snap.open_interest,
                "from_price": oldest.price,
                "to_price": snap.price,
                "window_samples": self.window,
            },
        )


# --------------------------------------------------------------------------- #
# SMC: Liquidity Pool Analyzer
# --------------------------------------------------------------------------- #


@dataclass
class LiquidityPool:
    """
    A liquidity pool == a cluster of equal/near-equal swing highs (sell-side
    pool, where stops above are resting) or swing lows (buy-side pool, where
    stops below are resting). Smart money sweeps these to grab liquidity.
    """

    symbol: str
    side: str           # "sell_side" (highs) | "buy_side" (lows)
    level: float
    timeframe: str
    formed_ts: int
    touch_count: int
    last_touch_ts: int
    swept: bool = False
    swept_ts: int | None = None
    sweep_wick_to_body: float | None = None


class LiquidityPoolAnalyzer:
    """
    Identifies liquidity pools and detects when they get swept (SMC concept).

    Pool formation rules:
        - Two or more swing highs within `cluster_pct` of each other form a
          sell-side liquidity pool at their average level.
        - Two or more swing lows within `cluster_pct` of each other form a
          buy-side pool likewise.

    Sweep detection:
        - A bar's wick pierces the pool level by >= `sweep_pierce_bps` bps
          and the bar closes back through the level (failed breakout).
        - Wick / body ratio >= `wick_to_body_min` to filter noise.
        - Once swept, the pool is marked as such and not retriggered.

    Swing-point detection uses the classic 2-side fractal:
        a bar is a swing high if its high > the highs of the `fractal`
        bars immediately before and after.
    """

    def __init__(
        self,
        fractal: int = 2,
        cluster_pct: float = 0.0015,   # 15 bps band considered "equal"
        max_pools: int = 8,            # keep recent N pools per side per symbol
        sweep_pierce_bps: float = 5.0, # pierce level by 5 bps to count as sweep
        wick_to_body_min: float = 1.5,
        min_history: int = 0,          # min bars before any analysis (defaults to 2*fractal+1)
    ):
        self.fractal = fractal
        self.cluster_pct = cluster_pct
        self.max_pools = max_pools
        self.sweep_pierce_bps = sweep_pierce_bps
        self.wick_to_body_min = wick_to_body_min
        self.min_history = max(min_history, 2 * fractal + 1)

        self._bars: dict[str, deque[Kline]] = {}
        self._pools: dict[str, list[LiquidityPool]] = {}

    # --------------------------- public API --------------------------- #

    def feed(self, exchange: str, symbol: str, bar: Kline) -> list[SignalEvent]:
        key = f"{exchange}:{symbol}:{bar.timeframe}"
        bars = self._bars.setdefault(key, deque(maxlen=200))
        bars.append(bar)
        pools = self._pools.setdefault(key, [])

        events: list[SignalEvent] = []

        if len(bars) < self.min_history:
            return events

        # 1) check sweeps on existing pools first (using the latest bar)
        for pool in pools:
            if pool.swept or pool.symbol != symbol or pool.timeframe != bar.timeframe:
                continue
            sweep_ev = self._maybe_sweep(exchange, pool, bar)
            if sweep_ev is not None:
                events.append(sweep_ev)

        # 2) try to detect a NEW swing point at index -(fractal+1)
        new_pivot = self._detect_swing_point(bars)
        if new_pivot is not None:
            side, pivot_bar = new_pivot
            new_event = self._integrate_pivot(exchange, symbol, bar.timeframe, pools, side, pivot_bar)
            if new_event is not None:
                events.append(new_event)

        # cap pool list
        if len(pools) > self.max_pools * 2:
            pools.sort(key=lambda p: p.last_touch_ts)
            del pools[: len(pools) - self.max_pools * 2]

        return events

    def pools_for(self, exchange: str, symbol: str, timeframe: str = "1m") -> list[LiquidityPool]:
        key = f"{exchange}:{symbol}:{timeframe}"
        return list(self._pools.get(key, []))

    # --------------------------- internals --------------------------- #

    def _detect_swing_point(self, bars: deque[Kline]) -> tuple[str, Kline] | None:
        """
        Look at the bar at offset -(fractal+1). If its high is the strict max
        of the surrounding 2*fractal+1 window, it's a swing high; symmetric
        for swing low.
        """
        n = len(bars)
        if n < 2 * self.fractal + 1:
            return None
        idx = n - self.fractal - 1
        candidate = bars[idx]
        window: Iterable[Kline] = (bars[i] for i in range(idx - self.fractal, idx + self.fractal + 1) if i != idx)
        window_list = list(window)
        if all(candidate.high > b.high for b in window_list):
            return "sell_side", candidate
        if all(candidate.low < b.low for b in window_list):
            return "buy_side", candidate
        return None

    def _integrate_pivot(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        pools: list[LiquidityPool],
        side: str,
        pivot: Kline,
    ) -> SignalEvent | None:
        level = pivot.high if side == "sell_side" else pivot.low

        # try to attach to an existing pool of the same side within cluster band
        for p in pools:
            if p.side != side or p.swept:
                continue
            if abs(p.level - level) / max(p.level, 1e-12) <= self.cluster_pct:
                # cluster: average the level, bump touch count
                p.level = (p.level * p.touch_count + level) / (p.touch_count + 1)
                p.touch_count += 1
                p.last_touch_ts = pivot.ts
                if p.touch_count == 2:
                    # pool is now "confirmed" — emit once
                    return SignalEvent(
                        kind=SignalKind.LIQUIDITY_POOL_FORMED,
                        symbol=symbol,
                        ts=pivot.ts,
                        exchange=exchange,
                        payload={
                            "side": side,
                            "level": p.level,
                            "timeframe": timeframe,
                            "touch_count": p.touch_count,
                        },
                    )
                return None

        # no match — create a new (single-touch) candidate pool
        pools.append(
            LiquidityPool(
                symbol=symbol,
                side=side,
                level=level,
                timeframe=timeframe,
                formed_ts=pivot.ts,
                touch_count=1,
                last_touch_ts=pivot.ts,
            )
        )
        return None

    def _maybe_sweep(self, exchange: str, pool: LiquidityPool, bar: Kline) -> SignalEvent | None:
        # SMC strict: a single swing high/low is not a "liquidity pool".
        # Only equal highs / equal lows (>=2 touches) constitute resting
        # liquidity worth sweeping. This also prevents the pool's own second
        # pivot bar from being misclassified as the sweep of itself.
        if pool.touch_count < 2:
            return None
        pierce = pool.level * self.sweep_pierce_bps / 10_000.0
        if pool.side == "sell_side":
            # need wick above level, but close back below
            if bar.high < pool.level + pierce:
                return None
            if bar.close >= pool.level:
                return None
            if bar.upper_wick < self.wick_to_body_min * max(bar.body, 1e-12):
                return None
        else:
            if bar.low > pool.level - pierce:
                return None
            if bar.close <= pool.level:
                return None
            if bar.lower_wick < self.wick_to_body_min * max(bar.body, 1e-12):
                return None

        wick = bar.upper_wick if pool.side == "sell_side" else bar.lower_wick
        ratio = wick / max(bar.body, 1e-12)

        pool.swept = True
        pool.swept_ts = bar.ts
        pool.sweep_wick_to_body = ratio

        return SignalEvent(
            kind=SignalKind.LIQUIDITY_SWEEP,
            symbol=pool.symbol,
            ts=bar.ts,
            exchange=exchange,
            payload={
                "side": pool.side,
                "level": pool.level,
                "wick_to_body": round(ratio, 3),
                "bar_close": bar.close,
                "timeframe": pool.timeframe,
                "pool_age_ms": bar.ts - pool.formed_ts,
                "touch_count_before_sweep": pool.touch_count,
            },
        )


# --------------------------------------------------------------------------- #
# Screener — async orchestration
# --------------------------------------------------------------------------- #


SignalSink = Callable[[SignalEvent], Awaitable[None]]


class Screener:
    """
    Async fan-in for ccxt.pro WebSocket streams across multiple exchanges and
    symbols. Routes each incoming snapshot to the appropriate detector(s).

    The class is intentionally thin: detectors do the heavy lifting and stay
    pure, so all behavior is testable without ccxt or a network connection
    (see `tests/test_screener_mock.py`).

    Usage:
        async def sink(event: SignalEvent) -> None:
            print(event)

        sc = Screener(exchanges=["binance"], symbols=["RAVEUSDT"], sink=sink)
        await sc.run()
    """

    def __init__(
        self,
        exchanges: list[str],
        symbols: list[str],
        sink: SignalSink,
        timeframes: tuple[str, ...] = ("1m", "5m"),
        volume_detector: VolumeSpikeDetector | None = None,
        funding_detector: FundingAnomalyDetector | None = None,
        oi_detector: OISurgeDetector | None = None,
        liquidity_analyzer: LiquidityPoolAnalyzer | None = None,
        funding_poll_sec: float = 30.0,
        oi_poll_sec: float = 60.0,
    ):
        self.exchanges = exchanges
        self.symbols = symbols
        self.timeframes = timeframes
        self.sink = sink
        self.funding_poll_sec = funding_poll_sec
        self.oi_poll_sec = oi_poll_sec

        self.volume_detector = volume_detector or VolumeSpikeDetector()
        self.funding_detector = funding_detector or FundingAnomalyDetector()
        self.oi_detector = oi_detector or OISurgeDetector()
        self.liquidity_analyzer = liquidity_analyzer or LiquidityPoolAnalyzer()

        self._stop = asyncio.Event()

    # ---- public dispatch hooks (used by both live and replay paths) ---- #

    async def on_kline(self, exchange: str, symbol: str, bar: Kline) -> None:
        ev = self.volume_detector.feed(exchange, symbol, bar)
        if ev is not None:
            await self.sink(ev)
        for ev in self.liquidity_analyzer.feed(exchange, symbol, bar):
            await self.sink(ev)

    async def on_funding(self, exchange: str, snap: FundingSnapshot) -> None:
        for ev in self.funding_detector.feed(exchange, snap):
            await self.sink(ev)

    async def on_oi(self, exchange: str, snap: OISnapshot) -> None:
        ev = self.oi_detector.feed(exchange, snap)
        if ev is not None:
            await self.sink(ev)

    # ---- live runner using ccxt.pro ---- #

    async def run(self) -> None:
        """Start one task per (exchange, symbol, stream-type). Cancels on stop."""
        try:
            import ccxt.pro as ccxtpro  # noqa: WPS433 — lazy import on purpose
        except ImportError as e:  # pragma: no cover - import-time only
            raise RuntimeError(
                "ccxt.pro is required for live mode. Install with `pip install ccxt`."
            ) from e

        clients: dict[str, Any] = {}
        for ex_name in self.exchanges:
            klass = getattr(ccxtpro, ex_name, None)
            if klass is None:
                raise ValueError(f"ccxt.pro does not support exchange {ex_name!r}")
            client = klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            clients[ex_name] = client

        tasks: list[asyncio.Task] = []
        try:
            for ex_name, client in clients.items():
                for sym in self.symbols:
                    for tf in self.timeframes:
                        tasks.append(asyncio.create_task(self._run_klines(ex_name, client, sym, tf)))
                    tasks.append(asyncio.create_task(self._run_funding(ex_name, client, sym)))
                    tasks.append(asyncio.create_task(self._run_oi(ex_name, client, sym)))
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for client in clients.values():
                with _swallow():
                    await client.close()

    def stop(self) -> None:
        self._stop.set()

    # ---- per-stream loops with auto-reconnect ---- #

    async def _run_klines(self, exchange: str, client: Any, symbol: str, tf: str) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ohlcv = await client.watch_ohlcv(symbol, tf)
                # ccxt returns a list of [ts, o, h, l, c, v] arrays. Use the last.
                if not ohlcv:
                    continue
                ts, o, h, lo, c, v = ohlcv[-1]
                bar = Kline(ts=int(ts), open=o, high=h, low=lo, close=c, volume=v, timeframe=tf)
                await self.on_kline(exchange, symbol, bar)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("kline stream error %s/%s/%s: %s", exchange, symbol, tf, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_funding(self, exchange: str, client: Any, symbol: str) -> None:
        while not self._stop.is_set():
            try:
                # Funding rate stream support varies by exchange; fall back to REST poll.
                if hasattr(client, "watch_funding_rate"):
                    fr = await client.watch_funding_rate(symbol)
                else:
                    fr = await client.fetch_funding_rate(symbol)
                    await asyncio.sleep(self.funding_poll_sec)
                rate = float(fr.get("fundingRate") or fr.get("rate") or 0.0)
                ts = int(fr.get("timestamp") or time.time() * 1000)
                next_ts = fr.get("fundingTimestamp")
                snap = FundingSnapshot(ts=ts, symbol=symbol, rate=rate, next_funding_ts=next_ts)
                await self.on_funding(exchange, snap)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("funding stream error %s/%s: %s", exchange, symbol, e)
                await asyncio.sleep(2.0)

    async def _run_oi(self, exchange: str, client: Any, symbol: str) -> None:
        while not self._stop.is_set():
            try:
                # Most exchanges only expose OI via REST; poll periodically.
                if hasattr(client, "fetch_open_interest"):
                    oi = await client.fetch_open_interest(symbol)
                    ticker = await client.fetch_ticker(symbol)
                    snap = OISnapshot(
                        ts=int(time.time() * 1000),
                        symbol=symbol,
                        open_interest=float(oi.get("openInterestAmount") or oi.get("openInterest") or 0.0),
                        price=float(ticker.get("last") or 0.0),
                    )
                    await self.on_oi(exchange, snap)
                await asyncio.sleep(self.oi_poll_sec)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("oi poll error %s/%s: %s", exchange, symbol, e)
                await asyncio.sleep(5.0)


class _swallow:
    """Context manager that swallows exceptions during shutdown."""

    def __enter__(self) -> _swallow:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401, ANN001
        return True
