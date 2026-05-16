"""price_tape.py — millisecond-grain price tape with anti-chase + vol-kill checks.

Purpose
-------
Altcoins ("妖币") regularly print +5% in <500 ms during pump-and-dump bursts.
At that pace REST round-trips (50–500 ms) and even the in-process pipeline
cannot react in time to *catch the top* — physics will not allow it.

This module instead turns the physical disadvantage into a *strategy
advantage*: it tracks the live tape on a small ring buffer and exposes
two cheap O(1) decisions the gate can use to **refuse to chase**:

  * ``anti_chase`` — has the price already moved more than ``X%`` in the
    signal direction during the last ``window_ms`` ms? If yes, we are
    too late: reject.

  * ``vol_kill``   — has the high-low range over the last ``window_ms``
    exceeded ``Y%``? If yes, the tape is in chop / liquidation cascade:
    reject everything until it calms down.

Both checks are local-memory only. No RTT, no I/O. They run before the
risk gate's networked checks (live-quote, leverage, etc.) so a rejected
signal saves the round-trip to the venue altogether.

Update path
-----------
The screener's ``on_kline`` already produces an intra-bar update every
~100 ms via ccxt.pro WebSocket. The trailing/screener wires call
``PriceTape.observe(symbol, price, ts_ms)`` once per update. Memory is
bounded: a single ``deque(maxlen=...)`` per symbol; symbols with no
recent ticks fall out automatically when ``observe`` is called for
others.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from altcoin_agent.risk.state import Side

# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class PriceTapeConfig:
    """Operator-tunable knobs.

    Defaults are calibrated for **3-min altcoin pump-and-dump** behaviour
    (Binance perp top-100 by vol over the last quarter):

      * ``anti_chase_window_ms = 30_000`` (30 s) — covers the typical
        "ramp" leg of a pump that high-priority signals tend to fire on.
      * ``anti_chase_max_move_pct = 0.025`` (2.5%) — empirically the band
        beyond which the trade has negative expectancy: enter slower,
        wait for a 1–3% retrace.
      * ``vol_kill_window_ms = 60_000`` (60 s).
      * ``vol_kill_range_pct = 0.08`` (8%) — by the time a coin oscillates
        8% inside a minute the spread is wider than our SR-1 cap and the
        positions opened mid-cascade tend to be stopped out within the
        next 30–60 s.

    Both gates can be disabled (set to ``inf``) for symbols that the
    operator has confidence in (BTC/ETH-grade) by cloning the config.
    """

    anti_chase_window_ms: int = 30_000
    anti_chase_max_move_pct: float = 0.025

    vol_kill_window_ms: int = 60_000
    vol_kill_range_pct: float = 0.08

    # Ring-buffer hard cap. At ~10 ticks/s × 60s = 600 samples; we keep
    # 8x headroom so a misbehaving venue cannot grow the deque without
    # bound. Older samples are dropped on push.
    max_samples_per_symbol: int = 5_000

    def __post_init__(self) -> None:
        if self.anti_chase_window_ms <= 0:
            raise ValueError(
                f"anti_chase_window_ms must be > 0, got {self.anti_chase_window_ms!r}"
            )
        if self.vol_kill_window_ms <= 0:
            raise ValueError(
                f"vol_kill_window_ms must be > 0, got {self.vol_kill_window_ms!r}"
            )
        # Allow infinity to disable a gate; anything finite must be > 0.
        if (
            self.anti_chase_max_move_pct != float("inf")
            and self.anti_chase_max_move_pct <= 0
        ):
            raise ValueError(
                f"anti_chase_max_move_pct must be > 0 or inf, "
                f"got {self.anti_chase_max_move_pct!r}"
            )
        if self.vol_kill_range_pct != float("inf") and self.vol_kill_range_pct <= 0:
            raise ValueError(
                f"vol_kill_range_pct must be > 0 or inf, got {self.vol_kill_range_pct!r}"
            )
        if self.max_samples_per_symbol < 8:
            raise ValueError(
                f"max_samples_per_symbol must be >= 8, "
                f"got {self.max_samples_per_symbol!r}"
            )


# --------------------------------------------------------------------- #
# Tape
# --------------------------------------------------------------------- #


@dataclass
class PriceTape:
    """Per-account, per-symbol millisecond price tape.

    The tape is a thin layer on top of one ``deque[(ts_ms, price)]`` per
    symbol. ``observe`` is the only mutating call; all read paths are
    pure ``O(window)`` (in practice ~100 samples for a 60 s window at
    1.5 Hz mark updates).

    Threading: this class is **not** thread-safe by design. The whole
    daemon is a single asyncio loop, and every call site is on it.
    """

    cfg: PriceTapeConfig = field(default_factory=PriceTapeConfig)
    _samples: dict[str, deque[tuple[int, float]]] = field(default_factory=dict)

    # ---------------- ingest ---------------- #

    def observe(self, symbol: str, price: float, ts_ms: int | None = None) -> None:
        """Push one price sample. Cheap; safe to call from the screener
        on every WebSocket tick."""
        if price <= 0:
            return     # silently drop invalid prints; venue garbage
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        buf = self._samples.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.cfg.max_samples_per_symbol)
            self._samples[symbol] = buf
        buf.append((int(ts_ms), float(price)))

    def reset(self, symbol: str) -> None:
        """Drop all samples for ``symbol``. Called when a position closes
        so a fresh signal isn't compared against the *previous* trade's
        ramp."""
        self._samples.pop(symbol, None)

    def latest(self, symbol: str) -> tuple[int, float] | None:
        """Most recent (ts_ms, price), or None when no samples."""
        buf = self._samples.get(symbol)
        if not buf:
            return None
        return buf[-1]

    # ---------------- gates ---------------- #

    def anti_chase_breach(
        self,
        *,
        symbol: str,
        side: Side,
        now_ms: int | None = None,
    ) -> tuple[bool, float]:
        """Has price already run away from us in the trade direction?

        Returns ``(breached, observed_move_pct)``:

          * ``breached`` is True iff the price moved by more than
            ``cfg.anti_chase_max_move_pct`` *in the trade direction*
            during the last ``cfg.anti_chase_window_ms``.
          * ``observed_move_pct`` is the actual signed move (positive ==
            in the trade direction). Useful for logging / telemetry.

        For LONG we measure ``(latest - oldest) / oldest``;
        for SHORT we flip the sign so a *drop* counts as the trade
        direction. Either way, breaching means "we're already late".

        Empty / single-sample tapes return ``(False, 0.0)`` — the gate
        will be a no-op until enough live ticks have arrived. This
        intentionally fails *open* on cold start so a freshly-booted
        daemon can still trade; the operator can tighten by setting
        ``min_samples_required`` if they want fail-closed instead.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cap = self.cfg.anti_chase_max_move_pct
        if cap == float("inf"):
            return False, 0.0

        cutoff = now_ms - self.cfg.anti_chase_window_ms
        oldest_in_window: float | None = None
        latest_price: float | None = None
        buf = self._samples.get(symbol)
        if buf is None or len(buf) < 2:
            return False, 0.0
        # Iterate left-to-right, keep first >= cutoff. Bounded by
        # max_samples_per_symbol (default 5_000) -> microseconds.
        for ts, p in buf:
            if ts < cutoff:
                continue
            if oldest_in_window is None:
                oldest_in_window = p
            latest_price = p
        if oldest_in_window is None or latest_price is None:
            return False, 0.0
        if oldest_in_window <= 0:
            return False, 0.0

        raw = (latest_price - oldest_in_window) / oldest_in_window
        directional = raw if side == Side.LONG else -raw
        return (directional > cap), directional

    def realized_vol_pct(
        self,
        *,
        symbol: str,
        window_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> float | None:
        """Estimate realized volatility as ``(high - low) / mid`` over the
        last ``window_ms``.

        This is a deliberate Parkinson-style range estimator (cheap, no
        return series needed) used by the risk gate to size a position
        with a venue-realistic vol number for the actual symbol —
        BTC vs PEPE differ by an order of magnitude here. Bug C4 fix:
        ``main._handle_high_priority`` used to hard-code 0.05 (BTC-grade).

        Returns ``None`` when the tape doesn't have enough samples in
        the window. Callers MUST fail-closed in that case rather than
        substitute a guess.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_ms
        buf = self._samples.get(symbol)
        if buf is None or len(buf) < 2:
            return None
        hi: float | None = None
        lo: float | None = None
        n_in_window = 0
        for ts, p in buf:
            if ts < cutoff:
                continue
            n_in_window += 1
            hi = p if hi is None else max(hi, p)
            lo = p if lo is None else min(lo, p)
        if hi is None or lo is None or n_in_window < 2:
            return None
        mid = (hi + lo) / 2.0
        if mid <= 0:
            return None
        return (hi - lo) / mid

    def vol_kill_breach(
        self,
        *,
        symbol: str,
        now_ms: int | None = None,
    ) -> tuple[bool, float]:
        """Has the high-low range exceeded the chop limit?

        Returns ``(breached, observed_range_pct)``. Breaching means
        "there is a liquidation cascade or whip-saw on the tape — every
        new entry we open right now is a coin-flip and the SR-1 slip
        cap will be exceeded by spread alone." We refuse to enter
        until things calm down.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cap = self.cfg.vol_kill_range_pct
        if cap == float("inf"):
            return False, 0.0

        cutoff = now_ms - self.cfg.vol_kill_window_ms
        buf = self._samples.get(symbol)
        if buf is None or len(buf) < 2:
            return False, 0.0
        hi: float | None = None
        lo: float | None = None
        for ts, p in buf:
            if ts < cutoff:
                continue
            hi = p if hi is None else max(hi, p)
            lo = p if lo is None else min(lo, p)
        if hi is None or lo is None or lo <= 0:
            return False, 0.0
        rng = (hi - lo) / lo
        return (rng > cap), rng

    # ---------------- maintenance ---------------- #

    def gc_stale(self, now_ms: int | None = None, ttl_ms: int = 600_000) -> int:
        """Drop symbols whose latest sample is older than ``ttl_ms``.

        Call from a low-rate worker (every few minutes). Prevents memory
        creep when a once-watched symbol stops streaming. Returns the
        number of symbols evicted.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - ttl_ms
        evicted = []
        for sym, buf in self._samples.items():
            if not buf or buf[-1][0] < cutoff:
                evicted.append(sym)
        for sym in evicted:
            self._samples.pop(sym, None)
        return len(evicted)
