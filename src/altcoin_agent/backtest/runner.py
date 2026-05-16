"""runner.py — Phase tagging + minimal backtest driver (QUADRANT Phase 3).

This is the Phase 1-3 deliverable: drive ``PumpPhaseFSM`` over a stream
of cached klines and emit a phase-tagged record per bar. The trainer
(Phase 4) consumes these records to learn rule weights; the full
matching engine + slippage simulation (Phase B.4) plugs in later by
wrapping ``PhaseTaggedBar`` with fill / PnL fields.

What this module *does*:

    * ``compute_phase_inputs(...)`` derives the FSM's ``PhaseInputs`` from
      a rolling window of bars + a vol z-score series. All math is
      vectorised over plain lists (no numpy/pandas dep) so running a
      backtest doesn't pull in extra deps.
    * ``BacktestRunner`` walks a list/iterator of bars in order, drives
      one FSM, and yields ``PhaseTaggedBar`` records.
    * It also exposes ``BacktestRunner.summary()`` returning per-phase
      counts and the list of phase transitions — small report needed
      by both the smoke test and the trainer.

What this module *deliberately does not do* (kept for Phase B.4):

    * No order matching, fill, slippage, fees.
    * No PnL. No equity curve.
    * No walk-forward windowing.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from altcoin_agent.risk.pump_phase import (
    KlineBar,
    PhaseInputs,
    PhaseThresholds,
    PumpPhase,
    PumpPhaseFSM,
)

logger = logging.getLogger(__name__)


# Bar-counts per timeframe used to compute lookback windows for the
# ``PhaseInputs`` deltas. We only support 1m bars in v1; the trainer
# rolls higher TFs by aggregating.
BARS_PER_HOUR_1M = 60
BARS_PER_DAY_1M = 60 * 24
BARS_PER_30D_1M = 30 * BARS_PER_DAY_1M


# --------------------------------------------------------------------- #
# Output record
# --------------------------------------------------------------------- #


@dataclass
class PhaseTaggedBar:
    """One bar plus the phase the FSM was in *after* ingesting it.

    ``inputs`` is the derived input vector — useful for the trainer,
    optional for the smoke test.
    """

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vol_z_score: float
    phase: PumpPhase
    inputs: PhaseInputs

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


# --------------------------------------------------------------------- #
# Input derivation
# --------------------------------------------------------------------- #


def compute_phase_inputs(
    history: list[KlineBar],
    *,
    last_pump_idx: int | None = None,
) -> PhaseInputs:
    """Derive the ``PhaseInputs`` for the most recent bar in ``history``.

    ``history`` is *all bars seen so far*, oldest-first, including the
    just-appended current bar. ``last_pump_idx`` is the index of the
    last detected pump bar (for ``days_since_last_pump``); ``None`` =
    "never seen". The runner maintains it.

    Uses 1-minute bar counts internally. Aggregating to coarser TFs is
    the caller's responsibility (the trainer flattens 5m/1d into 1m
    before driving us, so this stays simple).
    """
    n = len(history)
    if n == 0:
        return _zero_inputs()
    cur = history[-1]

    pct_24h = _pct_change_lookback(history, BARS_PER_DAY_1M)
    pct_6h = _pct_change_lookback(history, 6 * BARS_PER_HOUR_1M)
    pct_1h = _pct_change_lookback(history, BARS_PER_HOUR_1M)

    daily_window = history[-BARS_PER_DAY_1M:] if n >= BARS_PER_DAY_1M else history
    daily_upper_wick_to_body, daily_close_pos = _wick_geometry(daily_window)

    intrabar_lower_wick = _bar_lower_wick_pct(cur)

    # 30d realised vol approximation: stddev of bar close-to-close
    # log returns over the last 30 days * sqrt(bars/day) to annualise
    # *to a daily* horizon (simplified — full implementation in trainer).
    realized_vol = _realized_vol(history, BARS_PER_30D_1M)

    if last_pump_idx is None or last_pump_idx >= n:
        days_idle = 9999
    else:
        bars_since = max(0, (n - 1) - last_pump_idx)
        days_idle = bars_since // BARS_PER_DAY_1M

    return PhaseInputs(
        pct_change_24h=pct_24h,
        pct_change_6h=pct_6h,
        pct_change_1h=pct_1h,
        daily_upper_wick_to_body_ratio=daily_upper_wick_to_body,
        daily_close_pos_in_range=daily_close_pos,
        intrabar_lower_wick_pct=intrabar_lower_wick,
        realized_vol_pct_30d=realized_vol,
        days_since_last_pump=days_idle,
    )


def _zero_inputs() -> PhaseInputs:
    return PhaseInputs(
        pct_change_24h=0.0,
        pct_change_6h=0.0,
        pct_change_1h=0.0,
        daily_upper_wick_to_body_ratio=0.0,
        daily_close_pos_in_range=0.5,
        intrabar_lower_wick_pct=0.0,
        realized_vol_pct_30d=0.0,
        days_since_last_pump=9999,
    )


def _pct_change_lookback(history: list[KlineBar], lookback: int) -> float:
    if not history:
        return 0.0
    cur = history[-1].close
    if cur <= 0:
        return 0.0
    if len(history) <= lookback:
        ref = history[0].close
    else:
        ref = history[-lookback - 1].close
    if ref <= 0:
        return 0.0
    return (cur - ref) / ref


def _wick_geometry(window: list[KlineBar]) -> tuple[float, float]:
    """Aggregate a multi-bar window into a synthetic 'daily' candle.

    The PumpPhaseFSM expects upper-wick-to-body and close-position-in-range
    derived from the most recent *day* of bars. We aggregate the window:
      open  = first bar's open
      close = last bar's close
      high  = max(high)
      low   = min(low)
    """
    if not window:
        return 0.0, 0.5
    o = window[0].open
    c = window[-1].close
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    rng = hi - lo
    body = abs(c - o)
    upper_wick = hi - max(o, c)
    if body <= 0:
        wick_ratio = upper_wick / max(rng, 1e-9) * 4.0
    else:
        wick_ratio = upper_wick / body
    if rng <= 0:
        close_pos = 0.5
    else:
        close_pos = (c - lo) / rng
    return max(0.0, wick_ratio), max(0.0, min(1.0, close_pos))


def _bar_lower_wick_pct(bar: KlineBar) -> float:
    """Lower wick as a fraction of the bar's open price."""
    if bar.open <= 0:
        return 0.0
    lower_wick = min(bar.open, bar.close) - bar.low
    if lower_wick <= 0:
        return 0.0
    return lower_wick / bar.open


def _realized_vol(history: list[KlineBar], lookback: int) -> float:
    """Stddev of close-to-close log returns over the last ``lookback`` bars."""
    n = min(len(history), lookback)
    if n < 2:
        return 0.0
    sample = history[-n:]
    rets: list[float] = []
    prev = sample[0].close
    for bar in sample[1:]:
        if prev > 0 and bar.close > 0:
            rets.append(math.log(bar.close / prev))
        prev = bar.close
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #


@dataclass
class BacktestRunner:
    """Drive a single ``PumpPhaseFSM`` over an ordered bar stream.

    Use ``iter_phase_tagged()`` for streaming, or ``run()`` for a buffered
    list result. Both are deterministic.

    The runner keeps an in-memory rolling window of up to 32 days of 1m
    bars (the longest lookback the FSM needs is 30d for realised vol);
    if a caller drives a multi-year run they can rely on this bound.
    """

    fsm: PumpPhaseFSM = field(default_factory=PumpPhaseFSM)
    thresholds: PhaseThresholds = field(default_factory=PhaseThresholds)
    history_max_bars: int = 32 * BARS_PER_DAY_1M
    # Vol z-score window: bars used to compute the rolling z-score of
    # the current bar's volume. 30 days of 1m bars by default.
    vol_zscore_window: int = BARS_PER_30D_1M

    _history: deque[KlineBar] = field(default_factory=deque)
    _vol_buf: deque[float] = field(default_factory=deque)
    _last_pump_idx: int | None = None
    _phase_counts: Counter = field(default_factory=Counter)
    _start_idx: int = 0

    def __post_init__(self) -> None:
        # Keep the FSM and thresholds aligned.
        self.fsm.thresholds = self.thresholds
        self._history = deque(maxlen=self.history_max_bars)
        self._vol_buf = deque(maxlen=self.vol_zscore_window)

    # ---- public ---- #

    def run(self, bars: Iterable[Any]) -> list[PhaseTaggedBar]:
        return list(self.iter_phase_tagged(bars))

    def iter_phase_tagged(
        self, bars: Iterable[Any]
    ) -> Iterator[PhaseTaggedBar]:
        for raw in bars:
            kbar = self._coerce_bar(raw)
            if kbar is None:
                continue
            self._vol_buf.append(kbar.volume)
            z = self._vol_zscore(kbar.volume)
            kbar = KlineBar(
                ts_ms=kbar.ts_ms,
                open=kbar.open,
                high=kbar.high,
                low=kbar.low,
                close=kbar.close,
                volume=kbar.volume,
                vol_z_score=z,
            )
            self._history.append(kbar)
            history_list = list(self._history)
            inputs = compute_phase_inputs(
                history_list, last_pump_idx=self._last_pump_idx,
            )
            phase = self.fsm.advance(kbar, inputs)
            # Mark a pump bar (for days_since_last_pump). A bar is
            # considered a pump bar if its vol z >= ramp threshold.
            if z >= self.thresholds.ramp_min_vol_z:
                self._last_pump_idx = len(history_list) - 1
            self._phase_counts[phase] += 1
            yield PhaseTaggedBar(
                ts_ms=kbar.ts_ms,
                open=kbar.open,
                high=kbar.high,
                low=kbar.low,
                close=kbar.close,
                volume=kbar.volume,
                vol_z_score=z,
                phase=phase,
                inputs=inputs,
            )

    # ---- summary ---- #

    def summary(self) -> dict[str, Any]:
        return {
            "phase_counts": {p.value: int(self._phase_counts[p])
                              for p in PumpPhase},
            "transitions": [
                {
                    "ts_ms": t.ts_ms,
                    "from": t.from_phase.value,
                    "to": t.to_phase.value,
                    "reason": t.reason,
                }
                for t in self.fsm.transitions
            ],
            "final_state": self.fsm.state.value,
        }

    # ---- helpers ---- #

    @staticmethod
    def _coerce_bar(raw: Any) -> KlineBar | None:
        """Accept a ``KlineBar`` or a [ts, o, h, l, c, v] sequence."""
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
                vol_z_score=0.0,  # set later by the runner
            )
        except (TypeError, ValueError):
            return None

    def _vol_zscore(self, current_volume: float) -> float:
        """Population z-score of ``current_volume`` over the rolling buf.

        Population (not sample) stddev so a single observation -> 0.
        We need the ramp / parabolic z-thresholds to be reachable on
        synthetic test data, so a small buffer (< 30 bars) computes
        a useful z out of what we have rather than returning 0.
        """
        n = len(self._vol_buf)
        if n < 2:
            return 0.0
        # Exclude the current bar from the reference distribution so a
        # legitimate spike doesn't raise its own bar of the variance.
        # ``_vol_buf`` already includes the current; pop it virtually
        # by iterating over the first n-1 entries.
        ref = list(self._vol_buf)[:-1] if n > 1 else list(self._vol_buf)
        if not ref:
            return 0.0
        m = sum(ref) / len(ref)
        var = sum((v - m) ** 2 for v in ref) / max(1, len(ref))
        sd = math.sqrt(var)
        if sd <= 0:
            return 0.0
        return (current_volume - m) / sd


__all__ = [
    "BacktestRunner",
    "PhaseTaggedBar",
    "compute_phase_inputs",
]
