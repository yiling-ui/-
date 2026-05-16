"""pump_phase.py — Pump-phase finite state machine (QUADRANT Phase 1.C / 3).

Identifies which of seven canonical phases a 妖币 (alt-pump) symbol is in,
based on a stream of 1-minute klines plus a few rolling stats:

    ACCUMULATION  -> RAMP -> PARABOLIC -> BLOWOFF_TOP -> CRASH -> BLEED -> DEAD

The exact transition rules come from QUADRANT_STRATEGY_PLAN section 四.
This module is the rule encoding; the trainer (Phase 4) is allowed to
*tune* the thresholds via constructor args, but the topology is fixed.

Design constraints:

* **Pure / mock-friendly.** No async, no I/O, no clocks. The caller provides
  the kline + stats; the FSM returns the next state and the transitions
  observed since the last call.
* **Deterministic.** Identical input streams produce identical state
  trajectories (so the backtest runner and live runner agree).
* **Cheap.** O(1) work per kline; the state machine carries a tiny ring of
  recent close prices for blow-off detection but nothing more. Multi-symbol
  callers spin up one FSM per symbol.
* **Conservative on cold start.** Until ``min_history_bars`` klines have
  been ingested the FSM stays in ``ACCUMULATION`` so a fresh start can't
  fire RAMP/PARABOLIC on the first 5 bars.

The phase labels are *advisory* — they feed the ConfidenceGate and the
risk parameters but they do not by themselves open or close orders.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class PumpPhase(str, Enum):
    ACCUMULATION = "accumulation"
    RAMP = "ramp"
    PARABOLIC = "parabolic"
    BLOWOFF_TOP = "blowoff_top"
    CRASH = "crash"
    BLEED = "bleed"
    DEAD = "dead"


# --------------------------------------------------------------------- #
# Inputs the FSM consumes
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class KlineBar:
    """One 1-minute (or other timeframe) bar.

    ``vol_z_score`` is the bar's volume z-score over a rolling window
    (e.g. the last 30 days). Computing it lives upstream — keeping the
    FSM oblivious to window choice keeps it trivially testable.
    """

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vol_z_score: float


@dataclass(frozen=True)
class PhaseInputs:
    """All non-bar context the FSM needs for a single ``advance()`` call.

    These are computed by the caller (price tape / screener / backtest
    runner) so the FSM can stay numerically pure.

    All percentages are *fractional* (0.30 = +30%, never 30.0). The plan
    text uses "% form" but our codebase conventionally uses fractions
    everywhere else; we follow the codebase.
    """

    # Returns ending at the current bar's close.
    pct_change_24h: float          # close vs 24h-ago close
    pct_change_6h: float           # close vs 6h-ago close
    pct_change_1h: float           # close vs 1h-ago close
    # Wick geometry of the most recent 1-day bar (for BLOWOFF_TOP).
    daily_upper_wick_to_body_ratio: float
    daily_close_pos_in_range: float  # 0 = closed at low, 1 = at high
    # Wick geometry of the most recent 5-minute bar (for CRASH).
    intrabar_lower_wick_pct: float
    # Aggregate context.
    realized_vol_pct_30d: float    # historical vol-of-vol, used by BLEED
    days_since_last_pump: int      # > N idle days promotes BLEED -> DEAD


# --------------------------------------------------------------------- #
# Threshold defaults (plan-section 四). Trainer can override.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseThresholds:
    # ACCUMULATION
    accumulation_max_vol_z: float = 1.0
    accumulation_price_band_pct: float = 0.05  # ±5%
    # RAMP
    ramp_min_vol_z: float = 3.0
    ramp_min_24h_pct: float = 0.30
    ramp_max_24h_pct: float = 2.00
    # PARABOLIC
    parabolic_min_vol_z: float = 6.0
    parabolic_min_6h_pct: float = 1.00
    # BLOWOFF_TOP
    blowoff_min_upper_wick_to_body: float = 2.0
    blowoff_max_close_pos_in_range: float = 0.50
    # CRASH
    crash_min_1h_drop_pct: float = 0.30   # at least -30% in an hour
    crash_min_intrabar_lower_wick_pct: float = 0.05
    # BLEED -> DEAD
    dead_min_idle_days: int = 14
    bleed_min_idle_days: int = 3


# --------------------------------------------------------------------- #
# Transition record
# --------------------------------------------------------------------- #


@dataclass
class PhaseTransition:
    ts_ms: int
    from_phase: PumpPhase
    to_phase: PumpPhase
    reason: str


# --------------------------------------------------------------------- #
# The FSM
# --------------------------------------------------------------------- #


@dataclass
class PumpPhaseFSM:
    """One FSM per symbol. Drive it with ``advance(bar, inputs)``.

    The FSM never moves *backwards* through the canonical sequence
    (ACCUMULATION → RAMP → PARABOLIC → BLOWOFF_TOP → CRASH → BLEED → DEAD)
    *during a single pump cycle*. From DEAD, it *can* loop back to
    ACCUMULATION when an idle period is interrupted by a new vol spike,
    which mirrors the empirical pattern of meme coins reactivating after
    a year-long sleep.
    """

    thresholds: PhaseThresholds = field(default_factory=PhaseThresholds)
    min_history_bars: int = 5
    # Phase the FSM is currently in.
    state: PumpPhase = PumpPhase.ACCUMULATION
    # Compact history for the backtester to replay.
    transitions: list[PhaseTransition] = field(default_factory=list)
    _bars_seen: int = 0
    _recent_closes: deque[float] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    _last_bar_ts: int = 0

    # ---- public API ---- #

    def advance(self, bar: KlineBar, inputs: PhaseInputs) -> PumpPhase:
        """Ingest one bar + its derived inputs; return the resulting phase.

        Multiple transitions per call are possible (e.g. ACCUMULATION
        directly to PARABOLIC if a single bar lights up both a 6.0 vol z
        and a +100% 6h move). We chase transitions until the state
        stabilises so the same bar yields the same final state regardless
        of how many transitions it hops through. We cap the chase at 4
        hops to avoid pathological loops if a future trainer breaks the
        topology assumption.
        """
        # Reject out-of-order bars: the FSM is implicitly single-stream.
        # ccxt.pro WS reconnects can replay; ignoring late bars is the
        # same defensive pattern used by RegimeFilter.
        if bar.ts_ms <= self._last_bar_ts and self._last_bar_ts > 0:
            return self.state

        self._last_bar_ts = bar.ts_ms
        self._recent_closes.append(bar.close)
        self._bars_seen += 1

        if self._bars_seen < self.min_history_bars:
            return self.state

        # Multi-hop chase, bounded.
        for _ in range(4):
            next_state, reason = self._evaluate(bar, inputs)
            if next_state is self.state:
                break
            self.transitions.append(
                PhaseTransition(
                    ts_ms=bar.ts_ms,
                    from_phase=self.state,
                    to_phase=next_state,
                    reason=reason,
                )
            )
            self.state = next_state
        return self.state

    def reset(self, *, to: PumpPhase = PumpPhase.ACCUMULATION) -> None:
        self.state = to
        self.transitions.clear()
        self._bars_seen = 0
        self._recent_closes.clear()
        self._last_bar_ts = 0

    # ---- internal: per-state transition rules ---- #

    def _evaluate(
        self, bar: KlineBar, x: PhaseInputs
    ) -> tuple[PumpPhase, str]:
        """Return (next_state, reason). May equal self.state (no-op)."""
        t = self.thresholds
        s = self.state

        # CRASH is the highest-priority transition: a sudden drop is
        # actionable from any non-DEAD state.
        if s is not PumpPhase.DEAD and self._is_crash(x):
            return (
                PumpPhase.CRASH,
                f"crash:1h={x.pct_change_1h:+.2%},"
                f"wick={x.intrabar_lower_wick_pct:.2%}",
            )

        if s is PumpPhase.ACCUMULATION:
            if self._is_ramp(x, bar):
                return (
                    PumpPhase.RAMP,
                    f"vol_z={bar.vol_z_score:.2f},24h={x.pct_change_24h:+.2%}",
                )
            return s, ""

        if s is PumpPhase.RAMP:
            if self._is_parabolic(x, bar):
                return (
                    PumpPhase.PARABOLIC,
                    f"vol_z={bar.vol_z_score:.2f},6h={x.pct_change_6h:+.2%}",
                )
            # RAMP can collapse straight to BLEED if vol dies.
            if x.days_since_last_pump >= t.bleed_min_idle_days:
                return PumpPhase.BLEED, "ramp_to_bleed:idle_days"
            return s, ""

        if s is PumpPhase.PARABOLIC:
            if self._is_blowoff(x):
                return (
                    PumpPhase.BLOWOFF_TOP,
                    f"upper_wick={x.daily_upper_wick_to_body_ratio:.1f},"
                    f"close_pos={x.daily_close_pos_in_range:.2f}",
                )
            return s, ""

        if s is PumpPhase.BLOWOFF_TOP:
            # The next state must be CRASH (handled above) or BLEED.
            if x.days_since_last_pump >= t.bleed_min_idle_days:
                return PumpPhase.BLEED, "blowoff_to_bleed:cooled"
            return s, ""

        if s is PumpPhase.CRASH:
            if x.days_since_last_pump >= t.bleed_min_idle_days:
                return PumpPhase.BLEED, "crash_settled"
            return s, ""

        if s is PumpPhase.BLEED:
            if x.days_since_last_pump >= t.dead_min_idle_days:
                return PumpPhase.DEAD, "bleed_to_dead:idle"
            return s, ""

        if s is PumpPhase.DEAD:
            # Reawakening: a meme coin going from "dead" back to "ramp"
            # via a fresh vol spike is the empirical pattern we want.
            if self._is_ramp(x, bar):
                return PumpPhase.ACCUMULATION, "reawaken:ramp_signal"
            return s, ""

        return s, ""

    # ---- predicates ---- #

    def _is_ramp(self, x: PhaseInputs, bar: KlineBar) -> bool:
        t = self.thresholds
        return (
            bar.vol_z_score >= t.ramp_min_vol_z
            and t.ramp_min_24h_pct <= x.pct_change_24h <= t.ramp_max_24h_pct
        )

    def _is_parabolic(self, x: PhaseInputs, bar: KlineBar) -> bool:
        t = self.thresholds
        return (
            bar.vol_z_score >= t.parabolic_min_vol_z
            and x.pct_change_6h >= t.parabolic_min_6h_pct
        )

    def _is_blowoff(self, x: PhaseInputs) -> bool:
        t = self.thresholds
        return (
            x.daily_upper_wick_to_body_ratio
            >= t.blowoff_min_upper_wick_to_body
            and x.daily_close_pos_in_range <= t.blowoff_max_close_pos_in_range
        )

    def _is_crash(self, x: PhaseInputs) -> bool:
        t = self.thresholds
        # Either an outright -30% hour, or a deep intrabar lower wick
        # (the classic flash-crash candle).
        return (
            x.pct_change_1h <= -t.crash_min_1h_drop_pct
            or x.intrabar_lower_wick_pct
            >= t.crash_min_intrabar_lower_wick_pct
            and x.pct_change_1h <= 0
        )


__all__ = [
    "KlineBar",
    "PhaseInputs",
    "PhaseThresholds",
    "PhaseTransition",
    "PumpPhase",
    "PumpPhaseFSM",
]
