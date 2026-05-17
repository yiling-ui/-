"""trailing.py — TrailingStopFSM.

State machine:
    INIT -> ARMED -> BREAKEVEN -> TRAILING -> {TARGET_REACHED | CLOSED}

Transitions:
    INIT       -> ARMED      on first tick after entry
    ARMED      -> BREAKEVEN  when |unrealized PnL| >= 1R: stop -> entry
    BREAKEVEN  -> TRAILING   when |unrealized PnL| >= 2R: stop -> price -/+ atr_mult * ATR
    TRAILING   -> TRAILING   each tick: stop monotonically tightens toward price
    TRAILING   -> TARGET_REACHED  on SHORT: when (entry - price)/entry >= short_target_cap_pct
                                  -> stop is pinned just above price to force fill

Hard invariant (proved by construction):
    LONG  positions: stop is monotonically NON-DECREASING.
    SHORT positions: stop is monotonically NON-INCREASING.

R6 — pump-phase awareness (optional, default OFF).

When the caller passes ``phase`` to ``tick``, the FSM tightens the
trailing ATR multiplier in late-stage phases so the daemon books
profits before the parabolic-to-crash collapse:

    ACCUMULATION / RAMP / PARABOLIC  -> normal ``atr_multiplier``
    BLOWOFF_TOP                      -> atr_multiplier * blowoff_atr_tighten
    CRASH / BLEED / DEAD             -> atr_multiplier * crash_atr_tighten

Defaults of 0.5 / 0.3 mean a 2.5x ATR trail at parabolic phase
becomes a 1.25x trail at blowoff and 0.75x trail at crash. Existing
callers that don't pass ``phase`` get the v1.0 behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from altcoin_agent.risk.pump_phase import PumpPhase
from altcoin_agent.risk.state import Position, Side
from altcoin_agent.risk.symbol_profile import QuadrantParams


class TrailingState(str, Enum):
    INIT = "init"
    ARMED = "armed"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"
    TARGET_REACHED = "target_reached"
    CLOSED = "closed"


@dataclass
class TrailingStopFSM:
    """Stateless per call: caller passes Position and gets back (state, new_stop).

    `new_stop` is None when no replacement should be placed (state has not
    moved or the proposed stop would weaken the existing one).
    """

    atr_multiplier: float = 2.0
    breakeven_at_r: float = 1.0
    trailing_at_r: float = 2.0
    short_target_cap_pct: float = 0.70   # SHORT: force-close at -70% from entry

    # R6 — phase-aware tightening multipliers. Applied to
    # ``atr_multiplier`` when the caller passes ``phase`` to ``tick``.
    # Defaults preserve v1.0 behaviour for callers that omit ``phase``.
    blowoff_atr_tighten: float = 0.5     # half the trail at blowoff_top
    crash_atr_tighten: float = 0.3       # 30% of normal trail at crash/bleed/dead

    # ---- factories ---- #

    @classmethod
    def from_quadrant_params(
        cls, params: QuadrantParams, **overrides,
    ) -> "TrailingStopFSM":
        """Build an FSM tuned to a quadrant's risk profile.

        Pulls ``trailing_atr_mult`` and ``breakeven_at_r`` from
        ``QuadrantParams``; everything else stays at the FSM's
        defaults unless an explicit override is supplied. The
        ``QuadrantRiskFactory`` (Phase 5) already calls equivalent
        constructor kwargs; this classmethod is a less-verbose
        alternative for ad-hoc R6 callers.
        """
        kwargs = {
            "atr_multiplier": params.trailing_atr_mult,
            "breakeven_at_r": params.breakeven_at_r,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def tick(
        self,
        *,
        position: Position,
        current_price: float,
        atr: float,
        current_state: TrailingState,
        phase: PumpPhase | None = None,
    ) -> tuple[TrailingState, float | None, str]:
        """Return (next_state, proposed_stop or None, reason).

        Caller is responsible for: keeping the state, calling cancel+replace
        on the exchange, and refusing to apply a None stop.

        ``phase`` is optional. When provided, the trailing stop tightens
        in late pump phases so we book gains before the collapse.
        Omitting ``phase`` preserves v1.0 behaviour byte-for-byte.
        """
        if position.closed or current_state == TrailingState.CLOSED:
            return TrailingState.CLOSED, None, "position_closed"

        r_unit = position.r_unit
        if r_unit <= 0:
            return current_state, None, "no_r_unit"

        if position.side == Side.LONG:
            unrealized = (current_price - position.entry_price) / r_unit
        else:
            unrealized = (position.entry_price - current_price) / r_unit

        # Effective ATR multiplier for this tick.
        atr_mult = self._effective_atr_multiplier(phase)

        # SHORT-only: target cap (force close near -70%).
        if position.side == Side.SHORT:
            drop_pct = (position.entry_price - current_price) / position.entry_price
            if drop_pct >= self.short_target_cap_pct:
                pinned = self._pin_stop_just_against_price(position, current_price)
                if self._monotonic_ok(position, pinned):
                    return TrailingState.TARGET_REACHED, pinned, (
                        f"short_target_cap_reached:{drop_pct:.2%}"
                    )
                return TrailingState.TARGET_REACHED, None, (
                    "short_target_cap_already_armed"
                )

        # ----- normal progression -----
        if current_state == TrailingState.INIT:
            return TrailingState.ARMED, None, "armed"

        # ARMED -> BREAKEVEN
        if current_state in (TrailingState.ARMED,) and unrealized >= self.breakeven_at_r:
            new_stop = position.entry_price
            if self._monotonic_ok(position, new_stop):
                return TrailingState.BREAKEVEN, new_stop, "breakeven"
            return TrailingState.BREAKEVEN, None, "breakeven_no_op"

        # BREAKEVEN -> TRAILING
        if current_state == TrailingState.BREAKEVEN and unrealized >= self.trailing_at_r:
            proposed = self._atr_stop(position, current_price, atr, atr_mult)
            if self._monotonic_ok(position, proposed):
                return TrailingState.TRAILING, proposed, (
                    f"enter_trailing(atr_mult={atr_mult:.2f},phase={_phase_label(phase)})"
                )
            return TrailingState.TRAILING, None, "enter_trailing_no_op"

        # TRAILING: tighten toward price each tick.
        if current_state == TrailingState.TRAILING:
            proposed = self._atr_stop(position, current_price, atr, atr_mult)
            if self._monotonic_ok(position, proposed):
                return TrailingState.TRAILING, proposed, (
                    f"tighten(atr_mult={atr_mult:.2f},phase={_phase_label(phase)})"
                )
            return TrailingState.TRAILING, None, "tighten_no_op"

        # No state transition.
        return current_state, None, "no_change"

    # ---------------- helpers ---------------- #

    def _effective_atr_multiplier(self, phase: PumpPhase | None) -> float:
        """R6: late-phase tightening.

        Returns the FSM's base ``atr_multiplier`` multiplied by a
        phase-specific shrink factor. Unknown / None phase is a
        no-op so legacy callers see identical behaviour.
        """
        if phase is None:
            return self.atr_multiplier
        if phase == PumpPhase.BLOWOFF_TOP:
            return self.atr_multiplier * self.blowoff_atr_tighten
        if phase in (PumpPhase.CRASH, PumpPhase.BLEED, PumpPhase.DEAD):
            return self.atr_multiplier * self.crash_atr_tighten
        return self.atr_multiplier

    def _atr_stop(
        self,
        position: Position,
        price: float,
        atr: float,
        atr_mult: float | None = None,
    ) -> float:
        m = self.atr_multiplier if atr_mult is None else atr_mult
        if position.side == Side.LONG:
            return price - m * atr
        return price + m * atr

    @staticmethod
    def _pin_stop_just_against_price(pos: Position, price: float) -> float:
        """For SHORT target-cap: pin the stop just above current price."""
        if pos.side == Side.LONG:
            return price * 0.9995    # 5 bps below
        return price * 1.0005        # 5 bps above

    @staticmethod
    def _monotonic_ok(pos: Position, new_stop: float) -> bool:
        """LONG: new_stop must be >= current_stop. SHORT: <=."""
        if pos.side == Side.LONG:
            return new_stop >= pos.current_stop
        return new_stop <= pos.current_stop


def _phase_label(phase: PumpPhase | None) -> str:
    """Compact label for FSM transition reasons."""
    return phase.value if phase is not None else "n/a"
