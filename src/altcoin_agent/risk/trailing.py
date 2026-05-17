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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from altcoin_agent.risk.state import Position, Side


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

    def tick(
        self,
        *,
        position: Position,
        current_price: float,
        atr: float,
        current_state: TrailingState,
    ) -> tuple[TrailingState, float | None, str]:
        """Return (next_state, proposed_stop or None, reason).

        Caller is responsible for: keeping the state, calling cancel+replace
        on the exchange, and refusing to apply a None stop.
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
            proposed = self._atr_stop(position, current_price, atr)
            if self._monotonic_ok(position, proposed):
                return TrailingState.TRAILING, proposed, "enter_trailing"
            return TrailingState.TRAILING, None, "enter_trailing_no_op"

        # TRAILING: tighten toward price each tick.
        if current_state == TrailingState.TRAILING:
            proposed = self._atr_stop(position, current_price, atr)
            if self._monotonic_ok(position, proposed):
                return TrailingState.TRAILING, proposed, "tighten"
            return TrailingState.TRAILING, None, "tighten_no_op"

        # No state transition.
        return current_state, None, "no_change"

    # ---------------- helpers ---------------- #

    def _atr_stop(self, position: Position, price: float, atr: float) -> float:
        if position.side == Side.LONG:
            return price - self.atr_multiplier * atr
        return price + self.atr_multiplier * atr

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
