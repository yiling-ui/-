"""
trailing.py — trailing stop FSM.

State machine:
    INIT       -> ARMED        on entry fill (initial stop already at OB edge)
    ARMED      -> BREAKEVEN    when unrealized_r >= 1.0
    BREAKEVEN  -> TRAILING     when unrealized_r >= 2.0 (start ATR trailing)
    TRAILING   -> TRAILING     stop tightens via cancel+replace; never widens
    *          -> CLOSED       on exit fill / liquidation / manual

Invariant (proved by `_propose`):
    For LONG:  new_stop >= prev_stop  (monotone non-decreasing)
    For SHORT: new_stop <= prev_stop  (monotone non-increasing)

The FSM ONLY proposes new stop levels. Actually placing them on the
exchange via cancel+replace lives in the executor, because if the replace
fails we MUST keep the old hard stop in force (SR-2).
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
    CLOSED = "closed"


@dataclass
class TrailingStopFSM:
    """
    Pure logic. Inputs: position + current price + ATR. Output: optional
    new_stop suggestion. The FSM never lowers the protective level for
    longs or raises it for shorts.
    """

    breakeven_r: float = 1.0
    trail_start_r: float = 2.0
    atr_multiplier: float = 2.0
    state: TrailingState = TrailingState.ARMED

    def tick(
        self,
        *,
        position: Position,
        current_price: float,
        atr: float,
    ) -> tuple[TrailingState, float | None]:
        """
        Returns (new_state, new_stop_or_None).

        ``None`` means: no change required.
        """
        if position.closed or self.state == TrailingState.CLOSED:
            return TrailingState.CLOSED, None

        r_unit = position.stop_distance
        if r_unit <= 0:
            return self.state, None

        # signed unrealized R
        if position.side == Side.LONG:
            unrealized_r = (current_price - position.entry_price) / r_unit
        else:
            unrealized_r = (position.entry_price - current_price) / r_unit

        # ARMED -> BREAKEVEN
        if self.state == TrailingState.ARMED and unrealized_r >= self.breakeven_r:
            new_stop = self._propose(position, position.entry_price)
            if new_stop is not None:
                self.state = TrailingState.BREAKEVEN
                return self.state, new_stop

        # BREAKEVEN -> TRAILING (transition can happen directly from ARMED if a
        # bar gaps past 2R)
        if self.state in (TrailingState.ARMED, TrailingState.BREAKEVEN) and unrealized_r >= self.trail_start_r:
            candidate = self._atr_stop(position, current_price, atr)
            new_stop = self._propose(position, candidate)
            if new_stop is not None:
                self.state = TrailingState.TRAILING
                return self.state, new_stop

        # TRAILING — keep tightening
        if self.state == TrailingState.TRAILING:
            candidate = self._atr_stop(position, current_price, atr)
            new_stop = self._propose(position, candidate)
            if new_stop is not None:
                return TrailingState.TRAILING, new_stop

        return self.state, None

    def close(self) -> None:
        self.state = TrailingState.CLOSED

    # ---------------------- helpers ---------------------- #

    def _atr_stop(self, position: Position, price: float, atr: float) -> float:
        if position.side == Side.LONG:
            return price - self.atr_multiplier * atr
        return price + self.atr_multiplier * atr

    @staticmethod
    def _propose(position: Position, candidate: float) -> float | None:
        """Apply the monotone-tighten invariant."""
        if position.side == Side.LONG:
            if candidate <= position.current_hard_stop:
                return None
            return candidate
        # SHORT: stop tightens DOWN, so new_stop must be LOWER than current
        if candidate >= position.current_hard_stop:
            return None
        return candidate
