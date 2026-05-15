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
    TARGET_REACHED = "target_reached"     # SHORT only: -70% from entry hit
    CLOSED = "closed"


@dataclass
class TrailingStopFSM:
    """
    Pure logic. Inputs: position + current price + ATR. Output: optional
    new_stop suggestion. The FSM never lowers the protective level for
    longs or raises it for shorts.

    SHORT TARGET CAP (per user mandate):
        Altcoin shorts riding a "to zero" move face two real risks:
          (1) liquidity collapses near the bottom — exiting becomes expensive
          (2) funding rate flips deeply negative — paying to hold the short
        For SHORT positions, when price has dropped >= ``short_target_cap_pct``
        from entry (default 70%), the FSM reports state=TARGET_REACHED with
        new_stop = current_price + tiny buffer. Caller MUST market-close.
        This prevents the "wait for zero" greed trap.
    """

    breakeven_r: float = 1.0
    trail_start_r: float = 2.0
    atr_multiplier: float = 2.0
    short_target_cap_pct: float = 0.70   # SR: see class docstring
    short_target_close_buffer: float = 0.0005  # 0.05% above current price for safety
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

        # SHORT TARGET CAP: hard ceiling on greed.
        # If the short has already collected >= short_target_cap_pct of entry
        # price, force-close. We surface this by signalling TARGET_REACHED with
        # a stop placed JUST ABOVE current price (so the next exchange tick
        # triggers it). This keeps the executor path identical to a normal
        # tighten — no separate "close" code path needed.
        if position.side == Side.SHORT and position.entry_price > 0:
            drop_pct = (position.entry_price - current_price) / position.entry_price
            if drop_pct >= self.short_target_cap_pct:
                close_stop = current_price * (1.0 + self.short_target_close_buffer)
                # Only act once: if we've already moved into TARGET_REACHED and
                # current_price hasn't materially changed, skip.
                if (
                    self.state != TrailingState.TARGET_REACHED
                    or close_stop < position.current_hard_stop
                ):
                    new_stop = self._propose(position, close_stop)
                    if new_stop is not None:
                        self.state = TrailingState.TARGET_REACHED
                        return self.state, new_stop

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
