"""state.py — shared dataclasses for the risk + execution package."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def opposite(self) -> Side:
        return Side.SHORT if self == Side.LONG else Side.LONG


@dataclass
class PositionLeg:
    """A single executed entry within a (possibly rolled) position.

    Rolling-positions: a position is now a *bag of legs*, all sharing
    one trailing stop. Leg 0 is always the original entry; legs 1+ are
    rolled additions whose margin came from the position's unrealised
    PnL at the moment of the roll.

    Invariants:
      * All legs share the same ``side`` (rolling never reverses direction).
      * ``margin_source`` is "initial" for leg 0 and "rolled_unrealized"
        for additions.
      * ``size`` is in base units, exactly like ``Position.size``.
    """

    leg_id: int
    side: Side
    size: float
    entry_price: float
    entry_ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    margin_source: str = "initial"   # "initial" | "rolled_unrealized"
    trigger_score: float | None = None  # FusedSignal.final_score at time of roll


@dataclass
class Position:
    """A live position with all the bookkeeping the trailing FSM needs.

    Multi-leg support (rolling positions):
      ``legs`` is the source of truth for size / avg entry once a roll
      happens. When ``legs`` is empty (V1.0 path), ``size`` and
      ``entry_price`` are used directly so all existing behaviour is
      preserved verbatim.
    """

    symbol: str
    exchange: str
    side: Side
    entry_price: float
    size: float                       # contracts (or base units, exchange-dependent)
    leverage: float
    initial_stop: float               # the stop set at entry — never changes
    current_stop: float               # the stop currently resting on the exchange
    stop_order_id: str | None = None  # exchange-side hard-stop id, must always exist
    opened_at_ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    trace_id: str | None = None
    closed: bool = False
    legs: list[PositionLeg] = field(default_factory=list)

    @property
    def r_unit(self) -> float:
        """1R distance — always positive.

        Computed from the *original* entry / initial_stop because that is
        what the trailing FSM keys its breakeven and trailing-armed
        thresholds off. Adding rolled legs must not change what '1R from
        the original setup' means.
        """
        return abs(self.entry_price - self.initial_stop)

    @property
    def total_size(self) -> float:
        """Sum of all leg sizes. For a V1.0 single-leg position
        (``legs == []``) this returns ``self.size`` so callers that
        haven't been adapted still see the right number."""
        if not self.legs:
            return self.size
        return sum(L.size for L in self.legs)

    @property
    def avg_entry_price(self) -> float:
        """Size-weighted average entry across all legs.

        Used by the rolling controller to compute unrealised PnL for the
        whole position, and by the close-handler to compute realised
        PnL accurately when multiple legs share a single STOP_MARKET
        fill. Falls back to ``entry_price`` when no legs are recorded
        (V1.0 path)."""
        if not self.legs:
            return self.entry_price
        total = sum(L.size for L in self.legs)
        if total <= 0:
            return self.entry_price
        return sum(L.size * L.entry_price for L in self.legs) / total

    @property
    def num_rolled_legs(self) -> int:
        """Number of *added* legs (excluding the original)."""
        if not self.legs:
            return 0
        return max(0, len(self.legs) - 1)

    def unrealised_pnl_usdt(self, mark_price: float) -> float:
        """Unrealised PnL across all legs at ``mark_price``.

        Sign matches conventional long/short: positive == in profit.
        """
        if mark_price <= 0:
            return 0.0
        if self.side == Side.LONG:
            return (mark_price - self.avg_entry_price) * self.total_size
        return (self.avg_entry_price - mark_price) * self.total_size


@dataclass
class AccountState:
    """All non-position-level state Risk Gate cares about."""

    equity_usdt: float = 10_000.0
    starting_equity_today_usdt: float = 10_000.0
    realized_pnl_today_usdt: float = 0.0
    open_positions: dict[str, Position] = field(default_factory=dict)
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    cooldown_until_ts_ms: dict[str, int] = field(default_factory=dict)
    daily_stoploss_hits: int = 0
    reconciliation_complete: bool = False
    global_trading_halted: bool = False
    halt_reason: str | None = None
    # Bug #3 fix: ISO date string ("YYYY-MM-DD", UTC) of the last day we
    # rolled over. ``None`` means "never rolled over" — the next call to
    # ``maybe_roll_over_day`` will simply stamp today without resetting
    # any counters (boot-time semantics).
    last_rollover_date_utc: str | None = None
    # Anchor hour (0-23, UTC) at which a "trading day" ends and the next
    # one begins. Defaults to 00:00 UTC, which is the conventional
    # boundary used by Binance / most prop desks. Override via
    # ``AppConfig.rollover_anchor_utc_hour`` if your reporting day is on
    # a different anchor.
    rollover_anchor_utc_hour: int = 0

    @property
    def daily_drawdown_pct(self) -> float:
        if self.starting_equity_today_usdt <= 0:
            return 0.0
        return -self.realized_pnl_today_usdt / self.starting_equity_today_usdt

    def position(self, symbol: str) -> Position | None:
        return self.open_positions.get(symbol)

    def is_in_cooldown(self, symbol: str, now_ms: int) -> bool:
        until = self.cooldown_until_ts_ms.get(symbol, 0)
        return now_ms < until

    def set_cooldown(self, symbol: str, duration_sec: int, now_ms: int) -> None:
        self.cooldown_until_ts_ms[symbol] = now_ms + duration_sec * 1000

    def halt(self, reason: str) -> None:
        self.global_trading_halted = True
        self.halt_reason = reason

    # ------------------------------------------------------------------ #
    # Bug #3 fix — daily rollover.
    #
    # ``starting_equity_today_usdt``, ``realized_pnl_today_usdt`` and
    # ``daily_stoploss_hits`` were stamped exactly once at boot (in
    # ``App.run``) and never reset. The daily-drawdown circuit breaker
    # therefore became a ONE-WAY latch: a couple of small losing days
    # would peg the running sum past 6%, and the gate would refuse every
    # subsequent signal forever. Same for the 3-strike rule.
    #
    # ``maybe_roll_over_day`` is idempotent: it checks today's UTC date
    # against ``last_rollover_date_utc`` and only resets when the date
    # changes. It is safe to call from many places (the App's startup,
    # a background ticker, the hot path right before the gate) and is
    # the foundation of the "fail-recover" posture for daily breakers.
    # ------------------------------------------------------------------ #

    def _trading_day_iso(self, now_ms: int) -> str:
        """Map a wall-clock ms timestamp to its trading-day ISO date.

        With ``rollover_anchor_utc_hour=0`` this is exactly UTC date.
        With other anchors we shift the wall clock so the anchor falls
        at "naive midnight", then take the date of that shifted moment;
        this gives a stable, monotonically-incrementing day key for any
        anchor in 0..23.
        """
        seconds = now_ms / 1000.0
        # Shift backwards by the anchor so e.g. anchor=8 means a
        # "trading day" that runs 08:00 UTC -> 08:00 UTC next day.
        anchor_offset_sec = (self.rollover_anchor_utc_hour % 24) * 3600
        shifted = seconds - anchor_offset_sec
        return datetime.fromtimestamp(shifted, tz=timezone.utc).date().isoformat()

    def maybe_roll_over_day(self, now_ms: int | None = None) -> bool:
        """Reset daily-scoped counters when the trading day flips.

        Resets:
          * ``starting_equity_today_usdt`` -> current ``equity_usdt``
          * ``realized_pnl_today_usdt``    -> 0
          * ``daily_stoploss_hits``        -> 0

        Deliberately preserves:
          * ``consecutive_losses``     — per-symbol streaks span days.
          * ``cooldown_until_ts_ms``   — symbol cooldowns are wall-clock
            timed; let the natural deadline expire.
          * ``global_trading_halted`` / ``halt_reason`` — manual ops
            interventions are sticky by design. To restart trading after
            a hard halt the operator must explicitly clear it.

        Returns True iff a rollover happened, False if same day or first
        boot stamping. Stamping the first-ever day is logged distinctly
        because there's nothing to "reset" yet.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        today = self._trading_day_iso(now_ms)
        prev = self.last_rollover_date_utc
        if prev == today:
            return False
        if prev is None:
            # First boot stamp — don't reset (everything was just initialised).
            self.last_rollover_date_utc = today
            return False
        # Genuine day flip.
        self.starting_equity_today_usdt = self.equity_usdt
        self.realized_pnl_today_usdt = 0.0
        self.daily_stoploss_hits = 0
        self.last_rollover_date_utc = today
        return True
