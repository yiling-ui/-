"""state.py — shared dataclasses for the risk + execution package."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------ #
    # Phase B.1.3 — event-driven persistence hook.
    #
    # Old design persisted at three hard-coded points (boot, post-close,
    # one shutdown path). Cooldowns set by ``CCXTExecutor`` on stop /
    # entry failure, ``account.halt`` from the ``KillSwitchWatcher``,
    # and any future mutators were *not* persisted; a daemon crash
    # right after such a mutation would silently drop the change. The
    # ``MISS_PENALTY_AND_PRODUCTION_PLAN.md`` Phase B.1.3 calls out
    # this gap as 🟠 severe and prescribes "save() on every state
    # mutation".
    #
    # ``_on_change`` is a single callable (typically
    # ``AccountPersistor.save``) invoked whenever a daily-scoped or
    # halt/cooldown field changes. It's registered via
    # ``register_change_listener`` from ``main.App.run``. Failures in
    # the listener are caught and logged so a flaky disk never blocks
    # the trading loop. Tests can register their own listener to
    # observe and assert that every mutator triggered exactly one
    # save.
    #
    # We deliberately do NOT auto-fire on raw attribute assignment
    # (``account.equity_usdt = X``); the dataclass would have to be
    # frozen + a ``__setattr__`` override, breaking every test that
    # constructs an AccountState mid-flight. Instead we expose
    # explicit mutator helpers (``record_pnl`` / ``register_loss`` /
    # ``clear_consecutive_loss``) and wire the existing mutators
    # (``set_cooldown`` / ``halt`` / ``maybe_roll_over_day``) to
    # invoke the listener.
    _on_change: Callable[[AccountState], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def register_change_listener(
        self,
        listener: Callable[[AccountState], None] | None,
    ) -> None:
        """Attach (or detach with ``None``) a hook fired after each
        persistable mutation. Idempotent: re-registering replaces the
        previous listener.
        """
        self._on_change = listener

    def _notify_change(self) -> None:
        """Internal: invoke the registered listener. Failures are
        logged and swallowed so persistence never breaks trading.
        """
        if self._on_change is None:
            return
        try:
            self._on_change(self)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "AccountState change listener raised (swallowed): %s", e,
            )

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
        # Phase B.1.3: persist after every mutation to the cooldown
        # map. Without this, a stop-failure cooldown set in the
        # executor disappeared on restart and the symbol was trade-
        # eligible the moment the daemon came back up.
        self._notify_change()

    def halt(self, reason: str) -> None:
        self.global_trading_halted = True
        self.halt_reason = reason
        # Phase B.1.3: persist halt state immediately. The kill-switch
        # only lives in memory otherwise; a process restart would
        # silently un-halt the account.
        self._notify_change()

    # ------------------------------------------------------------------ #
    # Phase B.1.3 — PnL accounting helpers.
    #
    # ``main._on_position_close`` historically did the bookkeeping
    # inline with raw attribute mutation (``realized_pnl_today_usdt
    # += ...``, ``daily_stoploss_hits += 1``, ``consecutive_losses[...]
    # = ...``) and then called ``persistor.save`` at the end. Three
    # problems with that:
    #   (1) every other call site that wanted to record a fill had to
    #       repeat the same 6-line pattern;
    #   (2) any forgotten ``save()`` (e.g. emergency-close path in
    #       executor, or future code) lost data on crash;
    #   (3) the dashboard / metrics didn't have a single hook to
    #       observe "something just changed".
    #
    # We centralise the daily rollups here and fire ``_notify_change``
    # exactly once per logical event. The old call sites delegate to
    # these helpers; behaviour is byte-for-byte identical.
    # ------------------------------------------------------------------ #

    def record_pnl(
        self,
        symbol: str,
        realized_pnl_usdt: float,
        *,
        is_loss: bool | None = None,
    ) -> None:
        """Record a realised fill against the daily roll-ups + per-symbol
        consecutive-loss counter.

        Args:
            symbol: position symbol (used for consec-loss bookkeeping).
            realized_pnl_usdt: signed PnL of the close in USDT.
            is_loss: when None, treat ``realized_pnl_usdt < 0`` as a
                loss. Callers can override (rare) when post-mortem
                attribution differs from raw sign — kept for forward
                compatibility with the upcoming MissPenaltyEngine.
        """
        if is_loss is None:
            is_loss = realized_pnl_usdt < 0
        self.realized_pnl_today_usdt += realized_pnl_usdt
        self.equity_usdt += realized_pnl_usdt
        if is_loss:
            self.daily_stoploss_hits += 1
            self.consecutive_losses[symbol] = (
                self.consecutive_losses.get(symbol, 0) + 1
            )
        else:
            self.consecutive_losses.pop(symbol, None)
        self._notify_change()

    def clear_consecutive_loss(self, symbol: str) -> None:
        """Drop the per-symbol consec-loss streak. Intended for paths
        that close a position without it being a loss but where
        ``record_pnl`` isn't appropriate (e.g. manual flatten).
        """
        if self.consecutive_losses.pop(symbol, None) is not None:
            self._notify_change()

    # ------------------------------------------------------------------ #
    # Operational patch — external equity adjustment (manual deposits /
    # withdrawals to bank).
    #
    # Why this exists: an operator who wires 5,000 USDT out of the
    # exchange to their bank does NOT generate a realised PnL event,
    # but the exchange-side equity drops by 5k. Two failure modes if
    # we don't model this:
    #   (1) ``equity_usdt`` (used by sizing) gradually de-syncs from
    #       the venue truth — every new entry is sized off a phantom
    #       balance that hasn't existed for hours.
    #   (2) ``daily_drawdown_pct`` = ``-realized_pnl / starting_equity``
    #       — once the daily reconciler corrects ``equity_usdt`` we'd
    #       otherwise be tempted to also re-derive ``starting_equity``
    #       to "undo" the drawdown, which is the wrong fix because it
    #       would mask a real losing streak.
    #
    # The right fix (this method): bring ``equity_usdt`` to the new
    # balance AND shift ``starting_equity_today_usdt`` by the SAME
    # delta so the *ratio* in ``daily_drawdown_pct`` stays unchanged.
    # That preserves the circuit breaker's meaning (same realised loss
    # in USDT but expressed as the same %-of-start) while letting
    # sizing operate against the truthful number.
    #
    # The ``WithdrawalDetector`` (in main.py wiring) is the one
    # legitimate caller in production. Tests can call it directly to
    # simulate any bank-flow scenario.
    # ------------------------------------------------------------------ #

    def adjust_equity_baseline(
        self,
        new_equity_usdt: float,
        *,
        reason: str = "external_balance_adjustment",
    ) -> float:
        """Reconcile local equity to a venue-observed balance without
        polluting the daily drawdown breaker.

        Returns the *delta* applied (positive = deposit, negative =
        withdrawal). Callers can log / notify on the magnitude.
        """
        if new_equity_usdt < 0:
            # Negative balance is nonsensical; ignore and log.
            logger.warning(
                "adjust_equity_baseline: refused negative new_equity=%.2f "
                "(reason=%s); leaving state untouched",
                new_equity_usdt, reason,
            )
            return 0.0

        delta = float(new_equity_usdt) - self.equity_usdt
        if delta == 0.0:
            return 0.0

        # Preserve the daily_drawdown ratio across the adjustment.
        # daily_drawdown_pct = -realized_pnl_today / starting_equity_today.
        # If we just shifted starting_equity by ``delta`` (additive), the
        # ratio would change because the numerator (realised PnL) is
        # USDT-absolute, not proportional. The right invariant is:
        # rescale starting_equity by the SAME multiplicative factor as
        # equity, so the ratio = -pnl / (starting * scale) cannot be
        # preserved with a single transform — we have to choose.
        #
        # Choice (and rationale): we scale starting_equity by the same
        # multiplicative factor as equity_usdt. This means a 50%
        # withdrawal halves both numbers; the next ``daily_drawdown_pct``
        # query then shows -pnl / halved_starting which is 2× the
        # previous value — that is the CORRECT semantic, because in
        # USDT terms a 200 USDT loss against a halved account IS a
        # bigger drawdown. The breaker should fire sooner if the
        # operator pulls capital while the day was already negative.
        #
        # An alternative ("preserve-the-ratio") would be additive on
        # both sides, but that masks the post-withdrawal capital
        # constraint: a 6% breaker against a 5,000 starting still
        # represents 300 USDT remaining buffer when 600 USDT have
        # already been lost — an inconsistent statement.
        old_equity = self.equity_usdt
        if old_equity > 0 and self.starting_equity_today_usdt > 0:
            scale = float(new_equity_usdt) / old_equity
            self.starting_equity_today_usdt = max(
                0.0, self.starting_equity_today_usdt * scale,
            )
        else:
            # Defensive: degenerate state, fall back to additive shift
            # so we don't divide by zero.
            self.starting_equity_today_usdt = max(
                0.0, self.starting_equity_today_usdt + delta,
            )

        self.equity_usdt = float(new_equity_usdt)

        logger.warning(
            "adjust_equity_baseline: %s delta=%+.2f USDT "
            "(equity %.2f, starting_today %.2f) reason=%s",
            "deposit" if delta > 0 else "withdrawal",
            delta, self.equity_usdt, self.starting_equity_today_usdt,
            reason,
        )
        self._notify_change()
        return delta

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
        # Phase B.1.3: a day-rollover wipes today's PnL counter and the
        # stoploss-hit counter; that is exactly the kind of mutation
        # we don't want to lose to a restart five minutes later.
        self._notify_change()
        return True
