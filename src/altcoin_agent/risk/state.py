"""state.py — shared dataclasses for the risk + execution package."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def opposite(self) -> Side:
        return Side.SHORT if self == Side.LONG else Side.LONG


@dataclass
class Position:
    """A live position with all the bookkeeping the trailing FSM needs."""

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

    @property
    def r_unit(self) -> float:
        """1R distance — always positive."""
        return abs(self.entry_price - self.initial_stop)


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
