"""
state.py — shared dataclasses for risk/execution layer.

Kept in a separate module to avoid circular imports between gate, sizing,
trailing, executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Local mirror of an exchange position."""

    symbol: str
    exchange: str
    side: Side
    entry_price: float
    size_contracts: float
    leverage: float
    opened_ts: int
    initial_stop: float
    # state of the trailing stop FSM
    current_hard_stop: float
    hard_stop_order_id: str | None = None
    realized_r: float = 0.0  # in R-multiples; useful for the FSM transitions
    closed: bool = False

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    @property
    def is_long(self) -> bool:
        return self.side == Side.LONG


@dataclass
class AccountState:
    """Snapshot of account-level state that the gate inspects."""

    equity_usdt: float
    realized_pnl_today_usdt: float = 0.0   # negative when in drawdown
    open_positions: list[Position] = field(default_factory=list)
    consecutive_losses_today: int = 0
    daily_stoploss_hits: int = 0
    circuit_breaker_engaged: bool = False
    reconciliation_complete: bool = False  # SR-2: gate refuses entries until True
    # symbol -> ms epoch when cooldown expires (e.g. failed hard-stop)
    symbol_cooldowns: dict[str, int] = field(default_factory=dict)
