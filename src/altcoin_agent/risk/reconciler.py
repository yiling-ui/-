"""
reconciler.py — startup state alignment (SR-2).

The very first action of the executor's lifecycle is to compare the
exchange's view of positions/orders to our local state and surface
inconsistencies. Until the reconciler reports success, the gate refuses
all new entries.

Behaviour:
    - Pull positions from every exchange.
    - Diff vs `account.open_positions`.
    - Anything on the exchange but not local -> ORPHAN.
        Default action: place a breakeven stop on the exchange and emit
        a human-actionable alert. The orphan is NOT auto-closed.
    - Cancel any open orders that have no local record.

The exchange interaction is encapsulated behind a small Protocol so the
real implementation (ccxt) and tests (in-memory fake) share the same code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from altcoin_agent.risk.state import AccountState, Side

logger = logging.getLogger(__name__)


class ExchangeReconcileAdapter(Protocol):
    async def fetch_positions(self) -> list[dict[str, Any]]: ...
    async def fetch_open_orders(self) -> list[dict[str, Any]]: ...
    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]: ...
    async def place_stop_order(
        self, symbol: str, side: Side, size: float, stop_price: float, reduce_only: bool = True,
    ) -> dict[str, Any]: ...


@dataclass
class OrphanPosition:
    exchange: str
    symbol: str
    side: Side
    size_contracts: float
    entry_price: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconcilerReport:
    success: bool
    orphans: list[OrphanPosition] = field(default_factory=list)
    cancelled_orders: list[str] = field(default_factory=list)
    stops_placed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Reconciler:
    """One reconciler instance per exchange."""

    exchange_name: str
    adapter: ExchangeReconcileAdapter

    async def run(
        self,
        account: AccountState,
        breakeven_stop_buffer_pct: float = 0.005,  # 0.5% pad to avoid immediate trigger
    ) -> ReconcilerReport:
        report = ReconcilerReport(success=True)

        # 1) positions
        try:
            ex_positions = await self.adapter.fetch_positions()
        except Exception as e:
            report.success = False
            report.errors.append(f"fetch_positions: {e}")
            return report

        local_keys = {(p.exchange, p.symbol) for p in account.open_positions if not p.closed}
        for raw in ex_positions:
            symbol = str(raw.get("symbol"))
            size = float(raw.get("contracts") or raw.get("size") or 0.0)
            if size == 0:
                continue
            side = Side.LONG if size > 0 else Side.SHORT
            entry = float(raw.get("entryPrice") or raw.get("entry_price") or 0.0)

            if (self.exchange_name, symbol) in local_keys:
                continue  # known position, no orphan

            orphan = OrphanPosition(
                exchange=self.exchange_name,
                symbol=symbol,
                side=side,
                size_contracts=abs(size),
                entry_price=entry,
                raw=raw,
            )
            report.orphans.append(orphan)
            logger.warning("ORPHAN POSITION detected: %s %s size=%s @ %s",
                           self.exchange_name, symbol, size, entry)

            # Default action: place breakeven stop on the exchange. Better to
            # have a stop than no stop. Buffer so we don't trigger immediately
            # on the next tick.
            buffer = entry * breakeven_stop_buffer_pct
            stop_price = entry - buffer if side == Side.LONG else entry + buffer
            try:
                order = await self.adapter.place_stop_order(
                    symbol=symbol,
                    side=Side.SHORT if side == Side.LONG else Side.LONG,
                    size=abs(size),
                    stop_price=stop_price,
                    reduce_only=True,
                )
                report.stops_placed.append(str(order.get("id")))
            except Exception as e:
                report.errors.append(f"place_breakeven_stop({symbol}): {e}")
                # Even if we cant place a stop here, we don't fail the whole
                # reconciler — the alert above is the human-actionable signal.

        # 2) orphan orders
        try:
            ex_orders = await self.adapter.fetch_open_orders()
        except Exception as e:
            report.errors.append(f"fetch_open_orders: {e}")
            return report

        # Anything not associated with a current position -> cancel.
        # (We don't track order ids locally yet; we are conservative and
        # only cancel orders explicitly tagged as "stale" by the adapter.)
        for raw in ex_orders:
            order_id = str(raw.get("id"))
            symbol = str(raw.get("symbol"))
            tag = str(raw.get("clientOrderId") or "")
            if tag.startswith("stale-"):
                try:
                    await self.adapter.cancel_order(order_id, symbol)
                    report.cancelled_orders.append(order_id)
                except Exception as e:
                    report.errors.append(f"cancel_order({order_id}): {e}")

        # The reconciler succeeds even when orphans exist — the gate stays
        # closed only if `success=False`. Orphans surface via the alert path.
        return report
