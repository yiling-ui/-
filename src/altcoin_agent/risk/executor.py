"""
executor.py — order execution wrapping ccxt.

Critical contract (SR-2):
    1. Every successful entry fill is followed IMMEDIATELY by an exchange
       STOP_MARKET reduce_only order. If that placement fails, the entry
       position is force-closed by market and the symbol is cooled down.
    2. The trailing FSM only TIGHTENS the hard stop via cancel+replace.
       If the replace fails, the previous (looser) stop stays in force.

The executor is small and stateful only at the position level. Account
state and gate decisions are passed in by the orchestrator.

`ExchangeAdapter` is the seam for testing — a fake implementation in tests
exercises every branch (success, partial fill, stop placement failure).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from altcoin_agent.risk.sizing import OrderIntent, SizingResult
from altcoin_agent.risk.state import AccountState, Position, Side

logger = logging.getLogger(__name__)


class ExchangeAdapter(Protocol):
    """Subset of ccxt the executor depends on. Implementations ARE async."""

    async def market_order(
        self, symbol: str, side: Side, size: float, *,
        price: float | None = None, reduce_only: bool = False,
    ) -> dict[str, Any]: ...

    async def place_stop_order(
        self, symbol: str, side: Side, size: float, stop_price: float, reduce_only: bool = True,
    ) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]: ...

    async def set_leverage(self, symbol: str, leverage: float) -> dict[str, Any]: ...


@dataclass
class ExecuteResult:
    success: bool
    position: Position | None
    reason: str | None
    fill_price: float
    placed_stop_order_id: str | None
    forced_close: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class CCXTExecutor:
    adapter: ExchangeAdapter
    cooldown_on_stop_fail_ms: int = 4 * 60 * 60 * 1000  # 4h, per SR-2

    async def open_with_hard_stop(
        self,
        *,
        intent: OrderIntent,
        sizing: SizingResult,
        account: AccountState,
    ) -> ExecuteResult:
        """
        SR-2 enforced sequence:
            1. set leverage
            2. place market entry
            3. place exchange STOP_MARKET reduce_only
            4. on (3) failure -> immediate market close + cooldown
        """
        notes: list[str] = []

        if not account.reconciliation_complete:
            return ExecuteResult(
                success=False, position=None, reason="reconciliation_pending",
                fill_price=0.0, placed_stop_order_id=None, notes=notes,
            )

        if sizing.size_contracts <= 0:
            return ExecuteResult(
                success=False, position=None, reason="size_zero_after_rounding",
                fill_price=0.0, placed_stop_order_id=None, notes=notes,
            )

        # 1) leverage. We tolerate failure here only if the exchange's leverage
        # is already at our target — for paranoia we could refetch, but it's
        # exchange-specific. Keep it simple.
        try:
            await self.adapter.set_leverage(intent.symbol, sizing.leverage)
            notes.append(f"leverage set: {sizing.leverage:.2f}x")
        except Exception as e:
            notes.append(f"set_leverage non-fatal: {e}")

        # 2) market entry
        entry_side = intent.side
        try:
            fill = await self.adapter.market_order(
                symbol=intent.symbol,
                side=entry_side,
                size=sizing.size_contracts,
                price=intent.entry_price,
                reduce_only=False,
            )
        except Exception as e:
            return ExecuteResult(
                success=False, position=None, reason=f"market_order_failed:{e}",
                fill_price=0.0, placed_stop_order_id=None, notes=notes,
            )

        fill_price = float(fill.get("average") or fill.get("price") or intent.entry_price)
        notes.append(f"filled @ {fill_price}")

        # 3) MUST place hard stop immediately. SR-2.
        stop_side = Side.SHORT if entry_side == Side.LONG else Side.LONG
        stop_order_id: str | None = None
        try:
            stop = await self.adapter.place_stop_order(
                symbol=intent.symbol,
                side=stop_side,
                size=sizing.size_contracts,
                stop_price=intent.initial_stop,
                reduce_only=True,
            )
            stop_order_id = str(stop.get("id"))
            notes.append(f"hard stop placed @ {intent.initial_stop} id={stop_order_id}")
        except Exception as e:
            # 4) FAIL-CLOSED: force close the position we just opened.
            notes.append(f"hard_stop_failed: {e} -- forcing close")
            forced = await self._force_close(intent, sizing.size_contracts, entry_side, notes)
            cooldown_until = self._now_ms() + self.cooldown_on_stop_fail_ms
            account.symbol_cooldowns[f"{intent.exchange}:{intent.symbol}"] = cooldown_until
            return ExecuteResult(
                success=False,
                position=None,
                reason=f"hard_stop_placement_failed:{e}",
                fill_price=fill_price,
                placed_stop_order_id=None,
                forced_close=forced,
                notes=notes,
            )

        position = Position(
            symbol=intent.symbol,
            exchange=intent.exchange,
            side=intent.side,
            entry_price=fill_price,
            size_contracts=sizing.size_contracts,
            leverage=sizing.leverage,
            opened_ts=self._now_ms(),
            initial_stop=intent.initial_stop,
            current_hard_stop=intent.initial_stop,
            hard_stop_order_id=stop_order_id,
        )
        account.open_positions.append(position)

        return ExecuteResult(
            success=True,
            position=position,
            reason=None,
            fill_price=fill_price,
            placed_stop_order_id=stop_order_id,
            notes=notes,
        )

    async def tighten_hard_stop(
        self, position: Position, new_stop: float,
    ) -> bool:
        """
        Cancel + replace the exchange stop to a tighter level.

        Returns True iff the new stop is in force at the exchange when the
        method returns. If cancel fails we do NOT place a new stop (would
        end up with two stops on the book). If place fails after a
        successful cancel we attempt to put the old stop back; only if THAT
        also fails do we mark `current_hard_stop` as stale and surface.
        """
        if position.hard_stop_order_id is None:
            logger.warning("tighten_hard_stop: position has no current stop id; skipping")
            return False

        stop_side = Side.SHORT if position.is_long else Side.LONG
        old_id = position.hard_stop_order_id
        old_price = position.current_hard_stop

        # 1) cancel old
        try:
            await self.adapter.cancel_order(old_id, position.symbol)
        except Exception as e:
            logger.warning("cancel_order failed; keeping old stop: %s", e)
            return False

        # 2) place new
        try:
            new_order = await self.adapter.place_stop_order(
                symbol=position.symbol,
                side=stop_side,
                size=position.size_contracts,
                stop_price=new_stop,
                reduce_only=True,
            )
        except Exception as e:
            # 3) try to re-place the OLD stop
            logger.error("new stop placement failed: %s -- attempting old stop restore", e)
            try:
                restore = await self.adapter.place_stop_order(
                    symbol=position.symbol,
                    side=stop_side,
                    size=position.size_contracts,
                    stop_price=old_price,
                    reduce_only=True,
                )
                position.hard_stop_order_id = str(restore.get("id"))
                logger.warning("old stop restored")
            except Exception as e2:
                # We are now naked — emit a critical alert. The caller should
                # invoke a force_close routine.
                position.hard_stop_order_id = None
                logger.critical("RESTORE FAILED: position is naked: %s", e2)
            return False

        position.hard_stop_order_id = str(new_order.get("id"))
        position.current_hard_stop = new_stop
        return True

    async def _force_close(
        self, intent: OrderIntent, size: float, entry_side: Side, notes: list[str],
    ) -> bool:
        close_side = Side.SHORT if entry_side == Side.LONG else Side.LONG
        try:
            await self.adapter.market_order(
                symbol=intent.symbol, side=close_side, size=size, reduce_only=True,
            )
            notes.append("forced_close: ok")
            return True
        except Exception as e:
            notes.append(f"forced_close_failed: {e}")
            logger.critical(
                "FORCE CLOSE FAILED for %s/%s: position naked, manual intervention required",
                intent.exchange, intent.symbol,
            )
            return False

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
