"""executor.py — order placement with hard-stop pairing (SR-2 fail-closed).

Contract:
    Every entry call MUST end in one of two states:
      (a) entry filled AND exchange-side STOP_MARKET in place.
      (b) entry filled BUT stop placement failed -> immediate market close
          + symbol cooldown + critical alert.

Soft stops (in-process Python timers, etc.) are explicitly not allowed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.state import AccountState, Position, Side

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    pass


@runtime_checkable
class ExchangeAdapter(Protocol):
    """Minimal interface the executor needs. ccxt-compatible by design."""

    async def market_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]: ...

    async def place_stop_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]: ...

    async def set_leverage(self, symbol: str, leverage: float) -> dict[str, Any]: ...

    async def fetch_positions(self) -> list[dict[str, Any]]: ...

    async def fetch_open_orders(self) -> list[dict[str, Any]]: ...


@dataclass
class CCXTExecutor:
    """ccxt-backed executor (works with any adapter satisfying the protocol)."""

    adapter: ExchangeAdapter
    exchange_name: str = "binance"
    place_stop_retries: int = 2
    stop_failure_cooldown_sec: int = 4 * 3600

    async def open(
        self,
        *,
        symbol: str,
        decision: RiskDecision,
        current_price: float,
        account: AccountState,
        trace_id: str | None = None,
    ) -> Position:
        if not decision.approved:
            raise ExecutionError(f"open called on rejected decision: {decision.reason}")
        if decision.side is None or decision.size is None or decision.leverage is None:
            raise ExecutionError("decision missing side/size/leverage")
        if decision.initial_stop is None:
            raise ExecutionError("decision missing initial_stop")

        # 1) leverage
        await self.adapter.set_leverage(symbol, decision.leverage)

        # 2) market entry
        entry_resp = await self.adapter.market_order(
            symbol=symbol,
            side=decision.side,
            size=decision.size,
            price=current_price,
            reduce_only=False,
        )
        avg_price = float(
            entry_resp.get("average") or entry_resp.get("price") or current_price
        )

        # 3) hard stop on the exchange — RETRY then fail-closed close.
        stop_side = decision.side.opposite
        stop_resp: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(self.place_stop_retries + 1):
            try:
                stop_resp = await self.adapter.place_stop_order(
                    symbol=symbol,
                    side=stop_side,
                    size=decision.size,
                    stop_price=decision.initial_stop,
                    reduce_only=True,
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(
                    "stop placement attempt %d failed: %s", attempt + 1, e,
                )
                await asyncio.sleep(0.5 * (2 ** attempt))

        if stop_resp is None:
            # CRITICAL: we have an open exposure with no hard stop.
            logger.critical(
                "STOP placement failed for %s after %d attempts (%s) — closing",
                symbol, self.place_stop_retries + 1, last_err,
            )
            try:
                await self.adapter.market_order(
                    symbol=symbol,
                    side=stop_side,
                    size=decision.size,
                    price=current_price,
                    reduce_only=True,
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE also failed for %s: %s — manual intervention required",
                    symbol, e,
                )
            account.set_cooldown(symbol, self.stop_failure_cooldown_sec, now_ms_default())
            raise ExecutionError(f"stop_placement_failed:{last_err}")

        # 4) record the position
        pos = Position(
            symbol=symbol,
            exchange=self.exchange_name,
            side=decision.side,
            entry_price=avg_price,
            size=decision.size,
            leverage=decision.leverage,
            initial_stop=decision.initial_stop,
            current_stop=decision.initial_stop,
            stop_order_id=str(stop_resp.get("id") or ""),
            trace_id=trace_id,
        )
        account.open_positions[symbol] = pos
        return pos

    async def tighten_hard_stop(
        self,
        position: Position,
        new_stop: float,
    ) -> bool:
        """Cancel the existing stop, place a tighter one. On failure, attempt
        to restore the OLD stop. Only if BOTH replace AND restore fail do we
        give up and let the caller decide (typical response: emergency close).
        Returns True iff the new stop is now resting.
        """
        old_id = position.stop_order_id
        old_stop = position.current_stop
        stop_side = position.side.opposite

        try:
            if old_id:
                await self.adapter.cancel_order(old_id, position.symbol)
        except Exception as e:
            logger.warning("cancel of old stop %s failed: %s — trying replace anyway",
                           old_id, e)

        try:
            new_resp = await self.adapter.place_stop_order(
                symbol=position.symbol,
                side=stop_side,
                size=position.size,
                stop_price=new_stop,
                reduce_only=True,
            )
            position.current_stop = new_stop
            position.stop_order_id = str(new_resp.get("id") or "")
            return True
        except Exception as e:
            logger.error("replace stop failed for %s @ %s: %s",
                         position.symbol, new_stop, e)
            # Try to restore the old stop so the position isn't naked.
            try:
                restored = await self.adapter.place_stop_order(
                    symbol=position.symbol,
                    side=stop_side,
                    size=position.size,
                    stop_price=old_stop,
                    reduce_only=True,
                )
                position.stop_order_id = str(restored.get("id") or "")
                logger.warning("restored old stop @ %s on %s after replace failure",
                               old_stop, position.symbol)
                return False
            except Exception as e2:
                logger.critical(
                    "RESTORE old stop also failed on %s: %s — POSITION IS NAKED",
                    position.symbol, e2,
                )
                position.stop_order_id = None
                return False


def now_ms_default() -> int:
    import time
    return int(time.time() * 1000)
