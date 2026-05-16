"""executor.py — order placement with hard-stop pairing (SR-2 fail-closed).

Contract:
    Every entry call MUST end in one of two states:
      (a) entry filled AND exchange-side STOP_MARKET in place.
      (b) entry filled BUT stop placement failed -> immediate market close
          + symbol cooldown + critical alert.

Soft stops (in-process Python timers, etc.) are explicitly not allowed.

Rolling positions:
    ``add_leg`` extends an existing position with a new same-side market
    order, then replaces the single resting STOP_MARKET so its size
    reflects the new aggregate ``total_size``. The same fail-closed
    posture as ``open`` applies: if we cannot resize the stop, we
    EMERGENCY-CLOSE the entire position (legs are inseparable on the
    venue), set the symbol cooldown, and raise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.state import AccountState, Position, PositionLeg, Side

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
    # Audit #16: minimum acceptable fill ratio. Below this we treat
    # the entry as a failed market order and emergency-close whatever
    # did fill. 0.95 is conservative enough to tolerate normal
    # rounding/lot-size truncation but tight enough to catch a real
    # IOC partial fill on a thin book.
    min_fill_ratio: float = 0.95

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

        # Audit #16: partial-fill detection. ccxt returns ``filled`` on
        # most venues; when the venue (or our adapter) does not, fall
        # back to the order's ``amount`` so legacy adapters preserve
        # current behaviour. A fill below ``min_fill_ratio`` of the
        # requested size is treated as a failure: we emergency-close
        # whatever did fill (using the actually-filled qty as the
        # reduce_only size), set a cooldown, and raise. This keeps
        # the position book honest in altcoin scenarios where the
        # IOC market order can wipe one level and stop.
        try:
            filled_raw = entry_resp.get("filled")
            if filled_raw is None:
                # Fall back to the response's "amount" (full original
                # size when missing -> ratio == 1.0).
                filled_raw = entry_resp.get("amount", decision.size)
            filled = abs(float(filled_raw))
        except (TypeError, ValueError):
            filled = float(decision.size)

        fill_ratio = (
            filled / decision.size if decision.size > 0 else 0.0
        )
        if fill_ratio < self.min_fill_ratio:
            logger.critical(
                "Partial fill on %s: requested=%.6f filled=%.6f "
                "ratio=%.4f < %.4f — emergency closing the partial leg",
                symbol, decision.size, filled, fill_ratio,
                self.min_fill_ratio,
            )
            if filled > 0:
                try:
                    await self.adapter.market_order(
                        symbol=symbol,
                        side=decision.side.opposite,
                        size=filled,
                        price=current_price,
                        reduce_only=True,
                    )
                except Exception as e:
                    logger.critical(
                        "EMERGENCY CLOSE of partial fill failed for %s: %s "
                        "— manual intervention required", symbol, e,
                    )
            account.set_cooldown(
                symbol, self.stop_failure_cooldown_sec, now_ms_default(),
            )
            raise ExecutionError(
                f"partial_fill_below_threshold:{fill_ratio:.4f}",
            )

        # Audit (third pass) #3: when fill_ratio is in [min_fill_ratio, 1.0)
        # the position is acceptable but the **actual** size on the venue is
        # ``filled``, not ``decision.size``. Using decision.size for the
        # STOP_MARKET reduce_only order would either be auto-clamped (Binance)
        # or rejected outright (Bybit/Gate) at trigger time. Worse, the local
        # ``Position.size`` would lie about the true exposure, throwing off
        # PnL math, leverage cap re-checks, and add_leg's stop resize.
        # Pin actual_size to the venue truth and use it everywhere downstream.
        actual_size = filled if filled > 0 else float(decision.size)

        # 3) hard stop on the exchange — RETRY then fail-closed close.
        stop_side = decision.side.opposite
        stop_resp: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(self.place_stop_retries + 1):
            try:
                stop_resp = await self.adapter.place_stop_order(
                    symbol=symbol,
                    side=stop_side,
                    size=actual_size,
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
                    size=actual_size,
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

        # 4) record the position — using actual_size so the local book
        # reflects venue truth.
        pos = Position(
            symbol=symbol,
            exchange=self.exchange_name,
            side=decision.side,
            entry_price=avg_price,
            size=actual_size,
            leverage=decision.leverage,
            initial_stop=decision.initial_stop,
            current_stop=decision.initial_stop,
            stop_order_id=str(stop_resp.get("id") or ""),
            trace_id=trace_id,
        )
        # Rolling-positions bookkeeping: leg 0 is the original entry.
        # Subsequent ``add_leg`` calls append; trailing/sizing always
        # reads from ``legs`` when present (``total_size`` /
        # ``avg_entry_price`` fall back to the legacy fields when
        # ``legs`` is empty, so existing code paths remain identical).
        pos.legs.append(PositionLeg(
            leg_id=0, side=decision.side, size=actual_size,
            entry_price=avg_price, margin_source="initial",
        ))
        account.open_positions[symbol] = pos
        return pos

    async def add_leg(
        self,
        *,
        position: Position,
        size: float,
        current_price: float,
        new_stop_for_full: float | None = None,
        account: AccountState | None = None,
        trigger_score: float | None = None,
    ) -> PositionLeg:
        """Append a new same-side leg to an open position.

        Steps (parallel to ``open``):
          1. market_order(side=position.side, size=new_leg_size).
          2. Resize the resting STOP_MARKET to cover ``total_size``. If
             ``new_stop_for_full`` is provided, also tighten to that price
             (used when trailing has already moved past the original stop).
             Otherwise keep ``position.current_stop`` and only resize.
          3. On stop-replacement failure: emergency-close the ENTIRE
             position (single venue-side stop covers all legs; we cannot
             leave a partially-protected position).

        The returned PositionLeg is also appended to ``position.legs``,
        and ``position.size`` (the legacy field) is updated to the new
        ``total_size`` so downstream code that hasn't been migrated to
        the legs API still sees the right aggregate.

        Raises ``ExecutionError`` on any unrecoverable failure. The
        caller is expected to log + notify; the position has already
        been emergency-closed in that case.
        """
        if position.closed:
            raise ExecutionError("add_leg called on closed position")
        if size <= 0:
            raise ExecutionError(f"add_leg with non-positive size: {size}")
        if account is None:
            raise ExecutionError("add_leg requires account for cooldown")

        old_size = position.total_size

        # 1) market order on the SAME side as the existing position.
        entry_resp = await self.adapter.market_order(
            symbol=position.symbol,
            side=position.side,
            size=size,
            price=current_price,
            reduce_only=False,
        )
        avg_price = float(
            entry_resp.get("average") or entry_resp.get("price") or current_price
        )

        # Audit (third pass) #3: pin actual_leg_size to venue truth.
        # ``add_leg`` historically used the requested ``size``; if the
        # IOC market order partially filled (common in thin altcoin
        # books) the legacy code resized the venue-side stop to cover
        # phantom contracts, eventually triggering reduce_only rejections
        # and an emergency-close of the entire (winning) position. We
        # now read the real ``filled`` and propagate it everywhere
        # downstream (PositionLeg.size, position.size, stop_resize size).
        try:
            filled_raw = entry_resp.get("filled")
            if filled_raw is None:
                filled_raw = entry_resp.get("amount", size)
            filled = abs(float(filled_raw))
        except (TypeError, ValueError):
            filled = float(size)
        actual_leg_size = filled if filled > 0 else float(size)
        if actual_leg_size < size * self.min_fill_ratio:
            # Severe partial fill on the leg — same fail-closed posture
            # as ``open``: close just the partial leg and bail. The
            # caller (RollingController) will see the ExecutionError
            # and may auto-disable rolling.
            logger.critical(
                "add_leg partial fill on %s: requested=%.6f filled=%.6f "
                "ratio=%.4f — closing partial leg only",
                position.symbol, size, actual_leg_size,
                actual_leg_size / max(size, 1e-9),
            )
            try:
                await self.adapter.market_order(
                    symbol=position.symbol,
                    side=position.side.opposite,
                    size=actual_leg_size,
                    price=current_price,
                    reduce_only=True,
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE of partial add_leg fill failed for "
                    "%s: %s — manual intervention required",
                    position.symbol, e,
                )
            account.set_cooldown(
                position.symbol, self.stop_failure_cooldown_sec, now_ms_default(),
            )
            raise ExecutionError(
                f"add_leg_partial_fill:{actual_leg_size / max(size, 1e-9):.4f}",
            )

        # 2) Replace the resting stop so it covers the new aggregate size.
        # We piggyback on tighten_hard_stop's cancel+place+restore-on-failure
        # semantics, but pass through the EXISTING current_stop unless the
        # caller asked for a different one. The new stop's size (which the
        # venue actually cares about) is read from ``position.size``, so
        # we update that BEFORE the call.
        next_leg_id = (max((L.leg_id for L in position.legs), default=-1) + 1)
        leg = PositionLeg(
            leg_id=next_leg_id,
            side=position.side,
            size=actual_leg_size,
            entry_price=avg_price,
            margin_source="rolled_unrealized",
            trigger_score=trigger_score,
        )
        position.legs.append(leg)
        # Keep the legacy ``size`` field in sync. The trailing FSM and
        # stop-placement code path read ``position.size`` directly.
        position.size = old_size + actual_leg_size

        target_stop = (
            new_stop_for_full
            if new_stop_for_full is not None
            else position.current_stop
        )
        ok = await self.tighten_hard_stop(position, target_stop)
        if not ok:
            # CRITICAL: a leg is in but stop is now smaller than total
            # exposure (or completely missing). Single venue-side stop
            # cannot protect a partial position; emergency-close the
            # whole thing.
            logger.critical(
                "add_leg: stop resize FAILED for %s — emergency-closing all "
                "%d legs (total_size=%.6f)",
                position.symbol, len(position.legs), position.size,
            )
            try:
                await self.adapter.market_order(
                    symbol=position.symbol,
                    side=position.side.opposite,
                    size=position.size,
                    price=current_price,
                    reduce_only=True,
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE on add_leg also failed for %s: %s — "
                    "manual intervention required",
                    position.symbol, e,
                )
            account.set_cooldown(
                position.symbol, self.stop_failure_cooldown_sec, now_ms_default(),
            )
            position.closed = True
            raise ExecutionError(
                f"add_leg_stop_resize_failed:{position.symbol}",
            )
        return leg

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
