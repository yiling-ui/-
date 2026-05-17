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

TICKET-001 (clientOrderId idempotency):
    Every market entry and every STOP_MARKET carries a venue-side
    ``clientOrderId`` that we generate locally and persist on
    ``Position.client_order_id`` / ``Position.stop_client_order_id``.
    Network retries inside ``CCXTExchangeAdapter`` reuse the same cid
    so a duplicate-create attempt is rejected by the venue (or returns
    the original order). The adapter additionally calls
    ``fetch_order(client_order_id=...)`` between retries to short-circuit
    "did the previous request actually land?" — see ``risk/retry.py``.

TICKET-002 (partial-fill detection on the real path):
    The adapter's ``_normalize_order`` now exposes ``amount`` /
    ``filled`` / ``remaining`` / ``status`` so the
    ``min_fill_ratio`` check actually runs against venue truth on
    every code path (legacy mocks that returned a flat dict are still
    accepted via the ``decision.size`` fallback).

TICKET-005 (retry classification):
    Lives in the adapter; the executor sees clean success / clean
    failure. The executor's own per-call retry loop is therefore
    GONE — keeping it would double-retry transient errors and make
    the wall-clock budget unbounded.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.state import AccountState, Position, PositionLeg, Side

logger = logging.getLogger(__name__)


# TICKET-001: cid generator. Format: ``alt`` + 23 hex chars = 26 chars
# total. Starts with a letter (Binance + OKX requirement), uses only
# alphanumerics (Gate.io's ``text`` field allows ``_-.`` too but we
# don't need them and it keeps the cid uniformly safe across venues).
_CID_PREFIX = "alt"
_CID_HEX_LEN = 23


def _new_client_order_id(prefix: str = _CID_PREFIX) -> str:
    return f"{prefix}{uuid.uuid4().hex[:_CID_HEX_LEN]}"


# TICKET-004: the executor signals close hints to the position-watcher
# so the close handler can later attribute the close to the right
# bucket (stop_filled / liquidation / manual_close / emergency_close).
# Default reason "exchange_close_detected" stays in place for the
# common "STOP_MARKET filled, watcher noticed it" path so existing
# tests / callers don't need to change.
EmergencyCloseHint = Callable[[str, str], None]


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
        client_order_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def place_stop_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]: ...

    async def set_leverage(self, symbol: str, leverage: float) -> dict[str, Any]: ...

    async def fetch_positions(self) -> list[dict[str, Any]]: ...

    async def fetch_open_orders(self) -> list[dict[str, Any]]: ...


@dataclass
class CCXTExecutor:
    """ccxt-backed executor (works with any adapter satisfying the protocol).

    Args:
        adapter: a ccxt-style adapter. Retry policy lives there
            (TICKET-005); the executor sees clean success / failure.
        exchange_name: the venue label persisted on ``Position.exchange``.
        place_stop_retries: deprecated; kept for back-compat with tests
            that pass it. Real retry is in the adapter now. Default 0.
        stop_failure_cooldown_sec: seconds the symbol stays in cooldown
            after a stop-placement failure or partial-fill emergency-close.
        min_fill_ratio: TICKET-002. Lower bound on (filled / requested)
            below which we treat the entry as failed and emergency-close
            whatever did fill. 0.95 tolerates normal lot-size truncation.
        on_emergency_close: TICKET-004 hook. Called as
            ``on_emergency_close(symbol, reason)`` after an executor-side
            emergency close so the PositionWatcher can attribute the
            subsequent disappear-from-exchange event correctly. Optional.
    """

    adapter: ExchangeAdapter
    exchange_name: str = "binance"
    place_stop_retries: int = 0       # legacy; retry is in the adapter now
    stop_failure_cooldown_sec: int = 4 * 3600
    min_fill_ratio: float = 0.95
    on_emergency_close: EmergencyCloseHint | None = field(default=None)

    # ------------------- public API ------------------- #

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

        # TICKET-001: cid we'll thread through both the entry market
        # order and the resting stop. Different cids for the two
        # orders so the adapter's idempotency probe can disambiguate.
        entry_cid = _new_client_order_id()
        stop_cid = _new_client_order_id()

        # 1) leverage — adapter handles retries; transient errors come
        # back as exceptions only after the policy has given up.
        await self.adapter.set_leverage(symbol, decision.leverage)

        # 2) market entry
        entry_resp = await self.adapter.market_order(
            symbol=symbol,
            side=decision.side,
            size=decision.size,
            price=current_price,
            reduce_only=False,
            client_order_id=entry_cid,
        )
        avg_price = float(
            entry_resp.get("average") or entry_resp.get("price") or current_price
        )

        # TICKET-002: status + fill ratio -- two independent gates.
        # Status check first because a "canceled" / "rejected" entry has
        # an unambiguous answer (zero fill, no recovery needed). The
        # ratio check is the second line: status is silent on some
        # venues (it's "" until the order is moved out of the book) so
        # we still need fill_ratio for the partial-on-thin-book case.
        status = (entry_resp.get("status") or "").lower()
        if status in ("canceled", "rejected", "expired"):
            logger.critical(
                "Entry %s status=%s on %s — order did not land; aborting open",
                entry_cid, status, symbol,
            )
            account.set_cooldown(
                symbol, self.stop_failure_cooldown_sec, _now_ms(),
            )
            raise ExecutionError(f"entry_not_filled:{status}")

        filled = self._extract_filled_qty(entry_resp, decision.size)
        fill_ratio = filled / decision.size if decision.size > 0 else 0.0

        if fill_ratio < self.min_fill_ratio:
            logger.critical(
                "Partial fill on %s (cid=%s): requested=%.6f filled=%.6f "
                "ratio=%.4f < %.4f — emergency closing the partial leg",
                symbol, entry_cid, decision.size, filled, fill_ratio,
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
                        client_order_id=_new_client_order_id(),
                    )
                except Exception as e:
                    logger.critical(
                        "EMERGENCY CLOSE of partial fill failed for %s: %s "
                        "— manual intervention required", symbol, e,
                    )
            self._emit_close_hint(symbol, "emergency_close_partial_fill")
            account.set_cooldown(
                symbol, self.stop_failure_cooldown_sec, _now_ms(),
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
        actual_size = filled if filled > 0 else float(decision.size)

        # 3) hard stop on the exchange. Adapter handles transient retries.
        # We read the resp's cid back rather than blindly trusting our
        # generated value — some venues echo a transformed cid (gate.io
        # may strip its "t-" prefix on the way back) and we want to
        # persist the canonical form for later fetch_order lookups.
        stop_side = decision.side.opposite
        try:
            stop_resp = await self.adapter.place_stop_order(
                symbol=symbol,
                side=stop_side,
                size=actual_size,
                stop_price=decision.initial_stop,
                reduce_only=True,
                client_order_id=stop_cid,
            )
        except Exception as e:
            logger.critical(
                "STOP placement failed for %s after retry budget (%s) — emergency-closing",
                symbol, e,
            )
            try:
                await self.adapter.market_order(
                    symbol=symbol,
                    side=stop_side,
                    size=actual_size,
                    price=current_price,
                    reduce_only=True,
                    client_order_id=_new_client_order_id(),
                )
            except Exception as e2:
                logger.critical(
                    "EMERGENCY CLOSE also failed for %s: %s — manual intervention required",
                    symbol, e2,
                )
            self._emit_close_hint(symbol, "emergency_close_stop_failed")
            account.set_cooldown(symbol, self.stop_failure_cooldown_sec, _now_ms())
            raise ExecutionError(f"stop_placement_failed:{e}") from e

        # 4) record the position — using actual_size so the local book
        # reflects venue truth; cid persisted so fetch_order can find
        # the stop on a later restart / tighten chain.
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
            client_order_id=str(
                entry_resp.get("client_order_id") or entry_cid
            ),
            stop_client_order_id=str(
                stop_resp.get("client_order_id") or stop_cid
            ),
            trace_id=trace_id,
        )
        # Rolling-positions bookkeeping: leg 0 is the original entry.
        pos.legs.append(PositionLeg(
            leg_id=0, side=decision.side, size=actual_size,
            entry_price=avg_price, margin_source="initial",
            client_order_id=pos.client_order_id,
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
        """
        if position.closed:
            raise ExecutionError("add_leg called on closed position")
        if size <= 0:
            raise ExecutionError(f"add_leg with non-positive size: {size}")
        if account is None:
            raise ExecutionError("add_leg requires account for cooldown")

        old_size = position.total_size
        leg_cid = _new_client_order_id()

        # 1) market order on the SAME side as the existing position.
        entry_resp = await self.adapter.market_order(
            symbol=position.symbol,
            side=position.side,
            size=size,
            price=current_price,
            reduce_only=False,
            client_order_id=leg_cid,
        )
        avg_price = float(
            entry_resp.get("average") or entry_resp.get("price") or current_price
        )

        # TICKET-002 status guard.
        leg_status = (entry_resp.get("status") or "").lower()
        if leg_status in ("canceled", "rejected", "expired"):
            logger.critical(
                "add_leg %s on %s status=%s — leg did not land",
                leg_cid, position.symbol, leg_status,
            )
            account.set_cooldown(
                position.symbol, self.stop_failure_cooldown_sec, _now_ms(),
            )
            raise ExecutionError(f"add_leg_not_filled:{leg_status}")

        filled = self._extract_filled_qty(entry_resp, size)
        actual_leg_size = filled if filled > 0 else float(size)
        if actual_leg_size < size * self.min_fill_ratio:
            logger.critical(
                "add_leg partial fill on %s (cid=%s): requested=%.6f "
                "filled=%.6f ratio=%.4f — closing partial leg only",
                position.symbol, leg_cid, size, actual_leg_size,
                actual_leg_size / max(size, 1e-9),
            )
            try:
                await self.adapter.market_order(
                    symbol=position.symbol,
                    side=position.side.opposite,
                    size=actual_leg_size,
                    price=current_price,
                    reduce_only=True,
                    client_order_id=_new_client_order_id(),
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE of partial add_leg fill failed for "
                    "%s: %s — manual intervention required",
                    position.symbol, e,
                )
            account.set_cooldown(
                position.symbol, self.stop_failure_cooldown_sec, _now_ms(),
            )
            raise ExecutionError(
                f"add_leg_partial_fill:{actual_leg_size / max(size, 1e-9):.4f}",
            )

        next_leg_id = (max((L.leg_id for L in position.legs), default=-1) + 1)
        leg = PositionLeg(
            leg_id=next_leg_id,
            side=position.side,
            size=actual_leg_size,
            entry_price=avg_price,
            margin_source="rolled_unrealized",
            trigger_score=trigger_score,
            client_order_id=str(
                entry_resp.get("client_order_id") or leg_cid
            ),
        )
        position.legs.append(leg)
        position.size = old_size + actual_leg_size

        target_stop = (
            new_stop_for_full
            if new_stop_for_full is not None
            else position.current_stop
        )
        ok = await self.tighten_hard_stop(position, target_stop)
        if not ok:
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
                    client_order_id=_new_client_order_id(),
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE on add_leg also failed for %s: %s — "
                    "manual intervention required",
                    position.symbol, e,
                )
            self._emit_close_hint(
                position.symbol, "emergency_close_add_leg_stop_resize_failed",
            )
            account.set_cooldown(
                position.symbol, self.stop_failure_cooldown_sec, _now_ms(),
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
        new_stop_cid = _new_client_order_id()

        # The cancel is best-effort; the adapter has its own retry
        # budget for transient errors. A genuine fatal (the stop was
        # already filled, or the venue rejected the cancel) will
        # surface here and we still try to place the new one because
        # the resting stop being absent is a much worse failure mode
        # than a duplicate.
        try:
            if old_id:
                await self.adapter.cancel_order(old_id, position.symbol)
        except Exception as e:
            logger.warning(
                "cancel of old stop %s failed: %s — trying replace anyway",
                old_id, e,
            )

        try:
            new_resp = await self.adapter.place_stop_order(
                symbol=position.symbol,
                side=stop_side,
                size=position.size,
                stop_price=new_stop,
                reduce_only=True,
                client_order_id=new_stop_cid,
            )
            position.current_stop = new_stop
            position.stop_order_id = str(new_resp.get("id") or "")
            position.stop_client_order_id = str(
                new_resp.get("client_order_id") or new_stop_cid
            )
            return True
        except Exception as e:
            logger.error(
                "replace stop failed for %s @ %s: %s",
                position.symbol, new_stop, e,
            )
            # Try to restore the old stop so the position isn't naked.
            restore_cid = _new_client_order_id()
            try:
                restored = await self.adapter.place_stop_order(
                    symbol=position.symbol,
                    side=stop_side,
                    size=position.size,
                    stop_price=old_stop,
                    reduce_only=True,
                    client_order_id=restore_cid,
                )
                position.stop_order_id = str(restored.get("id") or "")
                position.stop_client_order_id = str(
                    restored.get("client_order_id") or restore_cid
                )
                logger.warning(
                    "restored old stop @ %s on %s after replace failure",
                    old_stop, position.symbol,
                )
                return False
            except Exception as e2:
                logger.critical(
                    "RESTORE old stop also failed on %s: %s — POSITION IS NAKED",
                    position.symbol, e2,
                )
                position.stop_order_id = None
                position.stop_client_order_id = None
                return False

    # ------------------- internals ------------------- #

    @staticmethod
    def _extract_filled_qty(resp: dict[str, Any], requested: float) -> float:
        """Read the filled quantity from a normalised response.

        Preference: ``filled`` (the post-TICKET-002 canonical key) ->
        ``amount`` (legacy key from older mocks) -> ``requested`` (our
        intent, used as a last-resort safety net for adapters that
        echo neither).
        """
        for key in ("filled", "amount"):
            v = resp.get(key)
            if v is None:
                continue
            try:
                f = abs(float(v))
            except (TypeError, ValueError):
                continue
            # NaN / inf guard
            if f != f or f in (float("inf"), float("-inf")):
                continue
            return f
        return float(requested)

    def _emit_close_hint(self, symbol: str, reason: str) -> None:
        """Forward a close-reason hint to the PositionWatcher.

        TICKET-004: this is how the close handler later reports
        ``reason="emergency_close_*"`` instead of the generic default.
        Hooks are best-effort; an exception inside the hint sink must
        never escalate into a failed emergency close.
        """
        if self.on_emergency_close is None:
            return
        try:
            self.on_emergency_close(symbol, reason)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "on_emergency_close hook raised for %s/%s: %s",
                symbol, reason, e,
            )


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


# Back-compat alias retained for any external callers that imported
# the legacy spelling.
def now_ms_default() -> int:
    return _now_ms()
