"""reconciler.py — startup state alignment (SR-2).

On daemon startup we cannot trust local in-memory state. We:
  1. Pull all open positions from the exchange.
  2. For positions we DO know about locally, verify the size/side and warn on diff.
  3. For ORPHAN positions (on the exchange but not in our state), refuse to
     close them automatically (they may be manual user trades) BUT we DO
     attempt to attach a protective stop if one is missing -- 'orphan with
     no stop' is the single most dangerous state.

If reconciliation fails for any reason we keep ``account.reconciliation_complete``
as False; the gate then refuses ALL orders. This is fail-closed by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from altcoin_agent.risk.state import AccountState

if TYPE_CHECKING:
    from altcoin_agent.risk.executor import ExchangeAdapter

logger = logging.getLogger(__name__)


@dataclass
class ReconcilerReport:
    success: bool
    exchange_name: str
    positions_seen: int = 0
    orphans_found: int = 0
    orphans_protected: int = 0
    diffs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Reconciler[{self.exchange_name}] success={self.success} "
            f"positions={self.positions_seen} orphans={self.orphans_found} "
            f"protected={self.orphans_protected} "
            f"diffs={len(self.diffs)} errors={len(self.errors)}"
        )


class Reconciler:
    """One-shot per startup. Idempotent."""

    def __init__(self, *, exchange_name: str, adapter: ExchangeAdapter):
        self.exchange_name = exchange_name
        self.adapter = adapter

    async def run(self, account: AccountState) -> ReconcilerReport:
        report = ReconcilerReport(success=False, exchange_name=self.exchange_name)
        try:
            exch_positions = await self.adapter.fetch_positions()
            exch_orders = await self.adapter.fetch_open_orders()
        except Exception as e:
            report.errors.append(f"fetch_failed:{type(e).__name__}:{e}")
            logger.error("Reconciler fetch failed: %s", e)
            return report

        report.positions_seen = len(exch_positions)
        local_symbols = set(account.open_positions.keys())

        for raw in exch_positions:
            try:
                symbol = str(raw.get("symbol") or raw.get("info", {}).get("symbol") or "")
                size_raw = raw.get("contracts") or raw.get("size") or 0.0
                size = abs(float(size_raw))
                side_str = str(raw.get("side") or "").lower()
                if size <= 0:
                    continue

                if symbol in local_symbols:
                    local = account.open_positions[symbol]
                    if abs(local.size - size) / max(local.size, 1e-9) > 0.01:
                        msg = f"size_diff:{symbol} local={local.size} exch={size}"
                        report.diffs.append(msg)
                        logger.warning("Reconciler %s", msg)
                    if local.side.value != side_str:
                        msg = f"side_diff:{symbol} local={local.side.value} exch={side_str}"
                        report.diffs.append(msg)
                        logger.warning("Reconciler %s", msg)
                else:
                    # Orphan: not in our state. Don't close — could be manual.
                    report.orphans_found += 1
                    if not self._has_protective_stop(symbol, exch_orders):
                        protected = await self._attach_emergency_stop(raw)
                        if protected:
                            report.orphans_protected += 1
            except Exception as e:
                report.errors.append(f"row_error:{e}")
                logger.warning("Reconciler row error: %s", e)

        report.success = not report.errors
        account.reconciliation_complete = report.success
        return report

    # ---------------- helpers ---------------- #

    @staticmethod
    def _has_protective_stop(symbol: str, orders: list[dict]) -> bool:
        for o in orders:
            o_symbol = o.get("symbol", "")
            o_type = str(o.get("type", "")).lower()
            o_reduce = bool(o.get("reduceOnly") or o.get("reduce_only"))
            if o_symbol == symbol and "stop" in o_type and o_reduce:
                return True
        return False

    async def _attach_emergency_stop(self, raw_position: dict) -> bool:
        """Best-effort: place a wide protective stop on an orphan.

        Audit #17: the previous version always used 5% adverse from
        entry. On a 10x leveraged orphan that is 50% of equity wiped
        out before the stop fires; on a 25x position (Binance default
        for altcoin perp users) it is 125% — i.e. liquidation before
        stop. We now translate the equity-side guardrail (default
        ``max_equity_loss_pct=0.30``) to a price distance scaled by
        the venue-reported leverage. A position with no leverage info
        (rare; most adapters always populate it) falls back to the
        old 5% to keep behaviour conservative.
        """
        try:
            symbol = str(raw_position.get("symbol") or "")
            side_str = str(raw_position.get("side") or "long").lower()
            size = abs(float(raw_position.get("contracts") or raw_position.get("size") or 0.0))
            entry = float(
                raw_position.get("entryPrice") or raw_position.get("avgPrice") or 0.0
            )
            if not symbol or size <= 0 or entry <= 0:
                return False
            from altcoin_agent.risk.state import Side
            pos_side = Side.LONG if side_str == "long" else Side.SHORT
            # Stop distance = min(5% absolute, max_equity_loss_pct / leverage).
            # On a 10x position with max_equity_loss=30%, that's 3% adverse;
            # on 5x it's 6% (clamped to 5%); on no-leverage info we keep 5%.
            max_equity_loss_pct = 0.30
            absolute_cap_pct = 0.05
            # Audit (third pass) #8: ccxt unifies many but not all
            # ``fetch_positions`` fields. ``leverage`` is reliably
            # present at the top level only on a subset of venues (and
            # types: spot has no leverage; cross-margin has it under
            # ``crossLeverage`` on some adapters). On Binance USDT-M
            # and Bybit v5 the actual value lives under
            # ``info.leverage``; OKX uses ``info.lever``. We probe all
            # three so the leverage-aware stop actually fires in
            # production (the previous code degraded to 5% on every
            # real venue).
            lev = 0.0
            for candidate in (
                raw_position.get("leverage"),
                (raw_position.get("info") or {}).get("leverage"),
                (raw_position.get("info") or {}).get("lever"),
                (raw_position.get("info") or {}).get("crossLeverage"),
            ):
                if candidate is None:
                    continue
                try:
                    lev = float(candidate)
                except (TypeError, ValueError):
                    continue
                if lev > 0:
                    break
            if lev > 0:
                lev_aware = max_equity_loss_pct / lev
                stop_pct = min(absolute_cap_pct, lev_aware)
            else:
                stop_pct = absolute_cap_pct
            if pos_side == Side.LONG:
                stop_price = entry * (1.0 - stop_pct)
                stop_side = Side.SHORT
            else:
                stop_price = entry * (1.0 + stop_pct)
                stop_side = Side.LONG
            await self.adapter.place_stop_order(
                symbol=symbol,
                side=stop_side,
                size=size,
                stop_price=stop_price,
                reduce_only=True,
            )
            logger.warning(
                "Reconciler attached emergency stop on orphan %s %s @ %s "
                "(stop_pct=%.4f, leverage=%.2f)",
                symbol, side_str, stop_price, stop_pct, lev,
            )
            return True
        except Exception as e:
            logger.error("Reconciler emergency-stop failed: %s", e)
            return False
