"""matching_engine.py — In-memory exchange simulator (Phase B.4 / plan B.5.2).

Implements the ``ExchangeAdapter`` Protocol so ``CCXTExecutor`` and the
rest of the risk stack can drive the backtest path with **zero** code
changes from the live path. This is the plan's hard rule:

    "实盘 daemon 和回测使用同一套 RiskGate / Sizer / TrailingFSM 代码
     唯一差异在 IO 层：实盘 ccxt → 回测 HistoricalDataAdapter"

What this module simulates:

* **Market orders**: filled at the *current bar's* close + slippage.
  The current bar is whatever ``BacktestDataAdapter`` last yielded —
  the runner is responsible for advancing the data cursor before
  calling into the executor.

* **Stop orders**: kept in a per-symbol resting list. On every bar
  advance the engine checks each resting stop against the bar's
  intra-bar high/low and triggers (with stop-out slippage) any that
  the bar's range crossed. Triggered stops emit a synthetic position
  close.

* **Cancel + idempotency**: ``cancel_order(order_id)`` removes a
  resting stop; ``client_order_id`` is honoured (a duplicate market
  order with the same client_order_id within the in-memory dedupe
  window is a no-op returning the original fill).

* **Positions**: tracked symbol-by-symbol with a single net side. We
  do not simulate hedge-mode (Binance's "BOTH" position-side); the
  live daemon only uses one-way mode.

* **Balance**: a simple USDT balance tracks realised PnL + fees in
  real time. Margin / liquidation are *not* simulated in v1 — the
  plan defers that to Phase 5; for the rolling-positions logic we
  exercise here, the realised-PnL view is sufficient.

What the engine does **not** simulate (out of scope for v1):

* Order book depth → stop fills assume worst-case slippage from the
  formula but don't sweep through book levels.
* Funding payments → the runner can call ``apply_funding`` separately
  if the trainer ever cares.
* Partial fills → every market order fills 100% in one go.
* Maker / limit orders → not used by the live daemon today.

These omissions are explicit in the plan and tracked for Phase 5.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from altcoin_agent.backtest.data_adapter import BacktestDataAdapter
from altcoin_agent.backtest.slippage_model import (
    SlippageModel,
    SlippageObservation,
)
from altcoin_agent.risk.pump_phase import KlineBar
from altcoin_agent.risk.state import Side

logger = logging.getLogger(__name__)


# Status strings match the ccxt vocabulary closely so callers that
# inspect the dict result don't need to special-case the backtest path.
STATUS_FILLED = "filled"
STATUS_OPEN = "open"
STATUS_CANCELLED = "canceled"
STATUS_TRIGGERED = "triggered"


# --------------------------------------------------------------------- #
# Internal records
# --------------------------------------------------------------------- #


@dataclass
class _RestingStop:
    """A stop order parked on the engine, waiting for the bar to cross it."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: Side
    size: float
    stop_price: float
    reduce_only: bool
    placed_ts_ms: int


@dataclass
class _SimPosition:
    """Per-symbol net position. Mirrors what ccxt fetch_positions returns
    minimally — only the fields ``Reconciler`` reads."""

    symbol: str
    side: Side | None = None  # None means flat
    size: float = 0.0
    entry_notional: float = 0.0  # sum of (size * fill_price), used for avg

    @property
    def avg_entry(self) -> float:
        return self.entry_notional / self.size if self.size > 0 else 0.0

    @property
    def is_flat(self) -> bool:
        return self.side is None or self.size <= 0


# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #


@dataclass
class MatchingEngineConfig:
    """Operator-tunable knobs for the simulator."""

    # Default top-of-book depth in USDT used for the slippage formula
    # when the data adapter doesn't have a per-bar depth value (we don't
    # store depth in the backtest cache yet). Plan calibration target:
    # 200k for top altcoins, 20k for shitcoins.
    default_top_depth_usdt: float = 50_000.0
    # Default realized 30d vol (fraction). Used by slippage when the
    # runner doesn't pass a per-bar value.
    default_realized_vol: float = 0.05
    # Window in seconds within which two market orders sharing the same
    # client_order_id are deduped. Mirrors Binance's behaviour.
    client_order_id_dedupe_sec: int = 60
    # Optional callback fired whenever a stop triggers. Useful for
    # the runner to keep the audit log in sync.
    on_stop_trigger: Callable[[dict[str, Any]], None] | None = None
    # Optional callback for every fill (entry + stop). Same purpose.
    on_fill: Callable[[dict[str, Any]], None] | None = None


# --------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------- #


@dataclass
class MatchingEngine:
    """ExchangeAdapter-compatible in-memory simulator.

    ``data`` is the read-side adapter — ``current_bar(symbol)`` resolves
    the bar a market order is filling against. ``slippage`` provides the
    fill-price formula. The engine itself is sync; we expose async
    methods because ``ExchangeAdapter`` is async, but everything
    runs in-process with no I/O.
    """

    data: BacktestDataAdapter
    slippage: SlippageModel = field(default_factory=SlippageModel)
    cfg: MatchingEngineConfig = field(default_factory=MatchingEngineConfig)
    starting_balance_usdt: float = 10_000.0

    # --- mutable state ---
    balance_usdt: float = field(init=False)
    realized_pnl_usdt: float = 0.0
    fees_paid_usdt: float = 0.0
    positions: dict[str, _SimPosition] = field(default_factory=dict)
    resting_stops: dict[str, list[_RestingStop]] = field(default_factory=dict)
    fills: list[dict[str, Any]] = field(default_factory=list)
    triggered_stops: list[dict[str, Any]] = field(default_factory=list)
    observations: list[SlippageObservation] = field(default_factory=list)
    # Dedupe table for client_order_id. Bounded by an OrderedDict.
    _coid_results: OrderedDict[str, tuple[float, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )
    _now_ms: Callable[[], int] = field(default=lambda: int(time.time() * 1000), repr=False)

    def __post_init__(self) -> None:
        self.balance_usdt = float(self.starting_balance_usdt)

    # ----------------------------------------------------------------- #
    # ExchangeAdapter — order placement
    # ----------------------------------------------------------------- #

    async def market_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Fill a market order against the *current* bar of ``symbol``.

        ``price`` is honoured as the mark price if the caller passes it
        (the live daemon does for "best price" semantics in some venues).
        Otherwise we use ``current_bar.close``.
        """
        # 1. Idempotency check.
        if client_order_id:
            cached = self._dedupe_lookup(client_order_id)
            if cached is not None:
                return cached

        bar = self.data.current_bar(symbol)
        if bar is None:
            raise RuntimeError(
                f"matching_engine: no current bar for {symbol!r}; "
                "did the runner forget to advance the data adapter?"
            )

        mark = float(price) if price is not None else float(bar.close)
        notional = mark * size
        fill_price, fee = self.slippage.apply(
            side=side,
            mark_price=mark,
            notional_usdt=notional,
            top_depth_usdt=self.cfg.default_top_depth_usdt,
            realized_vol_pct=self.cfg.default_realized_vol,
        )
        fill = self._record_fill(
            symbol=symbol,
            side=side,
            size=size,
            mark=mark,
            fill_price=fill_price,
            fee=fee,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
            bar_ts_ms=bar.ts_ms,
            kind="market",
        )
        if client_order_id:
            self._dedupe_store(client_order_id, fill)
        return fill

    async def place_stop_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Park a stop order — it triggers when a future bar's range
        contains ``stop_price``."""
        if client_order_id:
            cached = self._dedupe_lookup(client_order_id)
            if cached is not None:
                return cached

        order_id = f"stop-{uuid.uuid4().hex[:12]}"
        rs = _RestingStop(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            size=float(size),
            stop_price=float(stop_price),
            reduce_only=bool(reduce_only),
            placed_ts_ms=self._now_ms(),
        )
        self.resting_stops.setdefault(symbol, []).append(rs)
        result = {
            "id": order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side.value,
            "size": float(size),
            "stop_price": float(stop_price),
            "type": "stop_market",
            "reduce_only": bool(reduce_only),
            "status": STATUS_OPEN,
            "ts_ms": rs.placed_ts_ms,
        }
        if client_order_id:
            self._dedupe_store(client_order_id, result)
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        stops = self.resting_stops.get(symbol, [])
        for i, rs in enumerate(stops):
            if rs.order_id == order_id:
                del stops[i]
                return {
                    "id": order_id,
                    "symbol": symbol,
                    "status": STATUS_CANCELLED,
                }
        # Match ccxt: cancelling an unknown order returns a "no-op" dict
        # rather than raising, since live + retry can race.
        return {"id": order_id, "symbol": symbol, "status": "not_found"}

    async def set_leverage(self, symbol: str, leverage: float) -> dict[str, Any]:
        # No-op in backtest — leverage is enforced by the sizer upstream
        # and we don't model margin.
        return {"symbol": symbol, "leverage": float(leverage)}

    async def fetch_positions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in self.positions.values():
            if p.is_flat:
                continue
            out.append({
                "symbol": p.symbol,
                "side": p.side.value if p.side else "flat",
                "contracts": p.size,
                "size": p.size,
                "entryPrice": p.avg_entry,
                "notional": p.size * p.avg_entry,
            })
        return out

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sym, stops in self.resting_stops.items():
            for s in stops:
                out.append({
                    "id": s.order_id,
                    "clientOrderId": s.client_order_id,
                    "symbol": sym,
                    "side": s.side.value,
                    "size": s.size,
                    "stop_price": s.stop_price,
                    "type": "stop_market",
                    "reduce_only": s.reduce_only,
                    "status": STATUS_OPEN,
                })
        return out

    # ----------------------------------------------------------------- #
    # Bar advance — runner calls this once per bar before consulting
    # any rest-side state. This is where stops fire.
    # ----------------------------------------------------------------- #

    def on_bar(self, symbol: str, bar: KlineBar) -> list[dict[str, Any]]:
        """Notify the engine a new bar arrived; return any stop fills.

        Algorithm:

        1. Walk the resting stops for ``symbol`` in the order they were
           placed (FIFO).
        2. A LONG-protective stop (sell) triggers if ``bar.low <= stop_price``.
        3. A SHORT-protective stop (buy) triggers if ``bar.high >= stop_price``.
        4. The fill price is the worse of (stop_price, bar.open) — the
           "gap-through" case where the stop level wasn't hit but the
           open already gapped past it. Slippage is applied on top.
        """
        triggered: list[dict[str, Any]] = []
        stops = self.resting_stops.get(symbol, [])
        if not stops:
            return triggered
        survivors: list[_RestingStop] = []
        for rs in stops:
            hit, ref_price = self._stop_hit_price(rs, bar)
            if not hit:
                survivors.append(rs)
                continue
            notional = ref_price * rs.size
            fill_price, fee = self.slippage.apply(
                side=rs.side,
                mark_price=ref_price,
                notional_usdt=notional,
                top_depth_usdt=self.cfg.default_top_depth_usdt,
                realized_vol_pct=self.cfg.default_realized_vol,
            )
            fill = self._record_fill(
                symbol=rs.symbol,
                side=rs.side,
                size=rs.size,
                mark=ref_price,
                fill_price=fill_price,
                fee=fee,
                reduce_only=rs.reduce_only,
                client_order_id=rs.client_order_id,
                bar_ts_ms=bar.ts_ms,
                kind="stop",
                origin_order_id=rs.order_id,
            )
            triggered.append(fill)
            self.triggered_stops.append(fill)
            cb = self.cfg.on_stop_trigger
            if cb is not None:
                try:
                    cb(fill)
                except Exception:
                    logger.exception("matching_engine: on_stop_trigger raised")
        self.resting_stops[symbol] = survivors
        return triggered

    @staticmethod
    def _stop_hit_price(rs: _RestingStop, bar: KlineBar) -> tuple[bool, float]:
        """Decide whether ``rs`` triggers on ``bar``; return the worse
        of (stop_price, bar.open) when it does.

        For a *protective* stop:

            * On a LONG position, the stop is a SELL placed *below* the
              entry. It triggers when ``bar.low <= stop_price``; if the
              bar opened *below* the stop (``bar.open < stop_price``),
              we use ``bar.open`` so the gap-through case isn't given a
              free ride.
            * On a SHORT position, the stop is a BUY placed *above*
              entry. Symmetric: triggers when ``bar.high >= stop_price``;
              gap-through reference is ``bar.open`` if it's already
              past.
        """
        if rs.side is Side.SHORT:
            # Stop sits below entry — protect a LONG position
            # (rs.side here is the *closing* side, which is SHORT for a
            # LONG-position protective stop).
            if bar.low <= rs.stop_price:
                if bar.open < rs.stop_price:
                    return True, bar.open
                return True, rs.stop_price
        else:  # rs.side is Side.LONG -> closing a SHORT position
            if bar.high >= rs.stop_price:
                if bar.open > rs.stop_price:
                    return True, bar.open
                return True, rs.stop_price
        return False, 0.0

    # ----------------------------------------------------------------- #
    # Helpers — fill recording, position math, dedupe
    # ----------------------------------------------------------------- #

    def _record_fill(
        self,
        *,
        symbol: str,
        side: Side,
        size: float,
        mark: float,
        fill_price: float,
        fee: float,
        reduce_only: bool,
        client_order_id: str | None,
        bar_ts_ms: int,
        kind: str,
        origin_order_id: str | None = None,
    ) -> dict[str, Any]:
        order_id = origin_order_id or f"mkt-{uuid.uuid4().hex[:12]}"
        # Update position + balance.
        realized_delta = self._apply_fill_to_position(
            symbol=symbol, side=side, size=size, fill_price=fill_price,
            reduce_only=reduce_only,
        )
        self.realized_pnl_usdt += realized_delta
        self.fees_paid_usdt += fee
        self.balance_usdt += realized_delta - fee
        slip_pct = (
            (fill_price - mark) / mark if mark > 0 else 0.0
        )
        # We want the *signed* slip from the trader's perspective:
        # LONG taker pays more (positive); SHORT taker receives less
        # (also positive when expressed as |delta|/mark). Flip sign for
        # SHORT so positive always means "worse than mark".
        if side is Side.SHORT:
            slip_pct = -slip_pct
        fill_record = {
            "id": order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side.value,
            "size": float(size),
            "mark_price": float(mark),
            "fill_price": float(fill_price),
            "fee_usdt": float(fee),
            "slippage_pct": float(slip_pct),
            "realized_pnl_usdt": float(realized_delta),
            "reduce_only": bool(reduce_only),
            "kind": kind,                # "market" | "stop"
            "status": STATUS_FILLED if kind == "market" else STATUS_TRIGGERED,
            "ts_ms": int(bar_ts_ms),
        }
        self.fills.append(fill_record)
        # Synthetic observation row — useful for live calibration parity.
        self.observations.append(SlippageObservation(
            ts_ms=int(bar_ts_ms),
            symbol=symbol,
            side=side.value,
            mark_price=float(mark),
            fill_price=float(fill_price),
            notional_usdt=float(mark * size),
            top_depth_usdt=float(self.cfg.default_top_depth_usdt),
            realized_vol_pct=float(self.cfg.default_realized_vol),
            actual_slippage=float(abs(slip_pct)),
        ))
        cb = self.cfg.on_fill
        if cb is not None:
            try:
                cb(fill_record)
            except Exception:
                logger.exception("matching_engine: on_fill raised")
        return fill_record

    def _apply_fill_to_position(
        self,
        *,
        symbol: str,
        side: Side,
        size: float,
        fill_price: float,
        reduce_only: bool,
    ) -> float:
        """Apply ``fill`` to the symbol's net position. Return realised
        PnL delta (positive = profit) attributable to this fill.

        Logic mirrors a venue's one-way mode:

            * If the position is flat OR the fill is on the same side
              and ``reduce_only`` is False → it's an *open*: extend
              size, accrue notional, no realised PnL.

            * If the fill is on the *opposite* side OR ``reduce_only``
              is True → it's a *close* (full or partial). Realised PnL
              is computed from the avg entry vs fill_price.
        """
        pos = self.positions.setdefault(symbol, _SimPosition(symbol=symbol))
        if pos.is_flat:
            if reduce_only:
                # Reduce-only on a flat position is a no-op (mirrors ccxt).
                return 0.0
            pos.side = side
            pos.size = float(size)
            pos.entry_notional = float(size * fill_price)
            return 0.0

        # Existing position. Same side -> add; opposite side -> reduce.
        if side is pos.side and not reduce_only:
            pos.size += float(size)
            pos.entry_notional += float(size * fill_price)
            return 0.0

        # Reduce / close path.
        # We're reducing by min(pos.size, size); any leftover *flips* the
        # position (this matches Binance one-way mode).
        avg = pos.avg_entry
        close_size = min(pos.size, size)
        # Sign: closing a LONG at fill_price > avg is profit; closing a
        # SHORT at fill_price < avg is profit.
        if pos.side is Side.LONG:
            pnl = (fill_price - avg) * close_size
        else:
            pnl = (avg - fill_price) * close_size

        leftover = size - close_size
        # Reduce existing position.
        pos.size -= close_size
        pos.entry_notional -= avg * close_size
        if pos.size <= 1e-12:
            pos.size = 0.0
            pos.entry_notional = 0.0
            pos.side = None
        # Flip remaining size into a new opposite position (only if not
        # reduce-only; reduce-only never flips).
        if leftover > 1e-12 and not reduce_only:
            pos.side = side
            pos.size = leftover
            pos.entry_notional = leftover * fill_price
        return float(pnl)

    # ---- dedupe ---- #

    def _dedupe_store(self, coid: str, result: dict[str, Any]) -> None:
        now = self._now_ms() / 1000.0
        self._coid_results[coid] = (now, dict(result))
        if len(self._coid_results) > 4096:
            # Bound memory; drop oldest.
            self._coid_results.popitem(last=False)

    def _dedupe_lookup(self, coid: str) -> dict[str, Any] | None:
        entry = self._coid_results.get(coid)
        if entry is None:
            return None
        ts, result = entry
        now = self._now_ms() / 1000.0
        if now - ts > self.cfg.client_order_id_dedupe_sec:
            self._coid_results.pop(coid, None)
            return None
        return dict(result)

    # ---- inspectors ---- #

    def equity(self) -> float:
        """Realised + open-position mark-to-market."""
        equity = self.balance_usdt
        for sym, pos in self.positions.items():
            if pos.is_flat:
                continue
            bar = self.data.current_bar(sym)
            if bar is None:
                continue
            mark = bar.close
            sign = 1.0 if pos.side is Side.LONG else -1.0
            equity += sign * (mark - pos.avg_entry) * pos.size
        return float(equity)

    def summary(self) -> dict[str, Any]:
        return {
            "starting_balance_usdt": float(self.starting_balance_usdt),
            "balance_usdt": float(self.balance_usdt),
            "realized_pnl_usdt": float(self.realized_pnl_usdt),
            "fees_paid_usdt": float(self.fees_paid_usdt),
            "equity": float(self.equity()),
            "fills": len(self.fills),
            "triggered_stops": len(self.triggered_stops),
            "open_positions": [
                p.symbol for p in self.positions.values() if not p.is_flat
            ],
        }


__all__ = [
    "MatchingEngine",
    "MatchingEngineConfig",
    "STATUS_CANCELLED",
    "STATUS_FILLED",
    "STATUS_OPEN",
    "STATUS_TRIGGERED",
]
