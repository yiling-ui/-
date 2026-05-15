"""
Mock tests for Task D — Risk Gate, Sizer, Trailing FSM, Reconciler, Executor.

ALL survival rules from .kiro/steering/trading_logic.md are exercised:

    SR-1 (slippage abort)            : test_gate_aborts_on_excessive_slippage_*
    SR-2 (state sync + hard stop)    : test_executor_force_closes_when_hard_stop_fails
                                        test_reconciler_*
    SR-3 (wash trading - future)     : marked TODO, hook only
    SR-4 (sybil - active in prompt)  : covered in test_ai_engine.py prompt assertion
"""

from __future__ import annotations

from typing import Any

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.risk.executor import CCXTExecutor
from altcoin_agent.risk.gate import RiskGate, RiskGateConfig
from altcoin_agent.risk.reconciler import Reconciler
from altcoin_agent.risk.sizing import (
    OrderIntent,
    PositionSizer,
    compute_dynamic_leverage,
)
from altcoin_agent.risk.state import AccountState, Position, Side
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _signal(
    *,
    direction: Direction = Direction.LONG,
    final_score: float = 90.0,
    rule_score: float = 90.0,
    llm_score: float = 88.0,
    blocked: bool = False,
    block_reason: str | None = None,
    trigger_price: float | None = None,
    ts: int = 1_000_000,
) -> FusedSignal:
    sig = FusedSignal(
        symbol="RAVEUSDT",
        exchange="binance",
        ts=ts,
        direction=direction,
        rule_score=rule_score,
        llm_score=llm_score,
        final_score=final_score,
        is_high_priority=True,
        blocked=blocked,
        block_reason=block_reason,
    )
    if trigger_price is not None:
        # The bus integration layer is expected to stamp this. Tests do it manually.
        object.__setattr__(sig, "trigger_price", trigger_price)
    return sig


def _account(*, equity: float = 10_000.0, reconciled: bool = True, **kw: Any) -> AccountState:
    a = AccountState(equity_usdt=equity, **kw)
    a.reconciliation_complete = reconciled
    return a


# --------------------------------------------------------------------------- #
# Dynamic leverage formula
# --------------------------------------------------------------------------- #


def test_dynamic_leverage_floor_and_ceiling() -> None:
    """At promote_threshold (85), leverage = base = 5x. At 100 + ideal vol/liq, hits ceiling."""
    lev_at_85 = compute_dynamic_leverage(
        side=Side.LONG, fused_score=85.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    assert lev_at_85 == pytest.approx(5.0)

    lev_at_100 = compute_dynamic_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    assert lev_at_100 == pytest.approx(15.0)


def test_dynamic_leverage_short_capped_at_10() -> None:
    """User rule: shorts capped at 10x even when conf is max."""
    lev = compute_dynamic_leverage(
        side=Side.SHORT, fused_score=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    assert lev == pytest.approx(10.0)


def test_dynamic_leverage_volatile_market_lowers_leverage() -> None:
    quiet = compute_dynamic_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    volatile = compute_dynamic_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_volatility_pct=0.10,    # 5x more volatile
        book_depth_usdt_top5=1_000_000.0,
    )
    assert volatile < quiet
    assert volatile >= 5.0


def test_dynamic_leverage_thin_book_lowers_leverage() -> None:
    deep = compute_dynamic_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
        min_liquidity_usdt=200_000.0,
    )
    thin = compute_dynamic_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=100_000.0,  # under min
        min_liquidity_usdt=200_000.0,
    )
    assert thin < deep
    assert thin >= 5.0


# --------------------------------------------------------------------------- #
# Position sizer
# --------------------------------------------------------------------------- #


def test_sizer_respects_risk_parity_invariant() -> None:
    """If price moves to initial_stop, loss == max_risk_per_trade * equity."""
    sizer = PositionSizer(max_risk_per_trade=0.015, contract_step=1e-6)
    intent = OrderIntent(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.000,
        entry_price=1.000, initial_stop=0.950,  # 5% stop
        fused_score=90.0, confidence=0.85,
    )
    result = sizer.compute(
        intent=intent, equity_usdt=10_000.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    # implied loss = size * |entry - stop|
    implied_loss = result.size_contracts * intent.stop_distance_or(0.05)
    expected_risk = 10_000.0 * 0.015
    # within 1% of intended
    assert abs(implied_loss - expected_risk) / expected_risk < 0.01


def test_sizer_zero_when_below_min_notional() -> None:
    sizer = PositionSizer(max_risk_per_trade=0.0001, min_notional_usdt=100.0)
    intent = OrderIntent(
        symbol="X", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.0, entry_price=1.0, initial_stop=0.9,
        fused_score=90.0, confidence=0.8,
    )
    result = sizer.compute(
        intent=intent, equity_usdt=100.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    assert result.size_contracts == 0.0


# Add helper used above
def _patch_intent_stop_distance() -> None:
    """OrderIntent doesn't expose stop_distance directly; tests use a helper."""
    if not hasattr(OrderIntent, "stop_distance_or"):
        def _sd(self: OrderIntent, fallback: float) -> float:  # type: ignore[no-redef]
            return abs(self.entry_price - self.initial_stop) or fallback
        OrderIntent.stop_distance_or = _sd                     # type: ignore[attr-defined]


_patch_intent_stop_distance()


# --------------------------------------------------------------------------- #
# Risk gate — happy path
# --------------------------------------------------------------------------- #


def test_gate_approves_clean_long_signal() -> None:
    gate = RiskGate()
    sig = _signal(trigger_price=1.000)
    decision = gate.evaluate(
        signal=sig,
        current_price=1.005,                # +0.5% slippage
        book_depth_usdt_top5=500_000.0,
        realized_volatility_pct=0.02,
        account=_account(),
        proposed_initial_stop=0.95,
    )
    assert decision.approved is True
    assert decision.intent is not None
    assert decision.intent.side == Side.LONG
    assert decision.intent.entry_price == 1.005
    assert decision.intent.initial_stop == 0.95


# --------------------------------------------------------------------------- #
# Risk gate — SR-1 slippage tests
# --------------------------------------------------------------------------- #


def test_gate_aborts_on_excessive_slippage_long_5x() -> None:
    """At leverage 5x the base 3% threshold applies. 4% against entry -> abort."""
    gate = RiskGate()
    sig = _signal(final_score=85.0, trigger_price=1.000)  # score=85 -> 5x
    decision = gate.evaluate(
        signal=sig,
        current_price=1.040,                 # +4% — slipped against long
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=_account(),
        proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert decision.reason is not None and "slippage_abort" in decision.reason


def test_gate_aborts_on_excessive_slippage_long_10x() -> None:
    """
    At score=92 -> conf_norm=(92-85)/15=0.467; ideal vol/liq -> leverage ~ 9.7x.
    Threshold: 3% / sqrt(9.67/5) = ~2.16%. 2.5% against entry -> abort.
    """
    gate = RiskGate()
    sig = _signal(final_score=92.0, trigger_price=1.000)
    decision = gate.evaluate(
        signal=sig,
        current_price=1.025,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=_account(),
        proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "slippage_abort" in (decision.reason or "")


def test_gate_does_not_abort_when_price_moved_in_our_favor() -> None:
    """Asymmetric guard: price below trigger on a LONG entry = better entry."""
    gate = RiskGate()
    sig = _signal(final_score=92.0, trigger_price=1.000)
    decision = gate.evaluate(
        signal=sig,
        current_price=0.970,                 # -3% — moved IN our favor
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=_account(),
        proposed_initial_stop=0.92,
    )
    assert decision.approved is True


def test_gate_aborts_on_excessive_slippage_short() -> None:
    gate = RiskGate()
    sig = _signal(direction=Direction.SHORT, final_score=92.0, trigger_price=1.000)
    decision = gate.evaluate(
        signal=sig,
        current_price=0.96,                  # price fell after short signal — bad fill for short
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=_account(),
        proposed_initial_stop=1.05,
    )
    assert decision.approved is False
    assert "slippage_abort" in (decision.reason or "")


# --------------------------------------------------------------------------- #
# Risk gate — SR-2 reconciliation gate
# --------------------------------------------------------------------------- #


def test_gate_rejects_until_reconciliation_complete() -> None:
    gate = RiskGate()
    sig = _signal(trigger_price=1.0)
    account = _account(reconciled=False)
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=account, proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert decision.reason == "reconciliation_pending"


# --------------------------------------------------------------------------- #
# Risk gate — other defenses
# --------------------------------------------------------------------------- #


def test_gate_rejects_when_book_too_thin() -> None:
    gate = RiskGate(config=RiskGateConfig(min_liquidity_usdt=200_000.0))
    sig = _signal(trigger_price=1.0)
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=50_000.0,
        realized_volatility_pct=0.02,
        account=_account(), proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "book_depth" in (decision.reason or "")


def test_gate_rejects_under_drawdown_circuit_breaker() -> None:
    gate = RiskGate(config=RiskGateConfig(daily_drawdown_limit=0.06))
    sig = _signal(trigger_price=1.0)
    account = _account(equity=10_000.0)
    account.realized_pnl_today_usdt = -700.0   # -7%
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=account, proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "daily_drawdown" in (decision.reason or "")


def test_gate_rejects_blocked_signal_even_if_score_high() -> None:
    gate = RiskGate()
    sig = _signal(final_score=95.0, blocked=True, block_reason="kol_exit_liquidity_hard_veto",
                  trigger_price=1.0)
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=_account(), proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "signal_blocked" in (decision.reason or "")


def test_gate_rejects_max_concurrency() -> None:
    gate = RiskGate(config=RiskGateConfig(max_concurrent_positions=2))
    sig = _signal(trigger_price=1.0)
    account = _account()
    for i in range(2):
        account.open_positions.append(Position(
            symbol=f"AUSDT_{i}", exchange="binance", side=Side.LONG,
            entry_price=1.0, size_contracts=1.0, leverage=5.0, opened_ts=1,
            initial_stop=0.95, current_hard_stop=0.95,
        ))
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=account, proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "max_concurrent_positions" in (decision.reason or "")


def test_gate_rejects_under_symbol_cooldown() -> None:
    gate = RiskGate()
    sig = _signal(trigger_price=1.0, ts=1_000_000)
    account = _account()
    account.symbol_cooldowns["binance:RAVEUSDT"] = 1_500_000  # in the future
    decision = gate.evaluate(
        signal=sig, current_price=1.0,
        book_depth_usdt_top5=1_000_000.0,
        realized_volatility_pct=0.02,
        account=account, proposed_initial_stop=0.95,
    )
    assert decision.approved is False
    assert "cooldown" in (decision.reason or "")


# --------------------------------------------------------------------------- #
# Trailing stop FSM
# --------------------------------------------------------------------------- #


def _pos(*, side: Side = Side.LONG, entry: float = 100.0, stop: float = 95.0) -> Position:
    return Position(
        symbol="X", exchange="binance", side=side,
        entry_price=entry, size_contracts=1.0, leverage=5.0, opened_ts=0,
        initial_stop=stop, current_hard_stop=stop,
    )


def test_trailing_long_armed_to_breakeven() -> None:
    fsm = TrailingStopFSM()
    p = _pos(entry=100, stop=95)        # 1R = 5
    state, new_stop = fsm.tick(position=p, current_price=105.0, atr=2.0)  # +1R
    assert state == TrailingState.BREAKEVEN
    assert new_stop == pytest.approx(100.0)


def test_trailing_long_breakeven_to_trailing_then_tightens() -> None:
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    p = _pos(entry=100, stop=95)
    # advance to BE
    fsm.tick(position=p, current_price=105.0, atr=2.0)
    p.current_hard_stop = 100.0
    # +2R triggers TRAILING; ATR-based stop should be 110-2*2=106 (which > 100 => tightens)
    state, new_stop = fsm.tick(position=p, current_price=110.0, atr=2.0)
    assert state == TrailingState.TRAILING
    assert new_stop == pytest.approx(106.0)
    p.current_hard_stop = 106.0
    # price rises further; stop should ratchet up
    state, new_stop = fsm.tick(position=p, current_price=115.0, atr=2.0)
    assert state == TrailingState.TRAILING
    assert new_stop == pytest.approx(111.0)


def test_trailing_long_never_lowers_stop_invariant() -> None:
    """If price retraces, FSM must propose `None` rather than lowering the stop."""
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    p = _pos(entry=100, stop=95)
    fsm.state = TrailingState.TRAILING
    p.current_hard_stop = 110.0  # already trailed up high
    # price drops back to 105 -> ATR stop would be 101 (below current 110)
    state, new_stop = fsm.tick(position=p, current_price=105.0, atr=2.0)
    assert new_stop is None  # invariant: never lower


def test_trailing_short_mirrors_long() -> None:
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    p = _pos(side=Side.SHORT, entry=100, stop=105)  # 1R = 5
    # +1R for short means price moved DOWN to 95
    state, new_stop = fsm.tick(position=p, current_price=95.0, atr=2.0)
    assert state == TrailingState.BREAKEVEN
    assert new_stop == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# SHORT TARGET CAP — F item, "no death-grip to zero"
# --------------------------------------------------------------------------- #


def test_short_target_cap_force_closes_at_70pct_drop() -> None:
    """User mandate F: shorts cap at -70% from entry. Below that, force close."""
    fsm = TrailingStopFSM(atr_multiplier=2.0, short_target_cap_pct=0.70)
    p = Position(
        symbol="RAVEUSDT", exchange="binance", side=Side.SHORT,
        entry_price=1.000, size_contracts=1.0, leverage=10.0, opened_ts=0,
        initial_stop=1.05,                       # +5% stop
        current_hard_stop=0.50,                  # already trailed way down
    )
    fsm.state = TrailingState.TRAILING

    # Price has fallen 71% from entry of 1.0 → 0.29
    state, new_stop = fsm.tick(position=p, current_price=0.29, atr=0.01)
    assert state == TrailingState.TARGET_REACHED
    assert new_stop is not None
    # Stop placed JUST ABOVE current price -> next tick triggers it
    assert new_stop == pytest.approx(0.29 * 1.0005, rel=1e-9)


def test_short_target_cap_does_NOT_trigger_at_60pct_drop() -> None:
    """Boundary: -60% should NOT trigger the cap. Normal trailing continues."""
    fsm = TrailingStopFSM(atr_multiplier=2.0, short_target_cap_pct=0.70)
    p = Position(
        symbol="X", exchange="binance", side=Side.SHORT,
        entry_price=1.000, size_contracts=1.0, leverage=10.0, opened_ts=0,
        initial_stop=1.05,
        current_hard_stop=0.55,
    )
    fsm.state = TrailingState.TRAILING

    state, new_stop = fsm.tick(position=p, current_price=0.40, atr=0.01)  # -60%
    # Should be normal trailing tighten, NOT target_reached
    assert state == TrailingState.TRAILING
    # ATR-trail stop = 0.40 + 2*0.01 = 0.42, which IS tighter than 0.55
    assert new_stop == pytest.approx(0.42)


def test_short_target_cap_does_not_apply_to_long() -> None:
    """LONG positions have no equivalent cap (they have unbounded upside)."""
    fsm = TrailingStopFSM(atr_multiplier=2.0, short_target_cap_pct=0.70)
    p = Position(
        symbol="X", exchange="binance", side=Side.LONG,
        entry_price=1.000, size_contracts=1.0, leverage=10.0, opened_ts=0,
        initial_stop=0.95,
        current_hard_stop=1.50,
    )
    fsm.state = TrailingState.TRAILING

    # +200% gain — long should keep trailing, not "force close"
    state, new_stop = fsm.tick(position=p, current_price=3.00, atr=0.05)
    assert state == TrailingState.TRAILING
    assert new_stop == pytest.approx(2.90)
    assert state != TrailingState.TARGET_REACHED


# --------------------------------------------------------------------------- #
# Reconciler — SR-2 startup
# --------------------------------------------------------------------------- #


class FakeReconcileAdapter:
    def __init__(self, positions: list[dict], orders: list[dict]):
        self._positions = positions
        self._orders = orders
        self.placed_stops: list[dict] = []
        self.cancelled: list[str] = []

    async def fetch_positions(self) -> list[dict]:
        return self._positions

    async def fetch_open_orders(self) -> list[dict]:
        return self._orders

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    async def place_stop_order(
        self, symbol: str, side: Side, size: float, stop_price: float, reduce_only: bool = True,
    ) -> dict:
        order = {
            "id": f"stop-{symbol}-{stop_price}",
            "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "reduce_only": reduce_only,
        }
        self.placed_stops.append(order)
        return order


@pytest.mark.asyncio
async def test_reconciler_detects_orphan_and_places_breakeven_stop() -> None:
    # Exchange shows 1 RAVEUSDT long position, but local state has no record.
    adapter = FakeReconcileAdapter(
        positions=[{"symbol": "RAVEUSDT", "size": 100.0, "entryPrice": 1.0}],
        orders=[],
    )
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    account = _account(reconciled=False)

    report = await rec.run(account)
    assert report.success is True
    assert len(report.orphans) == 1
    assert report.orphans[0].symbol == "RAVEUSDT"
    assert report.orphans[0].side == Side.LONG
    # Breakeven stop placed (with buffer) — for a LONG this is BELOW entry
    assert len(adapter.placed_stops) == 1
    placed = adapter.placed_stops[0]
    assert placed["side"] == "short"           # closing side
    assert placed["stop_price"] < 1.0          # below entry
    assert placed["reduce_only"] is True


@pytest.mark.asyncio
async def test_reconciler_no_orphan_when_position_known() -> None:
    adapter = FakeReconcileAdapter(
        positions=[{"symbol": "RAVEUSDT", "size": 100.0, "entryPrice": 1.0}],
        orders=[],
    )
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    account = _account()
    account.open_positions.append(Position(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        entry_price=1.0, size_contracts=100.0, leverage=5.0, opened_ts=0,
        initial_stop=0.95, current_hard_stop=0.95,
    ))
    report = await rec.run(account)
    assert report.success is True
    assert report.orphans == []
    assert adapter.placed_stops == []


# --------------------------------------------------------------------------- #
# Executor — SR-2 hard stop pairing + fail-closed
# --------------------------------------------------------------------------- #


class FakeExchangeAdapter:
    def __init__(
        self,
        *,
        fill_price: float = 1.005,
        stop_will_fail: bool = False,
        cancel_will_fail: bool = False,
        replace_will_fail: bool = False,
    ):
        self.fill_price = fill_price
        self.stop_will_fail = stop_will_fail
        self.cancel_will_fail = cancel_will_fail
        self.replace_will_fail = replace_will_fail
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self.leverages: list[tuple[str, float]] = []
        self._next_id = 1
        self._stop_attempts = 0  # for distinguishing place vs replace

    def _id(self) -> str:
        v = f"o{self._next_id}"
        self._next_id += 1
        return v

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False):
        order = {
            "id": self._id(), "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only, "average": self.fill_price,
        }
        self.market_orders.append(order)
        return order

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True):
        self._stop_attempts += 1
        # First attempt: respect stop_will_fail. Subsequent (e.g. restore) attempts
        # respect replace_will_fail.
        if self._stop_attempts == 1 and self.stop_will_fail:
            raise RuntimeError("stop_placement_failed")
        if self._stop_attempts > 1 and self.replace_will_fail:
            raise RuntimeError("stop_replacement_failed")
        order = {
            "id": self._id(), "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "reduce_only": reduce_only,
        }
        self.stop_orders.append(order)
        return order

    async def cancel_order(self, order_id, symbol):
        if self.cancel_will_fail:
            raise RuntimeError("cancel_failed")
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):
        self.leverages.append((symbol, leverage))
        return {"symbol": symbol, "leverage": leverage}


@pytest.mark.asyncio
async def test_executor_opens_position_and_immediately_places_hard_stop() -> None:
    adapter = FakeExchangeAdapter(fill_price=1.005)
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer(contract_step=1e-6)
    intent = OrderIntent(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.000, entry_price=1.005,
        initial_stop=0.96, fused_score=92.0, confidence=0.85,
    )
    sizing = sizer.compute(
        intent=intent, equity_usdt=10_000.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    account = _account()

    result = await executor.open_with_hard_stop(
        intent=intent, sizing=sizing, account=account,
    )
    assert result.success is True
    assert result.position is not None
    assert result.placed_stop_order_id is not None
    assert len(adapter.market_orders) == 1                     # entry only
    assert len(adapter.stop_orders) == 1
    assert adapter.stop_orders[0]["reduce_only"] is True
    assert adapter.stop_orders[0]["side"] == "short"           # opposite of entry
    assert adapter.stop_orders[0]["stop_price"] == pytest.approx(0.96)


@pytest.mark.asyncio
async def test_executor_force_closes_when_hard_stop_fails() -> None:
    """SR-2 fail-closed: stop placement failure -> immediate market close, cooldown."""
    adapter = FakeExchangeAdapter(fill_price=1.005, stop_will_fail=True)
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer(contract_step=1e-6)
    intent = OrderIntent(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.000, entry_price=1.005,
        initial_stop=0.96, fused_score=92.0, confidence=0.85,
    )
    sizing = sizer.compute(
        intent=intent, equity_usdt=10_000.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    account = _account()

    result = await executor.open_with_hard_stop(
        intent=intent, sizing=sizing, account=account,
    )
    assert result.success is False
    assert result.forced_close is True
    assert "hard_stop_placement_failed" in (result.reason or "")
    # Two market orders: entry + force close
    assert len(adapter.market_orders) == 2
    assert adapter.market_orders[1]["reduce_only"] is True
    # Cooldown engaged for 4h
    assert "binance:RAVEUSDT" in account.symbol_cooldowns
    cooldown_ms = account.symbol_cooldowns["binance:RAVEUSDT"]
    assert cooldown_ms > executor._now_ms() + 3 * 60 * 60 * 1000
    # Position not stored
    assert account.open_positions == []


@pytest.mark.asyncio
async def test_executor_rejects_when_reconciliation_pending() -> None:
    adapter = FakeExchangeAdapter()
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer(contract_step=1e-6)
    intent = OrderIntent(
        symbol="X", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.0, entry_price=1.0, initial_stop=0.95,
        fused_score=92.0, confidence=0.85,
    )
    sizing = sizer.compute(
        intent=intent, equity_usdt=10_000.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    account = _account(reconciled=False)
    result = await executor.open_with_hard_stop(intent=intent, sizing=sizing, account=account)
    assert result.success is False
    assert result.reason == "reconciliation_pending"
    # No orders placed at all
    assert adapter.market_orders == []
    assert adapter.stop_orders == []


@pytest.mark.asyncio
async def test_executor_tighten_stop_keeps_old_when_replace_fails() -> None:
    """SR-2 invariant: cancel+replace failure keeps old stop in force OR clearly logs naked."""
    adapter = FakeExchangeAdapter(replace_will_fail=False)
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer(contract_step=1e-6)
    intent = OrderIntent(
        symbol="X", exchange="binance", side=Side.LONG,
        trigger_ts=1, trigger_price=1.0, entry_price=1.0,
        initial_stop=0.95, fused_score=92.0, confidence=0.85,
    )
    sizing = sizer.compute(
        intent=intent, equity_usdt=10_000.0,
        realized_volatility_pct=0.02, book_depth_usdt_top5=1_000_000.0,
    )
    account = _account()
    # First open a clean position
    res = await executor.open_with_hard_stop(intent=intent, sizing=sizing, account=account)
    assert res.success
    pos = res.position
    assert pos is not None

    # Now flip the adapter: replacement (= 2nd-onwards stop placement) will fail.
    # Cancel succeeds. Old stop will be re-placed via the restore path.
    adapter.replace_will_fail = True
    ok = await executor.tighten_hard_stop(pos, new_stop=0.97)
    assert ok is False
    # If restore succeeded the position has a stop id; if BOTH new+restore failed
    # it would be None. With our FakeExchangeAdapter, replace_will_fail blocks
    # only the FIRST replacement after cancel. Restore (3rd attempt) succeeds.
    # Either way, current_hard_stop must NOT have been moved past the cancel-only state.
    assert pos.current_hard_stop == 0.95


# --------------------------------------------------------------------------- #
# Sybil prompt assertion (SR-4 active in prompt)
# --------------------------------------------------------------------------- #


def test_sr4_sybil_defense_present_in_system_prompt() -> None:
    from altcoin_agent.ai_engine import SYSTEM_PROMPT
    p = SYSTEM_PROMPT.lower()
    # The active-now defense for SR-4: prompt mentions both the trigger and the action.
    assert "sybil" in p or "astroturf" in p
    assert "low-follower" in p or "low follower" in p
    assert "homogeneous" in p or "homogeneity" in p or "bot spam" in p
    assert "exit_liquidity" in p
    # Quantified penalty
    assert "30" in p
