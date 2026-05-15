"""Mock tests for risk + execution package (sizing, gate, trailing, executor)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.risk import (
    AccountState,
    CCXTExecutor,
    DynamicLeverageConfig,
    Position,
    PositionSizer,
    Reconciler,
    RiskGate,
    RiskGateConfig,
    Side,
    TrailingState,
    TrailingStopFSM,
)

# --------------------------------------------------------------------- #
# Mock exchange adapter for executor / reconciler tests
# --------------------------------------------------------------------- #


class FakeAdapter:
    def __init__(self, *, fail_stop: bool = False, replace_fails: bool = False) -> None:
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self.leverages: list[tuple[str, float]] = []
        self.fetched_positions: list[dict] = []
        self.fetched_orders: list[dict] = []
        self.fail_stop = fail_stop
        self.replace_fails = replace_fails
        self._n = 0
        self._stop_called = 0

    def _id(self) -> str:
        self._n += 1
        return f"o-{self._n}"

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "average": price or 1.0, "reduce_only": reduce_only, "price": price}
        self.market_orders.append(rec)
        return rec

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True):  # noqa: ANN001
        self._stop_called += 1
        if self.fail_stop:
            raise RuntimeError("simulated stop placement failure")
        if self.replace_fails and self._stop_called > 1:
            raise RuntimeError("simulated replace failure")
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "stop_price": stop_price, "reduce_only": reduce_only}
        self.stop_orders.append(rec)
        return rec

    async def cancel_order(self, order_id, symbol):  # noqa: ANN001
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):  # noqa: ANN001
        self.leverages.append((symbol, leverage))
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_positions(self) -> list[dict[str, Any]]:
        return list(self.fetched_positions)

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        return list(self.fetched_orders)


def _signal(direction: Direction = Direction.LONG, *, score: float = 95.0,
            trigger: float | None = 1.000) -> FusedSignal:
    return FusedSignal(
        symbol="RAVEUSDT", exchange="binance", ts=1,
        direction=direction, rule_score=90.0, llm_score=90.0,
        final_score=score, is_high_priority=True, blocked=False,
        block_reason=None, trigger_price=trigger,
    )


# --------------------------------------------------------------------- #
# PositionSizer
# --------------------------------------------------------------------- #


def test_sizer_risk_parity_invariant() -> None:
    sizer = PositionSizer(max_risk_per_trade=0.015)
    size, notional, risk = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=1.000, initial_stop=0.95,
    )
    # Risk amount = 1.5% of equity = $150
    assert risk == pytest.approx(150.0)
    # Stop distance is 5% -> notional = 150 / 0.05 = $3000
    assert notional == pytest.approx(3000.0, rel=1e-3)
    assert size == pytest.approx(3000.0, rel=1e-3)


def test_sizer_below_min_notional_returns_zero() -> None:
    sizer = PositionSizer(max_risk_per_trade=0.015, min_notional_usdt=1_000_000)
    size, notional, _ = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=1.0, initial_stop=0.99,
    )
    assert size == 0.0 and notional == 0.0


def test_dynamic_leverage_long_at_top_score_high_liq() -> None:
    sizer = PositionSizer(leverage_cfg=DynamicLeverageConfig())
    lev = sizer.compute_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_vol_pct=0.05, top5_depth_usdt=400_000.0,
    )
    assert lev == pytest.approx(15.0)


def test_dynamic_leverage_short_capped_at_10x() -> None:
    sizer = PositionSizer(leverage_cfg=DynamicLeverageConfig())
    lev = sizer.compute_leverage(
        side=Side.SHORT, fused_score=100.0,
        realized_vol_pct=0.05, top5_depth_usdt=400_000.0,
    )
    assert lev == pytest.approx(10.0)


def test_dynamic_leverage_thin_book_drops_leverage() -> None:
    sizer = PositionSizer(leverage_cfg=DynamicLeverageConfig(liq_full_depth_usdt=200_000))
    lev_full = sizer.compute_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_vol_pct=0.05, top5_depth_usdt=200_000,
    )
    lev_thin = sizer.compute_leverage(
        side=Side.LONG, fused_score=100.0,
        realized_vol_pct=0.05, top5_depth_usdt=50_000,
    )
    assert lev_thin < lev_full


# --------------------------------------------------------------------- #
# RiskGate
# --------------------------------------------------------------------- #


def _account(**overrides) -> AccountState:
    a = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
    a.reconciliation_complete = True
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_gate_happy_path_approves() -> None:
    gate = RiskGate(PositionSizer())
    decision = gate.evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.000,
        top5_depth_usdt=400_000,
        realized_vol_pct=0.05,
        initial_stop=0.95,
    )
    assert decision.approved
    assert decision.side == Side.LONG
    assert decision.size > 0


def test_gate_rejects_when_reconciliation_pending() -> None:
    a = _account()
    a.reconciliation_complete = False
    gate = RiskGate(PositionSizer())
    d = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "reconciliation" in d.reason


def test_gate_rejects_blocked_signal() -> None:
    sig = _signal()
    sig.blocked = True
    sig.block_reason = "wash_trading_detected"
    gate = RiskGate(PositionSizer())
    d = gate.evaluate(
        signal=sig, account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "signal_blocked" in d.reason


def test_gate_rejects_low_liquidity() -> None:
    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=200_000))
    d = gate.evaluate(
        signal=_signal(), account=_account(), current_price=1.0,
        top5_depth_usdt=50_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "insufficient_liquidity" in d.reason


def test_gate_sr1_dynamic_slippage_at_10x() -> None:
    """At 10x leverage, allowed adverse drift is ~2.12%."""
    gate = RiskGate(PositionSizer())
    sig = _signal(score=100.0, trigger=1.000)
    # 3% adverse on a LONG -> exceeds threshold for any leverage > ~5x
    d = gate.evaluate(
        signal=sig, account=_account(), current_price=1.030,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "slippage_too_high" in d.reason


def test_gate_sr1_favorable_drift_does_not_abort() -> None:
    gate = RiskGate(PositionSizer())
    # LONG at trigger 1.000 but current is 0.99 (favourable for entry).
    d = gate.evaluate(
        signal=_signal(trigger=1.000), account=_account(), current_price=0.99,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert d.approved


def test_gate_daily_drawdown_circuit_breaker() -> None:
    a = _account()
    a.realized_pnl_today_usdt = -700  # -7%
    gate = RiskGate(PositionSizer(), RiskGateConfig(daily_drawdown_limit=0.06))
    d = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "daily_drawdown" in d.reason


def test_gate_max_concurrent_positions() -> None:
    a = _account()
    for i in range(3):
        a.open_positions[f"X{i}USDT"] = Position(
            symbol=f"X{i}USDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=1.0, leverage=5.0,
            initial_stop=0.95, current_stop=0.95,
        )
    gate = RiskGate(PositionSizer(), RiskGateConfig(max_concurrent_positions=3))
    d = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved
    assert "max_concurrent" in d.reason


# --------------------------------------------------------------------- #
# Trailing FSM
# --------------------------------------------------------------------- #


def _long_pos() -> Position:
    return Position(
        symbol="X", exchange="binance", side=Side.LONG,
        entry_price=1.000, size=10.0, leverage=10.0,
        initial_stop=0.950, current_stop=0.950,
    )


def _short_pos() -> Position:
    return Position(
        symbol="X", exchange="binance", side=Side.SHORT,
        entry_price=2.000, size=10.0, leverage=10.0,
        initial_stop=2.100, current_stop=2.100,
    )


def test_trailing_long_breakeven_to_trailing_progression() -> None:
    fsm = TrailingStopFSM(atr_multiplier=2.0)
    pos = _long_pos()
    state = TrailingState.INIT

    # Tick 1: arms.
    state, stop, _ = fsm.tick(position=pos, current_price=1.020, atr=0.005,
                               current_state=state)
    assert state == TrailingState.ARMED and stop is None

    # +1R -> breakeven (stop -> entry).
    state, stop, _ = fsm.tick(position=pos, current_price=1.050, atr=0.005,
                               current_state=state)
    assert state == TrailingState.BREAKEVEN and stop == pytest.approx(1.000)
    pos.current_stop = stop

    # +2R -> trailing (stop = price - 2*ATR).
    state, stop, _ = fsm.tick(position=pos, current_price=1.100, atr=0.005,
                               current_state=state)
    assert state == TrailingState.TRAILING
    assert stop is not None
    assert stop == pytest.approx(1.100 - 2 * 0.005)
    pos.current_stop = stop

    # Tighten further with higher price.
    state, stop, _ = fsm.tick(position=pos, current_price=1.200, atr=0.005,
                               current_state=state)
    assert state == TrailingState.TRAILING
    assert stop is not None and stop > pos.current_stop


def test_trailing_long_invariant_never_widens() -> None:
    """If the proposed stop is BELOW the current stop, it must not be applied."""
    fsm = TrailingStopFSM()
    pos = _long_pos()
    pos.current_stop = 1.080
    # Price retraces back to 1.10 from a higher peak.
    state, stop, _ = fsm.tick(position=pos, current_price=1.100, atr=0.020,
                               current_state=TrailingState.TRAILING)
    # Proposed atr stop = 1.10 - 0.04 = 1.06 < 1.08, must be rejected.
    assert stop is None


def test_trailing_short_target_cap_at_70_pct() -> None:
    """SHORT must force-close at -70% drop from entry."""
    fsm = TrailingStopFSM(short_target_cap_pct=0.70)
    pos = _short_pos()
    pos.current_stop = 1.000  # already moved down a lot

    # Price has dropped 71% from entry of 2.0 -> 0.58
    state, stop, reason = fsm.tick(
        position=pos, current_price=0.58, atr=0.01,
        current_state=TrailingState.TRAILING,
    )
    assert state == TrailingState.TARGET_REACHED
    # Stop is pinned just above current price (so next tick triggers fill).
    assert stop is not None and stop > 0.58 and stop < 1.000


def test_trailing_short_breakeven_mirror() -> None:
    fsm = TrailingStopFSM()
    pos = _short_pos()  # entry 2.000 stop 2.100; R=0.100
    state = TrailingState.INIT
    state, stop, _ = fsm.tick(position=pos, current_price=1.99, atr=0.01,
                               current_state=state)
    assert state == TrailingState.ARMED
    # +1R -> price 1.90 -> breakeven (stop -> entry)
    state, stop, _ = fsm.tick(position=pos, current_price=1.90, atr=0.01,
                               current_state=state)
    assert state == TrailingState.BREAKEVEN
    assert stop == pytest.approx(2.000)


# --------------------------------------------------------------------- #
# Executor / SR-2 hard-stop pairing
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_executor_open_pairs_entry_with_stop() -> None:
    adapter = FakeAdapter()
    executor = CCXTExecutor(adapter=adapter, exchange_name="binance")
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    decision = gate.evaluate(
        signal=_signal(), account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert decision.approved
    pos = await executor.open(
        symbol="RAVEUSDT", decision=decision, current_price=1.0,
        account=_account(),
    )
    assert pos.stop_order_id
    assert len(adapter.market_orders) == 1
    assert len(adapter.stop_orders) == 1
    assert adapter.stop_orders[0]["reduce_only"] is True


@pytest.mark.asyncio
async def test_executor_emergency_close_when_stop_fails() -> None:
    adapter = FakeAdapter(fail_stop=True)
    executor = CCXTExecutor(
        adapter=adapter, exchange_name="binance", place_stop_retries=1,
        stop_failure_cooldown_sec=3600,
    )
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    decision = gate.evaluate(
        signal=_signal(), account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert decision.approved

    account = _account()
    with pytest.raises(Exception) as exc_info:
        await executor.open(
            symbol="RAVEUSDT", decision=decision, current_price=1.0, account=account,
        )
    assert "stop_placement_failed" in str(exc_info.value)
    # Two market orders: the entry, then the emergency close.
    assert len(adapter.market_orders) == 2
    # Symbol cooldown is set.
    assert "RAVEUSDT" in account.cooldown_until_ts_ms


@pytest.mark.asyncio
async def test_executor_tighten_stop_replaces_old() -> None:
    adapter = FakeAdapter()
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    decision = gate.evaluate(
        signal=_signal(), account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    pos = await executor.open(
        symbol="RAVEUSDT", decision=decision, current_price=1.0, account=_account(),
    )
    old_id = pos.stop_order_id

    ok = await executor.tighten_hard_stop(pos, new_stop=1.000)
    assert ok
    assert pos.current_stop == pytest.approx(1.000)
    assert pos.stop_order_id and pos.stop_order_id != old_id
    assert old_id in adapter.cancelled


@pytest.mark.asyncio
async def test_executor_replace_failure_restores_old_stop() -> None:
    adapter = FakeAdapter(replace_fails=True)
    executor = CCXTExecutor(adapter=adapter)
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    decision = gate.evaluate(
        signal=_signal(), account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    pos = await executor.open(
        symbol="RAVEUSDT", decision=decision, current_price=1.0, account=_account(),
    )
    # Make the SECOND place_stop_order call fail (the replace).
    # FakeAdapter.replace_fails triggers on _stop_called > 1, so subsequent
    # tighten attempt fails. After failure, executor tries to RESTORE old.
    # But _stop_called would be > 1 again, also failing. Let's relax the
    # replace_fails behavior for the restore attempt.
    adapter.replace_fails = False
    # First call (cancel) ok; second (replace) was supposed to fail; but now we
    # re-enable success for the restore. Reset counter so restore succeeds.
    adapter._stop_called = 0
    # Re-enable failure only on the immediate replace.
    original_place = adapter.place_stop_order
    fail_once = {"n": 0}
    async def place_one_failure(*args, **kwargs):  # noqa: ANN001
        fail_once["n"] += 1
        if fail_once["n"] == 1:
            raise RuntimeError("simulated replace failure")
        return await original_place(*args, **kwargs)
    adapter.place_stop_order = place_one_failure  # type: ignore[assignment]

    ok = await executor.tighten_hard_stop(pos, new_stop=1.000)
    assert not ok
    # The position is NOT naked: a restored stop has an id.
    assert pos.stop_order_id is not None


# --------------------------------------------------------------------- #
# Reconciler
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reconciler_clean_state_marks_complete() -> None:
    adapter = FakeAdapter()
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = AccountState()
    report = await rec.run(a)
    assert report.success
    assert a.reconciliation_complete


@pytest.mark.asyncio
async def test_reconciler_protects_orphan_with_no_stop() -> None:
    adapter = FakeAdapter()
    adapter.fetched_positions = [
        {"symbol": "GHOSTUSDT", "side": "long", "contracts": 100.0,
         "entryPrice": 1.000},
    ]
    adapter.fetched_orders = []   # no protective stop
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = AccountState()
    report = await rec.run(a)
    assert report.success
    assert report.orphans_found == 1
    assert report.orphans_protected == 1
    assert len(adapter.stop_orders) == 1


@pytest.mark.asyncio
async def test_reconciler_does_not_fire_on_already_protected_orphan() -> None:
    adapter = FakeAdapter()
    adapter.fetched_positions = [
        {"symbol": "GHOSTUSDT", "side": "long", "contracts": 100.0,
         "entryPrice": 1.000},
    ]
    adapter.fetched_orders = [
        {"symbol": "GHOSTUSDT", "type": "stop_market", "reduceOnly": True},
    ]
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = AccountState()
    report = await rec.run(a)
    assert report.orphans_found == 1
    assert report.orphans_protected == 0


# --------------------------------------------------------------------- #
# SR-1 dynamic slippage formula sanity
# --------------------------------------------------------------------- #


def test_sr1_slippage_formula_matches_doc() -> None:
    gate = RiskGate(PositionSizer(), RiskGateConfig(base_slippage=0.03))
    # 5x -> 3.00%
    assert math.isclose(gate._dynamic_slippage_cap(5.0), 0.03, rel_tol=1e-3)
    # 10x -> ~2.12%
    assert math.isclose(gate._dynamic_slippage_cap(10.0), 0.03 / math.sqrt(2.0), rel_tol=1e-3)
    # 15x -> ~1.73%
    assert math.isclose(gate._dynamic_slippage_cap(15.0), 0.03 / math.sqrt(3.0), rel_tol=1e-3)
