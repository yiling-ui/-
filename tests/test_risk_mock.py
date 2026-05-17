"""Mock tests for risk + execution package (sizing, gate, trailing, executor)."""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.risk import (
    AccountState,
    CCXTExecutor,
    DynamicLeverageConfig,
    Position,
    PositionSizer,
    PositionWatcher,
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

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False, client_order_id=None):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "client_order_id": client_order_id,
               "symbol": symbol, "side": side.value, "size": size,
               "amount": size, "filled": size, "remaining": 0.0,
               "status": "closed",
               "average": price or 1.0, "reduce_only": reduce_only, "price": price}
        self.market_orders.append(rec)
        return rec

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True, *, client_order_id=None):  # noqa: ANN001
        self._stop_called += 1
        if self.fail_stop:
            raise RuntimeError("simulated stop placement failure")
        if self.replace_fails and self._stop_called > 1:
            raise RuntimeError("simulated replace failure")
        oid = self._id()
        rec = {"id": oid, "client_order_id": client_order_id,
               "symbol": symbol, "side": side.value, "size": size,
               "stop_price": stop_price, "reduce_only": reduce_only,
               "amount": size, "filled": 0.0, "remaining": size,
               "status": "open", "average": 0.0, "price": 0.0}
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
        symbol="RAVEUSDT", exchange="binance",
        ts=int(time.time() * 1000),
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


def test_sizer_clamps_notional_to_leverage_cap_on_tight_stop() -> None:
    """Bug #1 regression: a 0.05% stop on $10k equity used to produce
    $300k notional (= 30x equity), well past the configured 15x long cap.
    The clamp must hold the notional at exactly ``equity * leverage``,
    and the returned ``risk_amount`` must reflect the *actual* dollar
    risk that ends up at exchange, not the configured ``max_risk_per_trade``.
    """
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            max_leverage_long=15.0, max_leverage_short=10.0,
        ),
    )
    entry, stop = 100.0, 99.95          # 0.05% stop -> would risk-parity to 30x
    size, notional, risk = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=entry, initial_stop=stop,
        leverage=15.0,
    )
    # Clamped to equity * leverage
    assert notional == pytest.approx(150_000.0)
    assert size == pytest.approx(1_500.0)
    # Actual risk is stop_distance * size = 0.05 * 1500 = $75
    # (significantly LESS than the $150 risk_amount the formula would
    # have implied; the operator now sees the real exposure).
    assert risk == pytest.approx(75.0)


def test_sizer_does_not_clamp_when_risk_parity_within_leverage_cap() -> None:
    """Wide stops (5% here) produce a small notional that's well inside
    the leverage cap; no clamping should occur, and risk_amount stays
    at exactly ``equity * max_risk_per_trade``."""
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            max_leverage_long=15.0, max_leverage_short=10.0,
        ),
    )
    size, notional, risk = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=1.000, initial_stop=0.95,
        leverage=10.0,
    )
    # Risk-parity: notional = 150 / 0.05 = 3000, well under 10k * 10 = 100k
    assert notional == pytest.approx(3000.0, rel=1e-3)
    assert risk == pytest.approx(150.0, rel=1e-3)


def test_sizer_clamps_to_short_cap_when_leverage_passed_in() -> None:
    """SHORT side cap is tighter (10x default). With a tight stop the
    clamp must respect that lower ceiling, not the 15x long cap."""
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            max_leverage_long=15.0, max_leverage_short=10.0,
        ),
    )
    size, notional, _ = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=100.0, initial_stop=100.05,
        leverage=10.0,
    )
    assert notional == pytest.approx(100_000.0)   # 10k * 10x
    assert size == pytest.approx(1_000.0)


def test_sizer_default_leverage_used_when_omitted() -> None:
    """Back-compat: legacy callers that don't pass ``leverage`` still get a
    safe behaviour — clamped to ``max_leverage_long`` (the side-agnostic
    upper bound) so notional can never exceed configured caps."""
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            max_leverage_long=15.0, max_leverage_short=10.0,
        ),
    )
    size, notional, _ = sizer.compute_size(
        equity_usdt=10_000.0, entry_price=100.0, initial_stop=99.95,
    )
    # Defaulted to 15x -> notional capped at $150k
    assert notional == pytest.approx(150_000.0)
    assert size == pytest.approx(1_500.0)


def test_sizer_rejects_zero_or_negative_leverage() -> None:
    sizer = PositionSizer(max_risk_per_trade=0.015)
    for bad in (0.0, -1.0):
        size, notional, _ = sizer.compute_size(
            equity_usdt=10_000.0, entry_price=1.0, initial_stop=0.95,
            leverage=bad,
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


def test_gate_tight_stop_does_not_exceed_leverage_cap() -> None:
    """End-to-end Bug #1 regression: a 0.05% sweep stop on $10k equity used
    to size to $300k notional (30x), tripping Binance's leverage limit.
    The gate must now produce ``notional <= equity * leverage`` so the
    order will actually rest cleanly on the venue."""
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            max_leverage_long=15.0, max_leverage_short=10.0,
        ),
    )
    gate = RiskGate(sizer)
    decision = gate.evaluate(
        signal=_signal(score=100.0, trigger=100.0),
        account=_account(),
        current_price=100.0,
        top5_depth_usdt=400_000,
        realized_vol_pct=0.05,
        initial_stop=99.95,            # 0.05% — the buggy case
    )
    assert decision.approved
    assert decision.leverage is not None
    # The hard invariant: notional <= equity * leverage. Pre-fix this was
    # 30x; post-fix it must respect the configured cap.
    assert decision.notional_usdt is not None
    assert decision.notional_usdt <= 10_000.0 * decision.leverage + 1e-6


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



# --------------------------------------------------------------------- #
# PositionWatcher (Bug #1)
# --------------------------------------------------------------------- #


def _open_position(symbol: str = "RAVEUSDT", *, size: float = 100.0,
                   side: Side = Side.LONG) -> Position:
    return Position(
        symbol=symbol, exchange="binance", side=side,
        entry_price=1.000, size=size, leverage=10.0,
        initial_stop=0.950, current_stop=0.950,
        stop_order_id="stop-1",
    )


@pytest.mark.asyncio
async def test_position_watcher_no_close_when_position_still_present() -> None:
    adapter = FakeAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos
    adapter.fetched_positions = [
        {"symbol": pos.symbol, "side": "long", "contracts": pos.size,
         "entryPrice": pos.entry_price},
    ]

    closes: list[tuple[str, str]] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append((p.symbol, reason))

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=2,
    )
    # Two polls in a row, position is still there both times.
    await watcher.poll_once()
    await watcher.poll_once()
    assert closes == []
    assert pos.symbol in a.open_positions


@pytest.mark.asyncio
async def test_position_watcher_fires_on_close_after_threshold() -> None:
    adapter = FakeAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos
    # Exchange side: position no longer present (got closed by STOP_MARKET).
    adapter.fetched_positions = []

    closes: list[tuple[str, str]] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append((p.symbol, reason))

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=2,
    )
    # First poll = first miss; should not fire yet.
    await watcher.poll_once()
    assert closes == []
    assert pos.symbol in a.open_positions

    # Second poll = threshold reached; fires.
    await watcher.poll_once()
    assert closes == [(pos.symbol, "exchange_close_detected")]
    assert pos.symbol not in a.open_positions
    assert pos.closed is True


@pytest.mark.asyncio
async def test_position_watcher_treats_zero_size_as_closed() -> None:
    """Some venues return rows with size=0 instead of dropping them."""
    adapter = FakeAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos
    adapter.fetched_positions = [
        {"symbol": pos.symbol, "side": "long", "contracts": 0.0,
         "entryPrice": pos.entry_price},
    ]

    closes: list[Position] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append(p)

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=1,
    )
    await watcher.poll_once()
    assert len(closes) == 1
    assert pos.symbol not in a.open_positions


@pytest.mark.asyncio
async def test_position_watcher_resets_misses_when_position_returns() -> None:
    """A transient drop in one snapshot followed by a recovery must NOT
    cause a false close."""
    adapter = FakeAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos

    closes: list[Position] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append(p)

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=3,
    )

    # Miss #1
    adapter.fetched_positions = []
    await watcher.poll_once()
    # Recovers: position visible again.
    adapter.fetched_positions = [
        {"symbol": pos.symbol, "side": "long", "contracts": pos.size,
         "entryPrice": pos.entry_price},
    ]
    await watcher.poll_once()
    # Miss again, but counter must have reset.
    adapter.fetched_positions = []
    await watcher.poll_once()
    await watcher.poll_once()
    # Two misses after reset (< threshold of 3) -> no close yet.
    assert closes == []
    assert pos.symbol in a.open_positions


@pytest.mark.asyncio
async def test_position_watcher_swallows_fetch_errors() -> None:
    """fetch_positions raising must NOT advance miss counts."""

    class ExplodingAdapter(FakeAdapter):
        async def fetch_positions(self):  # type: ignore[override]
            raise RuntimeError("api timeout")

    adapter = ExplodingAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos

    closes: list[Position] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append(p)

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=1,
    )
    # Even with miss_threshold=1, a fetch error must not falsely close.
    for _ in range(5):
        await watcher.poll_once()
    assert closes == []
    assert pos.symbol in a.open_positions


@pytest.mark.asyncio
async def test_position_watcher_swallows_on_close_exception() -> None:
    """If on_close raises, the watcher still removes the position locally
    so we don't loop forever."""
    adapter = FakeAdapter()
    pos = _open_position()
    a = AccountState()
    a.open_positions[pos.symbol] = pos
    adapter.fetched_positions = []

    async def on_close(p: Position, reason: str) -> None:
        raise RuntimeError("downstream blew up")

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=1,
    )
    closed = await watcher.poll_once()
    assert len(closed) == 1
    assert pos.symbol not in a.open_positions
    assert pos.closed is True


@pytest.mark.asyncio
async def test_position_watcher_aggregates_hedge_mode_rows() -> None:
    """Hedge-mode shorts can return as negative ``contracts`` and may
    appear as two rows; we abs-sum so we don't conclude false-close."""
    adapter = FakeAdapter()
    pos = _open_position(size=100.0)
    a = AccountState()
    a.open_positions[pos.symbol] = pos
    adapter.fetched_positions = [
        {"symbol": pos.symbol, "side": "long", "contracts": 60.0,
         "entryPrice": 1.0},
        {"symbol": pos.symbol, "side": "long", "contracts": 40.0,
         "entryPrice": 1.0},
    ]

    closes: list[Position] = []

    async def on_close(p: Position, reason: str) -> None:
        closes.append(p)

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=1,
    )
    await watcher.poll_once()
    assert closes == []
    assert pos.symbol in a.open_positions


@pytest.mark.asyncio
async def test_position_watcher_run_loop_terminates_on_stop_event() -> None:
    """The long-running ``run()`` must exit promptly when stop_event is set."""
    adapter = FakeAdapter()
    a = AccountState()

    async def on_close(p: Position, reason: str) -> None:
        return None

    watcher = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=0.01, miss_threshold=2,
    )
    import asyncio
    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)



# --------------------------------------------------------------------- #
# AccountState.maybe_roll_over_day (Bug #3)
#
# Without rollover, the daily-drawdown breaker latches one-way: a couple
# of small losing days peg the running sum past 6%, and every subsequent
# signal is rejected forever. Same for the 3-strike rule.
#
# These tests pin down:
#   * idempotent on the same trading day,
#   * resets the right counters on day flip,
#   * preserves consecutive_losses (per-symbol streaks span days),
#   * preserves manual halts (operator must explicitly clear them),
#   * the gate stops rejecting once a rollover happens,
#   * non-default ``rollover_anchor_utc_hour`` honours custom day starts.
# --------------------------------------------------------------------- #

# 2026-05-15 00:00:00 UTC, in milliseconds.
_DAY1_MS = 1_778_803_200_000          # day-1 00:30 UTC
_DAY2_MS = 1_778_803_200_000 + 86_400_000  # +24h: day 2


def _ms_at(year: int, month: int, day: int, hour: int = 12) -> int:
    """Build a UTC ms timestamp deterministically (no time module DST risk)."""
    import datetime as dt
    return int(dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_rollover_first_call_stamps_without_resetting() -> None:
    """First call after construction is the boot stamp — counters mustn't
    move (everything was just initialised)."""
    a = AccountState(
        equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        realized_pnl_today_usdt=-50.0, daily_stoploss_hits=1,
    )
    rolled = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))
    assert not rolled
    assert a.last_rollover_date_utc == "2026-05-15"
    # Untouched.
    assert a.realized_pnl_today_usdt == pytest.approx(-50.0)
    assert a.daily_stoploss_hits == 1
    assert a.starting_equity_today_usdt == pytest.approx(10_000.0)


def test_rollover_idempotent_within_same_utc_day() -> None:
    a = AccountState()
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15, 0))
    a.realized_pnl_today_usdt = -100
    a.daily_stoploss_hits = 2
    # Calls later on the same day are no-ops.
    for h in (1, 6, 12, 18, 23):
        rolled = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15, h))
        assert not rolled
    assert a.realized_pnl_today_usdt == pytest.approx(-100)
    assert a.daily_stoploss_hits == 2


def test_rollover_resets_daily_counters_on_day_flip() -> None:
    """After a losing day, the next-day rollover must:
      * snap starting_equity_today_usdt to current equity,
      * zero realized_pnl_today_usdt,
      * zero daily_stoploss_hits.
    """
    a = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))   # boot stamp

    # Lose money during day 1.
    a.realized_pnl_today_usdt = -650.0
    a.equity_usdt = 9_350.0
    a.daily_stoploss_hits = 3

    # Day flips.
    rolled = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16, 0))
    assert rolled
    assert a.last_rollover_date_utc == "2026-05-16"
    assert a.starting_equity_today_usdt == pytest.approx(9_350.0)
    assert a.realized_pnl_today_usdt == pytest.approx(0.0)
    assert a.daily_stoploss_hits == 0
    # ``equity_usdt`` itself is the running balance; rollover must not
    # change it.
    assert a.equity_usdt == pytest.approx(9_350.0)
    # And daily_drawdown_pct now reads 0 again (off the new anchor).
    assert a.daily_drawdown_pct == pytest.approx(0.0)


def test_rollover_preserves_consecutive_losses_and_cooldowns() -> None:
    """consecutive_losses is a per-symbol streak, not a daily quota — it
    must survive the day flip. Same for symbol cooldowns (the wall-clock
    deadline reaches its natural expiry on its own)."""
    a = AccountState()
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))

    a.consecutive_losses["RAVEUSDT"] = 2
    a.cooldown_until_ts_ms["RAVEUSDT"] = _ms_at(2026, 5, 16, 4)

    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16))
    assert a.consecutive_losses["RAVEUSDT"] == 2
    assert a.cooldown_until_ts_ms["RAVEUSDT"] == _ms_at(2026, 5, 16, 4)


def test_rollover_preserves_manual_global_halt() -> None:
    """An operator-set hard halt must NOT auto-clear at midnight — it's a
    deliberate intervention. Restarting trading requires explicit ops."""
    a = AccountState()
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))
    a.halt("manual: investigation in progress")

    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16))
    assert a.global_trading_halted is True
    assert a.halt_reason == "manual: investigation in progress"


def test_rollover_unlatches_daily_drawdown_breaker() -> None:
    """End-to-end gate behaviour: after a losing day trips the daily-DD
    breaker, a rollover must let signals through again on the next day.
    Pre-fix, the breaker was a one-way latch."""
    a = _account()
    a.equity_usdt = 9_300.0          # already lost $700 = 7%
    a.realized_pnl_today_usdt = -700
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))  # boot stamp
    # Force the boot stamp to actually have a *previous* day so the next
    # call is a real rollover.
    a.last_rollover_date_utc = "2026-05-15"

    gate = RiskGate(PositionSizer(), RiskGateConfig(daily_drawdown_limit=0.06))
    blocked = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not blocked.approved
    assert "daily_drawdown" in blocked.reason

    # Day flips. starting_equity_today_usdt re-anchors to the new $9,300
    # equity and PnL/DD start fresh at zero.
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16))
    approved = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert approved.approved


def test_rollover_unlatches_three_strike_breaker() -> None:
    """Three stops in a day -> halt. Day flips -> back to clean slate."""
    a = _account()
    a.daily_stoploss_hits = 3
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))
    a.last_rollover_date_utc = "2026-05-15"

    gate = RiskGate(PositionSizer(), RiskGateConfig(daily_stoploss_hits_max=3))
    d = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert not d.approved and "stoploss_hits" in d.reason

    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16))
    d2 = gate.evaluate(
        signal=_signal(), account=a, current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert d2.approved


def test_rollover_anchor_at_8utc_shifts_day_boundary() -> None:
    """With anchor=8, a "trading day" runs 08:00 UTC -> 08:00 UTC. So
    07:59 UTC and 08:00 UTC must straddle a flip; 08:01 UTC and 23:59
    UTC of the same anchor day must NOT."""
    a = AccountState(rollover_anchor_utc_hour=8)
    # Boot at 08:30 UTC on day 1.
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15, 8))
    boot_day = a.last_rollover_date_utc
    a.realized_pnl_today_usdt = -50

    # Same anchor day: 23:59 UTC on May 15.
    rolled_late = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15, 23))
    assert not rolled_late
    assert a.realized_pnl_today_usdt == pytest.approx(-50)

    # Same anchor day: 07:59 UTC on May 16 — still inside day 1.
    rolled_early = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16, 7))
    assert not rolled_early
    assert a.realized_pnl_today_usdt == pytest.approx(-50)
    assert a.last_rollover_date_utc == boot_day

    # 08:00 UTC on May 16 — anchor crosses, day flips.
    rolled = a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16, 8))
    assert rolled
    assert a.last_rollover_date_utc != boot_day
    assert a.realized_pnl_today_usdt == pytest.approx(0.0)


def test_rollover_with_zero_starting_equity_still_safe() -> None:
    """A degenerate starting_equity (e.g., empty paper-trade boot) must
    not cause divide-by-zero in daily_drawdown_pct after a rollover."""
    a = AccountState(equity_usdt=0.0, starting_equity_today_usdt=0.0)
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 15))
    a.last_rollover_date_utc = "2026-05-15"
    a.maybe_roll_over_day(now_ms=_ms_at(2026, 5, 16))
    assert a.daily_drawdown_pct == 0.0
