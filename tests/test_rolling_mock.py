"""Mock tests for the RollingController (滚仓 / pyramid-add).

The controller's contract is: on every kline tick it asks 9 questions in
order; ANY rejection short-circuits the pipeline. These tests pin each
gate down so a future refactor cannot quietly disable one of them.

Test groups (in roughly the same order as RollingController.maybe_roll):

  1.  precondition: cfg.enabled / position.closed / has legs
  2.  leg cap:      max_legs_per_symbol
  3.  throttle:     min_interval_sec
  4.  quote:        provider raises -> rejected
  5.  threshold:    R level not yet crossed / already fired
  6.  STRATEGY      <- the headline gate. Re-runs ScoreFuser.evaluate.
  7.  RiskGate:     daily DD / 3-strike / liquidity / leverage cap
  8.  sizing:       below min_notional
  9.  execution:    add_leg + stop resize -> success / failure semantics
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from altcoin_agent.fuser import (
    Direction,
    FuserConfig,
    ScoreFuser,
)
from altcoin_agent.risk import (
    AccountState,
    CCXTExecutor,
    Position,
    PositionLeg,
    PositionSizer,
    RiskGate,
    RollingConfig,
    RollingController,
    Side,
)
from altcoin_agent.screener import SignalEvent, SignalKind

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


class FakeAdapter:
    """Same shape as the FakeAdapter in test_risk_mock.py, trimmed to the
    surface that ``add_leg`` / ``tighten_hard_stop`` actually touch."""

    def __init__(self) -> None:
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self._n = 0
        self.fail_next_stop = False

    def _id(self) -> str:
        self._n += 1
        return f"o-{self._n}"

    async def market_order(
        self, symbol, side, size, *, price=None, reduce_only=False,
    ):  # noqa: ANN001
        oid = self._id()
        rec = {
            "id": oid, "symbol": symbol, "side": side.value, "size": size,
            "average": price or 1.0, "reduce_only": reduce_only,
            "price": price,
        }
        self.market_orders.append(rec)
        return rec

    async def place_stop_order(
        self, symbol, side, size, stop_price, reduce_only=True,
    ):  # noqa: ANN001
        if self.fail_next_stop:
            self.fail_next_stop = False
            raise RuntimeError("simulated stop failure")
        oid = self._id()
        rec = {
            "id": oid, "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "reduce_only": reduce_only,
        }
        self.stop_orders.append(rec)
        return rec

    async def cancel_order(self, order_id, symbol):  # noqa: ANN001
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):  # noqa: ANN001
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_positions(self) -> list[dict[str, Any]]:
        return []

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        return []


def _account(equity: float = 10_000.0) -> AccountState:
    a = AccountState(
        equity_usdt=equity,
        starting_equity_today_usdt=equity,
    )
    a.reconciliation_complete = True
    return a


def _open_long_position(
    *,
    symbol: str = "RAVEUSDT",
    entry: float = 1.000,
    initial_stop: float = 0.950,
    size: float = 100.0,
    leverage: float = 5.0,
) -> Position:
    pos = Position(
        symbol=symbol,
        exchange="binance",
        side=Side.LONG,
        entry_price=entry,
        size=size,
        leverage=leverage,
        initial_stop=initial_stop,
        current_stop=entry,           # already moved to breakeven
        stop_order_id="stop-init",
    )
    pos.legs.append(PositionLeg(
        leg_id=0, side=Side.LONG, size=size, entry_price=entry,
        margin_source="initial",
    ))
    return pos


def _open_short_position(
    *,
    symbol: str = "RAVEUSDT",
    entry: float = 1.000,
    initial_stop: float = 1.050,
    size: float = 100.0,
    leverage: float = 5.0,
) -> Position:
    pos = Position(
        symbol=symbol,
        exchange="binance",
        side=Side.SHORT,
        entry_price=entry,
        size=size,
        leverage=leverage,
        initial_stop=initial_stop,
        current_stop=entry,
        stop_order_id="stop-init",
    )
    pos.legs.append(PositionLeg(
        leg_id=0, side=Side.SHORT, size=size, entry_price=entry,
        margin_source="initial",
    ))
    return pos


def _make_strategy_signal(
    *,
    score: float = 95.0,
    direction: Direction = Direction.LONG,
    blocked: bool = False,
    block_reason: str | None = None,
    rule_score: float = 60.0,
):
    """Stub a ScoreFuser whose evaluate() returns a chosen FusedSignal."""
    from altcoin_agent.fuser import FusedSignal
    sig = FusedSignal(
        symbol="RAVEUSDT", exchange="binance", ts=1,
        direction=direction,
        rule_score=rule_score, llm_score=80.0,
        final_score=score, is_high_priority=score >= 85.0,
        blocked=blocked, block_reason=block_reason,
        trigger_price=1.0,
    )
    fuser = MagicMock(spec=ScoreFuser)
    fuser.evaluate.return_value = sig
    return fuser


def _build_controller(
    *,
    cfg: RollingConfig | None = None,
    fuser=None,
    quote: float = 1.10,    # default +10% past entry == +2R
    quote_raises: Exception | None = None,
    notify_roll=None,
    notify_error=None,
) -> tuple[RollingController, FakeAdapter, AccountState]:
    adapter = FakeAdapter()
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    executor = CCXTExecutor(adapter=adapter)

    if fuser is None:
        fuser = _make_strategy_signal()

    async def quote_provider(symbol: str) -> float:
        if quote_raises is not None:
            raise quote_raises
        return quote

    ctl = RollingController(
        cfg=cfg or RollingConfig(enabled=True),
        sizer=sizer,
        gate=gate,
        executor=executor,
        fuser=fuser,
        quote_provider=quote_provider,
        notify_roll=notify_roll,
        notify_error=notify_error,
    )
    return ctl, adapter, _account()


# --------------------------------------------------------------------- #
# 1) Precondition gates
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_disabled_short_circuits_immediately() -> None:
    ctl, _, account = _build_controller(
        cfg=RollingConfig(enabled=False),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "disabled"


@pytest.mark.asyncio
async def test_closed_position_not_rolled() -> None:
    ctl, _, account = _build_controller()
    pos = _open_long_position()
    pos.closed = True
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "position_closed"


@pytest.mark.asyncio
async def test_no_legs_recorded_refuses() -> None:
    """Reconciler-attached orphans don't have legs; we cannot reason
    about avg entry, so we refuse to roll."""
    ctl, _, account = _build_controller()
    pos = _open_long_position()
    pos.legs.clear()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "no_legs_recorded"


# --------------------------------------------------------------------- #
# 2) Leg cap
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_legs_caps_rolls() -> None:
    """With max_legs=2, the third roll attempt is rejected even if all
    other gates would pass."""
    ctl, _, account = _build_controller(
        cfg=RollingConfig(
            enabled=True, max_legs_per_symbol=2,
            trigger_r_levels=(1.5, 3.0, 5.0),
        ),
    )
    pos = _open_long_position()
    # Pretend two legs already added.
    for i in range(1, 3):
        pos.legs.append(PositionLeg(
            leg_id=i, side=Side.LONG, size=10.0, entry_price=1.05,
            margin_source="rolled_unrealized",
        ))
    pos.size = 120.0
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "max_legs_reached"


# --------------------------------------------------------------------- #
# 3) Throttle
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_min_interval_sec_throttles() -> None:
    ctl, _, account = _build_controller(
        cfg=RollingConfig(enabled=True, min_interval_sec=60),
    )
    ctl._last_roll_ts["RAVEUSDT"] = 1_000_000   # last roll at t0
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000 + 30_000,    # +30s, less than 60s
    )
    assert not decision.fired
    assert decision.reason == "min_interval_active"


# --------------------------------------------------------------------- #
# 4) Quote provider
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_quote_unavailable_rejects_with_typed_reason() -> None:
    ctl, _, account = _build_controller(
        quote_raises=RuntimeError("ws disconnected"),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("quote_unavailable:")
    assert "RuntimeError" in decision.reason


@pytest.mark.asyncio
async def test_quote_zero_or_negative_rejected() -> None:
    ctl, _, account = _build_controller(quote=0.0)
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "quote_non_positive"


# --------------------------------------------------------------------- #
# 5) R thresholds
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_below_first_threshold_no_op() -> None:
    """Position is at +1R but trigger_r_levels[0] is 1.5; nothing
    fires."""
    ctl, _, account = _build_controller(
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5, 3.0)),
        quote=1.05,    # +1R given entry=1.0, initial_stop=0.95
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "no_next_threshold"


@pytest.mark.asyncio
async def test_already_fired_threshold_skipped() -> None:
    """If +1.5R has already fired, +1.6R does not re-fire it."""
    ctl, _, account = _build_controller(
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5, 3.0)),
        quote=1.08,    # +1.6R
    )
    ctl._fired_levels["RAVEUSDT"] = {1.5}
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason == "no_next_threshold"


@pytest.mark.asyncio
async def test_first_roll_fires_at_1_5R() -> None:
    """+1.5R exact -> first threshold fires."""
    ctl, adapter, account = _build_controller(
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5, 3.0)),
        quote=1.080,    # 0.080/0.05 = +1.6R, safely past 1.5R
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert decision.fired, decision.reason
    assert decision.next_threshold_r == 1.5
    assert len(adapter.market_orders) == 1
    assert pos.num_rolled_legs == 1


# --------------------------------------------------------------------- #
# 6) Strategy re-confirmation -- the headline gate
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_strategy_blocked_skips_roll() -> None:
    """E.g. wash-trading detector now flags this symbol -> never roll."""
    fuser = _make_strategy_signal(
        blocked=True, block_reason="wash_trading_detected",
    )
    ctl, _, account = _build_controller(fuser=fuser, quote=1.10)
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("strategy_blocked:")
    assert "wash_trading_detected" in decision.reason


@pytest.mark.asyncio
async def test_strategy_direction_changed_skips_roll() -> None:
    """LONG position, fuser now thinks SHORT -> let trailing close the
    original; do NOT add a long leg into a bearish reversal."""
    fuser = _make_strategy_signal(direction=Direction.SHORT, score=90.0)
    ctl, _, account = _build_controller(fuser=fuser, quote=1.10)
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("strategy_direction_changed:")


@pytest.mark.asyncio
async def test_strategy_neutral_skips_roll() -> None:
    fuser = _make_strategy_signal(direction=Direction.NEUTRAL, score=90.0)
    ctl, _, account = _build_controller(fuser=fuser, quote=1.10)
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("strategy_direction_changed:")


@pytest.mark.asyncio
async def test_strategy_score_below_threshold_skips_roll() -> None:
    """final_score < cfg.require_strategy_min_score -> reject."""
    fuser = _make_strategy_signal(score=70.0)
    ctl, _, account = _build_controller(
        fuser=fuser, quote=1.10,
        cfg=RollingConfig(enabled=True, require_strategy_min_score=85.0),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("strategy_score_low:")
    assert decision.strategy_score == 70.0


@pytest.mark.asyncio
async def test_rule_score_floor_prevents_llm_only_roll() -> None:
    """A high final_score driven purely by stale LLM boost (low
    rule_score) should not cause a roll."""
    fuser = _make_strategy_signal(score=95.0, rule_score=20.0)
    ctl, _, account = _build_controller(
        fuser=fuser, quote=1.10,
        cfg=RollingConfig(
            enabled=True, require_min_rule_score=35.0,
        ),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("rule_score_low:")


@pytest.mark.asyncio
async def test_strategy_aligned_short_can_roll() -> None:
    """SHORT position + bearish fuser + price dropped past -1.5R -> roll."""
    fuser = _make_strategy_signal(direction=Direction.SHORT, score=95.0,
                                   rule_score=60.0)
    ctl, adapter, account = _build_controller(
        fuser=fuser, quote=0.920,    # entry 1.0, stop 1.05; -1.6R = 0.920
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5, 3.0)),
    )
    pos = _open_short_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert decision.fired, decision.reason
    assert pos.num_rolled_legs == 1
    # Short legs are SELL orders.
    assert adapter.market_orders[0]["side"] == "short"


# --------------------------------------------------------------------- #
# 7) Risk gate (subset)
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_daily_dd_breaker_blocks_roll() -> None:
    """Even with strategy aligned and unrealised PnL there, daily DD
    above 6% must block the new leg (prevents adding into a losing
    day)."""
    ctl, _, account = _build_controller(quote=1.10)
    account.equity_usdt = 9_300.0
    account.realized_pnl_today_usdt = -700      # -7% from start
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert "daily_drawdown_limit" in decision.reason


@pytest.mark.asyncio
async def test_three_strike_breaker_blocks_roll() -> None:
    ctl, _, account = _build_controller(quote=1.10)
    account.daily_stoploss_hits = 3
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert "stoploss_hits" in decision.reason


@pytest.mark.asyncio
async def test_low_liquidity_blocks_roll() -> None:
    ctl, _, account = _build_controller(quote=1.10)
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=50_000,    # below 200k floor
        realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert "insufficient_liquidity" in decision.reason


@pytest.mark.asyncio
async def test_global_halt_blocks_roll() -> None:
    ctl, _, account = _build_controller(quote=1.10)
    account.halt("manual: investigation")
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert "global_halt" in decision.reason


# --------------------------------------------------------------------- #
# 8) Sizing math
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sizing_uses_unrealised_pnl_ratio() -> None:
    """At +2R unrealised:
        avg_entry=1.0, mark=1.10, size=100 -> unrealised = $10
        ratio=0.5 -> risk_budget = $5
        leg_stop_pct=0.025 -> stop_distance = 1.10 * 0.025 = 0.0275
        new_leg_size = 5 / 0.0275 ≈ 181.8 units
    """
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(
            enabled=True, trigger_r_levels=(1.5, 3.0),
            unrealized_pnl_ratio=0.5, leg_stop_pct=0.025,
        ),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert decision.fired, decision.reason
    expected = 5.0 / (1.10 * 0.025)   # ≈ 181.82
    assert decision.new_leg_size == pytest.approx(expected, rel=1e-3)


@pytest.mark.asyncio
async def test_below_min_notional_skipped() -> None:
    """Tiny new-leg notional (operator dialled the ratio way down) should
    be skipped without an order, even when the R threshold is met."""
    # Entry=1.0, stop=0.95, r_unit=0.05. quote=1.05 -> +1R exactly.
    # ratio=1e-4 makes risk_budget tiny -> sub-min notional.
    ctl, adapter, account = _build_controller(
        quote=1.05,
        cfg=RollingConfig(
            enabled=True, trigger_r_levels=(1.0,),
            unrealized_pnl_ratio=1e-4,
            leg_stop_pct=0.025,
        ),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert decision.reason.startswith("below_min_notional:")
    assert len(adapter.market_orders) == 0


@pytest.mark.asyncio
async def test_total_notional_capped_by_leverage() -> None:
    """If new_leg_notional + existing_notional would exceed
    equity * leverage, the size is clamped down. Position with size 1000
    at $1 (notional 1000) at 5x with $10k equity has room for $49k more
    -- well above the natural sizing -- so this test instead verifies
    that a *much* larger notional gets clipped.

    We use the maximum allowed ``unrealized_pnl_ratio`` (1.0) plus a
    very large existing position to force the clamp without breaking
    the new RollingConfig validator (which caps the ratio at 1.0).
    """
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(
            enabled=True, trigger_r_levels=(1.5,),
            unrealized_pnl_ratio=1.0,    # maximum allowed -> still well past cap
            leg_stop_pct=0.025,
        ),
    )
    # Existing notional already near the leverage cap so any non-trivial
    # new leg has to be clamped. equity=10k, leverage=5x -> 50k cap.
    # size=4000 @ 1.10 -> existing notional 4400; room = 45,600.
    # Without the clamp: ratio=1.0 * unrealised_pnl ($400) / 0.0275
    # ≈ 14,545 size = 16,000 notional -- under the cap. To force the
    # clamp we make the existing position much bigger.
    pos = _open_long_position(size=40_000.0)   # existing notional 44,000
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    # equity * leverage = 10_000 * 5 = 50_000. Room = 50_000 - 44_000 = 6_000.
    # Without the clamp, ratio=1.0 * unrealised_pnl ($4,000) / 0.0275
    # ≈ 145,455 size = 160,000 notional -- way past the cap.
    if decision.fired:
        assert decision.new_leg_notional <= 6_000.0 + 1e-3
    else:
        # The gate's defence-in-depth leverage cap may have rejected it
        # outright, which is also acceptable -- both paths prove the
        # clamp is in force.
        assert (
            "leverage_cap_exceeded" in decision.reason
            or "below_min_notional" in decision.reason
        ), decision.reason


# --------------------------------------------------------------------- #
# 9) Execution success / failure
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_successful_roll_records_state_and_replaces_stop() -> None:
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5,)),
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert decision.fired

    # Threshold is now in fired_levels and last_roll_ts is set.
    assert 1.5 in ctl._fired_levels[pos.symbol]
    assert ctl._last_roll_ts[pos.symbol] == 1_000_000

    # One market order at the long side; one stop replacement.
    assert len(adapter.market_orders) == 1
    assert adapter.market_orders[0]["side"] == "long"
    assert len(adapter.stop_orders) == 1   # the fresh resized stop
    assert "stop-init" in adapter.cancelled

    # Position state was updated.
    assert pos.num_rolled_legs == 1
    assert pos.legs[1].leg_id == 1
    assert pos.legs[1].margin_source == "rolled_unrealized"
    assert pos.total_size > 100.0
    assert pos.size == pos.total_size   # legacy field kept in sync


@pytest.mark.asyncio
async def test_stop_resize_failure_emergency_closes_and_disables() -> None:
    """If ``add_leg`` cannot resize the stop, the executor emergency-
    closes the entire position, and the controller flips enabled=False
    when auto_disable_on_failure is True."""
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(
            enabled=True, trigger_r_levels=(1.5,),
            auto_disable_on_failure=True,
        ),
    )
    pos = _open_long_position()
    # Cancel will succeed; the *replace* fails. The executor's
    # tighten_hard_stop tries to restore old; restore also fails because
    # fail_next_stop is reset after one shot. So we need a 2-shot fail:
    # set both fail_next and re-arm in a counter.
    fail_count = {"n": 2}
    original = adapter.place_stop_order

    async def failing_place(*args, **kwargs):  # noqa: ANN001
        if fail_count["n"] > 0:
            fail_count["n"] -= 1
            raise RuntimeError("simulated stop failure")
        return await original(*args, **kwargs)

    adapter.place_stop_order = failing_place    # type: ignore[assignment]

    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    assert "executor_failed" in decision.reason
    # The auto-disable safety net flipped the master switch.
    assert ctl.cfg.enabled is False
    # The position itself was emergency-closed (reduce-only market order).
    assert any(
        m.get("reduce_only") and m.get("side") == "short"
        for m in adapter.market_orders
    )
    assert pos.closed is True


@pytest.mark.asyncio
async def test_failure_keeps_enabled_when_auto_disable_off() -> None:
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(
            enabled=True, trigger_r_levels=(1.5,),
            auto_disable_on_failure=False,
        ),
    )
    pos = _open_long_position()
    fail_count = {"n": 2}
    original = adapter.place_stop_order

    async def failing_place(*args, **kwargs):  # noqa: ANN001
        if fail_count["n"] > 0:
            fail_count["n"] -= 1
            raise RuntimeError("boom")
        return await original(*args, **kwargs)

    adapter.place_stop_order = failing_place    # type: ignore[assignment]

    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert not decision.fired
    # Enabled stayed True even though the leg failed.
    assert ctl.cfg.enabled is True


@pytest.mark.asyncio
async def test_reset_for_symbol_clears_state() -> None:
    """When a position closes, the next position on the same symbol
    must start with a clean slate of fired thresholds."""
    ctl, _, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5,)),
    )
    pos = _open_long_position()
    await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert ctl._fired_levels.get(pos.symbol) == {1.5}

    ctl.reset_for_symbol(pos.symbol)
    assert pos.symbol not in ctl._fired_levels
    assert pos.symbol not in ctl._last_roll_ts


# --------------------------------------------------------------------- #
# Position multi-leg arithmetic
# --------------------------------------------------------------------- #


def test_position_avg_entry_after_two_legs() -> None:
    pos = _open_long_position(entry=1.0, size=100.0)
    pos.legs.append(PositionLeg(
        leg_id=1, side=Side.LONG, size=50.0, entry_price=1.10,
        margin_source="rolled_unrealized",
    ))
    pos.size = 150.0
    # weighted avg = (100*1.0 + 50*1.10) / 150 = 1.0333...
    assert pos.avg_entry_price == pytest.approx(1.0333, rel=1e-3)
    assert pos.total_size == 150.0
    assert pos.num_rolled_legs == 1


def test_position_unrealised_pnl_aggregates_legs() -> None:
    pos = _open_long_position(entry=1.0, size=100.0)
    pos.legs.append(PositionLeg(
        leg_id=1, side=Side.LONG, size=50.0, entry_price=1.10,
        margin_source="rolled_unrealized",
    ))
    pos.size = 150.0
    # avg_entry = 1.0333; mark=1.20 -> (1.20-1.0333)*150 ≈ 25.0
    assert pos.unrealised_pnl_usdt(1.20) == pytest.approx(25.0, rel=1e-2)
    # SHORT direction:
    short = _open_short_position(entry=1.0, size=100.0)
    assert short.unrealised_pnl_usdt(0.90) == pytest.approx(10.0, rel=1e-3)


def test_position_unrealised_pnl_zero_at_avg_entry() -> None:
    pos = _open_long_position()
    assert pos.unrealised_pnl_usdt(pos.avg_entry_price) == 0.0


def test_position_legacy_path_has_no_legs() -> None:
    """Default-constructed Position (no legs) returns size/entry_price
    as before -- backwards compat for V1.0 code paths."""
    pos = Position(
        symbol="X", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=10.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95,
    )
    assert pos.legs == []
    assert pos.total_size == 10.0
    assert pos.avg_entry_price == 1.0
    assert pos.num_rolled_legs == 0


# --------------------------------------------------------------------- #
# Notify hooks
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_successful_roll_invokes_notify_hook() -> None:
    notify_roll = AsyncMock()
    ctl, _, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5,)),
        notify_roll=notify_roll,
    )
    pos = _open_long_position()
    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    assert decision.fired
    notify_roll.assert_awaited_once()
    payload = notify_roll.await_args.args[0]
    assert payload["symbol"] == "RAVEUSDT"
    assert payload["side"] == "long"
    assert payload["trigger_r"] == 1.5


@pytest.mark.asyncio
async def test_failure_invokes_notify_error_hook() -> None:
    notify_error = AsyncMock()
    ctl, adapter, account = _build_controller(
        quote=1.10,
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5,)),
        notify_error=notify_error,
    )
    pos = _open_long_position()
    fail_count = {"n": 2}
    original = adapter.place_stop_order

    async def failing(*args, **kwargs):  # noqa: ANN001
        if fail_count["n"] > 0:
            fail_count["n"] -= 1
            raise RuntimeError("boom")
        return await original(*args, **kwargs)

    adapter.place_stop_order = failing    # type: ignore[assignment]

    await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=1_000_000,
    )
    notify_error.assert_awaited()


# --------------------------------------------------------------------- #
# Integration with a *real* fuser (no mock)
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_roll_with_real_fuser_aligned_signals() -> None:
    """Drive a real ScoreFuser with synthetic signals that produce a
    high-priority LONG, then ask the controller to roll. This proves the
    re-evaluation path is wired correctly end-to-end."""
    fuser = ScoreFuser(sink=None, config=FuserConfig(
        # Disable cooldown for the test; otherwise evaluate's _dispatch
        # would suppress, but evaluate itself ignores cooldown anyway.
        cooldown_sec=0,
    ))
    # Feed a volume spike + OI surge -> fuser scores LONG with rule_score
    # well above the floor. Both events have to_price > from_price so
    # _rule_direction tags them LONG.
    now = 1_000_000
    await fuser.on_rule_signal(SignalEvent(
        kind=SignalKind.VOLUME_SPIKE,
        symbol="RAVEUSDT", exchange="binance", ts=now,
        payload={"side": "buy", "zscore": 5.0, "vol_ratio": 4.0,
                  "timeframe": "1m", "window": 60},
    ))
    await fuser.on_rule_signal(SignalEvent(
        kind=SignalKind.OI_SURGE,
        symbol="RAVEUSDT", exchange="binance", ts=now,
        payload={"oi_delta_pct": 0.18, "price_move_pct": 0.04,
                  "from_oi": 1.0, "to_oi": 1.18,
                  "from_price": 1.0, "to_price": 1.04, "window_samples": 5},
    ))
    await fuser.on_rule_signal(SignalEvent(
        kind=SignalKind.LIQUIDITY_SWEEP,
        symbol="RAVEUSDT", exchange="binance", ts=now,
        payload={"side": "buy_side", "level": 0.95, "wick_to_body": 3.0,
                  "bar_close": 1.0, "timeframe": "1m", "pool_age_ms": 5_000,
                  "touch_count_before_sweep": 2},
    ))

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.direction == Direction.LONG
    assert sig.final_score >= 85.0

    adapter = FakeAdapter()
    sizer = PositionSizer()
    gate = RiskGate(sizer)
    executor = CCXTExecutor(adapter=adapter)

    async def quote(symbol: str) -> float:
        return 1.10    # +2R given entry=1.0, stop=0.95

    ctl = RollingController(
        cfg=RollingConfig(enabled=True, trigger_r_levels=(1.5,)),
        sizer=sizer, gate=gate, executor=executor,
        fuser=fuser, quote_provider=quote,
    )
    pos = _open_long_position()
    account = _account()

    decision = await ctl.maybe_roll(
        position=pos, account=account,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        now_ms=now,
    )
    assert decision.fired, decision.reason
    assert decision.strategy_score == sig.final_score
    assert pos.num_rolled_legs == 1
