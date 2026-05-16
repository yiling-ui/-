"""Integration tests for PR-A fixes that aren't covered by the unit tests:

  * Multi-leg close PnL uses ``avg_entry_price`` and ``total_size``,
    not the legacy leg-0 ``entry_price``/``size`` fields. Without this
    fix, a rolled position closing at the trailing stop would credit
    phantom PnL to the daily ledger.

  * ``RollingController.reset_for_symbol`` is called on every close so
    a re-entry on the same symbol starts with an empty fired-thresholds
    set. Without this, residual ``{1.5, 3.0}`` markers from the prior
    position would suppress the first roll on the new one.

  * The wiring path (AppConfig.rolling_enabled -> RollingController ->
    TrailingController.rolling) actually exists end-to-end so flipping
    the operator switch in YAML produces a controller in the running
    app.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from altcoin_agent.main import App, AppConfig, DryRunExchangeAdapter
from altcoin_agent.risk import (
    AccountState,
    Position,
    PositionLeg,
    RollingConfig,
    Side,
    TrailingStopFSM,
)


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
    """Same harness as test_main_integration_mock._running_app."""
    app = App(cfg=cfg)
    runner = asyncio.create_task(app.run())
    for _ in range(50):
        if app._screener is not None:
            break
        await asyncio.sleep(0.02)
    assert app._screener is not None

    async def _noop_run() -> None:
        await app._stop_event.wait()
    app._screener.run = _noop_run     # type: ignore[method-assign]

    try:
        for _ in range(50):
            if app.state.reconciliation_complete:
                break
            await asyncio.sleep(0.02)
        yield app
    finally:
        app.request_stop()
        await asyncio.wait_for(runner, timeout=5.0)


def _make_rolled_long_position(
    *,
    symbol: str = "RAVEUSDT",
    leg0_entry: float = 1.000,
    leg0_size: float = 100.0,
    leg1_entry: float = 1.100,
    leg1_size: float = 50.0,
    initial_stop: float = 0.950,
    current_stop: float = 1.050,    # trailing already moved past breakeven
) -> Position:
    """Build a 2-leg LONG position with a known weighted-avg entry."""
    pos = Position(
        symbol=symbol,
        exchange="binance",
        side=Side.LONG,
        entry_price=leg0_entry,
        size=leg0_size + leg1_size,   # legacy field synced to total
        leverage=5.0,
        initial_stop=initial_stop,
        current_stop=current_stop,
        stop_order_id="stop-resized",
    )
    pos.legs.append(PositionLeg(
        leg_id=0, side=Side.LONG, size=leg0_size, entry_price=leg0_entry,
        margin_source="initial",
    ))
    pos.legs.append(PositionLeg(
        leg_id=1, side=Side.LONG, size=leg1_size, entry_price=leg1_entry,
        margin_source="rolled_unrealized",
    ))
    return pos


# --------------------------------------------------------------------- #
# Multi-leg close PnL
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_uses_avg_entry_price_for_multi_leg() -> None:
    """A 2-leg LONG closing at 1.050:
        leg 0: 100 @ 1.000
        leg 1:  50 @ 1.100
        avg_entry = (100*1.0 + 50*1.1) / 150 = 1.0333...
        fill = current_stop = 1.050
        true PnL = (1.050 - 1.0333) * 150 = +2.50
    The buggy path used leg-0 entry directly:
        buggy PnL = (1.050 - 1.000) * 150 = +7.50 — three times too high
    so the daily-DD math and consec-loss tracking would be wrong.
    """
    cfg = AppConfig(
        healthz_port=18301, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)

        # Build the supporting fixtures _on_position_close needs.
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            TrailingStopFSM,
        )
        executor = CCXTExecutor(adapter=adapter)
        account = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )

        pos = _make_rolled_long_position()
        account.open_positions[pos.symbol] = pos
        # Simulate the watcher's pre-callback bookkeeping (it normally
        # runs on a fetched-positions miss).
        account.open_positions.pop(pos.symbol, None)
        pos.closed = True

        equity_before = account.equity_usdt
        await app._on_position_close(
            position=pos, reason="stop_filled",
            trailing=trailing, account=account,
        )

        # True PnL = (1.050 - 1.03333...) * 150 = ~2.50
        # Allow a small tolerance for floating point.
        expected_pnl = (1.050 - (100*1.0 + 50*1.1) / 150) * 150
        assert account.realized_pnl_today_usdt == pytest.approx(expected_pnl, rel=1e-6)
        assert account.equity_usdt == pytest.approx(equity_before + expected_pnl, rel=1e-9)

        # +2.50 PnL is a tiny but positive close -> not a loss, so the
        # 3-strike counter is not bumped.
        assert account.daily_stoploss_hits == 0


@pytest.mark.asyncio
async def test_close_handles_buggy_legacy_path_when_no_legs() -> None:
    """V1.0 single-leg path: ``Position`` with empty ``legs`` falls back
    to the legacy ``size`` / ``entry_price`` so existing behaviour is
    preserved verbatim. Pre-fix and post-fix should agree on this case.
    """
    cfg = AppConfig(
        healthz_port=18302, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import ATRCalculator, CCXTExecutor

        executor = CCXTExecutor(adapter=adapter)
        account = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )

        # Single-leg position, ``legs`` empty -> legacy path.
        pos = Position(
            symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=100.0, leverage=5.0,
            initial_stop=0.95, current_stop=1.02,
        )
        assert pos.legs == []
        account.open_positions[pos.symbol] = pos
        account.open_positions.pop(pos.symbol, None)
        pos.closed = True

        await app._on_position_close(
            position=pos, reason="stop_filled",
            trailing=trailing, account=account,
        )
        # PnL = (1.02 - 1.00) * 100 = +2.0
        assert account.realized_pnl_today_usdt == pytest.approx(2.0, rel=1e-6)


# --------------------------------------------------------------------- #
# reset_for_symbol on close
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_invokes_rolling_reset_for_symbol() -> None:
    """When a position closes, the rolling controller's per-symbol
    fired-thresholds set must be cleared so a re-entry on the same
    symbol can fire its first R level again.
    """
    cfg = AppConfig(
        healthz_port=18303, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        rolling_enabled=True,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        # Wiring assertion: cfg.rolling_enabled flipping to True actually
        # produces an attached controller and stashes it on _rolling.
        assert app._rolling is not None, "rolling controller not wired"
        assert app._rolling.cfg.enabled is True

        # Pre-populate fired-thresholds as if a previous position had
        # already crossed +1.5R and +3.0R.
        app._rolling._fired_levels["RAVEUSDT"] = {1.5, 3.0}
        app._rolling._last_roll_ts["RAVEUSDT"] = 1_000_000

        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            TrailingStopFSM,
        )
        executor = CCXTExecutor(adapter=adapter)
        account = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )

        pos = _make_rolled_long_position()
        account.open_positions[pos.symbol] = pos
        account.open_positions.pop(pos.symbol, None)
        pos.closed = True

        await app._on_position_close(
            position=pos, reason="stop_filled",
            trailing=trailing, account=account,
        )
        assert "RAVEUSDT" not in app._rolling._fired_levels
        assert "RAVEUSDT" not in app._rolling._last_roll_ts


@pytest.mark.asyncio
async def test_close_safe_when_rolling_disabled() -> None:
    """With ``cfg.rolling_enabled=False``, ``app._rolling`` is None and
    ``_on_position_close`` must still complete without error -- the
    reset call is a guarded no-op."""
    cfg = AppConfig(
        healthz_port=18304, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        rolling_enabled=False,
    )
    async with _running_app(cfg) as app:
        assert app._rolling is None
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import ATRCalculator, CCXTExecutor

        executor = CCXTExecutor(adapter=app._adapter)
        account = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )
        pos = _make_rolled_long_position()
        account.open_positions[pos.symbol] = pos
        account.open_positions.pop(pos.symbol, None)
        pos.closed = True

        # Must not raise.
        await app._on_position_close(
            position=pos, reason="stop_filled",
            trailing=trailing, account=account,
        )


# --------------------------------------------------------------------- #
# Wiring sanity check
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rolling_wires_into_trailing_controller_when_enabled() -> None:
    """End-to-end: cfg.rolling_enabled=True must produce a
    RollingController whose maybe_roll the trailing controller will
    call on each kline tick."""
    cfg = AppConfig(
        healthz_port=18305, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        rolling_enabled=True,
        rolling_trigger_r_levels=(1.5, 3.0),
        rolling_unrealized_pnl_ratio=0.4,
        rolling_leg_stop_pct=0.03,
        rolling_max_legs_per_symbol=2,
    )
    async with _running_app(cfg) as app:
        # The controller exists and reflects the config.
        rc = app._rolling
        assert rc is not None
        assert rc.cfg.enabled is True
        assert rc.cfg.trigger_r_levels == (1.5, 3.0)
        assert rc.cfg.unrealized_pnl_ratio == pytest.approx(0.4)
        assert rc.cfg.leg_stop_pct == pytest.approx(0.03)
        assert rc.cfg.max_legs_per_symbol == 2


def test_rolling_config_validates_yaml_finger_trouble() -> None:
    """A YAML typo (95 instead of 0.95 for the ratio) must abort the
    daemon at config-load time rather than silently sizing 30x notionals
    in production."""
    with pytest.raises(ValueError, match="unrealized_pnl_ratio"):
        RollingConfig(unrealized_pnl_ratio=95.0)
    with pytest.raises(ValueError, match="leg_stop_pct"):
        RollingConfig(leg_stop_pct=0.5)   # 50% leg stop is nonsense
    with pytest.raises(ValueError, match="strictly increasing"):
        RollingConfig(trigger_r_levels=(3.0, 1.5))
    with pytest.raises(ValueError, match="post-breakeven"):
        # Below 1R = before the trailing FSM has even moved to breakeven.
        RollingConfig(trigger_r_levels=(0.5, 1.5))
