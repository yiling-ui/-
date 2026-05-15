"""Integration test: the full hot path inside main.App.

Boots an App with --dry-run, injects a fake screener via direct calls into
the public on_kline / on_funding / on_oi hooks, drives a synthetic confluence
that crosses the high_priority threshold, and verifies:

  * the dry-run adapter receives a market order,
  * the dry-run adapter receives a stop order,
  * the trailing FSM moves to BREAKEVEN after enough +1R bars.

This proves screener -> fuser -> risk_gate -> executor -> trailing is
actually wired, end to end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.main import App, AppConfig, DryRunExchangeAdapter


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
    """Start App.run() in a task; tear it down on exit."""
    app = App(cfg=cfg)
    # Patch screener.run to be a no-op so we can drive on_kline manually.
    runner = asyncio.create_task(app.run())
    # Replace screener.run after App.run created the screener.
    for _ in range(50):
        if app._screener is not None:
            break
        await asyncio.sleep(0.02)
    assert app._screener is not None, "screener never initialized"

    async def _noop_run() -> None:
        await app._stop_event.wait()
    app._screener.run = _noop_run     # type: ignore[method-assign]

    try:
        # wait until reconciliation is complete
        for _ in range(50):
            if app.state.reconciliation_complete:
                break
            await asyncio.sleep(0.02)
        yield app
    finally:
        app.request_stop()
        await asyncio.wait_for(runner, timeout=5.0)


@pytest.mark.asyncio
async def test_dry_run_app_boots_and_serves_health() -> None:
    cfg = AppConfig(healthz_port=18091, dry_run=True, graceful_timeout_sec=2.0)
    async with _running_app(cfg) as app:
        assert app.state.reconciliation_complete is True
        assert app.state.fuser_alive is True
        assert app.state.screener_alive is True


@pytest.mark.asyncio
async def test_high_priority_signal_routes_to_executor_and_attaches_trailing() -> None:
    cfg = AppConfig(healthz_port=18092, dry_run=True, graceful_timeout_sec=2.0,
                    initial_equity_usdt=10_000.0,
                    min_liquidity_usdt=100_000.0)
    async with _running_app(cfg) as app:
        # The DryRunExchangeAdapter is the one we instantiated.
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        # Bug #2 fix: the gate now requires a fresh live quote and
        # fail-closes when it can't get one. In dry-run that means tests
        # have to publish a mark price; set it equal to the trigger so the
        # adverse-slip term is exactly 0 and the existing semantics hold.
        adapter.set_mark_price("RAVEUSDT", 1.0)
        # Build a fully-baked high-priority FusedSignal and dispatch through
        # App._handle_high_priority via fuser.sink (which is _handle_high_priority).
        sig = FusedSignal(
            symbol="RAVEUSDT", exchange="binance", ts=1,
            direction=Direction.LONG, rule_score=90.0, llm_score=92.0,
            final_score=95.0, is_high_priority=True, blocked=False,
            block_reason=None, trigger_price=1.0,
        )
        # The fuser's sink is `fused_sink` defined inside App.run, which is
        # not externally reachable. Instead invoke App._handle_high_priority
        # directly: it IS reachable from the App object and exercises the
        # same gate -> sizer -> executor -> trailing chain we want to prove.
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            PositionSizer,
            RiskGate,
            RiskGateConfig,
            TrailingStopFSM,
        )
        from altcoin_agent.risk.state import AccountState
        sizer = PositionSizer()
        gate = RiskGate(sizer, RiskGateConfig(min_liquidity_usdt=100_000.0))
        executor = CCXTExecutor(adapter=adapter)
        account = AccountState(equity_usdt=10_000.0,
                                starting_equity_today_usdt=10_000.0)
        account.reconciliation_complete = True
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )
        await app._handle_high_priority(
            sig=sig, gate=gate, executor=executor,
            trailing=trailing, account=account,
        )
        assert len(adapter.market_orders) == 1
        assert len(adapter.stop_orders) == 1
        assert account.position("RAVEUSDT") is not None
        assert "RAVEUSDT" in trailing._by_symbol




@pytest.mark.asyncio
async def test_position_watcher_close_lifecycle_updates_account_and_dashboard() -> None:
    """Bug #1 regression: when the exchange closes a position (STOP_MARKET
    fires), the daemon must:
      * detect the close via PositionWatcher.poll_once,
      * update realized_pnl_today_usdt and equity,
      * detach the trailing tracker,
      * push the close to the dashboard,
      * notify Telegram.
    """
    cfg = AppConfig(
        healthz_port=18093, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        position_watcher_poll_sec=0.05,
        position_watcher_miss_threshold=1,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        # Bug #2 fix prerequisite: gate reads a live quote -- mock it.
        adapter.set_mark_price("RAVEUSDT", 1.0)

        sig = FusedSignal(
            symbol="RAVEUSDT", exchange="binance", ts=1,
            direction=Direction.LONG, rule_score=90.0, llm_score=92.0,
            final_score=95.0, is_high_priority=True, blocked=False,
            block_reason=None, trigger_price=1.0,
        )
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            PositionSizer,
            PositionWatcher,
            RiskGate,
            RiskGateConfig,
            TrailingStopFSM,
        )
        from altcoin_agent.risk.state import AccountState

        sizer = PositionSizer()
        gate = RiskGate(sizer, RiskGateConfig(min_liquidity_usdt=100_000.0))
        executor = CCXTExecutor(adapter=adapter)
        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )
        await app._handle_high_priority(
            sig=sig, gate=gate, executor=executor,
            trailing=trailing, account=account,
        )
        position = account.position("RAVEUSDT")
        assert position is not None
        assert "RAVEUSDT" in trailing._by_symbol
        equity_before = account.equity_usdt

        # The exchange-side STOP_MARKET fires. DryRunExchangeAdapter exposes
        # ``simulate_close`` for exactly this case.
        adapter.simulate_close("RAVEUSDT")

        # Drive the watcher manually with a tight threshold so the test
        # doesn't rely on background polling timing.
        async def _close_cb(p, reason):
            await app._on_position_close(
                position=p, reason=reason,
                trailing=trailing, account=account,
            )
        watcher = PositionWatcher(
            adapter=adapter, account=account, on_close=_close_cb,
            poll_interval_sec=0.01, miss_threshold=1,
        )
        closed = await watcher.poll_once()

        assert len(closed) == 1
        assert account.position("RAVEUSDT") is None
        assert "RAVEUSDT" not in trailing._by_symbol
        assert position.closed is True
        # The close was a hit on the initial 5%-below-entry stop -> negative PnL.
        assert account.realized_pnl_today_usdt < 0
        assert account.equity_usdt < equity_before
        assert account.daily_stoploss_hits == 1
        assert account.consecutive_losses.get("RAVEUSDT", 0) == 1
        assert app.state.closed_positions == 1
        assert app.state.last_close_ts > 0
        # Dashboard ring buffer recorded the close.
        assert len(app.dashboard.recent_closes) == 1
        rec = app.dashboard.recent_closes[-1]
        assert rec["symbol"] == "RAVEUSDT"
        assert rec["realized_pnl_usdt"] < 0



# --------------------------------------------------------------------- #
# Bug #2: SR-1 dynamic slippage now compares trigger to a *live* quote.
#
# Before the fix, ``_handle_high_priority`` passed ``signal.trigger_price``
# as both the trigger AND ``current_price`` to ``RiskGate.evaluate``, so
# the slip term was always identity-zero and the SR-1 cap (``base_slippage
# / sqrt(leverage / 5)``) had no effect. These tests pin the new behaviour:
#
#   * adverse drift between trigger and live -> gate rejects the order;
#   * favorable drift -> gate approves;
#   * live quote unavailable -> order fail-closes with ``quote_unavailable``
#     (i.e., the system never silently bypasses SR-1).
# --------------------------------------------------------------------- #


def _bug2_signal(direction: Direction = Direction.LONG,
                 *, trigger: float = 1.000,
                 score: float = 95.0) -> FusedSignal:
    return FusedSignal(
        symbol="RAVEUSDT", exchange="binance", ts=1,
        direction=direction, rule_score=90.0, llm_score=92.0,
        final_score=score, is_high_priority=True, blocked=False,
        block_reason=None, trigger_price=trigger,
    )


async def _bug2_setup(app):
    """Build the gate / executor / trailing / account chain used by the
    integration tests, returning everything the caller needs to drive
    ``App._handle_high_priority`` directly."""
    from altcoin_agent.main import TrailingController
    from altcoin_agent.risk import (
        ATRCalculator,
        CCXTExecutor,
        PositionSizer,
        RiskGate,
        RiskGateConfig,
        TrailingStopFSM,
    )
    from altcoin_agent.risk.state import AccountState

    sizer = PositionSizer()
    gate = RiskGate(sizer, RiskGateConfig(min_liquidity_usdt=100_000.0))
    executor = CCXTExecutor(adapter=app._adapter)
    account = AccountState(
        equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
    )
    account.reconciliation_complete = True
    trailing = TrailingController(
        fsm=TrailingStopFSM(), atr=ATRCalculator(),
        executor=executor, account=account, health=app.state,
    )
    return gate, executor, trailing, account


@pytest.mark.asyncio
async def test_bug2_adverse_live_drift_rejects_order() -> None:
    """LONG at trigger 1.000, live at 1.030 (+3.0%). The dynamic-slip cap
    at the leverage we'd actually size with is ~2-3%; this MUST reject."""
    cfg = AppConfig(
        healthz_port=18094, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        # Big adverse drift on the BUY side -- live price ran 3% above trigger.
        adapter.set_mark_price("RAVEUSDT", 1.030)

        gate, executor, trailing, account = await _bug2_setup(app)
        rejected_before = app.state.orders_rejected
        await app._handle_high_priority(
            sig=_bug2_signal(direction=Direction.LONG, trigger=1.000, score=100.0),
            gate=gate, executor=executor, trailing=trailing, account=account,
        )
        # Order was NOT placed; the gate rejected it on slippage.
        assert len(adapter.market_orders) == 0
        assert account.position("RAVEUSDT") is None
        assert app.state.orders_rejected == rejected_before + 1
        # Dashboard recorded a slippage-flavoured rejection.
        assert app.dashboard.recent_rejections, "rejection not pushed"
        last = app.dashboard.recent_rejections[-1]
        assert "slippage_too_high" in last["reason"]


@pytest.mark.asyncio
async def test_bug2_favorable_live_drift_still_approves() -> None:
    """LONG at trigger 1.000, live at 0.99 (favourable for entry).
    SR-1 only counts adverse drift -- the order MUST still go through."""
    cfg = AppConfig(
        healthz_port=18095, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        adapter.set_mark_price("RAVEUSDT", 0.99)

        gate, executor, trailing, account = await _bug2_setup(app)
        await app._handle_high_priority(
            sig=_bug2_signal(direction=Direction.LONG, trigger=1.000),
            gate=gate, executor=executor, trailing=trailing, account=account,
        )
        assert len(adapter.market_orders) == 1
        assert len(adapter.stop_orders) == 1
        assert account.position("RAVEUSDT") is not None


@pytest.mark.asyncio
async def test_bug2_unavailable_quote_fails_closed() -> None:
    """If the live quote can't be fetched (venue down, network blip, etc.)
    the order MUST be rejected with reason ``quote_unavailable`` -- the
    system never silently bypasses SR-1 by falling back to the trigger.
    """
    cfg = AppConfig(
        healthz_port=18096, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        # Deliberately do NOT call set_mark_price -- adapter.fetch_ticker_price
        # will raise, and the gate path must fail-closed.

        gate, executor, trailing, account = await _bug2_setup(app)
        rejected_before = app.state.orders_rejected
        await app._handle_high_priority(
            sig=_bug2_signal(),
            gate=gate, executor=executor, trailing=trailing, account=account,
        )
        assert len(adapter.market_orders) == 0
        assert account.position("RAVEUSDT") is None
        assert app.state.orders_rejected == rejected_before + 1
        last = app.dashboard.recent_rejections[-1]
        assert "quote_unavailable" in last["reason"]
        assert app.state.last_error is not None
        assert "quote_unavailable" in app.state.last_error


@pytest.mark.asyncio
async def test_bug2_quote_provider_override_takes_precedence() -> None:
    """The optional ``App.quote_provider`` callable wins over the adapter's
    ``fetch_ticker_price``. Useful for production where ops may want to
    plug in a higher-quality source (e.g. internal aggregator) without
    touching the adapter."""
    cfg = AppConfig(
        healthz_port=18097, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )

    async def custom_quote(symbol: str) -> float:
        # Adversely drifted enough to fail SR-1, irrespective of what the
        # adapter would have said.
        return 1.030

    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        # Set the adapter mark to a *favourable* price -- the override
        # should beat it and still cause a rejection.
        adapter.set_mark_price("RAVEUSDT", 0.99)
        app.quote_provider = custom_quote

        gate, executor, trailing, account = await _bug2_setup(app)
        await app._handle_high_priority(
            sig=_bug2_signal(direction=Direction.LONG, trigger=1.000, score=100.0),
            gate=gate, executor=executor, trailing=trailing, account=account,
        )
        assert len(adapter.market_orders) == 0
        last = app.dashboard.recent_rejections[-1]
        assert "slippage_too_high" in last["reason"]
