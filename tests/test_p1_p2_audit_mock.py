"""tests/test_p1_p2_audit_mock.py — P1 (006-011) + P2 (012-014, 016) audit fixes.

Each test pins ONE invariant of one ticket so a regression is triagable
in seconds. The structure mirrors the operator's audit ticket numbering
so a failure points directly at the offending ticket.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from altcoin_agent.ai_engine import (
    AIVerdict,
    LLMEngine,
    SMCContext,
)
from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.main import (
    HealthState,
    TrailingController,
    _Tracked,
    make_health_app,
)
from altcoin_agent.pipeline import DelayedPostMortemScheduler
from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk import (
    ATRCalculator,
    CCXTExecutor,
    ClusterCapConfig,
    ClusterMap,
    PositionSizer,
    PositionWatcher,
    RegimeFilter,
    RegimeFilterConfig,
    RiskGate,
    RiskGateConfig,
    Side,
    TrailingState,
)
from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.state import AccountState, Position
from altcoin_agent.screener import Kline

# --------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------- #


def _now_ms() -> int:
    return int(time.time() * 1000)


def _signal(
    *,
    direction: Direction = Direction.LONG,
    score: float = 95.0,
    trigger: float = 1.0,
    ts: int | None = None,
) -> FusedSignal:
    return FusedSignal(
        symbol="PEPEUSDT", exchange="binance",
        ts=ts if ts is not None else _now_ms(),
        direction=direction,
        rule_score=90.0, llm_score=80.0,
        final_score=score,
        is_high_priority=True, blocked=False, block_reason=None,
        trigger_price=trigger,
    )


def _account() -> AccountState:
    a = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
    a.reconciliation_complete = True
    return a


def _gate() -> RiskGate:
    return RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))


# --------------------------------------------------------------------- #
# TICKET-006 — signal freshness gate
# --------------------------------------------------------------------- #


def test_ticket006_fresh_signal_passes_freshness_gate() -> None:
    """A signal with ``ts == now_ms`` must NOT be rejected as stale."""
    sig = _signal(ts=_now_ms())
    decision = _gate().evaluate(
        signal=sig, account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
    )
    assert decision.approved
    assert decision.reason == "ok"


def test_ticket006_stale_signal_is_rejected() -> None:
    """A signal whose ``ts`` is older than ``max_signal_age_sec`` must
    be rejected with ``signal_stale:..ms>..ms``."""
    now = _now_ms()
    # 30 seconds old > default 10s cap
    sig = _signal(ts=now - 30_000)
    decision = _gate().evaluate(
        signal=sig, account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
        now_ms=now,
    )
    assert not decision.approved
    assert decision.reason.startswith("signal_stale:")
    # Magnitude is correctly reported.
    assert "30" in decision.reason  # ms count starts with 30...


def test_ticket006_synthetic_low_ts_bypasses_freshness() -> None:
    """``ts=1`` (clearly synthetic, not a real ms timestamp) must NOT
    be rejected as stale — keeps unit-test fixtures useful without
    forcing every test to time-travel.
    """
    sig = _signal(ts=1)
    decision = _gate().evaluate(
        signal=sig, account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
        now_ms=_now_ms(),
    )
    assert decision.approved


def test_ticket006_disabled_when_max_age_zero() -> None:
    """Setting ``max_signal_age_sec=0`` must disable the gate even
    for a clearly-stale real timestamp (operator opt-out)."""
    gate = RiskGate(
        PositionSizer(),
        RiskGateConfig(min_liquidity_usdt=100_000, max_signal_age_sec=0),
    )
    now = _now_ms()
    sig = _signal(ts=now - 86_400_000)  # 24h old
    decision = gate.evaluate(
        signal=sig, account=_account(), current_price=1.0,
        top5_depth_usdt=400_000, realized_vol_pct=0.05, initial_stop=0.95,
        now_ms=now,
    )
    assert decision.approved


# --------------------------------------------------------------------- #
# TICKET-007 — evaluate_rolling re-runs the 4 micro-structure gates
# --------------------------------------------------------------------- #


def _parent_position(side: Side = Side.LONG) -> Position:
    return Position(
        symbol="PEPEUSDT", exchange="binance", side=side,
        entry_price=1.0, size=10.0, leverage=5.0,
        initial_stop=0.95 if side == Side.LONG else 1.05,
        current_stop=0.95 if side == Side.LONG else 1.05,
        stop_order_id="orig",
    )


def test_ticket007_rolling_anti_chase_blocks_after_runup() -> None:
    """A rolling add must be rejected by anti-chase when the tape
    shows a recent runup past the configured cap, even though
    evaluate_rolling skips per-symbol cooldown."""
    tape = PriceTape(
        cfg=PriceTapeConfig(
            anti_chase_window_ms=30_000,
            anti_chase_max_move_pct=0.025,
            vol_kill_window_ms=60_000,
            vol_kill_range_pct=0.99,  # vol-kill effectively off
        )
    )
    now = _now_ms()
    tape.observe("PEPEUSDT", 1.00, now - 25_000)
    tape.observe("PEPEUSDT", 1.10, now)  # +10% in 25s -> chase

    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))
    parent = _parent_position()
    a = _account()
    a.open_positions[parent.symbol] = parent

    decision = gate.evaluate_rolling(
        parent=parent, proposed_size=5.0, proposed_stop=1.07,
        account=a, current_price=1.10,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        trigger_price=1.10, now_ms=now,
        price_tape=tape,
    )
    assert not decision.approved
    assert decision.reason.startswith("chase_too_late:")


def test_ticket007_rolling_vol_kill_blocks_in_whipsaw() -> None:
    tape = PriceTape(
        cfg=PriceTapeConfig(
            anti_chase_window_ms=30_000,
            anti_chase_max_move_pct=0.99,  # anti-chase effectively off
            vol_kill_window_ms=60_000,
            vol_kill_range_pct=0.05,
        )
    )
    now = _now_ms()
    # 10% range inside window
    tape.observe("PEPEUSDT", 1.00, now - 50_000)
    tape.observe("PEPEUSDT", 1.10, now - 30_000)
    tape.observe("PEPEUSDT", 0.99, now - 10_000)
    tape.observe("PEPEUSDT", 1.05, now)

    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))
    parent = _parent_position()
    a = _account()
    a.open_positions[parent.symbol] = parent

    decision = gate.evaluate_rolling(
        parent=parent, proposed_size=5.0, proposed_stop=1.02,
        account=a, current_price=1.05,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        trigger_price=1.05, now_ms=now,
        price_tape=tape,
    )
    assert not decision.approved
    assert decision.reason.startswith("vol_kill_active:")


def test_ticket007_rolling_regime_filter_blocks_long_in_btc_drawdown() -> None:
    rf = RegimeFilter(RegimeFilterConfig(
        reference_symbol="BTC",
        btc_drop_block_long_pct=0.03,
        min_samples=2,
    ))
    rf.observe("BTC", 100.0, _now_ms() - 60_000)
    rf.observe("BTC", 95.0, _now_ms())  # -5% > 3% block

    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))
    parent = _parent_position(Side.LONG)
    a = _account()
    a.open_positions[parent.symbol] = parent

    decision = gate.evaluate_rolling(
        parent=parent, proposed_size=5.0, proposed_stop=0.95,
        account=a, current_price=1.05,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        trigger_price=1.05, now_ms=_now_ms(),
        regime_filter=rf,
    )
    assert not decision.approved
    assert "btc_regime_block_long" in decision.reason


def test_ticket007_rolling_no_optional_gates_falls_through() -> None:
    """When the new optional safety modules are omitted, the rolling
    gate must behave EXACTLY as before (no new false positives)."""
    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))
    parent = _parent_position()
    a = _account()
    a.open_positions[parent.symbol] = parent

    decision = gate.evaluate_rolling(
        parent=parent, proposed_size=50.0, proposed_stop=0.95,
        account=a, current_price=1.05,
        top5_depth_usdt=400_000, realized_vol_pct=0.05,
        trigger_price=1.05, now_ms=_now_ms(),
    )
    assert decision.approved, decision.reason


# --------------------------------------------------------------------- #
# TICKET-008 — PositionWatcher tightened poll + wake-event
# --------------------------------------------------------------------- #


class _StubAdapter:
    """Minimal adapter for the watcher tests."""

    def __init__(self) -> None:
        self.position_snapshots: list[list[dict[str, Any]]] = []
        self._cursor = 0

    async def fetch_positions(self) -> list[dict[str, Any]]:
        if not self.position_snapshots:
            return []
        i = min(self._cursor, len(self.position_snapshots) - 1)
        self._cursor += 1
        return list(self.position_snapshots[i])


@pytest.mark.asyncio
async def test_ticket008_kick_wakes_watcher_immediately() -> None:
    """When ``kick()`` is called between polls, the next loop iteration
    must run a poll right away instead of sleeping the full
    ``poll_interval_sec``.
    """
    adapter = _StubAdapter()
    a = _account()
    a.open_positions["PEPEUSDT"] = _parent_position()
    # First snapshot still has the position; second has none -> close.
    adapter.position_snapshots = [
        [{"symbol": "PEPEUSDT", "contracts": 10.0}],
        [],
        [],
    ]

    closed: list[tuple[Position, str]] = []

    async def on_close(p: Position, reason: str) -> None:
        closed.append((p, reason))

    # Poll interval intentionally LONG so the test timeout would fail
    # without the kick path working.
    pw = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=30.0, miss_threshold=1,
    )
    stop = asyncio.Event()
    runner = asyncio.create_task(pw.run(stop))

    # Give the run loop a beat to do its first poll.
    await asyncio.sleep(0.1)
    pw.kick()  # external "wake up"
    await asyncio.sleep(0.1)
    pw.kick()  # 2nd poll observes the empty snapshot -> close fires
    await asyncio.sleep(0.1)
    stop.set()
    with suppress(BaseException):
        await asyncio.wait_for(runner, timeout=2.0)

    assert closed, "kick() did not produce a close inside the test window"
    pos, reason = closed[0]
    assert pos.symbol == "PEPEUSDT"
    # Default reason when no hint was supplied.
    assert reason == "exchange_close_detected"


@pytest.mark.asyncio
async def test_ticket008_hint_close_reason_also_kicks() -> None:
    """``hint_close_reason`` must set the wake event so the close is
    detected inside the test budget even with a long poll interval."""
    adapter = _StubAdapter()
    a = _account()
    a.open_positions["PEPEUSDT"] = _parent_position()
    adapter.position_snapshots = [
        [{"symbol": "PEPEUSDT", "contracts": 10.0}],
        [],
    ]

    closed: list[tuple[Position, str]] = []

    async def on_close(p: Position, reason: str) -> None:
        closed.append((p, reason))

    pw = PositionWatcher(
        adapter=adapter, account=a, on_close=on_close,
        poll_interval_sec=30.0, miss_threshold=1,
    )
    stop = asyncio.Event()
    runner = asyncio.create_task(pw.run(stop))
    await asyncio.sleep(0.1)
    pw.hint_close_reason("PEPEUSDT", "emergency_close_executor")
    await asyncio.sleep(0.2)
    stop.set()
    with suppress(BaseException):
        await asyncio.wait_for(runner, timeout=2.0)

    assert closed, "hint_close_reason did not propagate kick"
    assert closed[0][1] == "emergency_close_executor"


@pytest.mark.asyncio
async def test_ticket008_last_poll_wall_ts_stamped_on_success() -> None:
    """``last_poll_wall_ts`` must be updated only by successful polls,
    so a long-running fetch_positions failure surfaces as a
    ``position_watcher_lag_sec`` spike on the dashboard."""
    adapter = _StubAdapter()
    a = _account()
    pw = PositionWatcher(
        adapter=adapter, account=a, on_close=lambda p, r: asyncio.sleep(0),
        poll_interval_sec=10.0, miss_threshold=1,
    )
    before = time.time()
    await pw.poll_once()
    assert pw.last_poll_wall_ts >= before


# --------------------------------------------------------------------- #
# TICKET-009 — trailing naked grace window
# --------------------------------------------------------------------- #


class _NakedAdapter:
    def __init__(self) -> None:
        self.market_orders: list[dict[str, Any]] = []
        self.placed_stops: list[dict[str, Any]] = []

    async def market_order(self, symbol, side, size, *, price=None,
                           reduce_only=False, client_order_id=None):  # noqa: ANN001
        self.market_orders.append({
            "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only,
        })
        return {"id": f"mo-{len(self.market_orders)}",
                "amount": size, "filled": size, "status": "closed",
                "average": price or 0.0, "reduce_only": reduce_only}

    async def place_stop_order(self, symbol, side, size, stop_price,
                               reduce_only=True, *, client_order_id=None):  # noqa: ANN001
        self.placed_stops.append({
            "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price,
        })
        return {"id": "ok"}

    async def cancel_order(self, order_id, symbol):  # noqa: ANN001
        return {"status": "ok"}

    async def set_leverage(self, *a, **kw):  # noqa: ANN001
        return {}

    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self):
        return []


class _AlwaysTightenFSM:
    def tick(self, *, position, current_price, atr, current_state):  # noqa: ANN001
        return TrailingState.ARMED, current_price * 0.97, "test_tighten"


@pytest.mark.asyncio
async def test_ticket009_grace_window_keeps_position_during_short_outage() -> None:
    """Within ``naked_grace_sec`` after a tighten/restore double failure,
    the trailing controller must NOT emergency-close — it must keep
    retrying via FSM ticks instead. This is the difference between
    "transient venue 502" and "really naked"."""
    adapter = _NakedAdapter()
    ex = CCXTExecutor(adapter=adapter)
    a = _account()
    pos = Position(
        symbol="PEPE", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="orig",
    )
    a.open_positions["PEPE"] = pos
    health = HealthState()

    async def fake_tighten(p, new_stop):  # noqa: ANN001
        p.stop_order_id = None  # truly naked
        return False
    ex.tighten_hard_stop = fake_tighten  # type: ignore[assignment]

    tc = TrailingController(
        fsm=_AlwaysTightenFSM(), atr=ATRCalculator(),
        executor=ex, account=a, health=health,
        naked_grace_sec=10.0,  # 10s grace
    )
    tc._by_symbol["PEPE"] = _Tracked(position=pos)

    bar1 = Kline(ts=1_000, open=1.05, high=1.06, low=1.04,
                 close=1.05, volume=100.0, timeframe="1m")
    await tc.on_kline("binance", "PEPE", bar1)
    # No emergency close yet; position still open in book.
    assert adapter.market_orders == []
    assert pos.closed is False
    assert "PEPE" in a.open_positions
    # Counter for stop_replace_failure_count was bumped.
    assert health.stop_replace_failure_count == 1
    # Naked timestamp recorded.
    assert tc._by_symbol["PEPE"].naked_since_ts_ms == 1_000


@pytest.mark.asyncio
async def test_ticket009_emergency_close_after_grace_expired() -> None:
    """Once ``naked_grace_sec`` has elapsed without a successful
    re-attach, the emergency close fires."""
    adapter = _NakedAdapter()
    ex = CCXTExecutor(adapter=adapter)
    a = _account()
    pos = Position(
        symbol="PEPE", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="orig",
    )
    a.open_positions["PEPE"] = pos
    health = HealthState()

    async def fake_tighten(p, new_stop):  # noqa: ANN001
        p.stop_order_id = None
        return False
    ex.tighten_hard_stop = fake_tighten  # type: ignore[assignment]

    tc = TrailingController(
        fsm=_AlwaysTightenFSM(), atr=ATRCalculator(),
        executor=ex, account=a, health=health,
        naked_grace_sec=5.0,
    )
    tc._by_symbol["PEPE"] = _Tracked(position=pos)

    bar1 = Kline(ts=1_000, open=1.05, high=1.06, low=1.04,
                 close=1.05, volume=100.0, timeframe="1m")
    bar2 = Kline(ts=8_000, open=1.05, high=1.06, low=1.04,
                 close=1.05, volume=100.0, timeframe="1m")
    await tc.on_kline("binance", "PEPE", bar1)  # grace starts at 1s
    await tc.on_kline("binance", "PEPE", bar2)  # 7s elapsed > 5s -> close

    assert adapter.market_orders, "emergency close not issued after grace"
    assert pos.closed is True
    assert health.emergency_close_count == 1


@pytest.mark.asyncio
async def test_ticket009_grace_resets_after_successful_recovery() -> None:
    """If the FSM tightens successfully on the next bar, the naked
    timestamp must be cleared so a later naked event starts a fresh
    grace window (not a cumulative one)."""
    adapter = _NakedAdapter()
    ex = CCXTExecutor(adapter=adapter)
    a = _account()
    pos = Position(
        symbol="PEPE", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="orig",
    )
    a.open_positions["PEPE"] = pos
    health = HealthState()

    # Toggle: first tighten fails-naked, second succeeds.
    state = {"calls": 0}

    async def maybe_tighten(p, new_stop):  # noqa: ANN001
        state["calls"] += 1
        if state["calls"] == 1:
            p.stop_order_id = None
            return False
        # Recovery
        p.stop_order_id = "recovered"
        p.current_stop = new_stop
        return True
    ex.tighten_hard_stop = maybe_tighten  # type: ignore[assignment]

    tc = TrailingController(
        fsm=_AlwaysTightenFSM(), atr=ATRCalculator(),
        executor=ex, account=a, health=health,
        naked_grace_sec=5.0,
    )
    tc._by_symbol["PEPE"] = _Tracked(position=pos)

    await tc.on_kline("binance", "PEPE",
                     Kline(ts=1_000, open=1.05, high=1.06, low=1.04,
                           close=1.05, volume=100.0, timeframe="1m"))
    assert tc._by_symbol["PEPE"].naked_since_ts_ms == 1_000
    await tc.on_kline("binance", "PEPE",
                     Kline(ts=2_000, open=1.05, high=1.06, low=1.04,
                           close=1.05, volume=100.0, timeframe="1m"))
    # Recovery cleared the naked stamp.
    assert tc._by_symbol["PEPE"].naked_since_ts_ms is None
    assert pos.closed is False
    assert adapter.market_orders == []


# --------------------------------------------------------------------- #
# TICKET-010 — magnitude_pct includes leverage
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ticket010_magnitude_pct_scales_with_leverage(
    tmp_path: Path,
) -> None:
    """A 1.5% LONG win at 5x leverage must record magnitude_pct ≈ 7.5%
    (= 1.5R-equivalent), not 1.5%. The bucket goes from ``pos_small``
    (raw) to ``pos_large`` (leverage-scaled), which is what the rule
    store should learn against."""
    from altcoin_agent.learning_engine import (
        EventResult,
        RuleStore,
        bucket_pct,
    )

    captured: dict[str, Any] = {}

    async def fake_run_post_mortem(**kwargs: Any):  # noqa: ANN401
        captured.update(kwargs)
        rr: EventResult = kwargs["realized_result"]
        # Mirror what RuleStore would do.
        captured["bucket"] = bucket_pct(rr.magnitude_pct)
        # Return a minimal report-shaped object so the scheduler can
        # log it without erroring.
        return type(
            "R",
            (),
            {
                "result": rr,
                "candidates": [],
                "picks": [],
                "updated_rules": [],
            },
        )()

    import altcoin_agent.pipeline as pipeline_mod
    orig = pipeline_mod.run_post_mortem
    pipeline_mod.run_post_mortem = fake_run_post_mortem  # type: ignore[assignment]
    try:
        store = RuleStore(json_path=tmp_path / "rules.json")
        sched = DelayedPostMortemScheduler(store=store, engine=None)
        task = sched.record(
            symbol="PEPEUSDT",
            entry_ts_ms=1_000,
            close_ts_ms=2_000,
            side="long",
            entry_price=1.000,
            fill_price=1.015,    # +1.5% raw
            realized_pnl_usdt=15.0,
            realized_r=1.5,
            close_reason="trailing_take_profit",
            leverage=5.0,
        )
        await task
    finally:
        pipeline_mod.run_post_mortem = orig  # type: ignore[assignment]

    rr: EventResult = captured["realized_result"]
    # 1.5% × 5 = 7.5% → bucket_pct returns "pos_large" (>= 5%)
    assert rr.magnitude_pct == pytest.approx(0.075, rel=1e-6)
    assert captured["bucket"] == "pos_large"


@pytest.mark.asyncio
async def test_ticket010_magnitude_pct_unchanged_without_leverage(
    tmp_path: Path,
) -> None:
    """Back-compat: when ``leverage`` is omitted, magnitude_pct is the
    raw price delta (legacy behaviour)."""
    from altcoin_agent.learning_engine import EventResult, RuleStore

    captured: dict[str, Any] = {}

    async def fake_run_post_mortem(**kwargs: Any):  # noqa: ANN401
        captured.update(kwargs)
        return type("R", (), {
            "result": kwargs["realized_result"],
            "candidates": [], "picks": [], "updated_rules": [],
        })()

    import altcoin_agent.pipeline as pipeline_mod
    orig = pipeline_mod.run_post_mortem
    pipeline_mod.run_post_mortem = fake_run_post_mortem  # type: ignore[assignment]
    try:
        store = RuleStore(json_path=tmp_path / "rules.json")
        sched = DelayedPostMortemScheduler(store=store, engine=None)
        task = sched.record(
            symbol="PEPEUSDT",
            entry_ts_ms=1_000,
            close_ts_ms=2_000,
            side="long",
            entry_price=1.000,
            fill_price=1.015,
            realized_pnl_usdt=15.0,
            realized_r=1.5,
            close_reason="legacy",
            # leverage omitted -> legacy unscaled behaviour
        )
        await task
    finally:
        pipeline_mod.run_post_mortem = orig  # type: ignore[assignment]

    rr: EventResult = captured["realized_result"]
    assert rr.magnitude_pct == pytest.approx(0.015, rel=1e-6)


@pytest.mark.asyncio
async def test_ticket010_magnitude_pct_clamped_at_one(
    tmp_path: Path,
) -> None:
    """A pathological 50x trade with a 5% raw win would otherwise
    magnitude=2.5 and bucket as something nonsensical. We clamp at
    +/- 1.0 so the bucket lands at the ``xlarge`` extremum."""
    from altcoin_agent.learning_engine import EventResult, RuleStore

    captured: dict[str, Any] = {}

    async def fake_run_post_mortem(**kwargs: Any):  # noqa: ANN401
        captured.update(kwargs)
        return type("R", (), {
            "result": kwargs["realized_result"],
            "candidates": [], "picks": [], "updated_rules": [],
        })()

    import altcoin_agent.pipeline as pipeline_mod
    orig = pipeline_mod.run_post_mortem
    pipeline_mod.run_post_mortem = fake_run_post_mortem  # type: ignore[assignment]
    try:
        store = RuleStore(json_path=tmp_path / "rules.json")
        sched = DelayedPostMortemScheduler(store=store, engine=None)
        task = sched.record(
            symbol="PEPEUSDT",
            entry_ts_ms=1_000,
            close_ts_ms=2_000,
            side="long",
            entry_price=1.000,
            fill_price=1.05,      # +5% raw
            realized_pnl_usdt=250.0,
            realized_r=5.0,
            close_reason="big_win",
            leverage=50.0,        # would yield mag = 2.5 unclamped
        )
        await task
    finally:
        pipeline_mod.run_post_mortem = orig  # type: ignore[assignment]

    rr: EventResult = captured["realized_result"]
    assert rr.magnitude_pct == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# TICKET-011 — LLMEngine.judge belt-and-braces deadline
# --------------------------------------------------------------------- #


class _SlowProvider:
    name = "slow"
    model = "slow-1"

    def __init__(self, sleep_for: float) -> None:
        self.sleep_for = sleep_for

    async def chat_json(self, messages, *, timeout):  # noqa: ANN001
        # Sleep IGNORING the per-request timeout the engine passes —
        # this simulates the IPv6-stall / TLS-renegotiation edge cases
        # the outer wrapper protects against.
        await asyncio.sleep(self.sleep_for)
        return '{"intent": "neutral", "confidence_score": 0, "reason": "ok"}', 1

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_ticket011_outer_deadline_returns_degraded_verdict() -> None:
    """When the provider sleeps past ``timeout + 2``, ``judge`` must
    return a degraded neutral verdict instead of awaiting forever."""
    eng = LLMEngine(provider=_SlowProvider(sleep_for=10.0), timeout=0.5)
    verdict = await asyncio.wait_for(
        eng.judge(
            symbol="X", exchange="binance",
            funding_rate=None, funding_deviation_z=None,
            smc=SMCContext(), posts=[],
        ),
        timeout=5.0,
    )
    assert isinstance(verdict, AIVerdict)
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 0
    assert "outer_timeout" in verdict.reason
    assert eng.degraded_count == 1


@pytest.mark.asyncio
async def test_ticket011_fast_provider_returns_real_verdict() -> None:
    eng = LLMEngine(provider=_SlowProvider(sleep_for=0.01), timeout=2.0)
    verdict = await eng.judge(
        symbol="X", exchange="binance",
        funding_rate=None, funding_deviation_z=None,
        smc=SMCContext(), posts=[],
    )
    assert verdict.intent == "neutral"
    # confidence_score=0 came from the provider's payload, not from
    # a degraded synthesis — the reason must NOT mention the deadline.
    assert "outer_timeout" not in verdict.reason
    assert eng.degraded_count == 0


# --------------------------------------------------------------------- #
# TICKET-012 — compute_leverage NaN/inf defence
# --------------------------------------------------------------------- #


def test_ticket012_compute_leverage_nan_vol_falls_back_to_target() -> None:
    sizer = PositionSizer()
    lev = sizer.compute_leverage(
        side=Side.LONG, fused_score=95.0,
        realized_vol_pct=float("nan"),
        top5_depth_usdt=400_000,
    )
    # Must be finite and within the configured band.
    import math
    assert math.isfinite(lev)
    assert sizer.leverage_cfg.min_leverage <= lev <= sizer.leverage_cfg.max_leverage_long


def test_ticket012_compute_leverage_inf_depth_clamped() -> None:
    sizer = PositionSizer()
    lev = sizer.compute_leverage(
        side=Side.SHORT, fused_score=95.0,
        realized_vol_pct=0.05,
        top5_depth_usdt=float("inf"),
    )
    import math
    assert math.isfinite(lev)
    assert lev <= sizer.leverage_cfg.max_leverage_short


def test_ticket012_compute_leverage_nan_score_does_not_propagate() -> None:
    sizer = PositionSizer()
    lev = sizer.compute_leverage(
        side=Side.LONG, fused_score=float("nan"),
        realized_vol_pct=0.05,
        top5_depth_usdt=400_000,
    )
    import math
    assert math.isfinite(lev)


# --------------------------------------------------------------------- #
# TICKET-016 — Health metrics on /healthz + /metrics
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ticket016_metrics_endpoint_exposes_p1_health_gauges(
    tmp_path: Path,
) -> None:
    """``/metrics`` must expose persistor_save_failures /
    position_watcher_lag_sec / llm_degraded_count /
    emergency_close_count / stop_replace_failure_count."""
    from aiohttp.test_utils import TestClient, TestServer

    state = HealthState()
    state.started_at = time.time() - 1.0
    state.persistor_save_failures = 7
    state.position_watcher_lag_sec = 12.5
    state.llm_degraded_count = 4
    state.emergency_close_count = 2
    state.stop_replace_failure_count = 9
    state.fuser_alive = True
    state.screener_alive = True
    state.reconciliation_complete = True

    app = await make_health_app(state)
    server = TestServer(app)
    async with TestClient(server) as client:
        resp = await client.get("/metrics")
        assert resp.status == 200
        text = await resp.text()
        assert "altcoin_agent_persistor_save_failures 7" in text
        assert "altcoin_agent_position_watcher_lag_sec 12.5" in text
        assert "altcoin_agent_llm_degraded_count 4" in text
        assert "altcoin_agent_emergency_close_count 2" in text
        assert "altcoin_agent_stop_replace_failure_count 9" in text
        # /healthz JSON also carries them.
        resp = await client.get("/healthz")
        payload = await resp.json()
        assert payload["persistor_save_failures"] == 7
        assert payload["position_watcher_lag_sec"] == 12.5
        assert payload["llm_degraded_count"] == 4
        assert payload["emergency_close_count"] == 2
        assert payload["stop_replace_failure_count"] == 9


@pytest.mark.asyncio
async def test_ticket016_refresh_hook_updates_persistor_failures(
    tmp_path: Path,
) -> None:
    """The refresh hook wired in ``App.run`` must pull
    ``consecutive_save_failures`` off the persistor onto the health
    state on every scrape."""
    from aiohttp.test_utils import TestClient, TestServer

    persistor = AccountPersistor(
        path=tmp_path / "acc.json",
        max_consecutive_save_failures=10,
    )
    persistor.consecutive_save_failures = 3
    state = HealthState()
    state.started_at = time.time()
    state.fuser_alive = state.screener_alive = state.reconciliation_complete = True

    def refresh() -> None:
        state.persistor_save_failures = persistor.consecutive_save_failures

    app = await make_health_app(state, refresh_metrics=refresh)
    server = TestServer(app)
    async with TestClient(server) as client:
        r = await client.get("/healthz")
        payload = await r.json()
        assert payload["persistor_save_failures"] == 3
        # Bump the counter on the persistor; next scrape must reflect.
        persistor.consecutive_save_failures = 9
        r = await client.get("/metrics")
        text = await r.text()
        assert "altcoin_agent_persistor_save_failures 9" in text


# --------------------------------------------------------------------- #
# Cross-cutting: ensure no regression on the headline RiskGate path
# --------------------------------------------------------------------- #


def test_p1_p2_combined_happy_path_still_approves() -> None:
    """End-to-end smoke: with a fresh signal, healthy tape, and BTC
    flat, the gate must approve. Catches any silent fail-closed
    introduced by the new gates."""
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000,
        anti_chase_max_move_pct=0.10,
        vol_kill_window_ms=60_000,
        vol_kill_range_pct=0.20,
    ))
    now = _now_ms()
    tape.observe("PEPEUSDT", 1.000, now - 20_000)
    tape.observe("PEPEUSDT", 1.005, now)

    rf = RegimeFilter(RegimeFilterConfig(
        reference_symbol="BTC",
        btc_drop_block_long_pct=0.03,
        btc_rip_block_short_pct=0.05,
        min_samples=2,
    ))
    rf.observe("BTC", 100.0, now - 60_000)
    rf.observe("BTC", 100.1, now)

    cm = ClusterMap(explicit={"PEPE": "meme"})
    cap = ClusterCapConfig(enabled=True, max_per_cluster=3)

    gate = RiskGate(PositionSizer(), RiskGateConfig(min_liquidity_usdt=100_000))
    decision = gate.evaluate(
        signal=_signal(ts=now, trigger=1.005),
        account=_account(),
        current_price=1.005,
        top5_depth_usdt=400_000,
        realized_vol_pct=0.05,
        initial_stop=0.95,
        now_ms=now,
        price_tape=tape,
        regime_filter=rf,
        cluster_map=cm,
        cluster_cap_cfg=cap,
    )
    assert decision.approved, decision.reason
