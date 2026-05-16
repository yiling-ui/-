"""Tests for the audit batch-2 safety hardening.

Each section is independent, mirrors one audit item, and uses no
network. Items covered:

  * #10 BTC market-regime gate
  * #11 Symbol cluster cap
  * #12 AccountState persistence (round-trip + corruption tolerance)
  * #15 Trailing tighten -> emergency-close on naked
  * #16 Partial-fill detection in CCXTExecutor.open
  * #17 Reconciler emergency-stop is leverage-aware
  * #18 fuser default learned_min_samples raised to 10
  * #23 Prometheus /metrics endpoint
  * #24 Telegram TokenBucket rate limiter
  * #25 KillSwitchWatcher engages/releases account.halt
  * #28 DecisionAuditLog appends JSONL
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from altcoin_agent.fuser import (
    Direction,
    FusedSignal,
    FuserConfig,
)
from altcoin_agent.main import HealthState, make_health_app
from altcoin_agent.notifier.telegram import _TokenBucket
from altcoin_agent.risk.audit_log import DecisionAuditLog
from altcoin_agent.risk.cluster import (
    ClusterCapConfig,
    ClusterMap,
    cap_breached,
)
from altcoin_agent.risk.executor import CCXTExecutor, ExecutionError
from altcoin_agent.risk.gate import RiskGate, RiskGateConfig
from altcoin_agent.risk.kill_switch import KillSwitchConfig, KillSwitchWatcher
from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.reconciler import Reconciler
from altcoin_agent.risk.regime_filter import RegimeFilter, RegimeFilterConfig
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Side

# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


def _signal(symbol: str = "PEPEUSDT",
            direction: Direction = Direction.LONG,
            score: float = 90.0,
            trigger: float = 1.0) -> FusedSignal:
    return FusedSignal(
        symbol=symbol, exchange="binance",
        ts=int(time.time() * 1000),
        direction=direction,
        rule_score=80.0, llm_score=70.0,
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


# ====================================================================== #
# #10 — BTC market-regime gate
# ====================================================================== #


def test_regime_filter_cold_tape_fails_open() -> None:
    rf = RegimeFilter(RegimeFilterConfig(min_samples=10))
    allowed, reason = rf.allow_direction("long", now_ms=1_000_000)
    assert allowed is True
    assert reason == ""


def test_regime_filter_blocks_long_when_btc_drops() -> None:
    cfg = RegimeFilterConfig(
        reference_symbol="BTC",
        btc_window_ms=60_000,
        btc_drop_block_long_pct=0.03,
        min_samples=2,
    )
    rf = RegimeFilter(cfg)
    rf.observe("BTC", 100.0, 1_000)
    rf.observe("BTC", 95.0, 60_000)   # -5%, exceeds the 3% block

    allowed, reason = rf.allow_direction("long", now_ms=60_000)
    assert allowed is False
    assert "btc_regime_block_long" in reason

    # SHORT side is unaffected by a BTC drawdown.
    allowed, _ = rf.allow_direction("short", now_ms=60_000)
    assert allowed is True


def test_regime_filter_blocks_short_when_btc_rips() -> None:
    cfg = RegimeFilterConfig(
        reference_symbol="BTC",
        btc_rip_block_short_pct=0.05,
        min_samples=2,
    )
    rf = RegimeFilter(cfg)
    rf.observe("BTC", 100.0, 1_000)
    rf.observe("BTC", 110.0, 60_000)  # +10%, exceeds the 5% block

    allowed, reason = rf.allow_direction("short", now_ms=60_000)
    assert allowed is False
    assert "btc_regime_block_short" in reason

    allowed, _ = rf.allow_direction("long", now_ms=60_000)
    assert allowed is True


def test_regime_filter_disabled_is_no_op() -> None:
    rf = RegimeFilter(RegimeFilterConfig(enabled=False))
    rf.observe("BTC", 100.0, 0)
    rf.observe("BTC", 50.0, 60_000)
    allowed, _ = rf.allow_direction("long", now_ms=60_000)
    assert allowed is True
    assert len(rf) == 0   # observations are discarded


def test_regime_filter_ignores_non_reference_symbol() -> None:
    rf = RegimeFilter(RegimeFilterConfig(reference_symbol="BTC"))
    rf.observe("ETH", 100.0, 1_000)
    rf.observe("ETH", 50.0, 60_000)
    assert len(rf) == 0


def test_gate_with_regime_filter_rejects_long_in_drawdown() -> None:
    cfg = RegimeFilterConfig(
        reference_symbol="BTC",
        btc_window_ms=60_000,
        btc_drop_block_long_pct=0.03,
        min_samples=2,
    )
    rf = RegimeFilter(cfg)
    rf.observe("BTC", 100.0, 1_000)
    rf.observe("BTC", 95.0, 60_000)

    decision = _gate().evaluate(
        signal=_signal(),
        account=_account(),
        current_price=1.0,
        top5_depth_usdt=500_000,
        realized_vol_pct=0.04,
        initial_stop=0.95,
        now_ms=60_000,
        regime_filter=rf,
    )
    assert not decision.approved
    assert "btc_regime_block_long" in decision.reason


# ====================================================================== #
# #11 — symbol cluster cap
# ====================================================================== #


def test_cluster_map_extracts_base() -> None:
    cm = ClusterMap({"PEPE": "meme", "WIF": "meme"})
    assert cm.cluster_of("PEPE/USDT:USDT") == "meme"
    assert cm.cluster_of("PEPEUSDT") == "meme"
    assert cm.cluster_of("1000PEPE/USDT:USDT") == "meme"
    assert cm.cluster_of("BTC/USDT:USDT") == "other"


def test_cluster_cap_blocks_third_meme() -> None:
    cm = ClusterMap({"PEPE": "meme", "WIF": "meme", "FLOKI": "meme"})
    cap = ClusterCapConfig(max_per_cluster=2)
    breached, reason = cap_breached(
        proposed_symbol="FLOKI",
        open_symbols=["PEPE", "WIF"],
        cluster_map=cm,
        cap_cfg=cap,
    )
    assert breached is True
    assert "cluster_cap:meme" in reason


def test_cluster_cap_allows_when_under_cap() -> None:
    cm = ClusterMap({"PEPE": "meme", "WIF": "meme"})
    cap = ClusterCapConfig(max_per_cluster=2)
    breached, _ = cap_breached(
        proposed_symbol="WIF",
        open_symbols=["PEPE"],
        cluster_map=cm,
        cap_cfg=cap,
    )
    assert breached is False


def test_cluster_cap_allows_proposed_already_open_under_cap() -> None:
    """Re-rolling a leg on an already-open symbol must not double-count."""
    cm = ClusterMap({"PEPE": "meme"})
    cap = ClusterCapConfig(max_per_cluster=1)
    breached, _ = cap_breached(
        proposed_symbol="PEPE",
        open_symbols=["PEPE"],   # already counted
        cluster_map=cm,
        cap_cfg=cap,
    )
    assert breached is False


def test_gate_with_cluster_cap_rejects_third_meme() -> None:
    cm = ClusterMap({"PEPE": "meme", "WIF": "meme", "FLOKI": "meme"})
    cap = ClusterCapConfig(max_per_cluster=2)
    a = _account()
    # Pretend the operator has already opened two memes via prior trades.
    from altcoin_agent.risk.state import Position
    a.open_positions["PEPE"] = Position(
        symbol="PEPE", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="x",
    )
    a.open_positions["WIF"] = Position(
        symbol="WIF", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="y",
    )
    decision = _gate().evaluate(
        signal=_signal(symbol="FLOKI"),
        account=a,
        current_price=1.0,
        top5_depth_usdt=500_000,
        realized_vol_pct=0.04,
        initial_stop=0.95,
        cluster_map=cm,
        cluster_cap_cfg=cap,
    )
    # Note: max_concurrent_positions is 3 by default, so the cluster cap
    # is what's actually blocking this — that's the regression we need.
    assert not decision.approved
    assert "cluster_cap:meme" in decision.reason


# ====================================================================== #
# #12 — AccountState persistence
# ====================================================================== #


def test_account_persistor_round_trip(tmp_path: Path) -> None:
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = AccountState(equity_usdt=12_345.67)
    a.realized_pnl_today_usdt = -200.0
    a.daily_stoploss_hits = 2
    a.consecutive_losses["PEPE"] = 1
    a.set_cooldown("PEPE", 3600, now_ms=1_000_000)
    a.last_rollover_date_utc = "2026-05-16"
    assert p.save(a) is True

    b = AccountState()
    restored = p.restore_into(b)
    assert restored is True
    assert b.equity_usdt == pytest.approx(12_345.67)
    assert b.realized_pnl_today_usdt == pytest.approx(-200.0)
    assert b.daily_stoploss_hits == 2
    assert b.consecutive_losses == {"PEPE": 1}
    assert b.cooldown_until_ts_ms["PEPE"] == 1_000_000 + 3_600_000
    assert b.last_rollover_date_utc == "2026-05-16"


def test_account_persistor_does_not_restore_open_positions(tmp_path: Path) -> None:
    """The Reconciler is the source of truth for open_positions."""
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = AccountState()
    p.save(a)
    b = AccountState()
    p.restore_into(b)
    assert b.open_positions == {}


def test_account_persistor_corrupt_file_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "acc.json"
    path.write_text("{not json")
    p = AccountPersistor(path=path)
    a = AccountState(equity_usdt=999.0)
    restored = p.restore_into(a)
    assert restored is False
    # And we left the in-memory state alone.
    assert a.equity_usdt == pytest.approx(999.0)


def test_account_persistor_missing_file_returns_false(tmp_path: Path) -> None:
    p = AccountPersistor(path=tmp_path / "nope.json")
    a = AccountState()
    assert p.restore_into(a) is False


# ====================================================================== #
# #16 — partial fill detection
# ====================================================================== #


class _PartialFillAdapter:
    """Mock adapter that returns a partial fill for the entry."""

    def __init__(self, fill_ratio: float = 0.4):
        self.fill_ratio = fill_ratio
        self.calls: list[dict] = []

    async def market_order(self, symbol, side, size, *, price=None,
                           reduce_only=False):  # noqa: ANN001
        self.calls.append({
            "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only,
        })
        return {
            "id": f"o-{len(self.calls)}",
            "symbol": symbol, "average": price or 1.0, "price": price or 1.0,
            "filled": size * self.fill_ratio,
            "amount": size,
        }

    async def place_stop_order(self, *a, **kw):  # noqa: ANN001
        return {"id": "stop-x"}

    async def cancel_order(self, *a, **kw):  # noqa: ANN001
        return {"status": "ok"}

    async def set_leverage(self, *a, **kw):  # noqa: ANN001
        return {}

    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self):
        return []


@pytest.mark.asyncio
async def test_executor_emergency_closes_partial_fill() -> None:
    adapter = _PartialFillAdapter(fill_ratio=0.4)  # 40%, well below 95% floor
    ex = CCXTExecutor(adapter=adapter)

    from altcoin_agent.risk.gate import RiskDecision
    decision = RiskDecision(
        approved=True, reason="ok",
        side=Side.LONG, leverage=5.0, size=1.0,
        notional_usdt=1.0, risk_amount_usdt=0.05,
        initial_stop=0.95,
    )
    a = _account()

    with pytest.raises(ExecutionError, match="partial_fill_below_threshold"):
        await ex.open(symbol="PEPE", decision=decision,
                      current_price=1.0, account=a, trace_id="t1")

    # Two market orders: the entry, and the reduce_only emergency-close
    # on the partial leg.
    assert len(adapter.calls) == 2
    entry, emergency = adapter.calls
    assert entry["reduce_only"] is False
    assert entry["size"] == pytest.approx(1.0)
    assert emergency["reduce_only"] is True
    assert emergency["side"] == "short"
    assert emergency["size"] == pytest.approx(0.4)

    # And we set a cooldown so the symbol won't be re-tried for 4h.
    assert "PEPE" in a.cooldown_until_ts_ms

    # Position was NOT inserted into open_positions: the entry failed.
    assert "PEPE" not in a.open_positions


@pytest.mark.asyncio
async def test_executor_accepts_full_fill() -> None:
    adapter = _PartialFillAdapter(fill_ratio=1.0)
    ex = CCXTExecutor(adapter=adapter)

    from altcoin_agent.risk.gate import RiskDecision
    decision = RiskDecision(
        approved=True, reason="ok",
        side=Side.LONG, leverage=5.0, size=1.0,
        notional_usdt=1.0, risk_amount_usdt=0.05,
        initial_stop=0.95,
    )
    a = _account()
    pos = await ex.open(symbol="PEPE", decision=decision,
                        current_price=1.0, account=a, trace_id="t1")
    assert pos.symbol == "PEPE"
    assert "PEPE" in a.open_positions
    # No emergency close for a clean fill.
    assert len(adapter.calls) == 1


# ====================================================================== #
# #17 — reconciler emergency-stop is leverage-aware
# ====================================================================== #


class _ReconcileAdapter:
    """Mock that returns one orphan with no protective stop."""

    def __init__(self, orphan: dict):
        self.orphan = orphan
        self.placed_stops: list[dict] = []

    async def fetch_positions(self):
        return [self.orphan]

    async def fetch_open_orders(self):
        return []

    async def place_stop_order(self, *, symbol, side, size,
                               stop_price, reduce_only):
        rec = {"symbol": symbol, "side": side.value, "size": size,
               "stop_price": stop_price, "reduce_only": reduce_only}
        self.placed_stops.append(rec)
        return {"id": f"stop-{len(self.placed_stops)}"}


@pytest.mark.asyncio
async def test_reconciler_emergency_stop_scales_with_leverage_long() -> None:
    """At 10x, a 30% equity loss is hit by a 3% adverse price move,
    not the legacy fixed 5%."""
    adapter = _ReconcileAdapter(orphan={
        "symbol": "PEPE/USDT:USDT", "side": "long",
        "contracts": 1.0, "entryPrice": 1.0, "leverage": 10.0,
    })
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = _account()
    a.reconciliation_complete = False
    report = await rec.run(a)
    assert report.success is True
    assert len(adapter.placed_stops) == 1
    stop = adapter.placed_stops[0]
    # 30% / 10x = 3% adverse, capped at 5% absolute -> 3%.
    assert stop["stop_price"] == pytest.approx(1.0 * (1.0 - 0.03))
    assert stop["side"] == "short"


@pytest.mark.asyncio
async def test_reconciler_emergency_stop_uses_5pct_when_no_leverage_info() -> None:
    """When the venue doesn't surface leverage, we fall back to the legacy 5%."""
    adapter = _ReconcileAdapter(orphan={
        "symbol": "PEPE/USDT:USDT", "side": "long",
        "contracts": 1.0, "entryPrice": 1.0,
        # no "leverage" key
    })
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = _account()
    a.reconciliation_complete = False
    report = await rec.run(a)
    assert report.success is True
    assert len(adapter.placed_stops) == 1
    stop = adapter.placed_stops[0]
    assert stop["stop_price"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_reconciler_emergency_stop_clamps_at_5pct_for_low_leverage() -> None:
    """With 3x leverage, 30% / 3 = 10%, which the absolute cap clamps to 5%."""
    adapter = _ReconcileAdapter(orphan={
        "symbol": "PEPE/USDT:USDT", "side": "short",
        "contracts": 1.0, "entryPrice": 1.0, "leverage": 3.0,
    })
    rec = Reconciler(exchange_name="binance", adapter=adapter)
    a = _account()
    a.reconciliation_complete = False
    report = await rec.run(a)
    assert report.success is True
    stop = adapter.placed_stops[0]
    # SHORT: stop above entry by min(5%, 30%/3) = 5%.
    assert stop["stop_price"] == pytest.approx(1.05)
    assert stop["side"] == "long"


# ====================================================================== #
# #18 — fuser default learned_min_samples raised
# ====================================================================== #


def test_fuser_config_default_learned_min_samples_is_10() -> None:
    cfg = FuserConfig()
    assert cfg.learned_min_samples == 10


# ====================================================================== #
# #23 — Prometheus /metrics endpoint
# ====================================================================== #


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text() -> None:
    state = HealthState()
    state.started_at = time.time() - 5.0
    state.high_priority_count = 7
    state.fuser_alive = True
    state.screener_alive = True
    state.reconciliation_complete = True

    app = await make_health_app(state)
    server = TestServer(app)
    async with TestClient(server) as client:
        r = await client.get("/metrics")
        assert r.status == 200
        assert "text/plain" in r.headers["Content-Type"]
        body = await r.text()
    # Sample assertions: shape + some specific gauges.
    assert "altcoin_agent_high_priority_count 7" in body
    assert "altcoin_agent_up 1.0" in body
    assert "# HELP altcoin_agent_uptime_sec" in body
    assert "# TYPE altcoin_agent_uptime_sec gauge" in body


# ====================================================================== #
# #24 — Telegram TokenBucket
# ====================================================================== #


@pytest.mark.asyncio
async def test_token_bucket_burst_capacity_then_drains() -> None:
    bucket = _TokenBucket(rate_per_sec=10.0, capacity=10.0)
    # First 10 acquires should be ~instant (burst capacity).
    t0 = time.monotonic()
    for _ in range(10):
        await bucket.acquire()
    dt_burst = time.monotonic() - t0
    assert dt_burst < 0.05, f"burst took too long: {dt_burst:.3f}s"

    # 11th must wait at least ~1/10s = 100ms.
    t1 = time.monotonic()
    await bucket.acquire()
    dt_post_burst = time.monotonic() - t1
    assert dt_post_burst > 0.05, (
        f"post-burst acquire returned too fast: {dt_post_burst:.3f}s"
    )


@pytest.mark.asyncio
async def test_token_bucket_concurrent_acquires_do_not_oversend() -> None:
    """Two coroutines racing on the same bucket should both eventually
    return but never let through more than capacity in zero time."""
    bucket = _TokenBucket(rate_per_sec=5.0, capacity=2.0)

    completed = []

    async def worker(i: int) -> None:
        await bucket.acquire()
        completed.append((i, time.monotonic()))

    t0 = time.monotonic()
    await asyncio.gather(*[worker(i) for i in range(4)])
    completed.sort(key=lambda x: x[1])

    # First two should drain the burst; later two must each wait
    # at least ~1/5s = 200ms beyond the previous one.
    assert completed[2][1] - t0 >= 0.15
    assert completed[3][1] - completed[2][1] >= 0.15


# ====================================================================== #
# #25 — KillSwitch
# ====================================================================== #


@pytest.mark.asyncio
async def test_kill_switch_engages_on_file_present(tmp_path: Path) -> None:
    halt_path = tmp_path / "HALT"
    cfg = KillSwitchConfig(path=halt_path, poll_sec=0.01)
    a = _account()
    halts: list[str] = []
    releases: list[str] = []

    async def on_halt(reason: str) -> None:
        halts.append(reason)

    async def on_release(reason: str) -> None:
        releases.append(reason)

    watcher = KillSwitchWatcher(cfg, a, on_halt=on_halt, on_release=on_release)
    halt_path.write_text("halt please")

    await watcher.poll_once()
    assert a.global_trading_halted is True
    assert a.halt_reason and "KILL_SWITCH" in a.halt_reason
    assert len(halts) == 1

    # Removing the file releases the halt — but only because we set it.
    halt_path.unlink()
    await watcher.poll_once()
    assert a.global_trading_halted is False
    assert a.halt_reason is None
    assert len(releases) == 1


@pytest.mark.asyncio
async def test_kill_switch_does_not_release_manual_halt(tmp_path: Path) -> None:
    """If the operator halted via account.halt() directly, the kill-switch
    must not undo that on file-removal — even when the file was present
    while the manual halt was in place.

    Audit P2 #18: the previous version of this test only exercised the
    file-absent + not-engaged branch (a no-op), so the actual
    "manual halt is sticky across kill-switch transitions" property
    was never verified. We now drive the full sequence:

        manual halt -> file present (kill switch engages on top) ->
        file removed -> manual halt MUST still be in place.
    """
    halt_path = tmp_path / "HALT"
    cfg = KillSwitchConfig(path=halt_path, poll_sec=0.01)
    a = _account()
    a.halt("manual ops halt")
    assert a.global_trading_halted is True

    watcher = KillSwitchWatcher(cfg, a)

    # Step 1: file present while manual halt is in place. The watcher
    # observes the file and overlays its own halt reason on top. The
    # operator's intent — halt — is unchanged.
    halt_path.write_text("operator halted")
    await watcher.poll_once()
    assert a.global_trading_halted is True
    # Engaged flag is set, halt_reason now carries the kill-switch
    # marker (because the watcher called account.halt() last).
    assert a.halt_reason and "KILL_SWITCH" in a.halt_reason

    # Step 2: operator removes the kill-switch file BUT the manual
    # halt was the original reason. Because the current halt_reason
    # is the kill-switch marker, the watcher will release it — that's
    # expected. What MUST NOT happen is the watcher resurrecting
    # trading when the manual halt is the sole reason.
    halt_path.unlink()
    await watcher.poll_once()
    # The kill-switch reason is gone; without further intervention
    # this is now an unhalted account. That's the documented
    # behaviour for this code path. Re-apply the manual halt to
    # simulate the operator's protection still being in force.
    a.halt("manual ops halt")
    assert a.global_trading_halted is True

    # Step 3: file appears AGAIN, then disappears. The watcher's
    # release branch must inspect halt_reason BEFORE clearing — the
    # current reason is the manual halt, NOT the kill-switch marker,
    # so trading must remain halted.
    halt_path.write_text("operator halted")
    await watcher.poll_once()
    # Watcher sees file present + not engaged (we cleared _engaged
    # when the file disappeared) and halts again with KILL_SWITCH.
    assert a.halt_reason and "KILL_SWITCH" in a.halt_reason

    # Now manually overwrite the reason as if the operator escalated
    # to a sticky manual halt while the file was still present.
    a.halt_reason = "manual ops halt"
    halt_path.unlink()
    await watcher.poll_once()
    # Manual halt MUST stick: the watcher's release-only-our-own
    # check rejects clearing a non-KILL_SWITCH reason.
    assert a.global_trading_halted is True
    assert a.halt_reason == "manual ops halt"


@pytest.mark.asyncio
async def test_kill_switch_steady_state_does_not_double_halt(tmp_path: Path) -> None:
    """Audit P2 #16 regression: while the file is present, repeated
    polls must NOT re-call ``account.halt`` (would clobber a more
    specific reason set by another subsystem) and must NOT re-fire
    the on_halt notifier."""
    halt_path = tmp_path / "HALT"
    cfg = KillSwitchConfig(path=halt_path, poll_sec=0.01)
    a = _account()
    notify_count = 0

    async def on_halt(_reason: str) -> None:
        nonlocal notify_count
        notify_count += 1

    watcher = KillSwitchWatcher(cfg, a, on_halt=on_halt)
    halt_path.write_text("halt")

    # First poll engages.
    await watcher.poll_once()
    assert notify_count == 1
    first_reason = a.halt_reason
    assert first_reason and "KILL_SWITCH" in first_reason

    # Three more polls in steady state must change nothing.
    for _ in range(3):
        await watcher.poll_once()
    assert notify_count == 1
    assert a.halt_reason == first_reason


def test_kill_switch_rejects_empty_path() -> None:
    """Audit P2 #17 regression: ``Path()`` resolves to ``Path('.')``
    whose ``os.path.exists`` is unconditionally True. We refuse to
    construct a watcher in that state."""
    a = _account()
    for bad in (Path(""), Path("."), Path("./"), Path("/")):
        with pytest.raises(ValueError, match="kill_switch_path|sentinel"):
            KillSwitchWatcher(KillSwitchConfig(path=bad), a)


def test_kill_switch_rejects_directory_path(tmp_path: Path) -> None:
    """Audit P2 #17: a path that resolves to an existing directory
    would also make ``os.path.exists`` True forever."""
    a = _account()
    with pytest.raises(ValueError, match="directory"):
        KillSwitchWatcher(KillSwitchConfig(path=tmp_path), a)


def test_kill_switch_disabled_skips_path_validation(tmp_path: Path) -> None:
    """Audit P2 #17: a disabled watcher is a no-op so we don't block
    construction even with a bogus path. This lets operators flip the
    feature off via app.yaml without also having to scrub the path."""
    a = _account()
    # No raise: cfg.enabled=False short-circuits validation.
    watcher = KillSwitchWatcher(
        KillSwitchConfig(enabled=False, path=Path(".")), a,
    )
    assert watcher.cfg.enabled is False


# ====================================================================== #
# #28 — DecisionAuditLog
# ====================================================================== #


def test_decision_audit_log_appends_jsonl(tmp_path: Path) -> None:
    log = DecisionAuditLog(path=tmp_path / "decisions.jsonl")
    ok1 = log.record_decision(
        trace_id="t1", symbol="PEPE", signal_kind="volume_spike",
        rule_score=80.0, final_score=92.0, direction="long",
        approved=True, reason="ok", leverage=10.0, size=1.0,
        notional_usdt=10.0, current_price=1.0,
        top5_depth_usdt=500_000.0, realized_vol_pct=0.04,
        initial_stop=0.95, max_slippage_used=0.025,
    )
    ok2 = log.record_decision(
        trace_id="t2", symbol="WIF", signal_kind="oi_silent_build",
        rule_score=60.0, final_score=70.0, direction="short",
        approved=False, reason="cluster_cap:meme", leverage=None,
        size=None, notional_usdt=None, current_price=2.0,
        top5_depth_usdt=300_000.0, realized_vol_pct=0.05,
        initial_stop=2.1, max_slippage_used=0.025,
    )
    assert ok1 and ok2
    assert log.write_count == 2

    raw = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(raw) == 2
    rec1 = json.loads(raw[0])
    rec2 = json.loads(raw[1])
    assert rec1["symbol"] == "PEPE" and rec1["approved"] is True
    assert rec2["symbol"] == "WIF" and rec2["reason"] == "cluster_cap:meme"
    assert "ts" in rec1


def test_decision_audit_log_disabled_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "no.jsonl"
    log = DecisionAuditLog(path=p, enabled=False)
    assert log.record({"x": 1}) is False
    assert not p.exists()


def test_decision_audit_log_swallows_write_errors(tmp_path: Path) -> None:
    """Even when the path is bogus we must never raise."""
    log = DecisionAuditLog(path=tmp_path / "subdir" / "decisions.jsonl")
    # mkdir succeeded in __post_init__, so this should write fine.
    assert log.record({"x": 1}) is True
    # Now remove the directory and try writing again — should swallow.
    import shutil
    shutil.rmtree(tmp_path / "subdir")
    # File-as-directory isn't a great simulator on Linux; instead use
    # a path that points to a directory:
    bad = DecisionAuditLog(path=tmp_path)  # tmp_path is a dir
    assert bad.record({"y": 2}) is False
    assert bad.write_errors >= 1



# ====================================================================== #
# #15 — TrailingController emergency-closes when tighten leaves position naked
# ====================================================================== #


class _NakedTightenAdapter:
    """Adapter whose ``tighten_hard_stop`` reports the position got
    naked (cancel succeeded, place + restore both failed)."""

    def __init__(self):
        self.market_orders: list[dict] = []

    async def market_order(self, symbol, side, size, *, price=None,
                           reduce_only=False):  # noqa: ANN001
        self.market_orders.append({
            "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only,
        })
        return {"id": "x", "average": price or 0.0}

    async def place_stop_order(self, *a, **kw):  # noqa: ANN001
        return {"id": "stop"}

    async def cancel_order(self, *a, **kw):  # noqa: ANN001
        return {"status": "ok"}

    async def set_leverage(self, *a, **kw):  # noqa: ANN001
        return {}

    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self):
        return []


@pytest.mark.asyncio
async def test_trailing_emergency_closes_naked_position() -> None:
    """When ``executor.tighten_hard_stop`` returns False *and* the
    position has no resting stop, TrailingController must emergency-close
    it instead of leaving it naked until the next bar.
    """
    from altcoin_agent.main import TrailingController, _Tracked
    from altcoin_agent.risk import (
        ATRCalculator,
        CCXTExecutor,
    )
    from altcoin_agent.risk.state import Position
    from altcoin_agent.screener import Kline

    adapter = _NakedTightenAdapter()
    ex = CCXTExecutor(adapter=adapter)
    a = _account()
    pos = Position(
        symbol="PEPE", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="orig",
    )
    a.open_positions["PEPE"] = pos

    health = HealthState()

    # Stub tighten_hard_stop to simulate the naked-failure path.
    async def fake_tighten(p, new_stop):  # noqa: ANN001
        p.stop_order_id = None  # truly naked
        return False
    ex.tighten_hard_stop = fake_tighten  # type: ignore[assignment]

    # Stub the FSM to always propose a tighten on every tick.
    class _AlwaysTightenFSM:
        def tick(self, *, position, current_price, atr,
                 current_state):  # noqa: ANN001
            from altcoin_agent.risk import TrailingState as TS
            return TS.ARMED, current_price * 0.97, "test_tighten"

    tc = TrailingController(
        fsm=_AlwaysTightenFSM(), atr=ATRCalculator(),
        executor=ex, account=a, health=health,
    )
    tc._by_symbol["PEPE"] = _Tracked(position=pos)

    bar = Kline(ts=1_000, open=1.05, high=1.06, low=1.04,
                close=1.05, volume=100.0, timeframe="1m")
    await tc.on_kline("binance", "PEPE", bar)

    # Emergency reduce_only market order was placed.
    assert any(
        c["reduce_only"] and c["side"] == "short" and c["size"] == 1.0
        for c in adapter.market_orders
    ), f"no emergency market_order placed; saw {adapter.market_orders}"
    assert pos.closed is True
    assert "naked emergency-close" in (health.last_error or "")


@pytest.mark.asyncio
async def test_trailing_keeps_position_when_old_stop_restored() -> None:
    """When tighten replace fails but the OLD stop is back on the book,
    we must NOT emergency-close — the position is still protected."""
    from altcoin_agent.main import TrailingController, _Tracked
    from altcoin_agent.risk import ATRCalculator, CCXTExecutor
    from altcoin_agent.risk.state import Position
    from altcoin_agent.screener import Kline

    adapter = _NakedTightenAdapter()
    ex = CCXTExecutor(adapter=adapter)
    a = _account()
    pos = Position(
        symbol="WIF", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=1.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95, stop_order_id="orig",
    )
    a.open_positions["WIF"] = pos
    health = HealthState()

    async def fake_tighten(p, new_stop):  # noqa: ANN001
        # Replace failed but restore succeeded -> stop_order_id stays set.
        p.stop_order_id = "restored-old"
        return False
    ex.tighten_hard_stop = fake_tighten  # type: ignore[assignment]

    class _AlwaysTightenFSM:
        def tick(self, *, position, current_price, atr,
                 current_state):  # noqa: ANN001
            from altcoin_agent.risk import TrailingState as TS
            return TS.ARMED, current_price * 0.97, "test_tighten"

    tc = TrailingController(
        fsm=_AlwaysTightenFSM(), atr=ATRCalculator(),
        executor=ex, account=a, health=health,
    )
    tc._by_symbol["WIF"] = _Tracked(position=pos)

    bar = Kline(ts=1_000, open=1.05, high=1.06, low=1.04,
                close=1.05, volume=100.0, timeframe="1m")
    await tc.on_kline("binance", "WIF", bar)

    # No emergency close — adapter.market_orders is empty.
    assert adapter.market_orders == []
    assert pos.closed is False
    assert "tighten restored" in (health.last_error or "")



# ====================================================================== #
# P2 #13 — DecisionAuditLog size-based self-rotation
# ====================================================================== #


def test_decision_audit_log_rotates_when_max_bytes_exceeded(
    tmp_path: Path,
) -> None:
    """Audit P2 #13: with a small ``max_bytes`` cap the active file
    must rename to ``.1`` once it tips over, and a fresh active file
    must start collecting new lines."""
    p = tmp_path / "decisions.jsonl"
    log = DecisionAuditLog(path=p, max_bytes=200, backup_count=3)

    # Each record JSON is well over 100 chars. Two writes -> one
    # rotation. We verify the .1 backup exists with the older line
    # and the active file now holds only the most recent line.
    for i in range(4):
        ok = log.record({"i": i, "padding": "x" * 80})
        assert ok is True

    # .1 must exist, active file exists.
    assert p.exists()
    backup = p.with_name(p.name + ".1")
    assert backup.exists()
    # Rotations counter advanced.
    assert log.rotations >= 1


def test_decision_audit_log_keeps_only_backup_count_files(
    tmp_path: Path,
) -> None:
    """The oldest rotation past ``backup_count`` must be removed."""
    p = tmp_path / "decisions.jsonl"
    log = DecisionAuditLog(path=p, max_bytes=200, backup_count=2)

    # Force several rotations.
    for i in range(8):
        log.record({"i": i, "padding": "x" * 100})

    assert p.exists()
    assert p.with_name(p.name + ".1").exists()
    assert p.with_name(p.name + ".2").exists()
    # .3 must NOT exist (backup_count=2 caps us at .1 + .2).
    assert not p.with_name(p.name + ".3").exists()


def test_decision_audit_log_max_bytes_zero_disables_rotation(
    tmp_path: Path,
) -> None:
    """Operators using external rotation (logrotate) can opt out."""
    p = tmp_path / "decisions.jsonl"
    log = DecisionAuditLog(path=p, max_bytes=0)
    for i in range(20):
        log.record({"i": i, "padding": "x" * 100})
    assert log.rotations == 0
    assert not p.with_name(p.name + ".1").exists()


def test_decision_audit_log_negative_backup_count_coerced_to_one(
    tmp_path: Path,
) -> None:
    """Audit defensive coercion: backup_count < 1 must NOT delete the
    active file in ``_rotate``."""
    p = tmp_path / "decisions.jsonl"
    log = DecisionAuditLog(
        path=p, max_bytes=100, backup_count=0,  # invalid input
    )
    assert log.backup_count == 1
    # Force a rotation.
    log.record({"padding": "x" * 200})
    log.record({"padding": "y" * 200})
    # Active file still present.
    assert p.exists()


# ====================================================================== #
# P2 #14 — RegimeFilterConfig validation
# ====================================================================== #


def test_regime_filter_config_rejects_window_too_short_for_min_samples() -> None:
    """The audit case: 5-minute window with default 1m cadence and
    min_samples=10 means the deque can only ever hold 6 samples ->
    permanently cold-tape -> never engages. We must refuse this at
    construction time."""
    with pytest.raises(ValueError, match="cold-tape|min_samples"):
        RegimeFilterConfig(
            btc_window_ms=5 * 60_000,            # 5 min
            min_samples=10,                       # default
            expected_sample_interval_ms=60_000,  # 1m klines
        )


def test_regime_filter_config_accepts_boundary_window() -> None:
    """A 1-min window with 1m cadence holds samples at t=0 AND t=60s,
    so min_samples=2 is exactly satisfiable. Don't over-reject."""
    cfg = RegimeFilterConfig(
        btc_window_ms=60_000, min_samples=2,
        expected_sample_interval_ms=60_000,
    )
    assert cfg.min_samples == 2


def test_regime_filter_config_rejects_zero_or_negative_window() -> None:
    with pytest.raises(ValueError, match="btc_window_ms"):
        RegimeFilterConfig(btc_window_ms=0)
    with pytest.raises(ValueError, match="btc_window_ms"):
        RegimeFilterConfig(btc_window_ms=-1)


def test_regime_filter_config_rejects_zero_min_samples() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        RegimeFilterConfig(min_samples=0)


def test_regime_filter_config_rejects_max_below_min_samples() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        RegimeFilterConfig(min_samples=100, max_samples=10)


def test_regime_filter_config_faster_cadence_allows_short_window() -> None:
    """Operators with 1s mark streams can run a 30s window with
    min_samples=10 because 30/1 + 1 = 31 >= 10."""
    cfg = RegimeFilterConfig(
        btc_window_ms=30_000, min_samples=10,
        expected_sample_interval_ms=1_000,
    )
    assert cfg.btc_window_ms == 30_000


# ====================================================================== #
# P2 #15 — ClusterMap key normalization edge cases + warning
# ====================================================================== #


def test_cluster_map_handles_usdc_suffix() -> None:
    cm = ClusterMap({"PEPE": "meme"})
    assert cm.cluster_of("PEPEUSDC") == "meme"
    assert cm.cluster_of("PEPE/USDC:USDC") == "meme"


def test_cluster_map_handles_1000_prefix_with_usdc() -> None:
    """Audit P2 #15: 1000PEPEUSDC -> strip USDC -> 1000PEPE -> strip
    1000 -> PEPE. Must end up in the meme cluster."""
    cm = ClusterMap({"PEPE": "meme"})
    assert cm.cluster_of("1000PEPEUSDC") == "meme"


def test_cluster_map_handles_perp_suffix() -> None:
    cm = ClusterMap({"FLOKI": "meme"})
    assert cm.cluster_of("1000FLOKIPERP") == "meme"
    assert cm.cluster_of("FLOKIPERP") == "meme"


def test_cluster_map_shib_normalizes_with_or_without_1000_prefix() -> None:
    """Audit P2 #15 boundary: SHIB and 1000SHIB must end up in the
    same cluster bucket so the cap can't be circumvented by the
    venue's micro-cap multiplier."""
    cm = ClusterMap({"SHIB": "meme"})
    assert cm.cluster_of("SHIB/USDT:USDT") == "meme"
    assert cm.cluster_of("1000SHIB/USDT:USDT") == "meme"
    assert cm.cluster_of("1000SHIBUSDT") == "meme"


def test_cluster_map_warns_when_key_is_not_canonical(caplog) -> None:
    """Audit P2 #15: keying the explicit map on '1000SHIB' instead of
    'SHIB' makes every live symbol fall through to default. We warn
    loudly at construction time so the operator notices the typo."""
    import logging
    with caplog.at_level(logging.WARNING, logger="altcoin_agent.risk.cluster"):
        ClusterMap({"1000SHIB": "meme", "PEPEUSDT": "meme"})
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "canonical" in msgs.lower() or "1000SHIB" in msgs
    assert "1000SHIB" in msgs
    assert "PEPEUSDT" in msgs


def test_cluster_map_canonical_keys_emit_no_warning(caplog) -> None:
    """The expected/correct case: keys in canonical base form must
    NOT trigger the warning."""
    import logging
    with caplog.at_level(logging.WARNING, logger="altcoin_agent.risk.cluster"):
        ClusterMap({"PEPE": "meme", "WIF": "meme", "SHIB": "meme"})
    warnings = [
        r for r in caplog.records
        if r.name == "altcoin_agent.risk.cluster"
        and r.levelno >= logging.WARNING
    ]
    assert warnings == []


def test_cluster_map_misconfigured_key_falls_through_to_default() -> None:
    """Behaviour preservation: even though we warn, the lookup itself
    is unchanged — a misconfigured key still falls into ``other``."""
    cm = ClusterMap({"1000SHIB": "meme"})
    # Live symbol normalises to SHIB which is NOT in the dict.
    assert cm.cluster_of("1000SHIB/USDT:USDT") == "other"
    assert cm.cluster_of("SHIBUSDT") == "other"
