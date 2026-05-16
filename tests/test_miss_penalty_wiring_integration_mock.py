"""Integration tests for Phase A wiring in main.py (A.6).

Each existing unit test in
- tests/test_miss_penalty_engine_mock.py
- tests/test_reject_reason_scorer_mock.py
- tests/test_threshold_auto_tuner_mock.py
- tests/test_reflection_mode_mock.py
exercises a single Phase-A module in isolation. This file boots an
actual ``App`` and asserts the wiring fires end-to-end:

  * cfg.miss_penalty_enabled=True builds the four actors
  * the suspension state in the controller blocks
    ``_handle_high_priority`` and gets a row in decisions.jsonl
  * an A-quadrant signal (final_score >= bypass) goes through during
    suspension
  * ``_run_miss_penalty_pass`` invoked synchronously runs audit ->
    score -> reflection check end-to-end
  * the Telegram callback re-uses ``notifier.error``
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.main import App, AppConfig, DryRunExchangeAdapter, TrailingController
from altcoin_agent.risk import (
    ATRCalculator,
    CCXTExecutor,
    PositionSizer,
    RiskGate,
    RiskGateConfig,
    TrailingStopFSM,
)
from altcoin_agent.risk.miss_penalty_engine import (
    MissedOpportunity,
    fixed_klines_fetcher,
)
from altcoin_agent.risk.reflection_mode import ReflectionState
from altcoin_agent.risk.state import AccountState


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
    """Boot ``App`` with a no-op screener so the test can drive the
    high-priority handler manually."""
    app = App(cfg=cfg)
    runner = asyncio.create_task(app.run())
    for _ in range(50):
        if app._screener is not None:
            break
        await asyncio.sleep(0.02)
    assert app._screener is not None, "screener never initialized"

    async def _noop_run() -> None:
        await app._stop_event.wait()
    app._screener.run = _noop_run  # type: ignore[method-assign]

    try:
        for _ in range(50):
            if app.state.reconciliation_complete:
                break
            await asyncio.sleep(0.02)
        yield app
    finally:
        app.request_stop()
        await asyncio.wait_for(runner, timeout=5.0)


def _signal(
    *,
    symbol: str = "RAVEUSDT",
    direction: Direction = Direction.LONG,
    score: float = 80.0,
    trigger: float = 1.0,
) -> FusedSignal:
    return FusedSignal(
        symbol=symbol, exchange="binance", ts=1,
        direction=direction,
        rule_score=70.0, llm_score=70.0,
        final_score=score,
        is_high_priority=True, blocked=False, block_reason=None,
        trigger_price=trigger,
    )


def _base_config(*, tmp_path: Path, port_base: int, **extras) -> AppConfig:
    """Common config preset: dry-run, miss-penalty enabled, all paths
    under tmp_path so tests don't pollute each other."""
    return AppConfig(
        healthz_port=port_base, dashboard_port=port_base + 1,
        dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0,
        min_liquidity_usdt=100_000.0,
        decision_audit_log_enabled=True,
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
        miss_penalty_enabled=True,
        miss_penalty_state_dir=str(tmp_path / "miss_penalty_state"),
        reflection_reports_dir=str(tmp_path / "reflection_reports"),
        # Tight thresholds so tests can trigger reflection with 3 misses.
        reflection_window_days=7,
        reflection_miss_threshold=3,
        reflection_trade_threshold=2,
        reflection_a_quadrant_bypass_score=95.0,
        # Disable safety modules unrelated to this test surface so we
        # don't have to feed them data.
        regime_filter_enabled=False,
        cluster_cap_enabled=False,
        kill_switch_enabled=False,
        **extras,
    )


def _make_helpers(app: App, account: AccountState):
    sizer = PositionSizer()
    gate = RiskGate(sizer, RiskGateConfig(min_liquidity_usdt=100_000.0))
    executor = CCXTExecutor(adapter=app._adapter)
    trailing = TrailingController(
        fsm=TrailingStopFSM(), atr=ATRCalculator(),
        executor=executor, account=account, health=app.state,
    )
    return gate, executor, trailing


# ====================================================================== #
# Wiring -- actors get built when enabled
# ====================================================================== #


@pytest.mark.asyncio
async def test_miss_penalty_actors_built_when_enabled(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path=tmp_path, port_base=19101)
    async with _running_app(cfg) as app:
        assert app._miss_penalty is not None
        assert app._reject_scorer is not None
        assert app._threshold_tuner is not None
        assert app._reflection is not None


@pytest.mark.asyncio
async def test_miss_penalty_actors_disabled_by_default(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=19103, dashboard_port=19104,
        dry_run=True, graceful_timeout_sec=2.0,
        miss_penalty_enabled=False,
        regime_filter_enabled=False,
        kill_switch_enabled=False,
    )
    async with _running_app(cfg) as app:
        assert app._miss_penalty is None
        assert app._reject_scorer is None
        assert app._threshold_tuner is None
        assert app._reflection is None


# ====================================================================== #
# Suspension blocks _handle_high_priority
# ====================================================================== #


@pytest.mark.asyncio
async def test_handle_high_priority_blocks_when_reflection_suspends(
    tmp_path: Path,
) -> None:
    cfg = _base_config(tmp_path=tmp_path, port_base=19105)
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        adapter.set_mark_price("RAVEUSDT", 1.0)

        # Force suspension on.
        assert app._reflection is not None
        future_ms = int(time.time() * 1000) + 10 * 60 * 1000
        app._reflection._update_state(ReflectionState(
            suspended_until_ts_ms=future_ms,
            acknowledged=False, pending_report_id="x",
        ))

        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        gate, executor, trailing = _make_helpers(app, account)

        # final_score=80 < bypass(95) -> rejected.
        before_rejected = app.state.orders_rejected
        await app._handle_high_priority(
            sig=_signal(score=80.0),
            gate=gate, executor=executor,
            trailing=trailing, account=account,
        )
        assert app.state.orders_rejected == before_rejected + 1

    # The decision audit log should record the rejection with the
    # reflection-mode reason prefix.
    log_path = Path(cfg.decision_audit_log_path)
    rows = [
        json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()
    ]
    matches = [r for r in rows if r.get("symbol") == "RAVEUSDT"]
    assert matches, f"no audit row written; got: {rows}"
    assert matches[0]["approved"] is False
    assert matches[0]["reason"].startswith("reflection_mode_suspended")


@pytest.mark.asyncio
async def test_handle_high_priority_a_quadrant_bypasses_suspension(
    tmp_path: Path,
) -> None:
    cfg = _base_config(tmp_path=tmp_path, port_base=19107)
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        adapter.set_mark_price("RAVEUSDT", 1.0)

        assert app._reflection is not None
        future_ms = int(time.time() * 1000) + 10 * 60 * 1000
        app._reflection._update_state(ReflectionState(
            suspended_until_ts_ms=future_ms,
            acknowledged=False, pending_report_id="x",
        ))

        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        gate, executor, trailing = _make_helpers(app, account)

        # final_score=96 >= bypass(95) -> reaches the gate, which then
        # approves (regime/cluster gates are off, depth provider gives
        # adequate liquidity).
        before_rejected = app.state.orders_rejected
        await app._handle_high_priority(
            sig=_signal(score=96.0),
            gate=gate, executor=executor,
            trailing=trailing, account=account,
        )

        # The bypass path should NOT log a "reflection_mode_suspended"
        # rejection (whether the order ultimately approved or not).
        log_path = Path(cfg.decision_audit_log_path)
        rows = [
            json.loads(ln)
            for ln in log_path.read_text().splitlines() if ln.strip()
        ]
        suspensions = [
            r for r in rows
            if r.get("reason", "").startswith("reflection_mode_suspended")
        ]
        assert not suspensions

    _ = before_rejected  # keep var consumed (count delta is implementation-detail)


@pytest.mark.asyncio
async def test_handle_high_priority_passthrough_when_not_suspended(
    tmp_path: Path,
) -> None:
    """When reflection is wired but NOT in suspension, the gate path
    should behave exactly like before -- no extra rejections."""
    cfg = _base_config(tmp_path=tmp_path, port_base=19109)
    async with _running_app(cfg) as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        adapter.set_mark_price("RAVEUSDT", 1.0)

        # Default ReflectionState: not suspended.
        assert app._reflection is not None
        assert app._reflection.is_suspended() is False

        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        gate, executor, trailing = _make_helpers(app, account)

        await app._handle_high_priority(
            sig=_signal(score=80.0),
            gate=gate, executor=executor,
            trailing=trailing, account=account,
        )

    log_path = Path(cfg.decision_audit_log_path)
    rows = [
        json.loads(ln)
        for ln in log_path.read_text().splitlines() if ln.strip()
    ]
    suspensions = [
        r for r in rows
        if r.get("reason", "").startswith("reflection_mode_suspended")
    ]
    assert not suspensions


# ====================================================================== #
# _run_miss_penalty_pass — full pipeline drives reflection
# ====================================================================== #


def _write_rejected_decision(
    log_path: Path, *, ts: float, trace_id: str,
    symbol: str = "PEPE/USDT:USDT",
    direction: str = "long",
    reason: str = "anti_chase:0.04",
    current_price: float = 1.0,
) -> None:
    rec = {
        "ts": ts, "trace_id": trace_id, "symbol": symbol,
        "direction": direction, "current_price": current_price,
        "approved": False, "reason": reason, "final_score": 78.0,
        "rule_score": 60.0, "signal_kind": "volume_spike",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _make_pump_klines(*, start_ms: int, base: float = 1.0):
    """Generate 1440 1m bars starting at base, ramping +200% over 12h
    then mild retrace to +180%. Each bar's high/low straddles close
    by ±0.1%; peak bar high is exactly the configured peak."""
    bars = []
    bars_count = 1440
    half = bars_count // 2
    peak = base * 3.0
    end_value = peak * 0.93
    for i in range(bars_count):
        ts = start_ms + i * 60_000
        if i <= half:
            close = base + (peak - base) * (i / half)
        else:
            close = peak + (end_value - peak) * ((i - half) / (bars_count - half))
        high = close * 1.001
        low = close * 0.999
        if i == half:
            high = peak
        bars.append((ts, close, high, low, close, 1.0))
    return bars


@pytest.mark.asyncio
async def test_run_miss_penalty_pass_triggers_reflection_end_to_end(
    tmp_path: Path,
) -> None:
    """Seed three rejected decisions whose 24h K-lines moon -> the
    daily pass labels them all as missed pumps -> reflection triggers
    -> markdown report on disk + Telegram callback fired."""
    cfg = _base_config(
        tmp_path=tmp_path, port_base=19111,
        # Low samples_required so a single test trigger is enough.
    )
    notifier_calls: list[dict] = []

    async with _running_app(cfg) as app:
        # Wire a custom notifier so we can observe the Telegram path.
        original_error = app.notifier.error  # type: ignore[union-attr]

        async def _fake_error(msg, *, payload=None):
            notifier_calls.append({"msg": msg, "payload": payload})
            await original_error(msg, payload=payload)

        app.notifier.error = _fake_error  # type: ignore[union-attr,method-assign]

        # Inject canned 24h K-lines for the three test symbols so the
        # audit reads moon-shaped windows.
        # Reject 30h ago, well within the 48h lookback but with enough
        # margin that the wall-clock drift between writing the records
        # and reading them doesn't push the oldest one past the
        # ``since_ts`` filter.
        rejected_at_s = time.time() - 30 * 3600
        rejected_at_ms = int(rejected_at_s * 1000)
        symbols = ["PEPE/USDT:USDT", "WIF/USDT:USDT", "DOGE/USDT:USDT"]
        bars_by_symbol = {
            sym: _make_pump_klines(start_ms=rejected_at_ms)
            for sym in symbols
        }
        assert app._miss_penalty is not None
        app._miss_penalty.kline_fetcher = fixed_klines_fetcher(bars_by_symbol)

        # Seed three rejected decisions in the audit log (with stable
        # trace ids so dedup works).
        log_path = Path(cfg.decision_audit_log_path)
        for i, sym in enumerate(symbols):
            _write_rejected_decision(
                log_path, ts=rejected_at_s + i,
                trace_id=f"reject-{sym}", symbol=sym,
                reason="anti_chase:0.04",
            )

        # Fire one daily pass synchronously. Pick a non-Sunday weekday
        # so the threshold-tuner branch is also exercised but doesn't
        # mutate state (we don't assert on overrides here -- separate
        # test).
        non_sunday_utc = datetime(
            2026, 5, 13, 2, 0, 0, tzinfo=timezone.utc,  # Wed
        )
        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        await app._run_miss_penalty_pass(
            account=account, now_utc=non_sunday_utc,
        )

        # missed_opportunities.jsonl contains 3 entries, all labelled.
        missed_path = (
            Path(cfg.miss_penalty_state_dir) / "missed_opportunities.jsonl"
        )
        assert missed_path.exists()
        rows = [
            json.loads(ln)
            for ln in missed_path.read_text().splitlines() if ln.strip()
        ]
        assert len(rows) == 3
        assert all(r["is_missed_pump"] is True for r in rows)

        # Reflection state -> suspended, pending review.
        assert app._reflection is not None
        assert app._reflection.is_suspended() is True
        assert app._reflection.has_pending_review() is True

        # Markdown report on disk.
        reports_dir = Path(cfg.reflection_reports_dir)
        reports = list(reports_dir.glob("reflection_*.md"))
        assert reports, "no reflection markdown report found"
        text = reports[0].read_text()
        assert "策略反思报告" in text
        assert "PEPE/USDT:USDT" in text

        # Telegram error path called with the reflection title.
        reflection_alerts = [
            c for c in notifier_calls
            if c["payload"] and c["payload"].get("kind") == "reflection_report"
        ]
        assert reflection_alerts, (
            f"telegram callback not fired; got: {notifier_calls}"
        )


@pytest.mark.asyncio
async def test_run_miss_penalty_pass_holds_when_trades_above_threshold(
    tmp_path: Path,
) -> None:
    """When the audit log shows >= reflection_trade_threshold approved
    decisions in the window, even 3 missed pumps must NOT trigger
    reflection mode."""
    cfg = _base_config(tmp_path=tmp_path, port_base=19113)

    async with _running_app(cfg) as app:
        rejected_at_s = time.time() - 30 * 3600
        rejected_at_ms = int(rejected_at_s * 1000)
        symbols = ["PEPE/USDT:USDT", "WIF/USDT:USDT", "DOGE/USDT:USDT"]

        # Three missed-pump rejections AND three approved decisions
        # within the 7-day window.
        log_path = Path(cfg.decision_audit_log_path)
        for i, sym in enumerate(symbols):
            _write_rejected_decision(
                log_path, ts=rejected_at_s + i,
                trace_id=f"reject-{sym}", symbol=sym,
            )
        # Approved decisions to lift the trade count above the threshold.
        for i in range(3):
            rec = {
                "ts": rejected_at_s + 100 + i,
                "trace_id": f"approved-{i}",
                "symbol": "BTC/USDT:USDT", "direction": "long",
                "current_price": 50_000.0,
                "approved": True, "reason": "ok",
                "final_score": 95.0, "rule_score": 80.0,
                "signal_kind": "volume_spike",
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        bars_by_symbol = {
            sym: _make_pump_klines(start_ms=rejected_at_ms)
            for sym in symbols
        }
        assert app._miss_penalty is not None
        app._miss_penalty.kline_fetcher = fixed_klines_fetcher(bars_by_symbol)

        wed = datetime(2026, 5, 13, 2, 0, 0, tzinfo=timezone.utc)
        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        await app._run_miss_penalty_pass(account=account, now_utc=wed)

        # Reflection NOT triggered.
        assert app._reflection is not None
        assert app._reflection.is_suspended() is False
        assert app._reflection.has_pending_review() is False

        # No markdown report should have been written.
        reports_dir = Path(cfg.reflection_reports_dir)
        if reports_dir.exists():
            assert list(reports_dir.glob("reflection_*.md")) == []


@pytest.mark.asyncio
async def test_run_miss_penalty_pass_emits_threshold_proposals_on_sunday(
    tmp_path: Path,
) -> None:
    """On the configured weekly weekday the threshold tuner should
    write threshold_overrides.json. The proposed values must respect
    the hard limits (anti_chase <= 6%)."""
    cfg = _base_config(
        tmp_path=tmp_path, port_base=19115,
        threshold_tuner_run_on_weekday=2,  # Wednesday for this test
    )

    async with _running_app(cfg) as app:
        # Seed the scorer state directly: a deeply-negative
        # anti_chase confidence with enough samples to qualify.
        rejected_at_s = time.time() - 48 * 3600

        log_path = Path(cfg.decision_audit_log_path)
        # 50 rejections + 30 missed pumps -> confidence ~= 50-90 = -40
        # Mix the symbols so each MissedOpportunity is unique.
        for i in range(50):
            _write_rejected_decision(
                log_path, ts=rejected_at_s + i,
                trace_id=f"r{i}", symbol=f"SYM{i}/USDT:USDT",
                reason="chase_too_late:0.04",
            )

        # Pre-seed missed_opportunities.jsonl so the scorer sees 30 misses.
        # We pre-seed BOTH the misses (r0..r29) AND the correct rejects
        # (r30..r49) -- the latter with is_missed_pump=False -- so the
        # audit step below has no work to do (every trace_id is in the
        # dedup set). Without pre-seeding the correct rejects, the audit
        # would call the empty kline fetcher for r30..r49 and label them
        # ``insufficient_data``, demoting the sample count below the
        # tuner's samples_required threshold.
        missed_dir = Path(cfg.miss_penalty_state_dir)
        missed_dir.mkdir(parents=True, exist_ok=True)
        missed_path = missed_dir / "missed_opportunities.jsonl"
        with missed_path.open("w", encoding="utf-8") as f:
            for i in range(50):
                opp = MissedOpportunity(
                    trace_id=f"r{i}", symbol=f"SYM{i}/USDT:USDT",
                    rejected_at_ts_ms=int((rejected_at_s + i) * 1000),
                    rejected_reason="chase_too_late:0.04",
                    rejected_reason_bucket="chase_too_late",
                    rejected_score=78.0, direction="long",
                    entry_price_if_taken=1.0,
                    realized_max_favorable_pct=2.0 if i < 30 else 0.10,
                    realized_max_adverse_pct=-0.05,
                    is_missed_pump=(i < 30),
                    miss_severity=0.85 if i < 30 else 0.0,
                    bars_observed=1440,
                )
                f.write(json.dumps(opp.to_dict()) + "\n")

        # Don't fetch any new K-lines -- empty fetcher, audit will
        # noop because trace ids are already in missed_opportunities.
        assert app._miss_penalty is not None
        app._miss_penalty.kline_fetcher = fixed_klines_fetcher({})
        # Hydrate dedup set so audit short-circuits on every record.
        app._miss_penalty._labelled_trace_ids = {
            f"r{i}" for i in range(50)
        }

        wed = datetime(2026, 5, 13, 2, 0, 0, tzinfo=timezone.utc)
        assert wed.weekday() == 2

        account = AccountState(
            equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
        )
        account.reconciliation_complete = True
        await app._run_miss_penalty_pass(account=account, now_utc=wed)

        overrides_path = (
            Path(cfg.miss_penalty_state_dir) / "threshold_overrides.json"
        )
        assert overrides_path.exists()
        data = json.loads(overrides_path.read_text())
        chase = next(
            ov for ov in data["overrides"]
            if ov["reason"] == "chase_too_late"
        )
        assert chase["direction_label"] == "loosen"
        assert chase["new"] > 0.025
        assert chase["new"] <= 0.06   # hard_max respected


# ====================================================================== #
# trade-counter helper (uses audit log; consumed by reflection trigger)
# ====================================================================== #


@pytest.mark.asyncio
async def test_count_recent_trades_matches_audit_log(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path=tmp_path, port_base=19117)
    async with _running_app(cfg) as app:
        log_path = Path(cfg.decision_audit_log_path)
        now = time.time()
        # 4 approved trades over the last week, 1 over a month ago.
        records = [
            {"ts": now - i * 86400, "trace_id": f"a{i}",
             "symbol": "X", "direction": "long", "current_price": 1.0,
             "approved": True, "reason": "ok",
             "final_score": 95.0, "rule_score": 80.0,
             "signal_kind": "volume_spike"}
            for i in (0, 1, 2, 3)
        ]
        records.append({
            "ts": now - 60 * 86400, "trace_id": "old",
            "symbol": "X", "direction": "long", "current_price": 1.0,
            "approved": True, "reason": "ok",
            "final_score": 95.0, "rule_score": 80.0,
            "signal_kind": "volume_spike",
        })
        # And a few rejections that should NOT count.
        records.extend([
            {"ts": now - i * 3600, "trace_id": f"r{i}",
             "symbol": "Y", "direction": "long", "current_price": 1.0,
             "approved": False, "reason": "anti_chase:0.04",
             "final_score": 70.0, "rule_score": 60.0,
             "signal_kind": "volume_spike"}
            for i in (1, 2, 3)
        ])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        count = app._count_recent_trades(window_sec=7 * 86400)
        assert count == 4
