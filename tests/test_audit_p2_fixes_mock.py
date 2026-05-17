"""Tests for the audit P-2 / P-3 vulnerability fixes.

Covers three vulnerabilities surfaced in the extreme-stress audit:

  * P-2.1  ``enforce_persistence_enabled_for_real_orders`` refuses to
           start LIVE / paper-trade when persistence is disabled,
           because that combination silently turns the daily-DD
           circuit breaker into a one-way latch.

  * P-2.2  ``AccountPersistor`` exposes ``save_count`` /
           ``save_errors`` / ``last_save_error`` and the /metrics
           endpoint surfaces them as Prometheus gauges, so a clogged
           disk becomes a visible alert instead of a silent log line.

  * P-3.1  ``LLMEngine.judge`` enforces ``total_budget_sec`` via
           ``asyncio.wait_for``, returning a neutral degraded
           ``AIVerdict`` when a slow provider would otherwise spend
           ~16.5s in the worst case.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from altcoin_agent.ai_engine import (
    AIVerdict,
    LLMEngine,
    SMCContext,
)
from altcoin_agent.main import (
    AppConfig,
    HealthState,
    enforce_persistence_enabled_for_real_orders,
    make_health_app,
)
from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.state import AccountState

# ====================================================================== #
# P-2.1 — persistence enforcement in LIVE / paper-trade
# ====================================================================== #


def test_p21_dry_run_with_persistence_off_is_allowed() -> None:
    """Dry-run never touches venue funds; persistence off is fine."""
    cfg = AppConfig(
        dry_run=True, paper_trade=False,
        account_persistence_enabled=False,
    )
    enforce_persistence_enabled_for_real_orders(cfg)  # no raise


def test_p21_live_with_persistence_on_is_allowed() -> None:
    cfg = AppConfig(
        dry_run=False, paper_trade=False,
        account_persistence_enabled=True,
    )
    enforce_persistence_enabled_for_real_orders(cfg)  # no raise


def test_p21_paper_trade_with_persistence_on_is_allowed() -> None:
    cfg = AppConfig(
        dry_run=False, paper_trade=True,
        account_persistence_enabled=True,
    )
    enforce_persistence_enabled_for_real_orders(cfg)  # no raise


def test_p21_live_with_persistence_off_systemexits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = AppConfig(
        dry_run=False, paper_trade=False,
        account_persistence_enabled=False,
    )
    with caplog.at_level(logging.CRITICAL, logger="altcoin_agent.main"):
        with pytest.raises(SystemExit) as exc:
            enforce_persistence_enabled_for_real_orders(cfg)
    assert exc.value.code == 4
    blob = "\n".join(r.getMessage() for r in caplog.records)
    # The error message must mention the exact knob to flip and the
    # consequence so the operator can fix it without grepping.
    assert "account_persistence_enabled" in blob
    assert "daily-drawdown" in blob


def test_p21_paper_trade_with_persistence_off_systemexits() -> None:
    cfg = AppConfig(
        dry_run=False, paper_trade=True,
        account_persistence_enabled=False,
    )
    with pytest.raises(SystemExit) as exc:
        enforce_persistence_enabled_for_real_orders(cfg)
    assert exc.value.code == 4


# ====================================================================== #
# P-2.2 — AccountPersistor save error counters & /metrics export
# ====================================================================== #


def test_p22_persistor_increments_save_count_on_success(tmp_path: Path) -> None:
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = AccountState()
    assert p.save_count == 0
    assert p.save_errors == 0

    assert p.save(a) is True
    assert p.save_count == 1
    assert p.save_errors == 0
    assert p.last_save_error is None
    assert p.last_saved_ts > 0


def test_p22_persistor_increments_save_errors_on_failure(tmp_path: Path) -> None:
    """A save() that fails after construction must increment save_errors,
    stash last_save_error, and never raise."""
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = AccountState()
    # First save succeeds and bumps the counter.
    assert p.save(a) is True
    assert p.save_count == 1
    # Now nuke the parent dir AND replace with a file so that the
    # tempfile.mkstemp call inside save() fails — the trading loop must
    # see a False, not an exception.
    import shutil
    shutil.rmtree(tmp_path / "acc.json", ignore_errors=True)
    p.path.unlink(missing_ok=True)
    # Replace the parent dir with a regular file so mkstemp fails.
    parent = p.path.parent
    shutil.rmtree(parent)
    parent.write_text("now a file, no longer a dir")

    assert p.save(a) is False
    assert p.save_count == 1   # unchanged
    assert p.save_errors == 1
    assert p.last_save_error is not None
    assert p.last_save_error_ts > 0


@pytest.mark.asyncio
async def test_p22_metrics_endpoint_exports_persistor_gauges() -> None:
    state = HealthState()
    state.started_at = time.time() - 5.0
    state.fuser_alive = True
    state.screener_alive = True
    state.reconciliation_complete = True
    state.persistor_save_count = 42
    state.persistor_save_errors = 3
    state.persistor_last_saved_ts = time.time() - 1.0

    app = await make_health_app(state)
    server = TestServer(app)
    async with TestClient(server) as client:
        r = await client.get("/metrics")
        assert r.status == 200
        body = await r.text()

    assert "altcoin_agent_persistor_save_count 42" in body
    assert "altcoin_agent_persistor_save_errors 3" in body
    assert "# HELP altcoin_agent_persistor_save_errors" in body
    assert "# TYPE altcoin_agent_persistor_save_errors gauge" in body


@pytest.mark.asyncio
async def test_p22_healthz_includes_persistor_fields() -> None:
    state = HealthState()
    state.started_at = time.time() - 5.0
    state.fuser_alive = True
    state.screener_alive = True
    state.reconciliation_complete = True
    state.persistor_save_count = 7
    state.persistor_save_errors = 1
    state.persistor_last_save_error = "OSError: disk full"

    app = await make_health_app(state)
    server = TestServer(app)
    async with TestClient(server) as client:
        r = await client.get("/healthz")
        assert r.status == 200
        body = await r.json()

    assert body["persistor_save_count"] == 7
    assert body["persistor_save_errors"] == 1
    assert body["persistor_last_save_error"] == "OSError: disk full"


# ====================================================================== #
# P-3.1 — LLMEngine.judge() total budget deadline
# ====================================================================== #


class _SlowProvider:
    """Async chat provider whose RTT we control with a parameter."""

    name = "slow"
    model = "slow-1"
    api_base = ""

    def __init__(self, delay_sec: float):
        self.delay_sec = delay_sec
        self.calls = 0

    async def chat_json(self, messages, *, timeout):  # noqa: ANN001
        self.calls += 1
        await asyncio.sleep(self.delay_sec)
        # Return a valid verdict so we know failure was the deadline,
        # not a parse / schema problem.
        return ('{"intent":"pump","confidence_score":80,'
                '"reason":"ok","kol_intent":"neutral",'
                '"key_evidence":["a"]}'), 10

    async def aclose(self) -> None:  # pragma: no cover - trivial
        return None


@pytest.mark.asyncio
async def test_p31_judge_returns_neutral_when_total_budget_exceeded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _SlowProvider(delay_sec=2.0)
    # Budget below the provider RTT -> the deadline must fire.
    engine = LLMEngine(provider=provider, total_budget_sec=0.1, timeout=5.0)

    smc = SMCContext()
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="altcoin_agent.ai_engine"):
        verdict = await engine.judge(
            symbol="PEPE", exchange="binance",
            funding_rate=None, funding_deviation_z=None,
            smc=smc, posts=[],
        )
    elapsed = time.monotonic() - t0

    # Deadline must trip well before the provider's 2s would complete.
    assert elapsed < 1.0, f"deadline did not fire (took {elapsed:.2f}s)"
    assert isinstance(verdict, AIVerdict)
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 0
    assert "total_budget_exceeded" in verdict.reason
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "total_budget_sec" in blob


@pytest.mark.asyncio
async def test_p31_judge_completes_when_under_budget() -> None:
    """Sanity: a fast provider must return its real verdict, not the
    degraded one."""
    provider = _SlowProvider(delay_sec=0.05)
    engine = LLMEngine(provider=provider, total_budget_sec=2.0, timeout=5.0)
    verdict = await engine.judge(
        symbol="PEPE", exchange="binance",
        funding_rate=None, funding_deviation_z=None,
        smc=SMCContext(), posts=[],
    )
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 80


@pytest.mark.asyncio
async def test_p31_judge_disables_deadline_when_set_to_zero() -> None:
    """Operators can opt out by setting ``total_budget_sec=0``; tests
    rely on this so an environment-induced slow CI doesn't false-fail."""
    provider = _SlowProvider(delay_sec=0.05)
    engine = LLMEngine(provider=provider, total_budget_sec=0.0, timeout=5.0)
    verdict = await engine.judge(
        symbol="PEPE", exchange="binance",
        funding_rate=None, funding_deviation_z=None,
        smc=SMCContext(), posts=[],
    )
    assert verdict.intent == "pump"
