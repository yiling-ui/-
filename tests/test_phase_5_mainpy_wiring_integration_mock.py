"""tests/test_phase_5_mainpy_wiring_integration_mock.py

Phase 5 main.py wiring integration tests.

What this pins:
* ``AppConfig`` defaults keep all three Phase 5 features OFF, so any
  existing dry-run that doesn't set the flags behaves byte-for-byte
  like v1.0.
* ``AppConfig.from_file`` round-trips the new YAML keys.
* When ``llm_cache_enabled=True`` the daemon attaches a ``LLMCache``
  to the engine so ``LLMEngine.judge`` can see + serve cached verdicts.
* When ``llm_budget_manager_enabled=True`` the daemon attaches a
  ``TokenBudgetManager`` and the engine's tier-aware gating is live.
* When ``llm_pre_rate_enabled=True`` the daemon constructs the
  ``LLMPreRater`` and starts its background worker.
* The pre-rater is gated on the cache being enabled too — otherwise
  it would burn tokens on every candidate without short-circuiting,
  which the plan explicitly forbids.
* On shutdown, the pre-rater stops cleanly before the engine closes
  so any in-flight rating finishes its provider call.

We use the same ``_running_app`` pattern as the Phase B.2/B.3 wiring
test so the daemon starts in dry-run mode (no real network).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from altcoin_agent.ai_engine import AIVerdict, LLMEngine
from altcoin_agent.llm.cache import LLMCache
from altcoin_agent.llm.pre_rater import LLMPreRater
from altcoin_agent.llm.token_budget import BudgetMode, TokenBudgetManager
from altcoin_agent.llm_provider import LLMProvider
from altcoin_agent.main import App, AppConfig

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


class _SilentProvider(LLMProvider):
    """Stub provider so the App can construct an engine without a
    network round-trip.  Never actually called in these tests because
    the App's run loop doesn't dispatch any signals before we shut
    it down."""

    name = "silent"
    model = "test"

    async def chat_json(self, messages, *, timeout=8.0):
        return ('{"intent": "neutral", "confidence_score": 0, '
                '"reason": "noop", "kol_intent": "neutral", '
                '"key_evidence": []}'), 0

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
    """Spin up the daemon in dry-run + neutered-screener mode so
    construction completes without external IO. Mirrors the helper
    in ``test_phase_b2_b3_wiring_integration_mock.py``.

    We patch ``DEEPSEEK_API_KEY`` and inject a stub provider so the
    engine's ``__post_init__`` doesn't try to build a real DeepSeek
    HTTP client.
    """

    # Pre-build the engine with our stub provider; assigning it via
    # the App slot lets ``run`` skip its own ``DeepSeekEngine()``
    # construction (which would require a real API key).
    app = App(cfg=cfg)
    app._llm_engine = LLMEngine(provider=_SilentProvider())  # type: ignore[assignment]

    # Pretend the env var is set so the run-time branch that checks
    # ``os.getenv("DEEPSEEK_API_KEY")`` doesn't overwrite our stub.
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-token"}):
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


# --------------------------------------------------------------------- #
# Defaults are OFF
# --------------------------------------------------------------------- #


def test_phase_5_defaults_all_off() -> None:
    cfg = AppConfig()
    assert cfg.llm_cache_enabled is False
    assert cfg.llm_budget_manager_enabled is False
    assert cfg.llm_pre_rate_enabled is False


def test_phase_5_default_paths_match_plan() -> None:
    cfg = AppConfig()
    assert cfg.llm_cache_path == ".kiro/state/llm_cache.json"
    assert cfg.llm_budget_state_path == ".kiro/state/token_usage.json"
    assert cfg.llm_budget_monthly_tokens == 5_000_000
    assert cfg.llm_pre_rate_min_score == 70.0


# --------------------------------------------------------------------- #
# YAML round-trip
# --------------------------------------------------------------------- #


def test_phase_5_yaml_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "llm_cache_enabled: true\n"
        "llm_cache_path: '/tmp/cache.json'\n"
        "llm_cache_max_entries: 256\n"
        "llm_cache_ttl_sec: 1800\n"
        "llm_budget_manager_enabled: true\n"
        "llm_budget_state_path: '/tmp/budget.json'\n"
        "llm_budget_monthly_tokens: 1000000\n"
        "llm_pre_rate_enabled: true\n"
        "llm_pre_rate_min_score: 80.0\n"
        "llm_pre_rate_queue_max: 32\n"
    )
    cfg = AppConfig.from_file(str(yaml_path))
    assert cfg.llm_cache_enabled is True
    assert cfg.llm_cache_path == "/tmp/cache.json"
    assert cfg.llm_cache_max_entries == 256
    assert cfg.llm_cache_ttl_sec == 1800
    assert cfg.llm_budget_manager_enabled is True
    assert cfg.llm_budget_state_path == "/tmp/budget.json"
    assert cfg.llm_budget_monthly_tokens == 1_000_000
    assert cfg.llm_pre_rate_enabled is True
    assert cfg.llm_pre_rate_min_score == 80.0
    assert cfg.llm_pre_rate_queue_max == 32


# --------------------------------------------------------------------- #
# Wiring — feature OFF baseline
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_baseline_off_constructs_no_phase_5_components(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18701,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        assert app._llm_cache is None
        assert app._token_budget_manager is None
        assert app._llm_pre_rater is None
        # Engine still has no cache / budget_manager attached.
        assert app._llm_engine is not None
        assert app._llm_engine.cache is None
        assert app._llm_engine.budget_manager is None


# --------------------------------------------------------------------- #
# Cache-only wiring
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cache_only_attaches_to_engine(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18702,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=True,
        llm_cache_path=str(tmp_path / "cache.json"),
        llm_cache_max_entries=64,
        llm_cache_ttl_sec=600,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        assert isinstance(app._llm_cache, LLMCache)
        assert app._llm_cache.max_entries == 64
        assert app._llm_cache.default_ttl_sec == 600
        # Engine must see the cache.
        assert app._llm_engine.cache is app._llm_cache
        # Budget manager / pre-rater stayed off.
        assert app._token_budget_manager is None
        assert app._llm_pre_rater is None


# --------------------------------------------------------------------- #
# Budget-manager wiring
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_manager_attaches_to_engine(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18703,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_budget_manager_enabled=True,
        llm_budget_state_path=str(tmp_path / "budget.json"),
        llm_budget_monthly_tokens=200_000,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        assert isinstance(app._token_budget_manager, TokenBudgetManager)
        assert app._token_budget_manager.monthly_budget == 200_000
        assert app._token_budget_manager.mode() is BudgetMode.FREE
        assert app._llm_engine.budget_manager is app._token_budget_manager


# --------------------------------------------------------------------- #
# Pre-rate worker wiring
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pre_rater_starts_when_cache_present(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18704,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=True,
        llm_cache_path=str(tmp_path / "cache.json"),
        llm_pre_rate_enabled=True,
        llm_pre_rate_min_score=75.0,
        llm_pre_rate_queue_max=16,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        assert isinstance(app._llm_pre_rater, LLMPreRater)
        assert app._llm_pre_rater.prerate_min_score == 75.0
        assert app._llm_pre_rater.queue_maxsize == 16
        # Worker task should be alive.
        assert app._llm_pre_rater._task is not None
        assert not app._llm_pre_rater._task.done()


@pytest.mark.asyncio
async def test_pre_rate_disabled_when_cache_disabled(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The plan explicitly forbids pre-rating without a cache (every
    rating would burn tokens with no future hit). The wiring must
    detect this and refuse to construct the worker."""
    cfg = AppConfig(
        healthz_port=18705,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=False,           # cache OFF
        llm_pre_rate_enabled=True,         # but pre-rate ON
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    with caplog.at_level("WARNING"):
        async with _running_app(cfg) as app:
            assert app._llm_cache is None
            assert app._llm_pre_rater is None
            # And the operator gets a warning so it's not silent drift.
            assert any(
                "pre-rate enabled but cache disabled" in r.message
                for r in caplog.records
            )


# --------------------------------------------------------------------- #
# Full Phase 5 stack — all three on
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_full_stack_wires_engine_to_cache_and_budget(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18706,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=True,
        llm_cache_path=str(tmp_path / "cache.json"),
        llm_budget_manager_enabled=True,
        llm_budget_state_path=str(tmp_path / "budget.json"),
        llm_budget_monthly_tokens=5_000_000,
        llm_pre_rate_enabled=True,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        # Each component constructed.
        assert app._llm_cache is not None
        assert app._token_budget_manager is not None
        assert app._llm_pre_rater is not None
        # Engine sees both collaborators.
        assert app._llm_engine.cache is app._llm_cache
        assert app._llm_engine.budget_manager is app._token_budget_manager
        # Pre-rater shares the same cache + budget manager (so cache
        # writes from pre-rating are visible to the hot path and the
        # budget gate is one shared counter).
        assert app._llm_pre_rater.cache is app._llm_cache
        assert app._llm_pre_rater.budget_manager is app._token_budget_manager
        assert app._llm_pre_rater.engine is app._llm_engine


# --------------------------------------------------------------------- #
# Shutdown order
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_shutdown_stops_pre_rater_before_engine(tmp_path: Path) -> None:
    """The pre-rater must stop before the engine closes its provider,
    so any in-flight rating call gets a chance to finish cleanly."""
    cfg = AppConfig(
        healthz_port=18707,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=True,
        llm_cache_path=str(tmp_path / "cache.json"),
        llm_pre_rate_enabled=True,
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        rater = app._llm_pre_rater
        assert rater is not None
        assert rater._task is not None
    # After ``_running_app`` exits, the pre-rater task must have ended.
    assert rater._task is None or rater._task.done()


# --------------------------------------------------------------------- #
# Engine-aware behaviours actually flow through
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_engine_uses_cache_after_wiring(tmp_path: Path) -> None:
    """End-to-end: the engine the App built should actually serve a
    cached verdict if we pre-fill the cache, validating that the
    engine reference is the same one wired up."""
    cfg = AppConfig(
        healthz_port=18708,
        dashboard_enabled=False,
        graceful_timeout_sec=2.0,
        dry_run=True,
        llm_cache_enabled=True,
        llm_cache_path=str(tmp_path / "cache.json"),
        account_persistence_path=str(tmp_path / "account.json"),
        decision_audit_log_path=str(tmp_path / "decisions.jsonl"),
    )
    async with _running_app(cfg) as app:
        # Pre-fill the cache directly so the ``_SilentProvider``
        # never has to be invoked.
        verdict = AIVerdict(
            intent="pump", confidence_score=85,
            reason="cache primed by test",
            kol_intent="neutral", key_evidence=[],
        )
        app._llm_cache.put(
            symbol="X/USDT:USDT",
            phase="ramp",
            social_hash="empty",
            verdict=verdict.model_dump(),
        )

        # Drive the engine via the same path the daemon would.
        from altcoin_agent.ai_engine import SMCContext
        out = await app._llm_engine.judge(
            symbol="X/USDT:USDT", exchange="binance",
            funding_rate=None, funding_deviation_z=None,
            smc=SMCContext(), posts=[],
            phase="ramp",
        )
        assert out.intent == "pump"
        assert out.confidence_score == 85
        assert "primed by test" in out.reason
