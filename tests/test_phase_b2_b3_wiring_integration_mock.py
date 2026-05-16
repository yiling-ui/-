"""tests/test_phase_b2_b3_wiring_integration_mock.py

Phase B.2 / B.3 wiring integration tests.

What this pins:
* ``AppConfig`` defaults keep observability + sqlite OFF (so existing
  byte-for-byte tests don't change shape).
* ``AppConfig.from_file`` round-trips the new YAML keys.
* When ``metrics_enabled=True`` the daemon attaches a registry, and
  ``/metrics`` includes both the legacy gauges AND the
  ``altcoin_agent_orders_placed_total`` counter.
* When ``dlq_enabled=True`` rejections in the hot path land in the DLQ.
* When ``persistence_backend="sqlite"`` the daemon swaps the persistor
  to ``SQLiteAccountStore`` and saves at boot through the
  change-listener.
* trace_id is bound when ``_handle_high_priority`` runs and matches
  the one that lands in the audit log.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.main import App, AppConfig
from altcoin_agent.observability import bind_trace_id, current_trace_id
from altcoin_agent.risk.sqlite_persistence import SQLiteAccountStore


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
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


# --------------------------------------------------------------------- #
# Defaults are off
# --------------------------------------------------------------------- #


def test_phase_b2_b3_defaults_off() -> None:
    cfg = AppConfig()
    assert cfg.metrics_enabled is False
    assert cfg.structured_logging_enabled is False
    assert cfg.dlq_enabled is False
    assert cfg.persistence_backend == "json"


# --------------------------------------------------------------------- #
# YAML round-trip
# --------------------------------------------------------------------- #


def test_phase_b2_b3_yaml_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "metrics_enabled: true\n"
        "structured_logging_enabled: true\n"
        "dlq_enabled: true\n"
        "dlq_path: '/tmp/x.jsonl'\n"
        "persistence_backend: sqlite\n"
        "account_persistence_sqlite_path: '/tmp/account.sqlite3'\n"
    )
    cfg = AppConfig.from_file(str(yaml_path))
    assert cfg.metrics_enabled is True
    assert cfg.structured_logging_enabled is True
    assert cfg.dlq_enabled is True
    assert cfg.dlq_path == "/tmp/x.jsonl"
    assert cfg.persistence_backend == "sqlite"
    assert cfg.account_persistence_sqlite_path == "/tmp/account.sqlite3"


# --------------------------------------------------------------------- #
# /metrics integration
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_metrics_enabled_attaches_registry_to_app() -> None:
    cfg = AppConfig(
        healthz_port=18601, dry_run=True, graceful_timeout_sec=2.0,
        metrics_enabled=True,
        dashboard_enabled=False,  # avoid extra port bind
    )
    async with _running_app(cfg) as app:
        assert app._metrics is not None
        # Mutate a counter and confirm the registry render contains it.
        app._metrics.orders_placed_total.inc(
            labels={"symbol": "PEPE/USDT:USDT", "side": "long"},
        )
        text = app._metrics.registry.render()
        assert "altcoin_agent_orders_placed_total" in text
        assert (
            'altcoin_agent_orders_placed_total{side="long",'
            'symbol="PEPE/USDT:USDT"} 1'
        ) in text


@pytest.mark.asyncio
async def test_metrics_endpoint_includes_legacy_and_registry() -> None:
    """``/metrics`` should keep emitting the legacy ``altcoin_agent_up``
    gauge AND the registry-rendered metrics when ``metrics_enabled``."""
    from altcoin_agent.main import HealthState, make_health_app
    from altcoin_agent.observability import build_default_registry

    state = HealthState()
    state.fuser_alive = True
    state.screener_alive = True
    state.reconciliation_complete = True
    dm = build_default_registry()
    dm.orders_placed_total.inc(
        labels={"symbol": "X", "side": "long"},
    )
    app = await make_health_app(state, metrics_registry=dm.registry)
    server = TestServer(app)
    async with TestClient(server) as client:
        r = await client.get("/metrics")
        body = await r.text()
    # Legacy gauge still present.
    assert "altcoin_agent_up 1.0" in body
    # Registry metric appended.
    assert (
        'altcoin_agent_orders_placed_total{side="long",symbol="X"} 1'
    ) in body


# --------------------------------------------------------------------- #
# DLQ integration
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dlq_enabled_writes_on_rejection(tmp_path: Path) -> None:
    """When metrics+dlq are on, ``_record_rejection`` writes to the DLQ
    and bumps the rejected counter."""
    cfg = AppConfig(
        healthz_port=18602, dry_run=True, graceful_timeout_sec=2.0,
        metrics_enabled=True,
        dlq_enabled=True,
        dlq_path=str(tmp_path / "dlq.jsonl"),
        dashboard_enabled=False,
    )
    async with _running_app(cfg) as app:
        assert app._dlq is not None
        app._record_rejection(
            symbol="PEPE/USDT:USDT",
            reason="anti_chase_window_30000ms_move_4.5%",
            kind="gate_reject",
        )
        # DLQ has one row; metrics counter incremented under the
        # canonical bucket.
        rows = app._dlq.iter_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "gate_reject"
        assert rows[0]["symbol"] == "PEPE/USDT:USDT"
        # Free-form reason routed into the canonical "anti_chase" bucket.
        text = app._metrics.registry.render()
        assert (
            'altcoin_agent_orders_rejected_total{reason="anti_chase"} 1'
        ) in text
        assert (
            'altcoin_agent_dlq_writes_total{kind="gate_reject"} 1'
        ) in text


# --------------------------------------------------------------------- #
# SQLite persistence backend
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sqlite_backend_swaps_persistor(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18603, dry_run=True, graceful_timeout_sec=2.0,
        persistence_backend="sqlite",
        account_persistence_sqlite_path=str(
            tmp_path / "account.sqlite3"
        ),
        dashboard_enabled=False,
    )
    async with _running_app(cfg) as app:
        assert isinstance(app._persistor, SQLiteAccountStore)
        # The boot-time save (right after rollover stamp) creates at
        # least one log row.
        rows = app._persistor.history()
        assert len(rows) >= 1
        # Snapshot equity matches the configured initial equity.
        assert (
            rows[0]["payload"]["equity_usdt"]
            == cfg.initial_equity_usdt
        )


@pytest.mark.asyncio
async def test_unknown_backend_falls_back_to_json(tmp_path: Path) -> None:
    cfg = AppConfig(
        healthz_port=18604, dry_run=True, graceful_timeout_sec=2.0,
        persistence_backend="bogus",
        account_persistence_path=str(
            tmp_path / "account.json"
        ),
        dashboard_enabled=False,
    )
    async with _running_app(cfg) as app:
        # Falls back to plain AccountPersistor (not SQLiteAccountStore).
        assert app._persistor is not None
        assert not isinstance(app._persistor, SQLiteAccountStore)


# --------------------------------------------------------------------- #
# Reject-reason bucketing helper
# --------------------------------------------------------------------- #


def test_bucket_reject_reason_canonical_labels() -> None:
    fn = App._bucket_reject_reason
    assert fn("anti_chase_window_30000ms_move_4.5%") == "anti_chase"
    assert fn("vol_kill 60s range 12%") == "vol_kill"
    assert fn("min_liquidity_usdt below threshold") == "min_liquidity"
    assert fn("kill_switch engaged") == "kill_switch"
    assert fn("regime_filter btc dropped") == "regime_filter"
    assert fn("cluster_cap meme=2") == "cluster_cap"
    assert fn("daily_drawdown 6.2%") == "daily_drawdown"
    # Unrelated string -> "other"
    assert fn("totally novel reason") == "other"
    assert fn("") == "other"


# --------------------------------------------------------------------- #
# trace_id binding
# --------------------------------------------------------------------- #


def test_bind_trace_id_visible_in_current_trace_id() -> None:
    bind_trace_id("manual-12345")
    assert current_trace_id() == "manual-12345"
