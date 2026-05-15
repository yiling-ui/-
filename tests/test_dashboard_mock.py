"""Tests for the dashboard module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from altcoin_agent.dashboard import DashboardState, install_dashboard
from altcoin_agent.main import HealthState
from altcoin_agent.risk.state import AccountState, Position, Side


@pytest.mark.asyncio
async def test_dashboard_endpoints_serve_state(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [
            {"feature_name": "f1", "bucket": "pos_xlarge", "side": "pump",
             "hits": 4, "total": 5, "last_seen_ts_ms": 1},
            {"feature_name": "f2", "bucket": "neutral", "side": "dump",
             "hits": 0, "total": 5, "last_seen_ts_ms": 1},
        ],
    }))

    state = DashboardState(rules_path=rules_path)
    state.health = HealthState(
        started_at=0.0, fuser_alive=True, screener_alive=True,
        reconciliation_complete=True, high_priority_count=3,
        rule_event_count=42, orders_placed=2, orders_rejected=1,
        open_positions=1, last_signal_ts=12345.0, last_error=None,
    )
    account = AccountState(equity_usdt=10_000.0)
    account.open_positions["RAVEUSDT"] = Position(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=10.0, leverage=5.0,
        initial_stop=0.95, current_stop=1.0,
    )
    state.account = account

    state.push_signal({
        "symbol": "RAVEUSDT", "direction": "long", "final_score": 95,
        "ts": 1, "trigger_price": 1.0,
    })
    state.push_order({"ts": 2, "symbol": "RAVEUSDT", "side": "long",
                       "size": 10, "type": "MARKET"})
    state.push_rejection({"ts": 3, "symbol": "BLA", "reason": "low_liq"})

    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")

    async with TestClient(TestServer(app)) as client:
        r = await client.get("/dashboard")
        assert r.status == 200
        text = await r.text()
        assert "Altcoin Agent" in text

        r = await client.get("/api/state")
        body = await r.json()
        assert body["status"] == "ok"
        assert body["mode"] == "DRY-RUN"
        assert body["high_priority_count"] == 3
        assert body["orders_placed"] == 2

        r = await client.get("/api/signals")
        sigs = await r.json()
        assert len(sigs) == 1
        assert sigs[0]["symbol"] == "RAVEUSDT"

        r = await client.get("/api/orders")
        ords = await r.json()
        assert ords[0]["type"] == "MARKET"

        r = await client.get("/api/rejections")
        rejs = await r.json()
        assert rejs[0]["reason"] == "low_liq"

        r = await client.get("/api/positions")
        pos = await r.json()
        assert len(pos) == 1
        assert pos[0]["symbol"] == "RAVEUSDT"
        assert pos[0]["side"] == "long"

        r = await client.get("/api/rules")
        rules = await r.json()
        assert rules["total"] == 2
        # f1 hit_rate = (4+1)/(5+2) ~= 0.71 -> top
        assert rules["top"][0]["feature_name"] == "f1"
        assert rules["top"][0]["hit_rate"] > 0.5


@pytest.mark.asyncio
async def test_dashboard_handles_missing_rules_file(tmp_path: Path) -> None:
    state = DashboardState(rules_path=tmp_path / "absent.json")
    state.health = HealthState(reconciliation_complete=True)
    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/api/rules")
        body = await r.json()
        assert body["total"] == 0
        assert body["top"] == []


@pytest.mark.asyncio
async def test_dashboard_handles_corrupt_rules_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid")
    state = DashboardState(rules_path=p)
    state.health = HealthState(reconciliation_complete=True)
    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/api/rules")
        body = await r.json()
        assert body["total"] == 0
        assert "error" in body
