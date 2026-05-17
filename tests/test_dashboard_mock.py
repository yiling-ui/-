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



@pytest.mark.asyncio
async def test_dashboard_pnl_curve_endpoint(tmp_path: Path) -> None:
    """Operational patch: ``/api/pnl-curve`` returns equity snapshots
    captured by ``record_equity_snapshot``."""
    state = DashboardState(rules_path=tmp_path / "absent.json")
    state.health = HealthState(reconciliation_complete=True)

    # Two snapshots representing a small profit run.
    account = AccountState(equity_usdt=10_000.0)
    account.realized_pnl_today_usdt = 0.0
    state.account = account
    state.record_equity_snapshot()

    account.equity_usdt = 10_500.0
    account.realized_pnl_today_usdt = 500.0
    state.record_equity_snapshot()

    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/api/pnl-curve")
        assert r.status == 200
        body = await r.json()
        assert isinstance(body, list)
        assert len(body) == 2
        # Most recent snapshot is at the end (FIFO append).
        assert body[-1]["equity_usdt"] == pytest.approx(10_500.0)
        assert body[-1]["realized_pnl_today_usdt"] == pytest.approx(500.0)
        # Required fields present on every point.
        for p in body:
            assert "ts" in p and "equity_usdt" in p
            assert "open_positions" in p


@pytest.mark.asyncio
async def test_dashboard_pnl_curve_skips_zero_or_missing_account(
    tmp_path: Path,
) -> None:
    """Defensive: ``record_equity_snapshot`` must not push points when
    the account is unwired or equity is zero (boot-time states). The
    curve stays empty under those conditions."""
    state = DashboardState(rules_path=tmp_path / "absent.json")
    state.health = HealthState(reconciliation_complete=True)

    # No account wired yet -> no snapshot.
    state.record_equity_snapshot()
    assert len(state.equity_curve) == 0

    # Account with zero equity -> still no snapshot.
    state.account = AccountState(equity_usdt=0.0)
    state.record_equity_snapshot()
    assert len(state.equity_curve) == 0

    # Healthy account -> snapshot pushed.
    state.account = AccountState(equity_usdt=500.0)
    state.record_equity_snapshot()
    assert len(state.equity_curve) == 1


@pytest.mark.asyncio
async def test_dashboard_external_flows_endpoint(tmp_path: Path) -> None:
    """``/api/external-flows`` exposes recent deposit/withdrawal events
    captured by the WithdrawalDetector wiring."""
    state = DashboardState(rules_path=tmp_path / "absent.json")
    state.health = HealthState(reconciliation_complete=True)

    state.push_external_flow({
        "ts": 1234.0, "kind": "withdrawal_detected",
        "delta_usdt": -5_000.0, "venue_balance": 5_000.0,
        "notes": "operator wire to bank",
    })
    state.push_external_flow({
        "ts": 1300.0, "kind": "deposit_detected",
        "delta_usdt": 2_000.0, "venue_balance": 7_000.0,
    })

    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/api/external-flows")
        body = await r.json()
        assert len(body) == 2
        kinds = {x["kind"] for x in body}
        assert kinds == {"withdrawal_detected", "deposit_detected"}


@pytest.mark.asyncio
async def test_dashboard_html_includes_chart_assets(tmp_path: Path) -> None:
    """The page must include the Chart.js CDN reference and the
    canvas element so the PnL curve renders."""
    state = DashboardState(rules_path=tmp_path / "absent.json")
    state.health = HealthState(reconciliation_complete=True)
    app = web.Application()
    install_dashboard(app, state, mode_label="DRY-RUN")
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/dashboard")
        text = await r.text()
        assert 'id="pnlchart"' in text
        assert "chart.js" in text.lower() or "Chart" in text
        assert "/api/pnl-curve" in text
        assert "/api/external-flows" in text
