"""P0 unified-fix invariants — TICKET-001 / 002 / 003 / 004 / 005 (+015).

This file pins the contracts that the audit demanded. Each section
maps 1:1 to a ticket so a future regression can be triaged in seconds.

  * TICKET-001 + 015 — clientOrderId idempotency end-to-end
  * TICKET-002       — _normalize_order exposes full venue truth
  * TICKET-003       — persistence watchdog + corrupt-fail-close + LIVE assertion
  * TICKET-004       — real fill price via fetch_my_trades + close-reason routing
  * TICKET-005       — RetryPolicy classification, retry budget, Retry-After

The tests are deliberately narrow: each one names ONE invariant. Don't
collapse them — when a P0 contract is silently flipped on by accident,
the failure name is what tells the operator which guardrail tripped.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.main import App, AppConfig, DryRunExchangeAdapter
from altcoin_agent.risk import (
    AccountPersistor,
    AccountState,
    CCXTExchangeAdapter,
    CCXTExecutor,
    Position,
    PositionLeg,
    PositionSizer,
    PositionWatcher,
    RiskGate,
    RiskGateConfig,
    Side,
)
from altcoin_agent.risk.ccxt_adapter import _CID_WRITE_PARAM
from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.retry import RetryPolicy, classify, parse_retry_after


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


def _account() -> AccountState:
    a = AccountState(
        equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0,
    )
    a.reconciliation_complete = True
    return a


def _decision(side: Side = Side.LONG) -> RiskDecision:
    return RiskDecision(
        approved=True, reason="ok", side=side, leverage=5.0, size=10.0,
        notional_usdt=10.0, risk_amount_usdt=0.5, initial_stop=0.95,
        max_slippage_used=0.03,
    )


class _FakeCCXT:
    """In-memory fake ccxt client; records every call so tests can
    assert on the params the adapter actually sent."""

    def __init__(
        self, *,
        market_status: str = "closed",
        market_filled_ratio: float = 1.0,
        raise_on_market: Exception | None = None,
        raise_on_create: Exception | None = None,
    ) -> None:
        self.market_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self.leverage_calls: list[dict] = []
        self.fetch_order_calls: list[dict] = []
        self.fetch_trade_calls: list[dict] = []
        self.market_status = market_status
        self.market_filled_ratio = market_filled_ratio
        self.raise_on_market = raise_on_market
        self.raise_on_create = raise_on_create
        self._n = 0
        self._fetch_order_responses: dict[str, dict[str, Any] | None] = {}
        self._fetch_trade_responses: dict[
            str, list[dict[str, Any]]
        ] = {}

    def _id(self) -> str:
        self._n += 1
        return f"id-{self._n}"

    async def create_market_order(
        self, symbol: str, side: str, amount: float, *,
        params: dict | None = None,
    ) -> dict:
        self.market_calls.append({
            "symbol": symbol, "side": side, "amount": amount,
            "params": params or {},
        })
        if self.raise_on_market is not None:
            exc = self.raise_on_market
            self.raise_on_market = None
            raise exc
        return {
            "id": self._id(),
            "symbol": symbol, "side": side,
            "amount": amount,
            "filled": amount * self.market_filled_ratio,
            "remaining": amount * (1.0 - self.market_filled_ratio),
            "average": 1.0, "price": 1.0,
            "status": self.market_status,
            "info": {
                "newClientOrderId": (params or {}).get("newClientOrderId"),
                "clOrdId": (params or {}).get("clOrdId"),
                "text": (params or {}).get("text"),
                "clientOrderId": (params or {}).get("clientOrderId"),
            },
            "clientOrderId": (params or {}).get("clientOrderId"),
        }

    async def create_order(
        self, symbol: str, type: str, side: str,  # noqa: A002
        amount: float, *, price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        self.create_calls.append({
            "symbol": symbol, "type": type, "side": side,
            "amount": amount, "price": price, "params": params or {},
        })
        if self.raise_on_create is not None:
            exc = self.raise_on_create
            self.raise_on_create = None
            raise exc
        return {
            "id": self._id(), "symbol": symbol, "side": side,
            "amount": amount, "filled": 0.0, "remaining": amount,
            "status": "open",
            "stopPrice": (params or {}).get("stopPrice"),
            "info": {
                "reduceOnly": (params or {}).get("reduceOnly", False),
                "clientOrderId": (params or {}).get("clientOrderId"),
                "newClientOrderId": (params or {}).get("newClientOrderId"),
                "clOrdId": (params or {}).get("clOrdId"),
                "text": (params or {}).get("text"),
            },
            "clientOrderId": (params or {}).get("clientOrderId"),
        }

    async def cancel_order(
        self, id: str, symbol: str | None = None,  # noqa: A002
        params: dict | None = None,
    ) -> dict:
        self.cancel_calls.append({"id": id, "symbol": symbol})
        return {"id": id, "status": "cancelled"}

    async def set_leverage(
        self, leverage: float, symbol: str,
        params: dict | None = None,
    ) -> dict:
        self.leverage_calls.append({
            "leverage": leverage, "symbol": symbol, "params": params or {},
        })
        return {"leverage": leverage, "symbol": symbol}

    async def fetch_positions(self, *args, **kwargs):  # noqa: ANN001
        return []

    async def fetch_open_orders(self, *args, **kwargs):  # noqa: ANN001
        return []

    async def fetch_order(
        self, id: str, symbol: str | None = None,  # noqa: A002
        params: dict | None = None,
    ) -> dict:
        params = params or {}
        cid = (
            params.get("clientOrderId")
            or params.get("origClientOrderId")
            or params.get("clOrdId")
            or params.get("text")
            or id
        )
        self.fetch_order_calls.append({
            "id": id, "symbol": symbol, "params": params, "cid_query": cid,
        })
        resp = self._fetch_order_responses.get(cid)
        if resp is None:
            # Default: simulate "OrderNotFound" so the retry path
            # continues with a fresh attempt.
            from ccxt.base.errors import OrderNotFound
            raise OrderNotFound(f"no order with cid {cid}")
        return resp

    async def fetch_my_trades(
        self, symbol: str, since: int | None = None,
        limit: int | None = None, params: dict | None = None,
    ) -> list[dict]:
        self.fetch_trade_calls.append({
            "symbol": symbol, "since": since, "limit": limit,
            "params": params or {},
        })
        return list(self._fetch_trade_responses.get(symbol, []))

    # test helpers
    def pin_fetch_order(self, cid: str, resp: dict | None) -> None:
        self._fetch_order_responses[cid] = resp

    def pin_my_trades(self, symbol: str, trades: list[dict]) -> None:
        self._fetch_trade_responses[symbol] = trades


# ===================================================================== #
# TICKET-001 + 015 — clientOrderId idempotency
# ===================================================================== #


@pytest.mark.asyncio
async def test_ticket001_executor_open_threads_cid_through_to_venue() -> None:
    """The executor MUST generate a cid and inject it into both the
    market entry and the resting STOP_MARKET. The cid is venue-safe
    (26 chars, letter-prefixed, alphanumeric)."""
    cli = _FakeCCXT()
    adapter = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    ex = CCXTExecutor(adapter=adapter, exchange_name="binance")
    pos = await ex.open(
        symbol="RAVEUSDT", decision=_decision(),
        current_price=1.0, account=_account(),
    )
    # Position carries cids.
    assert pos.client_order_id and pos.client_order_id.startswith("alt")
    assert pos.stop_client_order_id and pos.stop_client_order_id.startswith("alt")
    assert pos.client_order_id != pos.stop_client_order_id
    # Length / charset invariants.
    assert len(pos.client_order_id) == 26
    assert pos.client_order_id.isalnum()
    assert pos.client_order_id[0].isalpha()
    # The market order received the cid in Binance's slot.
    assert len(cli.market_calls) == 1
    market_params = cli.market_calls[0]["params"]
    assert market_params["newClientOrderId"] == pos.client_order_id
    # The stop order received its (different) cid.
    assert len(cli.create_calls) == 1
    stop_params = cli.create_calls[0]["params"]
    assert stop_params["newClientOrderId"] == pos.stop_client_order_id
    assert stop_params["newClientOrderId"] != market_params["newClientOrderId"]


@pytest.mark.asyncio
async def test_ticket001_okx_uses_clOrdId_param() -> None:
    cli = _FakeCCXT()
    adapter = CCXTExchangeAdapter(client=cli, exchange_name="okx")
    ex = CCXTExecutor(adapter=adapter, exchange_name="okx")
    await ex.open(
        symbol="RAVE-USDT-SWAP", decision=_decision(),
        current_price=1.0, account=_account(),
    )
    assert "clOrdId" in cli.market_calls[0]["params"]
    assert "clOrdId" in cli.create_calls[0]["params"]
    # OKX cid is the unprefixed cid.
    cid = cli.market_calls[0]["params"]["clOrdId"]
    assert cid.startswith("alt")


@pytest.mark.asyncio
async def test_ticket001_gateio_text_field_carries_t_prefix() -> None:
    """Gate.io's ``text`` field requires the literal "t-" prefix."""
    cli = _FakeCCXT()
    adapter = CCXTExchangeAdapter(client=cli, exchange_name="gateio")
    ex = CCXTExecutor(adapter=adapter, exchange_name="gateio")
    pos = await ex.open(
        symbol="RAVE_USDT", decision=_decision(),
        current_price=1.0, account=_account(),
    )
    text = cli.market_calls[0]["params"]["text"]
    assert text.startswith("t-")
    # The 28-char body limit (excluding "t-") is honoured.
    assert len(text) <= 30
    assert pos.client_order_id is not None


def test_ticket001_cid_param_table_covers_supported_venues() -> None:
    """Sanity — the per-venue cid map should at least cover the venues
    the daemon currently supports out of the box."""
    for venue in ("binance", "okx", "gateio"):
        assert venue in _CID_WRITE_PARAM


@pytest.mark.asyncio
async def test_ticket001_retry_uses_idempotency_check_to_avoid_dupes() -> None:
    """When the first ``create_market_order`` raises NetworkError, the
    adapter MUST consult ``fetch_order(client_order_id=...)`` before
    retrying. If the prior call did land, the cached order is used and
    no second venue write happens."""
    from ccxt.base.errors import NetworkError
    cli = _FakeCCXT(raise_on_market=NetworkError("transient"))
    # Pre-pin a fetch_order response: the prior write actually landed
    # on the venue even though we got a NetworkError back.
    # We don't know the cid until the executor generates it, so use a
    # custom adapter that exposes the cid for this test.
    captured: dict[str, str] = {}

    class _CapturingAdapter(CCXTExchangeAdapter):
        async def market_order(self, *args, **kwargs):  # type: ignore[override]
            captured["cid"] = kwargs.get("client_order_id", "")
            # Pin a "yes the prior write succeeded" response under the
            # cid that's about to be generated, BEFORE the retry runs.
            cli.pin_fetch_order(captured["cid"], {
                "id": "venue-id-9", "clientOrderId": captured["cid"],
                "amount": 10.0, "filled": 10.0, "remaining": 0.0,
                "status": "closed", "average": 1.0, "price": 1.0,
                "info": {},
                "side": "buy", "symbol": "RAVEUSDT",
            })
            return await super().market_order(*args, **kwargs)

    adapter = _CapturingAdapter(
        client=cli, exchange_name="binance",
        retry_policy=RetryPolicy(
            base_backoff_sec=0.0, max_backoff_sec=0.0,
            jitter_sec=0.0, sleeper=lambda _t: asyncio.sleep(0),
        ),
    )
    ex = CCXTExecutor(adapter=adapter, exchange_name="binance")
    pos = await ex.open(
        symbol="RAVEUSDT", decision=_decision(),
        current_price=1.0, account=_account(),
    )
    # The order DID land — fetch_order short-circuited the retry,
    # so create_market_order was called exactly once.
    assert len(cli.market_calls) == 1
    # Position got the venue id from the resolved fetch_order.
    assert pos.client_order_id == captured["cid"]


def test_ticket015_position_cid_round_trips_through_persistor(
    tmp_path: Path,
) -> None:
    """Persisted Position must roundtrip its cids so the Reconciler
    can re-attach the venue order on restart."""
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = _account()
    pos = Position(
        symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
        entry_price=1.0, size=10.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95,
        stop_order_id="stop-99",
        client_order_id="alt0000000000000000000001",
        stop_client_order_id="alt0000000000000000000002",
    )
    pos.legs.append(PositionLeg(
        leg_id=0, side=Side.LONG, size=10.0, entry_price=1.0,
        margin_source="initial",
        client_order_id="alt0000000000000000000001",
    ))
    a.open_positions[pos.symbol] = pos
    assert p.save(a) is True

    b = _account()
    assert p.restore_into(b) is True
    restored = b.open_positions["RAVEUSDT"]
    assert restored.client_order_id == "alt0000000000000000000001"
    assert restored.stop_client_order_id == "alt0000000000000000000002"
    assert restored.legs[0].client_order_id == "alt0000000000000000000001"
    assert restored.stop_order_id == "stop-99"


# ===================================================================== #
# TICKET-002 — _normalize_order exposes full venue truth
# ===================================================================== #


def test_ticket002_normalize_order_exposes_amount_filled_remaining() -> None:
    """Pre-fix, only ``size`` (= max(amount, filled)) was exposed and
    the executor's partial-fill detection was a no-op. Post-fix the
    full schema is present."""
    raw = {
        "id": "x", "symbol": "RAVEUSDT", "side": "buy",
        "amount": 100.0, "filled": 40.0, "remaining": 60.0,
        "price": 1.5, "average": 1.5, "status": "open",
    }
    out = CCXTExchangeAdapter._normalize_order(raw)
    assert out["amount"] == pytest.approx(100.0)
    assert out["filled"] == pytest.approx(40.0)
    assert out["remaining"] == pytest.approx(60.0)
    assert out["status"] == "open"
    # legacy back-compat key still present
    assert out["size"] == pytest.approx(100.0)


def test_ticket002_normalize_order_extracts_client_order_id() -> None:
    """cid round-trips from any of the venue-specific slots."""
    cases = [
        {"clientOrderId": "alt-binance"},
        {"info": {"newClientOrderId": "alt-binance-info"}},
        {"info": {"clOrdId": "alt-okx"}},
        {"info": {"text": "t-altgate"}},
        {"info": {"orderLinkId": "alt-bybit"}},
    ]
    for raw in cases:
        out = CCXTExchangeAdapter._normalize_order(
            {"id": "x", "amount": 1.0, **raw},
        )
        assert out["client_order_id"] is not None, raw


@pytest.mark.asyncio
async def test_ticket002_partial_fill_via_real_adapter_path() -> None:
    """Pre-fix: the safety-hardening tests only caught partial fills
    when the test mocked the adapter directly (bypassing
    ``_normalize_order``). With the real adapter+normalize chain the
    partial-fill emergency-close MUST still fire. This is the
    regression test that would have caught TICKET-002 in production."""
    cli = _FakeCCXT(market_filled_ratio=0.4)  # 40% fill, below 95%
    adapter = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    ex = CCXTExecutor(adapter=adapter, exchange_name="binance")
    a = _account()
    with pytest.raises(Exception) as exc_info:
        await ex.open(
            symbol="RAVEUSDT", decision=_decision(),
            current_price=1.0, account=a,
        )
    assert "partial_fill_below_threshold" in str(exc_info.value)
    # Two market orders: entry, then reduce_only emergency close on
    # the partial leg. The reduce_only one is sized to the ACTUAL fill
    # (4.0), not the requested 10.0.
    assert len(cli.market_calls) == 2
    emergency = cli.market_calls[1]
    assert emergency["params"]["reduceOnly"] is True
    assert emergency["amount"] == pytest.approx(4.0)
    # Symbol cooldown engaged.
    assert "RAVEUSDT" in a.cooldown_until_ts_ms
    # Position not registered locally — the entry failed.
    assert "RAVEUSDT" not in a.open_positions


@pytest.mark.asyncio
async def test_ticket002_canceled_status_aborts_open() -> None:
    """A market order that comes back ``status=canceled`` is not
    a fill. Pre-fix this would have been treated as success."""
    cli = _FakeCCXT(market_status="canceled", market_filled_ratio=0.0)
    adapter = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    ex = CCXTExecutor(adapter=adapter, exchange_name="binance")
    a = _account()
    with pytest.raises(Exception) as exc_info:
        await ex.open(
            symbol="RAVEUSDT", decision=_decision(),
            current_price=1.0, account=a,
        )
    assert "entry_not_filled:canceled" in str(exc_info.value)


# ===================================================================== #
# TICKET-003 — persistence watchdog + corrupt-fail-close + LIVE assertion
# ===================================================================== #


def test_ticket003_consecutive_save_failures_halt_account(
    tmp_path: Path,
) -> None:
    """N consecutive save failures -> ``account.halt(
    'persistence_unavailable')``. The threshold is the persistor's
    ``max_consecutive_save_failures`` (default 3)."""
    p = AccountPersistor(
        path=tmp_path / "acc.json", max_consecutive_save_failures=3,
    )
    a = _account()

    # Force ``save`` to fail by stubbing ``to_dict`` to raise. We use
    # an exception inside ``to_dict`` rather than an unserialisable
    # value because the persistor calls ``json.dumps(..., default=str)``
    # which serialises arbitrary objects via repr -- the only reliable
    # way to force a failure is to make the snapshot construction
    # itself raise.
    def _bad_to_dict(_a):
        raise RuntimeError("disk full simulation")
    p.to_dict = _bad_to_dict  # type: ignore[method-assign]
    assert p.save(a) is False
    assert p.save(a) is False
    assert a.global_trading_halted is False  # not yet
    assert p.save(a) is False
    assert a.global_trading_halted is True
    assert a.halt_reason == "persistence_unavailable"


def test_ticket003_save_failure_counter_resets_on_success(
    tmp_path: Path,
) -> None:
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = _account()
    # Two failures, then a success.
    def _raise(_a):
        raise RuntimeError("disk full")
    p.to_dict = _raise  # type: ignore[method-assign]
    assert p.save(a) is False
    assert p.save(a) is False
    assert p.consecutive_save_failures == 2
    # Restore real to_dict — next save succeeds.
    p.to_dict = AccountPersistor.to_dict.__get__(p, AccountPersistor)
    assert p.save(a) is True
    assert p.consecutive_save_failures == 0


def test_ticket003_corrupt_restore_halts_and_returns_false(
    tmp_path: Path,
) -> None:
    """Pre-fix, a corrupt JSON snapshot was logged-and-ignored; the
    daemon proceeded with default zeros, silently re-arming the daily
    DD breaker. Post-fix the account is halted and a sticky corrupt
    flag is set so ``main.App.run`` aborts boot."""
    path = tmp_path / "acc.json"
    path.write_text("{not json")
    p = AccountPersistor(path=path)
    a = _account()
    restored = p.restore_into(a)
    assert restored is False
    assert a.account_state_corrupt is True
    assert a.global_trading_halted is True
    assert a.halt_reason == "account_state_corrupt"


def test_ticket003_clean_missing_file_does_not_halt(tmp_path: Path) -> None:
    """An absent snapshot file is "clean first boot", NOT corruption.
    Don't halt in that case."""
    p = AccountPersistor(path=tmp_path / "never_existed.json")
    a = _account()
    assert p.restore_into(a) is False
    assert a.account_state_corrupt is False
    assert a.global_trading_halted is False


def test_ticket003_empty_file_does_not_halt(tmp_path: Path) -> None:
    """Empty file == zero-byte, treat same as missing."""
    path = tmp_path / "acc.json"
    path.write_text("")
    p = AccountPersistor(path=path)
    a = _account()
    assert p.restore_into(a) is False
    assert a.account_state_corrupt is False


def test_ticket003_higher_schema_halts() -> None:
    """A snapshot from a future build (schema_version > current)
    must NOT be auto-loaded — that's a downgrade-cum-data-loss
    scenario."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "acc.json"
        path.write_text(json.dumps({
            "schema_version": 999,
            "equity_usdt": 1.0,
        }))
        p = AccountPersistor(path=path)
        a = _account()
        assert p.restore_into(a) is False
        assert a.account_state_corrupt is True
        assert a.global_trading_halted is True


@pytest.mark.asyncio
async def test_ticket003_live_mode_requires_persistence() -> None:
    """``App.run`` MUST refuse to start in non-dry-run mode when
    persistence is disabled. Pre-fix this was silent."""
    cfg = AppConfig(
        healthz_port=18900, dry_run=False, paper_trade=True,
        graceful_timeout_sec=1.0,
        account_persistence_enabled=False,
    )
    app = App(cfg=cfg)
    with pytest.raises(SystemExit) as ei:
        await app.run()
    assert ei.value.code == 4


# ===================================================================== #
# TICKET-004 — real fill price via fetch_my_trades + close-reason
# ===================================================================== #


def _running_app_factory(cfg: AppConfig):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx():
        app = App(cfg=cfg)
        runner = asyncio.create_task(app.run())
        for _ in range(80):
            if app._screener is not None:
                break
            await asyncio.sleep(0.02)
        assert app._screener is not None

        async def _noop_run() -> None:
            await app._stop_event.wait()
        app._screener.run = _noop_run     # type: ignore[method-assign]
        try:
            for _ in range(80):
                if app.state.reconciliation_complete:
                    break
                await asyncio.sleep(0.02)
            yield app
        finally:
            app.request_stop()
            await asyncio.wait_for(runner, timeout=5.0)
    return _ctx


@pytest.mark.asyncio
async def test_ticket004_fill_price_uses_fetch_my_trades_vwap() -> None:
    """A LONG closed at avg fill 0.92 (not the resting stop 0.95)
    must credit PnL based on the REAL fill, not the expected stop fill."""
    cfg = AppConfig(
        healthz_port=18901, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        position_watcher_poll_sec=0.05,
        position_watcher_miss_threshold=1,
    )
    async with _running_app_factory(cfg)() as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            TrailingStopFSM,
        )
        executor = CCXTExecutor(adapter=adapter)
        account = _account()
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )
        # A LONG position whose stop is at 0.95 but ACTUAL fills came
        # in at 0.93 and 0.91 (= VWAP 0.92, given equal qty). Pre-fix
        # the close handler would have used 0.95.
        pos = Position(
            symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=100.0, leverage=5.0,
            initial_stop=0.95, current_stop=0.95,
            stop_client_order_id="alt-stop-1",
        )
        pos.opened_at_ts_ms = 1_000_000
        adapter.set_fill_trades("RAVEUSDT", [
            {"timestamp": 1_500_000, "side": "sell", "amount": 50.0,
             "price": 0.93, "client_order_id": "alt-stop-1"},
            {"timestamp": 1_500_001, "side": "sell", "amount": 50.0,
             "price": 0.91, "client_order_id": "alt-stop-1"},
        ])
        account.open_positions[pos.symbol] = pos
        account.open_positions.pop(pos.symbol)
        pos.closed = True

        await app._on_position_close(
            position=pos, reason="exchange_close_detected",
            trailing=trailing, account=account,
        )
        # PnL = (0.92 - 1.00) * 100 = -8.00.
        assert account.realized_pnl_today_usdt == pytest.approx(-8.0, rel=1e-6)


@pytest.mark.asyncio
async def test_ticket004_fill_price_falls_back_to_current_stop() -> None:
    """When the adapter has no trades for the symbol the close handler
    must fall back to the legacy ``current_stop`` proxy. Documented
    degradation, not a silent failure."""
    cfg = AppConfig(
        healthz_port=18902, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
    )
    async with _running_app_factory(cfg)() as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)
        from altcoin_agent.main import TrailingController
        from altcoin_agent.risk import (
            ATRCalculator,
            CCXTExecutor,
            TrailingStopFSM,
        )
        executor = CCXTExecutor(adapter=adapter)
        account = _account()
        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=ATRCalculator(),
            executor=executor, account=account, health=app.state,
        )
        pos = Position(
            symbol="ZZZUSDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=100.0, leverage=5.0,
            initial_stop=0.95, current_stop=0.95,
        )
        # No fill_trades pinned -> adapter returns []
        pos.closed = True
        await app._on_position_close(
            position=pos, reason="exchange_close_detected",
            trailing=trailing, account=account,
        )
        # PnL = (0.95 - 1.00) * 100 = -5.00 (legacy fallback)
        assert account.realized_pnl_today_usdt == pytest.approx(-5.0, rel=1e-6)


@pytest.mark.asyncio
async def test_ticket004_close_reason_hint_routes_through_watcher() -> None:
    """Executor's ``on_emergency_close`` MUST plumb a hint to the
    watcher so the next close on the symbol gets the hinted reason
    instead of the generic default."""
    cfg = AppConfig(
        healthz_port=18903, dry_run=True, graceful_timeout_sec=2.0,
        initial_equity_usdt=10_000.0, min_liquidity_usdt=100_000.0,
        position_watcher_poll_sec=0.05,
        position_watcher_miss_threshold=1,
    )
    async with _running_app_factory(cfg)() as app:
        adapter = app._adapter
        assert isinstance(adapter, DryRunExchangeAdapter)

        seen_reasons: list[str] = []

        # Build a watcher with a recording on_close callback.
        account = _account()
        adapter._open_positions["RAVEUSDT"] = {
            "symbol": "RAVEUSDT", "side": "long", "contracts": 1.0,
            "entryPrice": 1.0,
        }
        pos = Position(
            symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=1.0, leverage=5.0,
            initial_stop=0.95, current_stop=0.95,
        )
        account.open_positions[pos.symbol] = pos

        async def _on_close(p: Position, reason: str) -> None:
            seen_reasons.append(reason)

        watcher = PositionWatcher(
            adapter=adapter, account=account, on_close=_on_close,
            poll_interval_sec=0.01, miss_threshold=1,
        )
        # Hint a custom reason BEFORE the close fires.
        watcher.hint_close_reason("RAVEUSDT", "emergency_close_partial_fill")
        adapter.simulate_close("RAVEUSDT")
        await watcher.poll_once()
        assert seen_reasons == ["emergency_close_partial_fill"]
        # Hint is single-use: a second close on the same symbol falls
        # back to the default.
        adapter._open_positions["RAVEUSDT"] = {
            "symbol": "RAVEUSDT", "side": "long", "contracts": 1.0,
            "entryPrice": 1.0,
        }
        # Re-attach a fresh position, then simulate close again.
        account.open_positions["RAVEUSDT"] = Position(
            symbol="RAVEUSDT", exchange="binance", side=Side.LONG,
            entry_price=1.0, size=1.0, leverage=5.0,
            initial_stop=0.95, current_stop=0.95,
        )
        adapter.simulate_close("RAVEUSDT")
        await watcher.poll_once()
        assert seen_reasons[-1] == "exchange_close_detected"


# ===================================================================== #
# TICKET-005 — RetryPolicy classification + budget + Retry-After
# ===================================================================== #


def test_ticket005_classify_known_exception_names() -> None:
    """Sanity-check the classification table. We use the string-name
    fallback so the test runs even on machines without ccxt installed
    in the test path."""
    class NetworkError(Exception): pass
    class RateLimitExceeded(Exception): pass
    class InvalidOrder(Exception): pass
    class ExchangeError(Exception): pass
    class _Random(Exception): pass

    assert classify(NetworkError("x")) == "transient"
    assert classify(RateLimitExceeded("x")) == "transient"
    assert classify(InvalidOrder("x")) == "fatal"
    assert classify(ExchangeError("x")) == "one_shot"
    assert classify(_Random("x")) == "one_shot"


def test_ticket005_parse_retry_after_message() -> None:
    """RateLimitExceeded with parseable Retry-After hint."""
    class RateLimitExceeded(Exception): pass
    assert parse_retry_after(RateLimitExceeded(
        "rate limit exceeded — retry after 2.5s",
    )) == pytest.approx(2.5)
    assert parse_retry_after(RateLimitExceeded(
        "retry-after: 1500ms",
    )) == pytest.approx(1.5)
    assert parse_retry_after(RateLimitExceeded("nothing useful")) is None


@pytest.mark.asyncio
async def test_ticket005_transient_retries_then_succeeds() -> None:
    """A transient error followed by success returns the success
    value; the sleeper is called once."""
    class NetworkError(Exception): pass
    sleeps: list[float] = []

    async def _sleep(t: float) -> None:
        sleeps.append(t)

    policy = RetryPolicy(
        base_backoff_sec=0.1, max_backoff_sec=1.0, jitter_sec=0.0,
        sleeper=_sleep,
    )
    n = {"calls": 0}

    async def _factory():
        n["calls"] += 1
        if n["calls"] == 1:
            raise NetworkError("blip")
        return "ok"

    out = await policy.execute(_factory, op_name="test")
    assert out == "ok"
    assert n["calls"] == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_ticket005_fatal_re_raises_immediately() -> None:
    class InvalidOrder(Exception): pass
    sleeps: list[float] = []

    policy = RetryPolicy(
        base_backoff_sec=0.0, max_backoff_sec=0.0, jitter_sec=0.0,
        sleeper=lambda t: sleeps.append(t) or asyncio.sleep(0),
    )
    n = {"calls": 0}

    async def _factory():
        n["calls"] += 1
        raise InvalidOrder("price below tick size")

    with pytest.raises(InvalidOrder):
        await policy.execute(_factory, op_name="test")
    assert n["calls"] == 1     # NO retry
    assert sleeps == []        # NO sleep


@pytest.mark.asyncio
async def test_ticket005_transient_exhausts_after_max_attempts() -> None:
    class NetworkError(Exception): pass
    policy = RetryPolicy(
        max_attempts=3,
        base_backoff_sec=0.0, max_backoff_sec=0.0, jitter_sec=0.0,
        sleeper=lambda _t: asyncio.sleep(0),
    )
    n = {"calls": 0}

    async def _factory():
        n["calls"] += 1
        raise NetworkError("permanent")

    with pytest.raises(NetworkError):
        await policy.execute(_factory, op_name="test")
    assert n["calls"] == 3      # initial + 2 retries == max_attempts


@pytest.mark.asyncio
async def test_ticket005_one_shot_retries_exactly_once() -> None:
    class ExchangeError(Exception): pass
    n = {"calls": 0}

    async def _factory():
        n["calls"] += 1
        raise ExchangeError("oops")

    policy = RetryPolicy(
        base_backoff_sec=0.0, max_backoff_sec=0.0, jitter_sec=0.0,
        sleeper=lambda _t: asyncio.sleep(0),
    )
    with pytest.raises(ExchangeError):
        await policy.execute(_factory, op_name="test")
    assert n["calls"] == 2     # initial + 1 retry, then give up


@pytest.mark.asyncio
async def test_ticket005_idempotency_check_short_circuits_retry() -> None:
    """When the idempotency check returns a non-None value, that value
    is returned in lieu of the retry — the venue already accepted the
    prior write and re-issuing would duplicate."""
    class NetworkError(Exception): pass
    policy = RetryPolicy(
        base_backoff_sec=0.0, max_backoff_sec=0.0, jitter_sec=0.0,
        sleeper=lambda _t: asyncio.sleep(0),
    )
    n = {"calls": 0}

    async def _factory():
        n["calls"] += 1
        raise NetworkError("blip")

    async def _idem():
        return {"status": "closed", "id": "venue-id-99"}

    out = await policy.execute(
        _factory, op_name="test", idempotency_check=_idem,
    )
    assert out == {"status": "closed", "id": "venue-id-99"}
    # _factory was called exactly once — the retry was short-circuited.
    assert n["calls"] == 1


@pytest.mark.asyncio
async def test_ticket005_adapter_market_order_uses_retry_policy() -> None:
    """End-to-end at the adapter layer: a transient client error is
    retried; the second call succeeds and the caller never sees the
    NetworkError."""
    from ccxt.base.errors import NetworkError

    call_log: list[str] = []

    class _OneBlipClient(_FakeCCXT):
        def __init__(self) -> None:
            super().__init__()
            self._blipped = False

        async def create_market_order(self, *args, **kwargs):  # type: ignore[override]
            call_log.append("create_market_order")
            if not self._blipped:
                self._blipped = True
                raise NetworkError("transient")
            return await super().create_market_order(*args, **kwargs)

    cli = _OneBlipClient()
    policy = RetryPolicy(
        base_backoff_sec=0.0, max_backoff_sec=0.0, jitter_sec=0.0,
        sleeper=lambda _t: asyncio.sleep(0),
    )
    adapter = CCXTExchangeAdapter(
        client=cli, exchange_name="binance", retry_policy=policy,
    )
    out = await adapter.market_order(
        "RAVEUSDT", Side.LONG, 10.0, client_order_id="alttest1234",
    )
    # Both invocations were observed — the adapter masked the blip
    # by retrying. The second call landed; the first one raised before
    # appending to market_calls, hence only one entry there. The
    # call_log proves the retry actually fired.
    assert call_log == ["create_market_order", "create_market_order"]
    assert len(cli.market_calls) == 1
    assert out["client_order_id"] == "alttest1234"
    assert out["status"] == "closed"
