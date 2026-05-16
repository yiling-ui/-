"""Phase B.1 — P0 production hardening tests.

Covers the three deliverables from
``MISS_PENALTY_AND_PRODUCTION_PLAN.md`` Phase B.1:

  * **B.1.1 clientOrderId idempotency**:
        - executor generates a UUID for every market / stop / close
          order it places;
        - retries reuse the SAME UUID so the venue dedupes;
        - CCXTExchangeAdapter forwards the UUID via the right
          venue-specific param (``newClientOrderId`` on Binance,
          ``clOrdId`` on OKX, ``text`` with ``t-`` prefix on Gate).

  * **B.1.2 market-entry retry with ccxt error class differentiation**:
        - retriable failures (NetworkError-like) are retried with
          exponential backoff up to ``place_entry_retries`` times;
        - non-retriable failures (InsufficientFunds-like) short-
          circuit immediately;
        - the same ``client_order_id`` is sent on every retry.

  * **B.1.3 event-driven AccountState persistence**:
        - registering a change listener fires once per logical mutation
          (set_cooldown / halt / record_pnl / clear_consecutive_loss /
          maybe_roll_over_day actually flipping the day);
        - ``record_pnl`` matches the legacy inlined update from
          ``main._on_position_close`` byte-for-byte;
        - boot stamping (``maybe_roll_over_day`` first call) does NOT
          notify because nothing was reset.
"""

from __future__ import annotations

from typing import Any

import pytest

from altcoin_agent.risk.ccxt_adapter import CCXTExchangeAdapter
from altcoin_agent.risk.executor import (
    CCXTExecutor,
    ExecutionError,
    _build_ccxt_error_sets,
    _classify_error,
)
from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.state import AccountState, Side

# ====================================================================== #
# Shared fakes
# ====================================================================== #


class _RecordingAdapter:
    """Minimal ExchangeAdapter that records every call and supports
    scripted failures on ``market_order`` to exercise the retry path.

    Failures are popped off ``market_order_failures`` in order — once
    exhausted, the call succeeds. Each call records the kwargs it
    received so tests can assert on the ``client_order_id`` flow.
    """

    def __init__(self) -> None:
        self.market_calls: list[dict[str, Any]] = []
        self.market_order_failures: list[Exception] = []
        self.stop_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[tuple[str, str]] = []
        self.leverage_calls: list[tuple[str, float]] = []
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"o-{self._n}"

    async def market_order(
        self, symbol, side, size, *, price=None, reduce_only=False,
        client_order_id=None,
    ):  # noqa: ANN001
        self.market_calls.append({
            "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        if self.market_order_failures:
            err = self.market_order_failures.pop(0)
            raise err
        return {
            "id": self._id(), "symbol": symbol, "side": side.value,
            "size": size, "average": price or 1.0, "price": price or 1.0,
            "filled": size, "amount": size,
        }

    async def place_stop_order(
        self, symbol, side, size, stop_price, reduce_only=True,
        client_order_id=None,
    ):  # noqa: ANN001
        self.stop_calls.append({
            "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return {
            "id": self._id(), "symbol": symbol, "side": side.value,
            "size": size, "stop_price": stop_price,
            "reduce_only": reduce_only,
        }

    async def cancel_order(self, order_id, symbol):  # noqa: ANN001
        self.cancel_calls.append((order_id, symbol))
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):  # noqa: ANN001
        self.leverage_calls.append((symbol, leverage))
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self):
        return []


def _account() -> AccountState:
    a = AccountState(equity_usdt=10_000.0, starting_equity_today_usdt=10_000.0)
    a.reconciliation_complete = True
    return a


def _decision(size: float = 1.0) -> RiskDecision:
    return RiskDecision(
        approved=True, reason="test", side=Side.LONG, size=size,
        leverage=10.0, initial_stop=0.95,
    )


# ====================================================================== #
# B.1.1 — clientOrderId idempotency (executor side)
# ====================================================================== #


@pytest.mark.asyncio
async def test_executor_generates_unique_client_order_id_per_logical_order() -> None:
    """Every entry / stop / close gets its own client_order_id, and
    they're all distinct."""
    adapter = _RecordingAdapter()
    ex = CCXTExecutor(adapter=adapter, client_order_id_prefix="alt")
    a = _account()
    await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                  account=a, trace_id="t1")

    # entry market_order: 1 call, has a coid
    assert len(adapter.market_calls) == 1
    entry_coid = adapter.market_calls[0]["client_order_id"]
    assert entry_coid is not None
    assert entry_coid.startswith("alt-e-")  # entry "kind" letter

    # stop placement: 1 call, has a different coid
    assert len(adapter.stop_calls) == 1
    stop_coid = adapter.stop_calls[0]["client_order_id"]
    assert stop_coid is not None
    assert stop_coid.startswith("alt-s-")
    assert stop_coid != entry_coid


@pytest.mark.asyncio
async def test_executor_reuses_client_order_id_across_retries() -> None:
    """B.1.1 + B.1.2 invariant: when the entry market_order fails with
    a retriable error and we retry, the SAME UUID is sent again so the
    venue can dedupe a successfully-submitted-but-lost-response order.
    """
    adapter = _RecordingAdapter()
    # First two attempts fail with a retriable network-style error,
    # third one succeeds.
    adapter.market_order_failures = [
        RuntimeError("connection timeout"),
        RuntimeError("503 service unavailable"),
    ]
    ex = CCXTExecutor(
        adapter=adapter,
        place_entry_retries=2,
        entry_retry_base_delay_sec=0.001,  # keep test fast
    )
    a = _account()
    await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                  account=a, trace_id="t1")

    # 3 entry attempts total (1 initial + 2 retries).
    assert len(adapter.market_calls) == 3
    coids = [c["client_order_id"] for c in adapter.market_calls]
    # All three calls used the SAME idempotency token.
    assert len(set(coids)) == 1
    assert coids[0] is not None and coids[0].startswith("alt-e-")


@pytest.mark.asyncio
async def test_executor_short_circuits_on_non_retriable_error() -> None:
    """InsufficientFunds-style errors are not retried — we fail fast."""
    adapter = _RecordingAdapter()
    adapter.market_order_failures = [
        RuntimeError("insufficient funds for order"),
        RuntimeError("would-have-succeeded-on-retry"),  # never reached
    ]
    ex = CCXTExecutor(adapter=adapter, place_entry_retries=3,
                      entry_retry_base_delay_sec=0.001)
    a = _account()
    with pytest.raises(RuntimeError, match="insufficient"):
        await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                      account=a, trace_id="t1")
    # Only 1 call: short-circuit on first non-retriable error.
    assert len(adapter.market_calls) == 1


@pytest.mark.asyncio
async def test_executor_retries_exhausted_raises_last_error() -> None:
    """All retries failing means we re-raise the last exception."""
    adapter = _RecordingAdapter()
    adapter.market_order_failures = [
        RuntimeError("timeout 1"),
        RuntimeError("timeout 2"),
        RuntimeError("timeout 3"),
    ]
    ex = CCXTExecutor(adapter=adapter, place_entry_retries=2,
                      entry_retry_base_delay_sec=0.001)
    a = _account()
    with pytest.raises(RuntimeError, match="timeout 3"):
        await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                      account=a, trace_id="t1")
    # 3 attempts (1 initial + 2 retries), all failed.
    assert len(adapter.market_calls) == 3


@pytest.mark.asyncio
async def test_executor_emergency_close_uses_fresh_client_order_id() -> None:
    """When stop placement fails after retries, the executor emergency-
    closes the entry. That close gets its OWN client_order_id, distinct
    from the entry's, so a future retry of the close (e.g. a separate
    daemon path) doesn't double-submit it.
    """
    adapter = _RecordingAdapter()

    # Override place_stop_order to always fail.
    async def always_fail_stop(*args, **kwargs):
        adapter.stop_calls.append(kwargs)
        raise RuntimeError("simulated stop failure")
    adapter.place_stop_order = always_fail_stop  # type: ignore[assignment]

    ex = CCXTExecutor(
        adapter=adapter, place_stop_retries=1,
        stop_retry_base_delay_sec=0.001,
    )
    a = _account()
    with pytest.raises(ExecutionError, match="stop_placement_failed"):
        await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                      account=a, trace_id="t1")

    # Entry call(s) + emergency close call; close coid must differ.
    coids = [c["client_order_id"] for c in adapter.market_calls]
    assert len(coids) >= 2
    entry_coid, close_coid = coids[0], coids[-1]
    assert entry_coid != close_coid
    assert entry_coid.startswith("alt-e-")
    assert close_coid.startswith("alt-c-")


# ====================================================================== #
# B.1.1 — clientOrderId forwarding (CCXTExchangeAdapter side)
# ====================================================================== #


class _FakeCCXTClient:
    """Captures ``params`` so we can assert which idempotency key was used."""

    def __init__(self) -> None:
        self.market_calls: list[dict[str, Any]] = []
        self.order_calls: list[dict[str, Any]] = []

    async def create_market_order(self, symbol, side, amount, *, params=None):  # noqa: ANN001
        self.market_calls.append({"symbol": symbol, "side": side,
                                   "amount": amount, "params": dict(params or {})})
        return {"id": "x", "average": 1.0, "filled": amount}

    async def create_order(self, symbol, type, side, amount, *,  # noqa: A002
                            price=None, params=None):  # noqa: ANN001
        self.order_calls.append({"symbol": symbol, "type": type, "side": side,
                                   "amount": amount, "price": price,
                                   "params": dict(params or {})})
        return {"id": "y", "stopPrice": (params or {}).get("stopPrice")}


@pytest.mark.asyncio
async def test_ccxt_adapter_binance_uses_newClientOrderId() -> None:
    cli = _FakeCCXTClient()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.market_order("PEPE", Side.LONG, 1.0, client_order_id="alt-e-abc")
    p = cli.market_calls[0]["params"]
    assert p.get("newClientOrderId") == "alt-e-abc"


@pytest.mark.asyncio
async def test_ccxt_adapter_okx_uses_clOrdId() -> None:
    cli = _FakeCCXTClient()
    a = CCXTExchangeAdapter(client=cli, exchange_name="okx")
    await a.market_order("PEPE", Side.LONG, 1.0, client_order_id="alt-e-abc")
    p = cli.market_calls[0]["params"]
    assert p.get("clOrdId") == "alt-e-abc"
    assert p.get("newClientOrderId") is None


@pytest.mark.asyncio
async def test_ccxt_adapter_gate_prefixes_with_t_dash() -> None:
    cli = _FakeCCXTClient()
    a = CCXTExchangeAdapter(client=cli, exchange_name="gateio")
    await a.market_order("PEPE", Side.LONG, 1.0, client_order_id="alt-e-abc")
    p = cli.market_calls[0]["params"]
    # Gate.io requires user-supplied IDs to start with "t-".
    assert p.get("text", "").startswith("t-")


@pytest.mark.asyncio
async def test_ccxt_adapter_no_client_order_id_omits_param() -> None:
    """Backward-compat: legacy callers (no ``client_order_id`` kwarg)
    must not see ``newClientOrderId`` injected silently."""
    cli = _FakeCCXTClient()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.market_order("PEPE", Side.LONG, 1.0)
    p = cli.market_calls[0]["params"]
    assert "newClientOrderId" not in p


@pytest.mark.asyncio
async def test_ccxt_adapter_stop_order_also_forwards_client_order_id() -> None:
    cli = _FakeCCXTClient()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.place_stop_order("PEPE", Side.SHORT, 1.0, stop_price=0.95,
                              client_order_id="alt-s-xyz")
    p = cli.order_calls[0]["params"]
    assert p.get("newClientOrderId") == "alt-s-xyz"


# ====================================================================== #
# B.1.2 — ccxt error classification
# ====================================================================== #


def test_classify_error_retriable_message_hints() -> None:
    for msg in (
        "connection timeout",
        "Service Unavailable (503)",
        "Bad Gateway 502",
        "504 gateway timeout",
        "rate limit exceeded",
        "ddos protection",
        "ssl handshake failed",
    ):
        assert _classify_error(RuntimeError(msg)) is True, msg


def test_classify_error_non_retriable_message_hints() -> None:
    for msg in (
        "Insufficient funds for order",
        "Invalid order: price out of range",
        "min notional not met",
        "Bad Request: invalid symbol",
        "Authentication failed: invalid api key",
        "Permission denied",
    ):
        assert _classify_error(RuntimeError(msg)) is False, msg


def test_classify_error_unknown_defaults_retriable() -> None:
    """Defensive bias: unknown error => retry. The retry budget is
    bounded so worst case is N repeated failures, not silent loss."""
    assert _classify_error(RuntimeError("some-mystery-error-XYZ")) is True


def test_classify_error_uses_real_ccxt_classes_when_available() -> None:
    """When ccxt is installed, we should classify by isinstance, not
    just by message. This guards against a vendor changing its error
    message format and us getting the classification wrong."""
    retriable, non_retriable = _build_ccxt_error_sets()
    # ccxt is in dev dependencies, so these should be non-empty.
    assert len(retriable) > 0
    assert len(non_retriable) > 0
    # Spot-check one from each.
    import ccxt
    if hasattr(ccxt, "NetworkError"):
        # NetworkError("anything") with NO retriable/non-retriable hint
        # in the message must still classify as retriable.
        assert _classify_error(ccxt.NetworkError("plain error")) is True
    if hasattr(ccxt, "InsufficientFunds"):
        assert _classify_error(ccxt.InsufficientFunds("fine-print")) is False


# ====================================================================== #
# B.1.3 — event-driven persistence
# ====================================================================== #


def test_change_listener_fires_on_set_cooldown() -> None:
    a = _account()
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    a.set_cooldown("PEPE", 3600, now_ms=1_000_000)
    assert len(fired) == 1
    assert fired[0] is a


def test_change_listener_fires_on_halt() -> None:
    a = _account()
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    a.halt("manual: ops stopping for deploy")
    assert len(fired) == 1
    assert a.global_trading_halted is True


def test_change_listener_fires_on_record_pnl() -> None:
    a = _account()
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    # A losing fill -> PnL down, equity down, daily_stoploss_hits up,
    # consecutive_losses[symbol] up. Single notification.
    a.record_pnl("PEPE", -100.0)
    assert len(fired) == 1
    assert a.realized_pnl_today_usdt == pytest.approx(-100.0)
    assert a.equity_usdt == pytest.approx(9_900.0)
    assert a.daily_stoploss_hits == 1
    assert a.consecutive_losses == {"PEPE": 1}


def test_record_pnl_winning_clears_consecutive_loss_and_notifies() -> None:
    a = _account()
    a.consecutive_losses["PEPE"] = 2  # pre-existing streak
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    a.record_pnl("PEPE", +50.0)
    assert len(fired) == 1
    assert a.consecutive_losses == {}
    assert a.realized_pnl_today_usdt == pytest.approx(50.0)
    assert a.daily_stoploss_hits == 0


def test_clear_consecutive_loss_only_notifies_on_actual_change() -> None:
    a = _account()
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    # No existing streak -> no notification (idempotent no-op).
    a.clear_consecutive_loss("PEPE")
    assert len(fired) == 0
    a.consecutive_losses["PEPE"] = 1
    a.clear_consecutive_loss("PEPE")
    assert len(fired) == 1


def test_maybe_roll_over_day_first_boot_does_not_notify() -> None:
    """Boot stamping (prev=None -> today) must not fire a notification
    because nothing was reset; only an actual day-flip should persist."""
    a = _account()
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    rolled = a.maybe_roll_over_day(now_ms=1_700_000_000_000)
    assert rolled is False  # first stamp returns False
    assert len(fired) == 0


def test_maybe_roll_over_day_genuine_flip_notifies() -> None:
    """A real day flip wipes today's PnL counter + stoploss-hits and
    must persist."""
    a = _account()
    a.last_rollover_date_utc = "2026-05-15"
    a.realized_pnl_today_usdt = -250.0
    a.daily_stoploss_hits = 2
    fired: list[AccountState] = []
    a.register_change_listener(fired.append)
    # Day after the stamped one (2026-05-17 = 1747440000000ms).
    rolled = a.maybe_roll_over_day(now_ms=1_747_440_000_000)
    assert rolled is True
    assert len(fired) == 1
    assert a.realized_pnl_today_usdt == 0.0
    assert a.daily_stoploss_hits == 0


def test_persistor_round_trip_via_change_listener(tmp_path) -> None:
    """End-to-end: register AccountPersistor.save as the listener, then
    every mutation should write to disk synchronously."""
    p = AccountPersistor(path=tmp_path / "acc.json")
    a = _account()
    a.register_change_listener(lambda state: p.save(state))

    a.set_cooldown("PEPE", 3600, now_ms=1_000_000)
    assert p.path.exists()
    saved_after_cooldown = p.path.read_text()
    assert "PEPE" in saved_after_cooldown

    a.record_pnl("PEPE", -300.0)
    saved_after_loss = p.path.read_text()
    # Most recent save reflects the loss.
    assert '"realized_pnl_today_usdt": -300.0' in saved_after_loss
    assert '"daily_stoploss_hits": 1' in saved_after_loss

    a.halt("KILL_SWITCH:test")
    saved_after_halt = p.path.read_text()
    assert '"global_trading_halted": true' in saved_after_halt


def test_change_listener_failure_does_not_break_state_mutation() -> None:
    """If the listener throws (e.g. disk full), the mutation still
    sticks in memory — persistence is defence-in-depth, not a
    correctness invariant."""
    a = _account()

    def boom(_state: AccountState) -> None:
        raise RuntimeError("disk full")

    a.register_change_listener(boom)
    # Should not propagate.
    a.set_cooldown("PEPE", 3600, now_ms=1_000_000)
    assert "PEPE" in a.cooldown_until_ts_ms


def test_register_change_listener_replaces_previous() -> None:
    """Re-registration replaces; passing None disables."""
    a = _account()
    first: list[int] = []
    second: list[int] = []
    a.register_change_listener(lambda _s: first.append(1))
    a.set_cooldown("PEPE", 60, now_ms=0)
    a.register_change_listener(lambda _s: second.append(1))
    a.set_cooldown("WIF", 60, now_ms=0)
    a.register_change_listener(None)
    a.set_cooldown("DOGE", 60, now_ms=0)
    assert first == [1]
    assert second == [1]


# ====================================================================== #
# B.1 integration — executor mutations propagate to persistor via the
# new event-driven hook (closes the loop end-to-end).
# ====================================================================== #


@pytest.mark.asyncio
async def test_executor_stop_failure_cooldown_persists_via_listener(
    tmp_path,
) -> None:
    """When ``open()`` hits the stop-placement-failed path, the executor
    sets a 4h cooldown. With the change-listener wired, that cooldown
    must be on disk before the function returns — otherwise a crash
    right after would lose it.
    """
    adapter = _RecordingAdapter()

    async def always_fail_stop(*args, **kwargs):
        raise RuntimeError("simulated stop failure")
    adapter.place_stop_order = always_fail_stop  # type: ignore[assignment]

    p = AccountPersistor(path=tmp_path / "acc.json")
    a = _account()
    a.register_change_listener(lambda state: p.save(state))

    ex = CCXTExecutor(
        adapter=adapter, place_stop_retries=0,
        stop_retry_base_delay_sec=0.001,
    )
    with pytest.raises(ExecutionError):
        await ex.open(symbol="PEPE", decision=_decision(), current_price=1.0,
                      account=a, trace_id="t1")

    # Cooldown set in-memory.
    assert "PEPE" in a.cooldown_until_ts_ms
    # AND on disk (event-driven persistence kicked in).
    assert p.path.exists()
    import json
    snap = json.loads(p.path.read_text())
    assert "PEPE" in snap["cooldown_until_ts_ms"]
