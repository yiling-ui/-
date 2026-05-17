"""Tests for the order-execution-layer audit fixes:

  * #E1 — every order sent through ``CCXTExchangeAdapter`` carries a
          venue-specific ``clientOrderId`` derived from a deterministic
          per-position ``trace_id`` root. Retried calls reuse the same
          coid; legs / stop replacements / emergency closes derive
          *unique-but-related* coids so the venue's 24h coid uniqueness
          window doesn't reject legitimate follow-ups.
  * #E2 — transient errors (502 / 504 / 429 / connection reset / timeout
          / rate-limit phrasing) trigger full-jitter exponential-backoff
          retries inside the adapter. Logical errors (InvalidOrder /
          InsufficientFunds / BadSymbol) are re-raised immediately.
          ``total_timeout_sec`` caps the entire sequence so a hung TCP
          socket cannot deadlock the executor coroutine.

  * Executor wiring — ``CCXTExecutor.open`` passes a stable trace_root
          to every call so retries inside the adapter and stop attempts
          inside the executor share the right coids.
"""

from __future__ import annotations

import asyncio

import pytest

from altcoin_agent.risk.ccxt_adapter import (
    CCXTExchangeAdapter,
    _derive_client_oid,
    _inject_client_oid,
    _is_transient_error,
)
from altcoin_agent.risk.executor import (
    CCXTExecutor,
    _entry_coid,
    _stop_coid,
    _trace_root,
)
from altcoin_agent.risk.gate import RiskDecision
from altcoin_agent.risk.state import AccountState, Side

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _account() -> AccountState:
    return AccountState(equity_usdt=10_000.0,
                         starting_equity_today_usdt=10_000.0)


def _decision(size: float = 100.0) -> RiskDecision:
    return RiskDecision(
        approved=True, side=Side.LONG, size=size, leverage=5.0,
        notional_usdt=size * 1.0, initial_stop=0.95,
        max_slippage_used=0.005, reason="ok",
    )


# --------------------------------------------------------------------- #
# CCXTLike fakes
# --------------------------------------------------------------------- #


class _RecordingCCXT:
    """ccxt.pro fake that records every call's params verbatim."""

    def __init__(self) -> None:
        self.market_calls: list[dict] = []
        self.order_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"ccxt-{self._n}"

    async def create_market_order(self, symbol, side, amount, *, params=None):  # noqa: ANN001
        self.market_calls.append({"symbol": symbol, "side": side,
                                    "amount": amount,
                                    "params": dict(params or {})})
        return {"id": self._id(), "symbol": symbol, "side": side,
                "amount": amount, "filled": amount, "average": 1.0,
                "status": "closed",
                "clientOrderId": (params or {}).get("newClientOrderId")
                                  or (params or {}).get("clOrdId")
                                  or (params or {}).get("clientOrderId")
                                  or (params or {}).get("orderLinkId")
                                  or "",
                "info": {"clientOrderId":
                         (params or {}).get("newClientOrderId") or ""},
                }

    async def create_order(self, symbol, type, side, amount, *,  # noqa: A002, ANN001
                            price=None, params=None):
        self.order_calls.append({"symbol": symbol, "type": type, "side": side,
                                  "amount": amount, "price": price,
                                  "params": dict(params or {})})
        return {"id": self._id(), "symbol": symbol, "side": side,
                "amount": amount,
                "stopPrice": (params or {}).get("stopPrice"),
                "info": {"reduceOnly": (params or {}).get("reduceOnly", False)},
                "status": "open",
                "clientOrderId": (params or {}).get("newClientOrderId")
                                  or (params or {}).get("clOrdId")
                                  or "",
                }

    async def cancel_order(self, id, symbol=None, params=None):  # noqa: A002, ANN001
        self.cancel_calls.append({"id": id, "symbol": symbol})
        return {"id": id, "status": "cancelled"}

    async def set_leverage(self, leverage, symbol, params=None):  # noqa: ANN001
        return {"leverage": leverage, "symbol": symbol}

    async def fetch_positions(self, symbols=None, params=None):  # noqa: ANN001
        return []

    async def fetch_open_orders(self, symbol=None, params=None):  # noqa: ANN001
        return []


class _FlakyCCXT(_RecordingCCXT):
    """Fail the first ``fail_first`` ``create_market_order`` calls with the
    given exception, then succeed. Used to verify retry + idempotency."""

    def __init__(self, *, fail_first: int, exc: Exception) -> None:
        super().__init__()
        self.fail_first = fail_first
        self.exc = exc
        self.attempts = 0

    async def create_market_order(self, symbol, side, amount, *, params=None):  # noqa: ANN001
        self.attempts += 1
        if self.attempts <= self.fail_first:
            # Still record the params so the test can assert the same
            # coid was used across retries.
            self.market_calls.append({"symbol": symbol, "side": side,
                                        "amount": amount,
                                        "params": dict(params or {}),
                                        "outcome": "raised"})
            raise self.exc
        return await super().create_market_order(
            symbol, side, amount, params=params,
        )


# ====================================================================== #
# A. Transient error classification
# ====================================================================== #


@pytest.mark.parametrize("msg", [
    "HTTP 502 Bad Gateway",
    "Service Unavailable (503)",
    "Gateway timeout (504)",
    "429 Too Many Requests",
    "Rate limit exceeded",
    "Connection reset by peer",
    "Read timed out",
    "ddos protection triggered",
    "temporarily unavailable",
])
def test_classifier_marks_infrastructure_failures_as_transient(msg: str) -> None:
    assert _is_transient_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "InvalidOrder: lot size below minimum",
    "Insufficient funds",
    "Bad symbol PEPEUSDT",
    "authentication required",
    "permission denied",
])
def test_classifier_marks_logical_errors_as_non_transient(msg: str) -> None:
    # Pure RuntimeError("Insufficient funds") would also trigger the regex
    # nothing — but the *class-name* path is what matters in production
    # because real ccxt raises ccxt.InsufficientFunds. We simulate the
    # class-name path with a custom exception type.
    if "Insufficient funds" in msg:
        class InsufficientFunds(Exception):
            pass
        assert _is_transient_error(InsufficientFunds(msg)) is False
        return
    if "InvalidOrder" in msg:
        class InvalidOrder(Exception):
            pass
        assert _is_transient_error(InvalidOrder(msg)) is False
        return
    if "Bad symbol" in msg:
        class BadSymbol(Exception):
            pass
        assert _is_transient_error(BadSymbol(msg)) is False
        return
    # Authentication / permission don't match any retry regex either.
    assert _is_transient_error(RuntimeError(msg)) is False


def test_classifier_ignores_cancellation_and_systemexit() -> None:
    # The retry loop re-raises these without classifying. Verify the
    # classifier itself doesn't mis-flag them.
    assert _is_transient_error(asyncio.CancelledError()) is False
    assert _is_transient_error(KeyboardInterrupt()) is False
    assert _is_transient_error(SystemExit(0)) is False


# ====================================================================== #
# B. clientOrderId derivation + venue-specific injection
# ====================================================================== #


def test_derive_coid_is_deterministic_for_same_trace() -> None:
    a = _derive_client_oid(prefix="e", trace_id="abc-123_xyz!")
    b = _derive_client_oid(prefix="e", trace_id="abc-123_xyz!")
    assert a == b
    assert a.startswith("e")


def test_derive_coid_strips_non_alnum() -> None:
    out = _derive_client_oid(prefix="s", trace_id="trace#$%^&*42")
    assert all(ch.isalnum() for ch in out)
    assert out.startswith("s")


def test_derive_coid_truncates_to_30_chars() -> None:
    long_trace = "z" * 200
    out = _derive_client_oid(prefix="e", trace_id=long_trace, suffix="suffix")
    assert len(out) <= 30


def test_derive_coid_falls_back_when_trace_empty() -> None:
    a = _derive_client_oid(prefix="e", trace_id=None)
    b = _derive_client_oid(prefix="e", trace_id="")
    # Random per call — not deterministic across calls when trace is empty.
    assert a != b
    assert a.startswith("e") and b.startswith("e")


@pytest.mark.parametrize("venue,expected_key", [
    ("binance", "newClientOrderId"),
    ("okx", "clOrdId"),
    ("gateio", "text"),
    ("bybit", "orderLinkId"),
    ("kraken", "clientOrderId"),  # ccxt unified fallback
    ("UNKNOWN", "clientOrderId"),
])
def test_inject_coid_uses_venue_specific_field(venue: str,
                                                 expected_key: str) -> None:
    params: dict = {}
    _inject_client_oid(params, venue, "etrace42L0")
    assert expected_key in params
    if venue == "gateio":
        # Gate.io requires the 't-' prefix.
        assert params[expected_key].startswith("t-")
    else:
        assert params[expected_key] == "etrace42L0"


# ====================================================================== #
# C. CCXTExchangeAdapter wires coid into the real venue call
# ====================================================================== #


@pytest.mark.asyncio
async def test_market_order_attaches_binance_newClientOrderId() -> None:
    cli = _RecordingCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    out = await a.market_order("RAVEUSDT", Side.LONG, 10.0,
                                client_order_id="etrace42L0")
    assert cli.market_calls[0]["params"]["newClientOrderId"] == "etrace42L0"
    assert out["client_order_id"] == "etrace42L0"


@pytest.mark.asyncio
async def test_market_order_attaches_okx_clOrdId() -> None:
    cli = _RecordingCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="okx")
    await a.market_order("RAVE-USDT-SWAP", Side.LONG, 10.0,
                          client_order_id="etrace42L0")
    assert cli.market_calls[0]["params"]["clOrdId"] == "etrace42L0"


@pytest.mark.asyncio
async def test_market_order_attaches_gateio_text_with_prefix() -> None:
    cli = _RecordingCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="gateio")
    await a.market_order("RAVE_USDT", Side.LONG, 10.0,
                          client_order_id="etrace42L0")
    assert cli.market_calls[0]["params"]["text"] == "t-etrace42L0"


@pytest.mark.asyncio
async def test_place_stop_order_attaches_coid() -> None:
    cli = _RecordingCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.place_stop_order("RAVEUSDT", Side.SHORT, 10.0,
                              stop_price=0.95,
                              client_order_id="strace42S0")
    assert cli.order_calls[0]["params"]["newClientOrderId"] == "strace42S0"


@pytest.mark.asyncio
async def test_market_order_generates_coid_when_caller_omits_one() -> None:
    cli = _RecordingCCXT()
    a = CCXTExchangeAdapter(client=cli, exchange_name="binance")
    await a.market_order("RAVEUSDT", Side.LONG, 10.0)
    coid = cli.market_calls[0]["params"].get("newClientOrderId")
    assert coid and coid.startswith("e")  # entry prefix


# ====================================================================== #
# D. Retry on transient errors keeps the SAME clientOrderId
# ====================================================================== #


@pytest.mark.asyncio
async def test_market_order_retries_on_502_with_same_coid() -> None:
    cli = _FlakyCCXT(fail_first=2, exc=RuntimeError("HTTP 502 Bad Gateway"))
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=3, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    out = await a.market_order("RAVEUSDT", Side.LONG, 10.0,
                                client_order_id="etrace42L0")
    assert cli.attempts == 3
    # Every attempt — including the failing ones — saw the same coid.
    coids = [c["params"]["newClientOrderId"] for c in cli.market_calls]
    assert coids == ["etrace42L0", "etrace42L0", "etrace42L0"]
    assert out["client_order_id"] == "etrace42L0"


@pytest.mark.asyncio
async def test_market_order_retries_on_429_then_succeeds() -> None:
    cli = _FlakyCCXT(fail_first=1,
                       exc=RuntimeError("429 Too Many Requests"))
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=3, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    await a.market_order("RAVEUSDT", Side.LONG, 10.0,
                          client_order_id="etrace42L0")
    assert cli.attempts == 2


@pytest.mark.asyncio
async def test_market_order_does_not_retry_on_logical_error() -> None:
    class InvalidOrder(Exception):
        pass

    cli = _FlakyCCXT(fail_first=99,
                       exc=InvalidOrder("lot size below minimum"))
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=3, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    with pytest.raises(InvalidOrder):
        await a.market_order("RAVEUSDT", Side.LONG, 10.0,
                              client_order_id="etrace42L0")
    # Exactly one attempt — no retries on logical errors.
    assert cli.attempts == 1


@pytest.mark.asyncio
async def test_market_order_does_not_retry_on_insufficient_funds() -> None:
    class InsufficientFunds(Exception):
        pass

    cli = _FlakyCCXT(fail_first=99,
                       exc=InsufficientFunds("Account has 0 USDT"))
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=3, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    with pytest.raises(InsufficientFunds):
        await a.market_order("RAVEUSDT", Side.LONG, 10.0)
    assert cli.attempts == 1


@pytest.mark.asyncio
async def test_market_order_exhausts_retries_and_reraises() -> None:
    cli = _FlakyCCXT(fail_first=99,
                       exc=RuntimeError("HTTP 502 Bad Gateway"))
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=2, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    with pytest.raises(RuntimeError, match="502"):
        await a.market_order("RAVEUSDT", Side.LONG, 10.0)
    # Initial attempt + 2 retries.
    assert cli.attempts == 3


# ====================================================================== #
# E. total_timeout_sec deadline
# ====================================================================== #


class _SlowCCXT(_RecordingCCXT):
    def __init__(self, *, sleep_sec: float) -> None:
        super().__init__()
        self.sleep_sec = sleep_sec

    async def create_market_order(self, symbol, side, amount, *, params=None):  # noqa: ANN001
        await asyncio.sleep(self.sleep_sec)
        return await super().create_market_order(
            symbol, side, amount, params=params,
        )


@pytest.mark.asyncio
async def test_total_timeout_caps_a_hung_call() -> None:
    cli = _SlowCCXT(sleep_sec=2.0)  # well over total_timeout_sec
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=0, total_timeout_sec=0.10,
    )
    with pytest.raises(RuntimeError, match="total_timeout_exceeded"):
        await a.market_order("RAVEUSDT", Side.LONG, 10.0)


# ====================================================================== #
# F. cancel_order is also retry-wrapped
# ====================================================================== #


@pytest.mark.asyncio
async def test_cancel_order_retries_on_transient() -> None:
    class _Flaky(_RecordingCCXT):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def cancel_order(self, id, symbol=None, params=None):  # noqa: A002, ANN001
            self.attempts += 1
            if self.attempts <= 1:
                raise RuntimeError("Read timed out")
            return await super().cancel_order(id, symbol=symbol,
                                               params=params)

    cli = _Flaky()
    a = CCXTExchangeAdapter(
        client=cli, exchange_name="binance",
        transient_retries=3, retry_base_delay_sec=0.001,
        retry_max_delay_sec=0.005, total_timeout_sec=2.0,
    )
    await a.cancel_order("o-1", "RAVEUSDT")
    assert cli.attempts == 2


# ====================================================================== #
# G. Executor passes a stable trace_root through to the adapter
# ====================================================================== #


def test_trace_root_strips_non_alnum_and_is_deterministic() -> None:
    a = _trace_root("trace-id-#42!")
    b = _trace_root("trace-id-#42!")
    assert a == b
    assert all(ch.isalnum() for ch in a)


def test_trace_root_falls_back_when_empty() -> None:
    a = _trace_root(None)
    b = _trace_root("")
    # Different random tokens but both alnum 16-char.
    assert a != b
    assert len(a) == 16


def test_entry_and_stop_coids_share_root_but_differ() -> None:
    root = _trace_root("trace42")
    e0 = _entry_coid(root, leg_id=0)
    s0 = _stop_coid(root, replacement_count=0)
    s1 = _stop_coid(root, replacement_count=1)
    assert e0 != s0
    assert s0 != s1
    # Both prefixes contain the same root substring.
    assert root in e0 and root in s0


class _CountingAdapter:
    """Simple in-memory adapter that records every coid."""

    def __init__(self) -> None:
        self.market_calls: list[dict] = []
        self.stop_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"o-{self._n}"

    async def market_order(
        self, *, symbol, side, size, price=None, reduce_only=False,
        client_order_id=None,
    ):
        self.market_calls.append({
            "symbol": symbol, "side": side.value, "size": size,
            "reduce_only": reduce_only, "coid": client_order_id,
        })
        return {"id": self._id(), "symbol": symbol, "average": price or 1.0,
                "filled": size, "amount": size,
                "client_order_id": client_order_id}

    async def place_stop_order(
        self, *, symbol, side, size, stop_price, reduce_only=True,
        client_order_id=None,
    ):
        self.stop_calls.append({
            "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "coid": client_order_id,
        })
        return {"id": self._id(), "client_order_id": client_order_id}

    async def cancel_order(self, order_id, symbol):
        self.cancel_calls.append({"order_id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self):
        return []


@pytest.mark.asyncio
async def test_executor_open_emits_deterministic_coids() -> None:
    adapter = _CountingAdapter()
    ex = CCXTExecutor(adapter=adapter)
    pos = await ex.open(
        symbol="RAVEUSDT", decision=_decision(), current_price=1.0,
        account=_account(), trace_id="trace-99",
    )
    # Entry coid is e<root>L0.
    entry_coid = adapter.market_calls[0]["coid"]
    stop_coid = adapter.stop_calls[0]["coid"]
    assert entry_coid is not None and entry_coid.startswith("e")
    assert "trace99" in entry_coid
    assert entry_coid.endswith("L0")
    # Stop coid uses the same root + S0 suffix.
    assert stop_coid is not None and stop_coid.startswith("s")
    assert "trace99" in stop_coid
    assert stop_coid.endswith("S0")
    # Different IDs.
    assert entry_coid != stop_coid
    # Position keeps the cached root for downstream tighten_hard_stop.
    assert getattr(pos, "_coid_root", None)
    assert getattr(pos, "_stop_replacement_count", None) == 1


@pytest.mark.asyncio
async def test_executor_tighten_uses_next_stop_coid() -> None:
    adapter = _CountingAdapter()
    ex = CCXTExecutor(adapter=adapter)
    pos = await ex.open(
        symbol="RAVEUSDT", decision=_decision(), current_price=1.0,
        account=_account(), trace_id="trace-99",
    )
    ok = await ex.tighten_hard_stop(pos, new_stop=0.97)
    assert ok is True
    # Two stop_calls now: open (S0) + tighten (S1).
    coids = [c["coid"] for c in adapter.stop_calls]
    assert coids[0].endswith("S0")
    assert coids[1].endswith("S1")
    assert coids[0] != coids[1]
    # Counter should now be at 2 so the next tighten uses S2.
    assert getattr(pos, "_stop_replacement_count", None) == 2


@pytest.mark.asyncio
async def test_executor_open_handles_legacy_adapter_without_coid_kwarg() -> None:
    """Older adapters (in-tree test fakes) didn't accept ``client_order_id``.
    The executor must keep working with them via signature introspection,
    just without the idempotency guarantee. This pins that we don't break
    pre-existing behaviour."""

    class _LegacyAdapter:
        def __init__(self) -> None:
            self.market_calls: list[dict] = []

        async def market_order(self, symbol, side, size, *,
                                price=None, reduce_only=False):
            self.market_calls.append({"symbol": symbol})
            return {"id": "x", "filled": size, "amount": size,
                    "average": price or 1.0}

        async def place_stop_order(self, symbol, side, size, stop_price,
                                    reduce_only=True):
            return {"id": "stop"}

        async def cancel_order(self, order_id, symbol):
            return {"id": order_id, "status": "cancelled"}

        async def set_leverage(self, symbol, leverage):
            return {"symbol": symbol, "leverage": leverage}

        async def fetch_positions(self):
            return []

        async def fetch_open_orders(self):
            return []

    adapter = _LegacyAdapter()
    ex = CCXTExecutor(adapter=adapter)
    # Should not raise even though the adapter lacks client_order_id.
    pos = await ex.open(
        symbol="RAVEUSDT", decision=_decision(), current_price=1.0,
        account=_account(), trace_id="trace-99",
    )
    assert pos.symbol == "RAVEUSDT"


@pytest.mark.asyncio
async def test_executor_emergency_close_uses_distinct_coid() -> None:
    """When stop placement keeps failing the executor falls into the
    naked-exposure branch and emits a reduce_only market order to flat
    the position. That close must carry its OWN coid (not the entry's),
    otherwise the venue rejects it as a duplicate and we end up naked."""

    class _StopFailAdapter(_CountingAdapter):
        async def place_stop_order(self, **kw):
            raise RuntimeError("stop endpoint 502")

    adapter = _StopFailAdapter()
    ex = CCXTExecutor(adapter=adapter, place_stop_retries=1,
                       stop_failure_cooldown_sec=3600)
    with pytest.raises(Exception, match="stop_placement_failed"):
        await ex.open(
            symbol="RAVEUSDT", decision=_decision(), current_price=1.0,
            account=_account(), trace_id="trace-99",
        )
    # Two market orders: the entry, then the emergency reduce_only close.
    assert len(adapter.market_calls) == 2
    entry = adapter.market_calls[0]
    emergency = adapter.market_calls[1]
    assert entry["reduce_only"] is False
    assert emergency["reduce_only"] is True
    # Distinct coids — the entry uses e<root>L0, the close uses r<root>K...
    assert entry["coid"] != emergency["coid"]
    assert entry["coid"].startswith("e")
    assert emergency["coid"].startswith("r")
