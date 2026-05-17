"""Tests for the manual-deposit/withdrawal detector.

Operational patch: when an operator wires money in or out of the
exchange, the daemon must reconcile ``AccountState.equity_usdt`` to
the venue truth WITHOUT polluting the daily drawdown breaker.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from altcoin_agent.risk.state import AccountState
from altcoin_agent.risk.withdrawal_detector import (
    WithdrawalDetector,
    WithdrawalDetectorConfig,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


class _FakeAdapter:
    """Programmable stub of the ``_BalanceAdapter`` protocol."""

    def __init__(self, balances: list[float], raise_on: set[int] | None = None):
        self._balances = list(balances)
        self._raise_on = raise_on or set()
        self._calls = 0

    async def fetch_total_usdt_balance(self) -> float:
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError(f"simulated failure on call {idx}")
        if idx >= len(self._balances):
            return self._balances[-1]
        return self._balances[idx]


def _account(equity: float = 10_000.0) -> AccountState:
    a = AccountState(equity_usdt=equity, starting_equity_today_usdt=equity)
    a.reconciliation_complete = True
    return a


def _detector(account: AccountState, balances: list[float],
              raise_on: set[int] | None = None,
              min_delta: float = 100.0) -> tuple[WithdrawalDetector, _FakeAdapter]:
    adapter = _FakeAdapter(balances, raise_on=raise_on)
    det = WithdrawalDetector(
        adapter=adapter,
        account=account,
        cfg=WithdrawalDetectorConfig(
            enabled=True, poll_interval_sec=0.01,
            min_significant_delta_usdt=min_delta,
            startup_grace_sec=0.0,
        ),
    )
    return det, adapter


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_first_poll_establishes_baseline_no_action() -> None:
    """First successful poll must NOT touch ``account.equity_usdt``."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0])
    out = await det.poll_once()
    assert out["action"] == "baseline_set"
    assert out["observed"] == pytest.approx(10_000.0)
    # Account untouched.
    assert a.equity_usdt == pytest.approx(10_000.0)
    assert a.starting_equity_today_usdt == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_withdrawal_detected_adjusts_equity_baseline() -> None:
    """Operator wires 5,000 out: venue drops 10k -> 5k. Detector must
    bring local equity to 5k. starting_equity rescales proportionally
    (not additively) — see ``adjust_equity_baseline`` for the rationale.
    """
    a = _account(equity=10_000.0)
    # Some realised PnL during the day so we exercise the netting math.
    a.realized_pnl_today_usdt = -200.0  # -2% drawdown vs starting 10k

    det, _ = _detector(a, balances=[10_000.0, 5_000.0])
    await det.poll_once()  # baseline
    out = await det.poll_once()

    assert out["action"] == "withdrawal_detected"
    assert out["applied"] == pytest.approx(-5_000.0)

    # Equity reconciles to venue.
    assert a.equity_usdt == pytest.approx(5_000.0)
    # Starting equity rescaled by 0.5 (the same proportion as equity).
    assert a.starting_equity_today_usdt == pytest.approx(5_000.0)
    # Drawdown ratio DOUBLES because the same 200 USDT loss now
    # represents a larger fraction of a halved capital base. This is
    # the correct semantic — see ``adjust_equity_baseline`` docstring.
    assert a.daily_drawdown_pct == pytest.approx(0.04, abs=1e-6)


@pytest.mark.asyncio
async def test_deposit_detected_adjusts_equity_baseline() -> None:
    """Operator wires 5,000 IN: venue rises 10k -> 15k. Same rescale
    treatment, only with positive delta."""
    a = _account(equity=10_000.0)
    a.realized_pnl_today_usdt = -100.0  # -1% drawdown vs 10k

    det, _ = _detector(a, balances=[10_000.0, 15_000.0])
    await det.poll_once()
    out = await det.poll_once()

    assert out["action"] == "deposit_detected"
    assert out["applied"] == pytest.approx(5_000.0)
    assert a.equity_usdt == pytest.approx(15_000.0)
    # 15000/10000 = 1.5x scale -> starting goes 10000 -> 15000
    assert a.starting_equity_today_usdt == pytest.approx(15_000.0)
    # Drawdown ratio shrinks by 1.5x (same 100 USDT loss vs larger base).
    assert a.daily_drawdown_pct == pytest.approx(100.0 / 15_000.0, abs=1e-6)


@pytest.mark.asyncio
async def test_below_threshold_delta_is_silently_absorbed() -> None:
    """50 USDT drift (below the default 100 threshold) must NOT trigger
    a reconcile — but the snapshot must still update so cumulative
    drift doesn't compound across polls."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0, 9_950.0])
    await det.poll_once()
    out = await det.poll_once()

    assert out["action"] == "below_threshold"
    # equity untouched.
    assert a.equity_usdt == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_realised_pnl_change_does_not_trigger_event() -> None:
    """Between polls a winning trade closes for +500 PnL. Venue
    balance moves +500 accordingly. The detector must subtract this
    expected PnL flow and see no unexplained delta."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0, 10_500.0])
    await det.poll_once()  # baseline at 10k, pnl_total=0

    # Simulate trading recording a 500 USDT win. Note: record_pnl
    # also bumps equity_usdt. The detector compares local pnl_today
    # delta against venue delta — they cancel out.
    a.record_pnl("PEPE", 500.0)
    assert a.equity_usdt == pytest.approx(10_500.0)
    assert a.realized_pnl_today_usdt == pytest.approx(500.0)

    out = await det.poll_once()
    # Venue moved 500, PnL moved 500 -> unexplained 0 -> noop.
    assert out["action"] in ("below_threshold", "noop")
    assert out["unexplained"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_withdrawal_during_active_trading_isolates_external() -> None:
    """Realistic combined scenario:
       Poll 1: baseline at 10,000.
       Between polls: trade closes for +200, operator wires 3,000 out.
       Poll 2: venue shows 10,000 + 200 - 3,000 = 7,200.
       Detector must subtract the 200 pnl and flag 3,000 withdrawal."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0, 7_200.0])
    await det.poll_once()

    a.record_pnl("PEPE", 200.0)  # equity now 10_200, pnl_today=200

    out = await det.poll_once()
    assert out["action"] == "withdrawal_detected"
    # venue_delta = 7200 - 10000 = -2800
    # pnl_delta = 200 - 0 = +200
    # unexplained = -2800 - 200 = -3000
    assert out["unexplained"] == pytest.approx(-3_000.0, abs=1e-6)
    # Account.equity reconciled to venue truth (7200), NOT to local
    # 10,200 minus 3,000 — the venue is authoritative.
    assert a.equity_usdt == pytest.approx(7_200.0)


@pytest.mark.asyncio
async def test_fetch_failure_swallowed_and_retried_next_poll() -> None:
    """A transient adapter error must NOT crash the detector and must
    NOT advance the baseline (otherwise the next poll could then
    register a phantom event when balance returns)."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0, 5_000.0],
                       raise_on={1})  # second call raises
    await det.poll_once()  # baseline OK

    out = await det.poll_once()  # raises -> swallowed
    assert out["ok"] is False
    assert "error" in out
    # Account untouched.
    assert a.equity_usdt == pytest.approx(10_000.0)

    # Third call now sees the same 5,000 balance — but our adapter
    # returns the LAST element after the list is exhausted, so we
    # actually see 5,000 here. The reconcile fires on this poll.
    out = await det.poll_once()
    assert out["action"] == "withdrawal_detected"


@pytest.mark.asyncio
async def test_negative_balance_from_adapter_ignored() -> None:
    """A buggy adapter returning negative must NOT corrupt state."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0, -50.0])
    await det.poll_once()
    out = await det.poll_once()
    assert out["ok"] is False
    assert out["error"] == "negative_balance"
    assert a.equity_usdt == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_run_loop_respects_stop_event() -> None:
    """The long-running ``run`` must terminate promptly when the stop
    event is set — even mid-polling."""
    a = _account(equity=10_000.0)
    det, _ = _detector(a, balances=[10_000.0])

    stop = asyncio.Event()

    async def stopper():
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(det.run(stop), stopper())
    assert det.stats["polls"] >= 1


@pytest.mark.asyncio
async def test_disabled_detector_is_noop() -> None:
    """``cfg.enabled=False`` must short-circuit ``run`` immediately
    without polling."""
    a = _account(equity=10_000.0)
    adapter = _FakeAdapter([10_000.0])
    det = WithdrawalDetector(
        adapter=adapter, account=a,
        cfg=WithdrawalDetectorConfig(enabled=False),
    )
    stop = asyncio.Event()
    await det.run(stop)  # returns immediately
    assert det.stats["polls"] == 0


@pytest.mark.asyncio
async def test_on_event_callback_fires() -> None:
    """The optional ``on_event`` async callback receives reason +
    delta + diagnostic dict on every reconcile."""
    a = _account(equity=10_000.0)
    captured: list[tuple[str, float, dict[str, Any]]] = []

    async def listener(reason, delta, diag):
        captured.append((reason, delta, diag))

    det, _ = _detector(a, balances=[10_000.0, 5_000.0])
    det.on_event = listener
    await det.poll_once()
    await det.poll_once()

    assert len(captured) == 1
    reason, delta, diag = captured[0]
    assert reason == "withdrawal_detected"
    assert delta == pytest.approx(-5_000.0)
    assert "venue_balance" in diag
