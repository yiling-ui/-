"""withdrawal_detector.py — Detect manual deposits / withdrawals at the
exchange and reconcile local ``AccountState.equity_usdt`` to venue truth.

Operational patch
=================

The operator's flow looks like this::

    1. Daemon trades from 500 USDT on Binance.
    2. Account grows to 10,000 USDT over a couple of weeks.
    3. Operator wires 5,000 USDT out of Binance to their bank.
    4. Daemon should keep trading from the remaining 5,000 USDT — NOT
       from a phantom 10,000 that the local ``AccountState`` still
       thinks exists.

V1.0 had no mechanism to detect step (3). The exchange-side balance
silently dropped while ``account.equity_usdt`` stayed at the old
number, so:

  * The position sizer kept treating equity = 10,000 -> oversized
    every new entry by 2× until the next reconciliation noticed
    open-position diffs (which it never directly noticed because
    open positions are unrelated to spot/futures wallet balance).
  * The daily-drawdown circuit breaker became unreliable: ratios
    computed against ``starting_equity_today_usdt`` did not match the
    real bank account.

Design
======

* **Polling, not realtime.** Withdrawals are operator-driven (bank
  wires take minutes-to-hours to land); we poll every
  ``poll_interval_sec`` (default 5 minutes). Polling avoids needing
  WS account streams which are venue-specific.

* **Quiet by default.** A balance change of less than
  ``min_significant_delta_usdt`` is ignored — exchange fees,
  funding settlement, micro-fills mid-poll all drift the wallet by
  small amounts. Only changes ≥ this threshold (default 100 USDT)
  trigger a reconcile.

* **PnL-aware.** Between two polls the venue balance can change due
  to (a) realised PnL on closed positions and (b) operator
  deposits/withdrawals. We track ``last_observed_equity`` AND the
  cumulative ``realized_pnl_today_usdt`` snapshot at last poll, so
  the *expected* delta is the realised-PnL delta. Anything beyond
  that gap is an external flow.

* **Conservative on direction.** A negative external delta is logged
  as ``withdrawal``; a positive one as ``deposit``. Both flow
  through :meth:`AccountState.adjust_equity_baseline` which keeps
  daily drawdown invariant.

* **Fail-open.** Any error fetching the balance is logged at
  warning level and the detector skips this round. We never block
  the trading loop or halt the account on a balance-fetch failure —
  the next round will reconcile.

* **No realtime equity authority.** We do NOT replace the unrealised-
  PnL feed used by the trailing stop or by ``account.equity_usdt``
  during a live position. The detector only reconciles when the
  delta is unexplained by realised-PnL movement, and only adjusts
  the *baseline* (starting + current both shift by the same amount).
  Mark-to-market intra-position movement stays where it lives today.

Usage
=====

The detector is constructed by ``main.App.run`` after the executor
adapter is built::

    detector = WithdrawalDetector(
        adapter=ccxt_adapter,
        account=account,
        notifier=notifier,
        cfg=WithdrawalDetectorConfig(...),
    )
    asyncio.create_task(detector.run(stop_event))

Tests construct it directly with a fake adapter and call
:meth:`poll_once` to drive each scenario deterministically.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from altcoin_agent.risk.state import AccountState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Adapter contract
# ---------------------------------------------------------------------- #


@runtime_checkable
class _BalanceAdapter(Protocol):
    """Minimal async surface the detector needs.

    ``CCXTExchangeAdapter`` is extended in this PR to satisfy this
    protocol via :meth:`fetch_total_usdt_balance`. Tests pass any
    object with the same async method.
    """

    async def fetch_total_usdt_balance(self) -> float:
        """Return the operator's total USDT-equivalent balance on the
        venue. For a USDT-margined perpetuals account this is the
        wallet balance + unrealised PnL on open positions, since both
        contribute to margin / equity. The return value should be
        directly comparable to ``AccountState.equity_usdt``.

        Implementations must raise on transient errors (network, 5xx,
        throttling). The detector catches and skips the round.
        """
        ...


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #


@dataclass
class WithdrawalDetectorConfig:
    """Operator-tunable knobs.

    Defaults are chosen so a 500-10,000 USDT account does not see
    spurious notifications: a 100 USDT delta is large enough that
    no reasonable funding/fee combination would trigger it
    accidentally, but small enough that an operator's intentional
    bank wire (typically ≥ 500 USDT) is always caught.
    """

    enabled: bool = True
    # How often we ask the venue for the wallet balance.
    poll_interval_sec: float = 300.0  # 5 minutes
    # Minimum unexplained delta to act on. Below this we update
    # ``last_observed_equity`` quietly to avoid drift accumulation.
    min_significant_delta_usdt: float = 100.0
    # Initial grace period after detector start before the first
    # comparison — gives the rest of the daemon time to finish
    # reconciliation. Without this we'd compare an
    # almost-uninitialised ``account.equity_usdt`` against the venue
    # and (correctly) flag the entire account as a "deposit".
    startup_grace_sec: float = 30.0


# ---------------------------------------------------------------------- #
# Detector
# ---------------------------------------------------------------------- #


@dataclass
class WithdrawalDetector:
    """Polls the venue for balance changes and reconciles with
    :class:`AccountState`.

    The detector is intentionally stateless beyond:
      * ``_last_observed_equity`` — wallet balance snapshot from the
        last successful poll.
      * ``_last_realized_pnl_total`` — cumulative realised PnL from
        the local account at the last poll, so we can subtract
        between-poll trading flow from the wallet delta.

    Both fields are reset only via ``poll_once``; there is no
    persistence to disk. After a daemon restart the first poll
    establishes a fresh baseline (no false positive on boot because
    ``startup_grace_sec`` blocks the comparison).
    """

    adapter: _BalanceAdapter
    account: AccountState
    cfg: WithdrawalDetectorConfig = None  # type: ignore[assignment]
    on_event: Callable[[str, float, dict[str, Any]], Awaitable[None]] | None = None

    # Internal state.
    _last_observed_equity: float | None = None
    _last_realized_pnl_total: float = 0.0
    # Operational patch (post-review): the daemon zeros
    # ``account.realized_pnl_today_usdt`` at UTC midnight via
    # ``maybe_roll_over_day``. Without tracking the rollover marker
    # here, the next poll computed:
    #     pnl_delta   = 0  -  yesterdays_pnl       (e.g. -500)
    #     unexplained = venue_delta - pnl_delta    (e.g. +500)
    # and fired a phantom "deposit_detected" event of yesterday's
    # PnL — once per UTC day per running daemon. We snapshot
    # ``last_rollover_date_utc`` alongside the PnL total and, when
    # the dates differ between polls, reset both snapshots from the
    # post-rollover state without raising any flow event.
    _last_rollover_date_utc: str | None = None
    _polls: int = 0
    _events: int = 0
    _errors: int = 0

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = WithdrawalDetectorConfig()

    # --------------------------- public --------------------------- #

    async def run(self, stop_event: asyncio.Event) -> None:
        """Long-running poller. Blocks until ``stop_event`` is set."""
        if not self.cfg.enabled:
            logger.info("WithdrawalDetector disabled by config")
            return
        # Startup grace.
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=self.cfg.startup_grace_sec,
            )
            return  # stop fired during grace
        except asyncio.TimeoutError:
            pass
        logger.info(
            "WithdrawalDetector started: poll_interval=%.0fs "
            "min_delta=%.2f USDT",
            self.cfg.poll_interval_sec, self.cfg.min_significant_delta_usdt,
        )
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as e:
                # ``poll_once`` already swallows; this is belt-and-
                # braces for an unexpected programming error in the
                # hot path so we never crash the daemon.
                self._errors += 1
                logger.exception(
                    "WithdrawalDetector.poll_once unexpected: %s", e,
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.cfg.poll_interval_sec,
                )
            except asyncio.TimeoutError:
                continue

    async def poll_once(self) -> dict[str, Any]:
        """Single poll-and-reconcile cycle. Returns a diagnostic dict
        for tests / dashboards. Never raises.
        """
        self._polls += 1
        out: dict[str, Any] = {
            "ok": False, "delta": 0.0, "action": "noop",
        }

        try:
            observed = float(await self.adapter.fetch_total_usdt_balance())
        except Exception as e:
            self._errors += 1
            logger.warning(
                "WithdrawalDetector: fetch_total_usdt_balance failed "
                "(swallowed): %s", e,
            )
            out["error"] = str(e)
            return out

        if observed < 0:
            # Defensive: an adapter that returns negative is buggy.
            logger.warning(
                "WithdrawalDetector: adapter returned negative balance "
                "%.2f, ignoring", observed,
            )
            out["error"] = "negative_balance"
            return out

        # First successful poll: establish baseline, do not act.
        if self._last_observed_equity is None:
            self._last_observed_equity = observed
            self._last_realized_pnl_total = self.account.realized_pnl_today_usdt
            self._last_rollover_date_utc = self.account.last_rollover_date_utc
            logger.info(
                "WithdrawalDetector baseline established: "
                "venue_balance=%.2f USDT", observed,
            )
            out["ok"] = True
            out["action"] = "baseline_set"
            out["observed"] = observed
            return out

        # Operational patch (post-review): UTC-day rollover handling.
        # ``account.maybe_roll_over_day`` zeros
        # ``realized_pnl_today_usdt`` at the trading-day boundary; if
        # we don't notice that here the next netting computes a
        # phantom flow of yesterday's PnL. Detect by comparing
        # rollover date markers — when they differ, reset OUR snapshot
        # to the current (post-rollover) totals and skip the round so
        # the operator never sees a ghost event. The rollover itself
        # is logged at INFO so the audit trail is intact.
        cur_rollover_date = self.account.last_rollover_date_utc
        if (
            cur_rollover_date is not None
            and self._last_rollover_date_utc is not None
            and cur_rollover_date != self._last_rollover_date_utc
        ):
            prev_rollover_date = self._last_rollover_date_utc
            logger.info(
                "WithdrawalDetector: detected UTC-day rollover "
                "(%s -> %s); resetting PnL snapshot to avoid phantom "
                "flow event. observed=%.2f",
                prev_rollover_date, cur_rollover_date, observed,
            )
            self._last_observed_equity = observed
            self._last_realized_pnl_total = (
                self.account.realized_pnl_today_usdt
            )
            self._last_rollover_date_utc = cur_rollover_date
            out["ok"] = True
            out["action"] = "rollover_resync"
            out["observed"] = observed
            out["rollover_from"] = prev_rollover_date
            out["rollover_to"] = cur_rollover_date
            return out

        # Delta we actually saw on the venue.
        venue_delta = observed - self._last_observed_equity

        # Delta we EXPECT due to local PnL bookkeeping in the same
        # interval. Realised PnL only — unrealised intra-position
        # mark-to-market is captured by the venue balance directly,
        # so subtracting realised gives us the externally-driven
        # residual.
        pnl_now = self.account.realized_pnl_today_usdt
        pnl_delta = pnl_now - self._last_realized_pnl_total

        unexplained = venue_delta - pnl_delta

        out["observed"] = observed
        out["venue_delta"] = venue_delta
        out["pnl_delta"] = pnl_delta
        out["unexplained"] = unexplained
        out["ok"] = True

        # Always advance the local snapshots so the next round
        # measures from the latest known state. Tiny drift is
        # absorbed silently here.
        self._last_observed_equity = observed
        self._last_realized_pnl_total = pnl_now

        if abs(unexplained) < self.cfg.min_significant_delta_usdt:
            out["action"] = "below_threshold"
            return out

        # Significant unexplained delta. Reconcile local equity to
        # the venue truth. ``adjust_equity_baseline`` keeps daily
        # drawdown invariant.
        reason = "withdrawal_detected" if unexplained < 0 else "deposit_detected"
        applied = self.account.adjust_equity_baseline(
            new_equity_usdt=observed, reason=reason,
        )
        self._events += 1
        out["action"] = reason
        out["applied"] = applied

        logger.warning(
            "WithdrawalDetector: %s — venue_delta=%+.2f pnl_delta=%+.2f "
            "unexplained=%+.2f -> account.equity_usdt=%.2f",
            reason, venue_delta, pnl_delta, unexplained, self.account.equity_usdt,
        )

        if self.on_event is not None:
            try:
                await self.on_event(reason, applied, {
                    "venue_balance": observed,
                    "venue_delta": venue_delta,
                    "pnl_delta": pnl_delta,
                    "unexplained": unexplained,
                })
            except Exception as e:
                logger.warning(
                    "WithdrawalDetector on_event listener raised "
                    "(swallowed): %s", e,
                )
        return out

    # --------------------------- diagnostics --------------------------- #

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "polls": self._polls,
            "events": self._events,
            "errors": self._errors,
            "last_observed_equity": self._last_observed_equity,
            "last_realized_pnl_total": self._last_realized_pnl_total,
            "last_rollover_date_utc": self._last_rollover_date_utc,
        }
