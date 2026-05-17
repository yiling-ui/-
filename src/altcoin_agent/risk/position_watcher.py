"""position_watcher.py — exchange-side position close detector.

Closes the loop on the position life-cycle. The flow is:

    open -> exchange-side STOP_MARKET fires -> our local AccountState
            still says "open" forever (BUG #1)

The watcher periodically calls ``adapter.fetch_positions()`` and compares
against ``account.open_positions``. When a symbol that we track locally
disappears from the exchange snapshot (or returns with size <= 0) for
``miss_threshold`` consecutive polls, we declare it closed and invoke
``on_close(position, reason)``.

The debounce protects against transient fetch failures and the brief
window between a market_close call and the exchange propagating the
position-closed state.

Failure modes:
    * adapter.fetch_positions() raises  -> we skip this poll, log a warning,
      and reset the miss counter for safety. Repeated raises do NOT cause
      a false close.
    * on_close raises                   -> logged and swallowed. The
      position is removed from tracking either way (we already concluded
      it's closed on the exchange side; not removing it would loop
      forever).

The watcher is intentionally separate from the trailing controller and
risk gate so it can be tested in isolation against a fake adapter.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from altcoin_agent.risk.executor import ExchangeAdapter
from altcoin_agent.risk.state import AccountState, Position

logger = logging.getLogger(__name__)


CloseCallback = Callable[[Position, str], Awaitable[None]]


@dataclass
class PositionWatcher:
    """Polls the exchange and fires ``on_close`` when a tracked position
    has been closed exchange-side (e.g. STOP_MARKET filled).

    Args:
        adapter: any ExchangeAdapter (live ccxt or DryRunExchangeAdapter).
        account: AccountState whose open_positions we monitor.
        on_close: coroutine invoked once per detected close, with the
            Position object that was just closed plus a short reason
            string. The default reason is ``"exchange_close_detected"``;
            when the executor signals an in-flight emergency close via
            :meth:`hint_close_reason`, that hint wins for the next
            close event on that symbol.
        poll_interval_sec: how often to poll. Default 5s; reduce in tests.
        miss_threshold: consecutive misses required before declaring a
            position closed. Default 2 — at the default poll interval that
            is a 10s debounce, comfortably longer than typical exchange
            propagation lag.

    TICKET-004 close-reason routing:
        ``hint_close_reason(symbol, reason)`` is called by the executor
        the moment it issues an emergency close (partial fill cleanup,
        stop-replacement failure, naked-position close, etc.). The
        next close event on that symbol will report the hinted reason
        instead of the generic default. The dict is single-use:
        consuming the hint pops it. Hints not consumed within the
        watcher's lifetime are simply ignored.
    """

    adapter: ExchangeAdapter
    account: AccountState
    on_close: CloseCallback
    poll_interval_sec: float = 5.0
    miss_threshold: int = 2
    _miss_counts: dict[str, int] = field(default_factory=dict)
    # TICKET-004: short-lived hints from the executor.
    close_reason_hints: dict[str, str] = field(default_factory=dict)

    def hint_close_reason(self, symbol: str, reason: str) -> None:
        """Tell the watcher how to label the *next* close on ``symbol``.

        Called by the executor right after it issues an emergency
        ``market_order(reduce_only=True)``. The watcher will see the
        position vanish from ``fetch_positions`` shortly after and
        invoke ``on_close`` with the hinted reason. Calling this twice
        before the close fires keeps the most recent hint (operator
        intent: "the latest reason is the one that matters").
        """
        if not symbol or not reason:
            return
        self.close_reason_hints[symbol] = reason

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main loop. Returns when ``stop_event`` is set."""
        logger.info(
            "PositionWatcher running (poll=%.1fs, miss_threshold=%d)",
            self.poll_interval_sec, self.miss_threshold,
        )
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("PositionWatcher poll_once failed: %s", e)
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.poll_interval_sec,
                )
            except asyncio.TimeoutError:
                pass

    async def poll_once(self) -> list[Position]:
        """One poll cycle. Returns the list of positions we just closed.

        Detection logic:
            For every symbol in ``account.open_positions``:
              * if it appears in the snapshot with size > 0 -> reset miss
                counter.
              * else                                       -> increment.
            When the miss counter for a symbol reaches ``miss_threshold``
            we treat that position as closed: remove from
            ``account.open_positions`` and fire ``on_close``.
        """
        try:
            snapshot = await self.adapter.fetch_positions()
        except Exception as e:
            # Transient failure: do NOT advance miss counts (fail-safe —
            # we never want to falsely conclude that a position closed
            # because the exchange API is flaky).
            logger.warning("fetch_positions failed; skipping poll: %s", e)
            return []

        live_sizes = self._index_snapshot(snapshot)

        # Snapshot the keys before mutation so we can safely delete.
        tracked_symbols = list(self.account.open_positions.keys())
        closed: list[Position] = []

        for symbol in tracked_symbols:
            position = self.account.open_positions.get(symbol)
            if position is None:
                continue
            size = live_sizes.get(symbol, 0.0)
            if size > 0.0:
                self._miss_counts.pop(symbol, None)
                continue

            count = self._miss_counts.get(symbol, 0) + 1
            self._miss_counts[symbol] = count
            logger.info(
                "PositionWatcher: %s missing on exchange (%d/%d)",
                symbol, count, self.miss_threshold,
            )
            if count < self.miss_threshold:
                continue

            # Threshold reached -> the position is gone.
            self._miss_counts.pop(symbol, None)
            self.account.open_positions.pop(symbol, None)
            position.closed = True
            closed.append(position)
            # TICKET-004: prefer an executor-supplied hint when present.
            # ``pop`` so the same hint can't be reused on a future
            # re-entry of the same symbol.
            reason = self.close_reason_hints.pop(
                symbol, "exchange_close_detected",
            )
            try:
                await self.on_close(position, reason)
            except Exception as e:
                logger.exception(
                    "on_close handler failed for %s: %s",
                    symbol, e,
                )

        return closed

    @staticmethod
    def _index_snapshot(snapshot: list[dict[str, Any]]) -> dict[str, float]:
        """Build {symbol: size} from a fetch_positions snapshot. Treats
        ``contracts``/``size`` as the numeric size and absolute-values it
        because hedge-mode shorts come back negative."""
        out: dict[str, float] = {}
        for p in snapshot or []:
            symbol = str(p.get("symbol") or "")
            if not symbol:
                continue
            raw = p.get("contracts")
            if raw is None:
                raw = p.get("size")
            try:
                size = abs(float(raw or 0.0))
            except (TypeError, ValueError):
                size = 0.0
            # Aggregate in case hedge-mode returns two rows per symbol.
            out[symbol] = out.get(symbol, 0.0) + size
        return out
