"""kill_switch.py — operator-controlled hard halt (audit #25).

Background
----------
The only way to stop the daemon mid-day was to send SIGTERM to the
container. That kills the process and orphans positions. Operators
should be able to *halt new entries* while existing positions keep
their trailing stops alive — graceful, reversible, no SSH required.

Design
------
A simple file-watch primitive. Whenever the operator wants to halt:

    touch .kiro/state/HALT

The daemon's background ticker (run from ``main.App.run``) checks
the file every ``poll_sec`` and, when present, calls
``account.halt(reason)``. Once the file is removed, halts that
came from the kill switch are released; halts from other sources
(e.g. manual ops setting ``account.halt`` directly) stick because
the audit explicitly says manual halts must be sticky by design.

The reason string carries a marker (``KILL_SWITCH:``) so the release
path can tell its own halts apart from ones it shouldn't touch.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from altcoin_agent.risk.state import AccountState

logger = logging.getLogger(__name__)


_KILL_SWITCH_PREFIX = "KILL_SWITCH:"


@dataclass
class KillSwitchConfig:
    enabled: bool = True
    path: Path = Path(".kiro/state/HALT")
    poll_sec: float = 2.0


class KillSwitchWatcher:
    """File-presence watcher that toggles ``account.halt`` based on the
    existence of a sentinel file.

    Notification is delegated to a callable (so we can reuse the
    Notifier without coupling). Failures in the notifier are
    swallowed.
    """

    def __init__(
        self,
        cfg: KillSwitchConfig,
        account: AccountState,
        on_halt: Callable[[str], Awaitable[None]] | None = None,
        on_release: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.cfg = cfg
        self.account = account
        self.on_halt = on_halt
        self.on_release = on_release
        self._engaged = False

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self.cfg.enabled:
            return
        # Make sure parent dir exists so operators can ``touch`` even
        # before the daemon was first started.
        try:
            self.cfg.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover - filesystem edge case
            logger.warning("KillSwitchWatcher mkdir failed: %s", e)
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as e:
                logger.exception("KillSwitchWatcher poll_once failed: %s", e)
            try:
                await asyncio.wait_for(stop_event.wait(), self.cfg.poll_sec)
            except asyncio.TimeoutError:
                continue

    async def poll_once(self) -> None:
        present = os.path.exists(self.cfg.path)
        if present and not self._engaged:
            reason = (
                f"{_KILL_SWITCH_PREFIX}halt file present at "
                f"{self.cfg.path}"
            )
            self.account.halt(reason)
            self._engaged = True
            logger.warning(
                "Kill switch ENGAGED (operator file %s present); "
                "blocking new entries", self.cfg.path,
            )
            if self.on_halt is not None:
                try:
                    await self.on_halt(reason)
                except Exception as e:
                    logger.warning(
                        "kill switch on_halt notify swallowed: %s", e,
                    )
        elif not present and self._engaged:
            # Only release if the current halt reason came from us;
            # never undo a manual halt set by other code paths.
            cur_reason = self.account.halt_reason or ""
            if cur_reason.startswith(_KILL_SWITCH_PREFIX):
                self.account.global_trading_halted = False
                self.account.halt_reason = None
                logger.warning(
                    "Kill switch RELEASED (operator removed %s); "
                    "trading allowed again", self.cfg.path,
                )
                if self.on_release is not None:
                    try:
                        await self.on_release(cur_reason)
                    except Exception as e:
                        logger.warning(
                            "kill switch on_release notify swallowed: %s",
                            e,
                        )
            self._engaged = False
        elif not present:
            # File absent and we never engaged — nothing to do.
            return
        # else: present and engaged — steady state.
