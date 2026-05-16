"""persistence.py — AccountState durability across restarts (audit #12).

Background
----------
Without persistence, every restart resets ``realized_pnl_today_usdt = 0``,
``daily_stoploss_hits = 0``, and ``consecutive_losses = {}``. The
worst-case scenario the audit called out: an early-session -5% PnL gets
the daemon close to the daily-DD breaker, the process is restarted
(OOM, k8s reschedule, SIGTERM during deploy), and the now-fresh account
is allowed to lose ANOTHER 6% before the breaker fires. End-of-day
realised drawdown can therefore reach 11%+ even though the operator
configured a 6% cap.

Design
------
* Snapshot fields that the audit specifically cares about
  (today's PnL, equity, stoploss hits, consec losses, cooldowns,
  rollover stamp). We DO NOT persist ``open_positions`` — those are
  the exchange's source of truth and recovered by the Reconciler at
  startup.
* Atomic writes via tmp-file + ``os.replace`` so a crash mid-write
  never leaves a half-flushed file.
* Save is synchronous but <1ms for the small payload; called from the
  hot path after each state mutation. For dashboards / metrics, we
  also expose ``last_saved_ts``.
* On load we tolerate missing/corrupt files by returning a fresh
  state and logging a warning. Operators see the warning in
  ``last_error`` via the dashboard.

Out of scope
------------
* Multi-process locking (single daemon assumption).
* Encryption at rest (this is account *bookkeeping*, not credentials).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from altcoin_agent.risk.state import AccountState

logger = logging.getLogger(__name__)


@dataclass
class AccountPersistor:
    """Durable bookkeeping for daily-scoped AccountState fields.

    Usage:
        persistor = AccountPersistor(path=".kiro/state/account.json")
        persistor.restore_into(account)   # at boot, after AccountState is built
        ...
        persistor.save(account)            # after every PnL update
    """

    path: Path
    last_saved_ts: float = 0.0
    last_loaded_ts: float = 0.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------- save --------------------- #

    def to_dict(self, account: AccountState) -> dict[str, Any]:
        """Snapshot the persistable fields of an AccountState."""
        return {
            "schema_version": 1,
            "saved_at": time.time(),
            "equity_usdt": float(account.equity_usdt),
            "starting_equity_today_usdt": float(
                account.starting_equity_today_usdt
            ),
            "realized_pnl_today_usdt": float(
                account.realized_pnl_today_usdt
            ),
            "daily_stoploss_hits": int(account.daily_stoploss_hits),
            "consecutive_losses": {
                str(k): int(v)
                for k, v in account.consecutive_losses.items()
            },
            "cooldown_until_ts_ms": {
                str(k): int(v)
                for k, v in account.cooldown_until_ts_ms.items()
            },
            "global_trading_halted": bool(account.global_trading_halted),
            "halt_reason": account.halt_reason,
            "last_rollover_date_utc": account.last_rollover_date_utc,
            "rollover_anchor_utc_hour": int(
                account.rollover_anchor_utc_hour
            ),
        }

    def save(self, account: AccountState) -> bool:
        """Write the snapshot atomically. Returns True on success.

        Failures are logged but never raised; persistence is a defence
        in depth, not a correctness invariant. The trading loop must
        keep running even if the disk is full.
        """
        try:
            payload = self.to_dict(account)
            data = json.dumps(payload, indent=2, sort_keys=True)
            # Atomic write: tmp file in same directory, then os.replace.
            fd, tmp_path = tempfile.mkstemp(
                prefix=".account.", suffix=".tmp", dir=str(self.path.parent),
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(data)
                os.replace(tmp_path, self.path)
            finally:
                # If replace succeeded the tmp path is gone; otherwise
                # clean it up so we don't leak files in /tmp-style dirs.
                if os.path.exists(tmp_path):
                    with _suppress_errors():
                        os.remove(tmp_path)
            self.last_saved_ts = time.time()
            return True
        except Exception as e:
            logger.warning(
                "AccountPersistor.save failed (swallowed): %s", e,
            )
            return False

    # --------------------- load --------------------- #

    def restore_into(self, account: AccountState) -> bool:
        """Mutate ``account`` in place with the on-disk snapshot.

        Returns True iff a snapshot was found and applied. On corrupt
        / missing file returns False; the caller may then proceed
        with whatever defaults the AccountState was constructed with.

        We deliberately do NOT touch ``open_positions`` — the
        Reconciler is the sole source of truth there. Persisting that
        dict and "restoring" it would create a phantom-position class
        of bug if the exchange has since closed any of them.
        """
        try:
            if not self.path.exists():
                return False
            data = json.loads(self.path.read_text())
        except Exception as e:
            logger.warning(
                "AccountPersistor.restore_into failed; starting fresh: %s",
                e,
            )
            return False

        try:
            account.equity_usdt = float(data.get(
                "equity_usdt", account.equity_usdt,
            ))
            account.starting_equity_today_usdt = float(data.get(
                "starting_equity_today_usdt",
                account.starting_equity_today_usdt,
            ))
            account.realized_pnl_today_usdt = float(data.get(
                "realized_pnl_today_usdt", 0.0,
            ))
            account.daily_stoploss_hits = int(data.get(
                "daily_stoploss_hits", 0,
            ))
            account.consecutive_losses = {
                str(k): int(v)
                for k, v in (data.get("consecutive_losses") or {}).items()
            }
            account.cooldown_until_ts_ms = {
                str(k): int(v)
                for k, v in (data.get("cooldown_until_ts_ms") or {}).items()
            }
            account.global_trading_halted = bool(data.get(
                "global_trading_halted", False,
            ))
            account.halt_reason = data.get("halt_reason")
            account.last_rollover_date_utc = data.get(
                "last_rollover_date_utc",
            )
            self.last_loaded_ts = time.time()
            return True
        except Exception as e:
            logger.warning(
                "AccountPersistor.restore_into: malformed snapshot, "
                "starting fresh: %s", e,
            )
            return False


# --------------------- helpers --------------------- #


class _suppress_errors:
    def __enter__(self) -> _suppress_errors:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True
