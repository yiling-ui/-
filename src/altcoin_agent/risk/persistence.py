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

TICKET-003 (persistence watchdog)
---------------------------------
The original ``save`` returned False on disk failure WITHOUT raising;
the trader continued with in-memory state diverging from disk. After
N consecutive save failures we now call ``account.halt(
"persistence_unavailable")`` so the gate refuses every further entry
until ops intervenes.

The original ``restore_into`` returned False on a corrupt JSON file
and let the daemon proceed with default zeros — exactly the silent
re-arm scenario the persistence layer was supposed to prevent. We now
flag the account as corrupt (sticky bool) and halt it; ``main.App``
aborts boot rather than continuing on a re-zeroed snapshot.

TICKET-001 / 015 (open_positions persistence)
---------------------------------------------
We now persist ``open_positions`` too — including each leg's
``client_order_id``. The Reconciler at boot remains the source of
truth ("what the venue actually shows"), but the persisted snapshot
gives it the cids it needs to re-attach a leg to its venue order via
``adapter.fetch_order(client_order_id=...)`` (and to call
``fetch_my_trades`` for accurate close-side fill prices). Without
this the Reconciler can only see a contracts-side total and has to
guess the original entry / stop, which destroys trailing-stop and
risk-per-trade math.

Design
------
* Atomic writes via tmp-file + ``os.replace`` so a crash mid-write
  never leaves a half-flushed file.
* Save is synchronous but <1ms for the typical payload; called from
  the hot path after each state mutation.
* Schema-versioned (``schema_version``); future migrations bump the
  version and ``restore_into`` declines to read older versions
  rather than misinterpreting them.

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

from altcoin_agent.risk.state import (
    AccountState,
    Position,
    PositionLeg,
    Side,
)

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 2  # bumped from 1 after open_positions added (TICKET-015)


@dataclass
class AccountPersistor:
    """Durable bookkeeping for daily-scoped AccountState fields.

    Usage:
        persistor = AccountPersistor(path=".kiro/state/account.json")
        persistor.restore_into(account)   # at boot, after AccountState is built
        ...
        persistor.save(account)            # after every PnL update

    TICKET-003 watchdog:
        ``max_consecutive_save_failures`` (default 3) — after that many
        back-to-back ``save`` failures we ``account.halt(
        "persistence_unavailable")``. Successful saves reset the counter.
    """

    path: Path
    last_saved_ts: float = 0.0
    last_loaded_ts: float = 0.0
    max_consecutive_save_failures: int = 3
    consecutive_save_failures: int = 0
    last_save_error: str | None = None
    last_load_error: str | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------- save --------------------- #

    def to_dict(self, account: AccountState) -> dict[str, Any]:
        """Snapshot the persistable fields of an AccountState.

        TICKET-015: includes ``open_positions`` with full leg metadata
        + cids so a reconciler restart can re-attach the venue-side
        orders without guessing.
        """
        return {
            "schema_version": SCHEMA_VERSION,
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
            "open_positions": {
                sym: _position_to_dict(p)
                for sym, p in account.open_positions.items()
            },
        }

    def save(self, account: AccountState) -> bool:
        """Write the snapshot atomically. Returns True on success.

        TICKET-003: failures bump ``consecutive_save_failures``; once
        the threshold is reached we halt the account so the gate
        refuses every further entry. The trading loop keeps running
        (we don't raise) but the gate is now a wall.
        """
        try:
            payload = self.to_dict(account)
            data = json.dumps(payload, indent=2, sort_keys=True, default=str)
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
            self.last_save_error = None
            if self.consecutive_save_failures > 0:
                logger.info(
                    "AccountPersistor.save recovered after %d failures",
                    self.consecutive_save_failures,
                )
            self.consecutive_save_failures = 0
            return True
        except Exception as e:
            self.consecutive_save_failures += 1
            self.last_save_error = f"{type(e).__name__}:{e}"
            logger.warning(
                "AccountPersistor.save failed (%d/%d): %s",
                self.consecutive_save_failures,
                self.max_consecutive_save_failures, e,
            )
            if (
                self.consecutive_save_failures
                >= self.max_consecutive_save_failures
                and not account.global_trading_halted
            ):
                logger.critical(
                    "AccountPersistor.save failed %d times in a row — "
                    "halting account (reason=persistence_unavailable). "
                    "In-memory state can no longer be relied upon to "
                    "survive a restart.",
                    self.consecutive_save_failures,
                )
                account.halt("persistence_unavailable")
            return False

    # --------------------- load --------------------- #

    def restore_into(self, account: AccountState) -> bool:
        """Mutate ``account`` in place with the on-disk snapshot.

        Returns True iff a snapshot was found and applied. On a missing
        file we return False (clean first boot). On a CORRUPT file
        TICKET-003 says fail-closed: we set ``account.account_state_corrupt
        = True`` and ``account.halt("account_state_corrupt")`` AND return
        False so the caller can refuse to start. The previous behaviour
        was to log a warning and silently re-arm at zero, which is the
        worst possible failure mode for the daily-DD breaker.

        TICKET-015: ``open_positions`` are restored as a *speculative
        cache*. The Reconciler at boot is still authoritative — it will
        verify each restored position against the venue and detach
        anything that no longer exists. Persisting them here gives the
        reconciler the cids it needs (so ``fetch_order`` /
        ``fetch_my_trades`` can resolve the venue-side state without
        guessing).
        """
        try:
            if not self.path.exists():
                return False
            text = self.path.read_text()
            if not text.strip():
                # Empty file == fresh disk == not corrupt.
                return False
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self.last_load_error = f"JSONDecodeError:{e}"
            logger.critical(
                "AccountPersistor.restore_into: ON-DISK SNAPSHOT IS "
                "CORRUPT (%s). Halting account; refusing to silently "
                "re-arm at zero — operator must inspect %s and either "
                "repair it or delete it after confirming PnL state.",
                e, self.path,
            )
            account.account_state_corrupt = True
            account.halt("account_state_corrupt")
            return False
        except OSError as e:
            self.last_load_error = f"{type(e).__name__}:{e}"
            logger.warning(
                "AccountPersistor.restore_into IO failure: %s — starting fresh",
                e,
            )
            return False

        try:
            schema = int(data.get("schema_version", 1))
            if schema > SCHEMA_VERSION:
                self.last_load_error = (
                    f"schema_version:{schema}>known:{SCHEMA_VERSION}"
                )
                logger.critical(
                    "AccountPersistor: snapshot schema_version=%d but this "
                    "build only knows up to %d — refusing to load (could "
                    "be a downgrade); halting.",
                    schema, SCHEMA_VERSION,
                )
                account.account_state_corrupt = True
                account.halt("account_state_corrupt")
                return False

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
            # Schema 2+: open_positions. Older snapshots silently miss
            # this; the Reconciler will rebuild from the venue.
            raw_open = data.get("open_positions") or {}
            if isinstance(raw_open, dict):
                for sym, raw_pos in raw_open.items():
                    try:
                        pos = _position_from_dict(raw_pos)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "AccountPersistor: skipping malformed open_position "
                            "%s: %s", sym, e,
                        )
                        continue
                    account.open_positions[sym] = pos
            self.last_loaded_ts = time.time()
            self.last_load_error = None
            return True
        except Exception as e:  # noqa: BLE001
            self.last_load_error = f"{type(e).__name__}:{e}"
            logger.critical(
                "AccountPersistor.restore_into: malformed snapshot "
                "structure (%s). Halting account.", e,
            )
            account.account_state_corrupt = True
            account.halt("account_state_corrupt")
            return False


# --------------------- helpers --------------------- #


def _position_to_dict(p: Position) -> dict[str, Any]:
    """Serialise a ``Position`` (incl. legs) to a JSON-safe dict.

    TICKET-015: includes cid metadata so a reconciler restart can
    re-attach venue-side orders without re-issuing them.
    """
    return {
        "symbol": p.symbol,
        "exchange": p.exchange,
        "side": p.side.value,
        "entry_price": float(p.entry_price),
        "size": float(p.size),
        "leverage": float(p.leverage),
        "initial_stop": float(p.initial_stop),
        "current_stop": float(p.current_stop),
        "stop_order_id": p.stop_order_id,
        "client_order_id": p.client_order_id,
        "stop_client_order_id": p.stop_client_order_id,
        "opened_at_ts_ms": int(p.opened_at_ts_ms),
        "trace_id": p.trace_id,
        "closed": bool(p.closed),
        "legs": [
            {
                "leg_id": int(L.leg_id),
                "side": L.side.value,
                "size": float(L.size),
                "entry_price": float(L.entry_price),
                "entry_ts_ms": int(L.entry_ts_ms),
                "margin_source": L.margin_source,
                "trigger_score": (
                    float(L.trigger_score)
                    if L.trigger_score is not None else None
                ),
                "client_order_id": L.client_order_id,
            }
            for L in p.legs
        ],
    }


def _position_from_dict(d: dict[str, Any]) -> Position:
    """Inverse of ``_position_to_dict``. Raises on malformed input."""
    side = Side(d["side"])
    pos = Position(
        symbol=str(d["symbol"]),
        exchange=str(d.get("exchange") or "binance"),
        side=side,
        entry_price=float(d["entry_price"]),
        size=float(d["size"]),
        leverage=float(d["leverage"]),
        initial_stop=float(d["initial_stop"]),
        current_stop=float(d["current_stop"]),
        stop_order_id=d.get("stop_order_id"),
        client_order_id=d.get("client_order_id"),
        stop_client_order_id=d.get("stop_client_order_id"),
        opened_at_ts_ms=int(d.get("opened_at_ts_ms") or int(time.time() * 1000)),
        trace_id=d.get("trace_id"),
        closed=bool(d.get("closed", False)),
    )
    for raw_leg in (d.get("legs") or []):
        pos.legs.append(PositionLeg(
            leg_id=int(raw_leg["leg_id"]),
            side=Side(raw_leg["side"]),
            size=float(raw_leg["size"]),
            entry_price=float(raw_leg["entry_price"]),
            entry_ts_ms=int(raw_leg.get("entry_ts_ms") or int(time.time() * 1000)),
            margin_source=str(raw_leg.get("margin_source") or "initial"),
            trigger_score=(
                float(raw_leg["trigger_score"])
                if raw_leg.get("trigger_score") is not None else None
            ),
            client_order_id=raw_leg.get("client_order_id"),
        ))
    return pos


class _suppress_errors:
    def __enter__(self) -> _suppress_errors:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True
