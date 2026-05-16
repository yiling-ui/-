"""sqlite_persistence.py — SQLite + WAL replacement for AccountPersistor.

Phase B.3.1 of ``MISS_PENALTY_AND_PRODUCTION_PLAN.md``: the existing
``AccountPersistor`` writes the entire ``AccountState`` snapshot as a
single JSON file via tmp-file + ``os.replace``. That gives
**snapshot durability** but no WAL semantics: an audit reviewer who
asks "what was the equity 12 hours ago?" has no answer because each
snapshot overwrites the previous one.

This module ships a drop-in :class:`SQLiteAccountStore` with two
tables:

* ``state_snapshot``  — single row per ``schema_version`` with the
                        latest persistable fields. Updated on every
                        change. Provides O(1) restore at boot.

* ``state_log``       — append-only event log: one row per persistable
                        change, with ``ts``, ``event`` (free-form
                        category) and ``payload_json`` (the full
                        snapshot at that moment). Enables postmortem
                        reconstruction for any historical timestamp.

WAL mode (``PRAGMA journal_mode=WAL``) is enabled for crash safety —
SQLite guarantees that a successful ``COMMIT`` is durable even if the
process is SIGKILL'd mid-write. Tests can pin a ``:memory:`` database
to keep them hermetic.

Design notes
------------
* **Same call-shape as ``AccountPersistor``.** Both classes expose
  ``save(account)`` returning bool, ``restore_into(account)``
  returning bool, ``last_saved_ts`` and ``last_loaded_ts`` floats. The
  daemon picks one at startup based on ``cfg.persistence_backend``;
  the ``register_change_listener`` wiring on AccountState doesn't
  need to know which one it has.
* **Single connection, single thread.** SQLite-Python's connection is
  not thread-safe; we use a re-entrant lock so the asyncio loop's
  multiple coroutines (entry hot path + position-watcher + rollover
  worker) serialise cleanly. The daemon is single-threaded so this is
  cheap.
* **Failures swallowed.** ``save`` and ``restore_into`` log at
  WARNING and return False — the trading loop must keep running
  even if the disk is full.
* **Schema migrations.** ``schema_version`` lives in
  ``state_snapshot`` so a future change can detect "old layout" and
  migrate. V1 hardcodes version 1.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.state import AccountState

logger = logging.getLogger(__name__)


_SCHEMA_VERSION = 1


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS state_snapshot (
    schema_version INTEGER PRIMARY KEY,
    updated_at     REAL    NOT NULL,
    payload_json   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS state_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    event        TEXT    NOT NULL,
    payload_json TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_log_ts ON state_log (ts);
CREATE INDEX IF NOT EXISTS idx_state_log_event ON state_log (event);
"""


@dataclass
class SQLiteAccountStore:
    """SQLite-backed alternative to :class:`AccountPersistor`.

    Construction cost: opens the connection, applies WAL/synchronous
    pragmas, and runs the schema-creation SQL. Costs <5 ms on first
    boot; effectively free thereafter (file already exists).
    """

    path: Path
    last_saved_ts: float = 0.0
    last_loaded_ts: float = 0.0
    log_event_label: str = "snapshot"
    last_log_id: int = 0
    _conn: sqlite3.Connection | None = field(
        default=None, init=False, repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False,
    )
    _persistor: AccountPersistor | None = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        # We re-use the JSON ``AccountPersistor.to_dict`` payload format
        # so a single backup folder can contain BOTH the SQLite db AND
        # the previous JSON snapshot, and a future migration can read
        # either side. The instance is constructed without a path
        # because we never call its save/restore — only its dict
        # builders.
        self._persistor = AccountPersistor(
            path=Path(":sqlite-shim:"),
        ) if self.path else None
        # ``AccountPersistor.__post_init__`` mkdirs its parent. For our
        # shim path that creates a ``:sqlite-shim:`` directory in CWD,
        # which is harmless but ugly. We swap in our real path here so
        # any incidental mkdir in tests targets the right location.
        if self._persistor is not None:
            self._persistor.path = Path(self.path)
        self._open()

    # --------------------- connection lifecycle --------------------- #

    def _open(self) -> None:
        path = Path(self.path)
        if str(path) != ":memory:":
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover - fs edge
                logger.warning(
                    "SQLiteAccountStore: parent mkdir failed (%s): %s",
                    path.parent, e,
                )
        # ``check_same_thread=False`` because the daemon is asyncio
        # (one thread, many tasks) and we already serialise via
        # ``self._lock``. The default check_same_thread=True would
        # raise the moment a coroutine context-switches between two
        # awaits.
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None,
        )
        with self._lock:
            cur = self._conn.cursor()
            try:
                # WAL + NORMAL synchronous is the canonical "fast +
                # crash-safe" combo for embedded SQLite. NORMAL means
                # we accept that a power-cut might lose the most
                # recent transaction; the trading loop reconciles
                # against the exchange at boot anyway, so this is
                # acceptable.
                if str(path) != ":memory:":
                    cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.executescript(_SCHEMA_SQL)
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "SQLiteAccountStore.close failed: %s", e,
                    )
                self._conn = None

    # --------------------- save / restore --------------------- #

    def to_dict(self, account: AccountState) -> dict[str, Any]:
        """Build the same JSON payload ``AccountPersistor`` would.

        We delegate so the persistable-field set stays in lockstep
        with the JSON backend; if a future field is added there, it
        flows here automatically.
        """
        if self._persistor is None:  # pragma: no cover - constructed in __post_init__
            self._persistor = AccountPersistor(path=Path(self.path))
        return self._persistor.to_dict(account)

    def save(self, account: AccountState) -> bool:
        """Persist a fresh snapshot AND append an event-log row.

        Returns True on success. Failures are logged and swallowed.
        """
        try:
            payload = self.to_dict(account)
            payload_json = json.dumps(payload, sort_keys=True)
            now = time.time()
            with self._lock:
                if self._conn is None:
                    return False
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE")
                    cur.execute(
                        "INSERT OR REPLACE INTO state_snapshot "
                        "(schema_version, updated_at, payload_json) "
                        "VALUES (?, ?, ?)",
                        (_SCHEMA_VERSION, now, payload_json),
                    )
                    cur.execute(
                        "INSERT INTO state_log (ts, event, payload_json) "
                        "VALUES (?, ?, ?)",
                        (now, self.log_event_label, payload_json),
                    )
                    self.last_log_id = int(cur.lastrowid or 0)
                    cur.execute("COMMIT")
                finally:
                    cur.close()
            self.last_saved_ts = now
            return True
        except Exception as e:
            logger.warning(
                "SQLiteAccountStore.save failed (swallowed): %s", e,
            )
            try:
                with self._lock:
                    if self._conn is not None:
                        self._conn.execute("ROLLBACK")
            except Exception:
                pass
            return False

    def restore_into(self, account: AccountState) -> bool:
        """Mutate ``account`` in place with the latest snapshot.

        Returns True if a snapshot was found AND applied; False on
        first boot or corruption. Mirrors ``AccountPersistor`` exactly
        so the daemon's restore code is backend-agnostic.
        """
        try:
            with self._lock:
                if self._conn is None:
                    return False
                cur = self._conn.cursor()
                try:
                    cur.execute(
                        "SELECT payload_json FROM state_snapshot "
                        "WHERE schema_version = ? "
                        "ORDER BY updated_at DESC LIMIT 1",
                        (_SCHEMA_VERSION,),
                    )
                    row = cur.fetchone()
                finally:
                    cur.close()
            if row is None:
                return False
            data = json.loads(row[0])
        except Exception as e:
            logger.warning(
                "SQLiteAccountStore.restore_into failed; starting fresh: %s",
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
                "SQLiteAccountStore.restore_into: malformed snapshot, "
                "starting fresh: %s", e,
            )
            return False

    # --------------------- read helpers --------------------- #

    def history(
        self,
        *,
        since_ts: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` recent state-log rows.

        Each row has ``id``, ``ts``, ``event``, ``payload`` (parsed
        JSON). Used by tests and by an upcoming Phase B.6 OpenTelemetry
        path that walks the log to reconstruct PnL trajectories.
        """
        try:
            with self._lock:
                if self._conn is None:
                    return []
                cur = self._conn.cursor()
                try:
                    if since_ts is not None:
                        cur.execute(
                            "SELECT id, ts, event, payload_json "
                            "FROM state_log WHERE ts >= ? "
                            "ORDER BY ts DESC, id DESC LIMIT ?",
                            (since_ts, max(1, int(limit))),
                        )
                    else:
                        cur.execute(
                            "SELECT id, ts, event, payload_json "
                            "FROM state_log "
                            "ORDER BY ts DESC, id DESC LIMIT ?",
                            (max(1, int(limit)),),
                        )
                    rows = cur.fetchall()
                finally:
                    cur.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("SQLiteAccountStore.history failed: %s", e)
            return []

        out: list[dict[str, Any]] = []
        for row_id, ts, event, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = None
            out.append({
                "id": int(row_id),
                "ts": float(ts),
                "event": str(event),
                "payload": payload,
            })
        return out

    def log_size(self) -> int:
        """Return the number of rows in ``state_log``. Cheap (uses a
        COUNT, but the table is bounded by Phase B.3's retention
        policy)."""
        try:
            with self._lock:
                if self._conn is None:
                    return 0
                cur = self._conn.cursor()
                try:
                    cur.execute("SELECT COUNT(*) FROM state_log")
                    return int(cur.fetchone()[0])
                finally:
                    cur.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("SQLiteAccountStore.log_size failed: %s", e)
            return 0

    def vacuum_log(self, *, keep_last_n: int) -> int:
        """Trim ``state_log`` to the most-recent ``keep_last_n`` rows.

        Returns the number of rows removed. Designed to be called
        periodically (e.g. on daily rollover) so the log doesn't grow
        unbounded over months.
        """
        if keep_last_n <= 0:
            keep_last_n = 1
        try:
            with self._lock:
                if self._conn is None:
                    return 0
                cur = self._conn.cursor()
                try:
                    cur.execute(
                        "SELECT id FROM state_log "
                        "ORDER BY id DESC LIMIT 1 OFFSET ?",
                        (keep_last_n,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return 0
                    cutoff_id = int(row[0])
                    cur.execute(
                        "DELETE FROM state_log WHERE id <= ?",
                        (cutoff_id,),
                    )
                    return cur.rowcount or 0
                finally:
                    cur.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "SQLiteAccountStore.vacuum_log failed: %s", e,
            )
            return 0
