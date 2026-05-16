"""tests/test_sqlite_persistence_mock.py — Phase B.3.1 SQLite store tests.

Covers:

* ``save`` + ``restore_into`` round trip for every persistable field on
  ``AccountState``.
* ``state_log`` accumulates one row per save (event-driven persistence
  hook of Phase B.1.3 produces the same shape via this backend).
* ``history(since_ts=…)`` filters on the timestamp column.
* WAL mode is set on a real file-backed DB.
* ``vacuum_log`` trims to the most-recent N rows.
* Corruption / missing-snapshot paths return False from
  ``restore_into`` instead of raising.
* The change-listener hook on ``AccountState`` fires the SQLite save
  end-to-end via ``set_cooldown`` / ``record_pnl`` / ``halt`` —
  exercising the same integration the daemon uses.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from altcoin_agent.risk.sqlite_persistence import SQLiteAccountStore
from altcoin_agent.risk.state import AccountState


def _make_account() -> AccountState:
    a = AccountState(
        equity_usdt=12_345.67,
        starting_equity_today_usdt=10_000.0,
    )
    a.realized_pnl_today_usdt = -123.45
    a.daily_stoploss_hits = 2
    a.consecutive_losses = {"PEPE/USDT:USDT": 3}
    a.cooldown_until_ts_ms = {"WIF/USDT:USDT": 1_700_000_000_000}
    a.global_trading_halted = True
    a.halt_reason = "manual"
    a.last_rollover_date_utc = "2026-05-15"
    return a


def test_save_then_restore_round_trip(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    a = _make_account()
    assert store.save(a)

    fresh = AccountState()
    assert store.restore_into(fresh)
    assert fresh.equity_usdt == 12_345.67
    assert fresh.realized_pnl_today_usdt == -123.45
    assert fresh.daily_stoploss_hits == 2
    assert fresh.consecutive_losses == {"PEPE/USDT:USDT": 3}
    assert fresh.cooldown_until_ts_ms == {"WIF/USDT:USDT": 1_700_000_000_000}
    assert fresh.global_trading_halted is True
    assert fresh.halt_reason == "manual"
    assert fresh.last_rollover_date_utc == "2026-05-15"
    store.close()


def test_state_log_accumulates_one_row_per_save(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    a = _make_account()
    for _ in range(4):
        assert store.save(a)
    rows = store.history()
    assert len(rows) == 4
    # Default sort is most-recent first.
    assert rows[0]["id"] > rows[-1]["id"]
    assert rows[0]["event"] == "snapshot"
    # Payload is parsed JSON, not a string.
    assert rows[0]["payload"]["equity_usdt"] == 12_345.67
    store.close()


def test_history_filter_by_since_ts(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    a = _make_account()
    store.save(a)
    cutoff = store.last_saved_ts + 0.01
    # Sleep past the cutoff.
    import time
    time.sleep(0.02)
    store.save(a)
    after = store.history(since_ts=cutoff)
    assert len(after) == 1


def test_restore_returns_false_on_empty_db(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    assert store.restore_into(AccountState()) is False


def test_wal_mode_is_set_on_file_db(tmp_path: Path) -> None:
    p = tmp_path / "account.sqlite3"
    store = SQLiteAccountStore(path=p)
    store.save(_make_account())
    store.close()
    # Re-open with a plain sqlite3 connection and verify journal_mode.
    conn = sqlite3.connect(str(p))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_in_memory_db_works_without_path() -> None:
    store = SQLiteAccountStore(path=Path(":memory:"))
    assert store.save(_make_account())
    fresh = AccountState()
    assert store.restore_into(fresh)
    assert fresh.equity_usdt == 12_345.67
    store.close()


def test_vacuum_log_trims_oldest(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    a = _make_account()
    for _ in range(10):
        store.save(a)
    assert store.log_size() == 10
    removed = store.vacuum_log(keep_last_n=3)
    assert removed == 7
    assert store.log_size() == 3
    store.close()


def test_change_listener_integration(tmp_path: Path) -> None:
    """End-to-end: AccountState mutators trigger SQLite saves via
    ``register_change_listener``. Mirrors how main.App.run wires it up.
    """
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    a = AccountState(equity_usdt=10_000.0)
    a.register_change_listener(lambda acct, _s=store: _s.save(acct))
    a.set_cooldown("PEPE", duration_sec=600, now_ms=1_700_000_000_000)
    a.record_pnl("PEPE", -50.0, is_loss=True)
    a.halt("manual_halt")

    fresh = AccountState()
    assert store.restore_into(fresh)
    assert fresh.daily_stoploss_hits == 1
    assert fresh.global_trading_halted is True
    assert fresh.halt_reason == "manual_halt"
    assert "PEPE" in fresh.consecutive_losses
    # 3 saves -> 3 log rows.
    assert store.log_size() == 3
    store.close()


def test_corrupt_payload_is_not_fatal(tmp_path: Path) -> None:
    """If somehow the snapshot row contains malformed JSON,
    ``restore_into`` returns False instead of raising.
    """
    p = tmp_path / "account.sqlite3"
    store = SQLiteAccountStore(path=p)
    store.save(_make_account())
    # Corrupt the snapshot JSON.
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "UPDATE state_snapshot SET payload_json = ?",
            ("{not valid json",),
        )
        conn.commit()
    finally:
        conn.close()
    fresh = AccountState()
    assert store.restore_into(fresh) is False
    store.close()


def test_save_returns_false_after_close(tmp_path: Path) -> None:
    store = SQLiteAccountStore(path=tmp_path / "account.sqlite3")
    store.close()
    assert store.save(_make_account()) is False
