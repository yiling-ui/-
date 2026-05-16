"""tests/test_dlq_mock.py — Phase B.2.3 dead-letter-queue tests.

Covers:

* Round trip: ``put`` -> ``iter_recent``.
* Both ``DLQEntry`` dataclass and raw-dict shapes are accepted.
* Rotation triggers when active file size crosses ``max_bytes`` and
  preserves the most recent N backups.
* ``approx_size`` returns the active-file row count.
* Disabled flag short-circuits all writes.
* Failures inside ``put`` (unwritable path) increment ``write_errors``
  and never raise.
"""

from __future__ import annotations

import json
from pathlib import Path

from altcoin_agent.observability.dlq import DeadLetterQueue, DLQEntry


def test_dlq_round_trip_dataclass(tmp_path: Path) -> None:
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    dlq.put(DLQEntry(
        kind="executor_exception",
        symbol="PEPE/USDT:USDT",
        reason="ccxt.NetworkError: timeout",
        trace_id="abc",
        payload={"size": 1.0},
    ))
    rows = dlq.iter_recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "executor_exception"
    assert row["symbol"] == "PEPE/USDT:USDT"
    assert row["trace_id"] == "abc"
    assert row["reason"].startswith("ccxt.NetworkError")
    assert row["payload"] == {"size": 1.0}


def test_dlq_round_trip_raw_dict(tmp_path: Path) -> None:
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    dlq.put({"kind": "quote_unavailable", "symbol": "WIF", "reason": "x"})
    rows = dlq.iter_recent()
    assert rows[0]["kind"] == "quote_unavailable"
    assert rows[0]["symbol"] == "WIF"


def test_dlq_disabled_flag_skips_writes(tmp_path: Path) -> None:
    p = tmp_path / "dlq.jsonl"
    dlq = DeadLetterQueue(path=p, enabled=False)
    ok = dlq.put(DLQEntry(kind="x", symbol="Y", reason="z"))
    assert ok is False
    assert not p.exists()


def test_dlq_write_count_and_last_ts(tmp_path: Path) -> None:
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    for i in range(3):
        assert dlq.put(DLQEntry(kind=f"k{i}", reason=str(i)))
    assert dlq.write_count == 3
    assert dlq.last_write_ts > 0


def test_dlq_rotation_when_active_file_exceeds_max_bytes(
    tmp_path: Path,
) -> None:
    p = tmp_path / "dlq.jsonl"
    # Tiny rotation threshold so a few rows suffice.
    dlq = DeadLetterQueue(
        path=p, max_bytes=200, backup_count=2,
    )
    # Each row is ~120 bytes; writing 5 should trigger ≥ 1 rotation.
    for i in range(5):
        dlq.put(DLQEntry(
            kind="k", symbol=f"S{i}", reason="x" * 50,
        ))
    assert dlq.rotations >= 1
    assert p.exists()
    # ``.1`` always exists after the first rotation; ``.2`` after two.
    assert (tmp_path / "dlq.jsonl.1").exists()


def test_dlq_rotation_drops_oldest_when_backup_full(
    tmp_path: Path,
) -> None:
    p = tmp_path / "dlq.jsonl"
    dlq = DeadLetterQueue(
        path=p, max_bytes=120, backup_count=2,
    )
    # Force many rotations.
    for i in range(20):
        dlq.put(DLQEntry(kind=f"k{i}", reason="x" * 80))
    # No more than backup_count + 1 files should exist.
    files = sorted(tmp_path.glob("dlq.jsonl*"))
    # active + .1 + .2 = 3 files at most
    assert len(files) <= 3


def test_dlq_approx_size_counts_lines(tmp_path: Path) -> None:
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    for _ in range(7):
        dlq.put(DLQEntry(kind="k"))
    assert dlq.approx_size() == 7


def test_dlq_iter_recent_limit(tmp_path: Path) -> None:
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    for i in range(10):
        dlq.put(DLQEntry(kind=f"k{i}"))
    last3 = dlq.iter_recent(limit=3)
    assert len(last3) == 3
    assert [r["kind"] for r in last3] == ["k7", "k8", "k9"]


def test_dlq_iter_recent_skips_corrupt_lines(tmp_path: Path) -> None:
    p = tmp_path / "dlq.jsonl"
    dlq = DeadLetterQueue(path=p)
    dlq.put(DLQEntry(kind="ok", reason="r"))
    # Append a half-flushed line manually.
    with p.open("a") as f:
        f.write("{not valid json\n")
    dlq.put(DLQEntry(kind="ok2", reason="r2"))
    rows = dlq.iter_recent()
    assert [r["kind"] for r in rows] == ["ok", "ok2"]


def test_dlq_emits_well_formed_json_lines(tmp_path: Path) -> None:
    p = tmp_path / "dlq.jsonl"
    dlq = DeadLetterQueue(path=p)
    dlq.put(DLQEntry(kind="k", reason="r", trace_id="t"))
    text = p.read_text().strip()
    obj = json.loads(text)
    assert obj["kind"] == "k"
    assert obj["trace_id"] == "t"
    assert "ts" in obj


def test_dlq_swallows_write_errors_when_path_invalid() -> None:
    # ``/dev/full`` would be ideal but isn't portable; instead we
    # construct a queue rooted at a path that we then turn into a
    # file (so opening as a directory fails).
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        # Create a regular file so the queue's parent.mkdir succeeds
        # but writing to ``./dlq.jsonl`` underneath it fails.
        bad_parent = os.path.join(tmp, "bad")
        with open(bad_parent, "w") as f:
            f.write("not a directory")
        # Path uses ``bad`` as parent dir, which is actually a file.
        dlq = DeadLetterQueue(path=Path(tmp) / "bad" / "dlq.jsonl")
        ok = dlq.put(DLQEntry(kind="k"))
        assert ok is False
        assert dlq.write_errors == 1
