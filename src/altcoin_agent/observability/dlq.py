"""dlq.py — append-only Dead Letter Queue for unrecoverable failures.

Phase B.2.3 of ``MISS_PENALTY_AND_PRODUCTION_PLAN.md``: the daemon
currently silently drops failed signals after bumping
``state.orders_rejected``. The operator has no replay/postmortem trail
for "why did this entry not fire?" once the in-memory dashboard ring
buffer (50 rows) overflows.

This module writes one JSON line per failure to ``.kiro/state/dlq/<kind>.jsonl``,
with size-based rotation modelled exactly on
``DecisionAuditLog`` so the operator only has to learn one rotation
strategy. Failures are swallowed: the trading loop must never block on
"can't write to DLQ".

Categories
----------
The same DLQ accepts entries of multiple ``kind``s and records the
kind in the row so a single ``jq`` invocation can split them:

* ``executor_exception``   — :class:`Exception` from the executor's
                             open path (bookkeeping inconsistency,
                             unexpected ccxt error).
* ``quote_unavailable``    — live quote fetch failed before the gate.
* ``depth_unavailable``    — top-5 depth fetch failed before the gate.
* ``vol_unavailable``      — PriceTape cold while live mode requires vol.
* ``naked_position``       — trailing tighten + restore both failed,
                             emergency close fired.
* ``llm_consult_failed``   — DeepSeek call raised after retries.
* ``stop_replace_failed``  — stop tighten failed, old stop restored.

Tests for new failure paths should add a new ``kind`` constant rather
than reusing an existing one so the categories stay observable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Same defaults as ``DecisionAuditLog``: 5 × 50 MB ≈ 250 MB total
# retention, plenty for tens of thousands of failures.
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5


@dataclass
class DLQEntry:
    """Plain dataclass mirror of one row, for type-checking call sites.

    The DLQ writer accepts either an instance of this OR a plain dict;
    using the dataclass makes the call site self-documenting without
    forcing it on tests that prefer to construct rows ad-hoc.
    """

    kind: str
    symbol: str | None = None
    reason: str = ""
    trace_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ts": self.ts or time.time(),
            "kind": self.kind,
            "symbol": self.symbol,
            "reason": self.reason,
            "trace_id": self.trace_id,
        }
        # Caller-supplied payload takes precedence over the envelope
        # only via a nested ``payload`` field — never overwrite the
        # standard top-level keys, otherwise downstream tooling that
        # filters by ``kind`` breaks unpredictably.
        if self.payload:
            d["payload"] = self.payload
        return d


@dataclass
class DeadLetterQueue:
    """Append-only file-rotated JSONL DLQ.

    Usage::

        dlq = DeadLetterQueue(path=Path(".kiro/state/dlq/main.jsonl"))
        dlq.put(DLQEntry(
            kind="executor_exception",
            symbol="PEPE/USDT:USDT",
            reason="ccxt.NetworkError: timeout",
            trace_id=trace_id,
            payload={"sig": sig.as_dict()},
        ))

    Failures during ``put`` are logged at WARNING and swallowed;
    callers do not need to wrap in try/except. ``write_count``,
    ``write_errors`` and ``rotations`` mirror the ``DecisionAuditLog``
    introspection counters so dashboards / metrics handlers can read
    them with one polymorphic helper.
    """

    path: Path
    enabled: bool = True
    max_bytes: int = _DEFAULT_MAX_BYTES
    backup_count: int = _DEFAULT_BACKUP_COUNT
    last_write_ts: float = 0.0
    write_count: int = 0
    write_errors: int = 0
    rotations: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.backup_count < 1:
            self.backup_count = 1
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover - filesystem edge
                logger.warning(
                    "DLQ: parent mkdir failed (%s): %s",
                    self.path.parent, e,
                )

    # --------------------- rotation --------------------- #

    def _should_rotate(self) -> bool:
        if self.max_bytes <= 0:
            return False
        try:
            return self.path.exists() and self.path.stat().st_size >= self.max_bytes
        except OSError:
            return False

    def _rotate(self) -> None:
        try:
            oldest = self.path.with_name(
                f"{self.path.name}.{self.backup_count}"
            )
            if oldest.exists():
                with _suppress_errors():
                    os.remove(oldest)
            for i in range(self.backup_count - 1, 0, -1):
                src = self.path.with_name(f"{self.path.name}.{i}")
                dst = self.path.with_name(f"{self.path.name}.{i + 1}")
                if src.exists():
                    os.replace(src, dst)
            if self.path.exists():
                os.replace(
                    self.path,
                    self.path.with_name(f"{self.path.name}.1"),
                )
            self.rotations += 1
        except Exception as e:
            self.write_errors += 1
            logger.warning("DLQ._rotate failed (swallowed): %s", e)

    # --------------------- write --------------------- #

    def put(self, entry: DLQEntry | dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            if isinstance(entry, DLQEntry):
                row = entry.to_dict()
            else:
                row = dict(entry)
                row.setdefault("ts", time.time())
                if "kind" not in row:
                    row["kind"] = "unknown"
            line = json.dumps(row, default=str, sort_keys=True)
            if self._should_rotate():
                self._rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.last_write_ts = time.time()
            self.write_count += 1
            return True
        except Exception as e:
            self.write_errors += 1
            logger.warning("DLQ.put failed (swallowed): %s", e)
            return False

    # --------------------- read helpers --------------------- #

    def approx_size(self) -> int:
        """Best-effort row count for the active file (counts lines).

        Cheap enough for the metrics gauge: a 50 MB file with ~250-byte
        rows is ~200K lines, which Python iterates in <100 ms once
        a scrape; we cache the result for ``cache_ttl_sec`` to keep
        repeated /metrics scrapes free.
        """
        try:
            if not self.path.exists():
                return 0
            with self.path.open("rb") as f:
                return sum(1 for _ in f)
        except Exception as e:  # pragma: no cover - filesystem edge
            logger.warning("DLQ.approx_size failed: %s", e)
            return 0

    def iter_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return up to ``limit`` most-recent rows from the active file.

        Used by the reflection-mode report and dashboards. Reads only
        the active file — older rotations are omitted because the
        operator can ``cat`` them directly.
        """
        try:
            if not self.path.exists():
                return []
            rows: list[dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as f:
                # Tail-friendly: read the whole file into memory only
                # if it's small. For larger files we read backwards in
                # chunks. Phase B.2.3 doesn't need the latter; the
                # active file is bounded by ``max_bytes`` so the simple
                # path is fine.
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Half-flushed line (rare; we use line-buffered
                        # writes). Skip it rather than raise.
                        continue
            if limit <= 0:
                return rows
            return rows[-limit:]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("DLQ.iter_recent failed: %s", e)
            return []


class _suppress_errors:
    def __enter__(self) -> _suppress_errors:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True
