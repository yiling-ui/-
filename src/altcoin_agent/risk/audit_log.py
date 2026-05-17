"""audit_log.py — append-only RiskGate decision log (audit #28).

Background
----------
``RiskGate.evaluate`` returns a ``RiskDecision`` and the daemon
``logger.info(...)`` it. ``logger.info`` goes to stdout; under
``docker logs`` it gets rotated by the daemon (default ~10MB) and lost.
When a position closes badly an operator (or compliance) needs to be
able to answer "why did the gate let this trade through?" and "what
features did the LLM pick when it agreed with the rules?". stdout
rotation makes that impossible after a few hours of busy day.

Design
------
* JSONL file under ``logs/decisions.jsonl`` by default.
* Each line is a self-contained JSON object: trace id, ts, signal
  fingerprint, gate inputs (current_price, depth, vol), gate
  decision (approved / reason / leverage / size / notional). Non-JSON-
  serialisable values are coerced to ``str`` defensively.
* Append-only — but with a size-based self-rotation so a long-running
  daemon does NOT grow the file unboundedly. When the active file
  exceeds ``max_bytes`` we rename it to ``decisions.jsonl.1``,
  shifting the prior rotations up to ``decisions.jsonl.N``, and start
  a fresh active file. Anything beyond ``backup_count`` is removed.
  Operators that prefer external rotation (logrotate / fluent-bit
  sidecar) can disable this by passing ``max_bytes=0``.
* Failures are swallowed; we never block the trading loop because
  the audit log can't write.

Audit (P2 #13) note on persistence
----------------------------------
``logs/decisions.jsonl`` lives under the ``./logs`` directory which
``docker-compose.yml`` bind-mounts into the container. ``.dockerignore``
excludes ``logs`` from the *build* context (so historical logs aren't
baked into the image), but the bind mount means the file is on the
host and DOES survive container restarts. The audit reported the
opposite — please re-read ``docker-compose.yml`` if reviewing this
again.

Out of scope
------------
* Cryptographic signing / tamper-proofing.
* Forwarding to a central log store (do that from the file).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Defaults chosen so a busy daemon (a few signals/sec) keeps roughly a
# day of history on hand without unbounded growth: 5 * 50 MB = 250 MB.
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5


@dataclass
class DecisionAuditLog:
    """Append-only sink for gate decisions with size-based rotation."""

    path: Path
    enabled: bool = True
    # Audit P2 #13: bounded self-rotation. ``max_bytes <= 0`` disables
    # the in-process rotation entirely (operator delegates to logrotate).
    max_bytes: int = _DEFAULT_MAX_BYTES
    backup_count: int = _DEFAULT_BACKUP_COUNT
    last_write_ts: float = 0.0
    write_count: int = 0
    write_errors: int = 0
    rotations: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        # Defensive coercion so an operator passing 0 / negative
        # backup_count never deletes the active file in ``_rotate``.
        if self.backup_count < 1:
            self.backup_count = 1
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover - filesystem edge case
                # The path may itself be a directory (some test paths)
                # or the parent may be unwritable. Either way we keep
                # ``enabled`` True so ``record`` returns a clean False
                # via its own try/except, preserving prior contract.
                logger.warning(
                    "DecisionAuditLog: parent mkdir failed (%s): %s",
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
        """Shift ``decisions.jsonl[.N]`` up by one, dropping the oldest.

        Failures are swallowed: a rotation problem must never crash the
        trading loop. If rotation fails the active file simply keeps
        growing — the operator can step in via external tooling.
        """
        try:
            # Drop the oldest backup if it exists.
            oldest = self.path.with_name(
                f"{self.path.name}.{self.backup_count}"
            )
            if oldest.exists():
                with _suppress_errors():
                    os.remove(oldest)
            # Shift backups N-1 -> N, N-2 -> N-1, ...
            for i in range(self.backup_count - 1, 0, -1):
                src = self.path.with_name(f"{self.path.name}.{i}")
                dst = self.path.with_name(f"{self.path.name}.{i + 1}")
                if src.exists():
                    os.replace(src, dst)
            # Rename active -> .1.
            if self.path.exists():
                os.replace(
                    self.path,
                    self.path.with_name(f"{self.path.name}.1"),
                )
            self.rotations += 1
        except Exception as e:
            self.write_errors += 1
            logger.warning(
                "DecisionAuditLog._rotate failed (swallowed): %s", e,
            )

    def record(self, entry: dict[str, Any]) -> bool:
        """Append a single JSON line. Returns True on success."""
        if not self.enabled:
            return False
        try:
            # Audit (third pass) #9: previously used ``entry.pop("ts", ...)``
            # which mutated the caller's dict — second invocation with the
            # same dict (e.g. the dashboard's signal payload) would silently
            # drop its ``ts`` field. Build the row from a copy instead so
            # callers can pass aliased dicts safely.
            ts_val = entry.get("ts", time.time())
            row = {"ts": ts_val, **{k: v for k, v in entry.items() if k != "ts"}}
            line = json.dumps(row, default=str, sort_keys=True)
            # Audit P2 #13: roll over BEFORE writing so the line that
            # tipped the file over the threshold lands in the fresh
            # active file (keeps each rotation self-consistent).
            if self._should_rotate():
                self._rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.last_write_ts = time.time()
            self.write_count += 1
            return True
        except Exception as e:
            self.write_errors += 1
            logger.warning("DecisionAuditLog.record failed (swallowed): %s", e)
            return False

    def record_decision(
        self,
        *,
        trace_id: str | None,
        symbol: str,
        signal_kind: str,
        rule_score: float,
        final_score: float,
        direction: str,
        approved: bool,
        reason: str,
        leverage: float | None,
        size: float | None,
        notional_usdt: float | None,
        current_price: float | None,
        top5_depth_usdt: float | None,
        realized_vol_pct: float | None,
        initial_stop: float | None,
        max_slippage_used: float | None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Convenience helper to keep call sites short."""
        entry: dict[str, Any] = {
            "trace_id": trace_id,
            "symbol": symbol,
            "signal_kind": signal_kind,
            "direction": direction,
            "rule_score": rule_score,
            "final_score": final_score,
            "approved": approved,
            "reason": reason,
            "leverage": leverage,
            "size": size,
            "notional_usdt": notional_usdt,
            "current_price": current_price,
            "top5_depth_usdt": top5_depth_usdt,
            "realized_vol_pct": realized_vol_pct,
            "initial_stop": initial_stop,
            "max_slippage_used": max_slippage_used,
        }
        if extra:
            for k, v in extra.items():
                if k not in entry:
                    entry[k] = v
        return self.record(entry)


# --------------------- helpers --------------------- #


class _suppress_errors:
    """Context manager that swallows any exception in its body.

    We define our own instead of ``contextlib.suppress`` so the rotate
    helper stays self-contained with zero new imports outside stdlib.
    """

    def __enter__(self) -> _suppress_errors:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True
