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
* Append-only — never overwrite or rotate from inside this module.
  Operators rotate via logrotate / fluent-bit / k8s sidecars.
* Failures are swallowed; we never block the trading loop because
  the audit log can't write.

Out of scope
------------
* Cryptographic signing / tamper-proofing.
* Forwarding to a central log store (do that from the file).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionAuditLog:
    """Append-only sink for gate decisions."""

    path: Path
    enabled: bool = True
    last_write_ts: float = 0.0
    write_count: int = 0
    write_errors: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

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
