"""production_rules_loader.py — Hot-reload production_rules.json.

The walk-forward trainer (``WalkforwardTrainer.run``) writes
``production_rules.json`` to ``state_dir`` whenever a rule clears the
80% gate (samples >= 30, win_rate >= 0.80, sharpe >= 1.5,
validation_months_passed >= 3). On the live daemon side we want every
worker that consults learned-rule statistics to see the freshest
trainer output **without** restarting the daemon.

This module is the seam: a tiny class that

  * reads ``<state_dir>/production_rules.json`` from disk
  * keeps an in-memory snapshot keyed by ``rule_id``
  * polls the file's mtime at most every ``min_check_interval_sec``
    so the hot path doesn't pay an fstat per signal
  * fires an optional ``on_reload`` callback so subscribers (the
    fuser, the risk gate, the dashboard) can refresh their derived
    state without polling themselves

Failure modes are all *fail-soft*: a missing file leaves the snapshot
empty; a malformed JSON file logs a warning and keeps the previous
good snapshot. The rest of the daemon stays alive even if the trainer
writes garbage — exactly the same discipline ``fuser.RuleIndex`` uses
for ``dynamic_rules.json``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from altcoin_agent.training.rules_promoter import LearnedRule

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #


@dataclass
class ProductionRulesLoader:
    """Mtime-throttled, thread-safe view over ``production_rules.json``.

    Construct with the path to the trainer's state directory. ``rules``
    returns the latest snapshot; ``maybe_reload`` re-reads the file
    only when both (a) the throttle interval has elapsed and (b) the
    file's mtime has advanced. ``force_reload`` ignores the throttle.

    Thread-safety: a single lock guards the snapshot + bookkeeping.
    The hot path on the daemon (``rules()``) takes a short read-only
    copy under the lock; reload work happens outside the lock.
    """

    path: str
    min_check_interval_sec: float = 30.0
    on_reload: Callable[[list[LearnedRule]], None] | None = None
    _rules: dict[str, LearnedRule] = field(default_factory=dict)
    _last_mtime_ns: int = 0
    _last_check_monotonic: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- public API ---- #

    def maybe_reload(self) -> bool:
        """Reload from disk if both the throttle and mtime allow.

        Returns True iff a reload actually happened (file was read).
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_check_monotonic < self.min_check_interval_sec:
                return False
            self._last_check_monotonic = now
        return self._reload_if_changed()

    def force_reload(self) -> bool:
        """Bypass the throttle and re-read the file."""
        with self._lock:
            self._last_check_monotonic = time.monotonic()
        return self._reload_if_changed(force=True)

    def rules(self) -> list[LearnedRule]:
        """Return a list copy of the current snapshot (safe to mutate)."""
        with self._lock:
            return list(self._rules.values())

    def get(self, rule_id: str) -> LearnedRule | None:
        with self._lock:
            return self._rules.get(rule_id)

    def lookup_by_features(
        self, *, feature_keys: list[str], value_key: str,
    ) -> LearnedRule | None:
        """Find a rule whose ``extras["feature_keys"]`` + value match.

        Mirrors how the trainer encodes rules so callers can ask
        "does the trainer have a production-grade win_rate for
        (quadrant=A, phase=ramp)?" without reconstructing rule_ids
        by hand.
        """
        with self._lock:
            for r in self._rules.values():
                keys = r.extras.get("feature_keys") if r.extras else None
                value = r.extras.get("value") if r.extras else None
                if keys == feature_keys and value == value_key:
                    return r
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._rules)

    def stats(self) -> dict[str, Any]:
        """For the dashboard / metrics."""
        with self._lock:
            return {
                "rule_count": len(self._rules),
                "last_mtime_ns": self._last_mtime_ns,
                "last_check_monotonic": self._last_check_monotonic,
                "path": self.path,
            }

    # ---- internals ---- #

    def _reload_if_changed(self, *, force: bool = False) -> bool:
        try:
            mtime_ns = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            with self._lock:
                if self._rules:
                    self._rules = {}
                    self._last_mtime_ns = 0
                    logger.info(
                        "ProductionRulesLoader: %s disappeared; "
                        "snapshot cleared", self.path,
                    )
            return False
        except OSError as exc:
            logger.warning(
                "ProductionRulesLoader: stat failed for %s: %s",
                self.path, exc,
            )
            return False

        with self._lock:
            already_current = (
                not force and mtime_ns == self._last_mtime_ns
            )
        if already_current:
            return False

        try:
            with open(self.path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "ProductionRulesLoader: failed to parse %s: %s; "
                "keeping previous snapshot", self.path, exc,
            )
            return False

        entries = payload.get("rules") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            logger.warning(
                "ProductionRulesLoader: %s shape unexpected; "
                "keeping previous snapshot", self.path,
            )
            return False

        new_rules: dict[str, LearnedRule] = {}
        for raw in entries:
            try:
                r = LearnedRule.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "ProductionRulesLoader: skipping malformed rule "
                    "(%s) in %s", exc, self.path,
                )
                continue
            new_rules[r.rule_id] = r

        with self._lock:
            self._rules = new_rules
            self._last_mtime_ns = mtime_ns

        logger.info(
            "ProductionRulesLoader: reloaded %s rules from %s",
            len(new_rules), self.path,
        )

        if self.on_reload is not None:
            try:
                self.on_reload(list(new_rules.values()))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ProductionRulesLoader on_reload callback "
                    "raised: %s", exc,
                )
        return True


__all__ = ["ProductionRulesLoader"]
