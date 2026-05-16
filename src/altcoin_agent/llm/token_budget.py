"""token_budget.py — Monthly LLM token budget enforcement (QUADRANT 六).

Single source of truth for "is the agent allowed to spend tokens on this
inference right now?". Wraps three concerns:

    1. Monthly accounting   — track tokens used, persisted across restarts
    2. Tiered gating        — free / economy / emergency / freeze modes
    3. Quadrant prioritisation — A always wins, D loses first

Plan defaults (operator-tunable):
    monthly_budget = 5,000,000 tokens
    50 - 80%  used => economy mode  (only A/B quadrants)
    80 - 95%  used => emergency mode (only A + score >= 70)
    >= 95%    used => freeze mode    (no LLM at all)

Persistence: ``.kiro/state/token_usage.json`` — tiny JSON, atomic write.
We keep both the current month's bucket *and* a rolling 12-month history
so the dashboard can chart spend over time without scraping logs.

This module is sync + pure-data. Async wiring to ai_engine is in main.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------- #


class BudgetMode(str, Enum):
    FREE = "free"             # < 50% used: no restriction
    ECONOMY = "economy"       # 50-80%: A/B only
    EMERGENCY = "emergency"   # 80-95%: A + high score only
    FREEZE = "freeze"         # >= 95%: no LLM at all


# Mode thresholds, fractions of budget used. Sorted ascending.
_DEFAULT_THRESHOLDS: tuple[tuple[float, BudgetMode], ...] = (
    (0.50, BudgetMode.FREE),
    (0.80, BudgetMode.ECONOMY),
    (0.95, BudgetMode.EMERGENCY),
    (1.00, BudgetMode.FREEZE),
)


# --------------------------------------------------------------------- #
# Persisted state
# --------------------------------------------------------------------- #


@dataclass
class TokenBudgetState:
    """One month's bucket. Persisted shape on disk."""

    month_key: str       # "YYYY-MM"
    used: int = 0
    calls: int = 0
    budget: int = 5_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "month_key": self.month_key,
            "used": self.used,
            "calls": self.calls,
            "budget": self.budget,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenBudgetState:
        return cls(
            month_key=str(d["month_key"]),
            used=int(d.get("used", 0)),
            calls=int(d.get("calls", 0)),
            budget=int(d.get("budget", 5_000_000)),
        )


# --------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------- #


_QUADRANT_RANK: dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1}


@dataclass
class TokenBudgetManager:
    """Stateful month-rolling budget + tiered gating.

    Construction reads the persisted state if a path is given. ``record_usage``
    persists synchronously after every update — at 1 LLM call/min that's
    ~43k writes/month, perfectly fine for a JSON-on-disk store. Switching to
    SQLite (Phase B.3) is mechanical if it ever becomes a bottleneck.
    """

    monthly_budget: int = 5_000_000
    state_path: str | None = None
    # Override the threshold table for testing.
    thresholds: tuple[tuple[float, BudgetMode], ...] = _DEFAULT_THRESHOLDS
    # Per-mode minimum quadrant rank that may call the LLM.
    # ECONOMY: A/B (rank>=3). EMERGENCY: A only (rank>=4).
    economy_min_rank: int = 3
    emergency_min_rank: int = 4
    emergency_min_score: float = 70.0
    # Time injector for tests.
    now_fn: Any = field(default=time.time, repr=False)
    # Active month bucket.
    state: TokenBudgetState = field(init=False)
    # Rolling 12-month history (for dashboard).
    history: list[TokenBudgetState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = TokenBudgetState(
            month_key=self._current_month_key(),
            budget=self.monthly_budget,
        )
        if self.state_path and os.path.exists(self.state_path):
            self._load()

    # ---- queries ---- #

    def used_pct(self) -> float:
        if self.state.budget <= 0:
            return 1.0
        return self.state.used / self.state.budget

    def mode(self) -> BudgetMode:
        u = self.used_pct()
        # thresholds is sorted ascending by ceiling.
        for ceiling, mode in self.thresholds:
            if u < ceiling:
                return mode
        return BudgetMode.FREEZE

    def can_call_llm(
        self,
        *,
        quadrant: str,
        signal_score: float,
    ) -> tuple[bool, str]:
        """Decide whether a single LLM call is currently permitted.

        Returns (allowed, reason). ``reason`` is human-readable so the
        rejected call gets recorded with enough context to debug
        budget-related drops in the audit log.
        """
        self._maybe_rollover()
        m = self.mode()
        if m is BudgetMode.FREE:
            return True, "free"
        rank = _QUADRANT_RANK.get(quadrant, 0)
        if m is BudgetMode.ECONOMY:
            if rank >= self.economy_min_rank:
                return True, "economy_ok"
            return False, f"economy_block:quadrant={quadrant}"
        if m is BudgetMode.EMERGENCY:
            if rank < self.emergency_min_rank:
                return False, f"emergency_block:quadrant={quadrant}"
            if signal_score < self.emergency_min_score:
                return False, (
                    f"emergency_block:score={signal_score:.1f}<"
                    f"{self.emergency_min_score:.1f}"
                )
            return True, "emergency_ok"
        # FREEZE
        return False, "freeze:budget_exhausted"

    # ---- mutations ---- #

    def record_usage(self, tokens: int) -> None:
        """Add ``tokens`` to the current bucket, rolling months as needed."""
        if tokens <= 0:
            return
        self._maybe_rollover()
        self.state.used += int(tokens)
        self.state.calls += 1
        if self.state_path:
            self._save()
        if self.used_pct() >= 0.80:
            logger.warning(
                "TokenBudgetManager: %.1f%% of monthly budget used "
                "(%s tokens of %s); mode=%s",
                self.used_pct() * 100,
                self.state.used,
                self.state.budget,
                self.mode().value,
            )

    @staticmethod
    def estimate_tokens(prompt: str) -> int:
        """Cheap pre-call estimator (mixed CN/EN: ~3.5 chars/token).

        Used by callers that want to know if a planned prompt would tip
        them into a higher mode *before* sending; not authoritative.
        """
        if not prompt:
            return 0
        return max(1, int(len(prompt) / 3.5))

    # ---- month rollover ---- #

    def _current_month_key(self) -> str:
        ts = float(self.now_fn())
        gm = time.gmtime(ts)
        return f"{gm.tm_year:04d}-{gm.tm_mon:02d}"

    def _maybe_rollover(self) -> None:
        cur = self._current_month_key()
        if cur == self.state.month_key:
            return
        # Archive the old bucket; cap history at 12 months so the file
        # stays bounded.
        self.history.append(self.state)
        if len(self.history) > 12:
            self.history = self.history[-12:]
        self.state = TokenBudgetState(
            month_key=cur,
            budget=self.monthly_budget,
        )
        if self.state_path:
            self._save()
        logger.info("TokenBudgetManager: rolled over to month %s", cur)

    # ---- persistence ---- #

    def _save(self) -> None:
        assert self.state_path
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "saved_at_ts": int(self.now_fn()),
            "active": self.state.as_dict(),
            "history": [b.as_dict() for b in self.history],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".token_usage.", suffix=".json.tmp",
            dir=os.path.dirname(self.state_path) or ".",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2,
                          sort_keys=True)
            os.replace(tmp_path, self.state_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        assert self.state_path
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "TokenBudgetManager: failed to load %s (%s); starting fresh",
                self.state_path, exc,
            )
            return
        if not isinstance(raw, dict):
            return
        active = raw.get("active")
        if isinstance(active, dict):
            try:
                loaded = TokenBudgetState.from_dict(active)
            except (KeyError, ValueError, TypeError):
                loaded = None
            if loaded is not None:
                if loaded.month_key == self._current_month_key():
                    self.state = loaded
                else:
                    # Stale file: archive its active bucket, start fresh.
                    self.history.append(loaded)
        history = raw.get("history") or []
        if isinstance(history, list):
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                try:
                    self.history.append(TokenBudgetState.from_dict(entry))
                except (KeyError, ValueError, TypeError):
                    continue
            if len(self.history) > 12:
                self.history = self.history[-12:]


__all__ = [
    "BudgetMode",
    "TokenBudgetManager",
    "TokenBudgetState",
]
