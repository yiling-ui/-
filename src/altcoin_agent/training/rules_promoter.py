"""rules_promoter.py — 80% conviction gate for promoting learned rules.

Each rule the trainer accumulates carries (samples, wins, losses,
avg_pnl, sharpe, ...). The promoter turns that into a binary
``production_ready`` flag using the plan's hard thresholds:

    samples              >= 30
    win_rate             >= 0.80
    sharpe               >= 1.5
    validation_months    >= 3

Rules not meeting the bar stay in ``candidate_rules.json``; rules that
clear it move into ``production_rules.json``. The trainer in Phase 4
calls ``promote_all`` once a day to refresh both files.

Demotion: a rule that has been production_ready but slips below
``demotion_win_rate`` for ``demotion_grace_days`` consecutive days is
moved back to candidate. The plan calls for "30 days < 0.50 win_rate"
as the demotion trigger; we expose those values as config so a future
trainer can tune them without code change.

Both promotion + demotion are deterministic — no I/O, no time-of-day
behaviour beyond the input timestamps the caller passes. This makes
the trainer trivial to test on synthetic rules.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Rule shape
# --------------------------------------------------------------------- #


@dataclass
class LearnedRule:
    """One candidate or production rule.

    Field semantics match QUADRANT_STRATEGY_PLAN section 5.3 verbatim.
    Extra fields the trainer needs (e.g. feature vectors) live in a
    separate ``extras`` dict to keep this dataclass schema-stable.
    """

    rule_id: str
    samples: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    sharpe: float = 0.0
    first_observed_ts: int = 0
    last_observed_ts: int = 0
    validation_months_passed: int = 0
    confidence: float = 0.0
    production_ready: bool = False
    # Bookkeeping for demotion grace.
    consecutive_bad_days: int = 0
    last_evaluated_ts: int = 0
    # Free-form extras (rule premise, feature weights, notes).
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearnedRule:
        return cls(
            rule_id=str(d["rule_id"]),
            samples=int(d.get("samples", 0)),
            wins=int(d.get("wins", 0)),
            losses=int(d.get("losses", 0)),
            win_rate=float(d.get("win_rate", 0.0)),
            avg_pnl_pct=float(d.get("avg_pnl_pct", 0.0)),
            sharpe=float(d.get("sharpe", 0.0)),
            first_observed_ts=int(d.get("first_observed_ts", 0)),
            last_observed_ts=int(d.get("last_observed_ts", 0)),
            validation_months_passed=int(d.get("validation_months_passed", 0)),
            confidence=float(d.get("confidence", 0.0)),
            production_ready=bool(d.get("production_ready", False)),
            consecutive_bad_days=int(d.get("consecutive_bad_days", 0)),
            last_evaluated_ts=int(d.get("last_evaluated_ts", 0)),
            extras=dict(d.get("extras") or {}),
        )


# --------------------------------------------------------------------- #
# Promotion config
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromotionConfig:
    """Hard thresholds for production promotion + demotion."""

    min_samples: int = 30
    min_win_rate: float = 0.80
    min_sharpe: float = 1.5
    min_validation_months: int = 3
    # Demotion: while production_ready, if win_rate < demotion_win_rate
    # for ``demotion_grace_days`` consecutive evaluations, demote.
    demotion_win_rate: float = 0.50
    demotion_grace_days: int = 30


# --------------------------------------------------------------------- #
# Promoter
# --------------------------------------------------------------------- #


@dataclass
class RulesPromoter:
    """Promote / demote rules and persist both pools to JSON.

    Two files on disk:

        <state_dir>/production_rules.json   — passed the 80% gate
        <state_dir>/candidate_rules.json    — still being trained

    The trainer calls ``promote_all(rules, now_ts)`` once per training
    cycle. Existing files are read once on construction and rewritten
    after each promote_all (atomic via tmp+rename).
    """

    cfg: PromotionConfig = field(default_factory=PromotionConfig)
    state_dir: str | None = None
    _production: dict[str, LearnedRule] = field(default_factory=dict)
    _candidate: dict[str, LearnedRule] = field(default_factory=dict)
    _loaded: bool = False

    # ---- entrypoints ---- #

    def promote_all(
        self, rules: list[LearnedRule], now_ts: int | None = None,
    ) -> tuple[list[LearnedRule], list[LearnedRule]]:
        """Classify each rule into production / candidate.

        Returns ``(promoted_now, demoted_now)`` — the rules whose
        ``production_ready`` flag flipped during this call. Both files
        are rewritten to disk afterward.
        """
        self._ensure_loaded()
        ts = int(now_ts if now_ts is not None else time.time())
        promoted: list[LearnedRule] = []
        demoted: list[LearnedRule] = []

        for rule in rules:
            rule.last_evaluated_ts = ts
            was_production = self._production.get(rule.rule_id) is not None
            qualifies = self._qualifies(rule)
            if qualifies:
                rule.consecutive_bad_days = 0
                rule.production_ready = True
                self._production[rule.rule_id] = rule
                self._candidate.pop(rule.rule_id, None)
                if not was_production:
                    promoted.append(rule)
            else:
                if was_production:
                    # Production rule didn't qualify this cycle.
                    rule.consecutive_bad_days += 1
                    if self._should_demote(rule):
                        rule.production_ready = False
                        rule.consecutive_bad_days = 0
                        self._production.pop(rule.rule_id, None)
                        self._candidate[rule.rule_id] = rule
                        demoted.append(rule)
                    else:
                        # Keep it in production but record the bad day.
                        self._production[rule.rule_id] = rule
                else:
                    rule.production_ready = False
                    self._candidate[rule.rule_id] = rule

        if self.state_dir:
            self._save()
        return promoted, demoted

    def production_rules(self) -> list[LearnedRule]:
        self._ensure_loaded()
        return list(self._production.values())

    def candidate_rules(self) -> list[LearnedRule]:
        self._ensure_loaded()
        return list(self._candidate.values())

    # ---- predicates ---- #

    def _qualifies(self, rule: LearnedRule) -> bool:
        c = self.cfg
        return (
            rule.samples >= c.min_samples
            and rule.win_rate >= c.min_win_rate
            and rule.sharpe >= c.min_sharpe
            and rule.validation_months_passed >= c.min_validation_months
        )

    def _should_demote(self, rule: LearnedRule) -> bool:
        c = self.cfg
        if rule.win_rate >= c.demotion_win_rate:
            return False
        return rule.consecutive_bad_days >= c.demotion_grace_days

    # ---- io ---- #

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.state_dir:
            return
        for pool, fname in (
            (self._production, "production_rules.json"),
            (self._candidate, "candidate_rules.json"),
        ):
            path = os.path.join(self.state_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("RulesPromoter: failed to load %s: %s", path, exc)
                continue
            entries = raw if isinstance(raw, list) else raw.get("rules") or []
            for entry in entries:
                try:
                    r = LearnedRule.from_dict(entry)
                except (KeyError, ValueError, TypeError):
                    continue
                pool[r.rule_id] = r

    def _save(self) -> None:
        assert self.state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        for pool, fname in (
            (self._production, "production_rules.json"),
            (self._candidate, "candidate_rules.json"),
        ):
            path = os.path.join(self.state_dir, fname)
            payload = {
                "version": 1,
                "saved_at_ts": int(time.time()),
                "rules": [r.as_dict() for r in pool.values()],
            }
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=f".{fname}.", suffix=".tmp", dir=self.state_dir,
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2,
                              sort_keys=True)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise


__all__ = [
    "LearnedRule",
    "PromotionConfig",
    "RulesPromoter",
]
