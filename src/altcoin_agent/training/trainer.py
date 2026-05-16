"""trainer.py — Walk-forward trainer scaffolding (QUADRANT Phase 4 stub).

The full implementation lands in Phase 4. This file ships:

    * ``TrainerConfig`` — parameters for the walk-forward loop
    * ``TrainingReport`` — output schema
    * ``Trainer`` — class with run() that delegates to ``RulesPromoter``
      and writes a daily report; the actual rule mining is a TODO
      hook so the trainer can be plugged in incrementally.

Phase 1-3 keeps the file small but real: nothing here is dead code,
and the test suite exercises the report / config plumbing so future
work has a stable boundary.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from altcoin_agent.training.rules_promoter import (
    LearnedRule,
    PromotionConfig,
    RulesPromoter,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainerConfig:
    """Walk-forward parameters.

    The plan calls for monthly windows; we expose seconds so tests can
    use shorter horizons without monkey-patching.
    """

    # Walk-forward window length, seconds. 30 days default.
    window_sec: int = 30 * 24 * 3600
    # Validation lag — how far behind ``now`` the trainer's "as-of"
    # cursor sits, so it never trains on data that hasn't yet had a
    # chance to play out.
    validation_lag_sec: int = 24 * 3600
    # Monthly token cap for training-side LLM use (separate from the
    # global TokenBudgetManager which counts live signal tokens).
    monthly_training_token_cap: int = 200_000
    # Where to write daily reports.
    report_dir: str | None = None
    # Promotion thresholds (passed through to RulesPromoter).
    promotion: PromotionConfig = field(default_factory=PromotionConfig)


# --------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------- #


@dataclass
class TrainingReport:
    """One day's training cycle output."""

    started_at_ts: int
    finished_at_ts: int
    cycle_label: str  # e.g. "20260516"
    rules_seen: int = 0
    rules_promoted: int = 0
    rules_demoted: int = 0
    rules_production_total: int = 0
    rules_candidate_total: int = 0
    tokens_used: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------- #


@dataclass
class Trainer:
    """Walk-forward trainer (skeleton).

    ``run_cycle(rules, now_ts)`` is the unit a daily cron triggers. The
    rule-mining itself (turn audit logs into ``LearnedRule`` candidates)
    is the Phase 4 TODO; here we accept a pre-built list and exercise
    the promotion + report machinery end-to-end.
    """

    cfg: TrainerConfig
    promoter: RulesPromoter | None = None

    def __post_init__(self) -> None:
        if self.promoter is None:
            self.promoter = RulesPromoter(cfg=self.cfg.promotion)

    # ---- main entrypoint ---- #

    def run_cycle(
        self, rules: list[LearnedRule], now_ts: int | None = None,
    ) -> TrainingReport:
        ts = int(now_ts if now_ts is not None else time.time())
        started = ts
        promoted, demoted = self.promoter.promote_all(rules, now_ts=ts)
        prod = self.promoter.production_rules()
        cand = self.promoter.candidate_rules()
        report = TrainingReport(
            started_at_ts=started,
            finished_at_ts=int(time.time()),
            cycle_label=time.strftime("%Y%m%d", time.gmtime(ts)),
            rules_seen=len(rules),
            rules_promoted=len(promoted),
            rules_demoted=len(demoted),
            rules_production_total=len(prod),
            rules_candidate_total=len(cand),
        )
        if self.cfg.report_dir:
            self._write_report(report)
        return report

    # ---- io ---- #

    def _write_report(self, report: TrainingReport) -> None:
        assert self.cfg.report_dir
        os.makedirs(self.cfg.report_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.report_dir,
            f"daily_report_{report.cycle_label}.json",
        )
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".report.", suffix=".json.tmp",
            dir=self.cfg.report_dir,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(report.as_dict(), fh, ensure_ascii=False,
                          indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingReport",
]
