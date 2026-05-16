"""reject_reason_scorer.py — opportunity-cost scoring (Phase A.2).

Background
----------
``MissPenaltyEngine`` (A.1) labels each rejection as either a "missed
pump" or a "correct reject" 24h after the fact. Per the operator's
calibration in ``MISS_PENALTY_AND_PRODUCTION_PLAN.md§A.2.3``:

    correct reject  =  +1 point  (small credit for not chasing junk)
    missed pump     =  -3 points (3x penalty -- mistakes hurt more)

A reason that loses 25% of the time on missed pumps yields a net
score of:  0.75 * 1 + 0.25 * (-3)  =  0  -- at break-even. Anything
under that is candidate for threshold loosening (handled by A.3).

Why a 3x asymmetry?
-------------------
A reject that catches a real pump-and-dump saves the operator from a
~3% loss; missing a +200% pump costs ~30% of opportunity. The 1:3
ratio reflects this asymmetry without overshooting (a 1:10 ratio
would force the gate to loosen on the first miss, making the system
chase every signal).

Sample-size guard
-----------------
We deliberately refuse to act on a reason until ``samples_required``
observations are accumulated (default 50). Until then the score is
*reported* (so the dashboard can show what's stewing) but
:meth:`should_loosen` returns False. This protects against flipping
parameters on n=2 noise.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from altcoin_agent.risk.miss_penalty_engine import (
    MissedOpportunity,
    bucket_reject_reason,
    iter_decisions_with_rotations,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------- #


@dataclass
class RejectReasonScore:
    """Aggregate stats for one bucketed RiskGate reason.

    Field meanings:

    * ``correct_rejects``: rejection happened AND the symbol stayed
      flat / dropped within 24h (i.e. our reject saved us). Counted
      from rejected entries that did NOT make it onto
      ``missed_opportunities.jsonl`` after their audit window closed.
    * ``missed_pumps``: rejection happened AND symbol mooned (entry on
      ``missed_opportunities.jsonl`` with ``is_missed_pump=True``).
    * ``insufficient_data``: rejections we couldn't audit (not enough
      bars, recent reject inside the window). Excluded from the
      score; surfaced for visibility.
    * ``confidence_score``:  +1 * correct_rejects - 3 * missed_pumps.
      Higher == reason is doing its job.
    * ``win_rate``: ``correct_rejects / (correct_rejects + missed_pumps)``
      with Laplace smoothing so a 0/0 reason doesn't divide-by-zero.

    The score is intentionally additive (not normalised by total)
    so a popular-but-bad reason loses harder than an obscure one --
    that matches the operator's intuition that
    ``min_liquidity:200000`` firing 500x and missing 30 pumps is
    worse than ``regime_btc_falling_long_blocked`` firing 5x and
    missing 1 pump.
    """

    reason: str
    correct_rejects: int = 0
    missed_pumps: int = 0
    insufficient_data: int = 0
    last_updated_ts_ms: int = 0

    # Severity-weighted miss count (sum of miss_severity for misses).
    # When two reasons tie on count, the one with deeper misses
    # (severity 0.95 vs 0.55) gets loosened first.
    weighted_miss_severity: float = 0.0

    @property
    def total_audited(self) -> int:
        return self.correct_rejects + self.missed_pumps

    @property
    def confidence_score(self) -> float:
        # +1 / -3 weighting per the operator's calibration.
        return float(self.correct_rejects - 3 * self.missed_pumps)

    @property
    def win_rate(self) -> float:
        # Laplace smoothing: (correct + 1) / (total + 2).
        # A reason with no audited samples reports 50% (neutral),
        # not undefined.
        denom = self.total_audited + 2
        return (self.correct_rejects + 1) / denom

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence_score"] = self.confidence_score
        d["win_rate"] = self.win_rate
        d["total_audited"] = self.total_audited
        return d


@dataclass
class RejectReasonScorerConfig:
    """Tuning knobs."""

    # File the scores are persisted to.
    state_path: str = ".kiro/state/miss_penalty/reject_reason_scores.json"

    # Threshold below which we'd consider loosening (used by A.3).
    # The operator's calibration says: a reason with confidence < 0
    # is net-negative -- it's costing more in missed pumps than it's
    # saving in correct rejects. The default loosening trigger is
    # ``confidence_score <= -10`` AND ``samples >= samples_required``.
    samples_required: int = 50
    loosen_when_confidence_below: float = -10.0
    loosen_when_missed_pumps_at_least: int = 5

    # Reject-reason bucketing knob (passed through to A.1 helper).
    reason_bucket_separator: str = ":"

    # Reasons we deliberately never tune. Built-in safety / structural
    # gates (e.g. ``reconciliation_pending``, ``global_halt``, ``daily
    # _drawdown_limit``) MUST stay at their defaults regardless of
    # how many pumps we miss while reconciling -- relaxing those
    # defeats the purpose of the safety wall.
    untuneable_reasons: tuple[str, ...] = (
        "global_halt",
        "reconciliation_pending",
        "daily_drawdown_limit",
        "daily_stoploss_hits_exceeded",
        "signal_blocked",
        "signal_not_high_priority",
        "signal_direction_neutral",
        "unexpected_error",
    )


# --------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------- #


class RejectReasonScorer:
    """Maintains per-reason ``RejectReasonScore`` aggregates.

    The scorer is **stateless across invocations**: we recompute the
    table from scratch each run by walking ``decisions.jsonl`` (for
    correct rejects) + ``missed_opportunities.jsonl`` (for misses).
    This avoids drift and lets us treat the score file as a
    derived artifact: deleting it is harmless, the next run rebuilds.
    """

    def __init__(
        self,
        *,
        decisions_log_path: Path | str,
        missed_opportunities_path: Path | str,
        config: RejectReasonScorerConfig | None = None,
    ):
        self.decisions_log_path = Path(decisions_log_path)
        self.missed_opportunities_path = Path(missed_opportunities_path)
        self.cfg = config or RejectReasonScorerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def recompute(
        self, *, lookback_sec: int = 30 * 24 * 3600,
    ) -> dict[str, RejectReasonScore]:
        """Walk both logs, build the score table, persist + return it.

        ``lookback_sec`` defaults to 30 days, matching the operator's
        "rolling 30-day reflection window" requirement.
        """
        now_s = time.time()
        oldest_s = now_s - lookback_sec

        # 1) Build the missed-pump set: trace_id -> MissedOpportunity.
        missed_by_trace: dict[str, MissedOpportunity] = {}
        for opp in self._iter_missed_opportunities(since_ts=oldest_s):
            missed_by_trace[opp.trace_id] = opp

        # 2) Walk the decision log; for each rejected entry, classify.
        scores: dict[str, RejectReasonScore] = {}
        for rec in iter_decisions_with_rotations(
            self.decisions_log_path, since_ts=oldest_s,
        ):
            if rec.get("approved", False):
                continue
            raw_reason = str(rec.get("reason") or "unknown")
            bucket = bucket_reject_reason(
                raw_reason, separator=self.cfg.reason_bucket_separator,
            )
            score = scores.setdefault(bucket, RejectReasonScore(reason=bucket))
            score.last_updated_ts_ms = max(
                score.last_updated_ts_ms,
                int(_coerce_float(rec.get("ts"), default=0.0) * 1000),
            )
            trace_id = str(rec.get("trace_id") or "")
            if not trace_id:
                # No trace_id -> we can't cross-reference the audit.
                # Conservatively count as "insufficient_data" to keep
                # the row visible without poisoning the math.
                score.insufficient_data += 1
                continue
            opp = missed_by_trace.get(trace_id)
            if opp is None:
                # Two cases: (a) reject is too recent and the audit
                # window hasn't closed; (b) audit ran and decided
                # this wasn't a missed pump (the engine only
                # PERSISTS missed pumps -- see _iter_missed_opportunities).
                # We treat (b) as a "correct reject" only when the
                # decision is OLDER than 24h, otherwise it's still in
                # the audit pipeline.
                ts_s = _coerce_float(rec.get("ts"), default=0.0)
                if now_s - ts_s >= 24 * 3600:
                    score.correct_rejects += 1
                else:
                    score.insufficient_data += 1
                continue
            if opp.insufficient_data:
                score.insufficient_data += 1
            elif opp.is_missed_pump:
                score.missed_pumps += 1
                score.weighted_miss_severity += opp.miss_severity
            else:
                score.correct_rejects += 1

        self._persist(scores)
        return scores

    def load(self) -> dict[str, RejectReasonScore]:
        """Read the persisted table without recomputing.

        Returns ``{}`` when the file doesn't exist yet (first boot).
        """
        path = Path(self.cfg.state_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "RejectReasonScorer.load: %s; returning empty table", e,
            )
            return {}
        out: dict[str, RejectReasonScore] = {}
        for row in data.get("scores", []):
            if not isinstance(row, dict):
                continue
            try:
                # ``confidence_score`` / ``win_rate`` / ``total_audited``
                # are properties on the dataclass; drop them on load
                # so __init__ doesn't choke.
                clean = {
                    k: v for k, v in row.items()
                    if k in RejectReasonScore.__dataclass_fields__
                }
                out[row["reason"]] = RejectReasonScore(**clean)
            except (TypeError, KeyError):
                continue
        return out

    def should_loosen(self, score: RejectReasonScore) -> bool:
        """True when this reason qualifies for threshold loosening
        per the operator's rule:

            samples >= samples_required
            AND confidence_score <= loosen_when_confidence_below
            AND missed_pumps >= loosen_when_missed_pumps_at_least
            AND reason not in untuneable_reasons

        See A.3 for the actual threshold-tuning logic; this helper
        is the "is the dial qualified to move?" gate.
        """
        if score.reason in self.cfg.untuneable_reasons:
            return False
        if score.total_audited < self.cfg.samples_required:
            return False
        if score.missed_pumps < self.cfg.loosen_when_missed_pumps_at_least:
            return False
        return score.confidence_score <= self.cfg.loosen_when_confidence_below

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _iter_missed_opportunities(
        self, *, since_ts: float,
    ) -> Iterable[MissedOpportunity]:
        path = Path(self.missed_opportunities_path)
        if not path.exists():
            return []
        out: list[MissedOpportunity] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("missed_opportunities read failed: %s", e)
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            ts_ms = row.get("rejected_at_ts_ms")
            if isinstance(ts_ms, (int, float)) and (ts_ms / 1000.0) < since_ts:
                continue
            try:
                out.append(MissedOpportunity.from_dict(row))
            except TypeError:
                continue
        return out

    def _persist(self, scores: dict[str, RejectReasonScore]) -> None:
        path = Path(self.cfg.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at_ts_ms": int(time.time() * 1000),
                "config": {
                    "samples_required": self.cfg.samples_required,
                    "loosen_when_confidence_below": self.cfg.loosen_when_confidence_below,
                    "loosen_when_missed_pumps_at_least":
                        self.cfg.loosen_when_missed_pumps_at_least,
                },
                # Sort by confidence ASC so the worst-performing
                # reasons appear at the top of the file -- the only
                # ones an operator usually cares about.
                "scores": [
                    s.to_dict()
                    for s in sorted(
                        scores.values(),
                        key=lambda s: (s.confidence_score, -s.total_audited),
                    )
                ],
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(
                "RejectReasonScorer._persist failed (swallowed): %s", e,
            )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _coerce_float(v: object, *, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f
