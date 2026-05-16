"""rule_miner.py — Turn (features, pnl) observations into LearnedRule candidates.

Phase 4 deliverable. The trainer collects pairs of:

    * signal feature vector  — what the screener / fuser saw at decision time
    * realized PnL            — fractional return after the trade closed (or
                                would have closed, if we replayed the
                                MatchingEngine over historical data)

This module turns those pairs into per-bucket statistics. Each non-empty
bucket emits one ``LearnedRule`` with samples / wins / losses / win_rate
/ sharpe / avg_pnl_pct populated. The result is fed straight into
``RulesPromoter`` which applies the 80% production gate.

Bucketing strategy
------------------
We use *categorical buckets only* — discretised slices of feature space
that an operator can read in plain English. Continuous features (rule
score, signal score) are coarsely binned. This is by design:

  * The plan demands rules an operator can *audit*: "if quadrant=A and
    phase=ramp and rule_score>=80 then win_rate=0.83". Tree-induced
    splits on continuous variables would be more accurate but harder
    to defend in a post-mortem.
  * The trainer in v1 produces O(100) rules, well within the
    ``RulesPromoter`` storage. A regression model that scales to 10k
    rules would need a different schema.

Determinism
-----------
Pure data + logic, no I/O, no clocks (caller passes ``now_ts``). Same
inputs → same rules → same promotions. The walk-forward driver
(``walkforward_trainer.py``) is what introduces the temporal axis.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from altcoin_agent.training.rules_promoter import LearnedRule

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeObservation:
    """One closed trade — the input to rule mining.

    ``features`` is a flat dict of *categorical* labels (already
    discretised). ``pnl_pct`` is fractional (0.05 = +5%). Continuous
    raw features (rule_score=83.4) should be coarse-binned by the
    caller before constructing the observation; ``bucketize_score``
    helpers below make this convenient.

    ``ts_ms`` is the bar timestamp the trade closed at (used by the
    walk-forward driver to assign each observation to its window).
    """

    ts_ms: int
    symbol: str
    pnl_pct: float
    features: Mapping[str, str]


# --------------------------------------------------------------------- #
# Helpers — discretisation
# --------------------------------------------------------------------- #


def bucketize_score(score: float, *, edges: tuple[float, ...] = (60.0, 70.0, 80.0, 90.0)) -> str:
    """Bin a 0..100 score into a label like ``"score>=80"``.

    Edges are inclusive thresholds, ascending. For ``score=83`` and the
    default edges we return ``"score>=80"`` (the highest passed
    threshold). For ``score=55`` we return ``"score<60"``.
    """
    s = float(score)
    if s != s:  # NaN
        return "score=na"
    last = None
    for e in edges:
        if s >= e:
            last = e
        else:
            break
    if last is None:
        # Below the smallest edge.
        return f"score<{int(edges[0])}"
    return f"score>={int(last)}"


def bucketize_pct(value: float, *, edges: tuple[float, ...]) -> str:
    """Generic fractional-value binning. ``value=0.06`` with edges
    ``(0.0, 0.03, 0.06, 0.10)`` → ``"pct>=0.06"``."""
    v = float(value)
    if v != v:
        return "pct=na"
    last: float | None = None
    for e in edges:
        if v >= e:
            last = e
        else:
            break
    if last is None:
        return f"pct<{edges[0]:g}"
    return f"pct>={last:g}"


# --------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class MinerConfig:
    """Operator-tunable knobs.

    ``min_samples_per_bucket`` keeps spurious 1-sample 100% win-rate
    buckets out of the candidate pool. The promoter's 80% gate uses
    its own ``min_samples=30``; we use a smaller floor here so the
    promoter sees enough candidates to evaluate.

    ``feature_combos`` controls which feature subsets the miner
    enumerates. Each combo is a tuple of feature keys; the miner emits
    one rule per (combo, value) it sees in the data. A combo of
    ``("quadrant",)`` produces 4 rules max (A/B/C/D); a 2-feature combo
    produces up to ``|f1| × |f2|``. Triples explode quickly so we cap.
    """

    min_samples_per_bucket: int = 5
    feature_combos: tuple[tuple[str, ...], ...] = (
        ("quadrant",),
        ("phase",),
        ("score_bucket",),
        ("rejected_reason",),
        ("quadrant", "phase"),
        ("quadrant", "score_bucket"),
        ("phase", "score_bucket"),
    )
    # Cap on rules emitted per cycle — guards against pathological combos.
    max_rules_per_cycle: int = 500
    # Optional per-rule baseline confidence used as a tie-breaker the
    # promoter can read via ``LearnedRule.confidence``.
    confidence_floor: float = 0.0


def mine_rules(
    observations: Iterable[TradeObservation],
    cfg: MinerConfig | None = None,
) -> list[LearnedRule]:
    """Group observations by feature combos; return one rule per bucket.

    Skips buckets with fewer than ``cfg.min_samples_per_bucket``
    observations. Sharpe is the simple mean / population-stddev ratio
    on the bucket's PnL list (matches what ``walk_forward.stats_from_pnls``
    uses, so the trainer + walk-forward report agree).

    The returned rules are *unranked*: the promoter applies the 80%
    gate; downstream consumers can sort by ``avg_pnl_pct * win_rate``
    for display.
    """
    cfg = cfg or MinerConfig()
    obs = [o for o in observations if isinstance(o, TradeObservation)]
    if not obs:
        return []

    # bucket_key -> list of pnls
    by_bucket: dict[tuple[str, str], list[TradeObservation]] = {}
    for combo in cfg.feature_combos:
        for ob in obs:
            try:
                values = [str(ob.features[k]) for k in combo]
            except KeyError:
                # Observation is missing one of the features in this
                # combo; skip it for this combo. (It still gets
                # included in other combos that match.)
                continue
            combo_key = "+".join(combo)
            value_key = "/".join(values)
            by_bucket.setdefault((combo_key, value_key), []).append(ob)

    rules: list[LearnedRule] = []
    for (combo_key, value_key), bucket in sorted(by_bucket.items()):
        if len(bucket) < cfg.min_samples_per_bucket:
            continue
        rule = _build_rule(
            combo_key=combo_key,
            value_key=value_key,
            bucket=bucket,
            cfg=cfg,
        )
        rules.append(rule)
        if len(rules) >= cfg.max_rules_per_cycle:
            logger.warning(
                "rule_miner: hit max_rules_per_cycle=%d; truncating",
                cfg.max_rules_per_cycle,
            )
            break
    return rules


def _build_rule(
    *,
    combo_key: str,
    value_key: str,
    bucket: list[TradeObservation],
    cfg: MinerConfig,
) -> LearnedRule:
    pnls = [o.pnl_pct for o in bucket]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    samples = len(pnls)
    win_rate = wins / samples
    avg = statistics.fmean(pnls)
    if samples >= 2:
        sd = statistics.pstdev(pnls)
        sharpe = avg / sd if sd > 0 else 0.0
    else:
        sharpe = 0.0
    first_ts = min(o.ts_ms for o in bucket)
    last_ts = max(o.ts_ms for o in bucket)
    # validation_months_passed is the trainer's responsibility — it's
    # set by ``WalkforwardTrainer`` when the same rule_id repeats
    # across consecutive validate windows. Default 0 here.
    rule_id = f"{combo_key}={value_key}"
    return LearnedRule(
        rule_id=rule_id,
        samples=samples,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_pnl_pct=float(avg),
        sharpe=float(sharpe),
        first_observed_ts=first_ts,
        last_observed_ts=last_ts,
        validation_months_passed=0,
        confidence=max(cfg.confidence_floor, win_rate),
        production_ready=False,
        extras={
            "combo": combo_key,
            "value": value_key,
            "feature_keys": combo_key.split("+"),
        },
    )


# --------------------------------------------------------------------- #
# Aggregation across windows
# --------------------------------------------------------------------- #


@dataclass
class RuleAccumulator:
    """Combine the same rule across multiple walk-forward windows.

    The walk-forward driver mines rules per validate window. To know
    whether a rule has been *consistently* good for ≥ 3 months, we
    accumulate per-rule stats across windows and bump
    ``validation_months_passed`` when this window's win_rate clears
    a threshold.

    Use it as::

        acc = RuleAccumulator()
        for window_rules in all_window_rules:
            acc.add_window(window_rules)
        merged = acc.flush()  # list[LearnedRule]
    """

    qualifying_win_rate: float = 0.80
    qualifying_min_samples: int = 5
    _by_id: dict[str, LearnedRule] = field(default_factory=dict)
    _windows_seen: dict[str, int] = field(default_factory=dict)
    _qualifying_windows: dict[str, int] = field(default_factory=dict)

    def add_window(self, rules: list[LearnedRule]) -> None:
        for r in rules:
            existing = self._by_id.get(r.rule_id)
            if existing is None:
                self._by_id[r.rule_id] = LearnedRule(
                    rule_id=r.rule_id,
                    samples=r.samples,
                    wins=r.wins,
                    losses=r.losses,
                    win_rate=r.win_rate,
                    avg_pnl_pct=r.avg_pnl_pct,
                    sharpe=r.sharpe,
                    first_observed_ts=r.first_observed_ts,
                    last_observed_ts=r.last_observed_ts,
                    validation_months_passed=0,
                    confidence=r.confidence,
                    production_ready=False,
                    extras=dict(r.extras),
                )
            else:
                existing.samples += r.samples
                existing.wins += r.wins
                existing.losses += r.losses
                # Sample-weighted incremental mean.
                total = existing.samples
                if total > 0:
                    existing.win_rate = existing.wins / total
                # Pool the avg PnL the same way (samples-weighted).
                existing.avg_pnl_pct = _pool_mean(
                    existing.avg_pnl_pct, existing.samples - r.samples,
                    r.avg_pnl_pct, r.samples,
                )
                # Sharpe pooling is not algebraically clean across
                # disjoint windows; we approximate by recomputing from
                # the running mean and an upper-bound sd inferred from
                # the pooled win/loss distribution. For ranking purposes
                # this stays monotone with PnL quality.
                existing.sharpe = _approx_sharpe(
                    existing.avg_pnl_pct, existing.win_rate,
                )
                existing.first_observed_ts = min(
                    existing.first_observed_ts, r.first_observed_ts,
                )
                existing.last_observed_ts = max(
                    existing.last_observed_ts, r.last_observed_ts,
                )
                existing.confidence = max(
                    existing.confidence, r.confidence,
                )
            self._windows_seen[r.rule_id] = self._windows_seen.get(
                r.rule_id, 0
            ) + 1
            if (
                r.samples >= self.qualifying_min_samples
                and r.win_rate >= self.qualifying_win_rate
            ):
                self._qualifying_windows[r.rule_id] = self._qualifying_windows.get(
                    r.rule_id, 0
                ) + 1

    def flush(self) -> list[LearnedRule]:
        out: list[LearnedRule] = []
        for rid, rule in self._by_id.items():
            rule.validation_months_passed = self._qualifying_windows.get(rid, 0)
            out.append(rule)
        return out


def _pool_mean(
    mean_a: float, n_a: int, mean_b: float, n_b: int,
) -> float:
    """Sample-weighted mean of two disjoint groups."""
    total = n_a + n_b
    if total <= 0:
        return 0.0
    return (mean_a * n_a + mean_b * n_b) / total


def _approx_sharpe(avg_pnl: float, win_rate: float) -> float:
    """Recover an approximate sharpe from (avg, win_rate) only.

    We don't carry the full PnL list across windows (would balloon
    memory). The promoter's 80% gate cares about *direction* of sharpe
    rather than its precise value, so this approximation is good
    enough: assume a Bernoulli-like distribution with magnitude
    proportional to ``avg_pnl`` and stddev scaled by ``sqrt(win_rate
    * (1 - win_rate)) * 2``. Returns 0.0 when win_rate hits 0/1
    (degenerate distribution → undefined sharpe).
    """
    p = max(0.0, min(1.0, float(win_rate)))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    sd = math.sqrt(p * (1.0 - p)) * 2.0
    return avg_pnl / sd if sd > 0 else 0.0


__all__ = [
    "MinerConfig",
    "RuleAccumulator",
    "TradeObservation",
    "bucketize_pct",
    "bucketize_score",
    "mine_rules",
]
