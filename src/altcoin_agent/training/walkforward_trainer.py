"""walkforward_trainer.py — Drive the walk-forward training loop.

The Phase 4 deliverable that ties together everything Phase 1-3 + B.4
already shipped:

    BacktestRunner / MatchingEngine    →  realized PnL per trade
    rule_miner.mine_rules              →  LearnedRule per feature bucket
    RuleAccumulator                    →  cross-window pooling
    walk_forward.iter_splits           →  rolling (train, validate)
    RulesPromoter.promote_all          →  80% gate, write to disk

This module is what an operator actually runs (eventually via cron):

    trainer = WalkforwardTrainer(...)
    report  = trainer.run(start_ms=..., end_ms=..., observations=...)

It accepts a *callable* that yields ``TradeObservation`` for a given
window, so the production wiring (drive matching engine over historical
klines + reconstruct features) can be plugged in incrementally without
touching this module. Tests pass in a list of pre-built observations.

Determinism + isolation
-----------------------
Pure data plumbing: the heavy lifting (matching engine, feature
extraction) is the caller's responsibility. We expose enough hooks
that the trainer itself stays trivial to unit-test.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from altcoin_agent.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardSplit,
    WindowStats,
    iter_splits,
    stats_from_pnls,
)
from altcoin_agent.training.rule_miner import (
    MinerConfig,
    RuleAccumulator,
    TradeObservation,
    mine_rules,
)
from altcoin_agent.training.rules_promoter import (
    LearnedRule,
    PromotionConfig,
    RulesPromoter,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config + report
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class WalkforwardTrainerConfig:
    """Operator-tunable knobs.

    ``train_window_ms`` and ``validate_window_ms`` come from the plan's
    "1-month train, 1-month validate" cadence; expose them so tests
    can use shorter horizons without monkey-patching ``time.time``.
    """

    train_window_ms: int = 30 * 24 * 3600 * 1000
    validate_window_ms: int = 30 * 24 * 3600 * 1000
    step_ms: int = 30 * 24 * 3600 * 1000
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    miner: MinerConfig = field(default_factory=MinerConfig)
    state_dir: str | None = None  # forwarded to RulesPromoter
    # Hard cap on tokens per training cycle. The trainer doesn't make
    # LLM calls itself in v1 (rule mining is pure-rule); this is set
    # so the report carries a verifiable 0 and an audit can confirm
    # we stayed under the plan's 200K cap.
    monthly_training_token_cap: int = 200_000


@dataclass
class WalkforwardWindowReport:
    """Per-window summary (one entry per (train, validate) split)."""

    index: int
    train_start_ms: int
    train_end_ms: int
    validate_start_ms: int
    validate_end_ms: int
    train_observations: int
    validate_observations: int
    rules_mined: int
    win_stats: WindowStats

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start_ms": self.train_start_ms,
            "train_end_ms": self.train_end_ms,
            "validate_start_ms": self.validate_start_ms,
            "validate_end_ms": self.validate_end_ms,
            "train_observations": self.train_observations,
            "validate_observations": self.validate_observations,
            "rules_mined": self.rules_mined,
            "win_stats": self.win_stats.as_dict(),
        }


@dataclass
class WalkforwardTrainerReport:
    started_at_ts: int
    finished_at_ts: int
    splits: int
    windows: list[WalkforwardWindowReport] = field(default_factory=list)
    rules_after_pooling: int = 0
    rules_promoted_now: int = 0
    rules_demoted_now: int = 0
    rules_production_total: int = 0
    rules_candidate_total: int = 0
    tokens_used: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["windows"] = [w.as_dict() for w in self.windows]
        return d


# --------------------------------------------------------------------- #
# Observation provider — pluggable
# --------------------------------------------------------------------- #


# A callable that returns the observations for a half-open ``[start_ms,
# end_ms)`` window. The trainer calls this twice per split (once for
# the train window, once for validate) and the implementation can
# either:
#   (a) drive a MatchingEngine over historical klines and emit
#       TradeObservation per closed trade, or
#   (b) replay a pre-built list of observations (used by tests + the
#       trainer's first cycle when the operator is hand-feeding it).
#
# Returning an empty list is allowed — empty windows are recorded
# but contribute no rules.
ObservationProvider = Callable[[int, int], Iterable[TradeObservation]]


def list_observation_provider(
    observations: Iterable[TradeObservation],
) -> ObservationProvider:
    """Build a provider that filters a fixed observation list by window.

    Convenience for tests + offline trainer drives. The list is
    materialised once and indexed by ts on first call.
    """
    cached = sorted(observations, key=lambda o: o.ts_ms)

    def _provider(start_ms: int, end_ms: int) -> list[TradeObservation]:
        return [o for o in cached if start_ms <= o.ts_ms < end_ms]

    return _provider


# --------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------- #


@dataclass
class WalkforwardTrainer:
    """Walk-forward driver.

    The trainer owns:
      * ``RulesPromoter`` (for the 80% gate + persistence)
      * a ``RuleAccumulator`` rebuilt per ``run`` so each cycle's
        validation_months_passed counts only the windows of *this* run

    It does *not* own:
      * The historical data adapter / matching engine — those live
        in the caller-supplied ``ObservationProvider``.
      * Wall-clock time — caller passes ``now_ts``.
    """

    cfg: WalkforwardTrainerConfig = field(default_factory=WalkforwardTrainerConfig)
    promoter: RulesPromoter | None = None

    def __post_init__(self) -> None:
        if self.promoter is None:
            self.promoter = RulesPromoter(
                cfg=self.cfg.promotion,
                state_dir=self.cfg.state_dir,
            )

    # ---- entrypoint ---- #

    def run(
        self,
        *,
        start_ms: int,
        end_ms: int,
        provider: ObservationProvider,
        now_ts: int | None = None,
    ) -> WalkforwardTrainerReport:
        """Iterate splits, mine rules, accumulate, promote.

        Returns a full report with per-window stats + post-promotion
        totals. Side-effect: writes ``production_rules.json`` /
        ``candidate_rules.json`` if ``state_dir`` is set.
        """
        ts = int(now_ts if now_ts is not None else time.time())
        wf_cfg = WalkForwardConfig(
            train_window_ms=self.cfg.train_window_ms,
            validate_window_ms=self.cfg.validate_window_ms,
            step_ms=self.cfg.step_ms,
        )
        accumulator = RuleAccumulator(
            qualifying_win_rate=self.cfg.promotion.min_win_rate,
            qualifying_min_samples=max(
                1, self.cfg.miner.min_samples_per_bucket,
            ),
        )

        report = WalkforwardTrainerReport(
            started_at_ts=ts,
            finished_at_ts=ts,
            splits=0,
        )

        for split in iter_splits(start_ms=start_ms, end_ms=end_ms, cfg=wf_cfg):
            window_report = self._run_one_window(
                split=split,
                provider=provider,
                accumulator=accumulator,
            )
            report.windows.append(window_report)
            report.splits += 1

        # Hand the pooled rules to the promoter once at the end of
        # the run. The promoter applies the 80% gate + persists.
        pooled = accumulator.flush()
        report.rules_after_pooling = len(pooled)
        promoted_now, demoted_now = self.promoter.promote_all(
            pooled, now_ts=ts,
        )
        report.rules_promoted_now = len(promoted_now)
        report.rules_demoted_now = len(demoted_now)
        report.rules_production_total = len(self.promoter.production_rules())
        report.rules_candidate_total = len(self.promoter.candidate_rules())
        report.finished_at_ts = int(time.time())
        return report

    # ---- internals ---- #

    def _run_one_window(
        self,
        *,
        split: WalkForwardSplit,
        provider: ObservationProvider,
        accumulator: RuleAccumulator,
    ) -> WalkforwardWindowReport:
        train_obs = list(provider(split.train.start_ms, split.train.end_ms))
        validate_obs = list(provider(split.validate.start_ms, split.validate.end_ms))

        # Mine rules from the *train* window only — the plan's
        # walk-forward methodology forbids mining on the validate
        # data. We then judge each train-mined rule by its win_rate
        # *on the validate window* before passing it to the
        # accumulator.
        train_rules = mine_rules(train_obs, self.cfg.miner)
        validated = self._regrade_on_validate(train_rules, validate_obs)
        accumulator.add_window(validated)

        # Per-window walk-forward stats use the validate-window PnLs.
        validate_pnls = [o.pnl_pct for o in validate_obs]
        win_stats = stats_from_pnls(validate_pnls, index=split.index)

        return WalkforwardWindowReport(
            index=split.index,
            train_start_ms=split.train.start_ms,
            train_end_ms=split.train.end_ms,
            validate_start_ms=split.validate.start_ms,
            validate_end_ms=split.validate.end_ms,
            train_observations=len(train_obs),
            validate_observations=len(validate_obs),
            rules_mined=len(train_rules),
            win_stats=win_stats,
        )

    def _regrade_on_validate(
        self,
        train_rules: list[LearnedRule],
        validate_obs: list[TradeObservation],
    ) -> list[LearnedRule]:
        """Score each ``train_rule`` by its validation-window stats.

        For each rule we built from the train window we measure how it
        performs on the *validate* window: collect all validate
        observations matching the rule's feature combo, recompute
        win_rate / avg_pnl / sharpe, and emit a fresh ``LearnedRule``
        with those numbers. Rules whose validate window has zero
        matching observations are dropped (no signal either way).

        This is the standard walk-forward discipline: never let
        in-sample stats leak into the production pool.
        """
        if not validate_obs:
            return []

        out: list[LearnedRule] = []
        for rule in train_rules:
            keys = rule.extras.get("feature_keys") or []
            target_value = rule.extras.get("value")
            if not keys or target_value is None:
                continue
            target_parts = target_value.split("/")
            if len(target_parts) != len(keys):
                continue

            matching: list[TradeObservation] = []
            for ob in validate_obs:
                try:
                    values = [str(ob.features[k]) for k in keys]
                except KeyError:
                    continue
                if values == target_parts:
                    matching.append(ob)
            if not matching:
                continue
            from altcoin_agent.training.rule_miner import _build_rule  # type: ignore
            regraded = _build_rule(
                combo_key="+".join(keys),
                value_key=target_value,
                bucket=matching,
                cfg=self.cfg.miner,
            )
            # Carry the train-window evidence as an extra so the
            # operator can compare in / out of sample.
            regraded.extras["train_samples"] = rule.samples
            regraded.extras["train_win_rate"] = rule.win_rate
            out.append(regraded)
        return out


__all__ = [
    "ObservationProvider",
    "WalkforwardTrainer",
    "WalkforwardTrainerConfig",
    "WalkforwardTrainerReport",
    "WalkforwardWindowReport",
    "list_observation_provider",
]
