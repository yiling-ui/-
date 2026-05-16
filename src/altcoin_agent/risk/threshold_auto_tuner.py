"""threshold_auto_tuner.py — opportunity-cost threshold tuner (Phase A.3).

Background
----------
``RejectReasonScorer`` (A.2) flags a reason like ``anti_chase`` as
"costing the operator money" when the +1 / -3 point ledger is deeply
negative. This module is the *actuator*: given the score table, it
emits a ``threshold_overrides.json`` file that ``main.py`` loads on
startup (or hot-reloads weekly) and merges into ``RiskGateConfig`` /
``PriceTapeConfig``.

Scope
-----
We only tune three parameters end-to-end (per ``MISS_PENALTY_AND_
PRODUCTION_PLAN.md§A.2.4``):

* ``anti_chase_max_move_pct``  (PriceTapeConfig)
* ``min_liquidity_usdt``        (RiskGateConfig)
* ``consecutive_loss_cooldown_sec``  (RiskGateConfig)

The mapping reason -> tuneable knob is hard-coded in
``DEFAULT_TUNING_RULES`` because (a) the set of RiskGate reasons is
small and stable, and (b) we explicitly want the operator to be able
to ``git diff`` what knobs the system can move.

Hard limits
-----------
Every tuneable parameter has a ``hard_min`` / ``hard_max`` pair the
tuner is forbidden to cross *no matter how many pumps we miss*. The
defaults match the four-quadrant strategy table (A象限 most-permissive
column) so the tuner can loosen up to "A 妖币 setting" but never
beyond. A reject-reason that's STILL net-negative at the hard bound
won't push past it -- the operator must intervene by hand.

Step size
---------
Loosening is incremental: each weekly run nudges the value by
``step_pct`` of the distance from current to hard limit. This keeps
us from over-correcting on a single week's data:

    new_value = current + step_pct * (hard_limit - current)

For ``step_pct = 0.20`` the value reaches 80% of the hard limit after
~7 weekly invocations, which gives the operator multiple opportunities
to observe the change and revert before it gets dangerous.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from altcoin_agent.risk.reject_reason_scorer import (
    RejectReasonScore,
    RejectReasonScorer,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Tuning rules
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TuneableKnob:
    """A single (reason, parameter) tuning relationship.

    ``param_path`` is the dotted name the override file emits, e.g.
    ``"price_tape.anti_chase_max_move_pct"``. ``main.py`` reads the
    same path and merges it into the relevant config object before
    instantiating gates.

    ``direction``: +1 means "increase to loosen the gate" (e.g. raise
    ``anti_chase_max_move_pct`` from 2.5% to 4% lets later entries
    through); -1 means "decrease to loosen" (e.g. lower
    ``min_liquidity_usdt`` from 200k to 100k accepts thinner books).

    ``hard_min`` / ``hard_max`` cap the tuner regardless of how
    negative the confidence score gets. Both bounds are absolute
    numerics in the same unit as the parameter.
    """

    reason: str
    param_path: str
    default: float
    hard_min: float
    hard_max: float
    direction: int  # +1 or -1
    step_pct: float = 0.20  # 20% step toward the hard limit per run

    def loosen(self, current: float) -> float:
        """Return the next value when the gate is too tight.

        ``direction == +1``  ->  step toward ``hard_max``
        ``direction == -1``  ->  step toward ``hard_min``
        """
        if self.direction > 0:
            target = self.hard_max
        else:
            target = self.hard_min
        delta = target - current
        if abs(delta) < 1e-9:
            return current
        return current + self.step_pct * delta

    def clamp(self, value: float) -> float:
        return max(self.hard_min, min(self.hard_max, value))


# Default mapping: reject-reason bucket -> tuneable knob.
#
# Defaults below match the live config in ``main.py``. The hard limits
# match QUADRANT_STRATEGY_PLAN's A-quadrant column (the most permissive
# operator-approved setting):
#
#   anti_chase_max_move_pct: default 2.5% -> max 6.0%  (A-quadrant)
#   min_liquidity_usdt:      default 200k -> min 100k  (A-quadrant)
#   consecutive_loss_cooldown_sec: default 4h -> min 1h
DEFAULT_TUNING_RULES: dict[str, TuneableKnob] = {
    "chase_too_late": TuneableKnob(
        reason="chase_too_late",
        param_path="price_tape.anti_chase_max_move_pct",
        default=0.025,
        hard_min=0.025,   # never tighter than current default
        hard_max=0.06,    # A-quadrant ceiling
        direction=+1,
    ),
    "insufficient_liquidity": TuneableKnob(
        reason="insufficient_liquidity",
        param_path="risk_gate.min_liquidity_usdt",
        default=200_000.0,
        hard_min=100_000.0,
        hard_max=200_000.0,  # never go ABOVE the safe default
        direction=-1,        # smaller = looser
    ),
    "consecutive_loss_cooldown": TuneableKnob(
        reason="consecutive_loss_cooldown",
        param_path="risk_gate.consecutive_loss_cooldown_sec",
        default=4 * 3600,
        hard_min=1 * 3600,
        hard_max=4 * 3600,
        direction=-1,
    ),
}


# --------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------- #


@dataclass
class ThresholdOverride:
    """One emitted override row.

    ``reason`` is what triggered the change; ``param_path`` is what
    main.py applies; ``previous`` / ``new`` show the trajectory so the
    operator can audit.
    """

    reason: str
    param_path: str
    previous: float
    new: float
    confidence_score: float
    samples: int
    missed_pumps: int
    direction_label: str  # "loosen" / "hold"
    last_updated_ts_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdAutoTunerConfig:
    """Tuning knobs for the tuner itself (yes, meta)."""

    overrides_state_path: str = ".kiro/state/miss_penalty/threshold_overrides.json"

    # When all conditions for ``RejectReasonScorer.should_loosen``
    # already match, do we *also* require the previous override to be
    # at least N seconds old? Defaults to 6 days so weekly runs don't
    # double-step within a single week if the cron fires twice.
    min_seconds_between_loosens: int = 6 * 24 * 3600

    # Optional override for the rule table -- tests inject custom
    # rules; production uses ``DEFAULT_TUNING_RULES``.
    rules: Mapping[str, TuneableKnob] = field(
        default_factory=lambda: dict(DEFAULT_TUNING_RULES),
    )


# --------------------------------------------------------------------- #
# Tuner
# --------------------------------------------------------------------- #


class ThresholdAutoTuner:
    """Reads scorer output, emits ``threshold_overrides.json``."""

    def __init__(
        self,
        *,
        scorer: RejectReasonScorer,
        config: ThresholdAutoTunerConfig | None = None,
        clock: callable[..., float] | None = None,  # type: ignore[name-defined]
    ):
        self.scorer = scorer
        self.cfg = config or ThresholdAutoTunerConfig()
        self._clock = clock or time.time

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tune(
        self,
        *,
        scores: dict[str, RejectReasonScore] | None = None,
    ) -> dict[str, ThresholdOverride]:
        """Compute new overrides for every loosenable reason.

        ``scores`` lets callers pass a pre-computed table (e.g.
        from ``RejectReasonScorer.recompute()`` already invoked
        upstream); otherwise we read the persisted scores via
        ``scorer.load()``.

        Returns the full override table the caller will publish to
        disk. Reasons that don't qualify for loosening retain their
        previous value (or default) so the file always round-trips
        the full set -- ``main.py`` doesn't have to handle missing
        keys.
        """
        if scores is None:
            scores = self.scorer.load()

        previous_overrides = self._load_previous()
        now_ms = int(self._clock() * 1000)
        new_overrides: dict[str, ThresholdOverride] = {}

        for reason, knob in self.cfg.rules.items():
            score = scores.get(reason)
            previous = previous_overrides.get(reason)
            current_value = (
                previous.new if previous is not None else knob.default
            )

            should_loosen = (
                score is not None
                and self.scorer.should_loosen(score)
                and self._cooldown_elapsed(previous, now_ms)
            )

            if should_loosen:
                proposed = knob.loosen(current_value)
                proposed = knob.clamp(proposed)
                direction_label = (
                    "hold" if abs(proposed - current_value) < 1e-9
                    else "loosen"
                )
            else:
                proposed = current_value
                direction_label = "hold"

            new_overrides[reason] = ThresholdOverride(
                reason=reason,
                param_path=knob.param_path,
                previous=current_value,
                new=proposed,
                confidence_score=(
                    score.confidence_score if score is not None else 0.0
                ),
                samples=score.total_audited if score is not None else 0,
                missed_pumps=(
                    score.missed_pumps if score is not None else 0
                ),
                direction_label=direction_label,
                last_updated_ts_ms=(
                    now_ms if direction_label == "loosen"
                    else (previous.last_updated_ts_ms if previous else now_ms)
                ),
            )

        self._persist(new_overrides)
        return new_overrides

    def load_overrides(self) -> dict[str, ThresholdOverride]:
        """Read the persisted overrides without re-tuning."""
        return self._load_previous()

    def emit_param_overrides(
        self, overrides: dict[str, ThresholdOverride] | None = None,
    ) -> dict[str, float]:
        """Flatten ``{reason -> ThresholdOverride}`` into
        ``{param_path -> value}`` for ``main.py`` to apply.

        When two reasons map to the same ``param_path`` (shouldn't
        happen with the default rules, but the user can plug in a
        custom map), the most-permissive value wins. "Most permissive"
        is "furthest from the default in the loosening direction".
        """
        if overrides is None:
            overrides = self._load_previous()
        out: dict[str, float] = {}
        for ov in overrides.values():
            knob = self.cfg.rules.get(ov.reason)
            if knob is None:
                continue
            existing = out.get(ov.param_path)
            if existing is None:
                out[ov.param_path] = ov.new
                continue
            # Pick the more-permissive (further from default) of the two.
            if knob.direction > 0:
                out[ov.param_path] = max(existing, ov.new)
            else:
                out[ov.param_path] = min(existing, ov.new)
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _cooldown_elapsed(
        self, prev: ThresholdOverride | None, now_ms: int,
    ) -> bool:
        if prev is None:
            return True
        delta_ms = now_ms - prev.last_updated_ts_ms
        return delta_ms >= self.cfg.min_seconds_between_loosens * 1000

    def _load_previous(self) -> dict[str, ThresholdOverride]:
        path = Path(self.cfg.overrides_state_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "ThresholdAutoTuner._load_previous failed: %s", e,
            )
            return {}
        out: dict[str, ThresholdOverride] = {}
        for row in data.get("overrides", []):
            if not isinstance(row, dict):
                continue
            try:
                clean = {
                    k: v for k, v in row.items()
                    if k in ThresholdOverride.__dataclass_fields__
                }
                out[row["reason"]] = ThresholdOverride(**clean)
            except (TypeError, KeyError):
                continue
        return out

    def _persist(self, overrides: dict[str, ThresholdOverride]) -> None:
        path = Path(self.cfg.overrides_state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at_ts_ms": int(self._clock() * 1000),
                "overrides": [
                    ov.to_dict()
                    for ov in sorted(
                        overrides.values(),
                        key=lambda o: o.reason,
                    )
                ],
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(
                "ThresholdAutoTuner._persist failed (swallowed): %s", e,
            )
