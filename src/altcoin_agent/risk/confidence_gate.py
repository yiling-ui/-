"""confidence_gate.py — 80% conviction gate (QUADRANT Phase 1.D).

Combines five inputs into a single confidence in [0, 1]:

    rule_score          screener+fuser composite, normalised to [0, 1]
    llm_score           DeepSeek (or other LLM) verdict prob, [0, 1]
    profile             SymbolProfile (quadrant, hist win rate, scam score)
    phase               PumpPhaseFSM current state
    learned_rules_score [0, 1] from learning_engine's production rules

The gate enforces the plan's hardest constraint:

    confidence < quadrant.confidence_threshold (default 0.80) -> NO open.

A signal that would have opened but failed the gate is *still emitted*
to ai_engine in **observer mode** so the trainer keeps a labeled negative.

This module is intentionally orthogonal to ``RiskGate``: ``RiskGate`` does
the per-account risk math (consecutive losses, cooldowns, cluster cap,
notional caps), ``ConfidenceGate`` does the per-symbol conviction math.
Both must pass before an order leaves the daemon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from altcoin_agent.risk.pump_phase import PumpPhase
from altcoin_agent.risk.symbol_profile import SymbolProfile

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Mixing weights (tunable but not per-quadrant for v1)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights summing to 1.0 across the four primary inputs.

    The phase modifier is applied multiplicatively *after* the linear
    blend so that PARABOLIC can boost A-quadrant conviction without
    letting it leak to D where conviction should stay low.
    """

    rule: float = 0.40
    llm: float = 0.30
    profile_history: float = 0.15
    learned: float = 0.15

    def __post_init__(self) -> None:
        total = self.rule + self.llm + self.profile_history + self.learned
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ConfidenceWeights must sum to 1.0; got {total:.4f}"
            )


# Phase-multipliers: how much each phase adjusts the linear blend.
# Multiplicative, clamped to [0, 1] in the gate.
DEFAULT_PHASE_MULTIPLIER: dict[PumpPhase, float] = {
    PumpPhase.ACCUMULATION: 0.85,  # be slow on early accumulation
    PumpPhase.RAMP: 1.00,
    PumpPhase.PARABOLIC: 1.05,     # high vol z confirmed
    PumpPhase.BLOWOFF_TOP: 0.70,   # too late for fresh longs
    PumpPhase.CRASH: 0.55,         # only short-side conviction makes sense
    PumpPhase.BLEED: 0.40,
    PumpPhase.DEAD: 0.20,
}


# --------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------- #


@dataclass
class ConfidenceVerdict:
    """Result of evaluating a single signal.

    ``should_open`` and ``observer_mode`` are mutually exclusive:
        approved   -> should_open=True,  observer_mode=False
        rejected   -> should_open=False, observer_mode=True
                      (still recorded for training)
    """

    confidence: float
    threshold: float
    should_open: bool
    observer_mode: bool
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, float | bool | str | dict[str, float]]:
        return {
            "confidence": self.confidence,
            "threshold": self.threshold,
            "should_open": self.should_open,
            "observer_mode": self.observer_mode,
            "components": dict(self.components),
            "reason": self.reason,
        }


# --------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------- #


@dataclass
class ConfidenceGate:
    """Stateless mixer + threshold gate.

    No internal mutable state means the same gate instance is safe to
    share across symbols and across the live + backtest paths.
    """

    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    phase_multiplier: dict[PumpPhase, float] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_MULTIPLIER)
    )
    # Floor under which we never approve, regardless of quadrant override.
    hard_min_threshold: float = 0.50

    def evaluate(
        self,
        *,
        rule_score: float,
        llm_score: float,
        learned_rules_score: float,
        profile: SymbolProfile,
        phase: PumpPhase,
    ) -> ConfidenceVerdict:
        """Compute confidence and decide.

        ``rule_score`` from fuser is on a 0..100 scale; we normalise here.
        ``llm_score`` and ``learned_rules_score`` are already [0, 1]. We
        clamp every input so that a buggy upstream (e.g. NaN) maps to a
        defensive zero rather than poisoning the blend.
        """
        rule01 = _clamp01(rule_score / 100.0)
        llm01 = _clamp01(llm_score)
        learned01 = _clamp01(learned_rules_score)
        hist01 = _clamp01(profile.historical_win_rate)

        w = self.weights
        linear = (
            w.rule * rule01
            + w.llm * llm01
            + w.profile_history * hist01
            + w.learned * learned01
        )
        phase_mult = self.phase_multiplier.get(phase, 1.0)
        # Scam-score penalty: a symbol with a high known-scam score gets
        # its confidence shrunk linearly. scam_score is also expected in
        # [0, 1].
        scam_penalty = 1.0 - 0.5 * _clamp01(profile.scam_score)
        confidence = _clamp01(linear * phase_mult * scam_penalty)

        threshold = max(
            self.hard_min_threshold,
            profile.effective_confidence_threshold(),
        )
        approved = confidence >= threshold

        components = {
            "rule": rule01,
            "llm": llm01,
            "learned": learned01,
            "history": hist01,
            "linear_blend": linear,
            "phase_mult": phase_mult,
            "scam_penalty": scam_penalty,
        }
        reason = (
            "approved" if approved
            else f"below_threshold:conf={confidence:.3f}<thr={threshold:.3f}"
        )
        return ConfidenceVerdict(
            confidence=confidence,
            threshold=threshold,
            should_open=approved,
            observer_mode=not approved,
            components=components,
            reason=reason,
        )


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


def _clamp01(x: float) -> float:
    """Clamp ``x`` to [0, 1], coercing NaN to 0.0."""
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


__all__ = [
    "ConfidenceGate",
    "ConfidenceVerdict",
    "ConfidenceWeights",
    "DEFAULT_PHASE_MULTIPLIER",
]
