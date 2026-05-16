"""quadrant_factory.py — Build per-quadrant risk components from SymbolProfile.

Phase 5 wiring layer: turns the four-quadrant strategy matrix
(``DEFAULT_QUADRANT_PARAMS``) into concrete, ready-to-use instances of
the existing risk-stack classes:

    SymbolProfile  →  PositionSizer
    SymbolProfile  →  RiskGateConfig
    SymbolProfile  →  TrailingStopFSM

The point: ``sizing.py`` / ``gate.py`` / ``trailing.py`` are unchanged
(zero risk to existing tests + main.py wiring); the daemon switches
between symbol-specific configurations by *constructing the right
instance per signal* via this module.

Why a factory and not a per-class quadrant param
------------------------------------------------
Three reasons:

1. **No churn in 60+ existing fixtures.** ``RiskGate(sizer, config)``,
   ``PositionSizer(...)``, ``TrailingStopFSM(...)`` all keep their
   exact signatures. Tests that build them with literal values
   continue to work.

2. **Single source of truth.** The plan's quadrant matrix lives in one
   place (``DEFAULT_QUADRANT_PARAMS``); this module is the only
   translator. Future trainer-driven param adjustments override
   ``QuadrantParams`` and propagate through the factory automatically.

3. **Operator override.** ``app.yaml -> risk.quadrant_overrides.A``
   can shadow the defaults; the factory accepts an optional override
   map and merges field-by-field. The plan deliberately makes this
   override-on-default rather than from-scratch so a partial override
   ("just bump A.max_risk_per_trade to 3%") still gets the rest of
   the matrix.

Determinism + isolation
-----------------------
Pure construction — no I/O, no clocks, no global state. Every call
returns fresh instances; callers may keep them in a per-signal scope
or cache them per quadrant. Tests inject ``QuadrantParams`` directly
to exercise edge cases without round-tripping through the matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from altcoin_agent.risk.gate import RiskGateConfig
from altcoin_agent.risk.sizing import DynamicLeverageConfig, PositionSizer
from altcoin_agent.risk.symbol_profile import (
    DEFAULT_QUADRANT_PARAMS,
    Quadrant,
    QuadrantParams,
    SymbolProfile,
)
from altcoin_agent.risk.trailing import TrailingStopFSM

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Factory bundle
# --------------------------------------------------------------------- #


@dataclass
class QuadrantRiskBundle:
    """Three concrete risk components for one quadrant.

    Yielded by ``QuadrantRiskFactory.bundle_for`` so the daemon can
    grab everything it needs in a single call.
    """

    quadrant: Quadrant
    params: QuadrantParams
    sizer: PositionSizer
    gate_config: RiskGateConfig
    trailing: TrailingStopFSM


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #


@dataclass
class QuadrantRiskFactory:
    """Build per-quadrant ``QuadrantRiskBundle`` instances on demand.

    ``overrides`` is a dict ``{Quadrant -> {field_name -> value}}``.
    Operators populate it from app.yaml; everything not listed falls
    through to ``DEFAULT_QUADRANT_PARAMS``.

    The factory is stateless beyond the override table; ``bundle_for``
    is safe to call from arbitrary threads / tasks.
    """

    overrides: dict[Quadrant, dict[str, Any]] = field(default_factory=dict)

    # ---- public API ---- #

    def params_for(self, quadrant: Quadrant) -> QuadrantParams:
        """Return the (possibly overridden) ``QuadrantParams``.

        Unknown override keys are logged + skipped so a stale operator
        config can't silently change behaviour outside the factory.
        """
        base = DEFAULT_QUADRANT_PARAMS[quadrant]
        ov = self.overrides.get(quadrant)
        if not ov:
            return base
        # ``replace`` validates field names; unknown keys raise.
        # We tolerate them by filtering and warning.
        valid_fields = {f for f in QuadrantParams.__dataclass_fields__}
        clean: dict[str, Any] = {}
        for k, v in ov.items():
            if k == "quadrant":
                # Never let an override change which quadrant we're in.
                continue
            if k not in valid_fields:
                logger.warning(
                    "QuadrantRiskFactory: unknown override key %r for %s; "
                    "ignoring", k, quadrant.value,
                )
                continue
            clean[k] = v
        return replace(base, **clean)

    def bundle_for(self, quadrant: Quadrant) -> QuadrantRiskBundle:
        params = self.params_for(quadrant)
        sizer = self._build_sizer(params)
        gate_cfg = self._build_gate_config(params)
        trailing = self._build_trailing(params)
        return QuadrantRiskBundle(
            quadrant=quadrant,
            params=params,
            sizer=sizer,
            gate_config=gate_cfg,
            trailing=trailing,
        )

    def bundle_for_profile(self, profile: SymbolProfile) -> QuadrantRiskBundle:
        """Convenience: pick the bundle from the symbol's profile.

        Identical to ``bundle_for(profile.quadrant)``; exposed so daemon
        code reads naturally::

            bundle = factory.bundle_for_profile(profile)
            sizer = bundle.sizer
        """
        return self.bundle_for(profile.quadrant)

    # ---- internals ---- #

    @staticmethod
    def _build_sizer(params: QuadrantParams) -> PositionSizer:
        lev = DynamicLeverageConfig(
            max_leverage_long=params.max_leverage_long,
            max_leverage_short=params.max_leverage_short,
        )
        return PositionSizer(
            max_risk_per_trade=params.max_risk_per_trade,
            leverage_cfg=lev,
        )

    @staticmethod
    def _build_gate_config(params: QuadrantParams) -> RiskGateConfig:
        # The plan's ``daily_drawdown_limit`` and (implicitly) the
        # liquidity floor differ per quadrant. We map them onto the
        # existing RiskGateConfig fields:
        return RiskGateConfig(
            daily_drawdown_limit=params.daily_drawdown_limit,
        )

    @staticmethod
    def _build_trailing(params: QuadrantParams) -> TrailingStopFSM:
        return TrailingStopFSM(
            atr_multiplier=params.trailing_atr_mult,
            breakeven_at_r=params.breakeven_at_r,
        )


__all__ = [
    "QuadrantRiskBundle",
    "QuadrantRiskFactory",
]
