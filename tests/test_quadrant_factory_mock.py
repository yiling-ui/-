"""tests/test_quadrant_factory_mock.py — Phase 5 quadrant -> risk wiring."""

from __future__ import annotations

import logging

import pytest

from altcoin_agent.risk.quadrant_factory import (
    QuadrantRiskBundle,
    QuadrantRiskFactory,
)
from altcoin_agent.risk.symbol_profile import (
    DEFAULT_QUADRANT_PARAMS,
    Quadrant,
    SymbolProfile,
)

# ----------------------- params_for ----------------------- #


@pytest.mark.parametrize("q", list(Quadrant))
def test_default_params_match_matrix(q):
    """No overrides → factory returns the plan's matrix verbatim."""
    f = QuadrantRiskFactory()
    p = f.params_for(q)
    assert p == DEFAULT_QUADRANT_PARAMS[q]


def test_override_replaces_only_listed_field():
    f = QuadrantRiskFactory(
        overrides={Quadrant.A: {"max_risk_per_trade": 0.030}},
    )
    a = f.params_for(Quadrant.A)
    assert a.max_risk_per_trade == 0.030
    # All other fields stayed at the matrix defaults.
    base = DEFAULT_QUADRANT_PARAMS[Quadrant.A]
    assert a.max_leverage_long == base.max_leverage_long
    assert a.confidence_threshold == base.confidence_threshold


def test_override_is_per_quadrant():
    f = QuadrantRiskFactory(
        overrides={Quadrant.A: {"max_risk_per_trade": 0.030}},
    )
    # B is unaffected.
    b = f.params_for(Quadrant.B)
    assert b == DEFAULT_QUADRANT_PARAMS[Quadrant.B]


def test_override_cannot_change_quadrant_field():
    """Even if an operator includes ``quadrant: "Z"`` in app.yaml,
    the factory must refuse to mutate it."""
    f = QuadrantRiskFactory(
        overrides={Quadrant.A: {"quadrant": "B"}},
    )
    a = f.params_for(Quadrant.A)
    assert a.quadrant is Quadrant.A


def test_unknown_override_key_logged_and_ignored(caplog):
    f = QuadrantRiskFactory(
        overrides={Quadrant.A: {"this_does_not_exist": 99}},
    )
    with caplog.at_level(logging.WARNING):
        a = f.params_for(Quadrant.A)
    assert a == DEFAULT_QUADRANT_PARAMS[Quadrant.A]
    assert any("unknown override key" in r.message for r in caplog.records)


# ----------------------- bundle_for ----------------------- #


def test_bundle_for_a_quadrant_carries_matrix_values():
    f = QuadrantRiskFactory()
    b = f.bundle_for(Quadrant.A)
    assert b.quadrant is Quadrant.A
    assert b.params is DEFAULT_QUADRANT_PARAMS[Quadrant.A]
    # Sizer
    assert b.sizer.max_risk_per_trade == 0.025
    assert b.sizer.leverage_cfg.max_leverage_long == 15.0
    assert b.sizer.leverage_cfg.max_leverage_short == 10.0
    # Gate
    assert b.gate_config.daily_drawdown_limit == 0.12
    # Trailing
    assert b.trailing.atr_multiplier == 2.5
    assert b.trailing.breakeven_at_r == 1.5


def test_bundle_for_d_quadrant_is_strictest():
    f = QuadrantRiskFactory()
    b = f.bundle_for(Quadrant.D)
    assert b.sizer.max_risk_per_trade == 0.005
    assert b.gate_config.daily_drawdown_limit == 0.06
    assert b.trailing.atr_multiplier == 0.5


def test_bundle_for_profile_picks_right_quadrant():
    f = QuadrantRiskFactory()
    profile = SymbolProfile.from_scores("X", 90, 90)
    assert profile.quadrant is Quadrant.A
    b = f.bundle_for_profile(profile)
    assert b.quadrant is Quadrant.A
    assert b.sizer.max_risk_per_trade == 0.025


def test_bundles_are_independent_instances():
    """Mutating one bundle's sizer must not bleed into another."""
    f = QuadrantRiskFactory()
    b1 = f.bundle_for(Quadrant.A)
    b2 = f.bundle_for(Quadrant.A)
    assert b1 is not b2
    assert b1.sizer is not b2.sizer
    b1.sizer.max_risk_per_trade = 0.99
    assert b2.sizer.max_risk_per_trade == 0.025


def test_override_propagates_into_bundle():
    f = QuadrantRiskFactory(
        overrides={
            Quadrant.B: {
                "max_risk_per_trade": 0.020,
                "trailing_atr_mult": 2.0,
                "daily_drawdown_limit": 0.10,
            }
        }
    )
    b = f.bundle_for(Quadrant.B)
    assert b.sizer.max_risk_per_trade == 0.020
    assert b.trailing.atr_multiplier == 2.0
    assert b.gate_config.daily_drawdown_limit == 0.10


def test_short_leverage_cap_propagates():
    """The plan's matrix has max_leverage_short = 5x for D-quadrant.
    Sizer.compute_size must clamp to that, not the long cap."""
    f = QuadrantRiskFactory()
    b = f.bundle_for(Quadrant.D)
    assert b.sizer.leverage_cfg.max_leverage_short == 5.0
    assert b.sizer.leverage_cfg.max_leverage_long == 5.0


# ----------------------- shape ----------------------- #


def test_bundle_shape_matches_dataclass():
    f = QuadrantRiskFactory()
    b = f.bundle_for(Quadrant.A)
    assert isinstance(b, QuadrantRiskBundle)
    # Spot-check: every field is populated.
    assert b.params is not None
    assert b.sizer is not None
    assert b.gate_config is not None
    assert b.trailing is not None
