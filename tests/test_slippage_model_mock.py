"""tests/test_slippage_model_mock.py — Phase B.4 slippage formula + calibration."""

from __future__ import annotations

import json
import os

import pytest

from altcoin_agent.backtest.slippage_model import (
    SlippageModel,
    SlippageObservation,
    SlippageParams,
    append_observation,
    fit_params,
    load_observations,
)
from altcoin_agent.risk.state import Side
from altcoin_agent.risk.symbol_profile import Quadrant

# ----------------------- formula ----------------------- #


def test_long_pays_more_than_mark():
    m = SlippageModel(SlippageParams(base_spread=0.0010, impact_coeff=0.01))
    fill, fee = m.apply(
        side=Side.LONG, mark_price=100.0,
        notional_usdt=1_000.0, top_depth_usdt=100_000.0, realized_vol_pct=0.0,
    )
    assert fill > 100.0
    assert fee > 0.0  # taker fee on 1000 USDT


def test_short_receives_less_than_mark():
    m = SlippageModel(SlippageParams(base_spread=0.0010, impact_coeff=0.01))
    fill, _fee = m.apply(
        side=Side.SHORT, mark_price=100.0,
        notional_usdt=1_000.0, top_depth_usdt=100_000.0, realized_vol_pct=0.0,
    )
    assert fill < 100.0


def test_larger_notional_increases_slippage():
    m = SlippageModel()
    small = m.estimate_slippage_pct(
        notional_usdt=100.0, top_depth_usdt=100_000.0, realized_vol_pct=0.0,
    )
    big = m.estimate_slippage_pct(
        notional_usdt=10_000.0, top_depth_usdt=100_000.0, realized_vol_pct=0.0,
    )
    assert big > small


def test_higher_realised_vol_increases_slippage():
    m = SlippageModel(SlippageParams(vol_premium_coeff=0.10))
    calm = m.estimate_slippage_pct(
        notional_usdt=1000.0, top_depth_usdt=100_000.0, realized_vol_pct=0.0,
    )
    storm = m.estimate_slippage_pct(
        notional_usdt=1000.0, top_depth_usdt=100_000.0, realized_vol_pct=0.10,
    )
    assert storm > calm


def test_impact_clamped_at_max():
    m = SlippageModel(SlippageParams(
        base_spread=0.0,
        impact_coeff=10_000.0,           # absurd
        max_impact_pct=0.05,
    ))
    slip = m.estimate_slippage_pct(
        notional_usdt=1_000_000.0, top_depth_usdt=100.0, realized_vol_pct=0.0,
    )
    assert slip <= 0.05 + 1e-9


def test_zero_mark_price_rejected():
    with pytest.raises(ValueError):
        SlippageModel().apply(
            side=Side.LONG, mark_price=0.0,
            notional_usdt=1000.0, top_depth_usdt=10_000.0,
            realized_vol_pct=0.0,
        )


def test_quadrant_defaults_widen_with_quality_drop():
    a = SlippageModel.from_quadrant(Quadrant.A)
    d = SlippageModel.from_quadrant(Quadrant.D)
    assert a.params.base_spread < d.params.base_spread


def test_dust_order_uses_half_spread_only():
    m = SlippageModel(SlippageParams(base_spread=0.0010, min_notional_usdt=10.0))
    slip = m.estimate_slippage_pct(
        notional_usdt=1.0, top_depth_usdt=10_000.0, realized_vol_pct=0.0,
    )
    assert slip == pytest.approx(0.0005)


# ----------------------- I/O + calibration ----------------------- #


def test_observation_round_trip():
    o = SlippageObservation(
        ts_ms=1, symbol="X", side="long", mark_price=100.0, fill_price=100.5,
        notional_usdt=1000.0, top_depth_usdt=50_000.0,
        realized_vol_pct=0.05, actual_slippage=0.005,
    )
    d = o.as_dict()
    o2 = SlippageObservation.from_dict(d)
    assert o == o2


def test_load_observations_skips_corrupt_rows(tmp_path):
    path = str(tmp_path / "obs.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts_ms": 1, "symbol": "X", "side": "long", "mark_price": 100,
            "fill_price": 100.5, "notional_usdt": 1000, "top_depth_usdt": 50000,
            "realized_vol_pct": 0.0, "actual_slippage": 0.005,
        }) + "\n")
        fh.write("not json\n")
        fh.write("{\"missing\":\"fields\"}\n")
        fh.write("\n")
    obs = load_observations(path)
    assert len(obs) == 1
    assert obs[0].symbol == "X"


def test_load_observations_missing_file_returns_empty(tmp_path):
    assert load_observations(str(tmp_path / "absent.jsonl")) == []


def test_append_observation_creates_parent_dir(tmp_path):
    path = str(tmp_path / "subdir" / "obs.jsonl")
    o = SlippageObservation(
        ts_ms=1, symbol="X", side="long", mark_price=100.0, fill_price=100.5,
        notional_usdt=1000.0, top_depth_usdt=50_000.0,
        realized_vol_pct=0.05, actual_slippage=0.005,
    )
    append_observation(path, o)
    assert os.path.exists(path)
    obs = load_observations(path)
    assert len(obs) == 1


def test_fit_below_min_samples_returns_fallback():
    fallback = SlippageParams(base_spread=0.0099)
    out = fit_params([], fallback=fallback, min_samples=10)
    assert out is fallback or out.base_spread == 0.0099


def test_fit_recovers_known_coefficients():
    """Synthesize observations from known params, refit, expect close match."""
    truth = SlippageParams(
        base_spread=0.0010, impact_coeff=0.02, vol_premium_coeff=0.05,
    )
    m = SlippageModel(truth)

    obs: list[SlippageObservation] = []
    # Vary notional, depth, vol over a grid.
    for n in (500, 1000, 2000, 5000, 10_000):
        for d in (20_000, 50_000, 100_000):
            for rv in (0.0, 0.02, 0.05, 0.10):
                slip = m.estimate_slippage_pct(
                    notional_usdt=n, top_depth_usdt=d, realized_vol_pct=rv,
                )
                obs.append(SlippageObservation(
                    ts_ms=0, symbol="X", side="long",
                    mark_price=1.0, fill_price=1.0 + slip,
                    notional_usdt=n, top_depth_usdt=d,
                    realized_vol_pct=rv, actual_slippage=slip,
                ))
    fitted = fit_params(obs, min_samples=10)
    assert fitted.base_spread == pytest.approx(truth.base_spread, abs=5e-4)
    assert fitted.impact_coeff == pytest.approx(truth.impact_coeff, abs=5e-3)
    assert fitted.vol_premium_coeff == pytest.approx(
        truth.vol_premium_coeff, abs=5e-3,
    )


def test_fit_clamps_negative_coefficients_to_zero():
    """A degenerate dataset where regression yields negative slope must
    not be allowed to credit the trader."""
    obs = [
        SlippageObservation(
            ts_ms=0, symbol="X", side="long", mark_price=1.0, fill_price=1.0,
            notional_usdt=n, top_depth_usdt=10_000.0,
            realized_vol_pct=0.0, actual_slippage=0.0,
        )
        for n in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000) * 3
    ]
    fitted = fit_params(obs, min_samples=10)
    assert fitted.base_spread >= 0.0
    assert fitted.impact_coeff >= 0.0
    assert fitted.vol_premium_coeff >= 0.0
