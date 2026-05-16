"""slippage_model.py — Backtest slippage model (Phase B.4 / plan B.5).

The plan calls for:

    fill_price = mark_price × (1 + impact_pct + spread/2)
    impact_pct = sqrt(notional / top_depth_usdt) × impact_coeff
    spread     = base_spread + vol_premium × realized_vol_pct

This module ships the closed-form formula plus an
``observation -> coefficient`` calibration loader so the live daemon
can write actual fill slippages to
``.kiro/state/slippage_observations.jsonl`` and the backtest will
re-fit on next run. Calibration uses ordinary least squares on the
log-linearised form so we don't drag in ``sklearn``.

Design constraints:

* **Pure / numpy-free.** OLS is implemented by hand; the dataset is
  bounded (≤ 30k observations) so a Python loop is fine.
* **Side-aware.** For a LONG market entry the fill drifts *up* from the
  mark (taker pays more); for SHORT it drifts *down*. The single
  ``apply`` API returns a positive `fill_price` regardless.
* **Symmetric for stops.** A LONG stop-out is a SHORT market sell; the
  same formula applies with the opposite sign convention.
* **Deterministic.** No randomness — the matching engine is responsible
  for any noise injection if a Phase 5 trainer wants to stress-test.

The plan suggests ``base_spread = 5 bps`` for top altcoins and ``50 bps``
for shitcoins. The ``SlippageModel.from_quadrant(...)`` helper uses the
per-quadrant ``QuadrantParams`` to pick a default; the trainer can
override per symbol.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from altcoin_agent.risk.state import Side
from altcoin_agent.risk.symbol_profile import Quadrant

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlippageParams:
    """Closed-form slippage parameters.

    All percentages are *fractional*: ``base_spread=0.0005`` means 5 bps.
    ``impact_coeff`` calibrates the square-root market-impact term.
    ``vol_premium_coeff`` lets a high-volatility regime widen the spread.
    """

    base_spread: float = 0.0005           # 5 bps default (top altcoins)
    impact_coeff: float = 0.01            # plan default
    vol_premium_coeff: float = 0.02       # plan default
    taker_fee: float = 0.0004             # 4 bps Binance USDT-M default
    # Min fillable notional in USDT — below this we treat the fill as
    # zero-impact (no point modelling sub-dust noise).
    min_notional_usdt: float = 1.0
    # Hard ceiling on impact_pct so a degenerate (size >> depth) call
    # in a unit test doesn't return non-finite numbers.
    max_impact_pct: float = 0.10          # 1000 bps cap


# Per-quadrant defaults from the plan: A/B = top altcoins (5 bps base),
# C = manipulated mid-cap (15 bps), D = shitcoin (50 bps). Operators
# override via app.yaml when the trainer has enough fills to recalibrate.
_QUADRANT_DEFAULTS: dict[Quadrant, SlippageParams] = {
    Quadrant.A: SlippageParams(base_spread=0.0005),
    Quadrant.B: SlippageParams(base_spread=0.0008),
    Quadrant.C: SlippageParams(base_spread=0.0015),
    Quadrant.D: SlippageParams(base_spread=0.0050),
}


# --------------------------------------------------------------------- #
# Observation record (used by both calibration + audit log)
# --------------------------------------------------------------------- #


@dataclass
class SlippageObservation:
    """One realised fill — written by the live daemon, read by calibration.

    ``actual_slippage`` is signed in the *direction the price moved*
    against the trader: positive means the fill was worse than mark
    (taker disadvantage). Calibration regresses ``ln(actual_slippage)``
    against ``ln(notional / depth)`` and ``realized_vol_pct``.
    """

    ts_ms: int
    symbol: str
    side: str                  # "long" | "short"
    mark_price: float
    fill_price: float
    notional_usdt: float
    top_depth_usdt: float
    realized_vol_pct: float
    actual_slippage: float     # fractional, signed positive = worse-than-mark

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlippageObservation:
        return cls(
            ts_ms=int(d["ts_ms"]),
            symbol=str(d["symbol"]),
            side=str(d["side"]),
            mark_price=float(d["mark_price"]),
            fill_price=float(d["fill_price"]),
            notional_usdt=float(d["notional_usdt"]),
            top_depth_usdt=float(d["top_depth_usdt"]),
            realized_vol_pct=float(d["realized_vol_pct"]),
            actual_slippage=float(d["actual_slippage"]),
        )


# --------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------- #


@dataclass
class SlippageModel:
    """Stateless slippage formula + calibration helpers.

    ``params`` holds the coefficients; the model itself does no I/O —
    persistence is handled by ``load_observations`` / ``save_params``
    free functions.
    """

    params: SlippageParams = field(default_factory=SlippageParams)

    # ---- factory ---- #

    @classmethod
    def from_quadrant(cls, quadrant: Quadrant) -> SlippageModel:
        return cls(params=_QUADRANT_DEFAULTS[quadrant])

    # ---- core formula ---- #

    def estimate_slippage_pct(
        self,
        *,
        notional_usdt: float,
        top_depth_usdt: float,
        realized_vol_pct: float,
    ) -> float:
        """Return slippage as a fractional, *unsigned* number.

        ``apply`` adds the sign (LONG pays more, SHORT receives less).
        Clamps inputs so a misbehaving caller can't return NaN.
        """
        p = self.params
        n = max(0.0, float(notional_usdt))
        d = max(p.min_notional_usdt, float(top_depth_usdt))
        rv = max(0.0, float(realized_vol_pct))
        if n <= p.min_notional_usdt:
            # Sub-dust order: just the half-spread, no impact.
            return p.base_spread / 2.0
        # Square-root market impact: standard Kyle / Almgren shape.
        impact = math.sqrt(n / d) * p.impact_coeff
        impact = min(impact, p.max_impact_pct)
        spread = p.base_spread + p.vol_premium_coeff * rv
        return spread / 2.0 + impact

    def apply(
        self,
        *,
        side: Side,
        mark_price: float,
        notional_usdt: float,
        top_depth_usdt: float,
        realized_vol_pct: float,
        is_taker: bool = True,
    ) -> tuple[float, float]:
        """Return ``(fill_price, fee_paid_usdt)``.

        For a LONG market buy the fill is *above* mark; for a SHORT
        market sell it's *below*. ``fee_paid_usdt`` is the absolute
        taker fee on the *notional* value (always positive).
        """
        if mark_price <= 0:
            raise ValueError(f"mark_price must be positive, got {mark_price!r}")
        slip = self.estimate_slippage_pct(
            notional_usdt=notional_usdt,
            top_depth_usdt=top_depth_usdt,
            realized_vol_pct=realized_vol_pct,
        )
        # Taker eats the slippage in the direction that disadvantages them.
        sign = +1.0 if side is Side.LONG else -1.0
        fill_price = mark_price * (1.0 + sign * slip)
        fee = (
            notional_usdt * self.params.taker_fee if is_taker else 0.0
        )
        return float(fill_price), float(max(0.0, fee))


# --------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------- #


def load_observations(path: str) -> list[SlippageObservation]:
    """Read JSONL observations; skip malformed rows."""
    if not os.path.exists(path):
        return []
    out: list[SlippageObservation] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                obs = SlippageObservation.from_dict(d)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("slippage: skip bad obs %r: %s", line[:80], exc)
                continue
            out.append(obs)
    return out


def append_observation(path: str, obs: SlippageObservation) -> None:
    """Append one observation as JSONL. Caller ensures parent dir exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obs.as_dict(), sort_keys=True) + "\n")


def fit_params(
    observations: Iterable[SlippageObservation],
    *,
    fallback: SlippageParams | None = None,
    min_samples: int = 20,
) -> SlippageParams:
    """OLS-fit ``base_spread + impact_coeff * sqrt(n/d) + vol_premium_coeff * rv``.

    The model is a 3-term linear regression on absolute slippage:

        |slip| ≈ base_spread/2  +  impact_coeff * sqrt(n/d)  +  vol_coef * rv

    Three coefficients, no intercept beyond ``base_spread/2``. Below
    ``min_samples`` observations we return the ``fallback`` (or the
    plan defaults) — fitting on too little data overfits to noise.

    Coefficients are clamped to non-negative; the formula must never
    *credit* the trader for a market order.
    """
    obs = [o for o in observations if o.notional_usdt > 0 and o.top_depth_usdt > 0]
    if len(obs) < min_samples:
        return fallback or SlippageParams()

    # Build the design matrix manually so we stay numpy-free.
    # Rows: [1, sqrt(n/d), rv]   Target: |actual_slippage|
    # Solve normal equations: (X^T X) b = X^T y
    n = len(obs)
    # Accumulators for X^T X (3x3 symmetric) and X^T y (3-vec).
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for o in obs:
        sq = math.sqrt(o.notional_usdt / o.top_depth_usdt)
        rv = o.realized_vol_pct
        y = abs(o.actual_slippage)
        row = (1.0, sq, rv)
        for i in range(3):
            xty[i] += row[i] * y
            for j in range(3):
                xtx[i][j] += row[i] * row[j]
    coeffs = _solve_3x3(xtx, xty)
    if coeffs is None:
        logger.warning("slippage: OLS singular on n=%d obs; falling back", n)
        return fallback or SlippageParams()
    intercept, impact_coef, vol_coef = coeffs
    # The closed-form ``estimate_slippage_pct`` evaluates
    #     half_spread + impact_coef * sqrt(n/d) + vol_premium/2 * rv
    # because both the base spread and the vol-premium add to ``spread``
    # which is then halved for the fill. The regression therefore
    # recovers the half-magnitudes, and we double them back so the
    # fitted ``SlippageParams`` plug straight into the same formula
    # without a scale mismatch.
    base_spread = max(0.0, 2.0 * intercept)
    vol_premium = max(0.0, 2.0 * vol_coef)
    return SlippageParams(
        base_spread=base_spread,
        impact_coeff=max(0.0, impact_coef),
        vol_premium_coeff=vol_premium,
        taker_fee=(fallback or SlippageParams()).taker_fee,
        min_notional_usdt=(fallback or SlippageParams()).min_notional_usdt,
        max_impact_pct=(fallback or SlippageParams()).max_impact_pct,
    )


def _solve_3x3(
    a: list[list[float]], b: list[float]
) -> tuple[float, float, float] | None:
    """Tiny 3x3 linear solver via Cramer's rule. Returns None if singular."""
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    if abs(det) < 1e-18:
        return None

    def _det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    cols = []
    for i in range(3):
        m = [row[:] for row in a]
        for r in range(3):
            m[r][i] = b[r]
        cols.append(_det3(m) / det)
    return cols[0], cols[1], cols[2]


__all__ = [
    "SlippageModel",
    "SlippageObservation",
    "SlippageParams",
    "append_observation",
    "fit_params",
    "load_observations",
]
