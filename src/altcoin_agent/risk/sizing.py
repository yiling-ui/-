"""
sizing.py — position sizing + dynamic leverage.

Implements the user-agreed leverage formula:

    leverage = clip(5 + 10 * conf_norm * vol_adj * liq_adj, 5, 15)

with short-side capped at 10. ``conf_norm`` is the part of fused score above
85, normalised to [0,1]. ``vol_adj`` falls inversely with realised volatility.
``liq_adj`` falls inversely with shallow orderbook depth.

Position size uses risk-parity:

    size_quote = (equity * risk_pct) / stop_distance_pct

where ``stop_distance_pct = abs(entry - initial_stop) / entry``. The size is
then bumped through leverage to a notional and rounded down to the contract
step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from altcoin_agent.risk.state import Side


@dataclass(frozen=True)
class OrderIntent:
    """A directional intent ready for sizing — the output of fuser+gate."""

    symbol: str
    exchange: str
    side: Side
    trigger_ts: int
    trigger_price: float
    entry_price: float        # the live price at intent creation (used for sizing)
    initial_stop: float
    fused_score: float        # 0..100 from fuser
    confidence: float         # 0..1 from LLM/fuser blend (informational)


@dataclass
class DynamicLeverageConfig:
    base: float = 5.0
    span: float = 10.0
    min_leverage: float = 5.0
    max_leverage_long: float = 15.0
    max_leverage_short: float = 10.0   # SR: shorts capped lower (mean-reversion risk)
    promote_threshold: float = 85.0    # below this, never trade (gate rejects earlier)


def compute_dynamic_leverage(
    *,
    side: Side,
    fused_score: float,
    realized_volatility_pct: float,
    book_depth_usdt_top5: float,
    min_liquidity_usdt: float = 200_000.0,
    target_volatility_pct: float = 0.02,  # 2% / 1h ATR target
    cfg: DynamicLeverageConfig | None = None,
) -> float:
    """
    Pure function. Returns a leverage in [min, max-by-side].

    - conf_norm: part of score above promote_threshold, scaled to [0,1].
    - vol_adj:   target / max(realized, target)  -> ≤ 1, smaller when volatile.
    - liq_adj:   min(1, depth / min_liquidity)  -> smaller when book is thin.
    """
    cfg = cfg or DynamicLeverageConfig()

    if fused_score < cfg.promote_threshold:
        return cfg.min_leverage

    conf_norm = max(0.0, min(1.0, (fused_score - cfg.promote_threshold) / (100.0 - cfg.promote_threshold)))

    rv = max(realized_volatility_pct, 1e-6)
    vol_adj = min(1.0, target_volatility_pct / rv)

    if min_liquidity_usdt <= 0:
        liq_adj = 1.0
    else:
        liq_adj = max(0.0, min(1.0, book_depth_usdt_top5 / min_liquidity_usdt))

    raw = cfg.base + cfg.span * conf_norm * vol_adj * liq_adj
    side_cap = cfg.max_leverage_short if side == Side.SHORT else cfg.max_leverage_long
    return max(cfg.min_leverage, min(side_cap, raw))


@dataclass
class SizingResult:
    leverage: float
    risk_amount_usdt: float
    notional_usdt: float
    size_contracts: float
    stop_distance: float


@dataclass
class PositionSizer:
    """
    Risk-parity sizing with the dynamic leverage above.

    Args:
        max_risk_per_trade: fraction of equity at risk per trade (default 1.5%)
        contract_step:      smallest tradable contract increment
        min_notional_usdt:  exchange-imposed min order size
    """

    max_risk_per_trade: float = 0.015
    leverage_cfg: DynamicLeverageConfig | None = None
    contract_step: float = 0.001
    min_notional_usdt: float = 5.0

    def compute(
        self,
        *,
        intent: OrderIntent,
        equity_usdt: float,
        realized_volatility_pct: float,
        book_depth_usdt_top5: float,
        min_liquidity_usdt: float = 200_000.0,
    ) -> SizingResult:
        if equity_usdt <= 0:
            raise ValueError("equity_usdt must be positive")

        if intent.entry_price <= 0:
            raise ValueError("entry_price must be positive")

        stop_dist = abs(intent.entry_price - intent.initial_stop)
        if stop_dist <= 0:
            raise ValueError("initial_stop must differ from entry_price")

        leverage = compute_dynamic_leverage(
            side=intent.side,
            fused_score=intent.fused_score,
            realized_volatility_pct=realized_volatility_pct,
            book_depth_usdt_top5=book_depth_usdt_top5,
            min_liquidity_usdt=min_liquidity_usdt,
            cfg=self.leverage_cfg,
        )

        risk_amount = equity_usdt * self.max_risk_per_trade
        # how much notional we need so that stop_dist movement equals risk_amount
        notional = risk_amount * intent.entry_price / stop_dist
        # leverage caps the EFFECTIVE risk to margin available; we keep the
        # risk-parity sizing and rely on leverage purely to reduce required
        # margin, not to scale up risk. We do, however, cap notional to
        # equity * leverage to respect exchange margin.
        notional = min(notional, equity_usdt * leverage)

        size_contracts = notional / intent.entry_price
        # round DOWN to step
        if self.contract_step > 0:
            size_contracts = math.floor(size_contracts / self.contract_step) * self.contract_step

        # final notional after rounding
        final_notional = size_contracts * intent.entry_price

        if final_notional < self.min_notional_usdt:
            # rounding would zero us out; reject by returning size 0
            size_contracts = 0.0
            final_notional = 0.0

        return SizingResult(
            leverage=leverage,
            risk_amount_usdt=risk_amount,
            notional_usdt=final_notional,
            size_contracts=size_contracts,
            stop_distance=stop_dist,
        )
