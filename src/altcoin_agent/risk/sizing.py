"""sizing.py — Position sizing & dynamic leverage.

Risk-parity sizing:
    risk_amount   = equity * max_risk_per_trade
    stop_distance = |entry - initial_stop|
    notional_usdt = risk_amount / stop_distance * entry
    size          = notional_usdt / contract_value

Dynamic leverage (per architect call):
    leverage = clip(min_leverage + (max - min) * conf_norm * vol_adj * liq_adj,
                    min_leverage, side_cap)
where:
    conf_norm = (fused_score - high_priority_threshold)
                / (100 - high_priority_threshold), clipped to [0, 1]
    vol_adj   = min(1.0, target_vol_pct / max(realized_vol_pct, eps))
                — high realized vol -> smaller leverage
    liq_adj   = min(1.0, top5_depth_usdt / liq_full_depth_usdt)
                — thin book -> smaller leverage
SHORT side has a tighter cap (default 10x) than LONG (default 15x).
"""

from __future__ import annotations

from dataclasses import dataclass

from altcoin_agent.risk.state import Side


@dataclass
class DynamicLeverageConfig:
    min_leverage: float = 5.0
    max_leverage_long: float = 15.0
    max_leverage_short: float = 10.0
    target_vol_pct: float = 0.05         # 5%/h ATR is "neutral"
    liq_full_depth_usdt: float = 200_000.0  # depth at which liq_adj = 1.0
    score_anchor: float = 85.0


@dataclass
class PositionSizer:
    max_risk_per_trade: float = 0.015      # 1.5% of equity
    min_notional_usdt: float = 20.0        # exchange-side minimum
    leverage_cfg: DynamicLeverageConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.leverage_cfg is None:
            self.leverage_cfg = DynamicLeverageConfig()

    # ----------------------------- leverage ----------------------------- #

    def compute_leverage(
        self,
        *,
        side: Side,
        fused_score: float,
        realized_vol_pct: float,
        top5_depth_usdt: float,
    ) -> float:
        cfg = self.leverage_cfg
        side_cap = (
            cfg.max_leverage_long if side == Side.LONG else cfg.max_leverage_short
        )
        anchor = cfg.score_anchor
        conf_norm = max(0.0, min(1.0, (fused_score - anchor) / max(100.0 - anchor, 1e-9)))
        vol_adj = min(1.0, cfg.target_vol_pct / max(realized_vol_pct, 1e-4))
        liq_adj = min(1.0, top5_depth_usdt / max(cfg.liq_full_depth_usdt, 1.0))
        spread = side_cap - cfg.min_leverage
        leverage = cfg.min_leverage + spread * conf_norm * vol_adj * liq_adj
        return float(max(cfg.min_leverage, min(side_cap, leverage)))

    # ----------------------------- sizing ----------------------------- #

    def compute_size(
        self,
        *,
        equity_usdt: float,
        entry_price: float,
        initial_stop: float,
    ) -> tuple[float, float, float]:
        """Returns (size_in_base, notional_usdt, risk_amount_usdt).

        Treats one contract as one unit of base (size_in_base equals quantity).
        The exchange-specific contract face value translation belongs in the
        executor, not here.
        """
        if entry_price <= 0:
            return 0.0, 0.0, 0.0
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return 0.0, 0.0, 0.0
        risk_amount = equity_usdt * self.max_risk_per_trade
        size_quote = risk_amount / stop_distance * entry_price  # = risk_amount * entry / stop_distance
        notional = size_quote
        if notional < self.min_notional_usdt:
            return 0.0, 0.0, 0.0
        size_in_base = notional / entry_price
        return size_in_base, notional, risk_amount
