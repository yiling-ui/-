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
        leverage: float | None = None,
    ) -> tuple[float, float, float]:
        """Returns (size_in_base, notional_usdt, risk_amount_usdt).

        Treats one contract as one unit of base (size_in_base equals quantity).
        The exchange-specific contract face value translation belongs in the
        executor, not here.

        Bug #1 fix — leverage cap on notional:

        Risk-parity sizing alone produces ``notional = risk_amount * entry /
        stop_distance``. With a tight stop (e.g. 0.05% on a sweep entry) this
        can balloon to 30-100x equity, blowing through Binance's leverage cap
        and the operator-configured ``max_leverage_long/short``. The exchange
        will either reject the order (-> emergency close + 4h cooldown) or
        worse, accept it on cross margin and quietly oversize the book.

        We now clamp ``notional <= equity * leverage`` and recompute
        ``risk_amount`` from the clamped size so the returned tuple honestly
        reflects what was actually committed.

        ``leverage`` is the dynamic leverage produced by ``compute_leverage``;
        when omitted we default to ``leverage_cfg.max_leverage_long`` (the
        side-agnostic upper bound) which preserves existing risk-parity
        behaviour for any caller that hasn't been updated yet.
        """
        if entry_price <= 0:
            return 0.0, 0.0, 0.0
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return 0.0, 0.0, 0.0
        if equity_usdt <= 0:
            return 0.0, 0.0, 0.0

        if leverage is None:
            leverage = self.leverage_cfg.max_leverage_long
        # Clamp leverage into a sane band so a stale or buggy upstream value
        # cannot inflate the notional past the configured side caps.
        max_lev_cap = max(
            self.leverage_cfg.max_leverage_long,
            self.leverage_cfg.max_leverage_short,
        )
        leverage = float(max(0.0, min(max_lev_cap, leverage)))
        if leverage <= 0:
            return 0.0, 0.0, 0.0

        risk_amount = equity_usdt * self.max_risk_per_trade
        notional_risk_parity = risk_amount * entry_price / stop_distance
        notional_cap = equity_usdt * leverage
        notional = min(notional_risk_parity, notional_cap)

        if notional < self.min_notional_usdt:
            return 0.0, 0.0, 0.0

        size_in_base = notional / entry_price
        # When we clamped, the realised dollar risk is smaller than the
        # configured ``max_risk_per_trade``. Recompute it so the gate's
        # bookkeeping reflects reality (this is what gets logged & shown on
        # the dashboard).
        actual_risk = stop_distance * size_in_base
        return size_in_base, notional, actual_risk
