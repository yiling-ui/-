"""gate.py — RiskGate, the single hard wall before any order is placed.

Nine fail-closed checks, in order. Any single failure produces a rejection
with a reason. On unexpected exception the result is also a rejection
(fail-closed).

The 9 checks:
    1. Account globally halted (manual or auto)
    2. Reconciliation complete
    3. Daily drawdown circuit breaker
    4. Daily stop-loss hit count
    5. Per-symbol cooldown
    6. Per-symbol consecutive-loss cooldown
    7. Concurrent position cap
    8. Top-5 orderbook depth (liquidity)
    9. Slippage (SR-1, dynamic threshold by leverage)

Per architect call SR-1, the slippage threshold is asymmetric:
moves IN OUR FAVOUR are NEVER abort reasons. Only adverse drift counts.
The threshold itself shrinks with leverage:
    max_slippage = base_slippage / sqrt(leverage / 5)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Side

logger = logging.getLogger(__name__)


@dataclass
class RiskGateConfig:
    # Capacity / circuit breakers
    daily_drawdown_limit: float = 0.06      # 6% of equity
    daily_stoploss_hits_max: int = 3        # 3 stops in a day -> halt
    max_concurrent_positions: int = 3
    max_consecutive_losses: int = 2          # then per-symbol cooldown
    consecutive_loss_cooldown_sec: int = 4 * 3600
    # Liquidity
    min_liquidity_usdt: float = 200_000.0    # top-5 depth USDT
    # Slippage (SR-1)
    base_slippage: float = 0.03              # 3% at 5x leverage
    # Signal age
    max_signal_age_sec: int = 10
    # Per-symbol cooldown after high_priority emission
    symbol_cooldown_sec: int = 60


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    side: Side | None = None
    leverage: float | None = None
    size: float | None = None
    notional_usdt: float | None = None
    risk_amount_usdt: float | None = None
    initial_stop: float | None = None
    max_slippage_used: float | None = None


class RiskGate:
    """All trading goes through here. Nothing else can place orders."""

    def __init__(
        self,
        sizer: PositionSizer,
        config: RiskGateConfig | None = None,
    ):
        self.sizer = sizer
        self.cfg = config or RiskGateConfig()

    def evaluate(
        self,
        *,
        signal: FusedSignal,
        account: AccountState,
        current_price: float,
        top5_depth_usdt: float,
        realized_vol_pct: float,
        initial_stop: float,
        now_ms: int | None = None,
    ) -> RiskDecision:
        """Run all 9 checks. Returns an approved decision with sizing details
        on success, or a rejection with a reason on the first failure."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        try:
            # 0) signal must be a real, non-blocked, directional high_priority
            if signal.blocked:
                return RiskDecision(False, f"signal_blocked:{signal.block_reason}")
            if not signal.is_high_priority:
                return RiskDecision(False, "signal_not_high_priority")
            if signal.direction == Direction.NEUTRAL:
                return RiskDecision(False, "signal_direction_neutral")

            side = (
                Side.LONG if signal.direction == Direction.LONG else Side.SHORT
            )

            # 1) global halt
            if account.global_trading_halted:
                return RiskDecision(False, f"global_halt:{account.halt_reason}")

            # 2) reconciliation must be complete (SR-2)
            if not account.reconciliation_complete:
                return RiskDecision(False, "reconciliation_pending")

            # 3) daily drawdown
            if account.daily_drawdown_pct >= self.cfg.daily_drawdown_limit:
                return RiskDecision(
                    False,
                    f"daily_drawdown_limit:{account.daily_drawdown_pct:.4f}",
                )

            # 4) daily stop-loss hits
            if account.daily_stoploss_hits >= self.cfg.daily_stoploss_hits_max:
                return RiskDecision(False, "daily_stoploss_hits_exceeded")

            # 5) per-symbol cooldown
            if account.is_in_cooldown(signal.symbol, now_ms):
                return RiskDecision(False, "symbol_cooldown_active")

            # 6) per-symbol consecutive losses cooldown
            losses = account.consecutive_losses.get(signal.symbol, 0)
            if losses >= self.cfg.max_consecutive_losses:
                return RiskDecision(False, "consecutive_loss_cooldown")

            # 7) concurrency
            if len(account.open_positions) >= self.cfg.max_concurrent_positions:
                return RiskDecision(False, "max_concurrent_positions")

            # 8) liquidity
            if top5_depth_usdt < self.cfg.min_liquidity_usdt:
                return RiskDecision(
                    False,
                    f"insufficient_liquidity:{top5_depth_usdt:.0f}",
                )

            # 9) slippage check (SR-1) — uses dynamic leverage that we are
            # ABOUT to size with.
            leverage = self.sizer.compute_leverage(
                side=side,
                fused_score=signal.final_score,
                realized_vol_pct=realized_vol_pct,
                top5_depth_usdt=top5_depth_usdt,
            )
            trigger_price = signal.trigger_price or current_price
            max_slippage = self._dynamic_slippage_cap(leverage)
            slip = self._adverse_slip(side, trigger_price, current_price)
            if slip > max_slippage:
                return RiskDecision(
                    False,
                    f"slippage_too_high:{slip:.4f}>{max_slippage:.4f}@lev={leverage:.2f}",
                    max_slippage_used=max_slippage,
                )

            # 10) sizing
            size, notional, risk_amount = self.sizer.compute_size(
                equity_usdt=account.equity_usdt,
                entry_price=current_price,
                initial_stop=initial_stop,
            )
            if size <= 0 or notional <= 0:
                return RiskDecision(
                    False, "sizing_below_minimum", side=side, leverage=leverage,
                )

            return RiskDecision(
                approved=True,
                reason="ok",
                side=side,
                leverage=leverage,
                size=size,
                notional_usdt=notional,
                risk_amount_usdt=risk_amount,
                initial_stop=initial_stop,
                max_slippage_used=max_slippage,
            )

        except Exception as e:
            logger.exception("RiskGate unexpected failure: %s", e)
            return RiskDecision(False, f"unexpected_error:{type(e).__name__}")

    # ---------------- helpers ---------------- #

    def _dynamic_slippage_cap(self, leverage: float) -> float:
        """Slippage cap shrinks with leverage. At 5x = base; at 10x ~= base/√2."""
        denom = math.sqrt(max(leverage / 5.0, 1e-6))
        return self.cfg.base_slippage / denom

    @staticmethod
    def _adverse_slip(side: Side, trigger_price: float, current_price: float) -> float:
        """Slip in OUR DIRECTION OF HARM. Favourable moves return 0."""
        if trigger_price <= 0:
            return 0.0
        if side == Side.LONG:
            # bad = paying more than trigger
            return max(0.0, (current_price - trigger_price) / trigger_price)
        # short: bad = entering at a lower price than the trigger
        return max(0.0, (trigger_price - current_price) / trigger_price)
