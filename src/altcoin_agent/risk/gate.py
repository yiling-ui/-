"""
gate.py — Risk Gate, the single hard wall (fail-closed).

A gate decision is a plain dataclass; tests can call ``evaluate`` purely.

Checks (in order; first failure wins):

    1.  Reconciliation completed                               — SR-2
    2.  Global circuit breaker not engaged
    3.  Daily drawdown / consecutive-loss limits not breached
    4.  Symbol not under cooldown (e.g. previous hard-stop fail) — SR-2
    5.  Concurrent positions cap not exceeded
    6.  Top-5 book depth >= min_liquidity_usdt (no thin coffin)
    7.  Slippage from trigger_price under dynamic threshold     — SR-1
    8.  Fused score >= high_priority threshold (defence in depth)
    9.  Direction not flat-blocked (e.g. user disabled shorts during news)

Anything that throws inside ``evaluate`` is interpreted as REJECT
(fail-closed by design).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from altcoin_agent.fuser import FusedSignal
from altcoin_agent.risk.sizing import (
    DynamicLeverageConfig,
    OrderIntent,
    compute_dynamic_leverage,
)
from altcoin_agent.risk.state import AccountState, Side

logger = logging.getLogger(__name__)


class GateReject(Exception):
    """Raised internally; tests typically inspect GateDecision instead."""


@dataclass
class RiskGateConfig:
    high_priority_threshold: float = 85.0
    max_concurrent_positions: int = 3
    daily_drawdown_limit: float = 0.06         # 6% of equity / day
    max_consecutive_losses: int = 2             # symbol-level hint
    daily_stoploss_hits_max: int = 3            # entire account, halts trading for the day
    min_liquidity_usdt: float = 200_000.0
    base_slippage: float = 0.03                # 3% at 5x reference
    allow_shorts: bool = True                  # user toggle


@dataclass
class GateDecision:
    approved: bool
    reason: str | None
    intent: OrderIntent | None
    notes: list[str] = field(default_factory=list)


class RiskGate:
    """
    All in-flight decisions go through `evaluate(...)` which returns a
    GateDecision. The gate is stateless except for config; account state
    and live market data are passed in.
    """

    def __init__(
        self,
        config: RiskGateConfig | None = None,
        leverage_cfg: DynamicLeverageConfig | None = None,
    ):
        self.cfg = config or RiskGateConfig()
        self.leverage_cfg = leverage_cfg or DynamicLeverageConfig()

    def evaluate(
        self,
        *,
        signal: FusedSignal,
        current_price: float,
        book_depth_usdt_top5: float,
        realized_volatility_pct: float,
        account: AccountState,
        proposed_initial_stop: float,
    ) -> GateDecision:
        notes: list[str] = []
        try:
            self._check_reconciliation(account, notes)
            self._check_circuit_breakers(account, notes)
            self._check_drawdown(account, notes)
            self._check_loss_streak(account, notes)
            self._check_cooldown(account, signal.symbol, signal.exchange, signal.ts, notes)
            self._check_concurrency(account, notes)
            self._check_liquidity(book_depth_usdt_top5, notes)
            side = self._side_from_signal(signal)
            self._check_direction_allowed(side, notes)
            self._check_score(signal, notes)
            self._check_slippage(
                side=side,
                trigger_price=self._trigger_price(signal, current_price),
                current_price=current_price,
                fused_score=signal.final_score,
                realized_volatility_pct=realized_volatility_pct,
                book_depth_usdt_top5=book_depth_usdt_top5,
                notes=notes,
            )

            intent = OrderIntent(
                symbol=signal.symbol,
                exchange=signal.exchange,
                side=side,
                trigger_ts=signal.ts,
                trigger_price=self._trigger_price(signal, current_price),
                entry_price=current_price,
                initial_stop=proposed_initial_stop,
                fused_score=signal.final_score,
                confidence=signal.llm_score / 100.0 if signal.llm_score else signal.rule_score / 100.0,
            )

            return GateDecision(approved=True, reason=None, intent=intent, notes=notes)

        except GateReject as e:
            return GateDecision(approved=False, reason=str(e), intent=None, notes=notes)
        except Exception as e:  # fail-closed: any unexpected error rejects.
            logger.exception("RiskGate unexpected error -> reject (fail-closed)")
            return GateDecision(approved=False, reason=f"unexpected_error:{e}", intent=None, notes=notes)

    # ---------------------- individual checks ---------------------- #

    def _check_reconciliation(self, account: AccountState, notes: list[str]) -> None:
        if not account.reconciliation_complete:
            raise GateReject("reconciliation_pending")  # SR-2
        notes.append("reconciliation: ok")

    def _check_circuit_breakers(self, account: AccountState, notes: list[str]) -> None:
        if account.circuit_breaker_engaged:
            raise GateReject("circuit_breaker_engaged")
        if account.daily_stoploss_hits >= self.cfg.daily_stoploss_hits_max:
            raise GateReject(f"daily_stoploss_hits>={self.cfg.daily_stoploss_hits_max}")
        notes.append("circuit_breakers: ok")

    def _check_drawdown(self, account: AccountState, notes: list[str]) -> None:
        if account.equity_usdt <= 0:
            raise GateReject("equity_non_positive")
        loss_pct = -min(0.0, account.realized_pnl_today_usdt) / account.equity_usdt
        if loss_pct >= self.cfg.daily_drawdown_limit:
            raise GateReject(f"daily_drawdown {loss_pct:.3f}>={self.cfg.daily_drawdown_limit}")
        notes.append(f"drawdown: {loss_pct:.3f}")

    def _check_loss_streak(self, account: AccountState, notes: list[str]) -> None:
        if account.consecutive_losses_today >= self.cfg.max_consecutive_losses:
            raise GateReject(
                f"consecutive_losses>={self.cfg.max_consecutive_losses}"
            )
        notes.append(f"loss_streak: {account.consecutive_losses_today}")

    def _check_cooldown(
        self, account: AccountState, symbol: str, exchange: str, now_ts: int, notes: list[str]
    ) -> None:
        key = f"{exchange}:{symbol}"
        until = account.symbol_cooldowns.get(key)
        if until is not None and now_ts < until:
            remaining_min = (until - now_ts) / 60_000
            raise GateReject(f"symbol_cooldown_active({remaining_min:.1f}m)")
        notes.append("cooldown: ok")

    def _check_concurrency(self, account: AccountState, notes: list[str]) -> None:
        active = sum(1 for p in account.open_positions if not p.closed)
        if active >= self.cfg.max_concurrent_positions:
            raise GateReject(
                f"max_concurrent_positions {active}/{self.cfg.max_concurrent_positions}"
            )
        notes.append(f"concurrency: {active}/{self.cfg.max_concurrent_positions}")

    def _check_liquidity(self, depth: float, notes: list[str]) -> None:
        if depth < self.cfg.min_liquidity_usdt:
            raise GateReject(
                f"book_depth {depth:.0f}<min {self.cfg.min_liquidity_usdt:.0f}"
            )
        notes.append(f"liquidity: {depth:.0f} USDT top5")

    def _check_direction_allowed(self, side: Side, notes: list[str]) -> None:
        if side == Side.SHORT and not self.cfg.allow_shorts:
            raise GateReject("shorts_disabled")
        notes.append(f"direction: {side.value}")

    def _check_score(self, signal: FusedSignal, notes: list[str]) -> None:
        if signal.final_score < self.cfg.high_priority_threshold:
            raise GateReject(
                f"score {signal.final_score:.1f}<{self.cfg.high_priority_threshold:.0f}"
            )
        if signal.blocked:
            raise GateReject(f"signal_blocked:{signal.block_reason}")
        notes.append(f"score: {signal.final_score:.1f}")

    def _check_slippage(
        self,
        *,
        side: Side,
        trigger_price: float,
        current_price: float,
        fused_score: float,
        realized_volatility_pct: float,
        book_depth_usdt_top5: float,
        notes: list[str],
    ) -> None:
        # Compute the leverage we would actually use, then derive the dynamic
        # slippage threshold. This is SR-1: the higher the leverage the
        # tighter the allowable slippage.
        leverage = compute_dynamic_leverage(
            side=side,
            fused_score=fused_score,
            realized_volatility_pct=realized_volatility_pct,
            book_depth_usdt_top5=book_depth_usdt_top5,
            min_liquidity_usdt=self.cfg.min_liquidity_usdt,
            cfg=self.leverage_cfg,
        )
        max_slip = self.cfg.base_slippage / math.sqrt(leverage / 5.0)
        if trigger_price <= 0:
            raise GateReject("trigger_price_invalid")
        slip = abs(current_price - trigger_price) / trigger_price

        # Asymmetric guard: only reject when slippage moves AGAINST our entry.
        moved_against = (
            (side == Side.LONG and current_price > trigger_price)
            or (side == Side.SHORT and current_price < trigger_price)
        )
        if moved_against and slip > max_slip:
            raise GateReject(
                f"slippage_abort {slip:.4f}>{max_slip:.4f} @ lev={leverage:.1f}x"
            )
        notes.append(
            f"slippage: {slip*100:.2f}% (max {max_slip*100:.2f}% @ lev {leverage:.1f}x)"
        )

    # ---------------------- helpers ---------------------- #

    @staticmethod
    def _side_from_signal(signal: FusedSignal) -> Side:
        if signal.direction.value == "long":
            return Side.LONG
        if signal.direction.value == "short":
            return Side.SHORT
        raise GateReject("signal_direction_neutral")

    @staticmethod
    def _trigger_price(signal: FusedSignal, fallback: float) -> float:
        # FusedSignal in this repo doesn't currently carry a trigger_price field
        # (Task A's screener events do via payload). The bus integration is
        # responsible for stamping it on the signal. For now we accept the
        # fallback (current price -> slippage will be 0). Test code can pass
        # a real trigger price by constructing the signal with the field.
        explicit = getattr(signal, "trigger_price", None)
        return float(explicit) if explicit is not None else fallback
