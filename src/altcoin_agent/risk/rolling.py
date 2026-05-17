"""rolling.py — RollingController (滚仓 / pyramid-add).

Adds same-side legs to a winning position when:
  (1) the live strategy still rates the symbol at high_priority on the
      same side (re-runs ScoreFuser.evaluate against the live recent_cache);
  (2) unrealised PnL has crossed the next un-fired R-trigger;
  (3) Risk Gate's daily / 3-strike / halt / liquidity / slippage /
      leverage caps are all still honoured (via RiskGate.evaluate_rolling);
  (4) min_interval_sec since the last roll has elapsed;
  (5) max_legs_per_symbol has not been reached.

All five gates must pass; the order is short-circuit so cheap checks
fail fast.

Sizing (see also rolling-positions.md §3.6):
    risk_budget   = unrealised_pnl_usdt * unrealized_pnl_ratio
    stop_distance = mark * leg_stop_pct
    new_size      = risk_budget / stop_distance
    new_notional  = new_size * mark
    new_notional  = min(new_notional, equity*leverage - existing_notional)
    -> at least min_notional_usdt or skip

Failure semantics:
    Any exception in the gate / sizing / executor chain is logged at
    warning, surfaces a notifier alert, and (if auto_disable_on_failure)
    flips ``cfg.enabled = False`` so the next tick short-circuits.
    The position itself is left untouched if the failure happened
    BEFORE the new market order; if the failure was the stop-resize
    after a successful add_leg, the executor has already
    emergency-closed the whole position.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from altcoin_agent.fuser import Direction, ScoreFuser
from altcoin_agent.risk.executor import CCXTExecutor, ExecutionError
from altcoin_agent.risk.gate import RiskGate
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Position, Side

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class RollingConfig:
    """Operator-tunable knobs for rolling positions.

    Every field can be live-toggled from the dashboard or Telegram
    without restarting the daemon (see PR-B and PR-C).
    """

    enabled: bool = False
    """Master switch. Default OFF; operator opts in."""

    trigger_r_levels: tuple[float, ...] = (1.5, 3.0, 5.0)
    """Unrealised-R thresholds that fire a roll. Each level fires at
    most once per position."""

    unrealized_pnl_ratio: float = 0.5
    """Fraction of unrealised PnL to commit as risk budget for the new
    leg. 0.5 = use half of paper profit."""

    leg_stop_pct: float = 0.025
    """Initial stop distance for the new leg, as a fraction of mark
    price. Tighter than the original 5% because the position is already
    in profit and protected by the trailing stop."""

    max_legs_per_symbol: int = 3
    """How many ROLLED legs (excluding leg 0). 3 = up to 4 entries total."""

    min_interval_sec: int = 60
    """Minimum gap between two consecutive rolls on the same symbol."""

    auto_disable_on_failure: bool = True
    """When True, any exception in the rolling pipeline flips
    ``enabled = False`` so subsequent ticks short-circuit until the
    operator re-enables."""

    require_strategy_min_score: float = 85.0
    """The live FusedSignal.final_score must be >= this when re-evaluated.
    Defaults to FuserConfig.high_priority_threshold so a roll requires
    the same conviction as a fresh entry would."""

    require_min_rule_score: float = 35.0
    """The live FusedSignal.rule_score floor (mirrors
    FuserConfig.require_min_rule_score). Prevents a roll based purely
    on stale LLM boost."""

    def __post_init__(self) -> None:
        """Defensive bounds checking on operator-tunable knobs.

        These limits are intentionally generous — they exist to catch
        finger-trouble (someone typing 95.0 where they meant 0.95, or
        flipping the trigger ladder), not to police strategy choices.
        Anything materially out of band gets clamped or rejected with
        a clear ValueError so the daemon refuses to start with broken
        config rather than silently sizing 30x notionals.
        """
        if self.unrealized_pnl_ratio < 0.0 or self.unrealized_pnl_ratio > 1.0:
            raise ValueError(
                f"RollingConfig.unrealized_pnl_ratio must be in [0, 1], "
                f"got {self.unrealized_pnl_ratio!r}"
            )
        if self.leg_stop_pct <= 0.0 or self.leg_stop_pct > 0.10:
            raise ValueError(
                f"RollingConfig.leg_stop_pct must be in (0, 0.10], "
                f"got {self.leg_stop_pct!r}"
            )
        if self.max_legs_per_symbol < 0:
            raise ValueError(
                f"RollingConfig.max_legs_per_symbol must be >= 0, "
                f"got {self.max_legs_per_symbol!r}"
            )
        if self.min_interval_sec < 0:
            raise ValueError(
                f"RollingConfig.min_interval_sec must be >= 0, "
                f"got {self.min_interval_sec!r}"
            )
        if not self.trigger_r_levels:
            raise ValueError(
                "RollingConfig.trigger_r_levels must be non-empty"
            )
        prev = -float("inf")
        for lvl in self.trigger_r_levels:
            if lvl <= 0.0:
                raise ValueError(
                    f"RollingConfig.trigger_r_levels must all be > 0, "
                    f"got {self.trigger_r_levels!r}"
                )
            if lvl <= prev:
                raise ValueError(
                    f"RollingConfig.trigger_r_levels must be strictly "
                    f"increasing, got {self.trigger_r_levels!r}"
                )
            prev = lvl
        # Trigger ladder must clear breakeven (i.e., trailing has already
        # moved the stop to entry) — otherwise a roll could fire while
        # the FSM is still in INIT/ARMED and an adverse tick takes the
        # stop out before the new leg is protected.
        if self.trigger_r_levels[0] < 1.0:
            raise ValueError(
                f"RollingConfig.trigger_r_levels[0] must be >= 1.0 "
                f"(post-breakeven); got {self.trigger_r_levels[0]!r}"
            )
        if self.require_strategy_min_score < 0 or self.require_strategy_min_score > 100:
            raise ValueError(
                f"RollingConfig.require_strategy_min_score must be in [0, 100], "
                f"got {self.require_strategy_min_score!r}"
            )
        if self.require_min_rule_score < 0 or self.require_min_rule_score > 100:
            raise ValueError(
                f"RollingConfig.require_min_rule_score must be in [0, 100], "
                f"got {self.require_min_rule_score!r}"
            )


# --------------------------------------------------------------------- #
# Decision objects
# --------------------------------------------------------------------- #


@dataclass
class RollDecision:
    """Why a roll did or did not happen on a given tick."""

    fired: bool
    reason: str
    next_threshold_r: float | None = None
    new_leg_size: float | None = None
    new_leg_notional: float | None = None
    strategy_score: float | None = None


# --------------------------------------------------------------------- #
# Live-quote / strategy re-evaluation hooks
# --------------------------------------------------------------------- #


QuoteFn = Callable[[str], Awaitable[float]]
"""Async callable returning a fresh mark price for a symbol. Same
contract as ``App._get_live_quote`` (Bug #2 fix)."""


# --------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------- #


@dataclass
class RollingController:
    """Stateful per-account: tracks which R-thresholds have already
    fired for each open position, and the timestamp of the last roll
    per symbol so ``min_interval_sec`` can throttle bursts."""

    cfg: RollingConfig
    sizer: PositionSizer
    gate: RiskGate
    executor: CCXTExecutor
    fuser: ScoreFuser
    quote_provider: QuoteFn

    # Optional notifier + audit hook. Both must be no-throw on the hot
    # path; a failure here must never escape the rolling pipeline.
    notify_roll: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    notify_error: Callable[[str, dict[str, Any] | None], Awaitable[None]] | None = None

    _fired_levels: dict[str, set[float]] = field(default_factory=dict)
    """symbol -> R levels that already triggered for the CURRENT position
    on that symbol. Cleared by ``reset_for_symbol`` on close."""

    _last_roll_ts: dict[str, int] = field(default_factory=dict)
    """symbol -> ms timestamp of the last successful roll. Drives
    ``min_interval_sec``."""

    # ---------------- public API ---------------- #

    async def maybe_roll(
        self,
        *,
        position: Position,
        account: AccountState,
        top5_depth_usdt: float,
        realized_vol_pct: float,
        now_ms: int | None = None,
    ) -> RollDecision:
        """Single decision point. Called by the trailing worker on every
        kline tick.

        Order of cheap-to-expensive checks (each adds a fail-fast exit):

          1. enabled / not closed / has legs                -- O(1)
          2. max_legs_per_symbol                            -- O(1)
          3. min_interval_sec since last roll                -- O(1)
          4. fetch live mark price                           -- 1 RTT
          5. unrealised R >= next un-fired threshold         -- O(1)
          6. strategy still says BUY/SELL same direction     -- O(events)
          7. RiskGate.evaluate_rolling (DD/3-strike/halt/...) -- O(1)
          8. sizing produces >= min_notional                 -- O(1)
          9. executor.add_leg + stop-resize                  -- 2 RTT
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # 1) precondition
        if not self.cfg.enabled:
            return RollDecision(False, "disabled")
        if position.closed:
            return RollDecision(False, "position_closed")
        if not position.legs:
            # Defensive: every Position created via executor.open() has a
            # leg 0. If somehow legs is empty (e.g. reconciler-attached
            # orphan) we cannot reason about avg entry, so refuse.
            return RollDecision(False, "no_legs_recorded")

        # 2) leg cap
        if position.num_rolled_legs >= self.cfg.max_legs_per_symbol:
            return RollDecision(False, "max_legs_reached")

        # 3) min-interval throttle
        last = self._last_roll_ts.get(position.symbol, 0)
        if now_ms - last < self.cfg.min_interval_sec * 1000:
            return RollDecision(False, "min_interval_active")

        # 4) live mark price (Bug #2 fix path)
        try:
            mark = await self.quote_provider(position.symbol)
        except Exception as e:
            return RollDecision(
                False, f"quote_unavailable:{type(e).__name__}",
            )
        if mark <= 0:
            return RollDecision(False, "quote_non_positive")

        # 5) unrealised R reached the next un-fired threshold?
        r_unit = position.r_unit
        if r_unit <= 0:
            return RollDecision(False, "no_r_unit")
        unrealised_r = self._unrealised_r(position, mark)
        next_threshold = self._next_unfired_threshold(position.symbol, unrealised_r)
        if next_threshold is None:
            return RollDecision(False, "no_next_threshold")

        # 6) strategy re-confirmation -- THE KEY GATE.
        # Re-run the fuser with the LIVE recent_cache (which the screener
        # is continuously feeding). This gives us the same FusedSignal we
        # would have produced for a *fresh* entry on this symbol right now.
        sig = self.fuser.evaluate(
            position.symbol, position.exchange, now_ms,
        )
        sig_dir = (
            Direction.LONG if position.side == Side.LONG else Direction.SHORT
        )
        if sig.blocked:
            return RollDecision(
                False,
                f"strategy_blocked:{sig.block_reason}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        if sig.direction != sig_dir:
            # Strategy now favours the OPPOSITE side (or neutral). A
            # winning long that the system now thinks should be a short
            # is a textbook "let trailing close it" moment.
            return RollDecision(
                False,
                f"strategy_direction_changed:{sig.direction.value}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        if sig.final_score < self.cfg.require_strategy_min_score:
            return RollDecision(
                False,
                f"strategy_score_low:{sig.final_score:.1f}<"
                f"{self.cfg.require_strategy_min_score:.1f}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        if sig.rule_score < self.cfg.require_min_rule_score:
            # The fuser's rule floor: don't roll on stale LLM-only score.
            return RollDecision(
                False,
                f"rule_score_low:{sig.rule_score:.1f}<"
                f"{self.cfg.require_min_rule_score:.1f}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )

        # 7) RiskGate.evaluate_rolling -- daily DD / 3-strike / halt /
        # liquidity / SR-1 slippage / leverage cap.
        unrealised_pnl = position.unrealised_pnl_usdt(mark)
        # Sizing: risk-budget driven by unrealised PnL.
        risk_budget = unrealised_pnl * self.cfg.unrealized_pnl_ratio
        if risk_budget <= 0:
            return RollDecision(
                False, "no_unrealised_pnl",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        stop_distance = mark * self.cfg.leg_stop_pct
        if stop_distance <= 0:
            return RollDecision(
                False, "stop_distance_zero",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        new_leg_size = risk_budget / stop_distance
        new_leg_notional = new_leg_size * mark

        # Total-exposure clamp (parallel to Bug #1 fix in PositionSizer).
        existing_notional = position.total_size * mark
        max_total_notional = account.equity_usdt * position.leverage
        room = max(0.0, max_total_notional - existing_notional)
        if new_leg_notional > room:
            new_leg_notional = room
            new_leg_size = (
                new_leg_notional / mark if mark > 0 else 0.0
            )
        if new_leg_notional < self.sizer.min_notional_usdt:
            return RollDecision(
                False,
                f"below_min_notional:{new_leg_notional:.2f}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )

        gate_decision = self.gate.evaluate_rolling(
            parent=position,
            proposed_size=new_leg_size,
            proposed_stop=(
                mark - stop_distance
                if position.side == Side.LONG
                else mark + stop_distance
            ),
            account=account,
            current_price=mark,
            top5_depth_usdt=top5_depth_usdt,
            realized_vol_pct=realized_vol_pct,
            trigger_price=mark,
            now_ms=now_ms,
        )
        if not gate_decision.approved:
            return RollDecision(
                False,
                f"gate_rejected:{gate_decision.reason}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )

        # 8) execute the leg.
        try:
            leg = await self.executor.add_leg(
                position=position,
                size=new_leg_size,
                current_price=mark,
                # We do NOT tighten the trailing stop here. The trailing
                # FSM owns that on the next kline tick. We only need
                # add_leg to RESIZE the existing stop to cover the new
                # ``total_size``.
                new_stop_for_full=None,
                account=account,
                trigger_score=sig.final_score,
            )
        except ExecutionError as e:
            await self._handle_failure(
                f"executor_failed:{e}",
                position=position,
                next_threshold=next_threshold,
            )
            return RollDecision(
                False,
                f"executor_failed:{e}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )
        except Exception as e:
            await self._handle_failure(
                f"unexpected:{type(e).__name__}",
                position=position,
                next_threshold=next_threshold,
            )
            return RollDecision(
                False,
                f"unexpected:{type(e).__name__}",
                next_threshold_r=next_threshold,
                strategy_score=sig.final_score,
            )

        # Success: record the threshold so it cannot fire again.
        self._fired_levels.setdefault(position.symbol, set()).add(next_threshold)
        self._last_roll_ts[position.symbol] = now_ms

        if self.notify_roll is not None:
            try:
                await self.notify_roll({
                    "ts": now_ms,
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "leg_id": leg.leg_id,
                    "leg_size": new_leg_size,
                    "leg_notional_usdt": new_leg_notional,
                    "leg_entry_price": leg.entry_price,
                    "trigger_r": next_threshold,
                    "strategy_score": sig.final_score,
                    "total_size_after": position.total_size,
                    "avg_entry_after": position.avg_entry_price,
                })
            except Exception:
                logger.exception("notify_roll callback failed")

        return RollDecision(
            fired=True,
            reason="ok",
            next_threshold_r=next_threshold,
            new_leg_size=new_leg_size,
            new_leg_notional=new_leg_notional,
            strategy_score=sig.final_score,
        )

    def reset_for_symbol(self, symbol: str) -> None:
        """Called by App._on_position_close. Any future position on the
        same symbol starts with a clean fired-thresholds set."""
        self._fired_levels.pop(symbol, None)
        self._last_roll_ts.pop(symbol, None)

    # ---------------- internals ---------------- #

    @staticmethod
    def _unrealised_r(position: Position, mark: float) -> float:
        r_unit = position.r_unit
        if r_unit <= 0:
            return 0.0
        if position.side == Side.LONG:
            return (mark - position.avg_entry_price) / r_unit
        return (position.avg_entry_price - mark) / r_unit

    def _next_unfired_threshold(
        self, symbol: str, unrealised_r: float,
    ) -> float | None:
        """Smallest R level that (a) we have crossed and (b) has not
        already fired. Returns None when the position is below the
        first level OR every level has fired."""
        fired = self._fired_levels.get(symbol, set())
        for level in self.cfg.trigger_r_levels:
            if level in fired:
                continue
            if unrealised_r >= level:
                return level
            # Levels are evaluated low-to-high. The first level we have
            # NOT crossed is also the first one we still might.
            return None
        return None

    async def _handle_failure(
        self,
        reason: str,
        *,
        position: Position,
        next_threshold: float,
    ) -> None:
        logger.warning(
            "rolling: failure on %s at R=%.2f: %s",
            position.symbol, next_threshold, reason,
        )
        if self.cfg.auto_disable_on_failure:
            self.cfg.enabled = False
            logger.warning(
                "rolling: auto-disabled after failure on %s "
                "(set cfg.enabled = True to re-arm)",
                position.symbol,
            )
        if self.notify_error is not None:
            try:
                await self.notify_error(
                    f"rolling failure on {position.symbol}: {reason}",
                    {
                        "symbol": position.symbol,
                        "reason": reason,
                        "next_threshold": next_threshold,
                        "auto_disabled": (
                            self.cfg.auto_disable_on_failure
                        ),
                    },
                )
            except Exception:
                logger.exception("notify_error callback failed")
