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

Audit batch 2 additions
-----------------------
Two optional gates run BEFORE any networked check (mirroring the
PriceTape anti-chase / vol-kill design):

  * BTC regime gate (audit #10): when the configured RegimeFilter
    reports BTC is dropping fast we block LONG entries; when BTC is
    ripping we block SHORT entries. Keeps the strategy from getting
    crushed by market beta in a fast tape.

  * Symbol cluster cap (audit #11): correlated symbols (PEPE / WIF /
    FLOKI ...) effectively count as the same trade. We enforce a
    per-cluster maximum on top of the global ``max_concurrent_positions``.

Both are no-ops when the relevant kwargs are omitted, so existing
callers and tests keep working unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.price_tape import PriceTape
from altcoin_agent.risk.cluster import (
    ClusterCapConfig,
    ClusterMap,
    cap_breached,
)
from altcoin_agent.risk.pump_phase import PumpPhase
from altcoin_agent.risk.regime_filter import RegimeFilter
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Side
from altcoin_agent.risk.symbol_profile import SymbolProfile

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
    # Operational patch (depth-aware sizing): cap a single leg's notional
    # at ``max_notional_vs_depth_pct`` of the live top-5 depth so the
    # daemon cannot eat through the book on a thin-liquidity altcoin
    # once the account scales (500 USDT @ 5x = 2500 USDT, fine; 50,000 @
    # 15x = 750,000 USDT and you become the entire top-5 -> 5-10%
    # immediate slippage, blowing through the 5% initial stop on entry).
    # 0.0 disables the check (legacy behaviour for backtests / tests).
    # Set to e.g. 0.10 in production: a single entry can take at most
    # 10% of the visible top-5 depth.
    max_notional_vs_depth_pct: float = 0.0
    # Slippage (SR-1)
    base_slippage: float = 0.03              # 3% at 5x leverage
    # Signal age
    max_signal_age_sec: int = 10
    # Per-symbol cooldown after high_priority emission
    symbol_cooldown_sec: int = 60

    # ------------------------------------------------------------------ #
    # R6 — phase + confidence gate (optional, default permissive).
    #
    # When the caller passes ``phase`` and ``confidence`` to ``evaluate``
    # (or a ``SymbolProfile`` whose quadrant carries a custom
    # ``confidence_threshold``), the gate enforces:
    #
    #   1. ``confidence >= phase_min_confidence[phase]``
    #      Default thresholds preserve v1.0 behaviour for callers that
    #      pass the legacy 0.0 confidence: 0.0 always >= 0.0.
    #
    #   2. ``phase`` is on the ``allowed_entry_phases`` allow-list.
    #      Default allows everything, so legacy callers see no change.
    #      QUADRANT_STRATEGY_PLAN section 四 says CRASH/BLEED/DEAD are
    #      no-trade phases for entries; operators that want plan-spec
    #      behaviour set ``allowed_entry_phases = {accumulation, ramp,
    #      parabolic}`` in app.yaml.
    #
    # Both checks fail-open when ``phase`` / ``confidence`` are None
    # so existing tests stay green.
    # ------------------------------------------------------------------ #
    phase_min_confidence: dict[str, float] = field(
        default_factory=lambda: {
            PumpPhase.ACCUMULATION.value: 0.0,
            PumpPhase.RAMP.value: 0.0,
            PumpPhase.PARABOLIC.value: 0.0,
            PumpPhase.BLOWOFF_TOP.value: 0.0,
            PumpPhase.CRASH.value: 0.0,
            PumpPhase.BLEED.value: 0.0,
            PumpPhase.DEAD.value: 0.0,
        }
    )
    allowed_entry_phases: frozenset[str] = frozenset({
        PumpPhase.ACCUMULATION.value,
        PumpPhase.RAMP.value,
        PumpPhase.PARABOLIC.value,
        PumpPhase.BLOWOFF_TOP.value,
        PumpPhase.CRASH.value,
        PumpPhase.BLEED.value,
        PumpPhase.DEAD.value,
    })


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
        price_tape: PriceTape | None = None,
        regime_filter: RegimeFilter | None = None,
        cluster_map: ClusterMap | None = None,
        cluster_cap_cfg: ClusterCapConfig | None = None,
        # R6 — phase + confidence + symbol profile (all optional).
        phase: PumpPhase | None = None,
        confidence: float | None = None,
        symbol_profile: SymbolProfile | None = None,
    ) -> RiskDecision:
        """Run all 9 checks. Returns an approved decision with sizing details
        on success, or a rejection with a reason on the first failure.

        ``price_tape`` is optional; when provided, two extra checks
        (anti-chase, vol-kill) run BEFORE the networked ones so a
        rejection saves a venue round-trip altogether. See
        :class:`altcoin_agent.price_tape.PriceTape` for the rationale —
        in altcoin pump-and-dump bursts, REST RTT is too slow to *catch
        the top*; the only winning move is to refuse to chase.

        ``regime_filter`` (audit #10): when present, blocks LONG when
        BTC is in fast drawdown and SHORT when BTC is ripping.

        ``cluster_map`` + ``cluster_cap_cfg`` (audit #11): when both
        present, blocks any signal that would push the symbol's cluster
        (e.g. ``meme``) past the configured cap.

        ``phase`` / ``confidence`` / ``symbol_profile`` (R6): when
        provided, enforce the QUADRANT_STRATEGY_PLAN per-phase
        confidence thresholds and the allowed-entry-phase allow-list.
        ``symbol_profile`` (when set) overrides
        ``phase_min_confidence`` with its quadrant's
        ``effective_confidence_threshold`` -- so an A-quadrant symbol
        with a stricter trainer-tuned threshold gets its tighter
        gate. All three default to None and are fail-open, preserving
        v1.0 behaviour for legacy callers.

        All three optional checks fail-OPEN when their inputs are
        missing or stale, so existing callers and dry-run tests keep
        working unchanged.
        """
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

            # 0a) anti-chase / vol-kill — local memory only, O(window).
            #
            # Run BEFORE the live-quote / order RTTs so a rejection here
            # costs nothing on the venue side. These two checks are the
            # only credible defence in a sub-500ms +5% pump scenario:
            # we cannot catch the top, but we can refuse to enter on it.
            if price_tape is not None:
                breached, move = price_tape.anti_chase_breach(
                    symbol=signal.symbol, side=side, now_ms=now_ms,
                )
                if breached:
                    return RiskDecision(
                        False,
                        f"chase_too_late:{move:+.4f}>"
                        f"{price_tape.cfg.anti_chase_max_move_pct:+.4f}",
                    )
                vol_breached, rng = price_tape.vol_kill_breach(
                    symbol=signal.symbol, now_ms=now_ms,
                )
                if vol_breached:
                    return RiskDecision(
                        False,
                        f"vol_kill_active:{rng:.4f}>"
                        f"{price_tape.cfg.vol_kill_range_pct:.4f}",
                    )

            # 0b) BTC market-regime gate (audit #10).
            #
            # Blocks LONG when BTC is dropping fast, SHORT when BTC is
            # ripping. Cold tape -> fail-open (won't reject every
            # signal during the first 10 BTC bars after boot).
            if regime_filter is not None:
                allowed, reason = regime_filter.allow_direction(
                    direction=signal.direction.value, now_ms=now_ms,
                )
                if not allowed:
                    return RiskDecision(False, reason)

            # 0c) Symbol-cluster cap (audit #11).
            #
            # Prevents the daemon from being SHORT all three of
            # PEPE/WIF/FLOKI simultaneously: those are one trade,
            # not three. Counting includes the proposed symbol.
            if cluster_map is not None and cluster_cap_cfg is not None:
                breached, reason = cap_breached(
                    proposed_symbol=signal.symbol,
                    open_symbols=list(account.open_positions.keys()),
                    cluster_map=cluster_map,
                    cap_cfg=cluster_cap_cfg,
                )
                if breached:
                    return RiskDecision(False, reason)

            # 0d) R6 — phase allow-list + per-phase confidence floor.
            #
            # Two cheap local checks before any networked work. Both
            # fail-OPEN on missing inputs so legacy callers see no
            # change. Symbol profile (when supplied) overrides the
            # gate-level confidence floor with the quadrant's
            # ``effective_confidence_threshold``.
            if phase is not None:
                if phase.value not in self.cfg.allowed_entry_phases:
                    return RiskDecision(
                        False, f"phase_not_allowed:{phase.value}",
                    )
                if confidence is not None:
                    floor = self.cfg.phase_min_confidence.get(
                        phase.value, 0.0,
                    )
                    if symbol_profile is not None:
                        # Profile threshold supersedes gate floor.
                        floor = max(
                            floor,
                            symbol_profile.effective_confidence_threshold(),
                        )
                    if confidence < floor:
                        return RiskDecision(
                            False,
                            f"confidence_below_floor:"
                            f"{confidence:.3f}<{floor:.3f}@phase={phase.value}",
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
            #
            # Bug #1 fix: pass the dynamic ``leverage`` we just computed so
            # ``compute_size`` can clamp ``notional <= equity * leverage``.
            # Without this, a tight sweep stop (e.g. 0.05%) would produce
            # 30-100x equity notional and either be rejected by the venue
            # (-> emergency close + 4h cooldown) or quietly oversize the
            # book on cross margin.
            size, notional, risk_amount = self.sizer.compute_size(
                equity_usdt=account.equity_usdt,
                entry_price=current_price,
                initial_stop=initial_stop,
                leverage=leverage,
            )
            if size <= 0 or notional <= 0:
                return RiskDecision(
                    False, "sizing_below_minimum", side=side, leverage=leverage,
                )

            # 10b) depth-aware notional cap (operational patch).
            #
            # Why this matters: ``min_liquidity_usdt`` (闸门 #8) only checks
            # that the book is ABOVE a floor. It does NOT prevent a large
            # account from sizing a SINGLE order that eats most of that
            # floor. With equity=50,000 USDT @ 15x leverage on an altcoin
            # whose top-5 depth is 500,000 USDT, the gate happily approves
            # a 750,000 USDT notional — meaning we ARE the book + 50%, and
            # the actual fill price will sit well outside our stop.
            #
            # We therefore compare the *just-sized* notional against the
            # live depth and reject if the order would consume more than
            # ``max_notional_vs_depth_pct`` of the visible top-5. The
            # check is deliberately OFF by default (== 0.0) so the legacy
            # test suite that uses synthetic depths stays green; production
            # flips it on via ``app.yaml`` (recommended: 0.10 = 10%).
            if self.cfg.max_notional_vs_depth_pct > 0:
                cap_notional = (
                    top5_depth_usdt * self.cfg.max_notional_vs_depth_pct
                )
                if notional > cap_notional:
                    return RiskDecision(
                        False,
                        f"notional_exceeds_depth_cap:{notional:.0f}>"
                        f"{cap_notional:.0f}@"
                        f"pct={self.cfg.max_notional_vs_depth_pct:.2f}",
                        side=side,
                        leverage=leverage,
                        size=size,
                        notional_usdt=notional,
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



    # ---------------- rolling-position gate ---------------- #
    #
    # Adding a leg to an EXISTING winning position is a different
    # animal from opening a fresh position. We deliberately skip:
    #
    #   * check #5 (per-symbol cooldown):  the cooldown is meant to
    #     prevent re-entering after a high_priority signal flipped
    #     direction. Adding to a winner is the opposite intent --
    #     we are *staying* with the prior direction.
    #
    #   * check #7 (max_concurrent_positions): adding a leg does not
    #     increase the number of distinct symbols in account.open_positions.
    #
    # Everything else (halt / reconciliation / daily DD / 3-strike /
    # consecutive losses / liquidity / SR-1 slippage / leverage cap)
    # remains in force. The leverage cap is honoured by the caller
    # (RollingController computes new_leg_notional with a clamp), but
    # we re-check here as defence-in-depth.

    def evaluate_rolling(
        self,
        *,
        parent,                          # type: Position (avoid circular import)
        proposed_size: float,
        proposed_stop: float,
        account: AccountState,
        current_price: float,
        top5_depth_usdt: float,
        realized_vol_pct: float,
        trigger_price: float,
        now_ms: int | None = None,
    ) -> RiskDecision:
        """Like ``evaluate``, but for an additional leg on an open position.

        Returns a positive RiskDecision when the additional leg can be
        placed; the caller is expected to use ``proposed_size`` /
        ``proposed_stop`` directly (no re-sizing here).
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        try:
            side = parent.side

            # 1) global halt
            if account.global_trading_halted:
                return RiskDecision(False, f"global_halt:{account.halt_reason}")

            # 2) reconciliation must be complete
            if not account.reconciliation_complete:
                return RiskDecision(False, "reconciliation_pending")

            # 3) daily drawdown circuit breaker (Bug #3 daily rollover
            # already keeps this honest across UTC days)
            if account.daily_drawdown_pct >= self.cfg.daily_drawdown_limit:
                return RiskDecision(
                    False,
                    f"daily_drawdown_limit:{account.daily_drawdown_pct:.4f}",
                )

            # 4) daily stop-loss hits
            if account.daily_stoploss_hits >= self.cfg.daily_stoploss_hits_max:
                return RiskDecision(False, "daily_stoploss_hits_exceeded")

            # 5)  per-symbol cooldown   -- DELIBERATELY SKIPPED for rolls.
            # 6) per-symbol consecutive losses (still relevant: don't add
            # to a position whose recent history is bad even if currently
            # in profit)
            losses = account.consecutive_losses.get(parent.symbol, 0)
            if losses >= self.cfg.max_consecutive_losses:
                return RiskDecision(False, "consecutive_loss_cooldown")

            # 7) concurrency  -- DELIBERATELY SKIPPED for rolls.

            # 8) liquidity (still required: never add into a thin book)
            if top5_depth_usdt < self.cfg.min_liquidity_usdt:
                return RiskDecision(
                    False,
                    f"insufficient_liquidity:{top5_depth_usdt:.0f}",
                )

            # 9) SR-1 slippage -- use the parent's leverage so the cap
            # is identical to what guarded the original entry. Adverse
            # drift here is measured between the leg's intended trigger
            # and the live mark.
            max_slippage = self._dynamic_slippage_cap(parent.leverage)
            slip = self._adverse_slip(side, trigger_price, current_price)
            if slip > max_slippage:
                return RiskDecision(
                    False,
                    f"slippage_too_high:{slip:.4f}>{max_slippage:.4f}@lev"
                    f"={parent.leverage:.2f}",
                    max_slippage_used=max_slippage,
                )

            # 10) leverage cap -- defence-in-depth.
            existing_notional = parent.total_size * current_price
            new_leg_notional = proposed_size * current_price
            total_notional = existing_notional + new_leg_notional
            max_total = account.equity_usdt * parent.leverage
            if total_notional > max_total + 1e-6:
                return RiskDecision(
                    False,
                    f"leverage_cap_exceeded:{total_notional:.2f}>"
                    f"{max_total:.2f}",
                    side=side, leverage=parent.leverage,
                )

            if proposed_size <= 0 or new_leg_notional < self.sizer.min_notional_usdt:
                return RiskDecision(
                    False,
                    f"sizing_below_minimum:{new_leg_notional:.2f}",
                    side=side, leverage=parent.leverage,
                )

            # 10b) depth-aware notional cap (operational patch, rolling path).
            #
            # Same rationale as the entry path: a single market order that
            # consumes more than ``max_notional_vs_depth_pct`` of the
            # visible top-5 depth is guaranteed to slip badly. We check
            # the *new leg's* notional rather than the cumulative position
            # because (a) the existing exposure already trades, and (b)
            # the slippage that hurts us is the impact of THIS order.
            if self.cfg.max_notional_vs_depth_pct > 0:
                cap_notional = (
                    top5_depth_usdt * self.cfg.max_notional_vs_depth_pct
                )
                if new_leg_notional > cap_notional:
                    return RiskDecision(
                        False,
                        f"notional_exceeds_depth_cap:{new_leg_notional:.0f}>"
                        f"{cap_notional:.0f}@"
                        f"pct={self.cfg.max_notional_vs_depth_pct:.2f}",
                        side=side,
                        leverage=parent.leverage,
                        size=proposed_size,
                        notional_usdt=new_leg_notional,
                    )

            return RiskDecision(
                approved=True,
                reason="ok",
                side=side,
                leverage=parent.leverage,
                size=proposed_size,
                notional_usdt=new_leg_notional,
                risk_amount_usdt=proposed_size * abs(current_price - proposed_stop),
                initial_stop=proposed_stop,
                max_slippage_used=max_slippage,
            )

        except Exception as e:
            logger.exception("RiskGate.evaluate_rolling failure: %s", e)
            return RiskDecision(False, f"unexpected_error:{type(e).__name__}")
