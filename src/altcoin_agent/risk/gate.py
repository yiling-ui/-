"""gate.py — RiskGate, the single hard wall before any order is placed.

Thirteen fail-closed checks, in order. Any single failure produces a
rejection with a reason. On unexpected exception the result is also a
rejection (fail-closed).

Every check below is independent: each returns a rejection on its own,
no boolean ``and``/``or`` short-circuiting fuses two checks into one.
This is the invariant the architecture depends on so that a single
defect in one gate cannot silently disable another.

The 13 independent checks (in evaluation order)
-----------------------------------------------
Signal-validity (pre-network):
    1. Signal is not blocked upstream (``signal.blocked``).
    2. Signal is high-priority (``signal.is_high_priority``).
    3. Signal direction is not NEUTRAL.

Local-memory micro-structure (no network round-trip):
    4. Anti-chase (price tape): recent move not already too far in our
       favour to add.
    5. Vol-kill (price tape): tape not in whipsaw range expansion.
    6. BTC regime filter (audit #10): block LONG when BTC is dropping
       fast / SHORT when BTC is ripping (cold tape -> fail-open).
    7. Symbol-cluster cap (audit #11): correlated symbol set
       (PEPE/WIF/FLOKI ...) effectively counts as one trade.

Account-level circuit breakers:
    8. Global trading halt (manual or kill-switch).
    9. Reconciliation must be complete (SR-2).
    10. Daily drawdown circuit breaker.
    11. Daily stop-loss hit count cap.
    12. Per-symbol cooldown active.
    13. Per-symbol consecutive-loss cooldown.

Capacity and execution feasibility:
    14. Concurrent-position cap.
    15. Top-5 orderbook depth (liquidity).
    16. Slippage cap (SR-1, dynamic by leverage).

(Fifteen ``return RiskDecision(False, ...)`` veto points plus a final
post-sizing ``size <= 0`` rejection. The "13" headline counts the user-
facing risk gates; the three signal-validity checks above are mandatory
preconditions enforced at the same fail-closed level.)

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
from dataclasses import dataclass

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.price_tape import PriceTape
from altcoin_agent.risk.cluster import (
    ClusterCapConfig,
    ClusterMap,
    cap_breached,
)
from altcoin_agent.risk.regime_filter import RegimeFilter
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
        price_tape: PriceTape | None = None,
        regime_filter: RegimeFilter | None = None,
        cluster_map: ClusterMap | None = None,
        cluster_cap_cfg: ClusterCapConfig | None = None,
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
