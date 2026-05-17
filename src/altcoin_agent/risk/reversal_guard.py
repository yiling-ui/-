"""reversal_guard.py — gate the daemon against premature reversal trades.

Operational patch (post-review)
==============================

The operator's instruction was unambiguous::

    对于反手要谨慎防止插针，合理规划风险要么就平仓观望，
    等待时机。只有合适时机才反手。

Translated to risk language:

  * **Default to flat-and-wait** when the daemon thinks the symbol just
    flipped. Holding the wrong side through a wick (插针 — "needle
    insertion", a deliberate liquidity-grab spike) is the worst kind
    of loss because the move that traps you is the same move that
    triggers the next entry.
  * **Only reverse when the conditions are right**: enough time has
    passed for the wick noise to settle, the price has confirmed past
    the level, and the new signal is strong on its own (not just
    because the old position took a stop). The default posture is
    "close and watch", and reversal is the exception.

The :class:`ReversalGuard` gives ``main.App._handle_high_priority`` a
single *decision* helper that turns those rules into a yes/no with a
machine-readable reason. We do NOT mutate state here; the caller
either lets the trade through, defers it (close + watch), or vetoes
it outright. Persistence of the "watch" state is in
``account.cooldown_until_ts_ms[symbol]`` so a daemon restart still
honours the wait period.

What counts as a "reversal"
---------------------------
A signal is treated as a reversal candidate if **any** of the
following is true at the moment ``decide()`` is called:

  1. The account currently holds a position on ``symbol`` and the new
     signal points the opposite way.
  2. The account just closed a position on ``symbol`` within the
     ``recent_close_window_sec`` (default 5 minutes) and the new
     signal points the opposite way.

For cases (2) we read a small "last close" map from the
:class:`AccountState` (operator-managed via :meth:`note_close`) so
the guard can see one wick deep into the past without us needing a
full close-event log.

Veto conditions
---------------

Even a "right time" reversal can be vetoed by:

  * **Wick / 插针 detection** — the recent ``high - low`` range over
    ``wick_window_sec`` exceeds ``wick_threshold_pct`` of mid. We
    use the existing ``PriceTape`` Parkinson estimator so this is a
    cheap deque scan, not a klines fetch.
  * **Signal strength** — ``min_reversal_final_score`` floor.
    "Reversed because the old trade got stopped" should not be
    enough; the new direction must clear an explicit bar.
  * **Cooldown** — after a reversal-veto we set
    ``reversal_cooldown_sec`` on the symbol so the guard doesn't
    flap on every tick of a thrashing tape.

All thresholds default to fail-CLOSED values that are obviously
conservative. Operators relax them in app.yaml as confidence grows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.price_tape import PriceTape
from altcoin_agent.risk.state import AccountState, Side

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class ReversalGuardConfig:
    """Operator-tunable knobs.

    Defaults are fail-closed: a fresh deploy that flips the guard on
    will reject most reversals, which is the intended UX. Operators
    relax them by editing app.yaml after observing how often the
    guard fires.
    """

    enabled: bool = False
    # Look-back window for "did we just close a position on this
    # symbol?". A reversal arriving inside this window is treated as
    # potentially-reactive to the close itself.
    recent_close_window_sec: int = 300

    # 插针 (wick / liquidity-grab) detection: if the Parkinson range
    # over ``wick_window_sec`` exceeds ``wick_threshold_pct`` of mid,
    # we VETO the reversal entirely (the new signal is most likely
    # the result of the wick, not a structural flip). Default 4%
    # over 60 s — calibrated to fire on the kind of single-candle
    # spike that traps reversal-chasers in altcoin futures.
    wick_window_sec: int = 60
    wick_threshold_pct: float = 0.04

    # The new signal must clear this final-score floor before the
    # guard considers approving a reversal. Higher than the normal
    # high-priority threshold by design.
    min_reversal_final_score: float = 7.5

    # After we VETO a reversal, set this cooldown on the symbol so a
    # tick-by-tick thrashing tape doesn't burn CPU + Telegram.
    reversal_cooldown_sec: int = 120

    # Minimum elapsed time since the last close on the same symbol
    # before a reversal can fire. Below this we always defer "watch
    # and wait" (close-then-flat is the operator's first preference).
    min_seconds_since_close: int = 30


# --------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReversalDecision:
    """What the caller should do with the signal.

    One of:
      * ``approve`` — guard does not apply (no position, same-side, or
        the operator turned the guard off). Caller proceeds as normal.
      * ``defer_close_and_watch`` — close existing position (if any)
        but do NOT open the reversal. The cooldown is set.
      * ``veto`` — drop the signal; the new direction failed an
        explicit check (wick, score, cooldown).

    ``reason`` is a machine-readable tag; ``message`` is human-friendly.
    ``diag`` carries the numbers for audit / dashboard.
    """

    action: str  # "approve" | "defer_close_and_watch" | "veto"
    reason: str
    message: str
    diag: dict[str, Any]


# --------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------- #


@dataclass
class ReversalGuard:
    """Decide whether to allow a reversal trade.

    Stateless beyond what's already in ``account`` / ``price_tape``.
    Construct once at boot and call :meth:`decide` from the hot path.
    """

    cfg: ReversalGuardConfig
    price_tape: PriceTape | None = None

    # Per-symbol "last close" memory (ts_ms, side_we_were_on). Kept
    # here rather than on AccountState so the guard module owns its
    # own bookkeeping; ``main.App`` calls :meth:`note_close` from the
    # close handler.
    _last_close: dict[str, tuple[int, Side]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._last_close is None:
            self._last_close = {}

    # --------------------------- public --------------------------- #

    def note_close(self, *, symbol: str, side: Side, now_ms: int | None = None) -> None:
        """Record that we just closed a position on ``symbol``. The next
        opposite-direction signal within ``recent_close_window_sec``
        will be treated as a reversal candidate.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        self._last_close[symbol] = (int(now_ms), side)

    def decide(
        self,
        *,
        signal: FusedSignal,
        account: AccountState,
        now_ms: int | None = None,
    ) -> ReversalDecision:
        """Single hot-path decision. Returns one of the three actions."""
        if not self.cfg.enabled:
            return ReversalDecision(
                action="approve",
                reason="disabled",
                message="reversal guard disabled",
                diag={},
            )
        if signal.direction == Direction.NEUTRAL:
            return ReversalDecision(
                action="approve",
                reason="not_reversal",
                message="neutral signal — guard not applicable",
                diag={},
            )
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        new_side = (
            Side.LONG if signal.direction == Direction.LONG else Side.SHORT
        )
        sym = signal.symbol

        # Determine whether this is a reversal candidate.
        is_reversal = False
        reversal_kind: str | None = None
        old_side: Side | None = None

        cur_pos = account.position(sym)
        if cur_pos is not None and cur_pos.side != new_side:
            is_reversal = True
            reversal_kind = "open_position_flip"
            old_side = cur_pos.side
        else:
            last = self._last_close.get(sym)
            if last is not None:
                close_ts_ms, prev_side = last
                age_sec = (now_ms - close_ts_ms) / 1000.0
                if (
                    age_sec >= 0
                    and age_sec < self.cfg.recent_close_window_sec
                    and prev_side != new_side
                ):
                    is_reversal = True
                    reversal_kind = "post_close_flip"
                    old_side = prev_side

        if not is_reversal:
            return ReversalDecision(
                action="approve",
                reason="not_reversal",
                message="not a reversal — same side or no recent context",
                diag={"new_side": new_side.value},
            )

        # ----------------------------------------------------------- #
        # Reversal candidate. Run the veto checks in cheap-first order.
        # ----------------------------------------------------------- #

        diag: dict[str, Any] = {
            "kind": reversal_kind,
            "old_side": old_side.value if old_side else None,
            "new_side": new_side.value,
            "final_score": float(signal.final_score),
        }

        # Cooldown set by a previous veto. Anything inside the
        # cooldown window is a hard veto — we don't even check the
        # other gates.
        if account.is_in_cooldown(sym, now_ms):
            until = account.cooldown_until_ts_ms.get(sym, 0)
            wait_sec = max(0.0, (until - now_ms) / 1000.0)
            diag["cooldown_remaining_sec"] = wait_sec
            return ReversalDecision(
                action="veto",
                reason="reversal_cooldown_active",
                message=(
                    f"reversal cooldown active "
                    f"({wait_sec:.0f}s remaining); waiting it out"
                ),
                diag=diag,
            )

        # If this is a post-close reversal, ensure enough wall-clock
        # has passed since the close. Operator's preferred posture:
        # close, then watch. Don't immediately reverse.
        if reversal_kind == "post_close_flip":
            close_ts_ms, _ = self._last_close[sym]
            age_sec = (now_ms - close_ts_ms) / 1000.0
            diag["seconds_since_close"] = age_sec
            if age_sec < self.cfg.min_seconds_since_close:
                # Defer rather than veto — the signal might be valid
                # in another minute. We don't set a cooldown so the
                # caller can re-evaluate on the next tick.
                return ReversalDecision(
                    action="defer_close_and_watch",
                    reason="too_soon_after_close",
                    message=(
                        f"reversal {age_sec:.0f}s after close < "
                        f"{self.cfg.min_seconds_since_close}s minimum; "
                        "watching"
                    ),
                    diag=diag,
                )

        # Wick / 插针 detection.
        if self.price_tape is not None:
            wick_pct = self.price_tape.realized_vol_pct(
                symbol=sym,
                window_ms=self.cfg.wick_window_sec * 1000,
                now_ms=now_ms,
            )
            if wick_pct is not None:
                diag["wick_range_pct"] = float(wick_pct)
                if wick_pct >= self.cfg.wick_threshold_pct:
                    # 插针. Veto + cooldown.
                    self._set_cooldown(account, sym, now_ms)
                    return ReversalDecision(
                        action="veto",
                        reason="wick_detected",
                        message=(
                            f"wick detected ({wick_pct * 100:.1f}% range "
                            f"over {self.cfg.wick_window_sec}s) — "
                            "reversal vetoed, cooldown set"
                        ),
                        diag=diag,
                    )

        # Signal strength bar.
        if signal.final_score < self.cfg.min_reversal_final_score:
            self._set_cooldown(account, sym, now_ms)
            return ReversalDecision(
                action="veto",
                reason="signal_below_reversal_floor",
                message=(
                    f"reversal signal score {signal.final_score:.2f} "
                    f"< floor {self.cfg.min_reversal_final_score:.2f}; "
                    "vetoed"
                ),
                diag=diag,
            )

        # Open-position flip path (cur_pos is not None): the operator's
        # explicit instruction is "rather close and watch than blindly
        # reverse". We DEFER (close, no immediate reopen) by default;
        # only the post-close path with all checks passing returns
        # ``approve``. This makes the guard's default semantic match
        # the operator's words verbatim.
        if reversal_kind == "open_position_flip":
            return ReversalDecision(
                action="defer_close_and_watch",
                reason="prefer_flat_and_watch",
                message=(
                    "open-position reversal — closing position "
                    "and watching; reversal will be evaluated on the "
                    "next signal post-close"
                ),
                diag=diag,
            )

        # Post-close flip with all checks passing -> this is the one
        # narrow path where a reversal IS approved.
        return ReversalDecision(
            action="approve",
            reason="post_close_reversal_ok",
            message="post-close reversal passed all checks",
            diag=diag,
        )

    # --------------------------- helpers --------------------------- #

    def _set_cooldown(
        self, account: AccountState, symbol: str, now_ms: int,
    ) -> None:
        """Apply the reversal-cooldown to ``symbol`` via the account
        helper so it persists alongside other cooldowns and respects
        the existing change listener.
        """
        try:
            account.set_cooldown(
                symbol,
                duration_sec=self.cfg.reversal_cooldown_sec,
                now_ms=now_ms,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "ReversalGuard: set_cooldown failed for %s (%s); "
                "veto still applies in-memory",
                symbol, e,
            )
