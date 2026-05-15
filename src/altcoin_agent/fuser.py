"""
fuser.py — Task C: Score Fuser.

Fuses Task A's market/SMC rule signals with Task B's DeepSeek verdict into a
single ``FusedSignal`` and decides whether it qualifies as ``high_priority``
(crossing the threshold for the Risk Gate to consider).

Hard rules (per architect call):

    1.  Rule score is the BASE.
        Each detector contributes a non-negative weight; multiple detectors
        firing within the window stack additively (capped at ``rule_score_cap``).
        This rewards multi-modal confluence (volume spike + OI silent build +
        liquidity sweep) which is the textbook altcoin pump pattern.

    2.  LLM acts as a MULTIPLIER + VETO, never as a passive averager.
        - ``intent`` agrees with rule direction & confidence high  -> boost
        - ``intent == "neutral"``                                   -> no change
        - ``intent`` disagrees with rule direction                  -> VETO
          (drops final score to <=40 and sets blocked=True)

    3.  KOL "exit liquidity" handling (per user requirement, strict):
        - kol_intent == "exit_liquidity" AND llm.confidence >= 0.70
          -> HARD VETO. final_score forced to a punitive value, blocked=True.
        - kol_intent == "exit_liquidity" AND llm.confidence <  0.70
          -> SOFT CAP at ``kol_exit_soft_cap_score`` (default 70).
            Rule confluence can still drive the trade, but never with a
            "high priority" classification.

    4.  Direction conflict between rule signals (e.g. mixed buy/sell volume
        spikes within the window) -> downgrade by ``conflict_penalty``.

    5.  Stale data: any rule signal older than ``window_sec`` (default 90s)
        is dropped before scoring. LLM verdict older than 2x window is
        ignored entirely (treated as missing).

    6.  Cooldown: same symbol cannot emit high_priority more than once per
        ``cooldown_sec`` (default 60s). The fuser tracks last emission ts.

    7.  LLM missing / engine returned a degraded verdict (confidence == 0)
        -> rule-only mode. No multiplier, no veto. Score uses rule baseline.

The class is fully synchronous-pure aside from `dispatch()`, which awaits an
optional sink. All decision logic is in `evaluate()` for unit testing.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.screener import SignalEvent, SignalKind

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class FusedSignal:
    """Final output of the fuser."""

    symbol: str
    exchange: str
    ts: int
    direction: Direction
    rule_score: float          # 0..100 from rule detectors only
    llm_score: float           # 0..100 derived from LLM verdict
    final_score: float         # 0..100, after fusion / vetoes / caps
    is_high_priority: bool
    blocked: bool              # True if a veto fired (UI/log only)
    block_reason: str | None
    rule_signals: list[SignalEvent] = field(default_factory=list)
    llm_verdict: AIVerdict | None = None
    notes: list[str] = field(default_factory=list)
    # Mid-price at the moment Task A first detected the trigger condition.
    # Required by Risk Gate's slippage check (SR-1). Set by the bus
    # integration layer; tests construct it explicitly.
    trigger_price: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "ts": self.ts,
            "direction": self.direction.value,
            "rule_score": round(self.rule_score, 2),
            "llm_score": round(self.llm_score, 2),
            "final_score": round(self.final_score, 2),
            "is_high_priority": self.is_high_priority,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "rule_signal_kinds": [s.kind.value for s in self.rule_signals],
            "llm_verdict": (
                self.llm_verdict.model_dump() if self.llm_verdict is not None else None
            ),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Default rule weights
# --------------------------------------------------------------------------- #
#
# Each detector's contribution to the rule-only score (out of 100 BEFORE
# stacking cap is applied).  These mirror the language used in
# requirements.md FR-A2..A5: SMC sweeps and OI silent build are the highest
# signal because they reflect smart-money positioning; volume spike alone is
# noisy (could be exit liquidity).

DEFAULT_RULE_WEIGHTS: dict[SignalKind, float] = {
    SignalKind.VOLUME_SPIKE:       30.0,
    SignalKind.FUNDING_EXTREME:    25.0,
    SignalKind.FUNDING_DEVIATION:  15.0,
    SignalKind.OI_SURGE:           25.0,
    SignalKind.OI_SILENT_BUILD:    35.0,   # the textbook smart-money pattern
    SignalKind.LIQUIDITY_SWEEP:    35.0,   # SMC stop-hunt
    SignalKind.LIQUIDITY_POOL_FORMED: 5.0, # informational; only adds slight bias
}


# --------------------------------------------------------------------------- #
# Rule-direction mapping
# --------------------------------------------------------------------------- #


def _rule_direction(ev: SignalEvent) -> Direction:
    """Infer trade direction from a rule signal's payload."""
    p = ev.payload
    kind = ev.kind

    if kind == SignalKind.VOLUME_SPIKE:
        side = p.get("side")
        if side == "buy":
            return Direction.LONG
        if side == "sell":
            return Direction.SHORT
        return Direction.NEUTRAL

    if kind == SignalKind.FUNDING_EXTREME:
        # short_squeeze: shorts are crowded -> bias LONG
        # long_fragile: longs over-leveraged -> bias SHORT
        d = p.get("direction")
        if d == "short_squeeze":
            return Direction.LONG
        if d == "long_fragile":
            return Direction.SHORT
        return Direction.NEUTRAL

    if kind == SignalKind.FUNDING_DEVIATION:
        # short_avg deviating negative from baseline -> shorts loading -> LONG bias
        z = p.get("zscore", 0.0)
        if z <= -1.0:
            return Direction.LONG
        if z >= 1.0:
            return Direction.SHORT
        return Direction.NEUTRAL

    if kind in (SignalKind.OI_SURGE, SignalKind.OI_SILENT_BUILD):
        # Direction comes from PRICE motion, not from the kind alone.
        # OI growth + price up   = longs piling in   -> LONG
        # OI growth + price down = SHORTS piling in -> SHORT  (smart-money distribution!)
        # OI growth + price flat = ambiguous; let other rules decide -> NEUTRAL
        from_p = p.get("from_price", 0.0)
        to_p = p.get("to_price", 0.0)
        if from_p <= 0:
            return Direction.NEUTRAL
        move_pct = (to_p - from_p) / from_p
        if move_pct > 0.003:        # +0.30%
            return Direction.LONG
        if move_pct < -0.003:       # -0.30%
            return Direction.SHORT
        return Direction.NEUTRAL

    if kind == SignalKind.LIQUIDITY_SWEEP:
        # sweep above sell_side highs that fails -> reversal SHORT
        # sweep below buy_side lows that fails -> reversal LONG
        side = p.get("side")
        if side == "sell_side":
            return Direction.SHORT
        if side == "buy_side":
            return Direction.LONG
        return Direction.NEUTRAL

    if kind == SignalKind.LIQUIDITY_POOL_FORMED:
        # Pool formation is informational; the sweep is what trades.
        return Direction.NEUTRAL

    return Direction.NEUTRAL


def _llm_direction(verdict: AIVerdict) -> Direction:
    if verdict.intent == "pump":
        return Direction.LONG
    if verdict.intent == "dump":
        return Direction.SHORT
    return Direction.NEUTRAL


# --------------------------------------------------------------------------- #
# ScoreFuser
# --------------------------------------------------------------------------- #


FusedSink = Callable[[FusedSignal], Awaitable[None]]


@dataclass
class FuserConfig:
    high_priority_threshold: float = 85.0
    rule_score_cap: float = 95.0       # rule-only ceiling (LLM can boost above)
    final_score_cap: float = 100.0
    llm_boost_max: float = 1.25        # max multiplier when LLM strongly agrees
    llm_boost_min: float = 1.00        # neutral LLM = no change
    veto_disagree_score: float = 40.0  # final score when LLM disagrees with rules
    kol_exit_hard_veto_conf: float = 0.70
    kol_exit_hard_veto_score: float = 30.0
    kol_exit_soft_cap_score: float = 70.0
    conflict_penalty: float = 15.0     # mixed-direction rule signals
    # SR-3 / TA-07: wash trading detection. When a WASH_TRADING_DETECTED
    # event is fresh in the window AND the rule direction is LONG, we apply
    # a hard veto. SHORT entries are NOT vetoed — wash trading typically
    # precedes a real dump (manipulators artificially propping the price for
    # exit liquidity), so a short alongside it is exactly the trade we want.
    wash_trading_veto_score: float = 25.0
    window_sec: int = 90
    cooldown_sec: int = 60
    require_min_rule_score: float = 35.0  # below this, even high LLM can't promote


class ScoreFuser:
    """
    Stateful fuser. Maintains a rolling window of recent rule signals and the
    latest LLM verdict per (exchange, symbol). Call ``on_rule_signal`` /
    ``on_llm_verdict`` from the bus consumers. Each ingestion triggers an
    ``evaluate`` and, if the result crosses threshold, dispatches to the sink.

    For pure unit testing, call ``evaluate(symbol, exchange, now_ts)`` directly.
    """

    def __init__(
        self,
        sink: FusedSink | None = None,
        config: FuserConfig | None = None,
        rule_weights: dict[SignalKind, float] | None = None,
    ):
        self.sink = sink
        self.cfg = config or FuserConfig()
        self.weights = rule_weights or DEFAULT_RULE_WEIGHTS

        # rolling per-(exchange,symbol) state
        self._rules: dict[str, deque[SignalEvent]] = {}
        self._llm: dict[str, AIVerdict] = {}
        self._llm_ts: dict[str, int] = {}
        self._last_high_priority_ts: dict[str, int] = {}

    # -------------------------- ingestion API -------------------------- #

    async def on_rule_signal(self, ev: SignalEvent) -> FusedSignal | None:
        key = self._key(ev.exchange, ev.symbol)
        bucket = self._rules.setdefault(key, deque(maxlen=64))
        bucket.append(ev)
        return await self._dispatch_if_changed(ev.symbol, ev.exchange, ev.ts)

    async def on_llm_verdict(
        self, exchange: str, symbol: str, verdict: AIVerdict, ts: int,
    ) -> FusedSignal | None:
        key = self._key(exchange, symbol)
        self._llm[key] = verdict
        self._llm_ts[key] = ts
        return await self._dispatch_if_changed(symbol, exchange, ts)

    # ---------------------- evaluation core ---------------------- #

    def evaluate(self, symbol: str, exchange: str, now_ts: int) -> FusedSignal:
        """
        Pure decision function. Inputs come from the fuser's internal state;
        ``now_ts`` is the reference time used to drop stale signals.
        """
        key = self._key(exchange, symbol)

        # 1) drop stale rule signals
        window_ms = self.cfg.window_sec * 1000
        bucket = self._rules.get(key, deque())
        fresh: list[SignalEvent] = [s for s in bucket if now_ts - s.ts <= window_ms]

        # 2) maybe LLM
        verdict: AIVerdict | None = None
        v_ts = self._llm_ts.get(key)
        v = self._llm.get(key)
        if v is not None and v_ts is not None and now_ts - v_ts <= 2 * window_ms:
            verdict = v

        notes: list[str] = []

        # 3) rule-only score and direction
        rule_score, rule_direction, conflict = self._score_rules(fresh, notes)

        # 3b) Wash-trading hard veto (SR-3 / TA-07).
        # If a WASH_TRADING_DETECTED event sits inside the active window AND
        # the rule_direction is LONG, refuse to promote — this is the
        # definitional "fake pump" scenario. Shorts are explicitly allowed
        # to ride the wash, since wash spikes typically precede a real dump.
        wash_events = [
            e for e in fresh if e.kind == SignalKind.WASH_TRADING_DETECTED
        ]
        if wash_events and rule_direction == Direction.LONG:
            ev = wash_events[-1]
            notes.append(
                f"HARD VETO (wash trading on LONG): patterns="
                f"{ev.payload.get('patterns')}, "
                f"vol/count_z_ratio={ev.payload.get('volume_to_count_z_ratio')}"
            )
            return self._make_signal(
                symbol=symbol, exchange=exchange, ts=now_ts,
                rule_score=rule_score, llm_score=0.0,
                final_score=min(rule_score, self.cfg.wash_trading_veto_score),
                direction=Direction.NEUTRAL,
                is_high_priority=False,
                blocked=True,
                block_reason="wash_trading_detected",
                rule_signals=fresh,
                llm_verdict=None,
                notes=notes,
            )

        # 4) LLM score (independent informational, used in payload)
        llm_score = 0.0
        if verdict is not None and verdict.confidence_score > 0:
            # LLM score is its raw confidence_score, only meaningful as
            # context. The actual fusion uses the multiplier model below.
            llm_score = float(verdict.confidence_score)

        # 5) Vetoes BEFORE multiplication.  Order matters: KOL exit > direction.
        blocked = False
        block_reason: str | None = None
        final = rule_score
        direction = rule_direction

        if verdict is not None and verdict.confidence_score > 0:
            llm_dir = _llm_direction(verdict)

            # 5a) Direction conflict veto (highest priority - run before any boost)
            if (
                rule_direction != Direction.NEUTRAL
                and llm_dir != Direction.NEUTRAL
                and llm_dir != rule_direction
            ):
                notes.append(
                    f"VETO: rule_direction={rule_direction.value} "
                    f"vs llm_intent={verdict.intent} (conf={verdict.confidence:.2f})"
                )
                return self._make_signal(
                    symbol=symbol, exchange=exchange, ts=now_ts,
                    rule_score=rule_score, llm_score=llm_score,
                    final_score=min(self.cfg.veto_disagree_score, rule_score),
                    direction=Direction.NEUTRAL,
                    is_high_priority=False,
                    blocked=True,
                    block_reason="direction_conflict",
                    rule_signals=fresh,
                    llm_verdict=verdict,
                    notes=notes,
                )

            # 5b) KOL exit_liquidity handling — DIRECTION-AWARE.
            #     - On a LONG signal: KOL distributing IS the trap. VETO/CAP.
            #     - On a SHORT signal: KOL distributing IS the catalyst we want
            #       to ride. The "smart" play is to short alongside the dump
            #       liquidity. Do NOT veto, but slightly DAMPEN to require very
            #       strong confluence (we still need to confirm with rule signals).
            kol_exit = verdict.kol_intent == "exit_liquidity"
            kol_exit_aligned_short = kol_exit and rule_direction == Direction.SHORT

            if kol_exit and not kol_exit_aligned_short:
                # Trap territory: rule says LONG but KOLs are dumping.
                if verdict.confidence >= self.cfg.kol_exit_hard_veto_conf:
                    notes.append(
                        f"HARD VETO: kol_intent=exit_liquidity on LONG, llm.confidence="
                        f"{verdict.confidence:.2f} >= {self.cfg.kol_exit_hard_veto_conf}"
                    )
                    return self._make_signal(
                        symbol=symbol, exchange=exchange, ts=now_ts,
                        rule_score=rule_score, llm_score=llm_score,
                        final_score=min(rule_score, self.cfg.kol_exit_hard_veto_score),
                        direction=Direction.NEUTRAL,
                        is_high_priority=False,
                        blocked=True,
                        block_reason="kol_exit_liquidity_hard_veto",
                        rule_signals=fresh,
                        llm_verdict=verdict,
                        notes=notes,
                    )

            # 5c) LLM agreement -> multiplier boost.
            #     - Pure agreement (no exit_liquidity flag): full boost
            #     - SHORT + exit_liquidity (aligned): MILD boost (KOL behavior is
            #       the very evidence we're trading on, so it does add weight)
            #     - LONG + exit_liquidity (non-vetoed soft case): no boost
            if llm_dir == rule_direction and llm_dir != Direction.NEUTRAL:
                if not kol_exit or kol_exit_aligned_short:
                    spread = self.cfg.llm_boost_max - self.cfg.llm_boost_min
                    # Aligned short with exit_liquidity gets a HALF multiplier
                    # bump (still positive, since this is supporting evidence).
                    effective_conf = (
                        verdict.confidence * 0.5 if kol_exit_aligned_short
                        else verdict.confidence
                    )
                    multiplier = self.cfg.llm_boost_min + spread * effective_conf
                    before = final
                    final = min(final * multiplier, self.cfg.final_score_cap)
                    if kol_exit_aligned_short:
                        notes.append(
                            f"LLM agree (SHORT alongside KOL exit_liquidity): "
                            f"x{multiplier:.3f} ({before:.1f} -> {final:.1f})"
                        )
                    else:
                        notes.append(
                            f"LLM agree: x{multiplier:.3f} ({before:.1f} -> {final:.1f})"
                        )

            # 5d) KOL exit_liquidity SOFT CAP for the LONG-case-only.
            #     SHORT-aligned-with-exit_liquidity is NOT capped — we WANT to
            #     promote that to high_priority because it's the dump-front-run.
            if kol_exit and not kol_exit_aligned_short:
                notes.append(
                    f"SOFT CAP: kol_intent=exit_liquidity on LONG, llm.confidence="
                    f"{verdict.confidence:.2f} < {self.cfg.kol_exit_hard_veto_conf}, "
                    f"capping at {self.cfg.kol_exit_soft_cap_score}"
                )
                final = min(final, self.cfg.kol_exit_soft_cap_score)

        # 6) conflict penalty (intra-rule)
        if conflict:
            final = max(final - self.cfg.conflict_penalty, 0.0)
            notes.append(f"intra-rule direction conflict: -{self.cfg.conflict_penalty}")

        # 7) require minimum rule score for high_priority promotion
        is_high = (
            final >= self.cfg.high_priority_threshold
            and rule_score >= self.cfg.require_min_rule_score
            and direction != Direction.NEUTRAL
        )

        # 8) cooldown
        if is_high:
            last = self._last_high_priority_ts.get(key, 0)
            if now_ts - last < self.cfg.cooldown_sec * 1000:
                notes.append(
                    f"cooldown active ({self.cfg.cooldown_sec}s); not promoting to high_priority"
                )
                is_high = False

        return self._make_signal(
            symbol=symbol, exchange=exchange, ts=now_ts,
            rule_score=rule_score, llm_score=llm_score,
            final_score=round(final, 2),
            direction=direction,
            is_high_priority=is_high,
            blocked=blocked,
            block_reason=block_reason,
            rule_signals=fresh,
            llm_verdict=verdict,
            notes=notes,
        )

    # ---------------------- helpers ---------------------- #

    @staticmethod
    def _key(exchange: str, symbol: str) -> str:
        return f"{exchange}:{symbol}"

    def _score_rules(
        self, fresh: list[SignalEvent], notes: list[str]
    ) -> tuple[float, Direction, bool]:
        """Returns (rule_score, dominant_direction, conflict_flag)."""
        if not fresh:
            return 0.0, Direction.NEUTRAL, False

        # Dedupe: at most one contribution per SignalKind in the window
        # (multiple volume spikes in the same window shouldn't double-count).
        best_per_kind: dict[SignalKind, SignalEvent] = {}
        for ev in fresh:
            cur = best_per_kind.get(ev.kind)
            # later events win on tie (most recent payload wins)
            if cur is None or ev.ts >= cur.ts:
                best_per_kind[ev.kind] = ev

        score = 0.0
        long_w = 0.0
        short_w = 0.0
        for kind, ev in best_per_kind.items():
            w = self.weights.get(kind, 0.0)
            score += w
            d = _rule_direction(ev)
            if d == Direction.LONG:
                long_w += w
            elif d == Direction.SHORT:
                short_w += w

        score = min(score, self.cfg.rule_score_cap)

        # determine dominant direction
        if long_w == 0 and short_w == 0:
            direction = Direction.NEUTRAL
        elif long_w >= short_w * 1.5:
            direction = Direction.LONG
        elif short_w >= long_w * 1.5:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL  # mixed

        conflict = (long_w > 0 and short_w > 0) and direction == Direction.NEUTRAL
        if conflict:
            notes.append(
                f"rule directions split: long_w={long_w:.1f}, short_w={short_w:.1f}"
            )

        return score, direction, conflict

    def _make_signal(self, **kw: Any) -> FusedSignal:
        return FusedSignal(**kw)

    async def _dispatch_if_changed(
        self, symbol: str, exchange: str, now_ts: int,
    ) -> FusedSignal | None:
        signal = self.evaluate(symbol, exchange, now_ts)
        if signal.is_high_priority:
            self._last_high_priority_ts[self._key(exchange, symbol)] = now_ts
            if self.sink is not None:
                await self.sink(signal)
        return signal
