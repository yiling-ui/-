"""
fuser.py — Score Fuser with hot-loaded learned rules (V1.0).

This is the central decision module: it consumes ``SignalEvent`` (from
screener.py) and ``AIVerdict`` (from ai_engine.py) and emits ``FusedSignal``,
the single object the Risk Gate cares about.

Hard rules (committed with the architect):

  1. The rule score is the BASE. The LLM is a multiplier + veto.
  2. KOL exit_liquidity is direction-aware:
       - On LONG with high LLM confidence -> HARD VETO.
       - On LONG with lower confidence    -> SOFT CAP (final score capped).
       - On SHORT alongside KOL distribution -> ALLOWED, but LLM boost halved
         (we're shorting INTO their offload).
  3. Wash trading on LONG -> HARD VETO (SR-3). Shorts are not vetoed: the
     fake pump usually precedes a real dump, so shorting it is correct.
  4. Direction conflict between rules and LLM intent -> HARD VETO.
  5. Window: rule events older than ``window_sec`` are dropped.
  6. Cooldown: same symbol cannot emit ``high_priority`` twice within
     ``cooldown_sec``.
  7. Learned rules from ``dynamic_rules.json`` apply as additional
     multipliers, with strict caps:
        * mtime-throttled hot reload (default >= 5s between disk checks)
        * IO failure degrades to last-good index, never raises
        * Confidence-shrunk lift: n / (n + k), k=5 by default
        * Asymmetric: rewards cap at +20%, penalties cap at -30%
        * Reward path:  reward_mult = max(llm_boost, learned_reward),
                        capped at ``learned_overall_reward_cap`` (default 1.30)
        * Penalty path: penalties stack multiplicatively
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.learning_engine import (
    bucket_funding,
    bucket_pct,
    bucket_zscore,
)
from altcoin_agent.screener import SignalEvent, SignalKind

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------- #


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class FusedSignal:
    """The single object the Risk Gate consumes."""

    symbol: str
    exchange: str
    ts: int
    direction: Direction
    rule_score: float
    llm_score: float
    final_score: float
    is_high_priority: bool
    blocked: bool
    block_reason: str | None
    rule_signals: list[SignalEvent] = field(default_factory=list)
    llm_verdict: AIVerdict | None = None
    notes: list[str] = field(default_factory=list)
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
            "trigger_price": self.trigger_price,
            "rule_signal_kinds": [s.kind.value for s in self.rule_signals],
            "llm_verdict": (self.llm_verdict.model_dump()
                            if self.llm_verdict is not None else None),
            "notes": self.notes,
        }


# --------------------------------------------------------------------- #
# Default rule weights
# --------------------------------------------------------------------- #


DEFAULT_RULE_WEIGHTS: dict[SignalKind, float] = {
    SignalKind.VOLUME_SPIKE:          30.0,
    SignalKind.FUNDING_EXTREME:       25.0,
    SignalKind.FUNDING_DEVIATION:     15.0,
    SignalKind.OI_SURGE:              25.0,
    SignalKind.OI_SILENT_BUILD:       35.0,
    SignalKind.LIQUIDITY_SWEEP:       35.0,
    SignalKind.LIQUIDITY_POOL_FORMED:  5.0,
    # WASH_TRADING_DETECTED carries no positive weight; it only vetoes.
}


# --------------------------------------------------------------------- #
# Direction inference helpers
# --------------------------------------------------------------------- #


def _rule_direction(ev: SignalEvent) -> Direction:
    """Decide which side a single rule event is biased toward."""
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
        d = p.get("direction")
        if d == "short_squeeze":
            return Direction.LONG
        if d == "long_fragile":
            return Direction.SHORT
        return Direction.NEUTRAL

    if kind == SignalKind.FUNDING_DEVIATION:
        z = p.get("zscore", 0.0)
        if z <= -1.0:
            return Direction.LONG    # rate dropping fast -> longs about to be bid
        if z >= 1.0:
            return Direction.SHORT   # rate climbing fast -> longs over-leveraged
        return Direction.NEUTRAL

    if kind in (SignalKind.OI_SURGE, SignalKind.OI_SILENT_BUILD):
        from_p = float(p.get("from_price", 0.0))
        to_p = float(p.get("to_price", 0.0))
        if from_p <= 0:
            return Direction.NEUTRAL
        move = (to_p - from_p) / from_p
        if move > 0.003:
            return Direction.LONG
        if move < -0.003:
            return Direction.SHORT  # OI up + price down = SHORT distribution
        return Direction.NEUTRAL

    if kind == SignalKind.LIQUIDITY_SWEEP:
        side = p.get("side")
        if side == "sell_side":
            return Direction.SHORT  # swept the sell-side pool above price
        if side == "buy_side":
            return Direction.LONG   # swept the buy-side pool below price
        return Direction.NEUTRAL

    return Direction.NEUTRAL


def _llm_direction(verdict: AIVerdict) -> Direction:
    if verdict.intent == "pump":
        return Direction.LONG
    if verdict.intent == "dump":
        return Direction.SHORT
    return Direction.NEUTRAL


# --------------------------------------------------------------------- #
# SignalEvent -> learned rule key mapping
# --------------------------------------------------------------------- #
# Maps a live signal event to (feature_name, bucket, side) tuples that
# learning_engine.RuleStore could have stored. The bucket helpers come
# from learning_engine to guarantee the namespace matches.


def signal_to_learned_keys(
    ev: SignalEvent, side: Direction,
) -> list[tuple[str, str, str]]:
    if side == Direction.NEUTRAL:
        return []
    side_str = "pump" if side == Direction.LONG else "dump"
    p = ev.payload
    out: list[tuple[str, str, str]] = []

    if ev.kind == SignalKind.VOLUME_SPIKE:
        z = float(p.get("zscore", 0.0))
        out.append(("volume_zscore_last1h", bucket_zscore(z), side_str))

    elif ev.kind == SignalKind.FUNDING_EXTREME:
        rate = float(p.get("rate", 0.0))
        out.append(("funding_pre2h_extreme", bucket_funding(rate), side_str))

    elif ev.kind == SignalKind.FUNDING_DEVIATION:
        z = float(p.get("zscore", 0.0))
        # Approximate rate-slope bucket via z-score scale.
        out.append(("funding_slope_pre1h", bucket_pct(z / 100.0), side_str))

    elif ev.kind in (SignalKind.OI_SURGE, SignalKind.OI_SILENT_BUILD):
        oi_pct = float(p.get("oi_delta_pct", 0.0))
        out.append(("oi_growth_pre1h", bucket_pct(oi_pct), side_str))
        from_p = float(p.get("from_price", 0.0))
        to_p = float(p.get("to_price", 0.0))
        if from_p > 0:
            price_move = (to_p - from_p) / from_p
            decoupling = abs(oi_pct) - abs(price_move)
            out.append(("oi_price_decoupling", bucket_pct(decoupling), side_str))

    elif ev.kind == SignalKind.LIQUIDITY_SWEEP:
        wb = float(p.get("wick_to_body", 0.0))
        # Map wick:body asymmetry into the same wick-dominance namespace
        # learning_engine measures, scaling by /5 so a 5:1 wick lands in
        # the "large" bucket.
        feature = (
            "upper_wick_dominance_last1h"
            if side == Direction.SHORT
            else "lower_wick_dominance_last1h"
        )
        out.append((feature, bucket_pct(wb / 5.0), side_str))

    return out


# --------------------------------------------------------------------- #
# Learned-rule index (hot-loaded)
# --------------------------------------------------------------------- #


@dataclass
class _LearnedRule:
    feature_name: str
    bucket: str
    side: str
    hits: int
    total: int

    @property
    def hit_rate(self) -> float:
        return (self.hits + 1) / (self.total + 2)


@dataclass
class RuleIndex:
    """In-memory snapshot of dynamic_rules.json with mtime-throttled reload.

    Architectural decisions:
        * mtime is checked at most every ``min_check_interval_sec``
          (default 5s) -- never IO on every signal.
        * On any IO error, the previous good index stays in force.
        * Atomic swap: a fresh dict is built before replacing the old one.
    """

    json_path: Path
    min_check_interval_sec: float = 5.0
    _rules_by_key: dict[str, _LearnedRule] = field(default_factory=dict)
    _last_mtime: float = 0.0
    _last_check_ts: float = 0.0

    def maybe_reload(self) -> None:
        now = time.time()
        if now - self._last_check_ts < self.min_check_interval_sec:
            return
        self._last_check_ts = now
        try:
            if not self.json_path.exists():
                return
            mtime = self.json_path.stat().st_mtime
            if mtime <= self._last_mtime:
                return
            data = json.loads(self.json_path.read_text())
            new_rules: dict[str, _LearnedRule] = {}
            for r in data.get("rules", []):
                try:
                    rule = _LearnedRule(
                        feature_name=str(r["feature_name"]),
                        bucket=str(r["bucket"]),
                        side=str(r["side"]),
                        hits=int(r.get("hits", 0)),
                        total=int(r.get("total", 0)),
                    )
                except Exception as e:
                    logger.warning("RuleIndex: skipping malformed rule %s: %s", r, e)
                    continue
                key = f"{rule.feature_name}|{rule.bucket}|{rule.side}"
                new_rules[key] = rule
            # Atomic swap
            self._rules_by_key = new_rules
            self._last_mtime = mtime
            logger.info("RuleIndex hot-reloaded: %d rules from %s",
                        len(new_rules), self.json_path)
        except Exception as e:
            logger.warning("RuleIndex reload failed (keeping previous): %s", e)

    def lookup(self, feature_name: str, bucket: str, side: str) -> _LearnedRule | None:
        return self._rules_by_key.get(f"{feature_name}|{bucket}|{side}")

    def force_reload_now(self) -> None:
        """Test/admin hook: bypass the throttle on next maybe_reload()."""
        self._last_check_ts = 0.0
        self._last_mtime = 0.0

    def __len__(self) -> int:
        return len(self._rules_by_key)


# --------------------------------------------------------------------- #
# ScoreFuser
# --------------------------------------------------------------------- #


FusedSink = Callable[[FusedSignal], Awaitable[None]]


@dataclass
class FuserConfig:
    high_priority_threshold: float = 85.0
    rule_score_cap: float = 95.0
    final_score_cap: float = 100.0
    llm_boost_max: float = 1.25
    llm_boost_min: float = 1.00
    veto_disagree_score: float = 40.0
    kol_exit_hard_veto_conf: float = 0.70
    kol_exit_hard_veto_score: float = 30.0
    kol_exit_soft_cap_score: float = 70.0
    conflict_penalty: float = 15.0
    wash_trading_veto_score: float = 25.0
    window_sec: int = 90
    cooldown_sec: int = 60
    require_min_rule_score: float = 35.0

    # Learned-rule application
    learned_min_samples: int = 3                # below this -> no effect
    learned_reward_lift_cap: float = 0.20       # +20% max boost
    learned_penalty_lift_cap: float = 0.30      # -30% max penalty
    learned_overall_reward_cap: float = 1.30    # final reward never > 1.30x
    learned_confidence_k: float = 5.0           # n / (n + k) shrinkage

    dynamic_rules_path: Path | None = None


class ScoreFuser:
    def __init__(
        self,
        sink: FusedSink | None = None,
        config: FuserConfig | None = None,
        rule_weights: dict[SignalKind, float] | None = None,
        rule_index: RuleIndex | None = None,
    ):
        self.sink = sink
        self.cfg = config or FuserConfig()
        self.weights = rule_weights or DEFAULT_RULE_WEIGHTS

        # Resolve learned-rules path (env override -> cfg -> default).
        if rule_index is not None:
            self.rule_index = rule_index
        else:
            path_str = os.getenv("DYNAMIC_RULES_JSON")
            if path_str:
                p = Path(path_str)
            elif self.cfg.dynamic_rules_path is not None:
                p = self.cfg.dynamic_rules_path
            else:
                # Default: <repo>/.kiro/steering/dynamic_rules.json
                here = Path(__file__).resolve()
                p = here.parents[2] / ".kiro" / "steering" / "dynamic_rules.json"
            self.rule_index = RuleIndex(json_path=p)

        self._rules: dict[str, deque[SignalEvent]] = {}
        self._llm: dict[str, AIVerdict] = {}
        self._llm_ts: dict[str, int] = {}
        self._last_high_priority_ts: dict[str, int] = {}

    # ------------------- ingestion API ------------------- #

    async def on_rule_signal(self, ev: SignalEvent) -> FusedSignal | None:
        key = self._key(ev.exchange, ev.symbol)
        bucket = self._rules.setdefault(key, deque(maxlen=64))
        bucket.append(ev)
        return await self._dispatch(ev.symbol, ev.exchange, ev.ts)

    async def on_llm_verdict(
        self, exchange: str, symbol: str, verdict: AIVerdict, ts: int,
    ) -> FusedSignal | None:
        key = self._key(exchange, symbol)
        self._llm[key] = verdict
        self._llm_ts[key] = ts
        return await self._dispatch(symbol, exchange, ts)

    # ------------------- evaluation core ------------------- #

    def evaluate(self, symbol: str, exchange: str, now_ts: int) -> FusedSignal:
        # 0) Hot-load learned rules (mtime-throttled, IO-safe).
        self.rule_index.maybe_reload()

        key = self._key(exchange, symbol)
        window_ms = self.cfg.window_sec * 1000
        bucket = self._rules.get(key, deque())
        fresh: list[SignalEvent] = [s for s in bucket if now_ts - s.ts <= window_ms]

        # LLM verdicts age out at 2x the rule window.
        verdict: AIVerdict | None = None
        v_ts = self._llm_ts.get(key)
        v = self._llm.get(key)
        if v is not None and v_ts is not None and now_ts - v_ts <= 2 * window_ms:
            verdict = v

        notes: list[str] = []

        rule_score, rule_direction, conflict = self._score_rules(fresh, notes)
        trigger_price = self._extract_trigger_price(fresh)

        # 1) Wash trading hard veto on LONG (SR-3).
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
                direction=Direction.NEUTRAL, is_high_priority=False,
                blocked=True, block_reason="wash_trading_detected",
                rule_signals=fresh, llm_verdict=None, notes=notes,
                trigger_price=trigger_price,
            )

        llm_score = (
            float(verdict.confidence_score)
            if verdict is not None and verdict.confidence_score > 0
            else 0.0
        )

        final = rule_score
        direction = rule_direction
        llm_boost = 1.0
        kol_exit = False
        kol_exit_aligned_short = False

        if verdict is not None and verdict.confidence_score > 0:
            llm_dir = _llm_direction(verdict)
            kol_exit = verdict.kol_intent == "exit_liquidity"
            kol_exit_aligned_short = (
                kol_exit and rule_direction == Direction.SHORT
            )

            # 2) Direction conflict veto.
            if (rule_direction != Direction.NEUTRAL
                    and llm_dir != Direction.NEUTRAL
                    and llm_dir != rule_direction):
                notes.append(
                    f"VETO: rule_direction={rule_direction.value} "
                    f"vs llm_intent={verdict.intent} (conf={verdict.confidence:.2f})"
                )
                return self._make_signal(
                    symbol=symbol, exchange=exchange, ts=now_ts,
                    rule_score=rule_score, llm_score=llm_score,
                    final_score=min(self.cfg.veto_disagree_score, rule_score),
                    direction=Direction.NEUTRAL, is_high_priority=False,
                    blocked=True, block_reason="direction_conflict",
                    rule_signals=fresh, llm_verdict=verdict, notes=notes,
                    trigger_price=trigger_price,
                )

            # 3) KOL exit_liquidity HARD VETO (only on LONG with high LLM conf).
            if kol_exit and not kol_exit_aligned_short:
                if verdict.confidence >= self.cfg.kol_exit_hard_veto_conf:
                    notes.append(
                        f"HARD VETO: kol_intent=exit_liquidity on LONG, "
                        f"llm.confidence={verdict.confidence:.2f} >= "
                        f"{self.cfg.kol_exit_hard_veto_conf}"
                    )
                    return self._make_signal(
                        symbol=symbol, exchange=exchange, ts=now_ts,
                        rule_score=rule_score, llm_score=llm_score,
                        final_score=min(rule_score, self.cfg.kol_exit_hard_veto_score),
                        direction=Direction.NEUTRAL, is_high_priority=False,
                        blocked=True,
                        block_reason="kol_exit_liquidity_hard_veto",
                        rule_signals=fresh, llm_verdict=verdict, notes=notes,
                        trigger_price=trigger_price,
                    )

            # 4) LLM agreement boost (deferred — applied below).
            if llm_dir == rule_direction and llm_dir != Direction.NEUTRAL:
                spread = self.cfg.llm_boost_max - self.cfg.llm_boost_min
                # Halve boost when shorting alongside KOL distribution.
                eff_conf = (
                    verdict.confidence * 0.5
                    if kol_exit_aligned_short
                    else verdict.confidence
                )
                llm_boost = self.cfg.llm_boost_min + spread * eff_conf

        # 5) Learned-rule adjustment (asymmetric, capped, confidence-shrunk).
        learned_reward, learned_penalty = self._compute_learned_adjustment(
            fresh, rule_direction, notes,
        )
        # reward_mult = max(LLM agree, learned reward), capped overall.
        reward_mult = max(llm_boost, learned_reward)
        reward_mult = min(reward_mult, self.cfg.learned_overall_reward_cap)
        # Penalty stacks regardless.
        final = final * reward_mult * learned_penalty
        final = min(final, self.cfg.final_score_cap)

        if reward_mult > 1.0:
            notes.append(
                f"reward x{reward_mult:.3f} "
                f"(llm_boost={llm_boost:.3f}, learned={learned_reward:.3f}, "
                f"capped at {self.cfg.learned_overall_reward_cap}); "
                f"kol_exit_aligned_short={kol_exit_aligned_short}"
            )

        # 6) KOL exit_liquidity SOFT CAP for LONG (low LLM conf path).
        if kol_exit and not kol_exit_aligned_short and verdict is not None:
            notes.append(
                f"SOFT CAP: kol_intent=exit_liquidity on LONG, "
                f"llm.confidence={verdict.confidence:.2f} < "
                f"{self.cfg.kol_exit_hard_veto_conf}, "
                f"capping at {self.cfg.kol_exit_soft_cap_score}"
            )
            final = min(final, self.cfg.kol_exit_soft_cap_score)

        # 7) Intra-rule conflict penalty.
        if conflict:
            final = max(final - self.cfg.conflict_penalty, 0.0)
            notes.append(
                f"intra-rule direction conflict: -{self.cfg.conflict_penalty}"
            )

        # 8) High-priority gate.
        is_high = (
            final >= self.cfg.high_priority_threshold
            and rule_score >= self.cfg.require_min_rule_score
            and direction != Direction.NEUTRAL
        )

        # 9) Cooldown.
        if is_high:
            last = self._last_high_priority_ts.get(key, 0)
            if now_ts - last < self.cfg.cooldown_sec * 1000:
                notes.append(
                    f"cooldown active ({self.cfg.cooldown_sec}s); "
                    "not promoting to high_priority"
                )
                is_high = False

        return self._make_signal(
            symbol=symbol, exchange=exchange, ts=now_ts,
            rule_score=rule_score, llm_score=llm_score,
            final_score=round(final, 2), direction=direction,
            is_high_priority=is_high, blocked=False, block_reason=None,
            rule_signals=fresh, llm_verdict=verdict, notes=notes,
            trigger_price=trigger_price,
        )

    # ---------------- learned-rule helper ---------------- #

    def _compute_learned_adjustment(
        self,
        fresh: list[SignalEvent],
        rule_direction: Direction,
        notes: list[str],
    ) -> tuple[float, float]:
        """Returns (reward_multiplier, penalty_multiplier).

        reward_multiplier  in [1.00, 1 + learned_reward_lift_cap]
        penalty_multiplier in [1 - learned_penalty_lift_cap, 1.00]
        """
        if rule_direction == Direction.NEUTRAL:
            return 1.0, 1.0

        max_reward_lift = 0.0
        cumulative_penalty = 1.0
        applied: list[str] = []

        for ev in fresh:
            for fname, bucket, side in signal_to_learned_keys(ev, rule_direction):
                rule = self.rule_index.lookup(fname, bucket, side)
                if rule is None or rule.total < self.cfg.learned_min_samples:
                    continue
                hr = rule.hit_rate
                # Map hit_rate in [0,1] to raw lift in [-1, +1].
                raw_lift = (hr - 0.5) * 2.0
                # Confidence shrinkage.
                conf = rule.total / (rule.total + self.cfg.learned_confidence_k)
                adj = raw_lift * conf
                if adj > 0:
                    adj = min(adj, self.cfg.learned_reward_lift_cap)
                    max_reward_lift = max(max_reward_lift, adj)
                else:
                    pen = max(adj, -self.cfg.learned_penalty_lift_cap)
                    cumulative_penalty *= (1.0 + pen)
                applied.append(
                    f"{fname}|{bucket}|{side}"
                    f"({rule.hits}/{rule.total}={hr:.2%}, adj={adj:+.3f})"
                )

        reward_mult = 1.0 + max_reward_lift
        cumulative_penalty = max(
            cumulative_penalty, 1.0 - self.cfg.learned_penalty_lift_cap,
        )

        if applied:
            notes.append(
                f"learned rules applied: reward x{reward_mult:.3f}, "
                f"penalty x{cumulative_penalty:.3f} "
                f"[{'; '.join(applied[:5])}]"
            )

        return reward_mult, cumulative_penalty

    # ---------------- helpers ---------------- #

    @staticmethod
    def _key(exchange: str, symbol: str) -> str:
        return f"{exchange}:{symbol}"

    def _score_rules(
        self, fresh: list[SignalEvent], notes: list[str],
    ) -> tuple[float, Direction, bool]:
        if not fresh:
            return 0.0, Direction.NEUTRAL, False

        # Keep only the most recent event of each kind to avoid double-counting.
        best: dict[SignalKind, SignalEvent] = {}
        for ev in fresh:
            cur = best.get(ev.kind)
            if cur is None or ev.ts >= cur.ts:
                best[ev.kind] = ev

        score = 0.0
        long_w = 0.0
        short_w = 0.0
        for kind, ev in best.items():
            w = self.weights.get(kind, 0.0)
            score += w
            d = _rule_direction(ev)
            if d == Direction.LONG:
                long_w += w
            elif d == Direction.SHORT:
                short_w += w

        score = min(score, self.cfg.rule_score_cap)

        if long_w == 0 and short_w == 0:
            direction = Direction.NEUTRAL
        elif long_w >= short_w * 1.5:
            direction = Direction.LONG
        elif short_w >= long_w * 1.5:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        conflict = (long_w > 0 and short_w > 0) and direction == Direction.NEUTRAL
        if conflict:
            notes.append(
                f"rule directions split: long_w={long_w:.1f}, short_w={short_w:.1f}"
            )
        return score, direction, conflict

    @staticmethod
    def _extract_trigger_price(fresh: list[SignalEvent]) -> float | None:
        """Find a representative price from the freshest non-funding event."""
        for ev in reversed(fresh):
            p = ev.payload
            if "to_price" in p:
                return float(p["to_price"])
            if "bar_close" in p:
                return float(p["bar_close"])
        return None

    def _make_signal(self, **kw: Any) -> FusedSignal:
        return FusedSignal(**kw)

    async def _dispatch(
        self, symbol: str, exchange: str, now_ts: int,
    ) -> FusedSignal | None:
        signal = self.evaluate(symbol, exchange, now_ts)
        if signal.is_high_priority:
            self._last_high_priority_ts[self._key(exchange, symbol)] = now_ts
            if self.sink is not None:
                await self.sink(signal)
        return signal
