"""symbol_profile.py — Per-symbol quadrant profile (QUADRANT_STRATEGY_PLAN Phase 1).

Every symbol the agent watches gets classified into one of four quadrants
based on (social_score, liquidity_score), each in [0, 100]:

    A  high-quality 妖币:  social >= 70  AND  liq >= 70
    B  抱团 (cluster) 妖币:  social >= 70  AND  liq <  70
    C  庄拉 (manipulated) 妖币:  social <  70  AND  liq >= 70
    D  砸盘 (dump) 币:        social <  70  AND  liq <  70   (mostly skip)

Each quadrant carries a different risk parameter profile (max_risk_per_trade,
leverage caps, rolling, trailing tightness, anti-chase ceiling, daily DD,
confidence threshold, LLM call rate). The plan's matrix is encoded once in
``QuadrantParams`` and consumed by ``RiskGate`` / ``PositionSizer`` / ``Fuser``
via the resolved ``SymbolProfile``.

Profiles are persisted as JSON to ``.kiro/state/symbol_profiles.json`` so a
restart picks up where the last 6h refresh left off.

This module is intentionally pure-data + pure-logic; the *refresh* loop that
recomputes social/liquidity scores from live observations belongs upstream
(screener / dashboard cron). Phase 1 ships only the scaffolding + persistence;
the refresh worker is wired in Phase 4-5.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Quadrant(str, Enum):
    A = "A"  # high-quality 妖币
    B = "B"  # 抱团 (cluster)
    C = "C"  # 庄拉 (manipulated)
    D = "D"  # 砸盘 (dump)


# --------------------------------------------------------------------- #
# Quadrant parameter matrix (from QUADRANT_STRATEGY_PLAN section 三)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuadrantParams:
    """Risk + execution parameters for one quadrant.

    All values come straight from the plan's matrix. Operators may override
    via app.yaml -> risk.quadrant_overrides.<A|B|C|D>; that wiring lives in
    ``main.py`` and is *not* baked here so this dataclass stays a pure
    spec object.
    """

    quadrant: Quadrant
    max_risk_per_trade: float
    max_leverage_long: float
    max_leverage_short: float
    rolling_enabled: bool
    rolling_max_legs: int
    trailing_atr_mult: float
    anti_chase_max_move_pct: float
    breakeven_at_r: float
    daily_drawdown_limit: float
    short_on_blowoff_top: bool
    confidence_threshold: float
    # "high" / "medium" / "low" — interpreted by TokenBudgetManager.
    llm_call_rate: str

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quadrant"] = self.quadrant.value
        return d


# Plan section 三 verbatim. Keep this as the single source of truth.
DEFAULT_QUADRANT_PARAMS: dict[Quadrant, QuadrantParams] = {
    Quadrant.A: QuadrantParams(
        quadrant=Quadrant.A,
        max_risk_per_trade=0.025,
        max_leverage_long=15.0,
        max_leverage_short=10.0,
        rolling_enabled=True,
        rolling_max_legs=4,
        trailing_atr_mult=2.5,
        anti_chase_max_move_pct=0.06,
        breakeven_at_r=1.5,
        daily_drawdown_limit=0.12,
        short_on_blowoff_top=True,
        confidence_threshold=0.80,
        llm_call_rate="high",
    ),
    Quadrant.B: QuadrantParams(
        quadrant=Quadrant.B,
        max_risk_per_trade=0.015,
        max_leverage_long=10.0,
        max_leverage_short=8.0,
        rolling_enabled=True,
        rolling_max_legs=2,
        trailing_atr_mult=1.5,
        anti_chase_max_move_pct=0.04,
        breakeven_at_r=1.0,
        daily_drawdown_limit=0.08,
        short_on_blowoff_top=True,
        confidence_threshold=0.85,
        llm_call_rate="medium",
    ),
    Quadrant.C: QuadrantParams(
        quadrant=Quadrant.C,
        max_risk_per_trade=0.010,
        max_leverage_long=8.0,
        max_leverage_short=5.0,
        rolling_enabled=False,
        rolling_max_legs=1,
        trailing_atr_mult=1.0,
        anti_chase_max_move_pct=0.025,
        breakeven_at_r=0.7,
        daily_drawdown_limit=0.06,
        # Plan: "顶部反手 SHORT  ❌(庄控盘风险)" — manipulator can squeeze.
        short_on_blowoff_top=False,
        confidence_threshold=0.85,
        llm_call_rate="medium",
    ),
    Quadrant.D: QuadrantParams(
        quadrant=Quadrant.D,
        max_risk_per_trade=0.005,
        max_leverage_long=5.0,
        max_leverage_short=5.0,
        rolling_enabled=False,
        rolling_max_legs=0,
        trailing_atr_mult=0.5,
        anti_chase_max_move_pct=0.025,
        breakeven_at_r=0.5,
        daily_drawdown_limit=0.06,
        short_on_blowoff_top=True,
        confidence_threshold=0.90,
        llm_call_rate="low",
    ),
}


def quadrant_params(q: Quadrant) -> QuadrantParams:
    return DEFAULT_QUADRANT_PARAMS[q]


# --------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------- #


SOCIAL_THRESHOLD: float = 70.0
LIQUIDITY_THRESHOLD: float = 70.0


def classify(social_score: float, liquidity_score: float) -> Quadrant:
    """Map (social, liquidity) ∈ [0, 100]² to a quadrant.

    Inputs are clamped: a negative or NaN-ish score is treated as 0 so a
    misbehaving upstream can't crash classification. ``>=`` boundaries on
    both axes match the plan.
    """
    s = max(0.0, float(social_score)) if social_score == social_score else 0.0
    liq = max(0.0, float(liquidity_score)) if liquidity_score == liquidity_score else 0.0
    high_social = s >= SOCIAL_THRESHOLD
    high_liq = liq >= LIQUIDITY_THRESHOLD
    if high_social and high_liq:
        return Quadrant.A
    if high_social and not high_liq:
        return Quadrant.B
    if not high_social and high_liq:
        return Quadrant.C
    return Quadrant.D


# --------------------------------------------------------------------- #
# SymbolProfile
# --------------------------------------------------------------------- #


@dataclass
class SymbolProfile:
    """Per-symbol summary the rest of the system reads.

    Refreshed every 6h by the screener cron (Phase 4-5). Persisted between
    restarts. ``confidence_threshold`` is the per-symbol override of the
    quadrant default; the trainer (Phase 4) may tighten it for a symbol
    with a poor historical track record without retiring the whole quadrant.
    """

    symbol: str
    quadrant: Quadrant
    social_score: float
    liquidity_score: float
    historical_win_rate: float = 0.0
    last_pump_ts: int = 0
    scam_score: float = 0.0
    confidence_threshold: float | None = None  # None = use quadrant default
    samples: int = 0  # number of historical trades observed
    refreshed_at_ts: int = 0
    notes: list[str] = field(default_factory=list)

    # ---- params resolution ---- #

    def params(self) -> QuadrantParams:
        """The full risk/execution parameters for this symbol."""
        return DEFAULT_QUADRANT_PARAMS[self.quadrant]

    def effective_confidence_threshold(self) -> float:
        """Per-symbol override falls back to the quadrant default."""
        if self.confidence_threshold is not None:
            return self.confidence_threshold
        return self.params().confidence_threshold

    # ---- serialization ---- #

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quadrant": self.quadrant.value,
            "social_score": self.social_score,
            "liquidity_score": self.liquidity_score,
            "historical_win_rate": self.historical_win_rate,
            "last_pump_ts": self.last_pump_ts,
            "scam_score": self.scam_score,
            "confidence_threshold": self.confidence_threshold,
            "samples": self.samples,
            "refreshed_at_ts": self.refreshed_at_ts,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolProfile:
        return cls(
            symbol=str(d["symbol"]),
            quadrant=Quadrant(d.get("quadrant", "D")),
            social_score=float(d.get("social_score", 0.0)),
            liquidity_score=float(d.get("liquidity_score", 0.0)),
            historical_win_rate=float(d.get("historical_win_rate", 0.0)),
            last_pump_ts=int(d.get("last_pump_ts", 0)),
            scam_score=float(d.get("scam_score", 0.0)),
            confidence_threshold=(
                float(d["confidence_threshold"])
                if d.get("confidence_threshold") is not None
                else None
            ),
            samples=int(d.get("samples", 0)),
            refreshed_at_ts=int(d.get("refreshed_at_ts", 0)),
            notes=list(d.get("notes") or []),
        )

    @classmethod
    def from_scores(
        cls,
        symbol: str,
        social_score: float,
        liquidity_score: float,
        *,
        now_ts: int | None = None,
    ) -> SymbolProfile:
        return cls(
            symbol=symbol,
            quadrant=classify(social_score, liquidity_score),
            social_score=float(social_score),
            liquidity_score=float(liquidity_score),
            refreshed_at_ts=int(now_ts if now_ts is not None else time.time()),
        )


# --------------------------------------------------------------------- #
# SymbolProfileStore — JSON persistence
# --------------------------------------------------------------------- #


@dataclass
class SymbolProfileStore:
    """Simple JSON-on-disk store, atomic write via tmpfile + rename.

    Keeping this dependency-free (no SQLite, no aiofiles) so the Phase 1
    scaffolding is trivially mock-testable. Scaling concerns belong in the
    Phase 4+ training cron, which already has heavier persistence.
    """

    path: str
    _by_symbol: dict[str, SymbolProfile] = field(default_factory=dict)
    _loaded: bool = False

    # ---- io ---- #

    def load(self) -> None:
        """Idempotent. Missing file is treated as empty store."""
        if self._loaded:
            return
        if not os.path.exists(self.path):
            self._loaded = True
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "SymbolProfileStore: failed to load %s (%s); starting empty",
                self.path,
                exc,
            )
            self._loaded = True
            return
        items = raw.get("profiles") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            logger.warning(
                "SymbolProfileStore: %s is not in expected shape; ignoring",
                self.path,
            )
            self._loaded = True
            return
        for entry in items:
            try:
                p = SymbolProfile.from_dict(entry)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("SymbolProfileStore: skipping malformed entry: %s", exc)
                continue
            self._by_symbol[p.symbol] = p
        self._loaded = True

    def save(self) -> None:
        """Atomic write so a crash mid-flush can't corrupt the file."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "saved_at_ts": int(time.time()),
            "profiles": [p.as_dict() for p in self._by_symbol.values()],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".symbol_profiles.", suffix=".json.tmp",
            dir=os.path.dirname(self.path) or ".",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- accessors ---- #

    def get(self, symbol: str) -> SymbolProfile | None:
        if not self._loaded:
            self.load()
        return self._by_symbol.get(symbol)

    def upsert(self, profile: SymbolProfile) -> None:
        if not self._loaded:
            self.load()
        self._by_symbol[profile.symbol] = profile

    def remove(self, symbol: str) -> None:
        if not self._loaded:
            self.load()
        self._by_symbol.pop(symbol, None)

    def all_symbols(self) -> list[str]:
        if not self._loaded:
            self.load()
        return sorted(self._by_symbol)

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._by_symbol)


__all__ = [
    "DEFAULT_QUADRANT_PARAMS",
    "LIQUIDITY_THRESHOLD",
    "Quadrant",
    "QuadrantParams",
    "SOCIAL_THRESHOLD",
    "SymbolProfile",
    "SymbolProfileStore",
    "classify",
    "quadrant_params",
]
