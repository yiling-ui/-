"""
learning_engine.py — strategy evolution via post-mortem analysis.

The pipeline (per user mandate):

    1. Historical Slicer:  symbol + target_ts → 4h pre-event slice
       (OHLCV/OI/funding via OKX public APIs). Bounded resource use,
       runs once per confirmed event.

    2. Feature Extractor:  pure function, no LLM. Computes 8 quantitative
       candidate features at fixed lookback windows. Reports each as
       (name, value, direction-correlation). The point of this layer is
       to give the LLM a CLOSED CHOICE of grounded features instead of a
       blank canvas — so it cannot post-hoc invent signals.

    3. Result Computer:  derives the realized event from the slice
       (magnitude, side, time-to-extreme).

    4. DeepSeek Post-Mortem:  receives (features, result), picks 1-2 of
       the eight candidates as MOST PROGNOSTIC, and explains why. We
       enforce a strict JSON schema so the output is machine-consumable.

    5. RuleStore:  Bayesian-updated persistent rule set.
        - JSON is the source of truth (machine-readable).
        - dynamic_rules.md is REGENERATED from JSON on every update —
          never appended, never duplicated.
        - Rules with hit_rate < 0.5 AND samples >= 5 → archived.
        - Top-N active rules (by hit rate) live in the "Active Rules"
          section that the Fuser/AI prompt will pick up.
        - Capped file size; truly stale rules drop off.

    6. The Fuser and AI Engine read .kiro/steering/dynamic_rules.md as
       part of their prompt context (via the existing steering-file
       inclusion mechanism). Strategy evolves with every new event.

Anti-pattern guards:
    * No LLM-only invention: the LLM is constrained to pick from the
      pre-computed candidate features — it cannot conjure new ones.
    * No append-only sprawl: persistence is JSON; markdown is rendered.
    * Thresholds are quantized to a small set of buckets so similar
      observations dedupe to the same rule key (otherwise every event
      becomes a unique rule and the store explodes).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from altcoin_agent.ai_engine import DeepSeekEngine, EngineError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Domain types
# --------------------------------------------------------------------------- #


class EventSide(str, Enum):
    PUMP = "pump"
    DUMP = "dump"


@dataclass
class HistoricalSlice:
    """4h pre-event window of bound-checked market data."""

    symbol: str
    target_ts_ms: int
    start_ts_ms: int  # = target_ts_ms - 4h
    klines_1m: list[dict[str, Any]] = field(default_factory=list)   # OHLCV + trade_count
    funding_rates: list[dict[str, Any]] = field(default_factory=list)  # ts, rate
    open_interest: list[dict[str, Any]] = field(default_factory=list)  # ts, oi
    source: str = ""  # "okx" / "gate" / "test"

    def is_complete(self) -> bool:
        # We need at least 60 bars (1h of 1m) AND some funding to do
        # anything interesting. Anything less and we abort with a clear log.
        return len(self.klines_1m) >= 60 and len(self.funding_rates) >= 1


@dataclass
class CandidateFeature:
    """One quantitative feature observed in the pre-event window.

    `direction_hint` says whether this feature was POSITIVE (i.e.
    consistent with the realized event side) or NEGATIVE. Used by
    the post-mortem to filter for *prognostic* features only.
    """
    name: str
    value: float
    bucket: str          # quantized: e.g. "very_negative", "spike_5x", ...
    direction_hint: str  # "supports_pump" | "supports_dump" | "neutral"
    description: str


@dataclass
class EventResult:
    """The realized outcome derived from the slice tail."""
    side: EventSide
    magnitude_pct: float    # absolute price move from slice START to slice END
    minutes_to_extreme: int


# --------------------------------------------------------------------------- #
# Strict LLM output contract
# --------------------------------------------------------------------------- #


class _PostMortemPick(BaseModel):
    feature_name: str
    rank: int = Field(ge=1, le=2)
    rationale: str = Field(min_length=1, max_length=500)


class _PostMortemVerdict(BaseModel):
    """The schema we DEMAND from DeepSeek."""
    picks: list[_PostMortemPick]
    summary: str = Field(min_length=1, max_length=1000)

    @field_validator("picks")
    @classmethod
    def _at_most_two(cls, v: list[_PostMortemPick]) -> list[_PostMortemPick]:
        if not v:
            raise ValueError("must pick at least 1 feature")
        if len(v) > 2:
            raise ValueError("must pick at most 2 features")
        return v


# --------------------------------------------------------------------------- #
# 1. Historical Slicer — OKX public REST (no auth needed, low geo-risk)
# --------------------------------------------------------------------------- #


_OKX_BASE = "https://www.okx.com"


def _okx_inst_id(symbol: str) -> str:
    """Convert e.g. 'RAVE' or 'RAVEUSDT' -> 'RAVE-USDT-SWAP'."""
    s = symbol.upper().replace("USDT", "").rstrip("-_")
    return f"{s}-USDT-SWAP"


async def fetch_historical_slice(
    symbol: str,
    target_ts_ms: int,
    *,
    lookback_hours: float = 4.0,
    client: httpx.AsyncClient | None = None,
) -> HistoricalSlice:
    """Pull the pre-event window from OKX. Best-effort; returns a possibly
    partial slice with `is_complete()` flagging usability."""
    inst = _okx_inst_id(symbol)
    start_ms = target_ts_ms - int(lookback_hours * 3600 * 1000)
    snap = HistoricalSlice(
        symbol=symbol,
        target_ts_ms=target_ts_ms,
        start_ts_ms=start_ms,
        source="okx",
    )

    own_client = client is None
    cli = client or httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "altcoin-agent/0.1"},
    )
    try:
        # 1) 1m klines.  OKX history-candles takes (after, before, bar, limit)
        # in REVERSE chronological. We page until we cover the window.
        bars: list[list[Any]] = []
        cursor = target_ts_ms
        for _ in range(8):  # max 8 pages × 100 bars = 800 bars (~13 hours)
            url = (
                f"{_OKX_BASE}/api/v5/market/history-candles"
                f"?instId={inst}&bar=1m&limit=100&before={cursor}"
            )
            try:
                r = await cli.get(url)
                if r.status_code != 200:
                    break
                rows = r.json().get("data") or []
                if not rows:
                    break
                bars.extend(rows)
                # OKX returns newest-first; the LAST row is the oldest in this page
                oldest_in_page = int(rows[-1][0])
                if oldest_in_page <= start_ms:
                    break
                cursor = oldest_in_page
            except Exception as e:
                logger.warning("OKX history-candles error: %s", e)
                break

        # Normalize into ascending order, filter to the window.
        # OKX kline schema: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        # `vol` is in base ccy. `volCcyQuote` is USDT volume.
        for row in sorted(bars, key=lambda r: int(r[0])):
            ts = int(row[0])
            if ts < start_ms or ts > target_ts_ms:
                continue
            snap.klines_1m.append({
                "ts": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "volume_quote": float(row[7]) if len(row) > 7 else 0.0,
                # OKX doesn't expose trade_count on this endpoint;
                # leave as 0 (the wash-trading detector treats 0 as
                # "unknown" and will not produce false positives).
                "trade_count": 0,
            })

        # 2) Funding rate history (returns recent samples; we filter)
        try:
            r = await cli.get(
                f"{_OKX_BASE}/api/v5/public/funding-rate-history"
                f"?instId={inst}&limit=100"
            )
            if r.status_code == 200:
                for row in r.json().get("data") or []:
                    ts = int(row.get("fundingTime") or 0)
                    if start_ms <= ts <= target_ts_ms:
                        snap.funding_rates.append({
                            "ts": ts,
                            "rate": float(row.get("fundingRate") or 0),
                        })
                snap.funding_rates.sort(key=lambda r: r["ts"])
        except Exception as e:
            logger.warning("OKX funding-rate-history error: %s", e)

        # 3) Open interest history (5m granularity).  Period must be set.
        try:
            r = await cli.get(
                f"{_OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-history"
                f"?instId={inst}&period=5m&limit=100"
            )
            if r.status_code == 200:
                for row in r.json().get("data") or []:
                    ts = int(row[0])
                    if start_ms <= ts <= target_ts_ms:
                        snap.open_interest.append({
                            "ts": ts,
                            "oi": float(row[1]),
                            "oi_ccy": float(row[2]) if len(row) > 2 else 0.0,
                        })
                snap.open_interest.sort(key=lambda r: r["ts"])
        except Exception as e:
            logger.warning("OKX open-interest-history error: %s", e)

    finally:
        if own_client:
            await cli.aclose()

    return snap


# --------------------------------------------------------------------------- #
# 2. Feature Extractor — pure, deterministic
# --------------------------------------------------------------------------- #


def _bucket_funding(rate: float) -> str:
    if rate <= -0.0015:
        return "very_negative"     # < -0.15% / 8h
    if rate <= -0.0005:
        return "negative"
    if rate >= 0.0015:
        return "very_positive"
    if rate >= 0.0005:
        return "positive"
    return "neutral"


def _bucket_pct(p: float) -> str:
    """Bucketize a percentage move into coarse bands so similar
    observations cluster onto the same rule key."""
    a = abs(p)
    sign = "neg_" if p < 0 else "pos_"
    if a < 0.005:
        return "flat"
    if a < 0.02:
        return f"{sign}small"
    if a < 0.05:
        return f"{sign}medium"
    if a < 0.10:
        return f"{sign}large"
    return f"{sign}xlarge"


def _bucket_zscore(z: float) -> str:
    a = abs(z)
    sign = "neg_" if z < 0 else "pos_"
    if a < 1.0:
        return "calm"
    if a < 2.0:
        return f"{sign}elevated"
    if a < 4.0:
        return f"{sign}high"
    return f"{sign}extreme"


def _zscore(values: list[float], current: float) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (current - mean) / std


def extract_candidate_features(
    slc: HistoricalSlice, result: EventResult,
) -> list[CandidateFeature]:
    """Compute 8 grounded features the LLM gets to choose from.

    Each feature's `direction_hint` tells whether its bucket is
    consistent with the realized side. The LLM is then asked to
    pick the most prognostic ones from those that are CONSISTENT.
    """
    out: list[CandidateFeature] = []

    if not slc.klines_1m:
        return out

    bars = slc.klines_1m

    # The "event-edge" reference points
    end = slc.target_ts_ms
    pre_60 = end - 60 * 60 * 1000   # 1h before
    pre_120 = end - 120 * 60 * 1000  # 2h before

    bars_pre60 = [b for b in bars if b["ts"] <= pre_60]
    bars_pre120 = [b for b in bars if b["ts"] <= pre_120]
    bars_last_60 = [b for b in bars if b["ts"] > pre_60]

    side = result.side

    # --- F1: funding pre-2h extreme deviation -----------------------------
    if slc.funding_rates:
        rates_pre120 = [f["rate"] for f in slc.funding_rates if f["ts"] <= pre_120]
        [f["rate"] for f in slc.funding_rates if pre_120 < f["ts"] <= pre_60]
        if rates_pre120:
            extreme = min(rates_pre120) if side == EventSide.PUMP else max(rates_pre120)
            bucket = _bucket_funding(extreme)
            hint = "neutral"
            if side == EventSide.PUMP and extreme <= -0.0005:
                hint = "supports_pump"
            elif side == EventSide.DUMP and extreme >= 0.0005:
                hint = "supports_dump"
            out.append(CandidateFeature(
                name="funding_pre2h_extreme",
                value=extreme,
                bucket=bucket,
                direction_hint=hint,
                description=(
                    f"Funding rate extreme during the 2h-4h pre-event window: "
                    f"{extreme:.6f} ({bucket})"
                ),
            ))

    # --- F2: OI growth in the last hour ----------------------------------
    if len(slc.open_interest) >= 2:
        oi_start = slc.open_interest[0]["oi"]
        oi_pre60 = next(
            (o["oi"] for o in reversed(slc.open_interest) if o["ts"] <= pre_60),
            oi_start,
        )
        oi_growth = (oi_pre60 - oi_start) / oi_start if oi_start > 0 else 0
        bucket = _bucket_pct(oi_growth)
        hint = "neutral"
        if oi_growth > 0.10:
            # OI growth is non-directional alone; the price-coupling feature handles direction.
            hint = "supports_dump" if side == EventSide.DUMP else "supports_pump"
        out.append(CandidateFeature(
            name="oi_growth_pre1h",
            value=oi_growth,
            bucket=bucket,
            direction_hint=hint,
            description=(
                f"Open Interest grew {oi_growth*100:.2f}% from slice start to "
                f"1h pre-event ({bucket})."
            ),
        ))

        # --- F3: OI/price decoupling (silent build / silent distribution) -
        if bars_pre60:
            price_start = bars[0]["close"]
            price_pre60 = bars_pre60[-1]["close"]
            price_move = (price_pre60 - price_start) / price_start if price_start > 0 else 0
            decoupling = abs(oi_growth) - abs(price_move)
            hint = "neutral"
            # OI growth >> price move = silent positioning; consistent with
            # whichever side eventually fired.
            if oi_growth > 0.05 and abs(price_move) < 0.02:
                hint = "supports_dump" if side == EventSide.DUMP else "supports_pump"
            out.append(CandidateFeature(
                name="oi_price_decoupling",
                value=decoupling,
                bucket=_bucket_pct(decoupling),
                direction_hint=hint,
                description=(
                    f"OI grew {oi_growth*100:.2f}% while price moved "
                    f"{price_move*100:.2f}% — decoupling={decoupling*100:.2f}%."
                ),
            ))

    # --- F4: volume z-score in the last hour vs prior 3h baseline ---------
    if bars_pre120 and bars_last_60:
        baseline = [b["volume"] for b in bars_pre120]
        last_hr_vol = sum(b["volume"] for b in bars_last_60) / len(bars_last_60)
        z = _zscore(baseline, last_hr_vol)
        out.append(CandidateFeature(
            name="volume_zscore_last1h",
            value=z,
            bucket=_bucket_zscore(z),
            direction_hint=("supports_pump" if z > 1.5 and side == EventSide.PUMP
                            else "supports_dump" if z > 1.5 and side == EventSide.DUMP
                            else "neutral"),
            description=(
                f"Volume z-score in last hour vs 3h baseline = {z:.2f}."
            ),
        ))

    # --- F5: price compression then break (range squeezing) ---------------
    if bars_pre60:
        highs_pre60 = [b["high"] for b in bars_pre60]
        lows_pre60 = [b["low"] for b in bars_pre60]
        if highs_pre60 and lows_pre60:
            range_pre60 = (max(highs_pre60) - min(lows_pre60)) / min(lows_pre60)
            highs_pre120 = [b["high"] for b in bars_pre120] or highs_pre60
            lows_pre120 = [b["low"] for b in bars_pre120] or lows_pre60
            range_pre120 = (max(highs_pre120) - min(lows_pre120)) / min(lows_pre120)
            compression = range_pre120 - range_pre60  # positive = squeezing in
            out.append(CandidateFeature(
                name="range_compression_pre1h",
                value=compression,
                bucket=_bucket_pct(compression),
                direction_hint=(
                    "supports_pump" if compression > 0.01 and side == EventSide.PUMP
                    else "supports_dump" if compression > 0.01 and side == EventSide.DUMP
                    else "neutral"
                ),
                description=(
                    f"Range tightened from {range_pre120*100:.2f}% to "
                    f"{range_pre60*100:.2f}% in the last hour (squeeze before move)."
                ),
            ))

    # --- F6: wick:body asymmetry in last 30m ------------------------------
    last_30 = [b for b in bars_last_60 if b["ts"] > end - 30 * 60 * 1000]
    if last_30:
        upper_wicks = sum(
            b["high"] - max(b["open"], b["close"]) for b in last_30
        )
        lower_wicks = sum(
            min(b["open"], b["close"]) - b["low"] for b in last_30
        )
        bodies = sum(abs(b["close"] - b["open"]) for b in last_30) or 1e-9
        asym = (lower_wicks - upper_wicks) / bodies   # positive = lower-side rejection
        out.append(CandidateFeature(
            name="wick_asymmetry_last30m",
            value=asym,
            bucket=_bucket_pct(asym),
            direction_hint=(
                "supports_pump" if asym > 0.5 and side == EventSide.PUMP
                else "supports_dump" if asym < -0.5 and side == EventSide.DUMP
                else "neutral"
            ),
            description=(
                f"Last 30m wick asymmetry (lower − upper) / body = {asym:.2f} "
                "(>0 = bottom rejection; <0 = top rejection)."
            ),
        ))

    # --- F7: funding rate trend slope in last hour ------------------------
    if slc.funding_rates and len(slc.funding_rates) >= 3:
        recent = [f for f in slc.funding_rates if f["ts"] > pre_60]
        if len(recent) >= 2:
            slope = (recent[-1]["rate"] - recent[0]["rate"])
            hint = "neutral"
            if slope < -0.0005 and side == EventSide.PUMP:
                hint = "supports_pump"
            elif slope > 0.0005 and side == EventSide.DUMP:
                hint = "supports_dump"
            out.append(CandidateFeature(
                name="funding_slope_last1h",
                value=slope,
                bucket=_bucket_funding(slope),
                direction_hint=hint,
                description=f"Funding rate slope over last hour = {slope:.6f}.",
            ))

    # --- F8: late-window distribution wick (top reject before dump) -------
    # Specifically: did the upper wicks in the last 60m dominate, even before
    # the dump candle? (Useful for dump prediction.)
    if bars_last_60:
        upper_total = sum(
            b["high"] - max(b["open"], b["close"]) for b in bars_last_60
        )
        lower_total = sum(
            min(b["open"], b["close"]) - b["low"] for b in bars_last_60
        )
        upper_dom = (
            (upper_total - lower_total) / max(upper_total + lower_total, 1e-9)
        )
        out.append(CandidateFeature(
            name="upper_wick_dominance_last1h",
            value=upper_dom,
            bucket=_bucket_pct(upper_dom),
            direction_hint=(
                "supports_dump" if upper_dom > 0.3 and side == EventSide.DUMP
                else "supports_pump" if upper_dom < -0.3 and side == EventSide.PUMP
                else "neutral"
            ),
            description=(
                f"Upper-wick dominance ratio over last 1h = {upper_dom:.2f} "
                "(distributive top wicks vs accumulative bottom wicks)."
            ),
        ))

    return out


# --------------------------------------------------------------------------- #
# 3. Result Computer
# --------------------------------------------------------------------------- #


def compute_event_result(slc: HistoricalSlice) -> EventResult | None:
    """Derive the realized event from the slice's price extreme.

    "Magnitude" = max absolute % move from the slice OPEN to the
    extreme reached at or before the target_ts. The side is the sign
    of the dominant move.
    """
    if not slc.klines_1m:
        return None
    open_price = slc.klines_1m[0]["open"]
    if open_price <= 0:
        return None

    ext_high = max((b["high"] for b in slc.klines_1m), default=open_price)
    ext_low = min((b["low"] for b in slc.klines_1m), default=open_price)

    up = (ext_high - open_price) / open_price
    down = (open_price - ext_low) / open_price

    if up > down:
        side = EventSide.PUMP
        magnitude = up
        # find the bar that hit the high
        bar = next((b for b in slc.klines_1m if b["high"] >= ext_high), slc.klines_1m[-1])
    else:
        side = EventSide.DUMP
        magnitude = down
        bar = next((b for b in slc.klines_1m if b["low"] <= ext_low), slc.klines_1m[-1])

    minutes_to_extreme = max(0, (bar["ts"] - slc.klines_1m[0]["ts"]) // 60_000)
    return EventResult(
        side=side, magnitude_pct=round(magnitude, 4),
        minutes_to_extreme=int(minutes_to_extreme),
    )


# --------------------------------------------------------------------------- #
# 4. DeepSeek Post-Mortem
# --------------------------------------------------------------------------- #


_POSTMORTEM_SYSTEM_PROMPT = """You are a post-mortem analyst for a crypto altcoin
trading bot. You receive:

  1. A REALIZED EVENT (the coin pumped or dumped by some magnitude).
  2. A list of CANDIDATE FEATURES that were objectively measured in the
     pre-event window. Each feature comes with a name, value, bucket,
     direction_hint, and human description.

Your task: pick AT MOST 2 features that were MOST PROGNOSTIC of the realized
event. "Prognostic" means: present in the pre-event window AND consistent
with the eventual direction AND specific enough to be useful as a forward
trading rule (not a tautology like "price moved").

Hard rules:
  * You may ONLY pick from the candidates provided. Do NOT invent features.
  * Prefer features whose direction_hint matches the realized side.
  * Reject features with direction_hint="neutral" unless their numeric
    value is extreme.
  * Reject features whose appearance is clearly post-hoc (e.g. a feature
    measured AT the event time rather than BEFORE it).

You MUST respond with ONE single JSON object and NO surrounding text:

{
  "picks": [
    {
      "feature_name": "<exact name from candidates>",
      "rank": <1 for most prognostic, 2 for second>,
      "rationale": "<why this is prognostic, <= 300 chars>"
    },
    ...
  ],
  "summary": "<a 1-2 sentence narrative of the setup, <= 300 chars>"
}

DO NOT output markdown, code fences, or any text outside the JSON.
"""


async def post_mortem_via_deepseek(
    *,
    engine: DeepSeekEngine,
    symbol: str,
    result: EventResult,
    features: list[CandidateFeature],
) -> _PostMortemVerdict | None:
    """Ask DeepSeek to pick 1-2 prognostic features. Returns None on
    any failure (the caller falls back to top-2-by-direction-hint)."""
    if not features:
        return None

    payload = {
        "symbol": symbol,
        "event": {
            "side": result.side.value,
            "magnitude_pct": result.magnitude_pct,
            "minutes_to_extreme": result.minutes_to_extreme,
        },
        "candidate_features": [asdict(f) for f in features],
    }
    user_msg = (
        "POST-MORTEM CONTEXT (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the JSON verdict now."
    )
    messages = [
        {"role": "system", "content": _POSTMORTEM_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw, _used = await engine._call_api(messages)
    except EngineError:
        raise
    except Exception as e:
        logger.warning("DeepSeek post-mortem call failed: %s", e)
        return None
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        obj = json.loads(text)
        return _PostMortemVerdict.model_validate(obj)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("DeepSeek post-mortem returned bad JSON: %s", e)
        return None


def _fallback_pick(features: list[CandidateFeature]) -> _PostMortemVerdict:
    """If DeepSeek is unavailable, pick by `direction_hint` + bucket extremity."""
    rank_score = {"calm": 0, "flat": 0, "neutral": 0,
                  "elevated": 1, "small": 1, "medium": 1,
                  "high": 2, "large": 2, "negative": 1, "positive": 1,
                  "very_negative": 3, "very_positive": 3,
                  "extreme": 3, "xlarge": 3}

    def score(f: CandidateFeature) -> int:
        if f.direction_hint == "neutral":
            return 0
        # bucket may be e.g. "neg_high" → look at the suffix
        b = f.bucket.split("_")[-1]
        return rank_score.get(b, 1) + (1 if f.direction_hint != "neutral" else 0)

    ranked = sorted(features, key=score, reverse=True)
    picks = []
    for i, f in enumerate(ranked[:2], 1):
        if score(f) == 0:
            break
        picks.append(_PostMortemPick(
            feature_name=f.name,
            rank=i,
            rationale=f"Fallback: bucket={f.bucket}, hint={f.direction_hint}",
        ))
    if not picks and ranked:
        picks.append(_PostMortemPick(
            feature_name=ranked[0].name,
            rank=1,
            rationale=f"Fallback: best available (bucket={ranked[0].bucket})",
        ))
    return _PostMortemVerdict(
        picks=picks,
        summary="LLM unavailable; rule chosen by quantitative fallback.",
    )


# --------------------------------------------------------------------------- #
# 5. Persistent rule store with Bayesian update
# --------------------------------------------------------------------------- #


@dataclass
class DynamicRule:
    """One learned rule. Keyed by (feature_name, bucket, side)."""
    feature_name: str
    bucket: str
    side: EventSide
    hits: int = 0          # times we observed this AND the event matched
    total: int = 0         # times we observed this regardless of outcome
    last_seen_iso: str = ""
    last_summary: str = ""

    @property
    def rule_id(self) -> str:
        return f"{self.feature_name}|{self.bucket}|{self.side.value}"

    @property
    def hit_rate(self) -> float:
        # Laplace smoothing (alpha=beta=1).
        return (self.hits + 1) / (self.total + 2)

    @property
    def confidence_band(self) -> str:
        if self.total < 3:
            return "early"
        if self.hit_rate >= 0.75:
            return "strong"
        if self.hit_rate >= 0.6:
            return "decent"
        if self.hit_rate >= 0.5:
            return "marginal"
        return "weak"


@dataclass
class RuleStore:
    """JSON-backed source of truth + regenerated markdown view.

    Files:
        json_path:  source of truth (machine-readable)
        md_path:    rendered for LLM/human consumption (regenerated each save)
    """
    json_path: Path
    md_path: Path
    max_active_rules: int = 30
    archive_threshold_samples: int = 5
    archive_threshold_hit_rate: float = 0.5
    rules: dict[str, DynamicRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        try:
            data = json.loads(self.json_path.read_text())
        except Exception as e:
            logger.error("rule_store: failed to load %s: %s", self.json_path, e)
            return
        for d in data.get("rules", []):
            try:
                rule = DynamicRule(
                    feature_name=d["feature_name"],
                    bucket=d["bucket"],
                    side=EventSide(d["side"]),
                    hits=int(d.get("hits", 0)),
                    total=int(d.get("total", 0)),
                    last_seen_iso=d.get("last_seen_iso", ""),
                    last_summary=d.get("last_summary", ""),
                )
                self.rules[rule.rule_id] = rule
            except Exception as e:
                logger.warning("rule_store: skipping malformed rule %s: %s", d, e)

    def update_with_picks(
        self,
        verdict: _PostMortemVerdict,
        features: list[CandidateFeature],
        result: EventResult,
    ) -> list[DynamicRule]:
        """Apply Bayesian-style update for the picked features.

        - For each PICK: increment (hits, total) — these were ahead-of-event
          observations consistent with the realized side (the LLM only
          picks ones that were).
        - For each NON-PICKED candidate that ALSO had a supportive
          direction_hint: increment total only (i.e., the feature was
          present but NOT chosen as primary), so its hit-rate is
          gradually penalized.

        This implements an honest competition between candidate features
        without over-counting.
        """
        feature_by_name = {f.name: f for f in features}
        picked_names = {p.feature_name for p in verdict.picks}
        ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        touched: list[DynamicRule] = []

        # Picked → hits + total
        for pick in verdict.picks:
            f = feature_by_name.get(pick.feature_name)
            if f is None:
                continue
            rule = self._upsert(f, result.side, ts_iso, verdict.summary)
            rule.hits += 1
            rule.total += 1
            touched.append(rule)

        # Non-picked but supportive → only total (penalty when not picked)
        for f in features:
            if f.name in picked_names:
                continue
            if f.direction_hint != f"supports_{result.side.value}":
                continue
            rule = self._upsert(f, result.side, ts_iso, verdict.summary)
            rule.total += 1
            touched.append(rule)

        self._save()
        return touched

    def _upsert(
        self, feature: CandidateFeature, side: EventSide,
        ts_iso: str, summary: str,
    ) -> DynamicRule:
        # Quantize into the bucket so similar observations dedupe
        rid_key = f"{feature.name}|{feature.bucket}|{side.value}"
        if rid_key not in self.rules:
            self.rules[rid_key] = DynamicRule(
                feature_name=feature.name,
                bucket=feature.bucket,
                side=side,
            )
        rule = self.rules[rid_key]
        rule.last_seen_iso = ts_iso
        rule.last_summary = summary
        return rule

    def active_and_archived(self) -> tuple[list[DynamicRule], list[DynamicRule]]:
        active: list[DynamicRule] = []
        archived: list[DynamicRule] = []
        for r in self.rules.values():
            is_archived = (
                r.total >= self.archive_threshold_samples
                and r.hit_rate < self.archive_threshold_hit_rate
            )
            if is_archived:
                archived.append(r)
            else:
                active.append(r)
        active.sort(key=lambda r: (-r.hit_rate, -r.total))
        archived.sort(key=lambda r: (r.hit_rate, -r.total))
        return active[: self.max_active_rules], archived

    def _save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rules": [
                {
                    "feature_name": r.feature_name,
                    "bucket": r.bucket,
                    "side": r.side.value,
                    "hits": r.hits,
                    "total": r.total,
                    "last_seen_iso": r.last_seen_iso,
                    "last_summary": r.last_summary,
                }
                for r in self.rules.values()
            ],
        }
        self.json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        active, archived = self.active_and_archived()
        self.md_path.write_text(self._render_markdown(active, archived))

    @staticmethod
    def _render_markdown(
        active: list[DynamicRule], archived: list[DynamicRule],
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: list[str] = []
        lines.append("---")
        lines.append("inclusion: always")
        lines.append("description: Auto-generated trading rules learned from "
                     "post-mortem analysis. DO NOT EDIT BY HAND — regenerated "
                     "by learning_engine.py on every event.")
        lines.append("---")
        lines.append("")
        lines.append("# Dynamic Trading Rules (Learned)")
        lines.append("")
        lines.append(f"_Last updated: {ts}_")
        lines.append("")
        lines.append("These rules were derived by post-mortem analysis of real "
                     "altcoin pump/dump events. The fuser and AI engine read "
                     "this file as additional steering context. Each rule:")
        lines.append("")
        lines.append("- Is keyed by `(feature, bucket, side)`.")
        lines.append("- Has `hit_rate = (hits + 1) / (total + 2)` "
                     "(Laplace smoothing).")
        lines.append("- Auto-archives when `total >= 5 AND hit_rate < 0.5`.")
        lines.append("")
        lines.append("## Active Rules (sorted by hit_rate)")
        lines.append("")
        if not active:
            lines.append("_No rules learned yet._")
        else:
            lines.append("| Rank | Feature | Bucket | Side | Hits/Total | "
                         "Hit Rate | Confidence | Last Seen |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for i, r in enumerate(active, 1):
                lines.append(
                    f"| {i} | `{r.feature_name}` | `{r.bucket}` | "
                    f"**{r.side.value}** | {r.hits}/{r.total} | "
                    f"{r.hit_rate:.2%} | {r.confidence_band} | "
                    f"{r.last_seen_iso or '—'} |"
                )

        # Most-recent narrative summary
        if active:
            last = max(active, key=lambda r: r.last_seen_iso or "")
            if last.last_summary:
                lines.append("")
                lines.append("### Most recent post-mortem narrative")
                lines.append(f"> {last.last_summary}")

        lines.append("")
        lines.append("## Archived (low-confidence rules pruned)")
        lines.append("")
        if not archived:
            lines.append("_None._")
        else:
            lines.append("| Feature | Bucket | Side | Hits/Total | Hit Rate |")
            lines.append("|---|---|---|---|---|")
            for r in archived:
                lines.append(
                    f"| `{r.feature_name}` | `{r.bucket}` | "
                    f"{r.side.value} | {r.hits}/{r.total} | "
                    f"{r.hit_rate:.2%} |"
                )

        lines.append("")
        lines.append("## Usage")
        lines.append("")
        lines.append("- The score fuser can up-weight rule events whose "
                     "`(feature, bucket, side)` match an active rule with "
                     "`hit_rate >= 0.6`.")
        lines.append("- The AI engine's prompt context should include the "
                     "Active Rules table so DeepSeek can ground its verdict in "
                     "the agent's own historical experience.")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 6. Top-level orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class LearningRun:
    """Container for everything produced by one post-mortem cycle."""
    symbol: str
    target_ts_ms: int
    slice: HistoricalSlice
    result: EventResult | None
    features: list[CandidateFeature]
    verdict: _PostMortemVerdict | None
    used_fallback: bool
    touched_rules: list[DynamicRule]
    md_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_ts_ms": self.target_ts_ms,
            "result": (
                {"side": self.result.side.value,
                 "magnitude_pct": self.result.magnitude_pct,
                 "minutes_to_extreme": self.result.minutes_to_extreme}
                if self.result else None
            ),
            "feature_count": len(self.features),
            "picked_features": (
                [p.feature_name for p in self.verdict.picks]
                if self.verdict else []
            ),
            "used_fallback": self.used_fallback,
            "touched_rules": [r.rule_id for r in self.touched_rules],
            "md_path": str(self.md_path),
        }


async def run_post_mortem(
    *,
    symbol: str,
    target_ts_ms: int,
    engine: DeepSeekEngine | None = None,
    rule_store: RuleStore | None = None,
    slice_override: HistoricalSlice | None = None,
    lookback_hours: float = 4.0,
) -> LearningRun:
    """End-to-end. `slice_override` lets tests/demos inject synthetic data."""

    # 1) slice
    if slice_override is not None:
        slc = slice_override
    else:
        slc = await fetch_historical_slice(
            symbol, target_ts_ms, lookback_hours=lookback_hours,
        )
    if not slc.is_complete():
        logger.warning(
            "Incomplete slice for %s @ %s (kbars=%d, funding=%d, oi=%d). "
            "Proceeding best-effort.",
            symbol, target_ts_ms, len(slc.klines_1m),
            len(slc.funding_rates), len(slc.open_interest),
        )

    # 2) result + features
    result = compute_event_result(slc)
    if result is None:
        logger.error("Cannot compute event result; returning empty run.")
        return LearningRun(
            symbol=symbol, target_ts_ms=target_ts_ms, slice=slc,
            result=None, features=[], verdict=None, used_fallback=True,
            touched_rules=[], md_path=Path(),
        )
    features = extract_candidate_features(slc, result)

    # 3) LLM post-mortem (with fallback)
    verdict: _PostMortemVerdict | None = None
    used_fallback = False
    if engine is not None and engine.api_key:
        try:
            verdict = await post_mortem_via_deepseek(
                engine=engine, symbol=symbol, result=result, features=features,
            )
        except EngineError as e:
            logger.warning("DeepSeek refused (likely budget): %s", e)
    if verdict is None:
        verdict = _fallback_pick(features)
        used_fallback = True

    # 4) persist
    if rule_store is None:
        # Default location inside the workspace's .kiro/steering/
        # Walk up from this file to find the project root. Pathlib here is
        # cheap and synchronous; no need for trio/anyio path wrappers.
        here = Path(__file__).resolve()  # noqa: ASYNC240
        project_root = here.parents[2]   # src/altcoin_agent/learning_engine.py → project root
        rule_store = RuleStore(
            json_path=project_root / ".kiro" / "steering" / "dynamic_rules.json",
            md_path=project_root / ".kiro" / "steering" / "dynamic_rules.md",
        )
    touched = rule_store.update_with_picks(verdict, features, result)

    return LearningRun(
        symbol=symbol, target_ts_ms=target_ts_ms, slice=slc, result=result,
        features=features, verdict=verdict, used_fallback=used_fallback,
        touched_rules=touched, md_path=rule_store.md_path,
    )


# --------------------------------------------------------------------------- #
# Convenience for tests
# --------------------------------------------------------------------------- #


def synthesize_dump_slice(
    symbol: str = "RAVEUSDT",
    target_ts_ms: int | None = None,
) -> HistoricalSlice:
    """Build a deterministic 4h pre-DUMP fixture.

    The setup encodes textbook distribution-then-dump:
        * 0h–2h: price compresses near the highs (~$1.00)
        * 2h–3h: OI grows ~+22% while price stays flat (silent build by shorts)
        * 3h–3:50h: positive funding rate, upper-wick rejection
        * Final 10m: waterfall dump to ~$0.78 (-22%)
    """
    if target_ts_ms is None:
        target_ts_ms = 1_700_000_000_000
    start_ms = target_ts_ms - 4 * 3600 * 1000

    klines: list[dict[str, Any]] = []
    base_price = 1.000
    # 240 1-minute bars
    for i in range(240):
        ts = start_ms + i * 60_000
        if i < 180:
            # compression around 1.00 ± 0.5%, slight drift up
            o = base_price + (i % 7) * 0.0005
            c = o + (-1) ** i * 0.001
            h = max(o, c) + 0.003
            lo = min(o, c) - 0.001       # small lower wicks
            vol = 100 + (i % 11) * 2
        elif i < 230:
            # upper-wick rejection band
            o = 1.005 + (i % 5) * 0.0006
            c = o - 0.0008
            h = o + 0.012                # heavy upper wicks
            lo = o - 0.001
            vol = 130 + (i % 11) * 4
        else:
            # the dump candles (last 10 minutes)
            decay = (i - 229) * 0.025
            o = 1.000 - decay * 0.8
            c = 1.000 - decay
            h = o + 0.001
            lo = c - 0.005
            vol = 800 + (i - 229) * 200
        klines.append({
            "ts": ts, "open": o, "high": h, "low": lo, "close": c,
            "volume": vol, "volume_quote": vol * o, "trade_count": 0,
        })

    funding = []
    # Funding rate has been climbing positive over the 4h
    for i in range(8):
        ts = start_ms + (i * 30 + 60) * 60_000   # samples every 30m starting at +1h
        rate = 0.0002 + i * 0.00018             # 0.02% → 0.15%
        funding.append({"ts": ts, "rate": round(rate, 6)})

    oi = []
    # OI grew ~22% — driven by shorts piling in despite flat price
    for i in range(48):  # 5m granularity = 48 samples / 4h
        ts = start_ms + i * 5 * 60_000
        # roughly: 1,000,000 → 1,220,000 over 4h
        v = 1_000_000.0 * (1 + 0.22 * (i / 47))
        oi.append({"ts": ts, "oi": v, "oi_ccy": v * 1.0})

    return HistoricalSlice(
        symbol=symbol, target_ts_ms=target_ts_ms, start_ts_ms=start_ms,
        klines_1m=klines, funding_rates=funding, open_interest=oi,
        source="test",
    )
