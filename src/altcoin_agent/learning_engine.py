"""
learning_engine.py — Strategy Self-Evolution Engine.

Purpose
-------
Take a confirmed pump/dump event after the fact, slice the 4 hours of market
data leading up to it, extract 8 quantitative candidate features, and let
DeepSeek pick the 1-2 most-prognostic features. Persist their hit/total counts
in `dynamic_rules.json` (machine source of truth) and regenerate
`dynamic_rules.md` for human/LLM context inclusion.

Architectural decisions (committed with the user):

  1. The MARKDOWN file is REGENERATED on every save, never appended. The JSON
     file is the canonical store. This prevents unbounded growth and avoids
     duplicate / contradictory entries piling up.

  2. The LLM cannot invent features. We pre-compute 8 quantitative buckets and
     the LLM may only return their feature_name+bucket. This kills the most
     common LLM failure mode: hindsight-narrative features that don't exist.

  3. Hit counts use LAPLACE SMOOTHING:  hit_rate = (hits + 1) / (total + 2).
     A brand-new 1/1 rule scores 66.7%, not 100%, so the fuser's confidence
     shrinkage doesn't get fooled into immediate full multiplier.

  4. Buckets are quantized so similar events aggregate. Without quantization
     every event would create a unique rule_id and the store would explode.

The bucket helpers here MUST match the helpers `fuser.py::signal_to_learned_keys`
uses for live lookup, otherwise the live signal won't find its learned rule.
Both modules import the same `_bucket_*` functions defined in this file.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from altcoin_agent.ai_engine import DeepSeekEngine, EngineError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Shared bucket helpers — also imported by fuser.py
# --------------------------------------------------------------------- #


def bucket_pct(p: float) -> str:
    """Bucket a fractional move (e.g. 0.05 = +5%) into a coarse label."""
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


def bucket_zscore(z: float) -> str:
    a = abs(z)
    sign = "neg_" if z < 0 else "pos_"
    if a < 1.0:
        return "calm"
    if a < 2.0:
        return f"{sign}elevated"
    if a < 4.0:
        return f"{sign}high"
    return f"{sign}extreme"


def bucket_funding(rate: float) -> str:
    """Bucket a per-interval funding rate."""
    if rate <= -0.0015:
        return "very_negative"
    if rate <= -0.0005:
        return "negative"
    if rate >= 0.0015:
        return "very_positive"
    if rate >= 0.0005:
        return "positive"
    return "neutral"


# --------------------------------------------------------------------- #
# Domain types
# --------------------------------------------------------------------- #


Direction = Literal["pump", "dump"]


@dataclass(frozen=True)
class Bar:
    """Minimal OHLCV+trades+OI+funding row for a fixed cadence (1m)."""

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class HistoricalSlice:
    """Market data window around a target event.

    Two layouts are supported:

    * **Backward-only (legacy):**  `[target_ts_ms - hours_back, target_ts_ms]`.
      Used by `discover_events` / standalone backtests where the "event" is
      the whole 4h window. ``entry_ts_ms`` is 0; ``compute_event_result`` and
      ``extract_candidate_features`` then operate on the entire bar list.
    * **Around an entry (Bug #2 fix):**  `[entry_ts_ms - hours_back,
      entry_ts_ms + hours_forward]`. Used by the live post-mortem scheduler
      and any backtest that wants to evaluate what happened *after* a given
      decision point. When ``entry_ts_ms > 0``:
        - ``extract_candidate_features`` uses bars strictly before
          ``entry_ts_ms`` (the predictors).
        - ``compute_event_result`` evaluates the realized direction/magnitude
          using only bars at or after ``entry_ts_ms``, with the close of the
          last pre-entry bar as the reference price (the most accurate
          available proxy for our actual fill).

    ``target_ts_ms`` is kept as the *anchor* timestamp for both layouts; it's
    the field the slice cache keys on.
    """

    symbol: str
    target_ts_ms: int
    bars: list[Bar] = field(default_factory=list)        # oldest -> newest
    funding_rates: list[tuple[int, float]] = field(default_factory=list)  # (ts, rate)
    open_interest: list[tuple[int, float]] = field(default_factory=list)  # (ts, oi)
    # Bug #2 fix: when set, the slice spans both before and after this
    # timestamp. compute_event_result and extract_candidate_features split
    # the bars on this boundary so the realized result is measured AFTER
    # entry and the predictive features are measured BEFORE entry. When 0,
    # the legacy whole-slice behaviour is used.
    entry_ts_ms: int = 0

    @property
    def first_close(self) -> float:
        return self.bars[0].close if self.bars else 0.0

    @property
    def last_close(self) -> float:
        return self.bars[-1].close if self.bars else 0.0

    def pre_entry_bars(self) -> list[Bar]:
        """Bars strictly before ``entry_ts_ms``; the predictive window."""
        if self.entry_ts_ms <= 0:
            return list(self.bars)
        return [b for b in self.bars if b.ts_ms < self.entry_ts_ms]

    def post_entry_bars(self) -> list[Bar]:
        """Bars at or after ``entry_ts_ms``; the result-evaluation window."""
        if self.entry_ts_ms <= 0:
            return list(self.bars)
        return [b for b in self.bars if b.ts_ms >= self.entry_ts_ms]

    def entry_reference_price(self) -> float:
        """Close of the last bar before ``entry_ts_ms`` — the best proxy
        for our actual fill price. Falls back to first post-entry open."""
        if self.entry_ts_ms <= 0:
            return self.first_close
        pre = self.pre_entry_bars()
        if pre:
            return pre[-1].close
        post = self.post_entry_bars()
        return post[0].open if post else 0.0


@dataclass
class EventResult:
    direction: Direction
    magnitude_pct: float           # signed, e.g. -0.255 for a 25.5% dump
    minutes_to_extremum: int
    realized_at_ts_ms: int


@dataclass
class CandidateFeature:
    """One bucketed feature value extracted from a slice."""

    name: str
    bucket: str
    raw_value: float
    description: str


# --------------------------------------------------------------------- #
# Slice fetching (OKX REST — no login required)
# --------------------------------------------------------------------- #


OKX_REST = "https://www.okx.com"


async def fetch_historical_slice(
    symbol: str,
    target_ts_ms: int,
    *,
    hours_back: int = 4,
    hours_forward: int = 0,
    entry_ts_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> HistoricalSlice:
    """
    Pull a market-data slice for ``symbol`` via OKX public REST. Symbol is
    normalized to OKX swap form (e.g. RAVE-USDT-SWAP).

    Two modes:

    * **Legacy** (``hours_forward == 0`` and ``entry_ts_ms`` unset): pulls
      ``[target_ts_ms - hours_back, target_ts_ms]``, the original 4h-leading
      window used by ``discover_events`` and standalone backtests.
    * **Entry-aware** (Bug #2 fix): when ``entry_ts_ms`` is provided, pulls
      ``[entry_ts_ms - hours_back, entry_ts_ms + hours_forward]`` and tags
      the resulting ``HistoricalSlice.entry_ts_ms``. Downstream
      ``compute_event_result`` / ``extract_candidate_features`` then split
      bars on that boundary so realized outcomes are measured strictly
      AFTER entry and predictive features strictly BEFORE.
    """
    inst_id = _to_okx_inst_id(symbol)

    anchor_ms = entry_ts_ms if entry_ts_ms is not None else target_ts_ms
    start_ms = anchor_ms - hours_back * 3600 * 1000
    end_ms = anchor_ms + hours_forward * 3600 * 1000
    if end_ms < target_ts_ms:
        # Caller may have set both target_ts_ms and entry_ts_ms; the network
        # window must include both so the slice cache key remains valid.
        end_ms = target_ts_ms

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=15.0)

    try:
        bars = await _fetch_okx_candles(client, inst_id, start_ms, end_ms)
        funding = await _fetch_okx_funding(client, inst_id, start_ms, end_ms)
        oi = await _fetch_okx_oi(client, inst_id, start_ms, end_ms)
    finally:
        if owns_client:
            await client.aclose()

    return HistoricalSlice(
        symbol=symbol,
        target_ts_ms=target_ts_ms,
        bars=bars,
        funding_rates=funding,
        open_interest=oi,
        entry_ts_ms=entry_ts_ms or 0,
    )


def _to_okx_inst_id(symbol: str) -> str:
    s = symbol.upper().replace("/", "").replace(":USDT", "")
    if s.endswith("USDT") and "-" not in s:
        return f"{s[:-4]}-USDT-SWAP"
    return s


async def _fetch_okx_candles(
    client: httpx.AsyncClient, inst_id: str, start_ms: int, end_ms: int,
) -> list[Bar]:
    bars: list[Bar] = []
    cursor = end_ms
    # OKX returns up to 100 per page, newest first.
    while cursor > start_ms:
        params = {"instId": inst_id, "bar": "1m", "after": str(cursor), "limit": "100"}
        r = await client.get(f"{OKX_REST}/api/v5/market/candles", params=params)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            if ts < start_ms:
                continue
            bars.append(Bar(ts, float(row[1]), float(row[2]), float(row[3]),
                            float(row[4]), float(row[5])))
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor = oldest_ts
    bars.sort(key=lambda b: b.ts_ms)
    return [b for b in bars if start_ms <= b.ts_ms <= end_ms]


async def _fetch_okx_funding(
    client: httpx.AsyncClient, inst_id: str, start_ms: int, end_ms: int,
) -> list[tuple[int, float]]:
    try:
        r = await client.get(
            f"{OKX_REST}/api/v5/public/funding-rate-history",
            params={"instId": inst_id, "before": str(start_ms), "after": str(end_ms),
                    "limit": "100"},
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        return sorted(
            [(int(row["fundingTime"]), float(row["fundingRate"])) for row in rows],
            key=lambda x: x[0],
        )
    except Exception as e:
        logger.warning("funding history fetch failed: %s", e)
        return []


async def _fetch_okx_oi(
    client: httpx.AsyncClient, inst_id: str, start_ms: int, end_ms: int,
) -> list[tuple[int, float]]:
    try:
        r = await client.get(
            f"{OKX_REST}/api/v5/rubik/stat/contracts/open-interest-volume",
            params={"ccy": inst_id.split("-")[0], "begin": str(start_ms),
                    "end": str(end_ms), "period": "1m"},
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        return sorted([(int(row[0]), float(row[1])) for row in rows], key=lambda x: x[0])
    except Exception as e:
        logger.warning("oi history fetch failed: %s", e)
        return []


# --------------------------------------------------------------------- #
# Result computation
# --------------------------------------------------------------------- #


def compute_event_result(
    s: HistoricalSlice,
    *,
    expected_direction: Direction | None = None,
) -> EventResult:
    """Derive realized direction + magnitude from the slice.

    Entry-aware (Bug #2 fix): when ``s.entry_ts_ms > 0`` the realized result
    is computed strictly from bars at or after entry, with the close of the
    last pre-entry bar as the reference price (the best proxy for our actual
    fill). This guarantees we evaluate "what happened after we opened",
    not "what was the biggest move anywhere in the lookback window".

    Legacy (when ``s.entry_ts_ms == 0``): the whole slice is scanned with the
    first close as reference, preserving the behaviour expected by
    ``discover_events`` / standalone backtests.

    ``expected_direction`` (Bug #2 fix): if provided, the result is reported
    AS IF the trader took that direction, so a long that gets stopped out
    correctly registers as ``"pump"`` with a *negative* magnitude (a missed
    pump = a loss). This is what lets the rule store learn from losers as
    well as winners. When omitted, the larger-extremum-wins behaviour is
    preserved for backward compatibility.
    """
    bars = s.post_entry_bars() if s.entry_ts_ms > 0 else list(s.bars)
    if not bars:
        return EventResult(
            expected_direction or "pump", 0.0, 0,
            s.entry_ts_ms or s.target_ts_ms,
        )

    ref = s.entry_reference_price() if s.entry_ts_ms > 0 else bars[0].close
    if ref <= 0:
        return EventResult(
            expected_direction or "pump", 0.0, 0,
            bars[0].ts_ms,
        )

    max_up = 0.0
    max_dn = 0.0
    up_bar: Bar | None = None
    dn_bar: Bar | None = None
    for b in bars:
        up = (b.high - ref) / ref
        dn = (b.low - ref) / ref
        if up > max_up:
            max_up = up
            up_bar = b
        if dn < max_dn:
            max_dn = dn
            dn_bar = b

    if expected_direction == "pump":
        # Trader's thesis was UP. Magnitude is the run UP if it materialized,
        # else the (negative) drawdown — the move that took us out.
        if max_up > 0:
            target_bar = up_bar or bars[-1]
            mag = max_up
        else:
            target_bar = dn_bar or bars[-1]
            mag = max_dn
        direction: Direction = "pump"
    elif expected_direction == "dump":
        # Trader's thesis was DOWN. Magnitude is the run DOWN if it
        # materialized, else the (positive) drawdown.
        if max_dn < 0:
            target_bar = dn_bar or bars[-1]
            mag = max_dn
        else:
            target_bar = up_bar or bars[-1]
            mag = max_up
        direction = "dump"
    else:
        # Legacy: whichever extremum is bigger wins.
        if abs(max_up) >= abs(max_dn):
            target_bar = up_bar or bars[-1]
            mag = max_up
            direction = "pump"
        else:
            target_bar = dn_bar or bars[-1]
            mag = max_dn
            direction = "dump"

    minutes = max(0, (target_bar.ts_ms - bars[0].ts_ms) // 60_000)
    return EventResult(
        direction=direction,
        magnitude_pct=mag,
        minutes_to_extremum=minutes,
        realized_at_ts_ms=target_bar.ts_ms,
    )


# --------------------------------------------------------------------- #
# Feature extraction (8 candidates)
# --------------------------------------------------------------------- #


def _zscore(values: list[float], x: float) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (x - mean) / std


def extract_candidate_features(s: HistoricalSlice, result: EventResult) -> list[CandidateFeature]:
    """Compute 8 quantitative candidate features over the PRE-event window.

    Two windowing modes:

    * **Entry-aware** (``s.entry_ts_ms > 0``, the live post-mortem path
      after the Bug #2 fix): the predictive window is the bars strictly
      before ``entry_ts_ms``. This is the only window the LLM is allowed to
      see when deciding what predicted the post-entry move — it cannot peek
      at bars from after we opened.
    * **Legacy** (``s.entry_ts_ms == 0``): the predictive window is
      everything before ``result.realized_at_ts_ms``, preserving the
      historical behaviour used by ``discover_events`` / standalone
      backtests.

    The LLM may only pick from these features.
    """
    if s.entry_ts_ms > 0:
        pre = s.pre_entry_bars()
        # Anchor used for "last 1h / last 2h" funding/OI windows below.
        feature_anchor_ts_ms = s.entry_ts_ms
    else:
        pre = [b for b in s.bars if b.ts_ms < result.realized_at_ts_ms]
        if not pre:
            pre = s.bars
        feature_anchor_ts_ms = s.target_ts_ms

    out: list[CandidateFeature] = []
    if not pre:
        return out

    # Last 60min vs the prior baseline (everything older than last 60).
    last_60 = pre[-60:]
    baseline = pre[:-60] if len(pre) > 60 else pre

    # 1. volume_zscore_last1h  — average volume of last 60min vs baseline
    avg_last60 = sum(b.volume for b in last_60) / len(last_60)
    base_vols = [b.volume for b in baseline]
    z = _zscore(base_vols, avg_last60) if base_vols else 0.0
    out.append(CandidateFeature("volume_zscore_last1h", bucket_zscore(z), z,
                                "Mean 1m volume in last 60min vs prior 3h"))

    # 2. funding_pre2h_extreme — most-extreme funding in last 2h (pre-entry only)
    last_2h = [
        r for r in s.funding_rates
        if feature_anchor_ts_ms - 2 * 3600 * 1000 <= r[0] < feature_anchor_ts_ms
    ]
    extreme_rate = 0.0
    if last_2h:
        extreme_rate = max(last_2h, key=lambda r: abs(r[1]))[1]
    out.append(CandidateFeature("funding_pre2h_extreme", bucket_funding(extreme_rate),
                                extreme_rate, "Most-extreme funding rate in last 2h"))

    # 3. oi_growth_pre1h — % change in OI over the last hour of pre window
    oi_growth = 0.0
    if len(s.open_interest) >= 2:
        last_oi = [
            o for o in s.open_interest
            if feature_anchor_ts_ms - 3600 * 1000 <= o[0] < feature_anchor_ts_ms
        ]
        if len(last_oi) >= 2 and last_oi[0][1] > 0:
            oi_growth = (last_oi[-1][1] - last_oi[0][1]) / last_oi[0][1]
    out.append(CandidateFeature("oi_growth_pre1h", bucket_pct(oi_growth), oi_growth,
                                "OI % change over the last hour"))

    # 4. oi_price_decoupling — |oi% growth| - |price% move| over the last hour
    if len(last_60) >= 2 and last_60[0].close > 0:
        price_move = (last_60[-1].close - last_60[0].close) / last_60[0].close
    else:
        price_move = 0.0
    decoupling = abs(oi_growth) - abs(price_move)
    out.append(CandidateFeature("oi_price_decoupling", bucket_pct(decoupling), decoupling,
                                "Excess OI growth over price move (last 1h)"))

    # 5. range_compression_pre1h — 1h ATR-ish vs longer baseline ATR
    def avg_range(bars: list[Bar]) -> float:
        if not bars:
            return 0.0
        return sum(b.high - b.low for b in bars) / len(bars)
    short_atr = avg_range(last_60)
    long_atr = avg_range(baseline) if baseline else short_atr
    compression = (short_atr / long_atr - 1.0) if long_atr > 0 else 0.0
    out.append(CandidateFeature("range_compression_pre1h", bucket_pct(compression),
                                compression,
                                "Last-1h avg range vs prior baseline avg range"))

    # 6. upper_wick_dominance_last1h — upper wick / total range over last hour
    def wick_share(bars: list[Bar], side: str) -> float:
        if not bars:
            return 0.0
        total = sum(b.high - b.low for b in bars)
        if total <= 0:
            return 0.0
        if side == "upper":
            wick = sum(b.high - max(b.open, b.close) for b in bars)
        else:
            wick = sum(min(b.open, b.close) - b.low for b in bars)
        return wick / total
    upper = wick_share(last_60, "upper")
    out.append(CandidateFeature("upper_wick_dominance_last1h", bucket_pct(upper),
                                upper, "Upper-wick share of total bar range, last 1h"))

    # 7. lower_wick_dominance_last1h — same on the other side
    lower = wick_share(last_60, "lower")
    out.append(CandidateFeature("lower_wick_dominance_last1h", bucket_pct(lower),
                                lower, "Lower-wick share of total bar range, last 1h"))

    # 8. funding_slope_pre1h — funding rate trend over the last 1h
    slope = 0.0
    last_fr = [
        r for r in s.funding_rates
        if feature_anchor_ts_ms - 3600 * 1000 <= r[0] < feature_anchor_ts_ms
    ]
    if len(last_fr) >= 2:
        slope = last_fr[-1][1] - last_fr[0][1]
    out.append(CandidateFeature("funding_slope_pre1h", bucket_pct(slope * 100),
                                slope, "Funding rate change over last 1h"))

    return out


# --------------------------------------------------------------------- #
# Rule store (JSON SoT + regenerated MD)
# --------------------------------------------------------------------- #


@dataclass
class DynamicRule:
    feature_name: str
    bucket: str
    side: Direction      # "pump" | "dump" — the direction this rule predicts
    hits: int = 0
    total: int = 0
    last_seen_ts_ms: int = 0

    @property
    def hit_rate(self) -> float:
        # Laplace smoothing
        return (self.hits + 1) / (self.total + 2)

    @property
    def key(self) -> str:
        return f"{self.feature_name}|{self.bucket}|{self.side}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "bucket": self.bucket,
            "side": self.side,
            "hits": self.hits,
            "total": self.total,
            "last_seen_ts_ms": self.last_seen_ts_ms,
        }


class RuleStore:
    """Persists DynamicRule instances. JSON is canonical; MD is regenerated."""

    def __init__(
        self,
        json_path: Path | str,
        md_path: Path | str | None = None,
        *,
        archive_below_hit_rate: float = 0.5,
        archive_min_samples: int = 5,
    ):
        self.json_path = Path(json_path)
        self.md_path = Path(md_path) if md_path else self.json_path.with_suffix(".md")
        self.archive_below_hit_rate = archive_below_hit_rate
        self.archive_min_samples = archive_min_samples
        self._rules: dict[str, DynamicRule] = {}
        if self.json_path.exists():
            self._load()

    # ---------------- public ---------------- #

    def all_rules(self) -> list[DynamicRule]:
        return list(self._rules.values())

    def get(self, feature_name: str, bucket: str, side: Direction) -> DynamicRule | None:
        return self._rules.get(f"{feature_name}|{bucket}|{side}")

    def update(
        self,
        *,
        feature_name: str,
        bucket: str,
        side: Direction,
        hit: bool,
        ts_ms: int | None = None,
    ) -> DynamicRule:
        """Bayesian-style update: increments total; increments hits if this
        observation confirms the rule."""
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        rule = self._rules.get(f"{feature_name}|{bucket}|{side}")
        if rule is None:
            rule = DynamicRule(feature_name=feature_name, bucket=bucket, side=side)
            self._rules[rule.key] = rule
        rule.total += 1
        if hit:
            rule.hits += 1
        rule.last_seen_ts_ms = ts_ms
        return rule

    def save(self) -> None:
        self._save_json()
        self._regenerate_md()

    # ---------------- internals ---------------- #

    def _load(self) -> None:
        try:
            data = json.loads(self.json_path.read_text())
            for r in data.get("rules", []):
                rule = DynamicRule(
                    feature_name=r["feature_name"],
                    bucket=r["bucket"],
                    side=r["side"],
                    hits=int(r.get("hits", 0)),
                    total=int(r.get("total", 0)),
                    last_seen_ts_ms=int(r.get("last_seen_ts_ms", 0)),
                )
                self._rules[rule.key] = rule
        except Exception as e:
            logger.warning("RuleStore load failed (%s); starting empty", e)
            self._rules = {}

    def _save_json(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at_ts_ms": int(time.time() * 1000),
            "rules": [r.to_dict() for r in self._rules.values()],
        }
        # Atomic replace
        tmp = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self.json_path)

    def _regenerate_md(self) -> None:
        active: list[DynamicRule] = []
        archived: list[DynamicRule] = []
        for r in self._rules.values():
            if r.total >= self.archive_min_samples and r.hit_rate < self.archive_below_hit_rate:
                archived.append(r)
            else:
                active.append(r)
        active.sort(key=lambda r: (-r.hit_rate, -r.total))
        archived.sort(key=lambda r: (r.hit_rate, -r.total))

        lines: list[str] = []
        lines.append("---")
        lines.append("inclusion: always")
        lines.append("---")
        lines.append("")
        lines.append("# Dynamic Rules (auto-generated by learning_engine)")
        lines.append("")
        lines.append("**DO NOT EDIT BY HAND.** This file is regenerated every time the")
        lines.append("learning engine runs a post-mortem. The canonical source is the")
        lines.append("sibling `dynamic_rules.json`.")
        lines.append("")
        lines.append("Hit rate uses Laplace smoothing: `(hits + 1) / (total + 2)`.")
        lines.append("")
        lines.append("## Active Rules")
        lines.append("")
        lines.append("| # | feature | bucket | side | hits/total | hit_rate |")
        lines.append("|---|---------|--------|------|------------|----------|")
        for i, r in enumerate(active, 1):
            lines.append(
                f"| {i} | {r.feature_name} | {r.bucket} | {r.side} | "
                f"{r.hits}/{r.total} | {r.hit_rate:.2%} |"
            )
        if not active:
            lines.append("| _(none yet)_ |  |  |  |  |  |")

        lines.append("")
        lines.append("## Archived (low hit rate, kept for audit)")
        lines.append("")
        lines.append("| # | feature | bucket | side | hits/total | hit_rate |")
        lines.append("|---|---------|--------|------|------------|----------|")
        for i, r in enumerate(archived, 1):
            lines.append(
                f"| {i} | {r.feature_name} | {r.bucket} | {r.side} | "
                f"{r.hits}/{r.total} | {r.hit_rate:.2%} |"
            )
        if not archived:
            lines.append("| _(none yet)_ |  |  |  |  |  |")

        lines.append("")
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.md_path.with_suffix(self.md_path.suffix + ".tmp")
        tmp.write_text("\n".join(lines))
        os.replace(tmp, self.md_path)


# --------------------------------------------------------------------- #
# Post-mortem (LLM call OR fallback)
# --------------------------------------------------------------------- #


@dataclass
class PostMortemPick:
    feature_name: str
    bucket: str
    rationale: str


async def post_mortem_via_deepseek(
    *,
    engine: DeepSeekEngine,
    symbol: str,
    candidates: list[CandidateFeature],
    result: EventResult,
) -> list[PostMortemPick]:
    """Ask DeepSeek to pick the 1-2 most prognostic features from `candidates`.
    Closed-set: the LLM may only return names that exist in the input list.
    Falls back to `_fallback_pick` on parse / network / budget errors so the
    learning loop never breaks the system.
    """
    allowed: dict[str, CandidateFeature] = {c.name: c for c in candidates}
    user = (
        "You are reviewing a confirmed altcoin trade AFTER it was opened.\n"
        f"Symbol: {symbol}\n"
        f"Realized direction: {result.direction}\n"
        f"Magnitude (signed PnL relative to entry): {result.magnitude_pct * 100:+.2f}%\n"
        f"Minutes from entry to extremum: {result.minutes_to_extremum}\n\n"
        "Below are 8 PRE-ENTRY candidate features measured strictly before\n"
        "entry. Pick 1-2 features from this list (no others) that you judge\n"
        "most predictive of the realized post-entry move. A negative\n"
        "magnitude means the trader's thesis missed (e.g. a long got\n"
        "stopped out); the picks should still be the features that BEST\n"
        "explained the realized direction, even when that direction was\n"
        "the opposite of the trader's bet.\n\n"
    )
    for c in candidates:
        user += f"- name={c.name} bucket={c.bucket} value={c.raw_value:.6f}  ({c.description})\n"

    user += (
        "\nReply with ONE JSON object EXACTLY in this schema and nothing else:\n"
        '{"picks": [{"feature_name": "<one of the names above>", '
        '"bucket": "<that feature\'s bucket as given above>", '
        '"rationale": "<short>"}], "narrative": "<short>"}\n'
        "Hard rules: feature_name MUST be one of the names above; bucket MUST\n"
        "match the given bucket exactly; pick at most 2 features."
    )

    try:
        client = await engine._get_client()  # type: ignore[attr-defined]
        body = {
            "model": engine.model,
            "messages": [
                {"role": "system",
                 "content": ("You are a quantitative post-mortem analyst. You pick from a "
                             "closed list of features and never invent new ones. Output "
                             "STRICT JSON only.")},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {engine.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        engine.budget.add(int(resp.json().get("usage", {}).get("total_tokens", 0)))
        parsed = json.loads(content)
        picks: list[PostMortemPick] = []
        for raw in parsed.get("picks", [])[:2]:
            name = str(raw.get("feature_name", ""))
            bucket = str(raw.get("bucket", ""))
            rationale = str(raw.get("rationale", "")).strip() or "(no rationale)"
            if name not in allowed:
                logger.warning("LLM picked unknown feature %r — discarding", name)
                continue
            # Force bucket to match what we measured (LLM cannot edit reality)
            real_bucket = allowed[name].bucket
            if bucket != real_bucket:
                logger.warning("LLM picked bucket %r for %s but real bucket is %r — using real",
                               bucket, name, real_bucket)
                bucket = real_bucket
            picks.append(PostMortemPick(feature_name=name, bucket=bucket, rationale=rationale))
        if picks:
            return picks
        logger.warning("LLM returned no valid picks; using fallback")
    except (EngineError, httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.warning("DeepSeek post-mortem failed (%s) — falling back to heuristic", e)

    return _fallback_pick(candidates, result)


def _fallback_pick(
    candidates: list[CandidateFeature], result: EventResult,
) -> list[PostMortemPick]:
    """Heuristic when LLM is unavailable: pick the most extreme features
    aligned with the realized direction."""
    if not candidates:
        return []

    # Score each candidate by extremity of its bucket label.
    extremity_rank = {
        "calm": 0, "flat": 0, "neutral": 0,
        "pos_small": 1, "neg_small": 1, "positive": 1, "negative": 1,
        "pos_elevated": 2, "neg_elevated": 2,
        "pos_medium": 2, "neg_medium": 2,
        "pos_high": 3, "neg_high": 3,
        "pos_large": 3, "neg_large": 3,
        "very_positive": 3, "very_negative": 3,
        "pos_extreme": 4, "neg_extreme": 4,
        "pos_xlarge": 4, "neg_xlarge": 4,
    }
    ranked = sorted(
        candidates,
        key=lambda c: extremity_rank.get(c.bucket, 0),
        reverse=True,
    )
    top = [c for c in ranked if extremity_rank.get(c.bucket, 0) > 0][:2]
    if not top:
        top = ranked[:1]
    return [
        PostMortemPick(
            feature_name=c.name, bucket=c.bucket,
            rationale=f"Heuristic fallback: extreme bucket aligned with {result.direction}",
        )
        for c in top
    ]


# --------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------- #


@dataclass
class PostMortemReport:
    symbol: str
    target_ts_ms: int
    result: EventResult
    candidates: list[CandidateFeature]
    picks: list[PostMortemPick]
    updated_rules: list[DynamicRule] = field(default_factory=list)


async def run_post_mortem(
    *,
    symbol: str,
    target_ts_ms: int,
    store: RuleStore,
    engine: DeepSeekEngine | None = None,
    slice_override: HistoricalSlice | None = None,
    entry_ts_ms: int | None = None,
    expected_direction: Direction | None = None,
    hours_back: int = 4,
    hours_forward: int = 1,
) -> PostMortemReport:
    """Main entry point. Returns the full report and persists rule updates.

    `slice_override` lets tests / demos inject a synthetic slice without
    touching the network.

    Bug #2 fix:

    * When called with ``entry_ts_ms`` (the live post-mortem path), the slice
      spans ``[entry - hours_back, entry + hours_forward]``; ``compute_event_result``
      evaluates only the post-entry segment and ``extract_candidate_features``
      uses only the pre-entry segment. The features can no longer be
      contaminated by post-entry data, and the result can no longer be
      "the biggest move that happened before we even opened".
    * ``expected_direction`` makes the result direction-aware: a long that
      gets stopped out is recorded as a *negative-magnitude pump* (we bet
      pump, lost money), so the rule store learns from misses too. Without
      this, a stopped-out long would pick up the dump bucket of every
      losing pre-entry feature, training the system to short the very
      patterns it had marked as bullish.
    """
    if slice_override is not None:
        s = slice_override
    elif entry_ts_ms is not None:
        s = await fetch_historical_slice(
            symbol, target_ts_ms,
            hours_back=hours_back,
            hours_forward=hours_forward,
            entry_ts_ms=entry_ts_ms,
        )
    else:
        s = await fetch_historical_slice(symbol, target_ts_ms,
                                          hours_back=hours_back)

    result = compute_event_result(s, expected_direction=expected_direction)
    candidates = extract_candidate_features(s, result)

    if engine is not None and engine.api_key:
        picks = await post_mortem_via_deepseek(
            engine=engine, symbol=symbol, candidates=candidates, result=result,
        )
    else:
        logger.info("No DeepSeek engine available; using heuristic fallback")
        picks = _fallback_pick(candidates, result)

    updated: list[DynamicRule] = []
    picked_keys = {(p.feature_name, p.bucket) for p in picks}

    # Update every candidate feature's rule store entry: pick = hit, others = miss.
    # This is what gives non-picked features a way to grow their `total` and
    # eventually move into the archived band if their bucket value never wins.
    #
    # Rules are filed under ``result.direction``. For the entry-aware path
    # this is the realized direction RELATIVE TO THE TRADER'S BET (so a
    # losing long records under "pump" with a negative magnitude, training
    # the feature against false-positive pump signatures rather than
    # falsely teaching the system the same features predict dumps).
    for c in candidates:
        is_hit = (c.name, c.bucket) in picked_keys
        rule = store.update(
            feature_name=c.name,
            bucket=c.bucket,
            side=result.direction,
            hit=is_hit,
            ts_ms=target_ts_ms,
        )
        if is_hit:
            updated.append(rule)

    store.save()

    return PostMortemReport(
        symbol=symbol,
        target_ts_ms=target_ts_ms,
        result=result,
        candidates=candidates,
        picks=picks,
        updated_rules=updated,
    )


# --------------------------------------------------------------------- #
# Synthetic slice for demos / tests (textbook dump pattern)
# --------------------------------------------------------------------- #


def synthesize_dump_slice(
    symbol: str = "RAVEUSDT", target_ts_ms: int | None = None,
) -> HistoricalSlice:
    """Build a 4h slice that ends in a -25% waterfall dump. Used by demos
    and tests; no network involved.

    Pattern: 3h of compression → +22% silent OI build → upper-wick rejection
    → final 10min waterfall.
    """
    if target_ts_ms is None:
        target_ts_ms = int(time.time() * 1000)
    base_price = 1.0000
    bars: list[Bar] = []
    funding: list[tuple[int, float]] = []
    oi: list[tuple[int, float]] = []

    start_ms = target_ts_ms - 4 * 3600 * 1000
    n = 4 * 60  # 240 bars

    for i in range(n):
        ts = start_ms + i * 60_000
        # Phase 1 (0..180 min): tight range compression around 1.00
        if i < 180:
            o = base_price + (i % 5 - 2) * 0.001
            c = o + (i % 3 - 1) * 0.0005
            h = max(o, c) + 0.001
            lo = min(o, c) - 0.001
            v = 1000.0 + (i % 7) * 50
            oi_val = 800_000 + i * 200            # gentle climb
        # Phase 2 (180..230 min): silent OI build, upper wicks rejecting up-moves
        elif i < 230:
            o = base_price - 0.005 + (i % 3) * 0.001
            c = o - 0.002
            h = o + 0.012   # long upper wick
            lo = min(o, c) - 0.001
            v = 2500.0 + (i % 5) * 200
            oi_val = 800_000 + 36_000 + (i - 180) * 4000  # +22% over the hour
        # Phase 3 (230..240 min): waterfall dump
        else:
            step = i - 230
            o = base_price - 0.02 - step * 0.025
            c = o - 0.025
            h = o + 0.001
            lo = c - 0.005
            v = 8000.0 + step * 800
            oi_val = 1_000_000 - step * 5000

        bars.append(Bar(ts, o, h, lo, c, v))

        # Funding rate fixings every 10 min in the last 2 hours, climbing positive.
        if i >= 120 and i % 10 == 0:
            rate = 0.0002 + (i - 120) * 1e-5  # walks from +0.02% to +0.14%
            funding.append((ts, rate))
        if i % 5 == 0:
            oi.append((ts, oi_val))

    return HistoricalSlice(symbol=symbol, target_ts_ms=target_ts_ms,
                           bars=bars, funding_rates=funding, open_interest=oi)




def synthesize_long_stopout_slice(
    symbol: str = "RAVEUSDT",
    entry_ts_ms: int | None = None,
    *,
    hours_back: int = 4,
    hours_forward: int = 1,
) -> HistoricalSlice:
    """Build an entry-aware slice where a LONG would get stopped out.

    Bug #2 fix needed a fixture that exercises both halves of the slice:
      * Pre-entry (4h): a textbook bullish-looking compression+OI build.
        These are the features that *fired the long signal*.
      * Post-entry (1h): the price falls 6%, taking the long out.

    Used by ``test_compute_event_result_*`` and ``test_run_post_mortem_*``
    to prove the entry-aware path picks features from the bullish setup
    yet records the result with negative magnitude under ``"pump"`` (the
    expected direction), not ``"dump"``.
    """
    if entry_ts_ms is None:
        entry_ts_ms = int(time.time() * 1000)
    bars: list[Bar] = []
    funding: list[tuple[int, float]] = []
    oi: list[tuple[int, float]] = []

    pre_minutes = hours_back * 60
    post_minutes = hours_forward * 60

    base_price = 1.0000
    # ---- pre-entry: 4h bullish setup ending right at entry_ts_ms ----
    pre_start_ms = entry_ts_ms - pre_minutes * 60_000
    for i in range(pre_minutes):
        ts = pre_start_ms + i * 60_000
        if i < pre_minutes - 60:
            # Quiet compression for the first 3h
            o = base_price + (i % 5 - 2) * 0.001
            c = o + (i % 3 - 1) * 0.0005
            h = max(o, c) + 0.001
            lo = min(o, c) - 0.001
            v = 1000.0 + (i % 7) * 50
            oi_val = 800_000 + i * 200
        else:
            # Last 1h: bullish OI build + rising volume (the trigger)
            step = i - (pre_minutes - 60)
            o = base_price + step * 0.0008
            c = o + 0.001
            h = c + 0.001
            lo = o - 0.001
            v = 2500.0 + step * 100
            oi_val = 1_000_000 + step * 5000   # +30% over the hour
        bars.append(Bar(ts, o, h, lo, c, v))
        if i % 10 == 0 and i >= pre_minutes - 120:
            # Funding climbing positive in the last 2h
            rate = 0.0006 + (i - (pre_minutes - 120)) * 1e-5
            funding.append((ts, rate))
        if i % 5 == 0:
            oi.append((ts, oi_val))

    # entry close ~ base_price + 0.06 ish
    entry_ref = bars[-1].close

    # ---- post-entry: 1h waterfall down 6% ----
    for j in range(post_minutes):
        ts = entry_ts_ms + j * 60_000
        # Falls from entry_ref to entry_ref * 0.94 over 60 minutes.
        # We pin the bar high BELOW entry_ref so a "pump thesis" post-mortem
        # records a strictly negative magnitude (no spurious tiny wick up).
        frac = (j + 1) / post_minutes
        c = entry_ref * (1.0 - 0.06 * frac)
        o = entry_ref * (1.0 - 0.06 * (j / post_minutes))
        # cap high below entry by a small epsilon so max_up is exactly 0
        h = min(max(o, c) + 0.0005, entry_ref - 0.0005)
        lo = min(o, c) - 0.0010
        v = 5000.0 + j * 100
        bars.append(Bar(ts, o, h, lo, c, v))

    return HistoricalSlice(
        symbol=symbol,
        target_ts_ms=entry_ts_ms,
        bars=bars,
        funding_rates=funding,
        open_interest=oi,
        entry_ts_ms=entry_ts_ms,
    )
