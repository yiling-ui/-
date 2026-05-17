"""historical_analyzer.py — KOL historical hit-rate tracker.

The :mod:`social.crawler` aggregator already pulls Binance-Square posts
into :class:`SocialSnapshot.binance_square_posts`; the
:class:`ai_engine.LLMEngine` already classifies a snapshot's overall
``kol_intent`` (``frontrun_call`` / ``exit_liquidity`` / ``neutral``).
What was missing — and what the operator surfaced as "social signal
盲区 #2" — is a **memory of which authors actually called pumps vs.
dumped on followers**. Without that memory the fuser treats every
``exit_liquidity`` flag identically: a 200-follower bot's call carries
the same weight as a 200K-follower veteran's, and a perma-bear who has
never been right gets the same hard veto privileges as a calibrated
analyst.

This module closes that gap with the same shape as the rest of the
system:

* Per-author Bayesian Laplace-smoothed hit-rate counters, keyed by
  ``(author, intent)``. The smoothing prior gives a fresh author a
  neutral 50% rate after one observation, drifting toward the empirical
  mean as the sample grows; the same shape ``learning_engine.RuleStore``
  uses for rule features so an operator only learns one statistical
  model.
* Atomic JSON persistence (``tmp + os.replace``) under
  ``.kiro/state/social/kol_history.json`` -- mirrors
  :class:`risk.persistence.AccountPersistor` and
  :class:`learning_engine.RuleStore`. Corrupt files log a warning and
  start fresh; there is no scenario where a malformed history file
  blocks the trading loop.
* Pure-Python, stdlib-only. No external deps.
* The async batch helper :func:`build_observations_from_posts` lets an
  operator (or a one-shot CLI script) replay months of square scrapes
  through a kline fetcher of their choice (ccxt, fixture, mock) and
  populate the store before live trading starts. The same
  ``record_observation`` entry point is used by the post-mortem path
  online: a position closes ⇒ for every KOL that mentioned the symbol
  in the last hour, score them on what actually happened.

Public surface (re-exported via ``social.__init__``):

    KOLObservation
    KOLScore
    KOLHistoryStore
    HistoricalAnalyzer
    build_observations_from_posts

The fuser-side adjustment is implemented in
:meth:`HistoricalAnalyzer.adjust_kol_confidence`, which the
``ScoreFuser.evaluate`` path calls after the LLM verdict but before the
hard-veto / soft-cap branch. See ``fuser.py`` for the wiring.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Same intent vocabulary as :class:`ai_engine.AIVerdict.kol_intent`. We
# replicate the literal here instead of importing it to avoid a
# circular dependency (ai_engine imports social.* indirectly via the
# pipeline).
KOLIntent = Literal["frontrun_call", "exit_liquidity", "neutral"]

# Realised direction symbols. Align with
# :func:`learning_engine.compute_event_result` so a future cross-call
# can compose them without translation.
RealisedDirection = Literal["pump", "dump", "neutral"]

# Storage schema version. Bumping this triggers a clean wipe of any
# older file the next time the store loads — cheaper than building a
# migration framework for a tiny JSON dict.
_SCHEMA_VERSION = 1


# --------------------------------------------------------------------- #
# Author-key normalisation
# --------------------------------------------------------------------- #


_HANDLE_PREFIX = re.compile(r"^[@$]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_author(author: str | None) -> str:
    """Render an author handle in a canonical form.

    Binance Square authors arrive with mixed prefixes (``@kol_x``,
    ``$kol_x``), surrounding whitespace, and inconsistent casing.
    Normalising once at the boundary keeps the stored key stable across
    scrapers, locales and casing typos.

    Returns ``""`` on empty / None input — the analyzer treats an empty
    author as "anonymous" and skips both record and lookup paths so an
    unattributed post can't pollute the store.
    """
    if not author:
        return ""
    s = _WHITESPACE.sub("", str(author)).strip()
    s = _HANDLE_PREFIX.sub("", s)
    return s.casefold()


# --------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------- #


@dataclass
class KOLObservation:
    """One realised outcome for a single KOL × symbol × intent triple."""

    author: str
    symbol: str
    intent: KOLIntent
    ts_ms: int
    realised_direction: RealisedDirection
    magnitude_pct: float
    follower_count: int = 0
    notes: str = ""

    def is_correct(self) -> bool:
        """Did this observation realise the KOL's stated intent?

        ``frontrun_call`` is correct when the price PUMPS afterwards
        (the KOL was telling followers to buy in time).
        ``exit_liquidity`` is correct when the price DUMPS afterwards
        (the KOL was offloading bags).
        ``neutral`` calls are not scored; they're stored only for the
        sample-size denominator.
        """
        if self.intent == "frontrun_call":
            return self.realised_direction == "pump"
        if self.intent == "exit_liquidity":
            return self.realised_direction == "dump"
        return False


@dataclass
class KOLScore:
    """Bayesian-smoothed scorecard for one (author, intent) pair.

    ``hit_rate`` is the Laplace-shrunk Bernoulli mean
    ``(hits + 1) / (total + 2)`` so a fresh author with zero
    observations starts at 0.5 (no information) and a pristine 1/1
    record sits at 2/3 ≈ 0.667 rather than the fragile 100% the raw
    ratio would produce.

    ``samples`` mirrors ``total`` so consumers can apply their own
    minimum-sample-size gate without dipping into the dataclass.
    """

    author: str
    intent: KOLIntent
    hits: int
    total: int
    avg_magnitude_pct: float = 0.0
    last_seen_ts_ms: int = 0

    @property
    def hit_rate(self) -> float:
        return (self.hits + 1) / (self.total + 2)

    @property
    def samples(self) -> int:
        return self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "intent": self.intent,
            "hits": self.hits,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "avg_magnitude_pct": round(self.avg_magnitude_pct, 6),
            "last_seen_ts_ms": self.last_seen_ts_ms,
        }


# --------------------------------------------------------------------- #
# Persistent counter store
# --------------------------------------------------------------------- #


@dataclass
class KOLHistoryStore:
    """Thread-safe, atomically-persisted (author, intent) counter map.

    Storage layout::

        {
            "schema_version": 1,
            "counters": {
                "kol_x|frontrun_call": {
                    "hits": 4, "total": 6,
                    "avg_magnitude_pct": 0.034,
                    "last_seen_ts_ms": 17xxxxxxxx
                },
                ...
            }
        }

    Atomic-write pattern: ``json.dumps`` is rendered to a sibling tmp
    file, ``fsync`` is best-effort, then ``os.replace`` swaps the file
    atomically. A crashed write therefore either leaves the previous
    snapshot untouched or completes; never half-written.

    Concurrency: a coarse lock guards every read AND write so the
    daemon's asyncio fuser can call :meth:`get_score` while a
    background post-mortem worker is in :meth:`record`.
    """

    path: Path
    autosave: bool = True
    _counters: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, init=False,
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # pragma: no cover - filesystem edge
            logger.warning(
                "KOLHistoryStore: parent mkdir failed (%s): %s",
                self.path.parent, e,
            )
        self._load()

    # --------------------- key helpers --------------------- #

    @staticmethod
    def _key(author: str, intent: KOLIntent) -> str:
        return f"{normalize_author(author)}|{intent}"

    # --------------------- IO --------------------- #

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                raw = self.path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "KOLHistoryStore: read failed (%s): %s — starting fresh",
                    self.path, e,
                )
                return
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    "KOLHistoryStore: corrupt JSON at %s (%s) — "
                    "starting fresh", self.path, e,
                )
                return
            if not isinstance(data, dict):
                logger.warning(
                    "KOLHistoryStore: unexpected top-level shape "
                    "(%s) — starting fresh", type(data).__name__,
                )
                return
            schema = int(data.get("schema_version", 0))
            if schema != _SCHEMA_VERSION:
                # We deliberately wipe rather than migrate — tiny dict,
                # nothing irreplaceable, and a stale schema is more
                # likely a hand-edit gone wrong than a real upgrade.
                logger.warning(
                    "KOLHistoryStore: schema_version=%s (expected %s); "
                    "wiping and starting fresh.",
                    schema, _SCHEMA_VERSION,
                )
                return
            counters = data.get("counters")
            if not isinstance(counters, dict):
                return
            valid: dict[str, dict[str, Any]] = {}
            for k, v in counters.items():
                if not isinstance(k, str) or "|" not in k:
                    continue
                if not isinstance(v, dict):
                    continue
                try:
                    valid[k] = {
                        "hits": int(v.get("hits", 0)),
                        "total": int(v.get("total", 0)),
                        "avg_magnitude_pct": float(
                            v.get("avg_magnitude_pct", 0.0),
                        ),
                        "last_seen_ts_ms": int(
                            v.get("last_seen_ts_ms", 0),
                        ),
                    }
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "KOLHistoryStore: skipping malformed entry %r: %s",
                        k, e,
                    )
                    continue
            self._counters = valid

    def save(self) -> None:
        """Atomically persist the current state to disk.

        Failures are swallowed: persistence is defence-in-depth, never
        the system of record. The trading loop must not stall on a
        flaky disk.
        """
        with self._lock:
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "counters": dict(self._counters),
            }
            text = json.dumps(payload, sort_keys=True, indent=2)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            try:
                fd = os.open(str(tmp), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                # fsync isn't available on every filesystem (tmpfs
                # under some sandboxes); the os.replace below is still
                # atomic by POSIX guarantees, just less durable. Soft
                # failure mode is acceptable for a counter file.
                pass
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning(
                "KOLHistoryStore: save failed (%s) — state retained "
                "in memory only", e,
            )
            try:
                tmp.unlink()
            except OSError:
                pass

    # --------------------- mutators --------------------- #

    def record(self, obs: KOLObservation) -> KOLScore | None:
        """Apply one observation to the counters; returns the updated
        score for the (author, intent) pair, or ``None`` when the
        observation was rejected (anonymous author / neutral intent).

        Neutral calls are deliberately not stored: they have no
        directional bet to score, and including them in the
        ``frontrun_call`` / ``exit_liquidity`` denominators would dilute
        the hit-rate signal. We DO surface the magnitude in the score
        for downstream debugging via ``avg_magnitude_pct``.
        """
        if obs.intent == "neutral":
            return None
        norm_author = normalize_author(obs.author)
        if not norm_author:
            return None
        key = f"{norm_author}|{obs.intent}"
        with self._lock:
            row = self._counters.get(key)
            if row is None:
                row = {
                    "hits": 0, "total": 0,
                    "avg_magnitude_pct": 0.0,
                    "last_seen_ts_ms": 0,
                }
                self._counters[key] = row
            prev_total = row["total"]
            row["total"] = prev_total + 1
            if obs.is_correct():
                row["hits"] = int(row["hits"]) + 1
            # Running mean of the absolute magnitude — gives the fuser
            # a sense of "this KOL's calls move price ~5% on average"
            # without storing every observation.
            mag = abs(float(obs.magnitude_pct))
            prev_avg = float(row["avg_magnitude_pct"])
            row["avg_magnitude_pct"] = (
                (prev_avg * prev_total + mag) / row["total"]
            )
            row["last_seen_ts_ms"] = max(
                int(row["last_seen_ts_ms"]), int(obs.ts_ms),
            )
            score = KOLScore(
                author=norm_author,
                intent=obs.intent,
                hits=int(row["hits"]),
                total=int(row["total"]),
                avg_magnitude_pct=float(row["avg_magnitude_pct"]),
                last_seen_ts_ms=int(row["last_seen_ts_ms"]),
            )
        if self.autosave:
            self.save()
        return score

    def reset(self) -> None:
        """Wipe all counters. Used by tests and the operator's
        ``rebuild`` CLI."""
        with self._lock:
            self._counters.clear()
        if self.autosave:
            self.save()

    # --------------------- accessors --------------------- #

    def get_score(self, author: str, intent: KOLIntent) -> KOLScore | None:
        norm_author = normalize_author(author)
        if not norm_author or intent == "neutral":
            return None
        key = f"{norm_author}|{intent}"
        with self._lock:
            row = self._counters.get(key)
            if row is None:
                return None
            return KOLScore(
                author=norm_author,
                intent=intent,
                hits=int(row["hits"]),
                total=int(row["total"]),
                avg_magnitude_pct=float(row["avg_magnitude_pct"]),
                last_seen_ts_ms=int(row["last_seen_ts_ms"]),
            )

    def all_scores(
        self, intent: KOLIntent | None = None,
    ) -> list[KOLScore]:
        out: list[KOLScore] = []
        with self._lock:
            for key, row in self._counters.items():
                try:
                    author, k_intent = key.split("|", 1)
                except ValueError:
                    continue
                if intent is not None and k_intent != intent:
                    continue
                if k_intent not in ("frontrun_call", "exit_liquidity"):
                    continue
                out.append(KOLScore(
                    author=author,
                    intent=k_intent,  # type: ignore[arg-type]
                    hits=int(row["hits"]),
                    total=int(row["total"]),
                    avg_magnitude_pct=float(row["avg_magnitude_pct"]),
                    last_seen_ts_ms=int(row["last_seen_ts_ms"]),
                ))
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._counters)


# --------------------------------------------------------------------- #
# Fuser-facing analyzer
# --------------------------------------------------------------------- #


@dataclass
class HistoricalAnalyzerConfig:
    """Adjustment policy for the fuser hook.

    The default thresholds are deliberately conservative:

    * **min_samples=10**: same floor as
      :attr:`fuser.FuserConfig.learned_min_samples`. A 6-of-9 hit ratio
      reads as 70% but is statistically meaningless.
    * **strong_bound=0.65 / weak_bound=0.40**: the hit_rate boundaries
      that flip the LLM-attributed confidence one notch up or down.
      A KOL with hit_rate>=0.65 on ``exit_liquidity`` means his dump
      calls are well-calibrated, so we INCREASE the effective
      confidence (more aggressive veto).
    * **conf_lift_max / conf_drop_max=0.20**: same envelope as
      :attr:`fuser.FuserConfig.learned_reward_lift_cap`. Capped per
      adjustment so a single KOL can never flip the verdict outright.
    * **shrink_k=5.0**: aligns with
      :attr:`fuser.FuserConfig.learned_confidence_k`.
    """

    min_samples: int = 10
    strong_bound: float = 0.65
    weak_bound: float = 0.40
    conf_lift_max: float = 0.20
    conf_drop_max: float = 0.20
    shrink_k: float = 5.0


@dataclass
class KOLAdjustment:
    """The result of one ``adjust_kol_confidence`` call.

    Carrying the diagnostic fields alongside the adjusted confidence
    lets the fuser surface a human-readable note in
    ``FusedSignal.notes`` so an operator reviewing why a SHORT was let
    through can see *which* KOL's track record vetoed which audit
    decision.
    """

    original_confidence: float
    adjusted_confidence: float
    delta: float
    contributing_authors: list[str]
    intent: KOLIntent
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_confidence": round(self.original_confidence, 4),
            "adjusted_confidence": round(self.adjusted_confidence, 4),
            "delta": round(self.delta, 4),
            "contributing_authors": list(self.contributing_authors),
            "intent": self.intent,
            "note": self.note,
        }


@dataclass
class HistoricalAnalyzer:
    """KOL-history-aware confidence adjuster.

    Wraps a :class:`KOLHistoryStore`. The fuser calls
    :meth:`adjust_kol_confidence` once per LLM verdict (when
    ``kol_intent != "neutral"``); the post-mortem path calls
    :meth:`record_observation` once per closed position to update the
    counters.

    The class is **stateless beyond the wrapped store**, so test fixtures
    can construct a temp-dir store and exercise the full hook without
    threading or background workers.
    """

    store: KOLHistoryStore
    config: HistoricalAnalyzerConfig = field(
        default_factory=HistoricalAnalyzerConfig,
    )

    # ----- write side: post-mortem path ----- #

    def record_observation(
        self,
        *,
        author: str,
        symbol: str,
        intent: KOLIntent,
        ts_ms: int,
        realised_direction: RealisedDirection,
        magnitude_pct: float,
        follower_count: int = 0,
        notes: str = "",
    ) -> KOLScore | None:
        """Persist one realised outcome.

        Returns the updated :class:`KOLScore` on success, ``None`` when
        the observation was rejected (anonymous, neutral intent, or any
        other validation failure). Never raises into the trading loop.
        """
        try:
            obs = KOLObservation(
                author=author,
                symbol=symbol,
                intent=intent,
                ts_ms=int(ts_ms),
                realised_direction=realised_direction,
                magnitude_pct=float(magnitude_pct),
                follower_count=int(follower_count),
                notes=notes,
            )
        except (TypeError, ValueError) as e:
            logger.warning(
                "HistoricalAnalyzer: rejecting malformed observation "
                "for %r/%s: %s", author, symbol, e,
            )
            return None
        return self.store.record(obs)

    # ----- read side: fuser hook ----- #

    def lookup(self, author: str, intent: KOLIntent) -> KOLScore | None:
        return self.store.get_score(author, intent)

    def adjust_kol_confidence(
        self,
        *,
        kol_intent: KOLIntent,
        confidence: float,
        authors: Iterable[str],
    ) -> KOLAdjustment:
        """Tilt the LLM's KOL confidence by historical accuracy.

        The fuser already routes ``exit_liquidity`` through a
        confidence-graded HARD VETO / SOFT CAP path (see
        ``fuser.ScoreFuser.evaluate``). Without history, every
        ``exit_liquidity`` call lands in the same bucket regardless of
        who called it. With history:

        * If the cited KOLs are well-calibrated on this intent
          (avg hit_rate >= ``strong_bound`` after sample-size gate),
          we INCREASE confidence by up to ``conf_lift_max`` so the
          fuser is more likely to take the hard veto branch.
        * If the cited KOLs have been wrong more often than right
          (avg hit_rate <= ``weak_bound``), we DROP confidence by up
          to ``conf_drop_max`` so a noisy "everyone is a perma-bear"
          chorus can't trigger the hard veto.
        * Authors below the sample-size gate contribute zero — they
          still appear in ``contributing_authors`` (with the note
          ``insufficient_samples``) so the operator sees that the
          history layer ran but did not have data to act on.

        The function is symmetrical: the same logic applies when
        ``kol_intent == "frontrun_call"`` (hot calls); a known-good
        caller's "buy now" raises confidence, an unreliable one's
        lowers it.

        Always returns a :class:`KOLAdjustment` so the caller can
        unconditionally consult ``adjusted_confidence`` without nil
        checks. ``adjusted_confidence == original_confidence`` when no
        author crossed the sample-size gate.
        """
        original = float(confidence)
        # Clamp to the AIVerdict.confidence domain so a stray score of
        # 1.5 from a buggy provider doesn't propagate back to the
        # fuser as an even-larger 1.7.
        original = max(0.0, min(1.0, original))
        if kol_intent == "neutral":
            return KOLAdjustment(
                original_confidence=original,
                adjusted_confidence=original,
                delta=0.0,
                contributing_authors=[],
                intent=kol_intent,
                note="neutral_intent_skipped",
            )
        author_list = [a for a in authors if normalize_author(a)]
        if not author_list:
            return KOLAdjustment(
                original_confidence=original,
                adjusted_confidence=original,
                delta=0.0,
                contributing_authors=[],
                intent=kol_intent,
                note="no_authors",
            )
        scores: list[KOLScore] = []
        skipped_authors: list[str] = []
        for a in author_list:
            sc = self.lookup(a, kol_intent)
            if sc is None or sc.samples < self.config.min_samples:
                skipped_authors.append(normalize_author(a))
                continue
            scores.append(sc)
        if not scores:
            return KOLAdjustment(
                original_confidence=original,
                adjusted_confidence=original,
                delta=0.0,
                contributing_authors=skipped_authors,
                intent=kol_intent,
                note="insufficient_samples",
            )
        # Weighted by total observations (cap at 200 so a 10K-sample
        # outlier doesn't drown the rest); shrunk by samples / (samples + k).
        weighted_sum = 0.0
        weight_total = 0.0
        for sc in scores:
            cap = min(sc.samples, 200)
            weight = cap * (sc.samples / (sc.samples + self.config.shrink_k))
            weighted_sum += sc.hit_rate * weight
            weight_total += weight
        if weight_total <= 0.0:
            return KOLAdjustment(
                original_confidence=original,
                adjusted_confidence=original,
                delta=0.0,
                contributing_authors=[s.author for s in scores],
                intent=kol_intent,
                note="zero_weight",
            )
        avg_rate = weighted_sum / weight_total
        # Map the rate into a symmetric tilt around 0.5.
        if avg_rate >= self.config.strong_bound:
            # Linear interpolation between strong_bound..1.0 -> 0..conf_lift_max
            span = max(1e-9, 1.0 - self.config.strong_bound)
            lift = self.config.conf_lift_max * min(
                1.0, (avg_rate - self.config.strong_bound) / span,
            )
            delta = lift
        elif avg_rate <= self.config.weak_bound:
            # weak_bound..0.0 -> 0..conf_drop_max
            span = max(1e-9, self.config.weak_bound)
            drop = self.config.conf_drop_max * min(
                1.0, (self.config.weak_bound - avg_rate) / span,
            )
            delta = -drop
        else:
            delta = 0.0
        adjusted = max(0.0, min(1.0, original + delta))
        if math.isnan(adjusted):
            # Defensive: a NaN in the score map should never override a
            # valid LLM confidence. Hard-pin to original.
            adjusted = original
            delta = 0.0
        contributing = [s.author for s in scores]
        if skipped_authors:
            contributing.extend(f"{a}(insufficient)" for a in skipped_authors)
        note = (
            f"avg_hit_rate={avg_rate:.3f} "
            f"({len(scores)} authors >= {self.config.min_samples} samples; "
            f"{len(skipped_authors)} below floor) -> delta={delta:+.3f}"
        )
        return KOLAdjustment(
            original_confidence=original,
            adjusted_confidence=adjusted,
            delta=delta,
            contributing_authors=contributing,
            intent=kol_intent,
            note=note,
        )


# --------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------- #


# Type aliases for the batch helper. The kline fetcher returns a list
# of (open_ts_ms, high, low, close) tuples — same shape ccxt's
# ``fetch_ohlcv`` produces under the OHLCV mapping (timestamp, open,
# high, low, close, volume). We keep only the fields we need so a
# fixture fetcher in tests is trivial to write.
KlineFetcher = Callable[[str, int, int], Awaitable[list[tuple[int, float, float, float]]]]


@dataclass
class _PostHandle:
    """Trim of :class:`social.binance_square.SquarePost` we need.

    Carrying our own dataclass avoids importing the social package in
    test fixtures (where importing ``aiohttp`` + scraper internals is
    overkill) — any tuple ``(author, ts_ms, intent)`` produced by an
    operator script can be wrapped into one of these.
    """

    author: str
    symbol: str
    ts_ms: int
    intent: KOLIntent
    follower_count: int = 0


def _classify_post_intent(text: str) -> KOLIntent:
    """Cheap keyword-based intent classifier for the batch helper.

    The online path uses ``ai_engine.LLMEngine.judge`` to assign
    ``kol_intent``; the batch helper deliberately does not call into
    the LLM (it would need months of per-post LLM spend to rebuild a
    history file). Instead we use the same hard-coded vocabulary the
    SR-4 system prompt's heuristic uses internally — accurate enough
    for a coarse ``exit_liquidity`` vs ``frontrun_call`` split, and
    cheap to audit.

    Returns ``"neutral"`` when neither set fires; that observation is
    then dropped by :meth:`HistoricalAnalyzer.record_observation`.
    """
    if not text:
        return "neutral"
    t = text.lower()
    exit_words = (
        "take profit", "tp here", "exit", "selling here", "selling now",
        "dump", "rug", "be careful", "watch out", "trap", "exit liquidity",
        "distributing", "distribution",
    )
    call_words = (
        "buy", "long", "moon", "send it", "ape", "going up", "pump",
        "load up", "all in", "100x", "next leg", "breakout", "calling",
    )
    has_exit = any(w in t for w in exit_words)
    has_call = any(w in t for w in call_words)
    if has_exit and not has_call:
        return "exit_liquidity"
    if has_call and not has_exit:
        return "frontrun_call"
    return "neutral"


def _outcome_from_klines(
    klines: list[tuple[int, float, float, float]],
    *,
    pre_close: float,
    pump_threshold: float,
    dump_threshold: float,
) -> tuple[RealisedDirection, float]:
    """Classify a forward-window of klines into a ``RealisedDirection``.

    ``pre_close`` is the close of the bar immediately before the
    KOL post (best proxy for the price the KOL was reacting to). We
    measure the maximum absolute % move from there over the window:

    * If the upper extremum's pct change >= ``pump_threshold`` AND
      it dominates the lower extremum, return ``("pump", +pct)``.
    * Mirror for ``dump_threshold`` -> ``("dump", -pct)``.
    * Else ``("neutral", peak_pct)``.

    ``magnitude_pct`` is signed: positive for a pump, negative for a
    dump, the absolute value for neutral. The store cares only about
    the magnitude; the sign is preserved here for downstream
    inspection / debugging.
    """
    if not klines or pre_close <= 0:
        return "neutral", 0.0
    highs = [float(b[1]) for b in klines]
    lows = [float(b[2]) for b in klines]
    max_up = (max(highs) - pre_close) / pre_close
    max_dn = (pre_close - min(lows)) / pre_close
    if max_up >= pump_threshold and max_up >= max_dn:
        return "pump", max_up
    if max_dn >= dump_threshold and max_dn > max_up:
        return "dump", -max_dn
    # Neutral: report the larger leg as the magnitude (signed).
    if max_up >= max_dn:
        return "neutral", max_up
    return "neutral", -max_dn


async def build_observations_from_posts(
    *,
    posts: Iterable[_PostHandle | dict[str, Any]],
    fetch_klines: KlineFetcher,
    forward_window_sec: int = 4 * 3600,
    pre_window_sec: int = 5 * 60,
    pump_threshold: float = 0.05,
    dump_threshold: float = 0.05,
) -> list[KOLObservation]:
    """Replay a list of posts through a kline fetcher into observations.

    Used by the operator's one-shot CLI script
    (``scripts/rebuild_kol_history.py``, not part of this PR) and by
    integration tests. Each post yields at most one observation:

    1. Fetch klines from ``[post_ts - pre_window_sec, post_ts +
       forward_window_sec]``.
    2. Use the close of the most-recent pre-post bar as the reference
       price.
    3. Classify the forward leg via :func:`_outcome_from_klines`.
    4. Build a :class:`KOLObservation` with the inferred intent +
       realised direction.

    The fetcher is async so a real implementation can use ``ccxt`` or
    ``aiohttp``; tests pass an in-memory fixture. Per-post failures are
    logged and skipped — the function never raises into the loop that
    drives it.

    Returns the in-order list of successfully built observations. The
    caller is expected to feed them into a
    :class:`HistoricalAnalyzer.record_observation` loop (kept separate
    so a partial fetcher failure doesn't leave the store with a
    half-baked snapshot).
    """
    out: list[KOLObservation] = []
    for raw in posts:
        if isinstance(raw, dict):
            try:
                handle = _PostHandle(
                    author=str(raw.get("author") or ""),
                    symbol=str(raw.get("symbol") or ""),
                    ts_ms=int(raw.get("ts_ms") or 0),
                    intent=raw.get("intent") or _classify_post_intent(
                        str(raw.get("text") or ""),
                    ),
                    follower_count=int(raw.get("follower_count") or 0),
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "build_observations: malformed dict post: %s", e,
                )
                continue
        else:
            handle = raw
        if (
            not handle.author
            or not handle.symbol
            or handle.ts_ms <= 0
            or handle.intent == "neutral"
        ):
            continue
        since = handle.ts_ms - pre_window_sec * 1000
        until = handle.ts_ms + forward_window_sec * 1000
        try:
            klines = await fetch_klines(handle.symbol, since, until)
        except Exception as e:
            logger.warning(
                "build_observations: fetch failed for %s @ %s: %s",
                handle.symbol, handle.ts_ms, e,
            )
            continue
        if not klines:
            continue
        # Pre-post and post-post split. ``klines`` is assumed to be
        # ordered ascending by timestamp (ccxt's contract).
        pre_bars = [b for b in klines if b[0] <= handle.ts_ms]
        post_bars = [b for b in klines if b[0] > handle.ts_ms]
        if not pre_bars or not post_bars:
            continue
        pre_close = float(pre_bars[-1][3])
        direction, magnitude = _outcome_from_klines(
            post_bars,
            pre_close=pre_close,
            pump_threshold=pump_threshold,
            dump_threshold=dump_threshold,
        )
        out.append(KOLObservation(
            author=handle.author,
            symbol=handle.symbol,
            intent=handle.intent,
            ts_ms=handle.ts_ms,
            realised_direction=direction,
            magnitude_pct=magnitude,
            follower_count=handle.follower_count,
        ))
    return out


def utc_ms() -> int:
    """Compatibility shim used by the post-mortem callsite."""
    return int(time.time() * 1000)
