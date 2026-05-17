"""pipeline.py — Online wiring for LLM + social + self-evolution.

This module bridges three components that were already built but had not
been connected to the main daemon loop:

    1. Binance Square + auxiliary social aggregator (``social.crawler``)
    2. DeepSeek inference engine (``ai_engine``)
    3. Post-mortem learning loop (``learning_engine``)

Architectural decisions (committed with the architect):

  * Token thrift: only a small subset of rule events ever become LLM
    consults. ``CandidateGate`` enforces:
       - a kind allowlist (only events with real directional conviction),
       - a per-symbol cooldown (default 5min) so a burst of volume on the
         same coin cannot spam the LLM,
       - graceful skip when no API key is configured.

  * Social-source priority: Binance Square is PRIMARY. When the snapshot's
    ``primary_status != "ok"``, the consultor STILL calls the LLM but
    surfaces the degradation via ``extra={"primary_status": ...}`` so
    the system prompt (SR-4) can clamp confidence appropriately. The
    auxiliary OKX / DexScreener / CoinGecko payloads are forwarded ONLY
    when primary is degraded — never as a free confidence boost.

  * Self-evolution closes online: every position that actually OPENS
    schedules a delayed post-mortem (default 1h after open). The
    scheduler tracks tasks so shutdown can cancel them, and never raises
    out of the hot path.

All public entry points here are async and best-effort; they must NEVER
raise into the trading loop. Any failure in social / LLM / learning
degrades the system to the next inner layer (rules-only) but never
stops it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    EngineError,
    SMCContext,
    SocialPost,
)
from altcoin_agent.fuser import ScoreFuser
from altcoin_agent.learning_engine import RuleStore, run_post_mortem
from altcoin_agent.screener import (
    FundingSnapshot,
    OISnapshot,
    SignalEvent,
    SignalKind,
)
from altcoin_agent.social import CookieJar, ProxyConfig
from altcoin_agent.social.crawler import SocialSnapshot, focus_on_symbol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------- #


def cookie_jar_from_env(var: str = "BINANCE_SQUARE_COOKIE") -> CookieJar | None:
    """Build a ``CookieJar`` from a raw 'Cookie:' header in env. Returns
    ``None`` if the var is unset/empty so the scraper degrades to public
    endpoints transparently."""
    raw = os.getenv(var, "").strip()
    if not raw:
        return None
    return CookieJar.from_header_string(raw)


def proxy_config_from_env(
    pool_var: str = "PROXY_POOL", rotate_var: str = "PROXY_ROTATE",
) -> ProxyConfig | None:
    """Comma-separated proxy pool from env, rotation toggle. Returns
    ``None`` when the pool is empty so requests go direct."""
    pool_raw = os.getenv(pool_var, "").strip()
    if not pool_raw:
        return None
    pool = [p.strip() for p in pool_raw.split(",") if p.strip()]
    if not pool:
        return None
    rotate = os.getenv(rotate_var, "true").lower() in ("1", "true", "yes")
    return ProxyConfig(pool=pool, rotate=rotate)


# --------------------------------------------------------------------- #
# RecentSignalsCache
# --------------------------------------------------------------------- #


@dataclass
class _RecentBundle:
    rule_events: deque[SignalEvent] = field(
        default_factory=lambda: deque(maxlen=64),
    )
    funding: FundingSnapshot | None = None
    open_interest: OISnapshot | None = None


@dataclass
class RecentSignalsCache:
    """Per-symbol rolling cache of rule events, funding and OI snapshots.

    The cache is read by ``LLMConsultor`` to reconstruct an ``SMCContext``
    on demand, so we don't have to call back into the screener's internal
    state. ``window_sec`` matches the fuser's signal window so what the
    LLM sees is exactly what the fuser is currently scoring on.
    """

    window_sec: int = 90
    _by_symbol: dict[str, _RecentBundle] = field(default_factory=dict)

    def add_signal(self, ev: SignalEvent) -> None:
        b = self._by_symbol.setdefault(ev.symbol, _RecentBundle())
        b.rule_events.append(ev)

    def add_funding(self, snap: FundingSnapshot) -> None:
        b = self._by_symbol.setdefault(snap.symbol, _RecentBundle())
        b.funding = snap

    def add_oi(self, snap: OISnapshot) -> None:
        b = self._by_symbol.setdefault(snap.symbol, _RecentBundle())
        b.open_interest = snap

    def latest_funding(self, symbol: str) -> FundingSnapshot | None:
        b = self._by_symbol.get(symbol)
        return b.funding if b is not None else None

    def latest_oi(self, symbol: str) -> OISnapshot | None:
        b = self._by_symbol.get(symbol)
        return b.open_interest if b is not None else None

    def latest_funding_zscore(self, symbol: str) -> float | None:
        """Most-recent FUNDING_DEVIATION zscore in the cache, if any."""
        b = self._by_symbol.get(symbol)
        if b is None:
            return None
        for past in reversed(b.rule_events):
            if past.kind == SignalKind.FUNDING_DEVIATION:
                z = past.payload.get("zscore")
                if z is not None:
                    try:
                        return float(z)
                    except (TypeError, ValueError):
                        return None
        return None

    def build_smc_context(self, symbol: str, now_ts: int) -> SMCContext:
        """Roll up recent rule events into the structured bundle the LLM
        prompt expects. Stale events outside ``window_sec`` are filtered out.
        """
        b = self._by_symbol.get(symbol)
        if b is None:
            return SMCContext()
        window_ms = self.window_sec * 1000
        sweeps: list[dict[str, Any]] = []
        pools: list[dict[str, Any]] = []
        volume_spike: dict[str, Any] | None = None
        oi_event: dict[str, Any] | None = None
        # Iterate oldest -> newest so volume_spike / oi_event end up holding
        # the freshest of their kind.
        for ev in b.rule_events:
            if now_ts - ev.ts > window_ms:
                continue
            if ev.kind == SignalKind.LIQUIDITY_SWEEP:
                sweeps.append({"ts": ev.ts, **ev.payload})
            elif ev.kind == SignalKind.LIQUIDITY_POOL_FORMED:
                pools.append({"ts": ev.ts, **ev.payload})
            elif ev.kind == SignalKind.VOLUME_SPIKE:
                volume_spike = {"ts": ev.ts, **ev.payload}
            elif ev.kind in (SignalKind.OI_SURGE, SignalKind.OI_SILENT_BUILD):
                oi_event = {
                    "ts": ev.ts, "kind": ev.kind.value, **ev.payload,
                }
        return SMCContext(
            liquidity_sweeps=sweeps,
            liquidity_pools=pools,
            volume_spike=volume_spike,
            oi_event=oi_event,
        )


# --------------------------------------------------------------------- #
# CandidateGate
# --------------------------------------------------------------------- #


# Kinds with enough directional conviction to justify the cost of an LLM
# consult. Funding events alone are too noisy; LIQUIDITY_POOL_FORMED is
# too early in the SMC sequence to act on.
DEFAULT_CONSULT_KINDS: frozenset[SignalKind] = frozenset({
    SignalKind.OI_SILENT_BUILD,
    SignalKind.OI_SURGE,
    SignalKind.LIQUIDITY_SWEEP,
    SignalKind.VOLUME_SPIKE,
    SignalKind.WASH_TRADING_DETECTED,
})


@dataclass
class CandidateGate:
    """Decides whether a SignalEvent warrants an LLM consult.

    Two filters in order of cost:
      1. kind allowlist (cheapest)
      2. per-symbol cooldown so a burst of fast-fire signals on the same
         symbol never spams the LLM
    """

    cooldown_sec: int = 300
    kinds: frozenset[SignalKind] = DEFAULT_CONSULT_KINDS
    _last_consult_ts: dict[str, int] = field(default_factory=dict)

    def should_consult(self, ev: SignalEvent, now_ts: int | None = None) -> bool:
        if ev.kind not in self.kinds:
            return False
        ts = now_ts if now_ts is not None else ev.ts
        last = self._last_consult_ts.get(ev.symbol, 0)
        return ts - last >= self.cooldown_sec * 1000

    def mark_consulted(self, symbol: str, ts: int) -> None:
        self._last_consult_ts[symbol] = ts


# --------------------------------------------------------------------- #
# LLMConsultor
# --------------------------------------------------------------------- #


SocialFetcher = Callable[[str], Awaitable[SocialSnapshot]]


@dataclass
class LLMConsultor:
    """End-to-end LLM consultation: social → SMC → judge → fuser.

    All exceptions are swallowed: a degraded LLM path must never crash
    the trading bus. The two consequences of failure are:
      - the fuser never sees an LLM verdict for this signal (so it
        scores rules-only, which is the V1.0 default behaviour);
      - the candidate gate is NOT marked consulted, so the next eligible
        event after the cooldown window may try again.
    """

    engine: DeepSeekEngine
    fuser: ScoreFuser
    cache: RecentSignalsCache
    cookies: CookieJar | None = None
    proxy: ProxyConfig | None = None
    social_timeout_sec: float = 8.0
    max_posts: int = 12
    # Test/admin override: when set, replaces the call to focus_on_symbol.
    social_fetcher: SocialFetcher | None = None

    async def consult(self, ev: SignalEvent) -> AIVerdict | None:
        """Run a single consultation. Returns the verdict on success
        (also feeds the fuser internally), or None on any failure path.

        Phase B.6: wraps the consult in a ``llm_consult`` span. Inside
        we further open ``social.fetch`` and ``llm.judge`` child spans
        so a slow third-party (Binance Square cookie expired, OTel
        endpoint unreachable, OpenRouter throttled) is immediately
        attributable. All spans are no-ops when tracing isn't
        configured.
        """
        from altcoin_agent.observability.tracing import start_span

        with start_span(
            "llm_consult",
            attributes={
                "altcoin_agent.symbol": ev.symbol,
                "altcoin_agent.exchange": ev.exchange,
                "altcoin_agent.signal_kind": ev.kind.value,
            },
        ) as consult_span:
            with start_span(
                "social.fetch",
                attributes={"altcoin_agent.symbol": ev.symbol},
                kind="client",
            ) as social_span:
                snapshot = await self._safe_fetch_social(ev.symbol)
                with suppress(Exception):
                    social_span.set_attribute(
                        "altcoin_agent.primary_status", snapshot.primary_status,
                    )
                    social_span.set_attribute(
                        "altcoin_agent.post_count",
                        len(snapshot.binance_square_posts),
                    )
            posts = self._snapshot_to_posts(snapshot)
            smc = self.cache.build_smc_context(ev.symbol, ev.ts)
            funding_snap = self.cache.latest_funding(ev.symbol)
            funding_rate = (
                funding_snap.rate if funding_snap is not None else None
            )
            funding_z = self.cache.latest_funding_zscore(ev.symbol)

            extra: dict[str, Any] = {
                "primary_status": snapshot.primary_status,
                "trigger_kind": ev.kind.value,
                "trigger_payload": ev.payload,
                "post_count": len(posts),
            }
            # When the primary social source is degraded, surface the
            # auxiliary highlights so the LLM still has *something*.
            # Aux sources cannot raise confidence on their own; the
            # system prompt (SR-4) is responsible for clamping.
            if snapshot.primary_status != "ok":
                if snapshot.okx is not None:
                    extra["aux_okx"] = snapshot.okx
                if snapshot.dexscreener is not None:
                    extra["aux_dexscreener"] = snapshot.dexscreener
                if snapshot.coingecko is not None:
                    extra["aux_coingecko"] = snapshot.coingecko

            try:
                with start_span(
                    "llm.judge",
                    attributes={
                        "altcoin_agent.symbol": ev.symbol,
                        "altcoin_agent.post_count": len(posts),
                    },
                    kind="client",
                ) as judge_span:
                    verdict = await self.engine.judge(
                        symbol=ev.symbol,
                        exchange=ev.exchange,
                        funding_rate=funding_rate,
                        funding_deviation_z=funding_z,
                        smc=smc,
                        posts=posts,
                        extra=extra,
                    )
                    with suppress(Exception):
                        judge_span.set_attribute(
                            "altcoin_agent.intent", verdict.intent,
                        )
                        judge_span.set_attribute(
                            "altcoin_agent.confidence_score",
                            int(verdict.confidence_score),
                        )
                        judge_span.set_attribute(
                            "altcoin_agent.kol_intent", verdict.kol_intent,
                        )
            except EngineError as e:
                # No API key, exhausted budget, or similar non-recoverable
                # config error. Log once at warning, return None.
                logger.warning("LLM consult skipped for %s: %s", ev.symbol, e)
                with suppress(Exception):
                    consult_span.set_attribute(
                        "altcoin_agent.skip_reason", str(e),
                    )
                return None
            except Exception as e:
                logger.exception("LLM judge unexpectedly failed for %s: %s",
                                 ev.symbol, e)
                return None

            # Phase B.6 sister deliverable: forward the cited KOL authors
            # to the fuser so its KOL-history adjuster can tilt the
            # ``verdict.confidence`` before the exit_liquidity hard-veto /
            # soft-cap branches fire. We only forward when ``snapshot``
            # actually carried Square posts; otherwise pass an empty list
            # which the fuser interprets as "clear stale authors" so a
            # degraded social call doesn't leave the previous symbol's
            # authors attached to this verdict.
            kol_authors = [
                p.author for p in snapshot.binance_square_posts if p.author
            ]
            try:
                await self.fuser.on_llm_verdict(
                    ev.exchange, ev.symbol, verdict, ev.ts,
                    kol_authors=kol_authors,
                )
            except Exception as e:
                logger.exception("fuser.on_llm_verdict failed for %s: %s",
                                 ev.symbol, e)

            with suppress(Exception):
                consult_span.set_attribute(
                    "altcoin_agent.kol_authors_count", len(kol_authors),
                )
            return verdict

    # -------------- internals -------------- #

    async def _safe_fetch_social(self, symbol: str) -> SocialSnapshot:
        try:
            if self.social_fetcher is not None:
                return await self.social_fetcher(symbol)
            return await focus_on_symbol(
                symbol,
                cookies=self.cookies,
                proxy=self.proxy,
                timeout_sec=self.social_timeout_sec,
            )
        except Exception as e:
            logger.warning("social fetch failed for %s: %s", symbol, e)
            return SocialSnapshot(
                symbol=symbol,
                fetched_at_ts_ms=int(time.time() * 1000),
                primary_status=f"degraded:fetch_error:{type(e).__name__}",
                errors=[f"{type(e).__name__}: {e}"],
            )

    def _snapshot_to_posts(self, snap: SocialSnapshot) -> list[SocialPost]:
        out: list[SocialPost] = []
        for p in snap.binance_square_posts[: self.max_posts]:
            out.append(SocialPost(
                author=p.author,
                follower_count=p.follower_count,
                text=p.text,
                ts=p.ts_ms,
                source=p.source,
            ))
        return out


# --------------------------------------------------------------------- #
# DelayedPostMortemScheduler
# --------------------------------------------------------------------- #


@dataclass
class DelayedPostMortemScheduler:
    """Schedules a learning_engine post-mortem ``delay_sec`` after a
    position opens. Tracks tasks so they can be cancelled on shutdown.

    All exceptions inside the post-mortem are swallowed — the learning
    loop is best-effort and must never poison the trading loop.

    Phase B.6 sister deliverable: when ``historical_analyzer`` is wired,
    each post-mortem also records one KOL observation per cited author
    so the analyzer's hit-rate counters can converge over time. The
    realised direction + magnitude come straight from the post-mortem
    report so the same kline data drives both learning loops.
    """

    store: RuleStore
    engine: DeepSeekEngine | None = None
    delay_sec: int = 3600
    historical_analyzer: Any | None = None
    _tasks: set[asyncio.Task] = field(default_factory=set)

    def schedule(
        self,
        *,
        symbol: str,
        target_ts_ms: int,
        entry_ts_ms: int | None = None,
        expected_direction: str | None = None,
        kol_authors: list[str] | None = None,
        kol_intent: str | None = None,
    ) -> asyncio.Task:
        """Schedule a delayed post-mortem.

        Bug #2 fix: the live post-mortem path now passes ``entry_ts_ms``
        (the moment we actually opened the position) and
        ``expected_direction`` ("pump" for LONG / "dump" for SHORT) so
        ``run_post_mortem`` slices ``[entry - 4h, entry + 1h]``, evaluates
        the move strictly post-entry, and files the rule update under the
        trader's intended direction even when the trade lost.

        Phase B.6: ``kol_authors`` and ``kol_intent`` are the
        analyzer's record keys. Optional — when omitted, only the rule
        store is updated, which is the V1.0 behaviour.
        """
        task = asyncio.create_task(
            self._run(
                symbol=symbol, target_ts_ms=target_ts_ms,
                entry_ts_ms=entry_ts_ms,
                expected_direction=expected_direction,
                kol_authors=list(kol_authors) if kol_authors else None,
                kol_intent=kol_intent,
            ),
            name=f"post_mortem:{symbol}:{target_ts_ms}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(
        self,
        *,
        symbol: str,
        target_ts_ms: int,
        entry_ts_ms: int | None = None,
        expected_direction: str | None = None,
        kol_authors: list[str] | None = None,
        kol_intent: str | None = None,
    ) -> None:
        try:
            await asyncio.sleep(self.delay_sec)
        except asyncio.CancelledError:
            return
        try:
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "target_ts_ms": target_ts_ms,
                "store": self.store,
                "engine": self.engine,
            }
            if entry_ts_ms is not None:
                kwargs["entry_ts_ms"] = entry_ts_ms
            if expected_direction is not None:
                kwargs["expected_direction"] = expected_direction
            report = await run_post_mortem(**kwargs)
            logger.info(
                "post-mortem ran for %s: dir=%s mag=%.4f picks=%s",
                symbol, report.result.direction, report.result.magnitude_pct,
                [(p.feature_name, p.bucket) for p in report.picks],
            )
            # Phase B.6 sister deliverable: feed the same realised
            # outcome to the KOL history analyzer for every author
            # that appeared on the social side at entry time.
            # ``run_post_mortem`` already computed direction +
            # magnitude over the post-entry window; reusing it keeps
            # the two learning loops consistent (no chance of one path
            # seeing a "pump" while the other sees "dump" because they
            # disagree on the bar selection).
            if (
                self.historical_analyzer is not None
                and kol_authors
                and kol_intent in ("frontrun_call", "exit_liquidity")
                and entry_ts_ms is not None
            ):
                # Dedupe by NORMALISED author key: the social snapshot
                # may report the same KOL under multiple surface forms
                # ("@goat", "$goat", "goat"); without normalising here
                # a spammy KOL would inflate their own sample count
                # because :meth:`KOLHistoryStore.record` writes to a
                # single normalised key but our seen set would treat
                # each spelling as fresh.
                from altcoin_agent.social.historical_analyzer import (
                    normalize_author,
                )
                seen: set[str] = set()
                for author in kol_authors:
                    norm = normalize_author(author)
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    try:
                        self.historical_analyzer.record_observation(
                            author=author,
                            symbol=symbol,
                            intent=kol_intent,
                            ts_ms=entry_ts_ms,
                            realised_direction=report.result.direction,
                            magnitude_pct=report.result.magnitude_pct,
                        )
                    except Exception as e:
                        logger.warning(
                            "kol_history record_observation failed for "
                            "%s/%s: %s", symbol, author, e,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("post-mortem failed for %s: %s", symbol, e)

    async def shutdown(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
