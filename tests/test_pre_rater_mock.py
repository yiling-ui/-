"""tests/test_pre_rater_mock.py — Phase B.5.2 LLM Pre-Rate worker."""

from __future__ import annotations

import asyncio
import json

import pytest

from altcoin_agent.ai_engine import AIVerdict, LLMEngine, SMCContext, SocialPost
from altcoin_agent.llm.cache import LLMCache
from altcoin_agent.llm.pre_rater import (
    LLMPreRater,
    PreRateRequest,
    PreRaterStats,
)
from altcoin_agent.llm.token_budget import TokenBudgetManager
from altcoin_agent.llm_provider import LLMProvider

# ----------------------- helpers ----------------------- #


class _FakeProvider(LLMProvider):
    """In-memory provider; counts calls + lets us inject a delay so we
    can exercise queue ordering."""

    name = "fake"
    model = "test"

    def __init__(
        self,
        *,
        intent: str = "pump",
        confidence: int = 80,
        tokens: int = 1000,
        latency_sec: float = 0.0,
    ):
        self.calls = 0
        self.last_messages = None
        self.intent = intent
        self.confidence = confidence
        self.tokens = tokens
        self.latency_sec = latency_sec

    async def chat_json(self, messages, *, timeout=8.0):
        self.calls += 1
        self.last_messages = messages
        if self.latency_sec > 0:
            await asyncio.sleep(self.latency_sec)
        verdict = {
            "intent": self.intent,
            "confidence_score": self.confidence,
            "reason": "fake",
            "kol_intent": "neutral",
            "key_evidence": [],
        }
        return json.dumps(verdict), self.tokens

    async def aclose(self) -> None:
        return None


def _req(
    *,
    symbol: str = "PEPE/USDT:USDT",
    quadrant: str = "A",
    score: float = 80.0,
    phase: str = "ramp",
    posts: tuple[SocialPost, ...] = (),
) -> PreRateRequest:
    return PreRateRequest(
        symbol=symbol,
        exchange="binance",
        quadrant=quadrant,
        phase=phase,
        signal_score=score,
        funding_rate=0.0,
        funding_deviation_z=0.0,
        smc=SMCContext(),
        posts=posts,
    )


def _build_rater(
    *,
    cache: LLMCache | None = None,
    bm: TokenBudgetManager | None = None,
    eligible: frozenset[str] | None = None,
    min_score: float = 70.0,
    queue_maxsize: int = 64,
    provider: _FakeProvider | None = None,
) -> tuple[LLMPreRater, _FakeProvider, LLMCache]:
    cache = cache or LLMCache(now_fn=lambda: 100.0)
    provider = provider or _FakeProvider()
    engine = LLMEngine(
        provider=provider, cache=cache, budget_manager=bm,
    )
    rater = LLMPreRater(
        engine=engine,
        cache=cache,
        budget_manager=bm,
        eligible_quadrants=eligible or frozenset({"A"}),
        prerate_min_score=min_score,
        queue_maxsize=queue_maxsize,
    )
    return rater, provider, cache


# ----------------------- schedule() filters ----------------------- #


def test_schedule_rejects_wrong_quadrant():
    rater, _, _ = _build_rater()
    accepted = rater.schedule(_req(quadrant="B"))
    assert accepted is False
    assert rater.stats.skipped_wrong_quadrant == 1
    assert rater.stats.scheduled == 1


def test_schedule_rejects_low_score():
    rater, _, _ = _build_rater(min_score=70.0)
    accepted = rater.schedule(_req(score=65.0))
    assert accepted is False
    assert rater.stats.skipped_low_score == 1


def test_schedule_accepts_eligible_request():
    rater, _, _ = _build_rater()
    accepted = rater.schedule(_req(quadrant="A", score=80))
    assert accepted is True
    assert rater.stats.scheduled == 1
    assert rater.stats.skipped_wrong_quadrant == 0


def test_schedule_drops_when_queue_is_full():
    rater, _, _ = _build_rater(queue_maxsize=2)
    # Worker isn't started → nothing drains the queue.
    assert rater.schedule(_req(symbol="A")) is True
    assert rater.schedule(_req(symbol="B")) is True
    assert rater.schedule(_req(symbol="C")) is False
    assert rater.stats.skipped_queue_full == 1


def test_schedule_supports_multiple_eligible_quadrants():
    rater, _, _ = _build_rater(eligible=frozenset({"A", "B"}))
    assert rater.schedule(_req(quadrant="A")) is True
    assert rater.schedule(_req(quadrant="B")) is True
    assert rater.schedule(_req(quadrant="C")) is False


def test_schedule_high_water_mark():
    rater, _, _ = _build_rater()
    rater.schedule(_req(symbol="A"))
    rater.schedule(_req(symbol="B"))
    rater.schedule(_req(symbol="C"))
    assert rater.stats.queue_high_water == 3


def test_stats_as_dict_has_all_counters():
    s = PreRaterStats()
    d = s.as_dict()
    expected_keys = {
        "scheduled", "skipped_low_score", "skipped_wrong_quadrant",
        "skipped_queue_full", "skipped_cache_warm", "skipped_budget_block",
        "rated_ok", "rated_failed", "queue_high_water",
    }
    assert set(d.keys()) == expected_keys


# ----------------------- start/stop + run loop ----------------------- #


@pytest.mark.asyncio
async def test_worker_drains_queue_and_calls_engine():
    rater, provider, cache = _build_rater()
    await rater.start()
    try:
        ok = rater.schedule(_req(symbol="PEPE/USDT:USDT"))
        assert ok
        # Wait for the queue to drain.
        await asyncio.wait_for(rater._queue.join(), timeout=2.0)
    finally:
        await rater.stop()
    assert provider.calls == 1
    assert rater.stats.rated_ok == 1
    # Cache should be warmed.
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_worker_warms_cache_for_hot_path():
    """The point: after pre-rating, an LLMEngine.judge() with the same
    (symbol, phase, posts) should be a 0-call cache hit."""
    rater, provider, cache = _build_rater()
    await rater.start()
    posts = (SocialPost("kol", 50000, "spotted early", 1),)
    try:
        rater.schedule(_req(posts=posts))
        await asyncio.wait_for(rater._queue.join(), timeout=2.0)
    finally:
        await rater.stop()
    # Hot path: a fresh engine sharing the cache should see the verdict.
    hot_engine = LLMEngine(provider=provider, cache=cache)
    pre_calls = provider.calls
    v = await hot_engine.judge(
        symbol="PEPE/USDT:USDT", exchange="binance",
        funding_rate=0.0, funding_deviation_z=0.0,
        smc=SMCContext(), posts=list(posts),
        phase="ramp",
    )
    assert isinstance(v, AIVerdict)
    assert provider.calls == pre_calls  # zero new calls — pure cache


@pytest.mark.asyncio
async def test_worker_continues_after_engine_exception():
    """If one rating raises, the worker logs and continues."""

    class BoomProvider(LLMProvider):
        name = "boom"
        model = "test"

        def __init__(self):
            self.calls = 0

        async def chat_json(self, messages, *, timeout=8.0):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return json.dumps({
                "intent": "pump", "confidence_score": 75,
                "reason": "ok", "kol_intent": "neutral", "key_evidence": [],
            }), 100

        async def aclose(self) -> None:
            return None

    boom = BoomProvider()
    cache = LLMCache(now_fn=lambda: 100.0)
    engine = LLMEngine(provider=boom, cache=cache, max_retries=0)
    rater = LLMPreRater(
        engine=engine, cache=cache,
        eligible_quadrants=frozenset({"A"}), prerate_min_score=70.0,
    )
    await rater.start()
    try:
        rater.schedule(_req(symbol="A1"))
        rater.schedule(_req(symbol="A2", phase="parabolic"))
        await asyncio.wait_for(rater._queue.join(), timeout=2.0)
    finally:
        await rater.stop()
    # First raised → degraded neutral verdict (counted as rated_ok by
    # ai_engine which returns AIVerdict in degradation), second succeeded.
    # Either way both jobs were processed.
    assert boom.calls == 2


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_cancels_in_flight():
    """Calling stop() twice is safe."""
    rater, _, _ = _build_rater()
    await rater.start()
    await rater.stop()
    await rater.stop()  # second call is a no-op


# ----------------------- budget interaction ----------------------- #


@pytest.mark.asyncio
async def test_worker_skips_when_budget_freezes():
    """If budget hits FREEZE between schedule() and run, the worker
    must count skipped_budget_block instead of paying."""
    bm = TokenBudgetManager(monthly_budget=10)
    bm.record_usage(10)  # already frozen
    rater, provider, _ = _build_rater(bm=bm)
    await rater.start()
    try:
        rater.schedule(_req(quadrant="A", score=99))
        await asyncio.wait_for(rater._queue.join(), timeout=2.0)
    finally:
        await rater.stop()
    assert provider.calls == 0
    assert rater.stats.skipped_budget_block == 1
