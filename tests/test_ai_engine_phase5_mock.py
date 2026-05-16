"""tests/test_ai_engine_phase5_mock.py — Phase 5 LLMEngine integration.

Exercises the optional ``cache`` + ``budget_manager`` collaborators
added to ``LLMEngine.judge`` in Phase 5. The legacy single-provider
path (no cache, no budget manager) is already covered by the existing
ai_engine tests; here we focus on the new behaviour:

  1. cache hit short-circuits the network call (zero tokens spent)
  2. cache miss writes the verdict back so the next call hits
  3. budget_manager rejection returns a synthetic neutral verdict
  4. successful calls feed BOTH legacy budget and new budget_manager
  5. _hash_posts is stable + insensitive to ordering
"""

from __future__ import annotations

import json

import pytest

from altcoin_agent.ai_engine import (
    AIVerdict,
    LLMEngine,
    SMCContext,
    SocialPost,
    _hash_posts,
)
from altcoin_agent.llm.cache import LLMCache
from altcoin_agent.llm.token_budget import (
    BudgetMode,
    TokenBudgetManager,
)
from altcoin_agent.llm_provider import LLMProvider

# ----------------------- helpers ----------------------- #


class _RecordingProvider(LLMProvider):
    """Counts ``chat_json`` calls so tests can assert short-circuiting."""

    name = "recording"
    model = "test-model"

    def __init__(self, *, verdict: dict | None = None, tokens: int = 1000):
        self.calls = 0
        self.verdict = verdict or {
            "intent": "pump",
            "confidence_score": 80,
            "reason": "test",
            "kol_intent": "neutral",
            "key_evidence": [],
        }
        self.tokens = tokens

    async def chat_json(self, messages, *, timeout=8.0):
        self.calls += 1
        return json.dumps(self.verdict), self.tokens

    async def aclose(self) -> None:
        return None


def _smc() -> SMCContext:
    return SMCContext()


def _posts(n: int = 1) -> list[SocialPost]:
    return [
        SocialPost(author=f"a{i}", follower_count=100, text=f"text {i}", ts=i)
        for i in range(n)
    ]


def _judge_kwargs(**overrides):
    base = dict(
        symbol="PEPE/USDT:USDT",
        exchange="binance",
        funding_rate=0.0,
        funding_deviation_z=0.0,
        smc=_smc(),
        posts=_posts(),
        extra=None,
    )
    base.update(overrides)
    return base


# ----------------------- _hash_posts ----------------------- #


def test_hash_empty_is_stable_sentinel():
    assert _hash_posts([]) == "empty"


def test_hash_is_order_insensitive():
    a = SocialPost("a", 100, "hi", 1)
    b = SocialPost("b", 200, "yo", 2)
    assert _hash_posts([a, b]) == _hash_posts([b, a])


def test_hash_changes_with_text_change():
    a1 = SocialPost("a", 100, "hi", 1)
    a2 = SocialPost("a", 100, "different", 1)
    assert _hash_posts([a1]) != _hash_posts([a2])


def test_hash_changes_with_ts():
    a1 = SocialPost("a", 100, "hi", 1)
    a2 = SocialPost("a", 100, "hi", 2)
    assert _hash_posts([a1]) != _hash_posts([a2])


def test_hash_does_not_explode_on_unicode():
    p = SocialPost(author="🚀", follower_count=10,
                   text="妖币 to the moon 🌙", ts=1)
    h = _hash_posts([p])
    assert isinstance(h, str) and len(h) == 16


def test_hash_truncates_long_text_consistently():
    # Two posts that differ only after the 120-char prefix MUST hash
    # the same — that's the "cache shouldn't bust on cosmetic edits"
    # contract.
    text_a = "x" * 120 + "AAA"
    text_b = "x" * 120 + "BBB"
    p_a = SocialPost("a", 100, text_a, 1)
    p_b = SocialPost("a", 100, text_b, 1)
    assert _hash_posts([p_a]) == _hash_posts([p_b])


# ----------------------- cache layer ----------------------- #


@pytest.mark.asyncio
async def test_cache_miss_calls_provider_and_writes_back():
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    v = await engine.judge(**_judge_kwargs(phase="ramp"))
    assert isinstance(v, AIVerdict)
    assert provider.calls == 1
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_network():
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    # Warm the cache.
    await engine.judge(**_judge_kwargs(phase="ramp"))
    assert provider.calls == 1

    # Same key (same symbol+phase+post-bundle) → cache hit, no new call.
    await engine.judge(**_judge_kwargs(phase="ramp"))
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_cache_disabled_when_phase_omitted():
    """Legacy callers (no phase=...) must not touch the cache."""
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    await engine.judge(**_judge_kwargs())  # no phase
    assert len(cache) == 0
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_cache_hit_returns_zero_token_charge():
    """A cache hit must not bump the legacy ``budget`` counter."""
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider(tokens=500)
    engine = LLMEngine(provider=provider, cache=cache)

    await engine.judge(**_judge_kwargs(phase="ramp"))
    used_after_first = engine.budget.monthly_token_limit - engine.budget.remaining()
    assert used_after_first == 500

    await engine.judge(**_judge_kwargs(phase="ramp"))  # cache hit
    used_after_second = engine.budget.monthly_token_limit - engine.budget.remaining()
    assert used_after_second == 500  # unchanged


@pytest.mark.asyncio
async def test_different_phase_is_a_different_cache_key():
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    await engine.judge(**_judge_kwargs(phase="ramp"))
    await engine.judge(**_judge_kwargs(phase="parabolic"))
    # Both should miss (different keys) and call the provider.
    assert provider.calls == 2
    assert len(cache) == 2


@pytest.mark.asyncio
async def test_corrupt_cached_verdict_falls_through_to_provider():
    """A malformed cached entry shouldn't crash judge — fall through."""
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    # Inject a bad verdict directly into the cache.
    cache.put(
        symbol="PEPE/USDT:USDT", phase="ramp",
        social_hash=_hash_posts(_posts()),
        verdict={"intent": "garbage", "confidence_score": 9999},
    )

    v = await engine.judge(**_judge_kwargs(phase="ramp"))
    assert isinstance(v, AIVerdict)
    assert provider.calls == 1


# ----------------------- budget_manager layer ----------------------- #


@pytest.mark.asyncio
async def test_budget_manager_blocks_in_freeze_mode():
    """In FREEZE mode every call returns a synthetic neutral verdict."""
    bm = TokenBudgetManager(monthly_budget=100)
    bm.record_usage(100)  # exhaust → freeze
    assert bm.mode() is BudgetMode.FREEZE

    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, budget_manager=bm)

    v = await engine.judge(**_judge_kwargs(quadrant="A", signal_score=99.0))
    assert isinstance(v, AIVerdict)
    assert v.intent == "neutral"
    assert v.confidence_score == 0
    assert "budget_gate" in v.reason
    assert provider.calls == 0  # never reached


@pytest.mark.asyncio
async def test_budget_manager_economy_blocks_d_quadrant():
    bm = TokenBudgetManager(monthly_budget=1000)
    bm.record_usage(600)  # 60% used → economy mode
    assert bm.mode() is BudgetMode.ECONOMY

    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, budget_manager=bm)

    v = await engine.judge(**_judge_kwargs(quadrant="D", signal_score=80))
    assert v.intent == "neutral"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_budget_manager_economy_allows_a_quadrant():
    bm = TokenBudgetManager(monthly_budget=1000)
    bm.record_usage(600)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, budget_manager=bm)

    v = await engine.judge(**_judge_kwargs(quadrant="A", signal_score=80))
    assert v.intent == "pump"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_budget_manager_emergency_requires_high_score():
    bm = TokenBudgetManager(monthly_budget=1000)
    bm.record_usage(900)  # 90% → emergency
    assert bm.mode() is BudgetMode.EMERGENCY

    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, budget_manager=bm)

    # A-quadrant + low score → blocked.
    v_low = await engine.judge(**_judge_kwargs(quadrant="A", signal_score=50))
    assert v_low.intent == "neutral"
    assert provider.calls == 0
    # A-quadrant + high score → allowed.
    v_hi = await engine.judge(**_judge_kwargs(quadrant="A", signal_score=99))
    assert v_hi.intent == "pump"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_budget_manager_records_real_token_usage():
    bm = TokenBudgetManager(monthly_budget=10_000)
    provider = _RecordingProvider(tokens=750)
    engine = LLMEngine(provider=provider, budget_manager=bm)

    await engine.judge(**_judge_kwargs(quadrant="A", signal_score=80))
    assert bm.state.used == 750
    assert bm.state.calls == 1


@pytest.mark.asyncio
async def test_budget_manager_default_quadrant_is_d_for_safety():
    """If the caller forgets to pass quadrant, the gate treats them as
    quadrant D (most conservative). In ECONOMY mode that means
    blocking — the gate should fail closed, not open."""
    bm = TokenBudgetManager(monthly_budget=1000)
    bm.record_usage(600)  # economy mode
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, budget_manager=bm)

    v = await engine.judge(**_judge_kwargs(signal_score=99))  # no quadrant
    assert v.intent == "neutral"
    assert provider.calls == 0


# ----------------------- cache + budget interaction ----------------------- #


@pytest.mark.asyncio
async def test_cache_hit_bypasses_budget_check():
    """A cache hit returns BEFORE the budget gate runs — even FREEZE
    can serve cached verdicts (we already paid for them)."""
    cache = LLMCache(now_fn=lambda: 100.0)
    provider = _RecordingProvider()
    engine = LLMEngine(provider=provider, cache=cache)

    # Warm the cache (no budget manager → no gating).
    v_first = await engine.judge(**_judge_kwargs(phase="ramp"))
    assert provider.calls == 1

    # Now attach a frozen budget manager and call again.
    bm = TokenBudgetManager(monthly_budget=10)
    bm.record_usage(10)
    engine.budget_manager = bm

    v_cached = await engine.judge(
        **_judge_kwargs(phase="ramp", quadrant="A", signal_score=99)
    )
    assert v_cached.intent == v_first.intent
    assert v_cached.confidence_score == v_first.confidence_score
    assert provider.calls == 1  # unchanged: cache served
