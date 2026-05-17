"""R5 — LLMEngine.judge_batch tests.

We stub the provider so no network is touched. The fake returns a
canned JSON string + a synthetic token usage. Tests confirm:

  * empty input is a no-op
  * batch covers all symbols, even when the provider response is
    missing one (degrades to neutral)
  * cache hits short-circuit per-symbol so cached items don't
    re-enter the prompt
  * budget-manager rejection per-item produces a synthetic neutral
    verdict and excludes the item from the network call
  * malformed individual verdict object degrades to neutral without
    masking sibling successes
  * provider HTTP error degrades all *pending* items to neutral but
    still returns cache hits for already-cached items
  * write-back cache is populated for cache-miss items with phase
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from altcoin_agent.ai_engine import (
    AIVerdict,
    BatchJudgeItem,
    LLMEngine,
    SMCContext,
    SocialPost,
    build_batch_user_prompt,
)
from altcoin_agent.llm.cache import LLMCache


# --------------------------------------------------------------------- #
# Fake provider — implements LLMProvider Protocol shape
# --------------------------------------------------------------------- #


@dataclass
class FakeProvider:
    response_json: str = '{"verdicts": {}}'
    tokens: int = 1000
    raise_exc: Exception | None = None
    name: str = "fake"
    model: str = "fake-model"
    calls: list[list[dict]] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    async def chat_json(self, messages, *, timeout):
        self.calls.append(list(messages))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response_json, self.tokens

    async def aclose(self):
        pass


def _verdict_dict(intent="pump", score=80, kol="neutral"):
    return {
        "intent": intent,
        "confidence_score": score,
        "reason": "synthetic test verdict",
        "kol_intent": kol,
        "key_evidence": ["v"],
    }


def _make_item(symbol: str, *, phase: str | None = None,
               quadrant: str | None = None,
               signal_score: float | None = None) -> BatchJudgeItem:
    return BatchJudgeItem(
        symbol=symbol, exchange="binance",
        funding_rate=0.0, funding_deviation_z=0.0,
        smc=SMCContext(),
        posts=[SocialPost(author="a", follower_count=1, text="x", ts=1)],
        phase=phase, quadrant=quadrant, signal_score=signal_score,
    )


# --------------------------------------------------------------------- #
# Empty + smoke
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_batch_empty_returns_empty():
    eng = LLMEngine(provider=FakeProvider())
    result = await eng.judge_batch([])
    assert result == {}


@pytest.mark.asyncio
async def test_judge_batch_single_symbol_smoke():
    payload = {"verdicts": {"PEPE/USDT:USDT": _verdict_dict("pump", 75)}}
    provider = FakeProvider(response_json=json.dumps(payload))
    eng = LLMEngine(provider=provider)
    result = await eng.judge_batch([_make_item("PEPE/USDT:USDT")])
    assert "PEPE/USDT:USDT" in result
    assert result["PEPE/USDT:USDT"].intent == "pump"
    assert len(provider.calls) == 1


# --------------------------------------------------------------------- #
# Multi-symbol coverage
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_batch_returns_one_verdict_per_item():
    payload = {
        "verdicts": {
            "A/USDT:USDT": _verdict_dict("pump", 80),
            "B/USDT:USDT": _verdict_dict("dump", 60),
            "C/USDT:USDT": _verdict_dict("neutral", 30),
        }
    }
    provider = FakeProvider(response_json=json.dumps(payload))
    eng = LLMEngine(provider=provider)
    items = [
        _make_item("A/USDT:USDT"),
        _make_item("B/USDT:USDT"),
        _make_item("C/USDT:USDT"),
    ]
    result = await eng.judge_batch(items)
    assert set(result) == {"A/USDT:USDT", "B/USDT:USDT", "C/USDT:USDT"}
    assert result["A/USDT:USDT"].intent == "pump"
    assert result["B/USDT:USDT"].intent == "dump"
    # ONE network call total (the whole point of batch).
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_judge_batch_missing_symbol_degrades_to_neutral():
    """Provider response only covers A; B is silently dropped -> neutral."""
    payload = {"verdicts": {"A/USDT:USDT": _verdict_dict("pump", 80)}}
    provider = FakeProvider(response_json=json.dumps(payload))
    eng = LLMEngine(provider=provider)
    items = [_make_item("A/USDT:USDT"), _make_item("B/USDT:USDT")]
    result = await eng.judge_batch(items)
    assert result["A/USDT:USDT"].intent == "pump"
    assert result["B/USDT:USDT"].intent == "neutral"
    assert "missing_in_batch" in result["B/USDT:USDT"].reason


@pytest.mark.asyncio
async def test_judge_batch_malformed_item_degrades_only_that_one():
    payload = {
        "verdicts": {
            "A/USDT:USDT": _verdict_dict("pump", 80),
            "B/USDT:USDT": {"intent": "pump"},  # missing required fields
        }
    }
    provider = FakeProvider(response_json=json.dumps(payload))
    eng = LLMEngine(provider=provider)
    result = await eng.judge_batch([
        _make_item("A/USDT:USDT"),
        _make_item("B/USDT:USDT"),
    ])
    assert result["A/USDT:USDT"].intent == "pump"
    assert result["B/USDT:USDT"].intent == "neutral"
    assert "item_invalid" in result["B/USDT:USDT"].reason


# --------------------------------------------------------------------- #
# Cache integration
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_batch_cache_hit_skips_network(tmp_path):
    cache = LLMCache(state_path=str(tmp_path / "cache.json"), max_entries=10)
    # Build a provider that returns ONLY B in its batch response. We
    # populate A by running judge_batch once first; then invoke again
    # and assert A hits while B misses.
    payload = {"verdicts": {"A/USDT:USDT": _verdict_dict("pump", 90)}}
    provider = FakeProvider(response_json=json.dumps(payload))
    eng = LLMEngine(provider=provider, cache=cache)

    item_a = _make_item("A/USDT:USDT", phase="ramp")
    item_b = _make_item("B/USDT:USDT", phase="ramp")

    # First call: only A in items -> network, populates cache for A.
    await eng.judge_batch([item_a])
    assert len(provider.calls) == 1

    # Second call: A is now a cache hit, B miss.
    payload2 = {"verdicts": {"B/USDT:USDT": _verdict_dict("dump", 70)}}
    provider.response_json = json.dumps(payload2)
    provider.calls.clear()
    result = await eng.judge_batch([item_a, item_b])

    assert result["A/USDT:USDT"].intent == "pump"
    assert result["B/USDT:USDT"].intent == "dump"
    # Exactly ONE network call covering only B.
    assert len(provider.calls) == 1
    user_msg = provider.calls[0][1]["content"]
    assert "B/USDT:USDT" in user_msg
    assert "A/USDT:USDT" not in user_msg


# --------------------------------------------------------------------- #
# Budget manager integration
# --------------------------------------------------------------------- #


class FakeBudgetManager:
    """Stub TokenBudgetManager: rejects items in ``reject_quadrants``."""

    def __init__(self, reject_quadrants: set[str] | None = None):
        self.reject_quadrants = reject_quadrants or set()
        self.usage_recorded: list[int] = []

    def can_call_llm(self, *, quadrant: str, signal_score: float):
        if quadrant in self.reject_quadrants:
            return False, f"freeze_{quadrant}"
        return True, ""

    def record_usage(self, n: int) -> None:
        self.usage_recorded.append(n)


@pytest.mark.asyncio
async def test_judge_batch_budget_rejects_excludes_from_network():
    payload = {"verdicts": {"A/USDT:USDT": _verdict_dict("pump", 80)}}
    provider = FakeProvider(response_json=json.dumps(payload))
    budget = FakeBudgetManager(reject_quadrants={"D"})
    eng = LLMEngine(provider=provider, budget_manager=budget)
    items = [
        _make_item("A/USDT:USDT", quadrant="A", signal_score=80.0),
        _make_item("D/USDT:USDT", quadrant="D", signal_score=20.0),
    ]
    result = await eng.judge_batch(items)
    assert result["A/USDT:USDT"].intent == "pump"
    assert result["D/USDT:USDT"].intent == "neutral"
    assert "budget_gate" in result["D/USDT:USDT"].reason
    # The user prompt only includes A.
    user_msg = provider.calls[0][1]["content"]
    assert "A/USDT:USDT" in user_msg
    assert "D/USDT:USDT" not in user_msg


# --------------------------------------------------------------------- #
# Network failure
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_batch_http_error_degrades_pending_to_neutral():
    provider = FakeProvider(
        raise_exc=httpx.ConnectError("boom"),
    )
    eng = LLMEngine(provider=provider, max_retries=0)
    items = [_make_item("A/USDT:USDT"), _make_item("B/USDT:USDT")]
    result = await eng.judge_batch(items)
    assert result["A/USDT:USDT"].intent == "neutral"
    assert result["B/USDT:USDT"].intent == "neutral"
    assert "degraded" in result["A/USDT:USDT"].reason


# --------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------- #


def test_build_batch_user_prompt_serializes_all_items():
    items = [
        _make_item("A/USDT:USDT", phase="ramp", quadrant="A"),
        _make_item("B/USDT:USDT", phase="parabolic"),
    ]
    text = build_batch_user_prompt(items)
    # Both symbols present, the JSON is parseable.
    assert "A/USDT:USDT" in text
    assert "B/USDT:USDT" in text
    # Strip the leading "CONTEXT (JSON):\n" + trailing instructions.
    body = text.split("CONTEXT (JSON):\n", 1)[1].split("\n\n", 1)[0]
    payload = json.loads(body)
    assert "items" in payload
    assert {it["symbol"] for it in payload["items"]} == {
        "A/USDT:USDT", "B/USDT:USDT",
    }
    # Phase + quadrant get propagated.
    by_sym = {it["symbol"]: it for it in payload["items"]}
    assert by_sym["A/USDT:USDT"]["phase"] == "ramp"
    assert by_sym["A/USDT:USDT"]["quadrant"] == "A"
