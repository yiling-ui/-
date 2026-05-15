"""Tests for ai_engine.py (provider-agnostic refactor).

Strategy:
    - Inject a ``FakeProvider`` (satisfies ``LLMProvider``) instead of
      patching httpx — this exercises the same JSON-parse / repair /
      degradation logic deterministically.
    - Verify DeepSeekEngine still picks up DEEPSEEK_API_KEY for back-compat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    EngineError,
    LLMEngine,
    SMCContext,
    SocialPost,
    TokenBudget,
    build_user_prompt,
)
from altcoin_agent.llm_provider import LLMProvider

# --------------------------------------------------------------------------- #
# Fake provider
# --------------------------------------------------------------------------- #


@dataclass
class FakeProvider:
    name: str = "fake"
    model: str = "fake-1"
    responses: list[str | Exception] | None = None
    calls: list[list[dict[str, str]]] | None = None

    def __post_init__(self) -> None:
        if self.responses is None:
            self.responses = []
        if self.calls is None:
            self.calls = []

    async def chat_json(self, messages, *, timeout):  # noqa: ANN001
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise httpx.ReadTimeout("no more mock responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt, 100

    async def aclose(self) -> None:
        return None


def test_fake_provider_satisfies_protocol() -> None:
    p = FakeProvider()
    assert isinstance(p, LLMProvider)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


def _ctx() -> dict[str, Any]:
    return dict(
        symbol="RAVEUSDT",
        exchange="binance",
        funding_rate=-0.0025,
        funding_deviation_z=-3.5,
        smc=SMCContext(
            liquidity_sweeps=[{"side": "sell_side", "level": 1.10, "wick_to_body": 2.1}],
            liquidity_pools=[{"side": "sell_side", "level": 1.10, "touch_count": 2}],
            volume_spike={"zscore": 7.2, "side": "buy"},
            oi_event={"kind": "oi_silent_build", "oi_delta_pct": 0.22},
        ),
        posts=[
            SocialPost(author="@whale", follower_count=120_000, text="$RAVE is loading", ts=1, source="twitter"),
            SocialPost(author="@noob", follower_count=200, text="going to moon", ts=2, source="binance_square"),
        ],
    )


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


def test_build_user_prompt_contains_required_blocks() -> None:
    prompt = build_user_prompt(
        symbol="RAVEUSDT",
        exchange="binance",
        funding_rate=-0.0025,
        funding_deviation_z=-3.0,
        smc=SMCContext(volume_spike={"zscore": 5.0, "side": "buy"}),
        posts=[SocialPost(author="@a", follower_count=10, text="hi", ts=1)],
    )
    assert "RAVEUSDT" in prompt
    assert "binance" in prompt
    assert "funding" in prompt
    assert "social_posts" in prompt
    json_part = prompt.split("CONTEXT (JSON):\n", 1)[1].split("\n\n", 1)[0]
    parsed = json.loads(json_part)
    assert parsed["symbol"] == "RAVEUSDT"
    assert parsed["funding"]["current_rate"] == -0.0025


# --------------------------------------------------------------------------- #
# DeepSeekEngine: api_key + env
# --------------------------------------------------------------------------- #


def test_deepseek_engine_picks_up_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    eng = DeepSeekEngine()
    assert eng.provider is not None
    # The provider holds the key
    assert eng.provider.api_key == "sk-from-env"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_deepseek_engine_raises_engine_error_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    eng = DeepSeekEngine()
    assert eng.provider is None
    with pytest.raises(EngineError, match="DEEPSEEK_API_KEY"):
        await eng.judge(**_ctx())


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_returns_strict_verdict_on_clean_json() -> None:
    body = json.dumps({
        "intent": "pump",
        "confidence_score": 88,
        "reason": "Negative funding, OI silent build, sweep above equal highs.",
        "kol_intent": "frontrun_call",
        "key_evidence": ["funding -0.25%", "OI +22% in 5m", "sweep wick 2.1x body"],
    })
    eng = LLMEngine(provider=FakeProvider(responses=[body]))
    verdict = await eng.judge(**_ctx())
    assert isinstance(verdict, AIVerdict)
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 88
    assert verdict.kol_intent == "frontrun_call"
    assert "funding" in verdict.reason.lower()


@pytest.mark.asyncio
async def test_judge_normalizes_uppercase_intent_and_string_score() -> None:
    body = json.dumps({
        "intent": "PUMP", "confidence_score": "75",
        "reason": "ok", "kol_intent": "NEUTRAL", "key_evidence": [],
    })
    eng = LLMEngine(provider=FakeProvider(responses=[body]))
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 75
    assert verdict.kol_intent == "neutral"


@pytest.mark.asyncio
async def test_judge_handles_code_fenced_response() -> None:
    fenced = "```json\n" + json.dumps(
        {"intent": "neutral", "confidence_score": 30, "reason": "weak signal"}
    ) + "\n```"
    eng = LLMEngine(provider=FakeProvider(responses=[fenced]))
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 30


# --------------------------------------------------------------------------- #
# Repair / degradation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_retries_once_on_invalid_json_then_succeeds() -> None:
    bad = "this is not json at all"
    good = json.dumps({"intent": "dump", "confidence_score": 70, "reason": "ok"})
    fake = FakeProvider(responses=[bad, good])
    eng = LLMEngine(provider=fake, max_retries=1)
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "dump"
    assert verdict.confidence_score == 70

    assert len(fake.calls) == 2
    second_call = fake.calls[1]
    assert any("Re-emit ONE valid JSON" in m["content"]
               for m in second_call if m["role"] == "user")


@pytest.mark.asyncio
async def test_judge_degrades_to_neutral_when_all_retries_fail() -> None:
    fake = FakeProvider(responses=["garbage", "still garbage"])
    eng = LLMEngine(provider=fake, max_retries=1)
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 0
    assert verdict.reason.startswith("degraded:")


@pytest.mark.asyncio
async def test_judge_retries_on_schema_violation() -> None:
    bad_schema = json.dumps({"intent": "moon", "confidence_score": 999, "reason": "x"})
    good = json.dumps({"intent": "pump", "confidence_score": 60, "reason": "ok"})
    eng = LLMEngine(provider=FakeProvider(responses=[bad_schema, good]),
                     max_retries=1)
    v = await eng.judge(**_ctx())
    assert v.intent == "pump"


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_blocks_call_when_exhausted() -> None:
    budget = TokenBudget(monthly_token_limit=10)
    budget.add(20)
    eng = LLMEngine(provider=FakeProvider(responses=["{}"]), budget=budget)
    with pytest.raises(EngineError, match="budget exhausted"):
        await eng.judge(**_ctx())
