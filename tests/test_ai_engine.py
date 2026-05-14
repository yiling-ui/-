"""
Tests for ai_engine.py.

Strategy:
    - Patch DeepSeekEngine._call_api to avoid any HTTP — this also exercises
      the JSON-parsing / repair / degradation logic deterministically.
    - Verify the engine reads DEEPSEEK_API_KEY from env when api_key not given.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    EngineError,
    SMCContext,
    SocialPost,
    TokenBudget,
    build_user_prompt,
)

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


def _patch_api(monkeypatch: pytest.MonkeyPatch, eng: DeepSeekEngine, *responses: str) -> list[list[dict[str, str]]]:
    """Make eng._call_api return successive `responses` strings. Captures msgs."""
    calls: list[list[dict[str, str]]] = []
    queue = list(responses)

    async def fake_call(messages: list[dict[str, str]]) -> tuple[str, int]:
        calls.append([dict(m) for m in messages])
        if not queue:
            raise httpx.ReadTimeout("no more mock responses")
        return queue.pop(0), 100

    monkeypatch.setattr(eng, "_call_api", fake_call)
    return calls


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
    # JSON portion must be parseable
    json_part = prompt.split("CONTEXT (JSON):\n", 1)[1].split("\n\n", 1)[0]
    parsed = json.loads(json_part)
    assert parsed["symbol"] == "RAVEUSDT"
    assert parsed["funding"]["current_rate"] == -0.0025


# --------------------------------------------------------------------------- #
# API key handling
# --------------------------------------------------------------------------- #


def test_engine_picks_up_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    eng = DeepSeekEngine()
    assert eng.api_key == "sk-from-env"


@pytest.mark.asyncio
async def test_engine_raises_engine_error_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    eng = DeepSeekEngine(api_key=None)
    with pytest.raises(EngineError, match="DEEPSEEK_API_KEY"):
        await eng.judge(**_ctx())


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_returns_strict_verdict_on_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = DeepSeekEngine(api_key="sk-test")
    body = json.dumps(
        {
            "intent": "pump",
            "confidence_score": 88,
            "reason": "Negative funding, OI silent build, sweep above equal highs.",
            "kol_intent": "frontrun_call",
            "key_evidence": ["funding -0.25%", "OI +22% in 5m", "sweep wick 2.1x body"],
        }
    )
    _patch_api(monkeypatch, eng, body)

    verdict = await eng.judge(**_ctx())
    assert isinstance(verdict, AIVerdict)
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 88
    assert verdict.kol_intent == "frontrun_call"
    assert "funding" in verdict.reason.lower()


@pytest.mark.asyncio
async def test_judge_normalizes_uppercase_intent_and_string_score(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = DeepSeekEngine(api_key="sk-test")
    body = json.dumps(
        {
            "intent": "PUMP",
            "confidence_score": "75",
            "reason": "ok",
            "kol_intent": "NEUTRAL",
            "key_evidence": [],
        }
    )
    _patch_api(monkeypatch, eng, body)
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 75
    assert verdict.kol_intent == "neutral"


@pytest.mark.asyncio
async def test_judge_handles_code_fenced_response(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = DeepSeekEngine(api_key="sk-test")
    fenced = "```json\n" + json.dumps(
        {"intent": "neutral", "confidence_score": 30, "reason": "weak signal"}
    ) + "\n```"
    _patch_api(monkeypatch, eng, fenced)
    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 30


# --------------------------------------------------------------------------- #
# Repair / degradation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_retries_once_on_invalid_json_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = DeepSeekEngine(api_key="sk-test", max_retries=1)
    bad = "this is not json at all"
    good = json.dumps({"intent": "dump", "confidence_score": 70, "reason": "ok"})
    calls = _patch_api(monkeypatch, eng, bad, good)

    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "dump"
    assert verdict.confidence_score == 70

    assert len(calls) == 2
    # The second call must contain a corrective user message
    second_call_msgs = calls[1]
    assert any("Re-emit ONE valid JSON" in m["content"] for m in second_call_msgs if m["role"] == "user")


@pytest.mark.asyncio
async def test_judge_degrades_to_neutral_when_all_retries_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = DeepSeekEngine(api_key="sk-test", max_retries=1)
    _patch_api(monkeypatch, eng, "garbage", "still garbage")

    verdict = await eng.judge(**_ctx())
    assert verdict.intent == "neutral"
    assert verdict.confidence_score == 0
    assert verdict.reason.startswith("degraded:")


@pytest.mark.asyncio
async def test_judge_retries_on_schema_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = DeepSeekEngine(api_key="sk-test", max_retries=1)
    bad_schema = json.dumps({"intent": "moon", "confidence_score": 999, "reason": "x"})  # invalid intent + score
    good = json.dumps({"intent": "pump", "confidence_score": 60, "reason": "ok"})
    _patch_api(monkeypatch, eng, bad_schema, good)
    v = await eng.judge(**_ctx())
    assert v.intent == "pump"


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_blocks_call_when_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = TokenBudget(monthly_token_limit=10)
    budget.add(20)
    eng = DeepSeekEngine(api_key="sk-test", budget=budget)
    with pytest.raises(EngineError, match="budget exhausted"):
        await eng.judge(**_ctx())
