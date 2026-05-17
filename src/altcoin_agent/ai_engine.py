"""ai_engine.py — AI Inference Engine (provider-agnostic).

Refactored from a DeepSeek-only client to a provider-agnostic engine that
can target DeepSeek / OpenAI / OpenRouter / Moonshot / Qwen / Anthropic /
any OpenAI-compatible endpoint. Set LLM_PROVIDER env to switch.

Design points (per requirements.md FR-C1..C4 and design.md §3.3):

* Strict JSON contract enforced by pydantic; on parse failure we retry once
  with a corrective prompt, and on second failure we degrade to a neutral
  verdict instead of crashing the bus.
* Configurable timeout, retries, and an in-process token-budget guard so the
  hot path can never run away with the user's wallet.
* Provider keys are read from <BACKEND>_API_KEY env vars, never config files.
* Network IO is encapsulated in the LLMProvider, which is the single place
  to monkey-patch in tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from altcoin_agent.llm_provider import (
    LLMProvider,
    OpenAICompatibleProvider,
    build_default_provider,
    parse_chat_json,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Public types (unchanged from V1)
# --------------------------------------------------------------------- #


Intent = Literal["pump", "dump", "neutral"]
KOLIntent = Literal["frontrun_call", "exit_liquidity", "neutral"]


@dataclass
class SocialPost:
    """Normalized post fed into the inference engine."""

    author: str
    follower_count: int
    text: str
    ts: int
    source: str = "binance_square"

    def truncated(self, n: int = 280) -> str:
        return self.text if len(self.text) <= n else self.text[:n] + "..."


@dataclass
class SMCContext:
    """Minimal SMC features bundle. Mirrors what screener.py emits."""

    liquidity_sweeps: list[dict[str, Any]] = field(default_factory=list)
    liquidity_pools: list[dict[str, Any]] = field(default_factory=list)
    volume_spike: dict[str, Any] | None = None
    oi_event: dict[str, Any] | None = None


class AIVerdict(BaseModel):
    """Strict response contract for any LLM provider."""

    intent: Intent
    confidence_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=2000)
    kol_intent: KOLIntent = "neutral"
    key_evidence: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def confidence(self) -> float:
        """0..1 float (= confidence_score / 100)."""
        return self.confidence_score / 100.0


class EngineError(RuntimeError):
    """Raised for non-recoverable engine errors (no key, budget exhausted)."""


# --------------------------------------------------------------------- #
# Prompt template (provider-agnostic)
# --------------------------------------------------------------------- #


SYSTEM_PROMPT = """You are a senior cryptocurrency derivatives analyst specialized in
detecting early-stage altcoin pumps and dumps driven by smart money.

You will be given a JSON CONTEXT containing:
  - symbol and exchange
  - current funding rate and short-term deviation
  - SMC features (liquidity sweeps, liquidity pools, volume spikes, OI changes)
  - a list of recent social posts from Binance Square / KOLs

Your task:
  1. Decide whether the next 15-60 minutes is more likely a genuine pump,
     a genuine dump, or noise/neutral.
  2. Judge whether KOLs in the posts are providing front-run calls (sharing
     genuine early signals) or merely creating exit liquidity (shilling so
     they can dump on followers). Be skeptical of accounts that post only
     after price has already moved.
  3. Combine evidence with funding rate: extreme negative funding favors
     pumps (shorts about to be squeezed); extreme positive funding favors
     dumps (longs over-leveraged). Liquidity sweeps in the trade direction
     add weight.

You MUST respond with ONE single JSON object and NO surrounding text. The
JSON MUST have EXACTLY the following fields:

{
  "intent": "pump" | "dump" | "neutral",
  "confidence_score": <integer 0..100>,
  "reason": "<concise english explanation, <= 400 chars>",
  "kol_intent": "frontrun_call" | "exit_liquidity" | "neutral",
  "key_evidence": ["<short bullet>", "<short bullet>", ...]
}

Hard rules:
  - confidence_score MUST be an integer.
  - If unsure, output intent=neutral and confidence_score <= 40.
  - If KOL accounts are obviously low-follower or post-only-after-pump,
    set kol_intent="exit_liquidity" and lower confidence_score by at least 20.
  - SR-4 BOT-SPAM / SYBIL DEFENSE: if a meaningful share (>=30%) of the social
    posts look like coordinated retail bots — highly homogeneous wording,
    emoji-only or "to the moon"-only content, no original analysis,
    posted within a tight time window from accounts with low follower
    counts — treat the social signal as MANUFACTURED. In that case:
      * lower confidence_score by an additional 15-25,
      * never output intent="pump" with confidence_score >= 70 unless
        market features alone (funding, OI, sweep) independently justify it,
      * if KOLs ALSO appear to be distributing, set kol_intent="exit_liquidity".
    Genuine grass-roots discussion has variance: differing arguments,
    counter-takes, links to charts, varying follower counts. Manufactured
    shilling is uniform.
  - DO NOT output markdown, code fences, or any text outside the JSON.
"""


def build_user_prompt(
    symbol: str,
    exchange: str,
    funding_rate: float | None,
    funding_deviation_z: float | None,
    smc: SMCContext,
    posts: list[SocialPost],
    extra: dict[str, Any] | None = None,
) -> str:
    """Build the structured user message."""
    payload: dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "funding": {
            "current_rate": funding_rate,
            "short_window_zscore": funding_deviation_z,
        },
        "smc": {
            "liquidity_sweeps": smc.liquidity_sweeps,
            "liquidity_pools": smc.liquidity_pools,
            "volume_spike": smc.volume_spike,
            "oi_event": smc.oi_event,
        },
        "social_posts": [
            {
                "author": p.author,
                "followers": p.follower_count,
                "ts": p.ts,
                "source": p.source,
                "text": p.truncated(),
            }
            for p in posts[:20]
        ],
    }
    if extra:
        payload["extra"] = extra
    return (
        "CONTEXT (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the JSON verdict now."
    )


# --------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------- #


@dataclass
class TokenBudget:
    monthly_token_limit: int = 5_000_000
    _used: int = 0
    _month_key: str = ""

    def _current_month(self) -> str:
        t = time.gmtime()
        return f"{t.tm_year}-{t.tm_mon:02d}"

    def add(self, tokens: int) -> None:
        m = self._current_month()
        if m != self._month_key:
            self._month_key = m
            self._used = 0
        self._used += tokens

    def remaining(self) -> int:
        m = self._current_month()
        if m != self._month_key:
            return self.monthly_token_limit
        return max(self.monthly_token_limit - self._used, 0)

    def assert_available(self) -> None:
        if self.remaining() <= 0:
            raise EngineError(
                "monthly LLM token budget exhausted; degrading to rule-only mode"
            )


# --------------------------------------------------------------------- #
# LLMEngine — the provider-agnostic engine
# --------------------------------------------------------------------- #


@dataclass
class LLMEngine:
    """Provider-agnostic AI engine.

    By default (no provider passed), build one from env vars (LLM_PROVIDER,
    <backend>_API_KEY). For tests, inject a fake provider that satisfies
    ``LLMProvider``.

    Audit P-3.1: ``total_budget_sec`` puts a hard upper bound on the
    end-to-end wall-clock cost of a single ``judge()`` call.

    Without it, the worst case is roughly:

        ``timeout`` + ``0.5 * 2^attempt`` backoff * (max_retries) +
        ``timeout`` (retry attempt)

    With the defaults (timeout=8.0, max_retries=1, exponential backoff
    0.5s for attempt #0) that's ~16.5s per consult. The trading hot
    path doesn't await ``judge()`` directly (the LLM lives on a
    separate ``llm_q`` worker), so a slow LLM never deadlocks the
    venue path — but a slow LLM DOES cause queue backpressure that
    can drop legitimate consults.

    With ``total_budget_sec`` we wrap the entire judge() body in
    ``asyncio.wait_for``: when the deadline is exceeded the call
    returns the same neutral degraded ``AIVerdict`` we already return
    on parse / HTTP errors, so the rest of the system (ScoreFuser,
    CandidateGate, post-mortem) sees a uniform contract for "LLM
    didn't help this time". The default of 12.0s was chosen to allow
    one full timeout + one short retry but never the worst-case
    16.5s the back-off math allows.
    """

    provider: LLMProvider | None = None
    timeout: float = 8.0
    max_retries: int = 1
    budget: TokenBudget = field(default_factory=TokenBudget)
    total_budget_sec: float = 12.0

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = build_default_provider()

    @property
    def model(self) -> str:
        return self.provider.model if self.provider is not None else "<none>"

    @property
    def name(self) -> str:
        return self.provider.name if self.provider is not None else "none"

    async def aclose(self) -> None:
        if self.provider is not None:
            await self.provider.aclose()

    async def __aenter__(self) -> LLMEngine:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def judge(
        self,
        *,
        symbol: str,
        exchange: str,
        funding_rate: float | None,
        funding_deviation_z: float | None,
        smc: SMCContext,
        posts: list[SocialPost],
        extra: dict[str, Any] | None = None,
    ) -> AIVerdict:
        if self.provider is None:
            raise EngineError(
                "No LLM provider configured. Set LLM_PROVIDER and the matching "
                "<BACKEND>_API_KEY env var."
            )
        self.budget.assert_available()

        # Audit P-3.1 fix: total deadline. Wrap the inner judge body
        # with ``asyncio.wait_for`` so a slow provider can't blow past
        # the architectural budget for a single consult. On
        # ``TimeoutError`` we return the same neutral verdict the
        # other failure paths return so the contract for "the LLM
        # didn't help this time" stays uniform across error types.
        if self.total_budget_sec is not None and self.total_budget_sec > 0:
            try:
                return await asyncio.wait_for(
                    self._judge_inner(
                        symbol=symbol, exchange=exchange,
                        funding_rate=funding_rate,
                        funding_deviation_z=funding_deviation_z,
                        smc=smc, posts=posts, extra=extra,
                    ),
                    timeout=self.total_budget_sec,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM judge() exceeded total_budget_sec=%.1fs for %s; "
                    "degrading to neutral verdict",
                    self.total_budget_sec, symbol,
                )
                return AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason=(
                        f"degraded: total_budget_exceeded:"
                        f"{self.total_budget_sec:.1f}s"
                    ),
                    kol_intent="neutral", key_evidence=[],
                )
        return await self._judge_inner(
            symbol=symbol, exchange=exchange,
            funding_rate=funding_rate,
            funding_deviation_z=funding_deviation_z,
            smc=smc, posts=posts, extra=extra,
        )

    async def _judge_inner(
        self,
        *,
        symbol: str,
        exchange: str,
        funding_rate: float | None,
        funding_deviation_z: float | None,
        smc: SMCContext,
        posts: list[SocialPost],
        extra: dict[str, Any] | None = None,
    ) -> AIVerdict:
        """The original judge() body, kept as an inner method so the
        outer ``asyncio.wait_for`` can enforce ``total_budget_sec``."""
        assert self.provider is not None  # outer judge() validates this
        user_prompt = build_user_prompt(
            symbol=symbol, exchange=exchange,
            funding_rate=funding_rate,
            funding_deviation_z=funding_deviation_z,
            smc=smc, posts=posts, extra=extra,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_err: str = ""
        raw: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw, used_tokens = await self.provider.chat_json(
                    messages, timeout=self.timeout,
                )
                self.budget.add(used_tokens)
                obj = parse_chat_json(raw)
                # Normalize common drifts
                if isinstance(obj.get("confidence_score"), str) and obj["confidence_score"].isdigit():
                    obj["confidence_score"] = int(obj["confidence_score"])
                if isinstance(obj.get("intent"), str):
                    obj["intent"] = obj["intent"].lower()
                if isinstance(obj.get("kol_intent"), str):
                    obj["kol_intent"] = obj["kol_intent"].lower()
                return AIVerdict.model_validate(obj)
            except ValidationError as e:
                last_err = f"json schema invalid: {e.errors()[:3]}"
                logger.warning("LLM response failed validation (attempt %s): %s",
                                attempt + 1, last_err)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON or did not match "
                        "the required schema. Re-emit ONE valid JSON object "
                        "exactly matching the schema, with no surrounding text."
                    ),
                })
            except json.JSONDecodeError as e:
                last_err = f"json decode error: {e}"
                logger.warning("LLM response not JSON (attempt %s): %s",
                                attempt + 1, last_err)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Re-emit ONE valid JSON object exactly matching the schema. No prose, no markdown.",
                })
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                last_err = f"http error: {type(e).__name__}: {e}"
                logger.warning("LLM HTTP error (attempt %s): %s",
                                attempt + 1, last_err)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        logger.error("LLM inference failed after %s attempts: %s",
                     self.max_retries + 1, last_err)
        return AIVerdict(
            intent="neutral", confidence_score=0,
            reason=f"degraded: {last_err}",
            kol_intent="neutral", key_evidence=[],
        )


# --------------------------------------------------------------------- #
# Backward-compat alias
# --------------------------------------------------------------------- #


@dataclass
class DeepSeekEngine(LLMEngine):
    """Back-compat: defaults to a DeepSeek provider when no provider is set.

    Existing callers that did ``DeepSeekEngine(api_key=...)`` keep working.
    """

    api_key: str | None = None
    api_base: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    temperature: float = 0.2

    def __post_init__(self) -> None:
        # Don't call LLMEngine.__post_init__ (which builds default provider);
        # we want the DeepSeek-shaped behaviour.
        if self.provider is None:
            key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
            if key:
                self.provider = OpenAICompatibleProvider(
                    name="deepseek",
                    api_key=key,
                    api_base=self.api_base,
                    model=self.model_name,
                    supports_json_format=True,
                    temperature=self.temperature,
                )

    async def judge(self, **kwargs: Any) -> AIVerdict:
        if self.provider is None:
            raise EngineError(
                "DEEPSEEK_API_KEY is not set. Export it or pass api_key=... "
                "to DeepSeekEngine."
            )
        return await super().judge(**kwargs)


# --------------------------------------------------------------------- #
# Internal helpers (test-friendly)
# --------------------------------------------------------------------- #


def _parse_verdict(raw: str) -> AIVerdict:
    """Public helper kept for tests that exercise the parsing logic alone."""
    obj = parse_chat_json(raw)
    if isinstance(obj.get("confidence_score"), str) and obj["confidence_score"].isdigit():
        obj["confidence_score"] = int(obj["confidence_score"])
    if isinstance(obj.get("intent"), str):
        obj["intent"] = obj["intent"].lower()
    if isinstance(obj.get("kol_intent"), str):
        obj["kol_intent"] = obj["kol_intent"].lower()
    return AIVerdict.model_validate(obj)
