"""
ai_engine.py — Task B: AI Inference Engine (DeepSeek).

Design points (per requirements.md FR-C1..C4 and design.md §3.3):

* Strict JSON contract enforced by pydantic; on parse failure we retry once
  with a corrective prompt, and on second failure we degrade to a neutral
  verdict instead of crashing the bus.
* Configurable timeout, retries, and an in-process token-budget guard so the
  hot path can never run away with the user's wallet.
* The DeepSeek API key is read from `DEEPSEEK_API_KEY` env var, never from
  config files (security).
* Network IO is encapsulated in `_call_api`, which is the single place to
  monkey-patch in tests (`respx` does this transparently via httpx).

The output schema requested in the user task is:
    {"intent": "pump|dump|neutral", "confidence_score": 0..100, "reason": "..."}

We extend it lightly (kol_intent, key_evidence) to match design.md FR-C3, but
keep `intent` and `confidence_score` at the top level for backward compat with
any downstream consumer asking for the simpler shape.
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

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


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
    """Strict response contract from DeepSeek."""

    intent: Intent
    confidence_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=2000)
    kol_intent: KOLIntent = "neutral"
    key_evidence: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class EngineError(RuntimeError):
    """Raised for non-recoverable engine errors (no API key, budget exhausted)."""


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #


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
    """Build the structured user message. Kept as a pure function for testing."""
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


# --------------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------------- #


@dataclass
class TokenBudget:
    """In-process monthly token budget guard.

    For multi-process deployments this would be backed by Redis (see design.md);
    here we keep it simple and dependency-free for unit tests.
    """

    monthly_token_limit: int = 5_000_000  # ~$200 of deepseek-chat at quoted rates
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
            raise EngineError("monthly LLM token budget exhausted; degrading to rule-only mode")


# --------------------------------------------------------------------------- #
# DeepSeek engine
# --------------------------------------------------------------------------- #


@dataclass
class DeepSeekEngine:
    """
    Thin async wrapper around DeepSeek's OpenAI-compatible chat endpoint.

    Args:
        model: deepseek-chat / deepseek-reasoner. Default: deepseek-chat.
        api_base: override for self-hosted gateway / mock servers.
        api_key: explicit override; if None, read from DEEPSEEK_API_KEY env.
        timeout: per-request seconds.
        max_retries: corrective retries on JSON-parse failure (>=0).
        budget: optional TokenBudget instance.
    """

    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com"
    api_key: str | None = None
    timeout: float = 8.0
    max_retries: int = 1
    temperature: float = 0.2
    budget: TokenBudget = field(default_factory=TokenBudget)

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    # ------------------------- lifecycle ------------------------- #

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=self.timeout,
                headers={"User-Agent": "altcoin-momentum-agent/0.1"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> DeepSeekEngine:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------- public API ------------------------- #

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
        """
        One shot judgement. On any failure path that is *not* a budget /
        config error, returns a degraded neutral verdict with a useful
        ``reason`` instead of raising — the caller (fuser) can then decide
        weight reduction.
        """
        if not self.api_key:
            raise EngineError(
                "DEEPSEEK_API_KEY is not set. Export it or pass api_key=... to DeepSeekEngine."
            )
        self.budget.assert_available()

        user_prompt = build_user_prompt(
            symbol=symbol,
            exchange=exchange,
            funding_rate=funding_rate,
            funding_deviation_z=funding_deviation_z,
            smc=smc,
            posts=posts,
            extra=extra,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_err: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw, used_tokens = await self._call_api(messages)
                self.budget.add(used_tokens)
                return _parse_verdict(raw)
            except ValidationError as e:
                last_err = f"json schema invalid: {e.errors()[:3]}"
                logger.warning("DeepSeek response failed validation (attempt %s): %s", attempt + 1, last_err)
                messages.append({"role": "assistant", "content": raw if "raw" in locals() else ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON or did not match the required schema. "
                            "Re-emit ONE valid JSON object exactly matching the schema, with no surrounding text."
                        ),
                    }
                )
            except json.JSONDecodeError as e:
                last_err = f"json decode error: {e}"
                logger.warning("DeepSeek response not JSON (attempt %s): %s", attempt + 1, last_err)
                messages.append({"role": "assistant", "content": raw if "raw" in locals() else ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Re-emit ONE valid JSON object exactly matching the schema. No prose, no markdown.",
                    }
                )
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                last_err = f"http error: {type(e).__name__}: {e}"
                logger.warning("DeepSeek HTTP error (attempt %s): %s", attempt + 1, last_err)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        # All retries exhausted — degrade gracefully.
        logger.error("DeepSeek inference failed after %s attempts: %s", self.max_retries + 1, last_err)
        return AIVerdict(
            intent="neutral",
            confidence_score=0,
            reason=f"degraded: {last_err}",
            kol_intent="neutral",
            key_evidence=[],
        )

    # ------------------------- transport ------------------------- #

    async def _call_api(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        client = await self._get_client()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        used = int(usage.get("total_tokens") or 0)
        return content, used


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _parse_verdict(raw: str) -> AIVerdict:
    """Parse DeepSeek raw content into AIVerdict.

    DeepSeek with response_format=json_object returns a clean JSON string,
    but real models occasionally wrap it in ```json fences. We tolerate that.
    """
    text = raw.strip()
    if text.startswith("```"):
        # strip ``` and optional language tag
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    obj = json.loads(text)

    # Normalize a couple of common drifts seen from LLMs:
    #   "confidence_score": "88"   -> int
    #   "intent": "PUMP"           -> lower
    if isinstance(obj.get("confidence_score"), str) and obj["confidence_score"].isdigit():
        obj["confidence_score"] = int(obj["confidence_score"])
    if isinstance(obj.get("intent"), str):
        obj["intent"] = obj["intent"].lower()
    if isinstance(obj.get("kol_intent"), str):
        obj["kol_intent"] = obj["kol_intent"].lower()

    return AIVerdict.model_validate(obj)
