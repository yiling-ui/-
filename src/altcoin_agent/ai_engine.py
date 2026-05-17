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

Phase 5 additions (B.5 LLM Pre-Rate / token budget integration)
---------------------------------------------------------------
``LLMEngine`` learned three optional collaborators that, when provided,
implement the plan's "缓存优先 -> 预算优先 -> 真调用" hierarchy without
disturbing the existing rule-only fallback path:

* ``cache: LLMCache``       — keyed by ``(symbol, phase, social_hash)``.
  Cache hits short-circuit the HTTP call entirely (and don't touch the
  legacy ``budget``), which is the source of the plan's 60-80% token
  savings. Misses fall through to the provider call and the result is
  written back so the next ``judge`` for the same context returns 0ms.
* ``budget_manager: TokenBudgetManager`` — replaces the legacy
  per-process ``TokenBudget`` for callers ready to enforce the
  quadrant / mode-tiered policy (FREE -> ECONOMY -> EMERGENCY ->
  FREEZE). When ``can_call_llm`` rejects, we return a synthetic
  neutral verdict instead of paying for a call we'd refuse to act on
  anyway.
* ``quadrant`` / ``signal_score`` — passed through to the budget
  manager. Both default to None so callers without quadrant context
  (e.g. unit tests, debugging consults) keep working unchanged.

The legacy ``TokenBudget`` continues to count successful network
calls so callers that didn't migrate to ``TokenBudgetManager`` still
see the same exhaustion behaviour.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from altcoin_agent.llm.cache import LLMCache
from altcoin_agent.llm.token_budget import TokenBudgetManager
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


@dataclass
class BatchJudgeItem:
    """One symbol's worth of context for :meth:`LLMEngine.judge_batch`.

    Acts as a per-symbol bundle of everything ``judge`` would have
    received as kwargs. The batch entrypoint then merges all items into
    one prompt + one HTTP call. Per-item phase / quadrant / score are
    threaded through to the cache + budget gate so the batch path
    enforces the same policy as a sequence of single calls would.
    """

    symbol: str
    exchange: str
    funding_rate: float | None
    funding_deviation_z: float | None
    smc: "SMCContext"
    posts: list["SocialPost"] = field(default_factory=list)
    extra: dict[str, Any] | None = None
    # Phase 5 routing context (all optional)
    quadrant: str | None = None
    signal_score: float | None = None
    phase: str | None = None


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
# Batch prompt — R5: combine multiple symbols into one HTTP call.
# --------------------------------------------------------------------- #


BATCH_SYSTEM_PROMPT = """You are a senior cryptocurrency derivatives analyst.
You will receive a JSON CONTEXT containing a list of items, one per symbol.
For EACH item, judge whether the next 15-60 minutes is more likely a genuine
pump, a genuine dump, or noise/neutral, applying the same hard rules as the
single-symbol analyst (funding extremes, KOL exit_liquidity penalty, SR-4
bot-spam defense). Items are independent — judge each on its own merits;
do not let a strong signal on one symbol bleed into another.

You MUST respond with ONE single JSON object of EXACTLY the form:

{
  "verdicts": {
    "<symbol_1>": {
      "intent": "pump" | "dump" | "neutral",
      "confidence_score": <integer 0..100>,
      "reason": "<concise english explanation, <= 400 chars>",
      "kol_intent": "frontrun_call" | "exit_liquidity" | "neutral",
      "key_evidence": ["<short bullet>", ...]
    },
    "<symbol_2>": { ... },
    ...
  }
}

Hard rules:
  - The keys of ``verdicts`` MUST match the ``symbol`` of each item.
  - confidence_score MUST be an integer.
  - If unsure on a particular item, output intent=neutral and confidence_score <= 40.
  - The same KOL exit-liquidity / wash-trading / bot-spam rules apply per-item.
  - DO NOT output markdown, code fences, or any text outside the JSON.
"""


def build_batch_user_prompt(items: list["BatchJudgeItem"]) -> str:
    """Encode a list of :class:`BatchJudgeItem` as one CONTEXT JSON.

    Mirrors :func:`build_user_prompt` for shape so the model sees the
    same fields it learned from in single-symbol calls; each per-item
    payload becomes one entry of an ``items`` array.
    """
    payload_items: list[dict[str, Any]] = []
    for it in items:
        item_payload: dict[str, Any] = {
            "symbol": it.symbol,
            "exchange": it.exchange,
            "funding": {
                "current_rate": it.funding_rate,
                "short_window_zscore": it.funding_deviation_z,
            },
            "smc": {
                "liquidity_sweeps": it.smc.liquidity_sweeps,
                "liquidity_pools": it.smc.liquidity_pools,
                "volume_spike": it.smc.volume_spike,
                "oi_event": it.smc.oi_event,
            },
            "social_posts": [
                {
                    "author": p.author,
                    "followers": p.follower_count,
                    "ts": p.ts,
                    "source": p.source,
                    "text": p.truncated(),
                }
                for p in it.posts[:20]
            ],
        }
        if it.extra:
            item_payload["extra"] = it.extra
        if it.phase is not None:
            item_payload["phase"] = it.phase
        if it.quadrant is not None:
            item_payload["quadrant"] = it.quadrant
        payload_items.append(item_payload)

    return (
        "CONTEXT (JSON):\n"
        + json.dumps({"items": payload_items},
                     ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the batch verdicts JSON now."
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

    Phase 5 collaborators (all optional, default None for backward compat):

      cache            : LLMCache for symbol+phase+social_hash dedupe
      budget_manager   : quadrant/tier-aware monthly budget enforcer
                         (replaces or augments the legacy ``budget`` counter)
    """

    provider: LLMProvider | None = None
    timeout: float = 8.0
    max_retries: int = 1
    budget: TokenBudget = field(default_factory=TokenBudget)
    cache: LLMCache | None = None
    budget_manager: TokenBudgetManager | None = None

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
        # ---- Phase 5 additions: all optional, all backward-compat ----
        quadrant: str | None = None,
        signal_score: float | None = None,
        phase: str | None = None,
    ) -> AIVerdict:
        if self.provider is None:
            raise EngineError(
                "No LLM provider configured. Set LLM_PROVIDER and the matching "
                "<BACKEND>_API_KEY env var."
            )

        # ---- Layer 1: cache lookup ---- #
        # The cache is keyed by (symbol, phase, social_hash). A hit
        # returns the previous verdict in 0ms with zero tokens consumed.
        # We skip caching when phase is None (legacy callers) so the
        # behaviour for them is identical to pre-Phase-5.
        cache_key: str | None = None
        if self.cache is not None and phase is not None:
            social_hash = _hash_posts(posts)
            cache_key = LLMCache.make_key(symbol, phase, social_hash)
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    return AIVerdict.model_validate(cached)
                except ValidationError as e:
                    logger.warning(
                        "LLMCache returned malformed verdict for %s; "
                        "ignoring and falling through to LLM. Err=%s",
                        cache_key, e.errors()[:2],
                    )

        # ---- Layer 2: tiered budget gate ---- #
        # If a TokenBudgetManager is wired, ask it before paying. The
        # legacy ``budget`` (per-process counter) still gets the
        # "exhausted" check below as a backstop for callers that
        # haven't migrated.
        if self.budget_manager is not None:
            quadrant_for_gate = quadrant or "D"  # most conservative
            score_for_gate = (
                float(signal_score) if signal_score is not None else 0.0
            )
            allowed, reason = self.budget_manager.can_call_llm(
                quadrant=quadrant_for_gate, signal_score=score_for_gate,
            )
            if not allowed:
                logger.info(
                    "TokenBudgetManager rejected LLM call for %s "
                    "(quadrant=%s score=%.1f reason=%s); returning neutral.",
                    symbol, quadrant_for_gate, score_for_gate, reason,
                )
                return AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason=f"budget_gate:{reason}",
                    kol_intent="neutral", key_evidence=[],
                )

        self.budget.assert_available()

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
                if self.budget_manager is not None:
                    self.budget_manager.record_usage(used_tokens)
                obj = parse_chat_json(raw)
                # Normalize common drifts
                if isinstance(obj.get("confidence_score"), str) and obj["confidence_score"].isdigit():
                    obj["confidence_score"] = int(obj["confidence_score"])
                if isinstance(obj.get("intent"), str):
                    obj["intent"] = obj["intent"].lower()
                if isinstance(obj.get("kol_intent"), str):
                    obj["kol_intent"] = obj["kol_intent"].lower()
                verdict = AIVerdict.model_validate(obj)
                # ---- Layer 1 write-back ---- #
                if self.cache is not None and cache_key is not None:
                    self.cache.put(
                        symbol=symbol,
                        phase=phase or "",
                        social_hash=cache_key.split("|", 2)[2] if "|" in cache_key else "",
                        verdict=verdict.model_dump(),
                        tokens_estimate=used_tokens,
                    )
                return verdict
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

    # ------------------------------------------------------------------ #
    # R5 — batch judge: combine multiple symbols into a single LLM call.
    # ------------------------------------------------------------------ #

    async def judge_batch(
        self, items: list[BatchJudgeItem],
    ) -> dict[str, AIVerdict]:
        """Score a batch of items with one HTTP call (with cache + budget).

        Returns a dict ``{symbol: AIVerdict}`` covering EVERY input
        item — items that miss the cache + fail the budget gate get
        a synthetic neutral verdict; items that fail individual JSON
        parsing also degrade to neutral but never mask successful
        siblings.

        Three-layer hierarchy (matches single-symbol ``judge``):

          1. Cache lookup per item — items whose ``(symbol, phase,
             social_hash)`` is already cached get the previous
             verdict in 0ms with zero tokens consumed.
          2. Budget gate per item — pre-rejects items whose
             quadrant + score wouldn't survive the
             ``TokenBudgetManager``. The budget gate is consulted
             once per item so callers see the same accounting they
             would get from N sequential ``judge`` calls.
          3. Single HTTP call covering only the cache-miss /
             budget-allowed items. Each parsed verdict is written
             back to the cache. Items missing from the model's
             response degrade to neutral with a "missing_in_batch"
             reason rather than raising.

        Empty input is a no-op returning ``{}``. A batch of 1 still
        works — useful for callers that want a uniform interface.
        """
        if not items:
            return {}

        if self.provider is None:
            raise EngineError(
                "No LLM provider configured. Set LLM_PROVIDER and the matching "
                "<BACKEND>_API_KEY env var."
            )

        out: dict[str, AIVerdict] = {}
        # Per-item cache key (None for legacy / no-phase items).
        cache_keys: dict[str, str | None] = {}
        social_hashes: dict[str, str] = {}
        # Items that need a network call.
        pending: list[BatchJudgeItem] = []

        # ---- Layer 1 + 2: cache + budget gate per item ----
        for it in items:
            cache_key: str | None = None
            social_hash = _hash_posts(it.posts)
            social_hashes[it.symbol] = social_hash
            if self.cache is not None and it.phase is not None:
                cache_key = LLMCache.make_key(
                    it.symbol, it.phase, social_hash,
                )
                cached = self.cache.get(cache_key)
                if cached is not None:
                    try:
                        out[it.symbol] = AIVerdict.model_validate(cached)
                        continue
                    except ValidationError as e:
                        logger.warning(
                            "LLMCache returned malformed verdict for %s; "
                            "ignoring. Err=%s", cache_key, e.errors()[:2],
                        )
            cache_keys[it.symbol] = cache_key

            # Budget gate per item.
            if self.budget_manager is not None:
                quadrant_for_gate = it.quadrant or "D"
                score_for_gate = (
                    float(it.signal_score) if it.signal_score is not None
                    else 0.0
                )
                allowed, reason = self.budget_manager.can_call_llm(
                    quadrant=quadrant_for_gate, signal_score=score_for_gate,
                )
                if not allowed:
                    logger.info(
                        "TokenBudgetManager rejected batch item %s "
                        "(quadrant=%s score=%.1f reason=%s); neutral.",
                        it.symbol, quadrant_for_gate,
                        score_for_gate, reason,
                    )
                    out[it.symbol] = AIVerdict(
                        intent="neutral", confidence_score=0,
                        reason=f"budget_gate:{reason}",
                        kol_intent="neutral", key_evidence=[],
                    )
                    continue

            pending.append(it)

        # No network call needed (everything cached or rejected).
        if not pending:
            return out

        # Legacy budget exhaustion gate (defensive, same as judge()).
        try:
            self.budget.assert_available()
        except EngineError:
            for it in pending:
                out[it.symbol] = AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason="budget_exhausted",
                    kol_intent="neutral", key_evidence=[],
                )
            return out

        # ---- Layer 3: one HTTP call ----
        user_prompt = build_batch_user_prompt(pending)
        messages = [
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_err = ""
        raw = ""
        parsed_obj: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw, used_tokens = await self.provider.chat_json(
                    messages, timeout=self.timeout,
                )
                self.budget.add(used_tokens)
                if self.budget_manager is not None:
                    self.budget_manager.record_usage(used_tokens)
                parsed_obj = parse_chat_json(raw)
                break
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = f"json error: {e}"
                logger.warning(
                    "Batch LLM JSON error (attempt %s): %s",
                    attempt + 1, last_err,
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Re-emit ONE valid JSON object exactly matching the "
                        "batch schema. No prose, no markdown."
                    ),
                })
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                last_err = f"http error: {type(e).__name__}: {e}"
                logger.warning(
                    "Batch LLM HTTP error (attempt %s): %s",
                    attempt + 1, last_err,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        if parsed_obj is None:
            logger.error(
                "Batch LLM call failed after %s attempts: %s",
                self.max_retries + 1, last_err,
            )
            for it in pending:
                out[it.symbol] = AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason=f"degraded: {last_err}",
                    kol_intent="neutral", key_evidence=[],
                )
            return out

        verdicts_raw = parsed_obj.get("verdicts")
        if not isinstance(verdicts_raw, dict):
            logger.warning(
                "Batch response missing 'verdicts' object; got keys=%s",
                list(parsed_obj.keys())[:5],
            )
            verdicts_raw = {}

        for it in pending:
            obj = verdicts_raw.get(it.symbol)
            if obj is None:
                out[it.symbol] = AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason="missing_in_batch_response",
                    kol_intent="neutral", key_evidence=[],
                )
                continue
            try:
                # Normalize same as judge().
                if isinstance(obj.get("confidence_score"), str) \
                        and obj["confidence_score"].isdigit():
                    obj["confidence_score"] = int(obj["confidence_score"])
                if isinstance(obj.get("intent"), str):
                    obj["intent"] = obj["intent"].lower()
                if isinstance(obj.get("kol_intent"), str):
                    obj["kol_intent"] = obj["kol_intent"].lower()
                verdict = AIVerdict.model_validate(obj)
            except ValidationError as e:
                logger.warning(
                    "Batch verdict for %s failed validation: %s",
                    it.symbol, e.errors()[:2],
                )
                out[it.symbol] = AIVerdict(
                    intent="neutral", confidence_score=0,
                    reason=f"item_invalid: {e.errors()[:1]}",
                    kol_intent="neutral", key_evidence=[],
                )
                continue
            out[it.symbol] = verdict
            # Write-back cache (per-item) so the next single ``judge``
            # for the same context returns instantly.
            cache_key = cache_keys.get(it.symbol)
            if (
                self.cache is not None
                and cache_key is not None
                and it.phase is not None
            ):
                self.cache.put(
                    symbol=it.symbol,
                    phase=it.phase,
                    social_hash=social_hashes.get(it.symbol, ""),
                    verdict=verdict.model_dump(),
                    tokens_estimate=used_tokens // max(len(pending), 1),
                )

        return out


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


def _hash_posts(posts: list[SocialPost]) -> str:
    """Stable short fingerprint of the social-post bundle.

    The cache key needs to differ when materially-new posts arrive but
    stay identical for an unchanged bundle — even reordering must
    produce the same hash so cosmetic ordering changes don't bust
    the cache. We hash the sorted ``(author, ts, truncated_text)``
    triples and take the first 16 hex chars, which gives 64 bits of
    space — plenty for the bounded cache size (~1024 entries).

    Empty post lists collapse to a fixed sentinel so two signals that
    differ only in zero-post-vs-zero-post ordering still share a key.
    """
    if not posts:
        return "empty"
    triples = sorted(
        (p.author or "", int(p.ts), (p.text or "")[:120])
        for p in posts
    )
    h = hashlib.blake2b(digest_size=8)
    for author, ts, text in triples:
        h.update(author.encode("utf-8", "replace"))
        h.update(b"\x1f")
        h.update(str(ts).encode("ascii"))
        h.update(b"\x1f")
        h.update(text.encode("utf-8", "replace"))
        h.update(b"\x1e")
    return h.hexdigest()
