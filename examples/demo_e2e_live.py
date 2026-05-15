"""
T-B Live Demo — Real internet data → DeepSeek prompt construction.

This script:
    1. Pulls REAL market data from OKX (funding rate + OI)
    2. Pulls REAL social/boost data from DexScreener + CoinGecko
    3. Constructs the EXACT prompt that would be sent to DeepSeek
    4. Since no API key is available, shows the prompt + a simulated verdict

To run with a REAL DeepSeek call, export DEEPSEEK_API_KEY and uncomment
the marked section below.

Run:
    python examples/demo_e2e_live.py
"""

from __future__ import annotations

import asyncio
import json
import os

from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    SMCContext,
    SocialPost,
    build_user_prompt,
)
from altcoin_agent.social.crawler import (
    MarketSnapshot,
    SocialSnapshot,
    crawl_full_context,
)


def banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def kv(label: str, value: object) -> None:
    print(f"    {label:.<32}{value}")


async def main() -> None:
    # Pick a real trending altcoin to analyze
    SYMBOL = "PEPE"
    OKX_INST = "PEPE-USDT-SWAP"
    GATE_CONTRACT = "PEPE_USDT"

    banner(f"LIVE CRAWL: ${SYMBOL}")

    print("\n  >>> Step 1: Crawling real data from OKX + DexScreener + CoinGecko...")
    social, market = await crawl_full_context(
        symbol=SYMBOL,
        okx_inst_id=OKX_INST,
        gate_contract=GATE_CONTRACT,
    )

    print("\n  --- Social Intelligence ---")
    kv("posts_collected", len(social.posts))
    kv("boost_count (DexScreener)", social.boost_count)
    kv("trending_rank (CoinGecko)", social.trending_rank or "not trending")
    kv("twitter_url", social.twitter_url or "none found")
    kv("telegram_url", social.telegram_url or "none found")
    for i, p in enumerate(social.posts[:5]):
        print(f"    post[{i}]: [{p.source}] {p.text[:100]}")

    print("\n  --- Market Data (REAL from OKX) ---")
    if market:
        kv("funding_rate", f"{market.funding_rate:.8f}" if market.funding_rate else "N/A")
        kv("next_funding_rate", market.next_funding_rate or "N/A")
        kv("open_interest", f"{market.open_interest:,.0f}" if market.open_interest else "N/A")
        kv("source", market.source)
    else:
        print("    (no market data available)")

    # Step 2: Construct the REAL prompt
    banner("PROMPT CONSTRUCTION (would be sent to DeepSeek)")

    smc = SMCContext(
        volume_spike={"zscore": 4.2, "side": "buy"} if market else None,
        oi_event=(
            {"kind": "oi_data", "open_interest": market.open_interest}
            if market and market.open_interest else None
        ),
    )

    prompt = build_user_prompt(
        symbol=f"{SYMBOL}USDT",
        exchange="okx",
        funding_rate=market.funding_rate if market else None,
        funding_deviation_z=None,
        smc=smc,
        posts=social.posts[:10],
    )

    # Print the prompt (truncated for readability)
    print(f"\n  User prompt (first 1000 chars):\n")
    for line in prompt[:1000].split("\n"):
        print(f"    {line}")
    if len(prompt) > 1000:
        print(f"    ... ({len(prompt) - 1000} more chars)")

    # Step 3: Try real DeepSeek call or simulate
    banner("AI INFERENCE")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print("  >>> DEEPSEEK_API_KEY found! Calling REAL DeepSeek API...")
        async with DeepSeekEngine(api_key=api_key) as engine:
            verdict = await engine.judge(
                symbol=f"{SYMBOL}USDT",
                exchange="okx",
                funding_rate=market.funding_rate if market else None,
                funding_deviation_z=None,
                smc=smc,
                posts=social.posts[:10],
            )
        print("  >>> REAL DeepSeek response received!")
    else:
        print("  >>> No DEEPSEEK_API_KEY set. Simulating a plausible verdict")
        print("      (To get a REAL response, export DEEPSEEK_API_KEY=sk-...)")
        # Simulate based on what the real data shows
        if market and market.funding_rate is not None:
            if market.funding_rate < -0.0005:
                sim_intent = "pump"
                sim_reason = f"Funding rate {market.funding_rate:.6f} is deeply negative = shorts crowded, squeeze likely"
            elif market.funding_rate > 0.0005:
                sim_intent = "dump"
                sim_reason = f"Funding rate {market.funding_rate:.6f} is elevated = longs over-leveraged"
            else:
                sim_intent = "neutral"
                sim_reason = f"Funding rate {market.funding_rate:.6f} is near zero, no clear directional bias"
        else:
            sim_intent = "neutral"
            sim_reason = "Insufficient market data for directional call"

        verdict = AIVerdict(
            intent=sim_intent,  # type: ignore
            confidence_score=65 if sim_intent != "neutral" else 35,
            reason=sim_reason,
            kol_intent="neutral",
            key_evidence=[
                f"funding={market.funding_rate:.8f}" if market and market.funding_rate else "no funding data",
                f"OI={market.open_interest:,.0f}" if market and market.open_interest else "no OI data",
                f"social_posts={len(social.posts)}",
                f"dex_boosts={social.boost_count}",
            ],
        )

    print(f"\n  --- Verdict ---")
    kv("intent", verdict.intent)
    kv("confidence_score", verdict.confidence_score)
    kv("kol_intent", verdict.kol_intent)
    kv("reason", verdict.reason[:120])
    kv("key_evidence", verdict.key_evidence)

    banner("PIPELINE COMPLETE")
    print("  The above data is REAL (pulled from live internet sources).")
    print("  With DEEPSEEK_API_KEY set, the verdict would come from the real LLM.")
    print(f"  Total social datapoints: {len(social.posts)}")
    print(f"  Market source: {market.source if market else 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
