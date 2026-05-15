"""
Live probe — BinanceSquareScraper against real Binance endpoints.

This runs the actual scraper from the sandbox (no proxy, no cookies) so
you can see exactly which scenarios it handles gracefully:

    Scenario A: no cookies, no proxy
        - Auth-gated Square endpoints get skipped (no token to send)
        - Public CMS announcements endpoint is attempted
        - Either succeeds (in unblocked region) or raises ScraperGeoBlocked
          / ScraperError on 451 / 202-empty bounce

    Scenario B: focus_on_symbol() — full pipeline
        - Tries Binance Square (PRIMARY) for posts
        - Falls back to auxiliary sources (DexScreener / CoinGecko / OKX)
        - SocialSnapshot.primary_status surfaces the degradation reason

Run:
    python examples/demo_binance_probe.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from altcoin_agent.social.binance_square import (
    BinanceSquareScraper,
    ScraperAuthRequired,
    ScraperError,
    ScraperGeoBlocked,
)
from altcoin_agent.social.crawler import focus_on_symbol


def banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def kv(label: str, value: object) -> None:
    print(f"    {label:.<32}{value}")


async def scenario_a_no_auth() -> None:
    banner("SCENARIO A — no cookies, no proxy: only public CMS reachable")
    print("    Trying BinanceSquareScraper.fetch_by_keyword('BTC') ...\n")
    try:
        async with BinanceSquareScraper(cookies=None, proxy=None) as scraper:
            posts = await scraper.fetch_by_keyword("BTC", limit=5)
            kv("posts returned", len(posts))
            for i, p in enumerate(posts[:3]):
                kv(f"post[{i}].author", p.author)
                kv(f"post[{i}].title", p.title[:80])
                kv(f"post[{i}].source", p.source_endpoint)
                kv(f"post[{i}].url", p.url)
    except ScraperGeoBlocked as e:
        print(f"    EXPECTED IN SANDBOX: ScraperGeoBlocked")
        print(f"      reason: {e}")
        print("    Production fix: pass ProxyConfig(pool=[...residential VPN...]).")
    except ScraperAuthRequired as e:
        print(f"    EXPECTED WITHOUT COOKIES: ScraperAuthRequired")
        print(f"      reason: {e}")
        print("    Production fix: pass CookieJar.from_header_string(<browser-cookie>).")
    except ScraperError as e:
        print(f"    ScraperError: {type(e).__name__}: {e}")


async def scenario_b_announcements_only() -> None:
    banner("SCENARIO B — direct CMS announcements (always-open endpoint)")
    print("    Trying scraper.fetch_announcements() ...\n")
    try:
        async with BinanceSquareScraper() as scraper:
            posts = await scraper.fetch_announcements(page_size=5)
            kv("announcements", len(posts))
            for i, p in enumerate(posts[:3]):
                kv(f"ann[{i}].title", p.title[:80])
                kv(f"ann[{i}].url", p.url[:80])
    except ScraperGeoBlocked as e:
        print(f"    Geo-blocked even on public endpoint: {e}")
        print("    Need a proxy/VPN to get past it.")
    except ScraperError as e:
        print(f"    ScraperError: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"    Unexpected: {type(e).__name__}: {e}")


async def scenario_c_focus_pipeline() -> None:
    banner("SCENARIO C — focus_on_symbol('PEPE'): full primary + auxiliary pipeline")
    print("    This demonstrates the 妖币聚焦 workflow.\n")
    try:
        social, market = await focus_on_symbol(
            symbol="PEPE",
            okx_inst_id="PEPE-USDT-SWAP",
            gate_contract="PEPE_USDT",
        )
        kv("primary_status", social.primary_status)
        kv("binance_square_posts", len(social.binance_square_posts))
        kv("aux posts (DexScreener+CG+Gate)", len(social.posts) - len(social.binance_square_posts))
        kv("twitter_url", social.twitter_url or "(none)")
        kv("trending_rank", social.trending_rank or "(not in top)")
        kv("dex_boosts", social.boost_count)
        if market:
            kv("OKX funding_rate", f"{market.funding_rate:.8f}" if market.funding_rate is not None else "?")
            kv("OKX open_interest", f"{market.open_interest:,.0f}" if market.open_interest else "?")
        else:
            kv("market", "no data")

        if social.primary_status != "ok":
            print()
            print("    >>> Binance Square is DEGRADED in this run.")
            print("    >>> The pipeline still produced a snapshot using auxiliary")
            print("    >>> sources, but downstream consumers should derate")
            print("    >>> confidence per the architectural rule:")
            print("    >>>   missing Square = lower confidence ceiling.")

    except Exception as e:
        print(f"    Unexpected: {type(e).__name__}: {e}")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="    [%(levelname)s %(name)s] %(message)s",
        stream=sys.stderr,
    )
    banner("Binance Square scraper — live probe")
    print("    No cookies, no proxy. Demonstrating the graceful-degradation paths.")
    await scenario_a_no_auth()
    await scenario_b_announcements_only()
    await scenario_c_focus_pipeline()
    banner("DONE")
    print("    To unlock the auth-gated Square endpoints in production:")
    print("      1. Capture cookies from a logged-in browser session at")
    print("         binance.com (csrftoken, bnc-uuid, p20t at minimum).")
    print("      2. Provide them via:")
    print("           CookieJar.from_header_string(<raw cookie header>)")
    print("      3. If your IP is geo-blocked (451), provide a residential")
    print("         VPN as ProxyConfig(pool=['socks5://user:pass@host:port',")
    print("         ...]) — the scraper rotates per-request.")


if __name__ == "__main__":
    asyncio.run(main())
