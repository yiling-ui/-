"""
crawler.py — Social sentiment aggregator with Binance Square as PRIMARY.

Per the explicit user mandate: Binance Square is the authoritative
source. Every other source (DexScreener, CoinGecko, Gate.io funding,
etc.) is an *auxiliary cross-reference*, never a substitute. Auxiliary
sources should never raise the confidence ceiling on their own; they
exist to corroborate or contradict what Square already shows.

Workflow ("妖币聚焦"):
    1. Screener identifies a candidate coin via volume / OI / funding
       anomalies (Task A).
    2. Risk/Fuser confirms the candidate has potential (Task C).
    3. ``focus_on_symbol(symbol)`` is invoked. It runs for a bounded
       duration (default 30 min) and:
         a. PRIMARY: hits Binance Square via BinanceSquareScraper for
            real posts about this coin.
         b. AUXILIARY: pulls DexScreener / CoinGecko / OKX cross-data.
       This keeps resource usage focused on the live opportunity rather
       than scanning hundreds of coins continuously.

Failure handling:
    - If Binance Square returns geo-block / auth-required, we DO NOT
      silently substitute auxiliary sources for the primary feed.
      Instead the SocialSnapshot reports
      ``primary_status="degraded:<reason>"`` so downstream consumers
      (fuser, risk gate) can take a more cautious stance — the
      architectural rule is: missing Square = lower confidence ceiling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from altcoin_agent.ai_engine import SocialPost
from altcoin_agent.social.binance_square import (
    BinanceSquareScraper,
    CookieJar,
    ProxyConfig,
    ScraperAuthRequired,
    ScraperError,
    ScraperGeoBlocked,
    SquarePost,
    to_social_posts,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
UA = "altcoin-momentum-agent/0.1"


@dataclass
class MarketSnapshot:
    """Live market data pulled from exchange APIs."""

    symbol: str
    funding_rate: float | None = None
    next_funding_rate: float | None = None
    open_interest: float | None = None
    price_usd: float | None = None
    price_change_24h_pct: float | None = None
    volume_24h_usd: float | None = None
    source: str = ""


@dataclass
class SocialSnapshot:
    """Aggregated social intelligence for a single token.

    `primary_status` is the contract by which downstream consumers know
    whether to trust this snapshot:
        "ok"                     -> Binance Square delivered posts
        "degraded:auth_required" -> no logged-in cookie, fell back to
                                    public/auxiliary sources only
        "degraded:geo_blocked"   -> 451 from Binance, proxy needed
        "degraded:no_results"    -> Square reachable but no matching posts
        "degraded:error:...":    -> any other exception
    """

    symbol: str
    primary_status: str = "ok"
    binance_square_posts: list[SquarePost] = field(default_factory=list)
    posts: list[SocialPost] = field(default_factory=list)  # Square + aux merged, ai_engine-ready
    boost_count: int = 0
    boost_amount_total: float = 0.0
    trending_rank: int | None = None
    twitter_url: str | None = None
    telegram_url: str | None = None
    crawled_at: int = 0


# --------------------------------------------------------------------- #
# PRIMARY: Binance Square
# --------------------------------------------------------------------- #


async def fetch_binance_square_posts(
    keyword: str,
    *,
    cookies: CookieJar | None = None,
    proxy: ProxyConfig | None = None,
    limit: int = 30,
) -> tuple[list[SquarePost], str]:
    """Returns (posts, primary_status).

    Wraps BinanceSquareScraper to translate exception types into a
    machine-readable status string.
    """
    try:
        async with BinanceSquareScraper(
            cookies=cookies, proxy=proxy,
        ) as scraper:
            posts = await scraper.fetch_by_keyword(keyword, limit=limit)
            if not posts:
                return [], "degraded:no_results"
            return posts, "ok"
    except ScraperGeoBlocked as e:
        logger.warning("Binance Square geo-blocked: %s", e)
        return [], "degraded:geo_blocked"
    except ScraperAuthRequired as e:
        logger.warning("Binance Square auth-required: %s", e)
        return [], "degraded:auth_required"
    except ScraperError as e:
        logger.warning("Binance Square error: %s", e)
        return [], f"degraded:error:{type(e).__name__}"
    except Exception as e:
        logger.exception("Binance Square unexpected error")
        return [], f"degraded:error:{type(e).__name__}:{e}"


# --------------------------------------------------------------------- #
# AUXILIARY: DexScreener, CoinGecko, OKX, Gate.io
# These corroborate Square; they never substitute for it.
# --------------------------------------------------------------------- #


async def fetch_coingecko_trending(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Returns top trending coins by search volume on CoinGecko."""
    try:
        resp = await client.get("https://api.coingecko.com/api/v3/search/trending")
        resp.raise_for_status()
        return resp.json().get("coins", [])
    except Exception as e:
        logger.warning("CoinGecko trending failed: %s", e)
        return []


async def fetch_dexscreener_boosts(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Returns latest boosted tokens from DexScreener."""
    try:
        resp = await client.get("https://api.dexscreener.com/token-boosts/latest/v1")
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, list) else []
    except Exception as e:
        logger.warning("DexScreener boosts failed: %s", e)
        return []


async def fetch_dexscreener_search(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    """Search DexScreener for a token and return pairs with social info."""
    try:
        resp = await client.get(
            f"https://api.dexscreener.com/latest/dex/search?q={query}"
        )
        resp.raise_for_status()
        return resp.json().get("pairs", [])[:10]
    except Exception as e:
        logger.warning("DexScreener search failed for %s: %s", query, e)
        return []


async def fetch_okx_funding(client: httpx.AsyncClient, inst_id: str) -> MarketSnapshot | None:
    """Fetch real-time funding rate from OKX."""
    try:
        resp = await client.get(
            f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        fr = data[0]
        return MarketSnapshot(
            symbol=inst_id,
            funding_rate=float(fr.get("fundingRate") or 0),
            next_funding_rate=(
                float(fr.get("nextFundingRate") or 0)
                if fr.get("nextFundingRate") else None
            ),
            source="okx",
        )
    except Exception as e:
        logger.warning("OKX funding failed for %s: %s", inst_id, e)
        return None


async def fetch_okx_open_interest(client: httpx.AsyncClient, inst_id: str) -> float | None:
    """Fetch open interest from OKX."""
    try:
        resp = await client.get(
            f"https://www.okx.com/api/v5/public/open-interest"
            f"?instType=SWAP&instId={inst_id}"
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        return float(data[0].get("oi") or 0)
    except Exception as e:
        logger.warning("OKX OI failed for %s: %s", inst_id, e)
        return None


async def fetch_gate_funding(client: httpx.AsyncClient, contract: str) -> list[dict[str, Any]]:
    """Fetch last 3 funding rate samples from Gate.io."""
    try:
        resp = await client.get(
            f"https://api.gateio.ws/api/v4/futures/usdt/funding_rate"
            f"?contract={contract}&limit=3"
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Gate.io funding failed for %s: %s", contract, e)
        return []


# --------------------------------------------------------------------- #
# Auxiliary aggregator (used after primary)
# --------------------------------------------------------------------- #


async def _crawl_auxiliary_for_symbol(
    client: httpx.AsyncClient, symbol: str, snap: SocialSnapshot,
) -> None:
    """Mutate `snap` in place with auxiliary data. Best-effort, never raises."""

    # DexScreener pair search (gives socials + per-DEX price/volume)
    pairs = await fetch_dexscreener_search(client, symbol)
    for pair in pairs:
        info = pair.get("info", {}) or {}
        for s in info.get("socials", []) or []:
            if s.get("type") == "twitter" and not snap.twitter_url:
                snap.twitter_url = s["url"]
            if s.get("type") == "telegram" and not snap.telegram_url:
                snap.telegram_url = s["url"]
        base = pair.get("baseToken", {}) or {}
        if base.get("symbol", "").upper() == symbol.upper():
            vol = (pair.get("volume", {}) or {}).get("h24", 0)
            price_change = (pair.get("priceChange", {}) or {}).get("h24", 0)
            snap.posts.append(SocialPost(
                author=f"dexscreener/{pair.get('dexId', 'unknown')}",
                follower_count=0,
                text=(
                    f"${symbol} on {pair.get('dexId', '?')} "
                    f"({pair.get('chainId', '?')}): "
                    f"price=${pair.get('priceUsd', '?')} "
                    f"vol24h=${vol:.0f} change24h={price_change}%"
                ),
                ts=int(time.time()),
                source="dexscreener",
            ))

    # DexScreener boosts (paid promotion proxy = potential exit liquidity)
    boosts = await fetch_dexscreener_boosts(client)
    snap.boost_count = len(boosts)
    snap.boost_amount_total = float(sum(b.get("amount", 0) or 0 for b in boosts))

    # CoinGecko trending position
    trending = await fetch_coingecko_trending(client)
    for i, coin in enumerate(trending):
        item = coin.get("item", {}) or {}
        if item.get("symbol", "").upper() == symbol.upper():
            snap.trending_rank = i + 1
            snap.posts.append(SocialPost(
                author="coingecko/trending",
                follower_count=0,
                text=(
                    f"${symbol} is #{i + 1} trending on CoinGecko "
                    f"(search volume spike)"
                ),
                ts=int(time.time()),
                source="coingecko_trending",
            ))


# --------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------- #


async def focus_on_symbol(
    symbol: str,
    *,
    okx_inst_id: str | None = None,
    gate_contract: str | None = None,
    binance_cookies: CookieJar | None = None,
    binance_proxy: ProxyConfig | None = None,
    primary_limit: int = 30,
) -> tuple[SocialSnapshot, MarketSnapshot | None]:
    """The 妖币聚焦 entry point.

    Called once a coin has been confirmed as a candidate by the screener
    + fuser. Aggregates Binance Square (PRIMARY) + auxiliary sources +
    market data (OKX/Gate.io). Returns ``(SocialSnapshot, MarketSnapshot)``.

    The snapshot's ``primary_status`` field tells downstream consumers
    whether the Binance Square pull succeeded. They are expected to
    derate confidence when Square is degraded.
    """
    snap = SocialSnapshot(symbol=symbol, crawled_at=int(time.time() * 1000))

    # PRIMARY — Binance Square
    bsq_posts, status = await fetch_binance_square_posts(
        symbol,
        cookies=binance_cookies,
        proxy=binance_proxy,
        limit=primary_limit,
    )
    snap.primary_status = status
    snap.binance_square_posts = bsq_posts
    snap.posts.extend(to_social_posts(bsq_posts))

    # AUXILIARY — Dex / CG / OKX / Gate.io
    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": UA},
    ) as client:
        await _crawl_auxiliary_for_symbol(client, symbol, snap)

        market: MarketSnapshot | None = None
        if okx_inst_id:
            market = await fetch_okx_funding(client, okx_inst_id)
            if market:
                market.open_interest = await fetch_okx_open_interest(
                    client, okx_inst_id,
                )

        if gate_contract:
            gate_rates = await fetch_gate_funding(client, gate_contract)
            for gr in gate_rates[:1]:
                snap.posts.append(SocialPost(
                    author="gate.io/funding",
                    follower_count=0,
                    text=(
                        f"Gate.io {symbol} funding: rate={gr.get('r')} "
                        f"at t={gr.get('t')}"
                    ),
                    ts=int(gr.get("t", time.time())),
                    source="gate_funding",
                ))

    return snap, market


# Backward-compat alias used by the existing demo_e2e_live.py.
crawl_full_context = focus_on_symbol
