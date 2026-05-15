"""crawler.py — Multi-source social aggregator.

Workflow (committed with the user):
  1. The market screener decides a coin is "妖币 candidate".
  2. ONLY THEN do we call `focus_on_symbol(symbol)` to spend network on
     social/auxiliary data. This conserves resources.
  3. Binance Square is the PRIMARY source; everything else is auxiliary
     and CANNOT raise the confidence ceiling on its own.
  4. `primary_status` exposes WHY Binance Square data is missing if it is,
     so downstream consumers can clamp confidence appropriately.

Auxiliary sources (best-effort, public, no-auth):
  - OKX  (current funding + OI for cross-validation)
  - DexScreener (token social-link metadata + paid boost activity)
  - CoinGecko (search-trending position; a proxy for retail attention)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from altcoin_agent.social.binance_square import (
    BinanceSquareScraper,
    CookieJar,
    ProxyConfig,
    ScraperAuthRequired,
    ScraperError,
    ScraperGeoBlocked,
    SquarePost,
)

logger = logging.getLogger(__name__)


@dataclass
class SocialSnapshot:
    symbol: str
    fetched_at_ts_ms: int
    primary_status: str = "ok"           # "ok" | "degraded:auth_required" | "degraded:geo_blocked" | "degraded:no_results" | "degraded:error:..."
    binance_square_posts: list[SquarePost] = field(default_factory=list)
    okx: dict[str, Any] | None = None
    dexscreener: dict[str, Any] | None = None
    coingecko: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def primary_ok(self) -> bool:
        return self.primary_status == "ok"

    def auxiliary_only(self) -> bool:
        return not self.primary_ok and (self.okx or self.dexscreener or self.coingecko)


async def focus_on_symbol(
    symbol: str,
    *,
    cookies: CookieJar | None = None,
    proxy: ProxyConfig | None = None,
    timeout_sec: float = 10.0,
) -> SocialSnapshot:
    """Focused fetch — only call AFTER the market screener has flagged the
    symbol as a candidate. Pulls Binance Square first (primary), then
    auxiliary sources in parallel. Always returns a snapshot, even on
    primary failure (degraded mode)."""
    snap = SocialSnapshot(symbol=symbol, fetched_at_ts_ms=int(time.time() * 1000))

    # ---- PRIMARY: Binance Square ----
    scraper = BinanceSquareScraper(cookies=cookies, proxy=proxy, request_timeout_sec=timeout_sec)
    try:
        posts = await scraper.fetch_by_keyword(symbol)
        snap.binance_square_posts = posts
        if not posts:
            snap.primary_status = "degraded:no_results"
        else:
            snap.primary_status = "ok"
    except ScraperAuthRequired as e:
        snap.primary_status = "degraded:auth_required"
        snap.errors.append(f"binance_square_auth: {e}")
    except ScraperGeoBlocked as e:
        snap.primary_status = "degraded:geo_blocked"
        snap.errors.append(f"binance_square_geo: {e}")
    except ScraperError as e:
        snap.primary_status = f"degraded:error:{type(e).__name__}"
        snap.errors.append(f"binance_square: {e}")
    except Exception as e:
        snap.primary_status = f"degraded:error:{type(e).__name__}"
        snap.errors.append(f"binance_square_unexpected: {e}")

    # ---- AUXILIARY: parallel best-effort ----
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        results = await asyncio.gather(
            _fetch_okx(client, symbol),
            _fetch_dexscreener(client, symbol),
            _fetch_coingecko_trending(client, symbol),
            return_exceptions=True,
        )
    okx_r, dex_r, cg_r = results
    if isinstance(okx_r, dict):
        snap.okx = okx_r
    elif isinstance(okx_r, Exception):
        snap.errors.append(f"okx: {okx_r}")
    if isinstance(dex_r, dict):
        snap.dexscreener = dex_r
    elif isinstance(dex_r, Exception):
        snap.errors.append(f"dexscreener: {dex_r}")
    if isinstance(cg_r, dict):
        snap.coingecko = cg_r
    elif isinstance(cg_r, Exception):
        snap.errors.append(f"coingecko: {cg_r}")

    return snap


# --------------------------------------------------------------------- #
# Auxiliary source fetchers (public, no-auth)
# --------------------------------------------------------------------- #


async def _fetch_okx(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    """Pull current funding + OI for the OKX swap form of `symbol`."""
    base = symbol.upper().replace("USDT", "")
    inst_id = f"{base}-USDT-SWAP"
    out: dict[str, Any] = {"inst_id": inst_id}
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": inst_id},
        )
        if r.status_code == 200:
            rows = r.json().get("data", [])
            if rows:
                out["funding_rate"] = float(rows[0].get("fundingRate", 0))
    except Exception as e:
        out["funding_error"] = str(e)
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/public/open-interest",
            params={"instId": inst_id},
        )
        if r.status_code == 200:
            rows = r.json().get("data", [])
            if rows:
                out["open_interest"] = float(rows[0].get("oi", 0))
    except Exception as e:
        out["oi_error"] = str(e)
    return out


async def _fetch_dexscreener(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    """Search DexScreener for the token symbol; return the most-liquid pair."""
    q = symbol.upper().replace("USDT", "")
    out: dict[str, Any] = {"query": q}
    r = await client.get(f"https://api.dexscreener.com/latest/dex/search?q={q}")
    r.raise_for_status()
    pairs = (r.json() or {}).get("pairs", []) or []
    if not pairs:
        out["pair"] = None
        return out
    pairs.sort(key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0.0), reverse=True)
    top = pairs[0]
    out["pair"] = {
        "chain": top.get("chainId"),
        "dex": top.get("dexId"),
        "price_usd": float(top.get("priceUsd") or 0.0),
        "vol_h24_usd": float((top.get("volume") or {}).get("h24") or 0.0),
        "price_change_h24": float((top.get("priceChange") or {}).get("h24") or 0.0),
        "liquidity_usd": float((top.get("liquidity") or {}).get("usd") or 0.0),
        "url": top.get("url"),
        "socials": (top.get("info") or {}).get("socials", []),
    }
    out["pair_count"] = len(pairs)
    return out


async def _fetch_coingecko_trending(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {"symbol": symbol}
    try:
        r = await client.get("https://api.coingecko.com/api/v3/search/trending")
        r.raise_for_status()
        coins = (r.json() or {}).get("coins", [])
        target = symbol.upper().replace("USDT", "")
        rank: int | None = None
        for i, c in enumerate(coins, 1):
            item = c.get("item") or {}
            if item.get("symbol", "").upper() == target:
                rank = i
                break
        out["trending_rank"] = rank
        out["trending_total"] = len(coins)
    except Exception as e:
        out["error"] = str(e)
    return out
