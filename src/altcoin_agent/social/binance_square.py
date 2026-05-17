"""binance_square.py — Binance Square (币安广场) scraper.

Architectural notes (committed with the user):

  1. Binance does not offer an official public API for the Square feed; the
     site uses internal `bapi/composite/v1/...` endpoints that require a
     logged-in browser session (CSRF token + cookies). We therefore support:
        - cookie injection from a raw 'Cookie:' header (CookieJar)
        - HTTP / SOCKS5 proxy pool injection (ProxyConfig) for IPs blocked
          with HTTP 451 (geo-block) -- common from cloud providers.
     The actual proxy/cookie values are NOT bundled; the user mounts them
     at deploy time via env vars or files.

  2. The scraper tries multiple internal endpoints in waterfall order. If
     all auth-required endpoints fail, it gracefully degrades to the public
     CMS announcements feed -- this is rate-limited but at least gives the
     downstream pipeline SOMETHING to reason about.

  3. Typed exceptions distinguish between "we are geo-blocked" (451),
     "we need to log in" (401 / Binance error code 100001005), and any
     other transport failure. The aggregator (crawler.py) uses these to
     choose how to surface degradation to downstream consumers.

  4. The sole network entry point is `_do_request`, so unit tests can
     monkey-patch it without touching the asyncio internals.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# User-Agent pool (recent Chrome / Firefox / Safari)
# --------------------------------------------------------------------- #

_USER_AGENTS: list[str] = [
    # Chrome 124 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 124 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox 125 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 "
    "Firefox/125.0",
    # Safari 17 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


# --------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------- #


class ScraperError(RuntimeError):
    """Generic transport / parse failure."""


class ScraperGeoBlocked(ScraperError):
    """HTTP 451 — IP is in a Binance-restricted region."""


class ScraperAuthRequired(ScraperError):
    """HTTP 401 or Binance code 100001005 — need a logged-in session cookie."""


@dataclass
class SquarePost:
    """One post from Binance Square (or a CMS announcement, normalized)."""

    post_id: str
    author: str
    follower_count: int
    text: str
    ts_ms: int
    source: str = "binance_square"
    url: str | None = None

    def truncated(self, n: int = 280) -> str:
        return self.text if len(self.text) <= n else self.text[:n] + "..."


@dataclass
class ProxyConfig:
    """Single proxy or rotating proxy pool.

    `url` is a single proxy URL (used if `pool` is empty).
    `pool` rotates one URL per request when `rotate=True`.
    Supported schemes: http://, https://, socks5:// (requires aiohttp_socks).
    """

    url: str | None = None
    pool: list[str] = field(default_factory=list)
    rotate: bool = True

    def pick(self) -> str | None:
        if self.pool:
            return random.choice(self.pool) if self.rotate else self.pool[0]
        return self.url

    def is_socks(self, url: str | None = None) -> bool:
        u = url or self.pick() or ""
        return u.startswith("socks")


@dataclass
class CookieJar:
    """Browser session cookies. Capture from a logged-in browser DevTools."""

    cookies: dict[str, str] = field(default_factory=dict)
    csrftoken: str | None = None

    @classmethod
    def from_header_string(cls, header: str) -> CookieJar:
        """Parse a raw 'Cookie:' header value into name->value pairs."""
        cookies: dict[str, str] = {}
        for chunk in header.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            k, _, v = chunk.partition("=")
            cookies[k.strip()] = v.strip()
        csrftoken = cookies.get("csrftoken")
        return cls(cookies=cookies, csrftoken=csrftoken)

    def header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    @property
    def has_session(self) -> bool:
        # Binance considers a session valid when at least these are set.
        return bool(self.cookies.get("p20t") or self.cookies.get("bnc-uuid"))


# --------------------------------------------------------------------- #
# Endpoints (waterfall — try in order)
# --------------------------------------------------------------------- #


_BASE = "https://www.binance.com"

# Authenticated Square feed endpoints. All of these require a logged-in
# session cookie + CSRF token. The order is empirical: at the time of
# writing, the first one is the most reliable.
_AUTH_FEED_ENDPOINTS: list[tuple[str, str]] = [
    ("POST", "/bapi/composite/v1/private/content/feed/get-feeds"),
    ("POST", "/bapi/composite/v1/private/content/feed/post/list"),
    ("POST", "/bapi/composite/v1/private/content/feed-list"),
]

# Public CMS announcement endpoint. No auth required; returns official
# Binance announcements. Useful for "new listing" alerts and as a degraded
# fallback when the authenticated feeds are unreachable.
_PUBLIC_CMS_ENDPOINT = "/bapi/composite/v1/public/cms/article/list/query"


# --------------------------------------------------------------------- #
# Scraper
# --------------------------------------------------------------------- #


@dataclass
class BinanceSquareScraper:
    """Binance Square scraper with cookie + proxy injection.

    Args:
        cookies: optional CookieJar with a logged-in session.
        proxy: optional ProxyConfig.
        request_timeout_sec: per-request timeout.
        accept_language: passed via header.
        min_request_interval_sec: minimum gap between consecutive requests
            (politeness throttle).
    """

    cookies: CookieJar | None = None
    proxy: ProxyConfig | None = None
    request_timeout_sec: float = 10.0
    accept_language: str = "en-US,en;q=0.9"
    min_request_interval_sec: float = 0.4

    _last_request_ts: float = 0.0

    # ----------------------- public API ----------------------- #

    async def fetch_by_keyword(self, keyword: str, *, page_size: int = 20) -> list[SquarePost]:
        """Fetch recent Square posts mentioning `keyword` (e.g. "$RAVE", "PEPE").

        Tries authenticated feed endpoints first; falls back to the public
        CMS announcements (filtered by keyword) on auth failure. Raises
        `ScraperGeoBlocked` if the IP is rejected with 451.
        """
        last_auth_err: ScraperAuthRequired | None = None

        if self.cookies is not None and self.cookies.has_session:
            for method, path in _AUTH_FEED_ENDPOINTS:
                try:
                    posts = await self._fetch_auth_feed(method, path, keyword, page_size)
                    if posts:
                        return posts
                except ScraperAuthRequired as e:
                    last_auth_err = e
                    logger.warning("Square auth feed %s rejected: %s", path, e)
                    continue
                except ScraperGeoBlocked:
                    raise
                except Exception as e:
                    logger.warning("Square auth feed %s failed: %s", path, e)
                    continue
            if last_auth_err is not None:
                # Auth was required everywhere; the user's cookies likely expired.
                logger.warning("All authenticated Square endpoints rejected; degrading to CMS")

        # Fallback: public announcements, filtered by keyword.
        try:
            return await self.fetch_announcements(keyword=keyword, page_size=page_size)
        except ScraperGeoBlocked:
            raise
        except Exception as e:
            if last_auth_err is not None:
                raise last_auth_err from e
            raise ScraperError(f"all endpoints failed: {e}") from e

    async def fetch_announcements(
        self,
        *,
        keyword: str | None = None,
        page_size: int = 40,
    ) -> list[SquarePost]:
        """Pull recent Binance announcements (no auth required)."""
        body = {"type": 1, "pageNo": 1, "pageSize": page_size}
        data = await self._do_request("POST", _PUBLIC_CMS_ENDPOINT, json=body)
        articles = (data.get("data") or {}).get("articles") or []
        if not articles:
            articles = data.get("data") or []  # alternative shape
        out: list[SquarePost] = []
        kw = keyword.lower().lstrip("$").lower() if keyword else None
        for a in articles:
            title = str(a.get("title") or "")
            body_text = str(a.get("body") or a.get("description") or "")
            text = (title + " " + body_text).strip()
            if kw and kw not in text.lower():
                continue
            ts = int(a.get("releaseDate") or a.get("publishDate") or time.time() * 1000)
            out.append(SquarePost(
                post_id=str(a.get("id") or a.get("code") or ts),
                author="Binance Announcements",
                follower_count=10_000_000,
                text=text,
                ts_ms=ts,
                source="binance_announcements",
                url=str(a.get("url") or ""),
            ))
        return out

    async def stream_keyword(
        self,
        keyword: str,
        *,
        duration_sec: int = 30 * 60,
        poll_interval_sec: float = 30.0,
        page_size: int = 20,
    ) -> list[SquarePost]:
        """Limited-duration polling stream. De-dupes by post_id."""
        seen: set[str] = set()
        collected: list[SquarePost] = []
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            try:
                batch = await self.fetch_by_keyword(keyword, page_size=page_size)
            except (ScraperAuthRequired, ScraperGeoBlocked) as e:
                logger.warning("stream_keyword aborting: %s", e)
                raise
            except ScraperError as e:
                logger.warning("stream_keyword transient error: %s", e)
                await asyncio.sleep(poll_interval_sec)
                continue
            for p in batch:
                if p.post_id in seen:
                    continue
                seen.add(p.post_id)
                collected.append(p)
            await asyncio.sleep(poll_interval_sec)
        return collected

    # ----------------------- internals ----------------------- #

    async def _fetch_auth_feed(
        self, method: str, path: str, keyword: str, page_size: int,
    ) -> list[SquarePost]:
        body = {
            "scene": "homePage",
            "scenes": ["homePage"],
            "pageType": "square",
            "pageSize": page_size,
            "pageIndex": 1,
            "keyword": keyword,
            "topicCode": keyword.lstrip("$").upper(),
        }
        data = await self._do_request(method, path, json=body)
        items: list[dict] = []
        d = data.get("data")
        if isinstance(d, dict):
            items = d.get("vos") or d.get("list") or d.get("posts") or []
        elif isinstance(d, list):
            items = d
        out: list[SquarePost] = []
        for it in items:
            try:
                pid = str(it.get("id") or it.get("postId") or it.get("uuid") or "")
                if not pid:
                    continue
                user = it.get("authorName") or (it.get("author") or {}).get("name") or "anon"
                follower = int((it.get("author") or {}).get("followerCount", 0))
                text = str(it.get("content") or it.get("text") or it.get("title") or "")
                ts = int(it.get("createTime") or it.get("publishTime") or time.time() * 1000)
                out.append(SquarePost(
                    post_id=pid, author=str(user), follower_count=follower,
                    text=text, ts_ms=ts, source="binance_square",
                ))
            except Exception as e:
                logger.debug("skipping malformed Square item: %s", e)
        return out

    async def _do_request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Politeness throttle
        delta = time.time() - self._last_request_ts
        if delta < self.min_request_interval_sec:
            await asyncio.sleep(self.min_request_interval_sec - delta)

        url = _BASE + path
        headers = self._build_headers()
        proxy_url = self.proxy.pick() if self.proxy else None

        # SOCKS proxies need aiohttp_socks; HTTP proxies are native.
        connector: aiohttp.BaseConnector | None = None
        if proxy_url and (self.proxy.is_socks(proxy_url) if self.proxy else False):
            try:
                from aiohttp_socks import ProxyConnector  # type: ignore
                connector = ProxyConnector.from_url(proxy_url)
                proxy_url = None  # passed via connector instead
            except ImportError as e:  # pragma: no cover
                raise ScraperError(
                    "aiohttp_socks is required for SOCKS proxies. "
                    "Install with `pip install aiohttp_socks`."
                ) from e

        timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers,
        ) as sess:
            try:
                async with sess.request(method, url, json=json, proxy=proxy_url) as resp:
                    self._last_request_ts = time.time()
                    if resp.status == 451:
                        raise ScraperGeoBlocked(f"HTTP 451 from {path}")
                    if resp.status == 401:
                        raise ScraperAuthRequired(f"HTTP 401 from {path}")
                    text = await resp.text()
                    if resp.status >= 400:
                        raise ScraperError(f"HTTP {resp.status} from {path}: {text[:200]}")
                    try:
                        data = await resp.json(content_type=None)
                    except aiohttp.ContentTypeError as e:
                        raise ScraperError(f"non-json response from {path}: {text[:200]}") from e
                    # Binance internal error code 100001005 == "please log in first"
                    if isinstance(data, dict) and str(data.get("code")) == "100001005":
                        raise ScraperAuthRequired(
                            f"Binance code 100001005 (login required) on {path}"
                        )
                    return data
            except (ScraperGeoBlocked, ScraperAuthRequired, ScraperError):
                raise
            except aiohttp.ClientError as e:
                raise ScraperError(f"transport error: {e}") from e
            except asyncio.TimeoutError as e:
                raise ScraperError(f"timeout after {self.request_timeout_sec}s") from e

    def _build_headers(self) -> dict[str, str]:
        ua = random.choice(_USER_AGENTS)
        h: dict[str, str] = {
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": self.accept_language,
            "Origin": _BASE,
            "Referer": f"{_BASE}/en/square",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/json",
        }
        if self.cookies and self.cookies.cookies:
            h["Cookie"] = self.cookies.header()
            if self.cookies.csrftoken:
                h["csrftoken"] = self.cookies.csrftoken
                h["X-CSRF-Token"] = self.cookies.csrftoken
        return h
