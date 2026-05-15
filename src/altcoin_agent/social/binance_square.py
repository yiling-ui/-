"""
binance_square.py — hardened Binance Square scraper.

Status of Binance's Square API (verified by reverse-engineering the live
www.binance.com/en/square frontend):

    Endpoint                                               | Auth     | Notes
    -------------------------------------------------------+----------+----------
    GET  /bapi/composite/v1/public/cms/article/list/query  | none     | Official
                                                                       announcements
                                                                       (listings,
                                                                       maintenance)
    POST /bapi/composite/v1/private/content/feed/post/list | session  | Square posts
    POST /bapi/composite/v1/private/content/community/     | session  | Square feed
         square/feed/get-feeds                                          (alt)
    POST /bapi/composite/v1/private/content/community/     | session  | Square feed
         square/feed-list                                               (alt)

Public access to Square POSTS is closed: those endpoints return
{"code":"100001005","message":"Please log in first."} unless you provide
a logged-in session cookie. The CMS announcements endpoint is open and
useful for new-listing alerts.

Geo-blocking: many sandbox / cloud-IP environments get a 451 or a
202-with-empty-body bounce. This scraper supports HTTP / SOCKS5 proxy
injection so the user can route via a residential VPN at runtime.

Design decisions:
    * aiohttp client, not httpx — the user explicitly asked for the
      aiohttp idiom and it has first-class proxy support including
      SOCKS via aiohttp_socks.
    * Multi-endpoint waterfall: try authenticated Square post endpoints
      first, fall back to the public CMS announcements as a last resort.
    * Strict typing of the return shape (`SquarePost`) so consumers can
      rely on it regardless of which underlying endpoint produced it.
    * Per-request rotation: User-Agent pool, optional cookie injection,
      optional proxy injection. Both are passed in by the caller — this
      class never reads env vars, so deployment never accidentally leaks
      credentials.
    * Exponential backoff on 401/403/429/5xx; abort cleanly on any
      definitive 4xx that can't be retried (404).
    * The class is async-context-manager compatible.

The class is written so the same code path runs on the user's local
machine (with a real logged-in cookie + residential VPN) and in the
sandbox (where it will fail gracefully and emit a helpful error log).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# A small UA pool. In production callers should override with a much
# larger pool to look organic.
DEFAULT_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 "
    "Firefox/130.0",
)


# ----------------------------------------------------------------------- #
# Types
# ----------------------------------------------------------------------- #


@dataclass(frozen=True)
class SquarePost:
    """A normalized Binance Square / CMS post."""

    post_id: str
    author: str
    follower_count: int
    text: str
    title: str
    ts_ms: int
    url: str
    source_endpoint: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    """How to route HTTP requests.

    Use exactly one of:
      * `url` — a single proxy URL (`http://...`, `https://...`,
        `socks5://...`)
      * `pool` — a list of proxy URLs to rotate through randomly per
        request (residential / sticky-session VPN style)

    Auth credentials should be embedded in the URL:
      ``http://user:pass@host:port``
    """

    url: str | None = None
    pool: list[str] = field(default_factory=list)
    rotate: bool = True

    def pick(self) -> str | None:
        if self.url:
            return self.url
        if not self.pool:
            return None
        return random.choice(self.pool) if self.rotate else self.pool[0]


@dataclass
class CookieJar:
    """Logged-in session cookies for authenticated Square endpoints.

    Required when calling the `private/...` family of endpoints. Capture
    these from a real browser session (`document.cookie` after logging
    into binance.com) and pass them in. Treat as secret.

    The two values that matter most are usually:
      * `csrftoken`
      * `bnc-uuid`
      * `p20t` (login session token)

    But Binance occasionally renames these, so we just take a free-form
    dict and let the user paste whatever they captured.
    """

    cookies: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_header_string(cls, raw: str) -> CookieJar:
        """Parse a raw `Cookie:` header value into a CookieJar."""
        out: dict[str, str] = {}
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            out[k.strip()] = v.strip()
        return cls(cookies=out)

    def header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())


# ----------------------------------------------------------------------- #
# Endpoint registry
# ----------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Endpoint:
    """One Binance API endpoint we'll try, in priority order."""

    name: str
    method: str
    path: str
    requires_auth: bool

    def url(self) -> str:
        return f"https://www.binance.com{self.path}"


# Priority order:
#   1. authenticated Square post endpoints (richest data)
#   2. public CMS announcements (open, but listing/maintenance only)
_ENDPOINTS: tuple[_Endpoint, ...] = (
    _Endpoint(
        name="square_feed_by_tag",
        method="POST",
        path="/bapi/composite/v1/private/content/community/square/feed/get-feeds",
        requires_auth=True,
    ),
    _Endpoint(
        name="square_feed_list",
        method="POST",
        path="/bapi/composite/v1/private/content/community/square/feed-list",
        requires_auth=True,
    ),
    _Endpoint(
        name="square_post_list",
        method="POST",
        path="/bapi/composite/v1/private/content/feed/post/list",
        requires_auth=True,
    ),
    _Endpoint(
        name="cms_announcements",
        method="GET",
        path="/bapi/composite/v1/public/cms/article/list/query",
        requires_auth=False,
    ),
)


# ----------------------------------------------------------------------- #
# Errors
# ----------------------------------------------------------------------- #


class ScraperError(RuntimeError):
    """Raised when no endpoint produced usable data."""


class ScraperGeoBlocked(ScraperError):
    """Raised when every attempt returned a 451 / blocked response."""


class ScraperAuthRequired(ScraperError):
    """Raised when only auth-gated endpoints are available and no cookie
    jar was provided."""


# ----------------------------------------------------------------------- #
# Scraper
# ----------------------------------------------------------------------- #


@dataclass
class BinanceSquareScraper:
    """Hardened Binance Square scraper.

    Usage:
        async with BinanceSquareScraper(
            cookies=CookieJar.from_header_string(my_cookie),
            proxy=ProxyConfig(pool=["socks5://user:pass@host:1080", ...]),
        ) as scraper:
            posts = await scraper.fetch_by_keyword("RAVE", limit=20)

    Args:
        cookies: optional logged-in session jar. If absent, only the
            public CMS endpoint is reachable.
        proxy: optional proxy config (single URL or rotating pool).
        user_agents: pool of UAs to rotate. Defaults to a small bundled
            set; override for production.
        request_timeout_sec: per-request timeout.
        max_retries: how many times to retry on transient errors.
        backoff_base_sec: exponential backoff base.
    """

    cookies: CookieJar | None = None
    proxy: ProxyConfig | None = None
    user_agents: tuple[str, ...] = DEFAULT_USER_AGENTS
    request_timeout_sec: float = 15.0
    max_retries: int = 3
    backoff_base_sec: float = 1.0

    _session: Any = field(init=False, default=None, repr=False)

    # ------------------------- lifecycle ------------------------- #

    async def __aenter__(self) -> BinanceSquareScraper:
        await self._ensure_session()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import aiohttp  # local import so unit tests don't need the dep
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "aiohttp is required for BinanceSquareScraper. "
                "Install with: pip install aiohttp aiohttp_socks"
            ) from e
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------- public API ------------------------- #

    async def fetch_by_keyword(
        self, keyword: str, *, limit: int = 20,
    ) -> list[SquarePost]:
        """Get up to `limit` recent Square posts matching `keyword`.

        Tries auth-gated endpoints first; falls back to public CMS.
        """
        await self._ensure_session()
        last_err: Exception | None = None

        for ep in _ENDPOINTS:
            if ep.requires_auth and self.cookies is None:
                logger.debug("skipping %s: requires auth, no cookie jar", ep.name)
                continue
            try:
                posts = await self._call_endpoint(ep, keyword=keyword, limit=limit)
                if posts:
                    logger.info(
                        "BinanceSquareScraper: %d posts via endpoint=%s "
                        "for keyword=%s",
                        len(posts), ep.name, keyword,
                    )
                    return posts
            except ScraperGeoBlocked:
                # geo blocks aren't recoverable by switching endpoints,
                # only by switching proxy. Bubble up immediately.
                raise
            except Exception as e:
                last_err = e
                logger.warning(
                    "BinanceSquareScraper endpoint %s failed: %s", ep.name, e,
                )
                continue

        # No endpoint produced data
        if self.cookies is None:
            raise ScraperAuthRequired(
                "No Square posts available without a logged-in cookie jar; "
                "tried all public endpoints. Last error: " + str(last_err)
            )
        raise ScraperError(
            f"All Binance Square endpoints failed. Last error: {last_err}"
        )

    async def fetch_announcements(
        self, *, page_size: int = 20, type_id: int = 1,
    ) -> list[SquarePost]:
        """Always-open public path: official Binance announcements."""
        await self._ensure_session()
        ep = next(e for e in _ENDPOINTS if e.name == "cms_announcements")
        return await self._call_endpoint(ep, keyword="", limit=page_size, cms_type=type_id)

    async def stream_keyword(
        self, keyword: str, *, interval_sec: float = 30.0,
        max_seen: int = 5_000,
    ) -> AsyncIterator[SquarePost]:
        """Long-poll new posts matching `keyword`. Dedupes by post_id.

        Designed for the "妖币聚焦" mode: triggered after a coin is
        confirmed as a candidate, run for a bounded duration, then stop.
        """
        seen: set[str] = set()
        while True:
            try:
                posts = await self.fetch_by_keyword(keyword, limit=20)
            except ScraperError as e:
                logger.warning("stream_keyword: %s", e)
                posts = []
            for p in posts:
                if p.post_id in seen:
                    continue
                seen.add(p.post_id)
                if len(seen) > max_seen:
                    # forget the oldest half — keep the set bounded
                    seen = set(list(seen)[-max_seen // 2 :])
                yield p
            await asyncio.sleep(interval_sec)

    # ------------------------- internals ------------------------- #

    def _build_headers(self, ep: _Endpoint) -> dict[str, str]:
        ua = random.choice(self.user_agents)
        h = {
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.binance.com",
            "Referer": "https://www.binance.com/en/square",
            "lang": "en",
            "clienttype": "web",
            "device-info": "eyJzY3JlZW5fcmVzb2x1dGlvbiI6IjE5MjB4MTA4MCJ9",
        }
        if ep.method == "POST":
            h["Content-Type"] = "application/json"
        if self.cookies is not None and self.cookies.cookies:
            h["Cookie"] = self.cookies.header()
            csrf = self.cookies.cookies.get("csrftoken")
            if csrf:
                h["csrftoken"] = csrf
            uuid_ = self.cookies.cookies.get("bnc-uuid")
            if uuid_:
                h["bnc-uuid"] = uuid_
        return h

    def _build_body(
        self, ep: _Endpoint, *, keyword: str, limit: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Returns (json_body, query_params). Exactly one is non-None."""
        if ep.name == "cms_announcements":
            return None, {"type": 1, "pageNo": 1, "pageSize": limit}
        if ep.name in ("square_feed_by_tag",):
            return {
                "pageIndex": 1, "pageSize": limit, "tag": keyword,
                "scene": "topic", "tabId": 0,
            }, None
        if ep.name in ("square_feed_list",):
            return {
                "pageIndex": 1, "pageSize": limit, "scene": "search",
                "keyword": keyword,
            }, None
        if ep.name == "square_post_list":
            return {
                "pageIndex": 1, "pageSize": limit, "keyword": keyword,
                "tab": "For You",
            }, None
        return None, None

    async def _call_endpoint(
        self,
        ep: _Endpoint,
        *,
        keyword: str,
        limit: int,
        cms_type: int = 1,
    ) -> list[SquarePost]:
        body, params = self._build_body(ep, keyword=keyword, limit=limit)
        if ep.name == "cms_announcements" and params is not None:
            params["type"] = cms_type

        attempt = 0
        while True:
            attempt += 1
            proxy_url = self.proxy.pick() if self.proxy else None
            headers = self._build_headers(ep)
            try:
                resp_status, resp_text = await self._do_request(
                    method=ep.method,
                    url=ep.url(),
                    headers=headers,
                    json_body=body,
                    params=params,
                    proxy_url=proxy_url,
                )
            except Exception as e:
                if attempt >= self.max_retries:
                    raise
                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                logger.debug("transport error %s, retrying in %.2fs", e, wait)
                await asyncio.sleep(wait)
                continue

            if resp_status == 451:
                raise ScraperGeoBlocked(
                    f"Binance returned 451 (geo-blocked). "
                    f"Configure proxy.pool with a residential VPN. "
                    f"endpoint={ep.name}"
                )
            if resp_status == 401:
                raise ScraperAuthRequired(
                    f"endpoint={ep.name} requires a valid session cookie."
                )
            if resp_status == 200:
                import json
                try:
                    data = json.loads(resp_text)
                except Exception as e:
                    raise ScraperError(
                        f"endpoint={ep.name} returned non-JSON: "
                        f"{resp_text[:200]}"
                    ) from e
                return _parse_response(ep, data)
            if resp_status in (429, 500, 502, 503, 504):
                if attempt >= self.max_retries:
                    raise ScraperError(
                        f"endpoint={ep.name} status={resp_status} "
                        f"after {attempt} attempts"
                    )
                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                await asyncio.sleep(wait)
                continue
            # any other 4xx is non-recoverable
            raise ScraperError(
                f"endpoint={ep.name} status={resp_status}: {resp_text[:200]}"
            )

    async def _do_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        proxy_url: str | None,
    ) -> tuple[int, str]:
        """Perform a single HTTP request via aiohttp, returning status+text.

        Encapsulated so tests can monkey-patch this single method.
        """
        assert self._session is not None  # set by _ensure_session

        # SOCKS5 needs a special connector; HTTP/HTTPS proxies are
        # native to aiohttp.
        connector = None
        proxy_for_aiohttp: str | None = None
        if proxy_url:
            if proxy_url.startswith("socks"):
                try:
                    from aiohttp_socks import ProxyConnector
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError(
                        "aiohttp_socks is required for SOCKS proxy support. "
                        "Install: pip install aiohttp_socks"
                    ) from e
                connector = ProxyConnector.from_url(proxy_url)
            else:
                proxy_for_aiohttp = proxy_url

        # If we need a custom (SOCKS) connector, we have to spin up a
        # one-shot session for that request. For HTTP proxies we can
        # reuse the long-lived session.
        if connector is not None:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout,
            ) as one_shot:
                async with one_shot.request(
                    method, url, headers=headers,
                    json=json_body, params=params,
                ) as resp:
                    text = await resp.text()
                    return resp.status, text
        else:
            async with self._session.request(
                method, url, headers=headers,
                json=json_body, params=params,
                proxy=proxy_for_aiohttp,
            ) as resp:
                text = await resp.text()
                return resp.status, text


# ----------------------------------------------------------------------- #
# Response parsing
# ----------------------------------------------------------------------- #


def _parse_response(ep: _Endpoint, data: Any) -> list[SquarePost]:
    """Normalize the heterogeneous response shapes into SquarePost lists."""
    out: list[SquarePost] = []
    if not isinstance(data, dict):
        return out

    if ep.name == "cms_announcements":
        # Shape: {"data": {"catalogs": [{"articles": [...], ...}, ...]}}
        catalogs = data.get("data", {}).get("catalogs", []) or []
        articles: list[dict[str, Any]] = []
        for cat in catalogs:
            articles.extend(cat.get("articles", []) or [])
        for a in articles:
            out.append(_post_from_cms_article(a, source_endpoint=ep.name))
        return out

    # All Square (private) endpoints share a similar wrapper:
    #   {"code":"000000","data":{"posts": [...]}}  OR
    #   {"code":"000000","data":{"feeds": [...]}}  OR
    #   {"code":"000000","data":{"items": [...]}}
    payload = data.get("data") or {}
    items = (
        payload.get("posts")
        or payload.get("feeds")
        or payload.get("items")
        or payload.get("list")
        or []
    )
    if not isinstance(items, list):
        return out
    for item in items:
        out.append(_post_from_square(item, source_endpoint=ep.name))
    return out


def _post_from_cms_article(a: dict[str, Any], source_endpoint: str) -> SquarePost:
    article_id = str(a.get("id", a.get("code", "")))
    code = a.get("code", "")
    return SquarePost(
        post_id=article_id,
        author="binance_official",
        follower_count=0,
        title=str(a.get("title", "")),
        text=str(a.get("title", "")) + " " + str(a.get("body", "") or ""),
        ts_ms=int(a.get("releaseDate", a.get("createTime", 0))),
        url=f"https://www.binance.com/en/support/announcement/{code}" if code else "",
        source_endpoint=source_endpoint,
        raw=a,
    )


def _post_from_square(item: dict[str, Any], source_endpoint: str) -> SquarePost:
    # The Square endpoints use slightly different field names depending
    # on which one we hit. Be liberal in what we accept.
    user = item.get("user") or item.get("author") or {}
    if isinstance(user, str):
        author = user
        followers = 0
    else:
        author = (
            user.get("nickname")
            or user.get("name")
            or user.get("userName")
            or "unknown"
        )
        followers = int(
            user.get("followerCount") or user.get("followers") or 0
        )

    pid = str(item.get("id") or item.get("postId") or item.get("feedId") or "")
    text = (
        item.get("content")
        or item.get("text")
        or item.get("description")
        or ""
    )
    title = item.get("title") or ""
    ts = int(
        item.get("createTime")
        or item.get("publishTime")
        or item.get("timestamp")
        or 0
    )
    url = item.get("url") or item.get("shareUrl") or ""
    likes = int(item.get("likeCount") or item.get("likes") or 0)
    comments = int(item.get("commentCount") or item.get("comments") or 0)
    shares = int(item.get("shareCount") or item.get("shares") or 0)

    return SquarePost(
        post_id=pid,
        author=str(author),
        follower_count=followers,
        title=str(title),
        text=str(text),
        ts_ms=ts,
        url=str(url),
        source_endpoint=source_endpoint,
        likes=likes,
        comments=comments,
        shares=shares,
        raw=item,
    )


# ----------------------------------------------------------------------- #
# Convenience: turn SquarePosts into the shape ai_engine expects
# ----------------------------------------------------------------------- #


def to_social_posts(square_posts: Iterable[SquarePost]) -> list[Any]:
    """Convert SquarePosts into ai_engine.SocialPost objects."""
    from altcoin_agent.ai_engine import SocialPost
    out: list[SocialPost] = []
    for p in square_posts:
        out.append(SocialPost(
            author=p.author,
            follower_count=p.follower_count,
            text=(p.title + " " + p.text).strip(),
            ts=p.ts_ms // 1000 if p.ts_ms > 1_000_000_000_000 else p.ts_ms,
            source=f"binance_square:{p.source_endpoint}",
        ))
    return out
