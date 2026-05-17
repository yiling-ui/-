"""fetch_history.py — Bulk historical OHLCV (and optionally funding/OI) downloader.

Operator-facing CLI that drives :class:`HistoricalDataLoader` and the
funding/OI loaders end-to-end:

  * Connects to a ccxt exchange (Binance Futures by default)
  * Token-bucket rate-limits requests to stay under the venue cap
  * Writes per-month JSON cache files under
    ``.kiro/state/backtest_cache/{exchange}/{symbol}/{timeframe}/YYYY/MM.json``
  * Resumable: re-running skips months whose cache file already exists
    (unless ``--force`` is passed)

Usage examples:

  Pull 90 days of 1m klines for three meme coins, dry-run first to
  preview the work:

    python scripts/fetch_history.py \\
        --symbols PEPE/USDT:USDT,WIF/USDT:USDT,TRUMP/USDT:USDT \\
        --timeframe 1m --days 90 --dry-run

  Real fetch, 10 req/s cap (default), saving to repo-default cache:

    python scripts/fetch_history.py \\
        --symbols PEPE/USDT:USDT --timeframe 1m --days 30

  Pull funding history alongside klines:

    python scripts/fetch_history.py --symbols PEPE/USDT:USDT \\
        --timeframe 1m --days 90 --with-funding --with-oi

  Custom cache dir + a tighter rate limit for a small VPS:

    python scripts/fetch_history.py --symbols PEPE/USDT:USDT \\
        --timeframe 5m --days 365 \\
        --state-dir /var/cache/altcoin/backtest \\
        --max-rate 5

The CLI never touches private API keys; all calls are public REST.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

from altcoin_agent.backtest.historical_loader import (
    HistoricalDataLoader,
    RateLimiter,
)

logger = logging.getLogger("fetch_history")


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #


@dataclass
class CliArgs:
    exchange: str
    symbols: list[str]
    timeframe: str
    start_ms: int
    end_ms: int
    state_dir: str
    max_rate: float
    capacity: float
    chunk_limit: int
    inter_call_sleep_sec: float
    with_funding: bool
    with_oi: bool
    force: bool
    dry_run: bool
    log_level: str


def _parse_args(argv: list[str] | None = None) -> CliArgs:
    p = argparse.ArgumentParser(
        prog="fetch_history",
        description=(
            "Download + cache historical OHLCV (and optionally "
            "funding rate / open interest) for a list of symbols."
        ),
    )
    p.add_argument("--exchange", default="binance",
                   help="ccxt exchange id (default: binance)")
    p.add_argument("--symbols", required=True,
                   help="comma-separated ccxt symbols, e.g. 'PEPE/USDT:USDT,WIF/USDT:USDT'")
    p.add_argument("--timeframe", default="1m",
                   help="kline timeframe: 1m / 5m / 15m / 1h / 4h / 1d (default: 1m)")

    span = p.add_mutually_exclusive_group(required=True)
    span.add_argument("--days", type=int,
                      help="lookback window in days (end = now)")
    span.add_argument("--start", type=str,
                      help="ISO-8601 UTC start, e.g. 2024-09-01T00:00:00Z")
    p.add_argument("--end", type=str, default="",
                   help="ISO-8601 UTC end (default: now)")

    p.add_argument("--state-dir", default=".kiro/state/backtest_cache",
                   help="root directory for the on-disk cache")
    p.add_argument("--max-rate", type=float, default=10.0,
                   help="token-bucket refill rate, req/s (default: 10)")
    p.add_argument("--capacity", type=float, default=20.0,
                   help="token-bucket burst capacity (default: 20)")
    p.add_argument("--chunk-limit", type=int, default=1000,
                   help="bars per fetch_ohlcv call (default: 1000)")
    p.add_argument("--inter-call-sleep", type=float, default=0.0,
                   help=("legacy fixed sleep between calls in seconds "
                         "(default: 0; rely on --max-rate instead)"))

    p.add_argument("--with-funding", action="store_true",
                   help="also download funding-rate history")
    p.add_argument("--with-oi", action="store_true",
                   help="also download open-interest history")

    p.add_argument("--force", action="store_true",
                   help="re-download months whose cache file already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without fetching anything")
    p.add_argument("--log-level", default="INFO")

    ns = p.parse_args(argv)

    end_ms = (
        _iso_to_ms(ns.end)
        if ns.end
        else int(time.time() * 1000)
    )
    if ns.days:
        start_ms = end_ms - int(ns.days) * 86_400_000
    else:
        start_ms = _iso_to_ms(ns.start)
    if start_ms >= end_ms:
        p.error(f"start ({start_ms}) must be before end ({end_ms})")

    symbols = [s.strip() for s in ns.symbols.split(",") if s.strip()]
    if not symbols:
        p.error("--symbols cannot be empty")

    return CliArgs(
        exchange=ns.exchange,
        symbols=symbols,
        timeframe=ns.timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        state_dir=ns.state_dir,
        max_rate=float(ns.max_rate),
        capacity=float(ns.capacity),
        chunk_limit=int(ns.chunk_limit),
        inter_call_sleep_sec=float(ns.inter_call_sleep),
        with_funding=bool(ns.with_funding),
        with_oi=bool(ns.with_oi),
        force=bool(ns.force),
        dry_run=bool(ns.dry_run),
        log_level=ns.log_level,
    )


def _iso_to_ms(iso: str) -> int:
    """Parse ISO-8601 with optional Z suffix to ms-epoch."""
    from datetime import datetime, timezone
    txt = iso.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------- #
# Exchange wiring (only imported on demand so unit tests don't pull ccxt)
# --------------------------------------------------------------------- #


def _build_ccxt_fetcher(exchange_id: str):
    """Return a live ccxt instance configured for public REST."""
    try:
        import ccxt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ccxt is required to fetch live history. "
            "pip install ccxt"
        ) from exc
    if not hasattr(ccxt, exchange_id):
        raise SystemExit(f"unknown ccxt exchange: {exchange_id!r}")
    klass = getattr(ccxt, exchange_id)
    inst = klass({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # USDT-margined perp default
    })
    return inst


# --------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------- #


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    span_days = (args.end_ms - args.start_ms) / 86_400_000
    logger.info(
        "plan: exchange=%s symbols=%d timeframe=%s span=%.1fd "
        "with_funding=%s with_oi=%s state_dir=%s",
        args.exchange, len(args.symbols), args.timeframe, span_days,
        args.with_funding, args.with_oi, args.state_dir,
    )
    for s in args.symbols:
        logger.info("  - %s", s)

    if args.dry_run:
        logger.info(
            "dry-run: would download %d symbol(s) x %.1f days at "
            "max_rate=%.1f req/s; exiting without fetching.",
            len(args.symbols), span_days, args.max_rate,
        )
        return 0

    fetcher = _build_ccxt_fetcher(args.exchange)
    limiter = RateLimiter(
        max_rate=args.max_rate, capacity=args.capacity,
    )
    loader = HistoricalDataLoader(
        fetcher=fetcher,
        cache_root=args.state_dir,
        exchange_name=args.exchange,
        inter_call_sleep_sec=args.inter_call_sleep_sec,
        chunk_limit=args.chunk_limit,
        rate_limiter=limiter,
    )

    total = 0
    for sym in args.symbols:
        logger.info("downloading %s ...", sym)
        try:
            written = loader.download(
                symbol=sym, timeframe=args.timeframe,
                start_ms=args.start_ms, end_ms=args.end_ms,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", sym, exc)
            continue
        logger.info("  %s: %d bars written", sym, written)
        total += written

    if args.with_funding or args.with_oi:
        try:
            from altcoin_agent.backtest.funding_history_loader import (
                FundingHistoryLoader,
                OpenInterestHistoryLoader,
            )
        except ImportError as exc:
            logger.warning(
                "funding/OI loaders unavailable (%s); skipping", exc,
            )
        else:
            if args.with_funding:
                f_loader = FundingHistoryLoader(
                    fetcher=fetcher,
                    cache_root=args.state_dir,
                    exchange_name=args.exchange,
                    rate_limiter=limiter,
                )
                for sym in args.symbols:
                    try:
                        n = f_loader.download(
                            symbol=sym,
                            start_ms=args.start_ms, end_ms=args.end_ms,
                            force=args.force,
                        )
                        logger.info("  funding %s: %d records", sym, n)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("funding FAILED %s: %s", sym, exc)
            if args.with_oi:
                oi_loader = OpenInterestHistoryLoader(
                    fetcher=fetcher,
                    cache_root=args.state_dir,
                    exchange_name=args.exchange,
                    rate_limiter=limiter,
                )
                for sym in args.symbols:
                    try:
                        n = oi_loader.download(
                            symbol=sym,
                            start_ms=args.start_ms, end_ms=args.end_ms,
                            force=args.force,
                        )
                        logger.info("  OI %s: %d records", sym, n)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("OI FAILED %s: %s", sym, exc)

    logger.info("done. total bars written: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(run())
