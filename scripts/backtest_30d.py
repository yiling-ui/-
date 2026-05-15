"""backtest_30d.py — Use Learning Engine as a backtest driver.

Loads a list of (symbol, target_ts) "妖币 events" from a JSON file or stdin,
fetches each event's 4h slice from OKX (with optional SQLite cache),
runs post-mortem (heuristic by default, LLM if enabled), and accumulates
rule statistics into ``.kiro/steering/dynamic_rules.json``. The fuser
hot-loads the updated rules on its next evaluation cycle.

Usage:
    # Discover events first:
    python scripts/discover_events.py --symbols RAVEUSDT,MYXUSDT \
        --days 30 --min-move-pct 0.15 --out events.json

    # Then backtest:
    python scripts/backtest_30d.py events.json --use-llm

    # Or generate one event per symbol, dated 1 day ago:
    python scripts/backtest_30d.py --symbols RAVEUSDT,MYXUSDT --auto-events
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from altcoin_agent.ai_engine import DeepSeekEngine, LLMEngine
from altcoin_agent.learning_engine import (
    PostMortemReport,
    RuleStore,
    fetch_historical_slice,
    run_post_mortem,
)
from altcoin_agent.llm_provider import build_default_provider
from altcoin_agent.slice_cache import SliceCache

logger = logging.getLogger("backtest")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 30d post-mortem backtest")
    p.add_argument("events", nargs="?", default="-",
                   help="path to events JSON, or '-' for stdin")
    p.add_argument("--symbols", default="",
                   help="comma-separated symbol list (with --auto-events)")
    p.add_argument("--auto-events", action="store_true",
                   help="generate one event per symbol, dated 1 day ago")
    p.add_argument("--rules-json", default=".kiro/steering/dynamic_rules.json",
                   help="output JSON path for the rule store")
    p.add_argument("--rules-md", default=".kiro/steering/dynamic_rules.md",
                   help="output MD path (regenerated)")
    p.add_argument("--use-llm", action="store_true",
                   help="use the configured LLM provider for picks")
    p.add_argument("--cache", default=".kiro/cache/slices.sqlite",
                   help="SQLite slice cache path; pass empty string to disable")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the SQLite slice cache")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _load_events(args: argparse.Namespace) -> list[dict]:
    if args.auto_events:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            print("--auto-events requires --symbols", file=sys.stderr)
            sys.exit(2)
        ts = int(time.time() * 1000) - 86_400_000
        return [{"symbol": s, "ts_ms": ts} for s in symbols]
    if args.events == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(args.events).read_text())


def _build_engine(use_llm: bool) -> DeepSeekEngine | LLMEngine | None:
    if not use_llm:
        return None
    # Prefer the new provider abstraction so users can pick any backend.
    provider = build_default_provider()
    if provider is None:
        logger.warning("--use-llm requested but no provider configured; "
                        "falling back to heuristic")
        return None
    logger.info("LLM enabled: provider=%s model=%s",
                 provider.name, provider.model)
    return LLMEngine(provider=provider)


async def _run() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    events = _load_events(args)
    if not events:
        logger.warning("No events to process.")
        return 0

    rules_json = Path(args.rules_json)
    rules_md = Path(args.rules_md)
    store = RuleStore(json_path=rules_json, md_path=rules_md)

    engine = _build_engine(args.use_llm)

    cache: SliceCache | None = None
    if not args.no_cache and args.cache:
        cache = SliceCache(args.cache)
        logger.info("Slice cache: %s (existing slices=%d)",
                     args.cache, cache.stats()["total_slices"])

    n_ok = 0
    n_err = 0
    n_cached = 0
    reports: list[PostMortemReport] = []
    try:
        for ev in events:
            symbol = str(ev.get("symbol") or "")
            ts_ms = int(ev.get("ts_ms") or 0)
            if not symbol or ts_ms <= 0:
                logger.warning("Skipping malformed event: %s", ev)
                continue
            try:
                logger.info("post-mortem: %s @ %d", symbol, ts_ms)
                slice_ = None
                if cache is not None:
                    slice_ = cache.get(symbol, ts_ms)
                    if slice_ is not None:
                        n_cached += 1
                        logger.info("  (slice from cache)")
                if slice_ is None:
                    slice_ = await fetch_historical_slice(symbol, ts_ms)
                    if cache is not None:
                        cache.put(slice_)
                report = await run_post_mortem(
                    symbol=symbol, target_ts_ms=ts_ms,
                    store=store, engine=engine,
                    slice_override=slice_,
                )
                reports.append(report)
                n_ok += 1
                logger.info(
                    "  -> direction=%s magnitude=%+.2f%% picks=%s",
                    report.result.direction,
                    report.result.magnitude_pct * 100,
                    [(p.feature_name, p.bucket) for p in report.picks],
                )
            except Exception as e:
                logger.exception("post-mortem failed for %s: %s", symbol, e)
                n_err += 1
    finally:
        if engine is not None:
            await engine.aclose()
        if cache is not None:
            cache.close()

    rules = store.all_rules()
    logger.info("Done. ok=%d err=%d cache_hits=%d total_rules=%d",
                n_ok, n_err, n_cached, len(rules))
    logger.info("Rules JSON: %s", rules_json)
    logger.info("Rules MD:   %s", rules_md)

    rules.sort(key=lambda r: (-r.hit_rate, -r.total))
    print(json.dumps({
        "events_processed": n_ok,
        "events_failed": n_err,
        "cache_hits": n_cached,
        "total_rules": len(rules),
        "top_rules": [asdict(r) | {"hit_rate": r.hit_rate} for r in rules[:10]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
