"""discover_events.py — find historical pump/dump events automatically.

Scans OKX swap symbols for the last ``--days`` days and emits an events
JSON file suitable for feeding to ``backtest_30d.py``. An "event" is the
END of a 4h window in which the symbol moved by at least ``--min-move-pct``.

Usage:
    # Discover events for a curated list over the past 30 days
    python scripts/discover_events.py \
        --symbols RAVEUSDT,MYXUSDT,PEPEUSDT \
        --days 30 --min-move-pct 0.15 \
        --out events.json

    # Discover events across ALL OKX swap symbols (slower)
    python scripts/discover_events.py --all-okx --days 30 \
        --min-move-pct 0.20 --max-symbols 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger("discover")


OKX_REST = "https://www.okx.com"


def _to_okx_inst_id(symbol: str) -> str:
    s = symbol.upper().replace("/", "").replace(":USDT", "")
    if s.endswith("USDT") and "-" not in s:
        return f"{s[:-4]}-USDT-SWAP"
    return s


async def _list_okx_swaps(client: httpx.AsyncClient, max_n: int) -> list[str]:
    """List active OKX SWAP instruments (USDT-margined)."""
    r = await client.get(
        f"{OKX_REST}/api/v5/public/instruments",
        params={"instType": "SWAP"},
    )
    r.raise_for_status()
    rows = r.json().get("data", [])
    insts: list[str] = []
    for row in rows:
        inst_id = row.get("instId", "")
        if inst_id.endswith("-USDT-SWAP"):
            insts.append(inst_id)
        if len(insts) >= max_n:
            break
    return insts


async def _fetch_klines_4h(
    client: httpx.AsyncClient, inst_id: str, start_ms: int, end_ms: int,
) -> list[tuple[int, float, float, float, float]]:
    """Pull 4h candles. Returns list of (ts_ms, o, h, l, c)."""
    out: list[tuple[int, float, float, float, float]] = []
    cursor = end_ms
    while cursor > start_ms:
        r = await client.get(
            f"{OKX_REST}/api/v5/market/candles",
            params={"instId": inst_id, "bar": "4H",
                    "after": str(cursor), "limit": "100"},
        )
        if r.status_code != 200:
            logger.warning("OKX rejected %s: %s", inst_id, r.status_code)
            break
        rows = r.json().get("data", [])
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            if ts < start_ms:
                continue
            out.append((ts, float(row[1]), float(row[2]),
                        float(row[3]), float(row[4])))
        oldest = int(rows[-1][0])
        if oldest <= start_ms:
            break
        cursor = oldest
    return out


def _events_from_candles(
    symbol: str,
    candles: list[tuple[int, float, float, float, float]],
    *,
    min_move_pct: float,
) -> list[dict]:
    """Mark a candle as an event if |close - open| / open >= threshold."""
    events: list[dict] = []
    for row in candles:
        ts, o, h, lo, _c = row
        if o <= 0:
            continue
        pump = (h - o) / o
        dump = (o - lo) / o
        if max(pump, dump) >= min_move_pct:
            events.append({
                "symbol": symbol, "ts_ms": ts,
                "direction": "pump" if pump > dump else "dump",
                "magnitude_pct": round(max(pump, dump), 4),
            })
    return events


async def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="",
                        help="comma-separated symbols (e.g. RAVEUSDT,MYXUSDT)")
    parser.add_argument("--all-okx", action="store_true",
                        help="scan all OKX SWAP USDT-margined symbols")
    parser.add_argument("--max-symbols", type=int, default=30)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-move-pct", type=float, default=0.15,
                        help="minimum 4h candle move to count as an event")
    parser.add_argument("--out", default="events.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400_000

    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.all_okx:
            symbols_okx = await _list_okx_swaps(client, max_n=args.max_symbols)
            logger.info("Discovered %d OKX swaps", len(symbols_okx))
        else:
            sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
            if not sym_list:
                print("Either --symbols or --all-okx is required",
                      file=sys.stderr)
                return 2
            symbols_okx = [_to_okx_inst_id(s) for s in sym_list]

        all_events: list[dict] = []
        for inst in symbols_okx:
            try:
                candles = await _fetch_klines_4h(client, inst, start_ms, end_ms)
                # Convert OKX inst_id back to USDT-margin form for downstream tools.
                base = inst.replace("-USDT-SWAP", "")
                normalized = f"{base}USDT"
                evs = _events_from_candles(normalized, candles,
                                            min_move_pct=args.min_move_pct)
                if evs:
                    logger.info("  %s: %d events", inst, len(evs))
                all_events.extend(evs)
            except Exception as e:
                logger.warning("scan failed for %s: %s", inst, e)

    out = Path(args.out)
    out.write_text(json.dumps(all_events, indent=2))
    logger.info("Wrote %d events to %s", len(all_events), out)
    print(json.dumps({"events_written": len(all_events), "path": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
