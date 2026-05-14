"""
Quick standalone demo: feed synthetic OI + funding + volume + SMC sweep data
into the Screener and print the events. No network, no API key required.

Run:
    python examples/demo_mock.py
"""

from __future__ import annotations

import asyncio
import json

from altcoin_agent.screener import (
    FundingAnomalyDetector,
    FundingSnapshot,
    Kline,
    LiquidityPoolAnalyzer,
    OISnapshot,
    OISurgeDetector,
    Screener,
    SignalEvent,
    VolumeSpikeDetector,
)


async def main() -> None:
    received: list[SignalEvent] = []

    async def sink(ev: SignalEvent) -> None:
        received.append(ev)
        print(json.dumps(ev.as_dict(), indent=2, ensure_ascii=False))

    sc = Screener(
        exchanges=["binance"],
        symbols=["RAVEUSDT"],
        sink=sink,
        volume_detector=VolumeSpikeDetector(window=30, k_sigma=4.0, min_samples=20),
        funding_detector=FundingAnomalyDetector(consecutive=2),
        oi_detector=OISurgeDetector(window=3, surge_pct=0.15, silent_max_price_move=0.01),
        liquidity_analyzer=LiquidityPoolAnalyzer(),
    )

    print("=== feeding 25 calm 1m bars then a 50x volume spike ===")
    for i in range(25):
        await sc.on_kline("binance", "RAVEUSDT", Kline(
            ts=i * 60_000, open=1.0, high=1.005, low=0.995, close=1.001, volume=100.0 + (i % 3),
        ))
    await sc.on_kline("binance", "RAVEUSDT", Kline(
        ts=26 * 60_000, open=1.001, high=1.05, low=1.000, close=1.045, volume=5_000.0,
    ))

    print("=== feeding 3 calm OI samples then +25% with flat price (silent build) ===")
    for i in range(3):
        await sc.on_oi("binance", OISnapshot(ts=i * 60_000, symbol="RAVEUSDT",
                                             open_interest=1_000_000, price=1.045))
    await sc.on_oi("binance", OISnapshot(ts=4 * 60_000, symbol="RAVEUSDT",
                                         open_interest=1_250_000, price=1.046))

    print("=== feeding two consecutive extreme negative funding samples ===")
    await sc.on_funding("binance", FundingSnapshot(ts=1, symbol="RAVEUSDT", rate=-0.0022))
    await sc.on_funding("binance", FundingSnapshot(ts=2, symbol="RAVEUSDT", rate=-0.0028))

    print(f"\n=== TOTAL EVENTS: {len(received)} ===")
    for ev in received:
        print(f"  - {ev.kind.value} on {ev.symbol} @ {ev.ts}")


if __name__ == "__main__":
    asyncio.run(main())
