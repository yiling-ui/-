"""sim_pump_cycle.py — Drive the real RiskGate / PriceTape / AccountState
through the full pump-and-dump cycle the operator drew on the chart.

We do NOT mock the trading code. The simulation:

  * Constructs the actual ``PositionSizer``, ``RiskGate``, ``PriceTape``,
    ``RegimeFilter``, ``ClusterMap`` from the production package.
  * Walks the daily candle path 0.22 -> 28.30 -> 0.55 sample by sample.
  * Calls ``RiskGate.evaluate`` at each candidate entry, just like
    ``main.App`` would do.
  * Walks ATR / ratchet stop logic in pure Python so we can read the
    journal.
  * Prints a per-bar trade journal at the end with realised PnL.

Starting capital: 500 USDT.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk.cluster import ClusterCapConfig, ClusterMap
from altcoin_agent.risk.gate import RiskGate, RiskGateConfig
from altcoin_agent.risk.regime_filter import RegimeFilter, RegimeFilterConfig
from altcoin_agent.risk.sizing import DynamicLeverageConfig, PositionSizer
from altcoin_agent.risk.state import AccountState, Position, Side


# --------------------------------------------------------------------- #
# Synthetic candle path that matches the operator's chart
# --------------------------------------------------------------------- #
#
# 1 candle = 1 day. Roughly:
#   day 0..3   : flat bottom around 0.23 (accumulation)
#   day 4..7   : slow climb 0.30 -> 1.50
#   day 8..10  : parabolic 3 -> 12 -> 22
#   day 11     : top wick to 28.30 close 17 (the red doji on the chart)
#   day 12     : crash to 5
#   day 13..16 : bleed 5 -> 1.5 -> 0.7 -> 0.55

@dataclass
class Bar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float       # USDT volume — informs liquidity
    label: str = ""


CANDLES: list[Bar] = []
DAY_MS = 24 * 3600 * 1000
T0 = 1_700_000_000_000  # arbitrary anchor

# accumulation
for i, c in enumerate([0.228, 0.230, 0.225, 0.232]):
    CANDLES.append(Bar(
        ts_ms=T0 + i * DAY_MS, open=c, high=c * 1.02, low=c * 0.98, close=c,
        volume=300_000, label="accumulation",
    ))

# slow ramp
for i, c in enumerate([0.30, 0.55, 0.95, 1.50], start=4):
    CANDLES.append(Bar(
        ts_ms=T0 + i * DAY_MS, open=c * 0.85, high=c * 1.10, low=c * 0.80,
        close=c, volume=2_000_000, label="ramp",
    ))

# parabolic
CANDLES.append(Bar(
    ts_ms=T0 + 8 * DAY_MS, open=1.6, high=4.0, low=1.6, close=3.5,
    volume=20_000_000, label="parabolic",
))
CANDLES.append(Bar(
    ts_ms=T0 + 9 * DAY_MS, open=3.5, high=14.0, low=3.4, close=12.0,
    volume=120_000_000, label="parabolic",
))
CANDLES.append(Bar(
    ts_ms=T0 + 10 * DAY_MS, open=12.0, high=23.0, low=11.5, close=22.0,
    volume=300_000_000, label="parabolic",
))

# blow-off top — wick to 28.30, close at 17 (matches the red doji)
CANDLES.append(Bar(
    ts_ms=T0 + 11 * DAY_MS, open=22.0, high=28.30, low=15.0, close=17.0,
    volume=500_000_000, label="blow-off-top",
))

# crash bar — exact mirror of the chart: massive red, wick down
CANDLES.append(Bar(
    ts_ms=T0 + 12 * DAY_MS, open=17.0, high=18.0, low=4.5, close=5.2,
    volume=400_000_000, label="crash",
))

# bleed
for i, (o, h, l, c) in enumerate([
    (5.2, 5.5, 1.4, 1.6),
    (1.6, 1.8, 0.7, 0.75),
    (0.75, 0.80, 0.50, 0.60),
    (0.60, 0.62, 0.50, 0.548),
], start=13):
    CANDLES.append(Bar(
        ts_ms=T0 + i * DAY_MS, open=o, high=h, low=l, close=c,
        volume=80_000_000, label="bleed",
    ))


# --------------------------------------------------------------------- #
# Helpers — build a HighPriority signal that the gate would otherwise pass
# --------------------------------------------------------------------- #


def build_signal(symbol: str, side: Side, score: float, ts_ms: int,
                 trigger: float) -> FusedSignal:
    return FusedSignal(
        symbol=symbol,
        exchange="binance",
        ts=ts_ms,
        direction=Direction.LONG if side == Side.LONG else Direction.SHORT,
        rule_score=score - 5,
        llm_score=score - 10,
        final_score=score,
        is_high_priority=True,
        blocked=False,
        block_reason=None,
        trigger_price=trigger,
    )


@dataclass
class JournalEntry:
    day: int
    bar_label: str
    event: str
    price: float
    detail: str = ""


JOURNAL: list[JournalEntry] = []


def log(day: int, bar: Bar, event: str, detail: str = "") -> None:
    JOURNAL.append(JournalEntry(day, bar.label, event, bar.close, detail))


# --------------------------------------------------------------------- #
# Run the simulation
# --------------------------------------------------------------------- #


def run() -> None:
    SYMBOL = "MEME/USDT:USDT"

    account = AccountState(
        equity_usdt=500.0,
        starting_equity_today_usdt=500.0,
    )
    account.reconciliation_complete = True

    # Production sizer with the project defaults: 1.5% risk per trade,
    # dynamic leverage 5..15x for LONG, 5..10x for SHORT.
    sizer = PositionSizer(
        max_risk_per_trade=0.015,
        leverage_cfg=DynamicLeverageConfig(
            min_leverage=5.0,
            max_leverage_long=15.0,
            max_leverage_short=10.0,
            target_vol_pct=0.05,
            liq_full_depth_usdt=200_000.0,
            score_anchor=85.0,
        ),
    )

    gate = RiskGate(
        sizer,
        RiskGateConfig(
            min_liquidity_usdt=200_000.0,
            base_slippage=0.03,
            max_consecutive_losses=2,
            consecutive_loss_cooldown_sec=4 * 3600,
            symbol_cooldown_sec=60,
            max_concurrent_positions=3,
            daily_drawdown_limit=0.06,
            daily_stoploss_hits_max=3,
        ),
    )

    # Anti-chase / vol-kill: with 30s window + 2.5% cap on a DAILY
    # candle stream, we feed the close of the previous candle as the
    # "30s ago" sample so the tape has a meaningful baseline for the
    # ramp candles. (In production this fires on second-by-second
    # marks; here we adapt sampling to the candle granularity.)
    tape = PriceTape(cfg=PriceTapeConfig(
        anti_chase_window_ms=30_000,
        anti_chase_max_move_pct=0.025,
        vol_kill_window_ms=60_000,
        vol_kill_range_pct=0.08,
    ))

    # Regime filter pointed at a "BTC" stream we never observe ->
    # cold tape -> fail-open, which is the right default for this
    # simulation (the operator hasn't told us the BTC regime).
    regime = RegimeFilter(RegimeFilterConfig(reference_symbol="BTC"))

    # Cluster cap for memes: only 1 meme position at a time.
    cluster_map = ClusterMap({"MEME": "meme"})
    cluster_cap = ClusterCapConfig(enabled=True, max_per_cluster=1)

    # Position bookkeeping for the simulator (the daemon's executor
    # would track these via the actual exchange). We simulate in
    # one_position_at_a_time mode mirroring the cluster cap of 1.
    pos: Position | None = None
    realised_pnl_total = 0.0
    fills: list[dict] = []

    print(f"=== Simulation start: equity = {account.equity_usdt:.2f} USDT ===\n")

    for day, bar in enumerate(CANDLES):
        # Feed the price tape: simulate the 30s sample as
        # (ts - 30s, prev_close) and the live tick as (ts, current_high
        # midpoint). For anti-chase we sample at the bar OPEN.
        if day > 0:
            prev_close = CANDLES[day - 1].close
            tape.observe(SYMBOL, prev_close, ts_ms=bar.ts_ms - 30_000)
        tape.observe(SYMBOL, bar.open, ts_ms=bar.ts_ms)

        # ------------------------------------------------------------ #
        # 1) Manage existing position first: trailing / stop-out
        # ------------------------------------------------------------ #
        if pos is not None and not pos.closed:
            # ratchet stop:
            #   LONG  -> raise stop to max(current_stop, low - 1*ATR_proxy)
            #   SHORT -> lower stop to min(current_stop, high + 1*ATR_proxy)
            atr_proxy = bar.high - bar.low
            if pos.side == Side.LONG:
                # Did the day's LOW take out our stop?
                if bar.low <= pos.current_stop:
                    fill_px = pos.current_stop  # assume stop fills at the level
                    pnl = (fill_px - pos.entry_price) * pos.size
                    account.equity_usdt += pnl
                    realised_pnl_total += pnl
                    fills.append({
                        "day": day, "side": "LONG_EXIT", "price": fill_px,
                        "pnl": pnl, "reason": "stop_hit",
                    })
                    log(day, bar, "STOP HIT (LONG)",
                        f"fill={fill_px:.4f} pnl={pnl:+.2f} "
                        f"equity={account.equity_usdt:.2f}")
                    pos.closed = True
                    pos = None
                else:
                    # Trail using the bar's CLOSE as the latest mark.
                    candidate = bar.close - 1.5 * atr_proxy
                    if candidate > pos.current_stop:
                        log(day, bar, "trail tighten LONG",
                            f"{pos.current_stop:.4f} -> {candidate:.4f}")
                        pos.current_stop = candidate
            else:  # SHORT
                if bar.high >= pos.current_stop:
                    fill_px = pos.current_stop
                    pnl = (pos.entry_price - fill_px) * pos.size
                    account.equity_usdt += pnl
                    realised_pnl_total += pnl
                    fills.append({
                        "day": day, "side": "SHORT_EXIT", "price": fill_px,
                        "pnl": pnl, "reason": "stop_hit",
                    })
                    log(day, bar, "STOP HIT (SHORT)",
                        f"fill={fill_px:.4f} pnl={pnl:+.2f} "
                        f"equity={account.equity_usdt:.2f}")
                    pos.closed = True
                    pos = None
                else:
                    candidate = bar.close + 1.5 * atr_proxy
                    if candidate < pos.current_stop:
                        log(day, bar, "trail tighten SHORT",
                            f"{pos.current_stop:.4f} -> {candidate:.4f}")
                        pos.current_stop = candidate

        # If the stop fired, sync open_positions for the gate's
        # cluster-cap and concurrency checks.
        if pos is None:
            account.open_positions.clear()

        # ------------------------------------------------------------ #
        # 2) Decide whether to open a new position this bar
        # ------------------------------------------------------------ #
        candidate_side: Side | None = None
        candidate_score = 0.0
        candidate_stop = 0.0
        candidate_trigger = bar.open

        if bar.label == "accumulation":
            # Volume too thin in real life — skip entirely.
            log(day, bar, "no entry", "accumulation: volume too thin")
            continue

        if bar.label == "ramp" and pos is None:
            # Volume-spike LONG. Score scales with how much the ramp moved.
            candidate_side = Side.LONG
            candidate_score = 88.0 + min(7.0, (bar.close / bar.open - 1.0) * 30)
            candidate_stop = bar.open * 0.95     # 5% below entry
            candidate_trigger = bar.open

        elif bar.label == "parabolic" and pos is None:
            # The strategy WANTS to chase here; anti-chase should bite.
            candidate_side = Side.LONG
            candidate_score = 95.0
            candidate_stop = bar.open * 0.95
            candidate_trigger = bar.open

        elif bar.label == "blow-off-top" and pos is None:
            # First candle where SHORT becomes attractive — a long upper
            # wick + close well below the high.
            candidate_side = Side.SHORT
            candidate_score = 92.0
            # Stop above the top wick by 5%.
            candidate_stop = bar.high * 1.05
            candidate_trigger = bar.open

        elif bar.label == "crash":
            # We probably already SHORT; if not, the strategy still
            # tries to enter SHORT mid-cascade — vol-kill should bite.
            if pos is None:
                candidate_side = Side.SHORT
                candidate_score = 90.0
                candidate_stop = bar.open * 1.05
                candidate_trigger = bar.open

        elif bar.label == "bleed" and pos is None:
            # Bear-flag continuation SHORTs. Lower score because trend
            # has matured; cluster cap / consecutive losses may bite.
            candidate_side = Side.SHORT
            candidate_score = 87.0
            candidate_stop = bar.open * 1.07
            candidate_trigger = bar.open

        if candidate_side is None:
            continue

        signal = build_signal(
            symbol=SYMBOL,
            side=candidate_side,
            score=candidate_score,
            ts_ms=bar.ts_ms,
            trigger=candidate_trigger,
        )

        # Liquidity proxy: a fraction of daily USDT volume sits at top-5
        # at any given second. 0.5% is a generous estimate.
        top5_depth = bar.volume * 0.005
        # Realised vol proxy: today's range / open.
        realised_vol = (bar.high - bar.low) / max(bar.open, 1e-9)

        decision = gate.evaluate(
            signal=signal,
            account=account,
            current_price=bar.open,
            top5_depth_usdt=top5_depth,
            realized_vol_pct=realised_vol,
            initial_stop=candidate_stop,
            now_ms=bar.ts_ms,
            price_tape=tape,
            regime_filter=regime,
            cluster_map=cluster_map,
            cluster_cap_cfg=cluster_cap,
        )

        if not decision.approved:
            log(day, bar, f"REJECT {candidate_side.value}",
                f"reason={decision.reason} "
                f"score={candidate_score:.0f} depth={top5_depth:,.0f} "
                f"vol={realised_vol:.3f}")
            continue

        # Approved! Open the position.
        pos = Position(
            symbol=SYMBOL,
            exchange="binance",
            side=candidate_side,
            entry_price=bar.open,
            size=decision.size,
            leverage=decision.leverage,
            initial_stop=candidate_stop,
            current_stop=candidate_stop,
            stop_order_id=f"sim-{day}",
        )
        account.open_positions[SYMBOL] = pos
        fills.append({
            "day": day,
            "side": f"{candidate_side.value.upper()}_OPEN",
            "price": bar.open,
            "size": decision.size,
            "leverage": decision.leverage,
            "notional": decision.notional_usdt,
            "stop": candidate_stop,
        })
        log(day, bar, f"OPEN {candidate_side.value}",
            f"entry={bar.open:.4f} size={decision.size:.2f} "
            f"lev={decision.leverage:.2f}x "
            f"notional={decision.notional_usdt:.2f} "
            f"stop={candidate_stop:.4f} risk_amt={decision.risk_amount_usdt:.2f}")

    # Force-close anything still open at end of simulation.
    if pos is not None and not pos.closed:
        last_bar = CANDLES[-1]
        if pos.side == Side.LONG:
            pnl = (last_bar.close - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - last_bar.close) * pos.size
        account.equity_usdt += pnl
        realised_pnl_total += pnl
        fills.append({
            "day": len(CANDLES) - 1, "side": f"{pos.side.value.upper()}_CLOSE_EOD",
            "price": last_bar.close, "pnl": pnl, "reason": "end_of_sim",
        })
        log(len(CANDLES) - 1, last_bar, f"FORCE CLOSE {pos.side.value}",
            f"px={last_bar.close:.4f} pnl={pnl:+.2f}")

    # ------------------------------------------------------------ #
    # Print the journal
    # ------------------------------------------------------------ #
    print("--- Per-bar journal ---\n")
    for j in JOURNAL:
        print(f"day {j.day:2d} [{j.bar_label:13s}] @ {j.price:7.3f}  "
              f"{j.event:30s}  {j.detail}")

    print("\n--- Fills ---\n")
    for f in fills:
        kv = "  ".join(f"{k}={v}" for k, v in f.items() if k != "side")
        print(f"  {f['side']:18s}  {kv}")

    print("\n--- Result ---")
    print(f"  starting equity : 500.00 USDT")
    print(f"  ending equity   : {account.equity_usdt:.2f} USDT")
    print(f"  realised PnL    : {realised_pnl_total:+.2f} USDT "
          f"({realised_pnl_total / 5.0:+.2f}% on equity)")
    print(f"  trades          : {sum(1 for f in fills if 'OPEN' in f['side'])}")
    stops_hit = sum(1 for f in fills if f.get('reason') == 'stop_hit')
    print(f"  stops hit       : {stops_hit}")


if __name__ == "__main__":
    run()
