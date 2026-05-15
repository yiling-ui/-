"""
T-D-06 — End-to-end integration demo (Pump/Long + Dump/Short).

Runs two complete lifecycle scenarios with synthetic data and prints a
human-readable trace at every stage:

    Flow A (PUMP/LONG):
        - Negative funding (short squeeze prelude)
        - OI silent build with mild upward price drift
        - Liquidity sweep below buy-side equal lows (failed breakdown)
        - Mock LLM verdict: pump, KOL frontrun_call, conf 0.92
        -> Fuser: high_priority LONG
        -> Risk Gate: approves
        -> Executor: market entry @ exchange + STOP_MARKET hard stop
        -> Trailing FSM: ARMED -> BREAKEVEN -> TRAILING (ratchets up)
        -> Exit on stop trigger

    Flow B (DUMP/SHORT):
        - Positive funding (long fragile, ripe for cascade)
        - OI silent build with mild DOWNWARD price drift
          (the key dump-front-run signal - distribution)
        - Liquidity sweep above sell-side equal highs (failed breakout)
        - Mock LLM verdict: dump, KOL exit_liquidity, conf 0.90
          (KOLs creating exit liquidity is EXACTLY what we want to short)
        -> Fuser: high_priority SHORT (despite KOL exit_liquidity)
        -> Risk Gate: approves
        -> Executor: market short + STOP_MARKET above
        -> Trailing FSM: ARMED -> BREAKEVEN -> TRAILING -> TARGET_REACHED at -70%
        -> Exit at hard cap

No network. No ccxt. No DeepSeek. Run:
    python examples/demo_e2e.py
"""

from __future__ import annotations

import asyncio

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.fuser import Direction, FusedSignal, ScoreFuser
from altcoin_agent.risk import (
    AccountState,
    CCXTExecutor,
    PositionSizer,
    RiskGate,
    TrailingStopFSM,
)
from altcoin_agent.risk.state import Side
from altcoin_agent.risk.trailing import TrailingState
from altcoin_agent.screener import SignalEvent, SignalKind

# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #


def banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def step(text: str) -> None:
    print(f"\n  >>> {text}")


def kv(label: str, value: object) -> None:
    print(f"      {label:.<28}{value}")


# --------------------------------------------------------------------------- #
# Mock exchange (in-memory) - same surface as the real ccxt one
# --------------------------------------------------------------------------- #


class MockExchange:
    """Captures every order. Returns deterministic fills near requested price."""

    def __init__(self) -> None:
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self.leverages: list[tuple[str, float]] = []
        self._next_id = 1

    def _id(self) -> str:
        v = f"ord-{self._next_id}"
        self._next_id += 1
        return v

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False):
        order = {
            "id": self._id(), "symbol": symbol, "side": side.value, "size": size,
            "price": price, "reduce_only": reduce_only, "average": price,
        }
        self.market_orders.append(order)
        suffix = "  (reduce-only)" if reduce_only else ""
        print(f"      [exchange] MARKET {side.value.upper():<5} {size:.4f} "
              f"{symbol} @ {price:.4f}{suffix}  id={order['id']}")
        return order

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True):
        order = {
            "id": self._id(), "symbol": symbol, "side": side.value, "size": size,
            "stop_price": stop_price, "reduce_only": reduce_only,
        }
        self.stop_orders.append(order)
        print(f"      [exchange] STOP-MARKET {side.value.upper():<5} {size:.4f} "
              f"{symbol} stop_price={stop_price:.4f}  id={order['id']}")
        return order

    async def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        print(f"      [exchange] CANCEL id={order_id}")
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):
        self.leverages.append((symbol, leverage))
        print(f"      [exchange] SET LEVERAGE {leverage:.1f}x for {symbol}")
        return {"symbol": symbol, "leverage": leverage}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ev(kind: SignalKind, *, ts: int, payload: dict, symbol: str,
        exchange: str = "binance") -> SignalEvent:
    return SignalEvent(kind=kind, symbol=symbol, ts=ts, exchange=exchange, payload=payload)


def _print_stop_change(price: float, label: str, state, new_stop) -> None:
    if new_stop is not None:
        print(f"      [tick @ ${price:.4f}, {label}]  state={state.value:<16} "
              f"new_stop={new_stop:.4f}")
    else:
        print(f"      [tick @ ${price:.4f}, {label}]  state={state.value:<16} "
              f"no change")


# --------------------------------------------------------------------------- #
# Flow A - PUMP / LONG
# --------------------------------------------------------------------------- #


async def flow_long_pump(account: AccountState, exchange: MockExchange) -> None:
    banner("FLOW A - PUMP / LONG  (RAVEUSDT)")

    SYMBOL = "RAVEUSDT"
    EXCHANGE = "binance"
    now = 1_700_000_000_000

    step("Step 1: Screener emits rule signals")
    rule_signals = [
        _ev(SignalKind.FUNDING_EXTREME, ts=now-15_000, symbol=SYMBOL,
            payload={"rate": -0.0022, "direction": "short_squeeze", "consecutive": 2}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-10_000, symbol=SYMBOL,
            payload={"oi_delta_pct": 0.22, "from_price": 1.000, "to_price": 1.005,
                     "from_oi": 1_000_000, "to_oi": 1_220_000}),
        _ev(SignalKind.VOLUME_SPIKE, ts=now-3_000, symbol=SYMBOL,
            payload={"side": "buy", "zscore": 5.8, "vol_ratio": 18.3}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now, symbol=SYMBOL,
            payload={"side": "buy_side", "wick_to_body": 2.4, "level": 0.985,
                     "bar_close": 1.012}),
    ]
    for s in rule_signals:
        kv(s.kind.value, s.payload)

    step("Step 2: DeepSeek (mocked) returns verdict")
    verdict = AIVerdict(
        intent="pump",
        confidence_score=92,
        reason="Negative funding -0.22% indicates crowded shorts. OI +22% with "
               "stable price = pre-pump accumulation. Buy-side liquidity sweep "
               "+ KOLs sharing entry signals before move.",
        kol_intent="frontrun_call",
        key_evidence=["funding -0.22%", "OI +22% in 5m", "sweep wick 2.4x body"],
    )
    kv("intent", verdict.intent)
    kv("confidence_score", verdict.confidence_score)
    kv("kol_intent", verdict.kol_intent)
    kv("reason", verdict.reason[:80] + "...")

    step("Step 3: Fuser combines rule + LLM signals")
    # Realistic order: LLM verdict (slow-path) is cached BEFORE the rule
    # confluence completes. We capture the first high_priority emission via
    # the sink — that's the signal that would have been dispatched to the bus.
    fired: list[FusedSignal] = []

    async def _sink(s: FusedSignal) -> None:
        fired.append(s)

    fuser = ScoreFuser(sink=_sink)
    await fuser.on_llm_verdict(EXCHANGE, SYMBOL, verdict, ts=now - 20_000)
    for sig in rule_signals:
        await fuser.on_rule_signal(sig)
    assert fired, "expected fuser to emit a high_priority signal"
    fused = fired[0]
    fused.trigger_price = 1.012
    kv("direction", fused.direction.value)
    kv("rule_score", f"{fused.rule_score:.2f}")
    kv("llm_score", f"{fused.llm_score:.2f}")
    kv("final_score", f"{fused.final_score:.2f}")
    kv("is_high_priority", fused.is_high_priority)
    for n in fused.notes:
        kv("note", n)
    assert fused.is_high_priority and fused.direction == Direction.LONG

    step("Step 4: Risk Gate evaluates")
    gate = RiskGate()
    decision = gate.evaluate(
        signal=fused,
        current_price=1.015,                  # +0.30% from trigger
        book_depth_usdt_top5=850_000.0,
        realized_volatility_pct=0.025,
        account=account,
        proposed_initial_stop=0.965,
    )
    for n in decision.notes:
        kv("check", n)
    kv("approved", decision.approved)
    if not decision.approved:
        kv("reason", decision.reason)
        return
    assert decision.intent is not None
    intent = decision.intent

    step("Step 5: Position sizing (risk-parity + dynamic leverage)")
    sizer = PositionSizer(max_risk_per_trade=0.015, contract_step=0.001)
    sizing = sizer.compute(
        intent=intent,
        equity_usdt=account.equity_usdt,
        realized_volatility_pct=0.025,
        book_depth_usdt_top5=850_000.0,
    )
    kv("leverage", f"{sizing.leverage:.2f}x")
    kv("risk_amount_usdt", f"{sizing.risk_amount_usdt:.2f}")
    kv("notional_usdt", f"{sizing.notional_usdt:.2f}")
    kv("size_contracts", f"{sizing.size_contracts:.4f}")
    kv("stop_distance", f"{sizing.stop_distance:.4f}")

    step("Step 6: Executor - market entry + exchange STOP_MARKET (SR-2)")
    executor = CCXTExecutor(adapter=exchange)
    result = await executor.open_with_hard_stop(intent=intent, sizing=sizing, account=account)
    kv("success", result.success)
    kv("fill_price", f"{result.fill_price:.4f}")
    kv("hard_stop_id", result.placed_stop_order_id)
    assert result.success and result.position is not None
    pos = result.position

    step("Step 7: Trailing FSM lifecycle (price ratchets up)")
    fsm = TrailingStopFSM(breakeven_r=1.0, trail_start_r=2.0, atr_multiplier=2.0)
    R = pos.stop_distance
    atr = 0.012

    price_a = pos.entry_price + 1.0 * R
    state, new_stop = fsm.tick(position=pos, current_price=price_a, atr=atr)
    _print_stop_change(price_a, "+1.0R", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    price_b = pos.entry_price + 2.0 * R
    state, new_stop = fsm.tick(position=pos, current_price=price_b, atr=atr)
    _print_stop_change(price_b, "+2.0R", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    price_c = pos.entry_price + 4.0 * R
    state, new_stop = fsm.tick(position=pos, current_price=price_c, atr=atr)
    _print_stop_change(price_c, "+4.0R", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    print(f"      [tick @ ${pos.current_hard_stop:.4f}, exit]  "
          f"-> exchange triggers STOP_MARKET, position closed")
    await exchange.market_order(pos.symbol, Side.SHORT, pos.size_contracts,
                                price=pos.current_hard_stop, reduce_only=True)
    pos.closed = True
    fsm.close()

    step("Step 8: PnL")
    realized_r = (pos.current_hard_stop - pos.entry_price) / R
    pnl_usdt = pos.size_contracts * (pos.current_hard_stop - pos.entry_price)
    kv("realized_R", f"{realized_r:+.2f}R")
    kv("realized_USDT", f"{pnl_usdt:+.2f}")
    kv("equity_before", f"{account.equity_usdt:.2f}")
    account.equity_usdt += pnl_usdt
    kv("equity_after", f"{account.equity_usdt:.2f}")


# --------------------------------------------------------------------------- #
# Flow B - DUMP / SHORT
# --------------------------------------------------------------------------- #


async def flow_short_dump(account: AccountState, exchange: MockExchange) -> None:
    banner("FLOW B - DUMP / SHORT  (MYXUSDT)")

    SYMBOL = "MYXUSDT"
    EXCHANGE = "binance"
    now = 1_700_001_000_000

    step("Step 1: Screener emits dump-front-run rule signals")
    rule_signals = [
        _ev(SignalKind.FUNDING_EXTREME, ts=now-15_000, symbol=SYMBOL,
            payload={"rate": 0.0012, "direction": "long_fragile", "consecutive": 2}),
        # OI rises 20% while price DROPS 0.5% - distribution / shorts piling in
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-10_000, symbol=SYMBOL,
            payload={"oi_delta_pct": 0.20, "from_price": 2.000, "to_price": 1.990,
                     "from_oi": 800_000, "to_oi": 960_000}),
        _ev(SignalKind.VOLUME_SPIKE, ts=now-3_000, symbol=SYMBOL,
            payload={"side": "sell", "zscore": 6.2, "vol_ratio": 22.1}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now, symbol=SYMBOL,
            payload={"side": "sell_side", "wick_to_body": 2.6, "level": 2.060,
                     "bar_close": 1.985}),
    ]
    for s in rule_signals:
        kv(s.kind.value, s.payload)

    step("Step 2: DeepSeek (mocked) returns dump verdict + KOL exit_liquidity")
    verdict = AIVerdict(
        intent="dump",
        confidence_score=90,
        reason="Positive funding +0.12% with long-fragile bias. OI +20% while "
               "price compressing at the highs = late-stage distribution. "
               "Failed breakout sweep + KOLs and bot-spam shilling = retail "
               "front-running the trap. Smart money distributing to retail.",
        kol_intent="exit_liquidity",
        key_evidence=["funding +0.12%", "OI +20% / price -0.5% = distribution",
                      "sell-side sweep wick 2.6x body", "low-follower bot spam"],
    )
    kv("intent", verdict.intent)
    kv("confidence_score", verdict.confidence_score)
    kv("kol_intent", verdict.kol_intent)
    kv("reason", verdict.reason[:80] + "...")

    step("Step 3: Fuser combines (KOL exit_liquidity reinforces SHORT)")
    fired: list[FusedSignal] = []

    async def _sink(s: FusedSignal) -> None:
        fired.append(s)

    fuser = ScoreFuser(sink=_sink)
    await fuser.on_llm_verdict(EXCHANGE, SYMBOL, verdict, ts=now - 20_000)
    for sig in rule_signals:
        await fuser.on_rule_signal(sig)
    assert fired, "expected fuser to emit a high_priority SHORT signal"
    fused = fired[0]
    fused.trigger_price = 1.985
    kv("direction", fused.direction.value)
    kv("rule_score", f"{fused.rule_score:.2f}")
    kv("llm_score", f"{fused.llm_score:.2f}")
    kv("final_score", f"{fused.final_score:.2f}")
    kv("is_high_priority", fused.is_high_priority)
    kv("blocked", fused.blocked)
    for n in fused.notes:
        kv("note", n)
    assert fused.is_high_priority and fused.direction == Direction.SHORT
    assert not fused.blocked, "SHORT alongside exit_liquidity must NOT be vetoed"

    step("Step 4: Risk Gate evaluates")
    gate = RiskGate()
    decision = gate.evaluate(
        signal=fused,
        current_price=1.978,                  # -0.35% from trigger (in our favor for short)
        book_depth_usdt_top5=750_000.0,
        realized_volatility_pct=0.030,
        account=account,
        proposed_initial_stop=2.080,
    )
    for n in decision.notes:
        kv("check", n)
    kv("approved", decision.approved)
    if not decision.approved:
        kv("reason", decision.reason)
        return
    assert decision.intent is not None
    intent = decision.intent
    assert intent.side == Side.SHORT

    step("Step 5: Position sizing")
    sizer = PositionSizer(max_risk_per_trade=0.015, contract_step=0.001)
    sizing = sizer.compute(
        intent=intent,
        equity_usdt=account.equity_usdt,
        realized_volatility_pct=0.030,
        book_depth_usdt_top5=750_000.0,
    )
    kv("leverage", f"{sizing.leverage:.2f}x  (capped at 10x for shorts)")
    kv("risk_amount_usdt", f"{sizing.risk_amount_usdt:.2f}")
    kv("notional_usdt", f"{sizing.notional_usdt:.2f}")
    kv("size_contracts", f"{sizing.size_contracts:.4f}")
    kv("stop_distance", f"{sizing.stop_distance:.4f}")

    step("Step 6: Executor - market SHORT entry + exchange STOP_MARKET above")
    executor = CCXTExecutor(adapter=exchange)
    result = await executor.open_with_hard_stop(intent=intent, sizing=sizing, account=account)
    kv("success", result.success)
    kv("fill_price", f"{result.fill_price:.4f}")
    kv("hard_stop_id", result.placed_stop_order_id)
    assert result.success and result.position is not None
    pos = result.position

    step("Step 7: Trailing FSM - waterfall dump, ATR ratchets stop DOWN")
    fsm = TrailingStopFSM(breakeven_r=1.0, trail_start_r=2.0, atr_multiplier=2.0,
                          short_target_cap_pct=0.70)
    R = pos.stop_distance
    atr = 0.040

    price_a = pos.entry_price - 1.0 * R
    state, new_stop = fsm.tick(position=pos, current_price=price_a, atr=atr)
    _print_stop_change(price_a, "+1.0R", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    price_b = pos.entry_price - 2.0 * R
    state, new_stop = fsm.tick(position=pos, current_price=price_b, atr=atr)
    _print_stop_change(price_b, "+2.0R", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    price_c = pos.entry_price * 0.50
    state, new_stop = fsm.tick(position=pos, current_price=price_c, atr=atr)
    _print_stop_change(price_c, "-50%", state, new_stop)
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    price_d = pos.entry_price * 0.29
    state, new_stop = fsm.tick(position=pos, current_price=price_d, atr=atr)
    _print_stop_change(price_d, "-71% TARGET_REACHED", state, new_stop)
    assert state == TrailingState.TARGET_REACHED
    if new_stop is not None:
        await executor.tighten_hard_stop(pos, new_stop)

    print(f"      [tick @ ${pos.current_hard_stop:.4f}, exit]  "
          f"-> exchange triggers STOP_MARKET BUY, short closed")
    await exchange.market_order(pos.symbol, Side.LONG, pos.size_contracts,
                                price=pos.current_hard_stop, reduce_only=True)
    pos.closed = True
    fsm.close()

    step("Step 8: PnL")
    realized_r = (pos.entry_price - pos.current_hard_stop) / R
    pnl_usdt = pos.size_contracts * (pos.entry_price - pos.current_hard_stop)
    kv("realized_R", f"{realized_r:+.2f}R")
    kv("realized_USDT", f"{pnl_usdt:+.2f}")
    kv("equity_before", f"{account.equity_usdt:.2f}")
    account.equity_usdt += pnl_usdt
    kv("equity_after", f"{account.equity_usdt:.2f}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


async def main() -> None:
    exchange = MockExchange()
    account = AccountState(equity_usdt=10_000.0)
    account.reconciliation_complete = True

    banner("STARTUP")
    print("  Account equity: $10,000.00")
    print("  Reconciler: complete (no orphans on mock exchange)")
    print("  Config: max_risk_per_trade=1.5%, daily_dd_limit=6%, "
          "max_concurrent=3, lev_long_max=15x, lev_short_max=10x")

    await flow_long_pump(account, exchange)
    await flow_short_dump(account, exchange)

    banner("FINAL ACCOUNT STATE")
    kv("equity_usdt", f"{account.equity_usdt:.2f}")
    kv("market orders sent", len(exchange.market_orders))
    kv("stop orders placed", len(exchange.stop_orders))
    kv("orders cancelled", len(exchange.cancelled))
    kv("leverage sets", len(exchange.leverages))


if __name__ == "__main__":
    asyncio.run(main())
