---
inclusion: always
---

# Survival Rules (always-loaded)

These four rules are the floor of the system. Any code change that weakens them
should be rejected at review.

## SR-1 — Dynamic Slippage Abort

Before any market entry, compare the current price against the `trigger_price`
recorded when the rule signal fired. If the adverse drift exceeds
`base_slippage / sqrt(leverage / 5)` (default base = 3%), abort the order.

| Leverage | Allowed adverse drift |
|----------|-----------------------|
| 5x       | 3.00% |
| 10x      | 2.12% |
| 15x      | 1.73% |

Favourable drift is not an abort condition.
Implementation: `risk/gate.py::RiskGate._dynamic_slippage_cap`.

## SR-2 — State Sync + Exchange-Side Hard Stop

  * On startup, the `Reconciler` MUST run to completion before the gate
    accepts any signal. Until `account.reconciliation_complete` is True
    every order is rejected with `reconciliation_pending`.
  * Every successful entry MUST be paired with an exchange-side
    `STOP_MARKET` (reduce_only). Soft stops in Python memory are
    forbidden. If the exchange-side stop placement fails, the position is
    market-closed immediately and the symbol is put on a 4-hour cooldown.
  * The trailing FSM may only TIGHTEN the stop (cancel + replace). It
    cannot widen it.

Implementation: `risk/executor.py::CCXTExecutor.open` and
`risk/executor.py::CCXTExecutor.tighten_hard_stop`.

## SR-3 — Wash-Trading Filter

A `volume_spike` event accompanied by either of:

  * `volume_zscore / count_zscore >= 2.0`  ("ghost volume")
  * `avg_trade_size_zscore >= 4.0`         ("whale single-print")

emits `WASH_TRADING_DETECTED`. The fuser hard-vetoes LONG signals when
this event is in the window. SHORT signals are NOT vetoed -- the fake pump
typically precedes a real dump.

Implementation: `screener.py::WashTradingDetector` and
`fuser.py::ScoreFuser.evaluate` step (1).

## SR-4 — Bot-Spam / Sybil Defense

The DeepSeek system prompt instructs the model to detect coordinated
homogeneous shilling -- emoji-only posts, "to the moon" template messages,
low-follower accounts posting in a tight time window. When the detector
fires, the LLM must:

  * lower `confidence_score` by an additional 15-25 points,
  * never emit `intent="pump"` with `confidence_score >= 70` unless the
    market features alone (funding, OI, sweep) independently justify it,
  * mark `kol_intent="exit_liquidity"` if the KOLs are also distributing.

Implementation: `ai_engine.py::SYSTEM_PROMPT` SR-4 paragraph.
