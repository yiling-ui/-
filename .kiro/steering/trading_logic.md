---
inclusion: always
description: Survival rules for the altcoin trading agent. Hard constraints — must never be violated by any code change.
---

# Trading Logic — Survival Rules

These 4 rules are extracted from real "got rugged" experience. They take
precedence over any strategy code. If a strategy proposal conflicts with one
of these, the strategy must change — not the rule.

The first 2 rules are HARD requirements for Task D code.
The last 2 are partly active (prompt) and partly future-iteration markers.

---

## SR-1 — Slippage Abort (extreme latency protection)

**Why**: signal -> AI judgement -> risk gate -> order can take 4–8s. On a
low-cap altcoin a 1s candle can move 5%. By the time we submit the order,
entry is at the top of the move.

**Rules every executor MUST follow:**

1. Every `FusedSignal` carries `trigger_price` (mid at signal time) and
   `trigger_ts`. Never trade without these.
2. Before sending a market order, refetch `current_price` (orderbook
   mid or last trade — NOT the cached `trigger_price`).
3. Compute `slippage = abs(current - trigger) / trigger`.
4. Threshold is **dynamic by leverage**:
   `max_slippage = base_slippage / sqrt(leverage / 5)` with `base = 3%`.
   Examples: 5x → 3.0%, 10x → 2.12%, 15x → 1.73%.
5. If `slippage > max_slippage`:
   - Default: **ABORT** the entry, emit `risk.alert.slippage_abort`.
   - Do NOT auto-fall-back to a passive limit order. Pullbacks often
     don't come on altcoins; passive limits become naked exposure.
   - The fallback-to-limit behavior is opt-in via
     `execution.slippage.fallback_to_limit=true` and OFF by default.
6. The slippage check applies to BOTH long entries and short entries.
   Symmetric.

---

## SR-2 — State Sync & Exchange Hard Stop

**Why**: process restart loses memory; an in-memory soft stop is worthless
if the process dies. The exchange must be the source of truth.

**Rules every executor MUST follow:**

1. **Startup reconciliation (first action of `Executor.start()`):**
   - `fetch_positions()` from every configured exchange.
   - Diff against local state.
   - Any position present on exchange but absent locally = ORPHAN.
   - Default treatment for orphan: place a breakeven stop and surface a
     human-actionable alert. Do NOT silently auto-close (might be a
     legitimate user position).
   - Cancel any open orders that have no local record.

2. **Every successful entry fill MUST be followed immediately by:**
   - `create_order(type="STOP_MARKET", reduceOnly=true, ...)` placed on the
     exchange. This is the HARD stop. It outlives the process.
   - Failure to place the hard stop is a **fail-closed** event:
     - Immediately market-close the position just opened.
     - Emit alert.
     - Cool down the symbol for 4 hours.
     - DO NOT leave naked delta hoping the next retry succeeds.

3. **Trailing stop (Python side) only ever TIGHTENS the hard stop:**
   - Move-up only via `cancel + replace` of the exchange stop.
   - Never replace the exchange stop with a software-only stop.
   - If `cancel + replace` fails, the previous (looser) hard stop must
     remain in force — never end up with no stop at all.

---

## SR-3 — Wash Trading Filter (future iteration)

**Status**: not active yet. Marked as task `T-A-07`. Code locations must
preserve a hook so this can land without ripping out the fuser.

**Algorithm spec for when implemented:**

- Track `trade_count_per_min` alongside volume.
- A real pump requires `zscore(trade_count) ≈ zscore(volume)` (within ~1σ).
- Volume spike with flat trade count = wash trading suspect.
- Cross-check: `avg_trade_size = volume / trade_count`.
  - Real altcoin pump: trade_count explodes, avg_trade_size dips (retail piles in).
  - Suspected wash: trade_count flat, avg_trade_size explodes (own-account ping-pong).
- Output: `wash_score` in `[0, 1]`. When `>= 0.7`, fuser applies
  `rule_score *= 0.5` and lifts the high-priority threshold to 95.

**Required code seam right now (Task A or D, whichever lands first):**
- `screener.py`: leave a `WashTradingFilter` Protocol class with a no-op
  default implementation.
- `fuser.py`: accept an optional `wash_filter` dependency in the
  constructor; default `None` keeps current behavior.

---

## SR-4 — Sybil / Astroturfing Filter

**Status**: prompt-side defense is **active now**. Feature-side
clustering and account profiling are future work (`T-B-07`, `T-C-06`).

**Active now (DeepSeek system prompt addition):**

> When multiple posts come from low-follower accounts (< 1000) and the
> texts are highly homogeneous (repeating emoji/hashtag combos, no
> specific thesis, just price calls), treat them as sybil/astroturf.
> Penalize `confidence_score` by at least 30 and bias `kol_intent`
> toward `exit_liquidity`.

**Future:**
- Feature-side: SimHash on recent posts, cluster similarity ≥ 0.85
  collapses to one effective vote.
- Account-side: pull author age, follower growth, post density. New +
  low-quality + simultaneous = sybil cluster.
- Fuser hook: when `sybil_density >= 0.6`, force `kol_intent` into the
  soft-cap branch (cap at 70, never high_priority).

---

## Code review checklist

Any PR touching `executor`, `risk`, `fuser`, `ai_engine`, or `screener`
MUST be reviewed against this file. Reviewers should explicitly mark which
of SR-1..SR-4 the PR touches and confirm none are weakened.
