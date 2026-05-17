# Altcoin Momentum Agent V1.0

Multimodal AI trading agent for capturing early altcoin pumps and dumps,
combining real-time market microstructure (volume / OI / funding / SMC
liquidity), pluggable LLM intent analysis (DeepSeek by default; OpenAI /
Claude / OpenRouter / Moonshot / Qwen / any OpenAI-compatible endpoint
supported), social-feed signals, and a self-evolving rule library that
learns from every confirmed pump/dump event.

## Quick start (dry run, no real orders)

```bash
git clone https://github.com/yiling-ui/-.git altcoin-agent && cd altcoin-agent
cp .env.example .env             # DRY_RUN=true is the default
docker compose up --build -d
curl http://localhost:8080/healthz
open http://localhost:8080/dashboard
docker compose logs -f
```

The first run creates `.kiro/steering/dynamic_rules.json` and `.md` on the
fly. The fuser hot-reloads them every 5s, so the Learning Engine's output
takes effect without restarting the daemon.

## Operating modes

| Mode | How to enable | What it does |
|---|---|---|
| **DRY-RUN** (default) | `DRY_RUN=true` | logs every intended order; no exchange calls |
| **PAPER-TRADE** | `DRY_RUN=false` + `PAPER_TRADE=true` | real ccxt orders against the testnet/sandbox of the chosen venue |
| **LIVE** | `DRY_RUN=false` + `PAPER_TRADE=false` + valid API keys | real ccxt orders against mainnet |

## LLM provider — pluggable

Set `LLM_PROVIDER` in `.env` and provide the matching `<BACKEND>_API_KEY`:

| `LLM_PROVIDER` | Key env | Default model |
|---|---|---|
| `deepseek` (default) | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-3.5-sonnet` |
| `moonshot` | `MOONSHOT_API_KEY` | `moonshot-v1-8k` |
| `qwen` | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20240620` |
| `generic` | `LLM_API_KEY` + `LLM_API_BASE` + `LLM_MODEL` | (your own) |

Override the model with `LLM_MODEL` env at any time.

## Web dashboard

Set `DASHBOARD_ENABLED=true` (default) and visit `http://localhost:8080/dashboard`.

The page polls every 5 seconds and shows:

* System state (uptime, mode, reconciliation, error counters)
* Open positions (symbol / side / entry / current stop / leverage)
* Last 50 high-priority signals (sym / dir / score / trigger / notes)
* Last 50 orders sent (dry-run or live)
* Last 50 risk-gate rejections
* Top 20 dynamic-rule entries from `dynamic_rules.json` sorted by Bayesian-smoothed hit rate

JSON endpoints:

```
GET /healthz
GET /api/state         /api/signals     /api/positions
/api/orders            /api/rejections  /api/rules
```

## Telegram notifier

Set in `.env`:

```
TG_ENABLED=true
TG_BOT_TOKEN=<from @BotFather>
TG_CHAT_ID=<channel/group/user id>
```

Pushes structured cards on every:

* `📡 SIGNAL` — fused high-priority signal observed (with score, direction, rule kinds)
* `🟢 OPENED` — order placed, position opened (size, leverage, stop)
* `⚠️ REJECTED` — risk gate rejected a signal (with reason)
* `🚨 ERROR` — critical error in the daemon

Telegram failures (network down, bot rate-limited, 5xx) are swallowed —
the trading loop never pauses on the notifier.

## 30-day backtest pipeline

Three steps. End-to-end, the agent learns from history without manual
labelling:

```bash
# 1) Discover historical pump/dump events from OKX 4h candles
python scripts/discover_events.py \
    --symbols RAVEUSDT,MYXUSDT,PEPEUSDT \
    --days 30 --min-move-pct 0.15 \
    --out events.json

# Or scan all OKX swaps:
python scripts/discover_events.py --all-okx \
    --days 30 --min-move-pct 0.20 --max-symbols 30 --out events.json

# 2) Run post-mortem over each event (cached locally so re-runs are fast)
python scripts/backtest_30d.py events.json --use-llm

# 3) Inspect the converged rule library
cat .kiro/steering/dynamic_rules.json | jq '.rules | sort_by(-(.hits+1)/(.total+2))[0:10]'
# Or open http://localhost:8080/dashboard while the daemon is running
```

The fuser auto-loads `dynamic_rules.json` within 5 seconds of any change;
you do NOT need to restart the daemon between backtest passes.

## Architecture

```
                 ┌───── ccxt.pro (binance / okx / gateio) ─────┐
                 │                                             │
              klines, funding, OI                            trades
                 │                                             │
                 ▼                                             ▼
            Screener (4 detectors + WashTradingDetector)   TradeFlowAggregator
                 │                                             │
                 │  SignalEvent              Kline (with trade_count)
                 ▼                                             │
            asyncio.Queue ──────────────────────────────────► (same hot path)
                 │
                 ▼
            ScoreFuser ◄── dynamic_rules.json (mtime-throttled hot reload)
                 │      ◄── LLMEngine.judge() (pluggable backend, async)
                 │
                 ▼  FusedSignal (high-priority)
            RiskGate (9 fail-closed checks: SR-1..SR-4)
                 │
                 ▼  RiskDecision
            CCXTExecutor ── enters position + STOP_MARKET pair
                 │
                 ▼
            TrailingController (ARMED → BREAKEVEN → TRAILING)
                 │
                 ▼
            ATRCalculator (online, per (exchange, symbol, tf))

            ─── parallel ───
            Learning Engine (post-mortem) ──► dynamic_rules.json/.md
                                                    ↑ SliceCache
            Social Aggregator (Binance Square primary + OKX/Dex/Gecko aux)
            Notifier (Telegram) + Dashboard (aiohttp + vanilla HTML)
```

## Survival rules (auto-loaded into all LLM contexts)

See [`.kiro/steering/trading_logic.md`](.kiro/steering/trading_logic.md):

* **SR-1 dynamic slippage abort** — `base_slippage / sqrt(leverage / 5)`
* **SR-2 state sync + exchange-side hard stop** — reconciler at startup,
  every entry pairs with a `STOP_MARKET` (`reduce_only=True`); the trailing
  FSM may only TIGHTEN the stop.
* **SR-3 wash-trading filter** — ghost-volume + whale-single-print
  patterns hard-veto LONG (shorts allowed).
* **SR-4 bot-spam / sybil defense** — LLM is instructed to detect
  coordinated homogeneous shilling and cap confidence accordingly.

## LLM verdict is advisory, not gating (TICKET-014)

The fuser's high-priority gate fires on the **rule score alone**. When
`rule_score >= high_priority_threshold` (default 85) the `FusedSignal`
is dispatched and the executor places the order on the next loop tick;
the daemon does **not** wait for the LLM to weigh in.

Reasoning: typical altcoin pump bursts move 5%+ inside 500ms, well
inside any realistic LLM round-trip. Waiting for the LLM would forfeit
the winning entry. An LLM verdict that arrives **after** the order has
landed cannot retract the trade — its only effects are:

* the trailing FSM continues to manage the position via the
  exchange-side `STOP_MARKET`, capping the worst case;
* the next signal on the same symbol sees the LLM-modulated
  `final_score` (so a strongly-disagreeing post-arrival verdict
  damps the *next* high-priority promotion);
* the close-event-driven post-mortem learning loop persists the
  feature combination's hit/miss so future occurrences score lower.

Operational consequence: do not interpret a "REJECTED by LLM" log line
that appears after an `OPENED` line as a missed bug. It's the
documented behaviour. If you need rules-AND-LLM-must-agree semantics,
that is a separate `StrictFuser` build and explicitly out of scope for
V1.0.

## Operational health metrics (TICKET-016)

The dashboard's `/api/state` and the Prometheus `/metrics` endpoint
both expose a small set of leading-indicator gauges so an on-call
operator can triage in seconds:

| Metric | Leading indicator for |
|---|---|
| `persistor_save_failures` | Persistence layer is dropping writes (daily-DD breaker may not survive a restart) |
| `position_watcher_lag_sec` | Phantom positions (close not detected) |
| `llm_degraded_count` | LLM provider outage / budget exhaustion |
| `emergency_close_count` | Venue / network instability driving forced closes |
| `stop_replace_failure_count` | Trailing tighten misses (precedes naked-position emergency closes) |

## Module map

| File | Responsibility |
|---|---|
| `src/altcoin_agent/screener.py` | ccxt.pro fan-in + 5 detectors |
| `src/altcoin_agent/screener_extras.py` | TradeFlowAggregator |
| `src/altcoin_agent/fuser.py` | ScoreFuser + hot-loaded RuleIndex + asymmetric Laplace-shrunk learned multipliers, 1.30x reward cap |
| `src/altcoin_agent/llm_provider.py` | Pluggable LLM backends (DeepSeek / OpenAI / Anthropic / etc.) |
| `src/altcoin_agent/ai_engine.py` | LLMEngine: strict JSON schema, retry-once-then-degrade, monthly token budget |
| `src/altcoin_agent/learning_engine.py` | Slicer, 8 closed-set features, post-mortem, RuleStore (Bayesian) |
| `src/altcoin_agent/slice_cache.py` | SQLite cache for OKX historical slices |
| `src/altcoin_agent/risk/*` | gate / sizing / trailing / atr / executor / reconciler / ccxt_adapter |
| `src/altcoin_agent/social/*` | binance_square scraper + multi-source aggregator |
| `src/altcoin_agent/notifier/*` | Telegram bot + NullNotifier |
| `src/altcoin_agent/dashboard.py` | Live web dashboard (aiohttp + vanilla JS) |
| `src/altcoin_agent/main.py` | V1 daemon: /healthz + /dashboard, 3 modes |
| `scripts/discover_events.py` | Auto-discover historical pump/dump events |
| `scripts/backtest_30d.py` | Run Learning Engine over an event list |

## Tests

```bash
pip install -e '.[dev]'
pytest -q                # 134 passing
ruff check .             # clean
```
