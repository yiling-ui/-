# Altcoin Momentum Agent V1.0

Multimodal AI trading agent for capturing early altcoin pumps and dumps,
combining real-time market microstructure (volume / OI / funding / SMC
liquidity), DeepSeek LLM intent analysis, social-feed signals, and a
self-evolving rule library.

## Quick start (dry run, no real orders)

```bash
git clone https://github.com/yiling-ui/-.git altcoin-agent && cd altcoin-agent
cp .env.example .env             # DRY_RUN=true is the default
docker compose up --build -d
curl http://localhost:8080/healthz
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

`main.py` refuses to start LIVE/PAPER if API credentials are missing.

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
                 │      ◄── DeepSeekEngine.judge() (slow path, async)
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
            Social Aggregator (Binance Square primary + OKX/Dex/Gecko aux)
```

## Survival rules (auto-loaded into all LLM contexts)

See [`.kiro/steering/trading_logic.md`](.kiro/steering/trading_logic.md):

* **SR-1 dynamic slippage abort** — `base_slippage / sqrt(leverage / 5)`
* **SR-2 state sync + exchange-side hard stop** — reconciler at startup,
  every entry pairs with a `STOP_MARKET` (`reduce_only=True`); the trailing
  FSM may only TIGHTEN the stop.
* **SR-3 wash-trading filter** — ghost-volume + whale-single-print
  patterns hard-veto LONG (shorts allowed; fake pumps usually precede
  real dumps).
* **SR-4 bot-spam / sybil defense** — DeepSeek is instructed to detect
  coordinated homogeneous shilling and cap confidence accordingly.

## Module map

| File | Responsibility |
|---|---|
| `src/altcoin_agent/screener.py` | ccxt.pro fan-in + 5 detectors (Volume / Funding / OI / SMC liquidity / WashTrading) |
| `src/altcoin_agent/screener_extras.py` | TradeFlowAggregator: aggregates `watch_trades` into Klines with `trade_count` |
| `src/altcoin_agent/fuser.py` | ScoreFuser + hot-loaded RuleIndex + asymmetric Laplace-shrunk learned multipliers, 1.30x reward cap |
| `src/altcoin_agent/ai_engine.py` | DeepSeek client with strict JSON schema, retry-once-then-degrade, monthly token budget |
| `src/altcoin_agent/learning_engine.py` | Historical slicer, 8 closed-set features, post-mortem, RuleStore (Bayesian, JSON+MD) |
| `src/altcoin_agent/risk/gate.py` | 9 fail-closed checks (incl. SR-1) |
| `src/altcoin_agent/risk/sizing.py` | Risk-parity sizing + dynamic leverage (5..15 long, 5..10 short) |
| `src/altcoin_agent/risk/executor.py` | Order placement with hard-stop pairing, retry, fail-closed close |
| `src/altcoin_agent/risk/trailing.py` | TrailingStopFSM (ARMED → BREAKEVEN → TRAILING + SHORT -70% target cap) |
| `src/altcoin_agent/risk/atr.py` | Online ATR calculator |
| `src/altcoin_agent/risk/reconciler.py` | Startup state alignment + emergency-stop on orphans |
| `src/altcoin_agent/risk/ccxt_adapter.py` | Live ExchangeAdapter backed by ccxt.pro (Binance / OKX / Gate.io) |
| `src/altcoin_agent/social/binance_square.py` | Binance Square scraper (cookie + proxy injection, multi-endpoint waterfall, public CMS fallback) |
| `src/altcoin_agent/social/crawler.py` | Multi-source aggregator (Square primary; OKX / DexScreener / CoinGecko aux) |
| `src/altcoin_agent/main.py` | V1 daemon with /healthz, SIGINT/SIGTERM, DRY-RUN / PAPER / LIVE modes |
| `scripts/backtest_30d.py` | 30-day post-mortem backtest using Learning Engine |

## Running the 30-day backtest

```bash
# 1) Auto-generate events for a symbol list, dated 24h ago
python scripts/backtest_30d.py --auto-events --symbols RAVEUSDT,MYXUSDT

# 2) Or feed a curated event list (covers a week of past pumps / dumps)
cat events.json | python scripts/backtest_30d.py - --use-llm
```

The script writes/updates `.kiro/steering/dynamic_rules.json` and the
fuser picks up the new rules within 5 seconds.

## Tests

```bash
pip install -e '.[dev]'
pytest -q                # 101 passing, ~3s
ruff check .             # clean
```
