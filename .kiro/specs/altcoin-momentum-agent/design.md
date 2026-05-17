# Design — Altcoin Momentum Agent

> 与 `requirements.md` 配套，描述系统架构、数据流、关键算法、技术选型与失败模式处理。
>
> 文档版本：v0.1 / Draft

---

## 1. 架构总览

### 1.1 设计哲学

1. **事件驱动 + 总线解耦**：所有跨模块通信走 Redis Streams，无直接函数调用；任一模块挂掉不污染其他。
2. **Hot path 与 Slow path 分离**：规则引擎位于 hot path（毫秒级），LLM 位于 slow path（秒级），LLM 永不阻塞下单决策的最低底线。
3. **风控独立、不可绕过**：Risk Gate 是单独进程/模块，所有交易决策必须经过；策略 bug 不能击穿账户。
4. **适配器优先**：交易所、社交源、LLM provider 全部 Adapter 接口化，便于替换与降级。
5. **回测 = 实盘子集**：实盘走的代码路径，回测必须能 100% 复用，仅替换 IO 边界（数据源、撮合器）。

### 1.2 模块清单

| 编号 | 模块 | 进程 | 主要技术 |
|---|---|---|---|
| A | Market & On-Chain Screener | `screener` | ccxt.pro, asyncio, numpy, pandas |
| B | Social Sentiment Crawler | `social` | aiohttp, jieba, snscrape/Nitter |
| C | AI Inference Engine | `inference` | httpx, pydantic, DeepSeek SDK |
| D | Execution & Risk | `risk` + `executor` | ccxt, asyncio |
| E | RL & Backtest Lab | `lab`（离线） | pandas, polars, optuna, mlflow |
| ∞ | 总线 / 存储 / 观测 | infra | Redis Streams, TimescaleDB / Parquet, Prometheus, Grafana, Loki |

### 1.3 部署拓扑

```
                       ┌────────────────────────────────────────┐
                       │              单机 (Docker compose)      │
                       │                                        │
   Binance/OKX/Gate ──▶│ screener ──┐                           │
                       │            │                           │
   Binance Square ─────│ social ────┤                           │
   KOL Twitter/X ──────│            ▼                           │
                       │       Redis Streams (event bus)        │
                       │            │                           │
                       │            ▼                           │
                       │   inference  ──── DeepSeek API (HTTPS) │
                       │            │                           │
                       │            ▼                           │
                       │     risk_gate ──▶ executor ──▶ ccxt ──▶ Exchanges
                       │            │                           │
                       │            ▼                           │
                       │   TimescaleDB / Parquet (long-term)    │
                       │   Prometheus + Grafana + Loki          │
                       └────────────────────────────────────────┘
```

---

## 2. 数据流向（Data Flow）

### 2.1 实时主路径（Hot + Slow）

```
┌─────────────────┐    ws/rest    ┌──────────────┐
│ Binance/OKX/    │ ─────────────▶│  screener    │
│ Gate.io         │               │ (ccxt.pro)   │
└─────────────────┘               └──────┬───────┘
                                         │ 1) raw stream → market.raw.*
                                         │ 2) feature extract → market.feature.*
                                         │ 3) rule signals → signal.market.*
                                         ▼
┌─────────────────┐    crawl      ┌──────────────┐
│ Binance Square  │ ─────────────▶│  social      │
│ Twitter/X KOLs  │               │ (crawler)    │
└─────────────────┘               └──────┬───────┘
                                         │  signal.social.surge.*
                                         ▼
                                  ┌──────────────┐
                                  │ Redis Streams│
                                  │ (event bus)  │
                                  └──────┬───────┘
                                         │ subscribe by topic
                ┌────────────────────────┼─────────────────────┐
                │                        │                     │
        ┌───────▼────────┐      ┌────────▼─────────┐   ┌───────▼────────┐
        │ rule_aggregator│      │ inference_engine │   │ persister      │
        │ (实时融合)      │      │ (DeepSeek 异步) │   │ (TS/Parquet)   │
        └───────┬────────┘      └────────┬─────────┘   └────────────────┘
                │  rule_score             │ llm_score
                └──────────────┬──────────┘
                               ▼
                       ┌───────────────┐
                       │ score_fuser   │  potential_score_final
                       └───────┬───────┘
                               │  signal.high_priority (>=85)
                               ▼
                       ┌───────────────┐   reject
                       │  Risk Gate    │ ──────▶ alert
                       └───────┬───────┘
                               │ approve
                               ▼
                       ┌───────────────┐         ┌──────────┐
                       │  Executor     │ ──ccxt─▶│ Exchange │
                       └───────┬───────┘         └─────┬────┘
                               │                      │ fill events
                               ▼                      │
                       ┌───────────────┐  ◀───────────┘
                       │ Position Mgr  │  → trailing stop loop → Risk Gate
                       └───────────────┘
```

**关键点**：
- `rule_aggregator` 与 `inference_engine` **并行**消费同一信号，双输出由 `score_fuser` 融合；LLM 即便超时也不挡住 `rule_score` 直接进入 fuser（仅权重退化为 100% 规则）。
- `Risk Gate` 是**唯一**能产出真实下单指令的节点。
- 所有事件都被 `persister` 落 TimescaleDB / Parquet，回测时直接重放。

### 2.2 回测离线路径

```
TimescaleDB / Parquet
        │
        ▼
┌─────────────────┐
│ event_replayer  │  按真实时间戳还原 → 注入 Redis Streams（or in-process queue）
└────────┬────────┘
         │
         ▼
   (整套实时模块以 backtest mode 运行：
    - executor 替换为 sim_executor（基于 orderbook 深度撮合）
    - inference 可缓存历史 LLM 响应避免重复花钱
    - risk_gate 同实盘逻辑)
         │
         ▼
┌─────────────────┐    ┌────────────────┐
│ trade_log       │ ──▶│ analytics      │── HTML / mlflow report
└─────────────────┘    └────────────────┘
         │
         ▼
┌────────────────────────────┐
│ postmortem (LLM 复盘标注)  │ → 更新 policy_weights.yaml
└────────────────────────────┘
```

### 2.3 事件总线（Topic 设计）

Redis Streams，命名规范 `domain.entity.action`：

| Topic | Producer | Consumer | Payload 关键字段 |
|---|---|---|---|
| `market.raw.trade.{exchange}.{symbol}` | screener | persister, feature | ts, price, qty, side |
| `market.feature.kline.{tf}.{symbol}` | screener | rule_agg, inference | ohlcv + 计算特征 |
| `market.feature.funding.{symbol}` | screener | rule_agg | rate, ts |
| `market.feature.oi.{symbol}` | screener | rule_agg | oi, delta_pct |
| `signal.market.volume_spike` | rule_agg | fuser, inference | zscore, side, ... |
| `signal.market.smc.{type}` | rule_agg | fuser, inference | type, level, strength |
| `signal.social.surge.{symbol}` | social | fuser, inference | growth_rate, samples |
| `signal.llm.verdict.{symbol}` | inference | fuser | verdict, kol_intent, score |
| `signal.high_priority.{symbol}` | fuser | risk_gate | full payload |
| `order.intent.{symbol}` | risk_gate | executor | side, size, stops |
| `order.fill.{symbol}` | executor | position_mgr, persister | exec_price, fee, ... |
| `risk.alert.*` | risk_gate, executor | notifier | reason |

每条消息携带 `trace_id` 贯穿全链路（OpenTelemetry W3C 格式）。

---

## 3. 模块详细设计

### 3.1 模块 A — Screener

**子组件**：
- `MarketFeed`（per exchange × per symbol）：基于 `ccxt.pro` `watchTrades / watchOHLCV / watchOrderBook / watchFundingRate / watchOpenInterest`，每路独立 task；
- `FeatureCalculator`：滚动窗口（`collections.deque`）计算 zscore、ATR、OI delta、funding 平滑值，全部以**纯 numpy** 实现，避免 pandas 单点开销；
- `SMCDetector`：基于 fractal swing 点的有限状态机，识别 BOS / CHoCH / OB / Liquidity Sweep；
- `RuleEngine`：把上述特征做硬阈值 → 发布 `signal.market.*`。

**Volume Spike 算法（伪码）**：
```
zscore = (vol_t - rolling_mean(vol, N)) / rolling_std(vol, N)
if zscore >= k and sign(close - open) consistent with delta_volume_side:
    emit volume_spike
```

**SMC 关键定义**：
- Swing Point：fractal(window=2)；
- BOS：当前 swing high 突破前一个 swing high；
- CHoCH：原趋势 swing 被反向打破；
- Order Block：BOS 之前最后一根反向实体 K 线；
- Liquidity Sweep：wick 穿越关键 level 后回收，wick / body ≥ 1.5。

**性能优化**：
- 每个 symbol 一个独立 asyncio Task，`uvloop` 加速；
- 滚动窗口预分配 numpy buffer（不在 hot path 做 dict / DataFrame）；
- 特征计算结果写入 Redis 以供其它模块消费，不在内存中跨模块共享。

### 3.2 模块 B — Social Crawler

**Adapter 接口**：
```
class SocialSourceAdapter(Protocol):
    async def stream(self) -> AsyncIterator[Post]: ...
```
Post 字段：`source, author, follower_count, ts, text, raw_url, ticker_mentions, lang`.

**子组件**：
- `BinanceSquareAdapter`：HTTP 轮询热门列表，最小间隔 ≥ 1s；
- `TwitterKOLAdapter`：基于第三方镜像 / Nitter；带 KOL 白名单；
- `TickerExtractor`：正则 + 词典（`$RAVE`, `RAVE/USDT`, 中文"瑞夫币"等）；中英文分词后再做关键词匹配；
- `SurgeDetector`：同 Volume Spike 思路，但作用在每分钟 mention count 序列。

**反爬与合规**：
- UA 池 + 代理可选；
- 全局 rate limiter（`aiolimiter`）；
- 单一适配器允许一键禁用（配置项 `social.adapters.binance_square.enabled=false`）；
- 默认不持久化原文，仅持久化 hash + 元数据；用户开启 `archive_text=true` 才落地原文。

### 3.3 模块 C — Inference Engine

**架构**：
- 入口：消费 `signal.market.*` + `signal.social.*` 事件，按 `symbol` 维护 90s 滑动窗口；
- 触发器：当一窗口内同时存在「市场异常」+「社交异常」时入队 `llm_request_queue`；
- 调度器：单 symbol 冷却 60s；全局并发 ≤ N（默认 4）；月预算控制（token 计数）。

**Prompt 模板（节选）**：
```
You are a senior crypto market analyst...
Context (JSON):
{
  "symbol": "...",
  "market_features": {...},
  "recent_klines": [...],
  "social_posts": [{"author":"@x","followers":120000,"text":"..."}],
  "kol_history_accuracy": {...}
}
Return STRICT JSON with fields: verdict, confidence, kol_intent, key_evidence, potential_score.
Do not add explanation outside JSON.
```

**JSON 强约束**：
- 用 `pydantic` Model 解析；
- 解析失败 → 重试 1 次（带 "你的上一条响应不是合法 JSON" 修正提示）；
- 二次失败 → 降级为 `verdict=noise, potential_score=0` 并打 metric。

**模型路由**：
- 默认 `deepseek-chat`；
- 当 confidence < 0.6 且预算允许，升级到 `deepseek-reasoner` 复核（V1.1）。

**成本控制**：
- 累计 token 写 Redis counter，按月归零；
- 超预算 → emit `risk.alert.budget_exceeded`，自动切回纯规则模式。

### 3.4 模块 D — Risk Gate + Executor

**Risk Gate 检查清单**（按顺序，任一 fail 即拒）：
1. 全局熔断标志（人工 / 自动）；
2. 当日累计亏损是否触顶；
3. 当前持仓数；
4. 单笔风险敞口；
5. 标的流动性（5 档深度 USDT 估值）；
6. 信号年龄（>10s 视为陈旧）；
7. 价格滑点预估（基于当前 orderbook 模拟成交，>0.3% 拒单）。

**Position Sizing**：
```
risk_amount = equity * max_risk_per_trade
stop_dist = abs(entry - initial_stop)
size_quote = risk_amount / stop_dist * entry
size_contracts = round_to_lot(size_quote / contract_value)
```

**Trailing Stop FSM**：
```
states: INIT → ARMED → BREAKEVEN → TRAILING → CLOSED
- INIT: 入场后立即下条件单 stop = OB 反向边界 - buffer
- ARMED: 浮盈 < 1R
- BREAKEVEN: 浮盈 >= 1R, 上移 stop 至 entry
- TRAILING: 浮盈 >= 2R, stop = max(prev_stop, price - n*ATR(14))
- CLOSED: 触发 stop 或人工平仓
不变量: stop 单调朝有利方向移动
```

**Executor**：
- 限价 IOC 优先；价格漂移 > 0.3% 则放弃这次入场（避免追高被埋）；
- 失败重试：网络错（指数退避 0.5/1/2s）、API 限流（按交易所 retry-after）；
- 所有下单 / 撤单写 `order.*` 事件，便于审计与回放。

### 3.5 模块 E — Backtest & RL Lab

**事件回放器**：
- 数据源：TimescaleDB（trade/funding/oi）+ Parquet（orderbook 快照）+ JSON（社交）；
- 时间机：按 `event_ts` 全局有序合并，按真实间隔 sleep 或加速倍率（默认 10x）；
- 注入端：可选 in-process queue（快）或 Redis Streams（保真）；
- 撮合器 `SimExecutor`：使用历史 orderbook 快照模拟吃单深度与滑点；不允许"完美成交"假设。

**LLM 复盘**：
- 输入：启动时刻 ±2h 的全特征 + 社交摘要；
- 输出：`{trigger_features:[...], false_positive_features:[...], narrative:"..."}`；
- 持久化路径：`data/postmortem/{symbol}_{date}.json`。

**权重更新（V1：贝叶斯加权）**：
- 每个特征维护 `(hit, total)` 计数；
- 后验命中率 `p = (hit + α) / (total + α + β)`（α=β=1，Laplace 平滑）；
- `weight_i ∝ p_i / (1 - p_i)` 归一化；
- 写回 `policy_weights.yaml`，附 `version, updated_at, sample_window`；
- 必须保留最近 10 个版本，可一键回滚。

**报告**：
- 每次回测产出 `reports/{run_id}/index.html`，包含权益曲线、每特征 PnL attribution、错单 / 滑点分布；
- 同时写 mlflow 便于横向比较。

---

## 4. 关键技术选型

| 维度 | 选择 | 理由 |
|---|---|---|
| 异步运行时 | `asyncio` + `uvloop` | ccxt.pro 原生协程；I/O 密集 |
| 事件总线 | Redis Streams | 单机够用、有 consumer group、自带持久化、轻量；扩展时可换 NATS/Kafka |
| 时序存储 | TimescaleDB（连续聚合） | SQL 友好、兼容 Postgres 工具链 |
| 大对象存储 | Parquet on local disk / S3 | orderbook snapshot、原始社交文本压缩比高 |
| 配置 | YAML + pydantic-settings | 热更新友好、类型安全 |
| LLM 客户端 | httpx async + 自封装 retry/budget | 不依赖某个厂商 SDK；便于切换 |
| 观测 | Prometheus + Grafana + Loki | 标配；OpenTelemetry trace 透传 |
| 部署 | Docker compose | 单机最简；K8s 留给 V2 |
| 任务编排（离线） | `prefect` 或 makefile | 回测、复盘、权重更新流水线 |

**为什么不选**：
- Kafka：单机过重；
- ZeroMQ：缺乏持久化；
- Celery：Broker 用 Redis 还要再加 worker，与 Streams 直接消费冗余；
- Pandas as hot-path：DataFrame 创建开销大，hot path 用 numpy。

---

## 5. 配置示例（节选）

```yaml
# config/app.yaml
exchanges:
  - name: binance
    market: usdtm-futures
    enable: true
  - name: okx
    market: swap
  - name: gate
    market: usdt-futures

symbols:
  whitelist:
    - "*USDT"     # 通配
  blacklist:
    - "BTCUSDT"   # 大盘币不在猎单范围
  min_24h_volume_usdt: 5_000_000
  max_listed_days: 60   # 优先新币

screener:
  volume_spike:
    window: 60
    k_sigma: 4
  funding:
    extreme_low: -0.001
    extreme_high: 0.0015
  oi:
    silent_pct: 0.15
    silent_max_price_move: 0.01

inference:
  provider: deepseek
  model_default: deepseek-chat
  monthly_token_budget_usd: 200
  cooldown_per_symbol_sec: 60
  max_concurrent: 4

risk:
  max_risk_per_trade: 0.01
  daily_drawdown_limit: 0.04
  max_concurrent_positions: 3
  min_liquidity_usdt: 200_000
  slippage_max: 0.003

execution:
  order_type: limit_ioc
  retry: {max: 3, base_ms: 500}

policy_weights_file: config/policy_weights.yaml
```

---

## 6. 失败模式与降级

| 故障 | 行为 |
|---|---|
| 交易所 WS 断 | 自动重连 + REST 补齐；3 次失败标记该交易所 unhealthy，停止其下单 |
| DeepSeek 超时 / 限流 | LLM 分置 0，融合分纯规则；写 `risk.alert.llm_degraded` |
| 月预算耗尽 | 自动切纯规则模式 |
| 社交源失败 | 单源熔断，其它源继续；融合分降权 |
| Redis 不可用 | 全停（无总线无法工作）；持久化层有 WAL，恢复后自动续上 |
| 下单网络错 | 指数退避；3 次后告警 + 取消该信号 |
| 持仓与本地状态不一致 | 启动时全量 reconcile：交易所持仓为 source of truth |
| Risk Gate 自身 bug | 默认 fail-closed（拒单）而非 fail-open |

---

## 7. 安全

- API key 不入仓库；`.env` + OS keyring；
- 下单 key 与读 key 物理分离；下单 key 限制 IP 白名单；
- 配置变更走 git，禁止运行时直接改文件不留痕；
- 日志默认脱敏（mask key、订单 ID 部分截断）；
- 运行用户最小权限（容器内 non-root）。

---

## 8. 可观测指标（Prometheus）

- `screener_events_total{type=...}`
- `signal_emit_total{kind=...}`
- `llm_call_seconds_bucket`、`llm_call_errors_total{reason}`、`llm_tokens_total`
- `risk_reject_total{reason}`
- `order_submit_total`、`order_fill_seconds_bucket`、`order_slippage_bps`
- `position_pnl_usdt`、`equity_usdt`、`daily_drawdown_pct`
- `bus_lag_seconds{stream}`

告警示例：`bus_lag_seconds > 5`、`order_slippage_bps p99 > 50`、`llm_call_errors_total[5m] > 10`、`daily_drawdown_pct > 0.03`。

---

## 9. 目录结构

```
.
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── config/
│   ├── app.yaml
│   └── policy_weights.yaml
├── .kiro/
│   └── specs/altcoin-momentum-agent/{requirements,design,tasks}.md
├── src/altcoin_agent/
│   ├── bus/                 # Redis Streams 客户端封装
│   ├── screener/
│   │   ├── feeds/           # 各交易所 ccxt.pro 适配
│   │   ├── features/        # zscore / atr / smc
│   │   └── rules/
│   ├── social/
│   │   ├── adapters/{binance_square,twitter,nitter}.py
│   │   ├── extractors/
│   │   └── surge.py
│   ├── inference/
│   │   ├── prompt.py
│   │   ├── deepseek_client.py
│   │   ├── budget.py
│   │   └── fuser.py
│   ├── risk/
│   │   ├── gate.py
│   │   ├── sizing.py
│   │   └── trailing.py
│   ├── execution/
│   │   ├── ccxt_executor.py
│   │   └── sim_executor.py
│   ├── lab/
│   │   ├── replayer.py
│   │   ├── postmortem.py
│   │   ├── weights_update.py
│   │   └── reports/
│   ├── persist/             # timescale + parquet 写入
│   ├── observability/
│   └── cli.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 10. 后续演进（V1 → V2 路线）

- DEX 链上巨鲸地址监控（Etherscan / Arkham 风格）；
- 多账户、SaaS 化；
- 用 contextual bandit / PPO 替代贝叶斯权重；
- 自托管或微调小模型替代部分 DeepSeek 调用以降本；
- K8s + 跨 region 热备。
