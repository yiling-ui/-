# 社交驱动妖币策略改造计划单

> **本文件是项目唯一的"接力棒"**,任何下一轮对话只要打开这个文件就能立刻接手干活,
> 不需要操作员重述意图,也不会再次跑偏。
>
> **不要修改这个文件的章节标题层级**(后面有锚链接);可以追加新章节、可以勾选 checkbox、
> 可以更新进度,但不要重排或删除已有内容。

---

## 0. 操作员约定 (Single Source of Truth)

### 0.1 真实意图(以这一段为准,不接受任何代码注释/文档反向解释)

操作员要做的是 **"以币安广场等以及各路 KOL 各种讨论发帖的交易平台为主去寻找可能会发生
大幅度拉盘的妖币、其他为辅助,然后结合讨论去分析开仓做多以及平仓和做空,最后通过
强化学习,做到 10 倍以上的收益"**。

关键解释:
- **社交是发现层 (discovery)**,不是确认层 (confirmation)。
- "其他为辅助"指技术指标 (vol/OI/funding/sweep/wash) 是 **第二位** 的过滤,
  不是第一位的发现。
- **10x 收益目标 ≠ 每笔加杠杆赌 10x**;是"小仓位多次试错 + 单笔最大输 1R + 跑赢的
  留到 10R+"的赔率分布游戏。
- **强化学习** 在这里是宽义,当前阶段先做 **向量记忆 + agentic post-mortem**,
  正经 RL 训练环路放到 Phase 2 之后再考虑(样本量不够,现在做必过拟合)。

### 0.2 偏离的根源(为了让接手者一眼看懂为什么这个文件存在)

PR #1~#39 累积偏离了真实意图,核心错误注释固化在 `src/altcoin_agent/pipeline.py`
的 module docstring:

> "The market screener decides a coin is '妖币 candidate'. ONLY THEN do we call
> `focus_on_symbol(symbol)` to spend network on social/auxiliary data."

这把 "社交先选币" 反转成了 "技术先选币、社交二次确认"。本计划要把它改回来。

### 0.3 操作员已经拍板的预算/资源

| 项目 | 决策 |
|---|---|
| 币安广场 cookie | **单账号**(操作员手工维护) |
| Gate/OKX 公告 | RSS/API key + HTML scraper(双路,以 RSS 优先) |
| LLM 预算 | **DeepSeek $30/月** 上限 |
| Embedding | DeepSeek embedding API,**不引入本地 sentence-transformers**(操作员明确拒绝本地模型) |
| 风控基线 | 保留 PR #1~#39 的 14 道 RiskGate 闸,**不动** |
| 上线节奏 | 仍走 30 天 dry-run + paper-trade,实盘前必须 `LIVE_CONFIRM=I_UNDERSTAND` |

### 0.4 三个 PR 的执行顺序(不可并行,后者依赖前者)

```
PR-A: 动态 symbol 发现层      (1 周, 600~800 行)    ← 先做
PR-B: 三路加权 fuser          (1.5 周, 800~1000 行) ← 依赖 PR-A 的 symbol 池
PR-C: 向量记忆 + agent 学习   (2~3 周, 1500~2000 行) ← 依赖 PR-A/B 跑出真实交易
```

---

## 1. 当前代码地基(只读,不要改的部分)

接手前 5 分钟读完这一节就够了。**这些已经做好,不要重写**:

### 1.1 主链(保留)

```
ws → screener → fuser → RiskGate (14 闸) → CCXTExecutor (cid 幂等) → Position
                                                                          │
                       ┌──────────────────┬─────────────────┬─────────────┤
                       ▼                  ▼                 ▼             ▼
                TrailingStopFSM     RollingController  PositionWatcher  PostMortem
```

### 1.2 不要动的关键不变量

- **RiskGate 14 闸的次序**(`src/altcoin_agent/risk/gate.py`),只能 **加** 不能改顺序
- **CCXTExecutor.cid 幂等**(`alt-{e|s|l|c}-{uuid12hex}`)
- **AccountPersistor schema_version=2**(改要 bump 到 3 + 兼容旧文件)
- **TrailingStopFSM monotonic 不变量**(stop 只能收紧)
- **dynamic_rules.json 是 SoT,.md 自动重生,不要手改 .md**
- **dashboard token 中间件 + dashboard_bind=127.0.0.1 默认**
- **LIVE_CONFIRM=I_UNDERSTAND 启动门**

### 1.3 已知技术债(本计划不修,留在 backlog)

- `WashTradingDetector` 在 ccxt.pro 实盘路径上 trade_count <= 0,SR-3 可能 no-op
- `compute_leverage` 的 `vol_adj` / `liq_adj` 对妖币是反向激励(高波动/薄深度反而低杠杆)
- 没有真实回测 PnL/Sharpe/MaxDD 数据
- learning_engine 的 hit_rate 没扣手续费 + 资金费

这些是 V2.0 后再处理的事情,**本计划只解决"社交驱动发现"和"学习模式升级"**。

---

## 2. PR-A: 动态 Symbol 发现层

### 2.1 目标

把 `app.yaml.symbols` 的静态列表改成 **"yaml pinned ∪ 动态发现"**,让真正在
广场/Gate/OKX 上被讨论或异动的币种自动进入 screener 监视。

### 2.2 模块结构

```
src/altcoin_agent/discovery/                  [新目录]
    __init__.py
    discoverer.py            DiscoveredSymbol + SymbolDiscoverer 主类
    sources/
        __init__.py
        binance_ticker.py    /fapi/v1/ticker/24hr
        binance_square.py    复用 social/binance_square.py + 新增 fetch_trending()
        gate_contracts.py    /api/v4/futures/usdt/contracts + 公告 RSS
        okx_tickers.py       /api/v5/market/tickers (SWAP) + 公告
        dexscreener.py       trending API (筛选已有 CEX perp 的)
    ner/
        __init__.py
        ticker_extractor.py  从帖子文本里抽 ticker (regex + 白名单)
    pool.py                  SymbolPool (LRU + pinned)
```

### 2.3 数据结构

```python
# discovery/discoverer.py
@dataclass
class DiscoveredSymbol:
    symbol: str                      # "PEPE/USDT:USDT" 标准化
    score: float                     # 0~100
    sources: dict[str, float]        # {"binance_vol": 23.4, "square_mention": 87.1, ...}
    last_seen_ts_ms: int
    discovered_at_ts_ms: int
    reason: str                      # 人类可读的解释,落 dashboard 用
    metadata: dict[str, Any]         # 原始数据保留(debug 用)


@dataclass
class DiscovererConfig:
    # source weights (sum = 1.0, 校验)
    binance_vol_weight: float = 0.25
    gate_funding_weight: float = 0.15
    okx_price_weight: float = 0.10
    square_mention_weight: float = 0.30   # ← 操作员意图: 广场为主
    dex_trending_weight: float = 0.20

    # 入池阈值
    min_score_to_enter: float = 60.0
    min_score_to_keep: float = 40.0       # 已入池的, 跌破这个才踢

    # ws 容量
    max_discovered_pool: int = 80         # LRU 上限
    yaml_pinned_always: list[str] = ["BTC/USDT:USDT"]  # regime filter 必需

    # 频率
    discover_interval_sec: int = 300      # 5min
    eviction_grace_sec: int = 1800        # 入池后至少保留 30min, 避免抖动

    # source 各自的限流(防 451/429)
    binance_ticker_min_interval_sec: int = 60
    square_trending_min_interval_sec: int = 180   # 单账号 cookie, 不要太频繁
    gate_okx_announcements_min_interval_sec: int = 300
```

### 2.4 SymbolPool (LRU 管理)

```python
# discovery/pool.py
class SymbolPool:
    def __init__(self, cfg: DiscovererConfig)
    def upsert(self, ds: DiscoveredSymbol) -> set[str]:  # 返回新增的 symbols
    def evict_stale(self, now_ms: int) -> set[str]:      # 返回踢出的 symbols
    def current_symbols(self) -> set[str]                 # pinned ∪ discovered
    def snapshot(self) -> list[DiscoveredSymbol]          # 给 dashboard
```

**关键不变量**:
- `pinned` 永远不会被 evict
- `discovered` 用 `(score, last_seen_ts)` 双键 LRU
- 评分跌破 `min_score_to_keep` AND 进池超过 `eviction_grace_sec` → 才允许踢
- 同 symbol 重复 discover 时,**取最新 score 而不是累加**(防爆涨币种永久占池)

### 2.5 Screener 改造(最小侵入)

```python
# screener.py 新增
async def add_symbol(self, symbol: str) -> bool:
    """订阅新 symbol 的 ws stream. 已订阅则 no-op. 失败返回 False."""

async def remove_symbol(self, symbol: str) -> bool:
    """取消订阅. 同时清理该 symbol 在 detector 里的状态(rolling buffer)."""

@property
def active_symbols(self) -> frozenset[str]
```

**Binance ws 上限处理**:
- 单 ws 连接 ~200 stream(`<symbol>@kline_1m + @aggTrade + @markPrice = 3 stream`)
- 80 个 symbol × 3 stream = 240 个 → 必须分两个 ws 连接管理
- 在 `screener.py` 里加 `WSConnectionPool`,每连接最多 150 stream,超了开新连接

### 2.6 Worker 接入

```python
# main.py 新增
[W16] symbol_discovery_worker
    每 cfg.discover_interval_sec 跑一次:
    1. discoverer.discover() → list[DiscoveredSymbol]
    2. pool.upsert(...) → 新增集合
    3. pool.evict_stale(now) → 踢出集合
    4. for sym in 新增: await screener.add_symbol(sym)
       for sym in 踢出: await screener.remove_symbol(sym)
    5. dashboard.update_discovery_pool(pool.snapshot())
    6. 任何异常吞掉, 写 metric `discovery_failures_total`
```

### 2.7 AppConfig 新增字段(全部默认 OFF)

```yaml
# app.yaml
discovery_enabled: false              # PR-A 合并后默认 false, 操作员 dry-run 后再开
discovery_interval_sec: 300
discovery_min_score_to_enter: 60.0
discovery_min_score_to_keep: 40.0
discovery_max_pool_size: 80
discovery_eviction_grace_sec: 1800
discovery_yaml_pinned: ["BTC/USDT:USDT"]
discovery_source_weights:
  binance_vol: 0.25
  gate_funding: 0.15
  okx_price: 0.10
  square_mention: 0.30
  dex_trending: 0.20
```

### 2.8 测试要求

`tests/test_discovery_mock.py` 必须覆盖:

- [ ] `test_pool_pinned_never_evicted` — yaml symbols 永远在
- [ ] `test_pool_lru_evicts_lowest_score_when_full`
- [ ] `test_pool_eviction_grace_window` — 进池 < 30min 不能踢
- [ ] `test_pool_repeated_discover_takes_latest_score` (不累加)
- [ ] `test_screener_add_remove_symbol_idempotent`
- [ ] `test_screener_add_when_ws_full_opens_new_connection`
- [ ] `test_discoverer_weights_must_sum_to_one` — config 校验
- [ ] `test_discoverer_source_failure_partial_score` — 一个 source 挂了不影响其他
- [ ] `test_discoverer_all_sources_fail_no_pool_change` — 全挂时 pool 不变(不要清空)
- [ ] `test_worker_handles_screener_add_failure_gracefully`
- [ ] `test_binance_square_trending_451_falls_back_to_yaml`
- [ ] `test_ticker_extractor_filters_unknown_tickers` — "去问 SOL" 不会误识别 SOLANA

### 2.9 已知 bug / 风险 / 应对

| # | 风险 | 应对 |
|---|---|---|
| A1 | Binance Square 单账号 cookie 失效 → 451 | scraper 已有 typed exception (`ScraperGeoBlocked`/`ScraperError`),catch 后 source score=0,**不阻塞其他 source**;dashboard 加 `social_source_status` 红绿灯 |
| A2 | Binance trending 提及频次刷量(机器人灌水) | 用 `unique_authors_count / total_mention_count` 比值 < 0.3 时降权;复用 PR #21 的 ghost_volume 思路 |
| A3 | DexScreener 返回的 token 在 CEX 没 perp | 必须查 ccxt `markets` cache,不在的直接丢弃 |
| A4 | Ticker NER 把 "USDT" "BTC" 这种通用词误识别 | 用 `IGNORED_TICKERS` 黑名单 + 出现频次必须 ≥ 2 才进候选 |
| A5 | 5min 间隔太长,错过爆发币 | **不要降到 1min**(429 风险);应在 `BinanceSquareScraper` 内部加 `on_breakout_post` 推送钩子(如果检测到单帖 1h 内点赞数 >500),可触发 ad-hoc discover |
| A6 | 新 symbol 加入后 detector 冷启动期(60 bar)产生不了信号 | 接受这个延迟。dashboard 显示 `cold_bars_remaining`,操作员可以视觉判断。 |
| A7 | LRU 踢出时正在持仓 | **必须在 evict 前检查 `account.open_positions`,有持仓的 symbol 不能踢**;只能等仓位平了再踢 |
| A8 | RegimeFilter 依赖 BTC ws,如果 BTC 被误踢就废了 | `yaml_pinned` 默认包含 BTC,`pool.upsert` 拒绝 evict pinned |
| A9 | 一次 discover 引入 50+ 新 symbol → ws 风暴 | `add_symbol` 串行调用,每次间隔 200ms;单次 worker 最多新增 10 个,剩余下次再加 |
| A10 | discovery 太激进影响整体 metrics | `discovery_enabled=false` 默认关,操作员 paper-trade 7 天再开 |

### 2.10 PR-A 完成定义 (DoD)

- [ ] 全部新模块 ruff clean
- [ ] 测试矩阵全过(2.8 的 12 项)
- [ ] 全套 pytest 总数 ≥ 411 + 新增,无回归
- [ ] `discovery_enabled=false` 时 daemon 行为字节级与当前一致
- [ ] dashboard 新增 "Discovery Pool" 面板,显示当前 pool + 每个 symbol 的 score 来源
- [ ] `/metrics` 暴露 `discovery_pool_size` / `discovery_failures_total{source}` /
      `discovery_evictions_total{reason}` / `screener_active_symbols`
- [ ] PR 描述列出"和当前默认行为的差异",让 reviewer 5 分钟看完

---

## 3. PR-B: 三路加权 Fuser

### 3.1 目标

把 fuser 的契约从 "rule_score is base, LLM is multiplier" 改成
**"rule + social + LLM 三路加权平均"**,让广场和 Gate/OKX 的社交信号成为
**一等公民**,可以单独触发 high_priority(不需要技术信号)。

### 3.2 模块结构

```
src/altcoin_agent/social/                    [扩充]
    __init__.py
    binance_square.py        [已有, 扩充 fetch_trending]
    crawler.py               [已有, 改造 SocialSnapshot]
    historical_analyzer.py   [已有 PR #33, 不动]
    sources/                 [新]
        __init__.py
        gate_announcements.py   RSS + HTML scraper
        okx_announcements.py    API + HTML scraper
    social_scorer.py         [新] SocialScorer 主类
    ticker_classifier.py     [新] 帖子 → intent (pump/dump/exit_liquidity/neutral)

src/altcoin_agent/fuser.py                   [改造]
    + FuserConfig.signal_blend_weights
    + ScoreFuser._compute_three_way_score
    + ScoreFuser._maybe_social_solo_trigger
```

### 3.3 SocialScorer 数据结构

```python
@dataclass
class SocialScore:
    symbol: str
    score: float                    # 0~100, 综合社交分
    confidence: float               # 0~1, 来源覆盖度 + 样本量
    breakdown: dict[str, float]     # {"square_mention": 70, "gate_announce": 0, ...}
    direction_hint: str             # "long" | "short" | "neutral"
    direction_confidence: float     # 0~1
    kol_authors: list[str]          # 用于 PR #33 KOLHistoryStore 加权
    raw_posts: list[SquarePost]     # 给 LLM 用的素材
    fetched_at_ts_ms: int
    sources_status: dict[str, str]  # 每个 source 是否 ok / degraded


@dataclass
class SocialScorerConfig:
    # 各 source 在社交综合分里的权重 (sum = 1.0)
    square_weight: float = 0.55
    gate_weight: float = 0.25
    okx_weight: float = 0.20

    # 触发阈值
    mention_velocity_threshold: float = 3.0   # 1h 提及量 / 7d 平均 ≥ 3x
    min_unique_authors_for_signal: int = 5

    # 防灌水
    bot_density_threshold: float = 0.7        # bot 占比 ≥ 70% 直接降到 30 分

    # KOL 历史命中加权 (复用 PR #33)
    kol_history_weight_max: float = 0.20
```

### 3.4 SocialScorer 主流程

```
SocialScorer.score(symbol) → SocialScore
    │
    ├─ Binance Square                        weight 0.55
    │   ├─ posts = BinanceSquareScraper.fetch_for_symbol()
    │   ├─ mention_velocity = count_1h / mean_count_7d
    │   ├─ unique_authors_1h
    │   ├─ direction = TickerClassifier.classify(posts)
    │   ├─ KOLHistoryStore.adjust_confidence(posts)  ★ 复用 PR #33
    │   └─ square_subscore (0~100)
    │
    ├─ Gate                                  weight 0.25
    │   ├─ RSS: https://www.gate.com/announcements/rss/...
    │   ├─ fallback: HTML scraper (selectolax)
    │   ├─ keyword: "上线" "Spot" "Futures" "空投" "上币"
    │   └─ gate_subscore (0~100, 1h 衰减)
    │
    ├─ OKX                                   weight 0.20
    │   ├─ API: /api/v5/public/announcements?annType=announcements-new-listings
    │   ├─ fallback: HTML scraper
    │   └─ okx_subscore (0~100, 1h 衰减)
    │
    └─ 综合
        bot_density 检查 → 必要时降权
        weighted_avg → final score
        direction_hint = 多数 source 一致的方向, 否则 neutral
```

### 3.5 Fuser 改造

#### 3.5.1 三路融合公式

```python
# 当前 (PR-B 之前):
final_score = rule_score × llm_multiplier × kol_modifier

# PR-B 之后:
weights = cfg.signal_blend_weights  # rule:0.40, social:0.40, llm:0.20

# rule 永远有(本地计算)
# social/llm 可能不存在 → 该路 weight 重新归一化到剩余路
active_paths = [(score, w) for (score, w) in [
    (rule_score, weights.rule),
    (social_score, weights.social) if social_score else None,
    (llm_score, weights.llm) if llm_verdict else None,
] if exists]

normalized_w = renormalize(active_paths)
final_score = sum(score × w for score, w in active_paths)
```

#### 3.5.2 Social-solo 触发

```python
# PR-B 新增: 即使 rule_score 很低, social_score 高 + 强方向 → 也触发 high_priority
def _maybe_social_solo_trigger(self, social: SocialScore, rule_score: float) -> bool:
    if social.score < 80:                    return False
    if social.direction_confidence < 0.7:    return False
    if social.confidence < 0.5:              return False
    if social.unique_authors_count < 5:      return False
    # 即使 rule_score 是 0 也允许触发
    return True

# 但 RiskGate 14 道闸照常跑 — anti-chase / vol-kill 可能仍然否决
```

#### 3.5.3 KOL exit_liquidity 升级

PR #33 已经把 KOL exit_liquidity 实现成"hard veto LONG @ conf≥0.7 / soft cap LONG @ conf<0.7"。
**不动这个逻辑**,只是现在三路融合下,同一个 social signal 也喂给 SHORT 触发器:

```python
# 新增: SHORT 主动开仓由 KOL exit_liquidity 触发
if social.direction_hint == "short" and \
   "exit_liquidity" in social.intents and \
   social.direction_confidence >= 0.7 and \
   kol_history_avg_hit_rate >= 0.55:
    # 这是 PR #33 之前完全做不到的:KOL 喊出货 → 主动做空
    return FusedSignal(direction=SHORT, ...)
```

### 3.6 AppConfig 新增

```yaml
# app.yaml
social_scorer_enabled: false   # PR-B 默认关
signal_blend_weights:
  rule: 0.60                    # PR-B 起步 (保守)
  social: 0.20                  # 起步小,paper-trade 2 周稳了升到 0.40
  llm: 0.20

social_solo_trigger_enabled: false   # 极保守,默认关
social_solo_min_score: 80
social_solo_min_direction_confidence: 0.7
social_solo_min_unique_authors: 5

# Gate
gate_announcements_rss_url: "https://www.gate.com/announcements/rss/..."
gate_announcements_fallback_html: "https://www.gate.com/announcements/article/..."

# OKX
okx_announcements_api_url: "https://www.okx.com/api/v5/public/announcements"
okx_announcements_fallback_html: "https://www.okx.com/help/section/announcements-..."

# 反爬
scraper_user_agent_pool: [...]   # 5~10 个真实浏览器 UA
scraper_proxy_pool: []            # 空 = 不用代理
scraper_request_timeout_sec: 8
scraper_circuit_breaker_failures: 5    # 连续 5 次失败开启熔断
scraper_circuit_breaker_recovery_sec: 600
```

### 3.7 反爬抗性设计 (操作员明确要求)

```
src/altcoin_agent/social/scraper_base.py     [新]
    class ResilientScraper:
        - request() 走自己的 httpx.AsyncClient
        - rotating user-agent
        - exponential backoff (0.5s, 1s, 2s, 4s, 8s)
        - typed exceptions (Geoblocked / RateLimited / AuthRequired / ParseError)
        - circuit breaker per host (5 fail → 10min cooldown)
        - persistent cookie jar (.kiro/state/scrapers/{host}.cookies)
        - 失败时 emit metric `scraper_failures_total{host, reason}`
```

**关键设计**:
- **不要并发抓**:同一个 host 串行,间隔 ≥ 2s。多 host 之间可以并发。
- **HTML 解析用 selectolax**(比 lxml 快 5x,纯 Python 后端可降级)。
- **RSS 优先**:Gate/OKX 都有 RSS,如果 RSS 拿到了就跳过 HTML。

### 3.8 测试要求

`tests/test_social_scorer_mock.py`:

- [ ] `test_three_source_weighted_avg`
- [ ] `test_one_source_down_renormalizes_others`
- [ ] `test_all_sources_down_returns_none_no_crash`
- [ ] `test_bot_density_above_threshold_caps_score`
- [ ] `test_kol_history_lifts_high_reputation_author`
- [ ] `test_kol_history_drops_known_dumper`
- [ ] `test_direction_disagreement_returns_neutral`
- [ ] `test_mention_velocity_below_threshold_no_signal`

`tests/test_fuser_three_way_mock.py`:

- [ ] `test_three_way_blend_basic`
- [ ] `test_only_rule_path_when_social_disabled`
- [ ] `test_renormalize_weights_when_llm_missing`
- [ ] `test_social_solo_triggers_high_priority`
- [ ] `test_social_solo_blocked_when_rule_strongly_opposes` — rule 70 short, social 90 long → 矛盾,
      veto
- [ ] `test_kol_exit_liquidity_triggers_short` (新场景)
- [ ] `test_yaml_round_trip_signal_blend_weights`
- [ ] `test_blend_weights_sum_validation`

`tests/test_scraper_resilience_mock.py`:

- [ ] `test_circuit_breaker_opens_after_5_failures`
- [ ] `test_circuit_breaker_recovers_after_cooldown`
- [ ] `test_user_agent_rotates_per_request`
- [ ] `test_cookie_jar_persists_across_restarts`
- [ ] `test_rss_preferred_over_html`
- [ ] `test_html_fallback_when_rss_404`

### 3.9 已知 bug / 风险 / 应对

| # | 风险 | 应对 |
|---|---|---|
| B1 | 广场单账号 cookie 一旦被风控就全停 | 已有 `degraded:auth_required` typed exception;**新增 dashboard 红色告警 + Telegram 通知**;social_score 这一路降到 0,fuser 自动 renormalize 到 rule + llm 两路 |
| B2 | Gate/OKX RSS 改格式或下线 | RSS 解析失败 → 自动 fallback HTML;HTML 也挂 → circuit breaker 开,该 source weight=0 |
| B3 | Bot 灌水 social_score 虚高 | `bot_density > 70%` 直接 cap 到 30 分;`unique_authors / total_posts < 0.3` 也降权 |
| B4 | 新交易所公告被解析成"上币"但其实是空投/合约调整 | TickerClassifier 必须区分意图;先用关键词,后期接 DeepSeek tag。 **保守做法**:只把"现货上线"和"合约上线"识别成 pump 信号,其他归 neutral |
| B5 | KOL exit_liquidity 误判把好币砸了 | 复用 PR #33 的 hit_rate 校验,**`min_samples=10`**(已是默认);新作者一律 confidence ≤ 0.5,只能 soft cap 不能 hard veto |
| B6 | social_solo_trigger 接连开错方向 | 同 symbol 连续亏 2 次后 social_solo 对该 symbol 自动失效 24h |
| B7 | TickerClassifier 把 ETH/SOL 这种主流币标记成妖币 candidate | yaml 维护一份 `excluded_majors`(BTC/ETH/SOL/BNB/XRP/...);social_solo_trigger 对这些直接关 |
| B8 | 多 source 投票方向不一致(广场看多,Gate 公告中性) | 当 unanimous_direction_confidence < 0.6 时 direction_hint = neutral,不进 fuser 方向决策 |
| B9 | Gate/OKX HTML 改版后正则全废 | `selectolax` + selector 写在 yaml 里,不要硬编码;启动时跑一次 selector 健康检查,失败 logger.critical |
| B10 | 三路融合后 high_priority 信号量翻 3 倍 | RiskGate 14 闸照样过;但要监控 `orders_placed/min` 和 `orders_rejected/min` 比值变化;dashboard 加 alert 阈值 |
| B11 | Telegram 通知爆表 | 已有 PR #21 的 TokenBucket 28 msg/s 限流,合理;但 social signal 可能让 signal 流量翻倍 → 调 dashboard 把 SIGNAL 通知改成 batched 5min summary |
| B12 | DeepSeek 月预算超 $30 | TokenBudgetManager 已有 FREEZE 档,**新加 social-only 的 LLM 预算桶**(最多预算的 30%),互不干扰 |

### 3.10 PR-B 完成定义 (DoD)

- [ ] 全部测试矩阵过(3.8 的 22 项)
- [ ] `social_scorer_enabled=false` AND `signal_blend_weights={rule:1.0, social:0, llm:0}` 时,
      行为字节级与 PR-A 之后一致
- [ ] dashboard 新增 "Social Signals" 面板:每 symbol 的 square/gate/okx 三路分 + 方向 + KOL 列表
- [ ] `/metrics` 新增 `social_score{symbol,source}` / `social_signals_total{direction}` /
      `social_solo_triggers_total` / `scraper_circuit_breaker_open{host}`
- [ ] Gate/OKX 公告 RSS 健康检查在启动时跑,失败 logger.warning 但不阻塞启动

---

## 4. PR-C: 向量记忆 + Agentic 学习

### 4.1 目标

把 learning_engine 从 "8 个 closed-set 特征 + bucket + Laplace 平滑" 升级成
**"向量记忆 (kNN 检索) + agentic post-mortem (DeepSeek 工具调用)"**。

### 4.2 模块结构

```
src/altcoin_agent/memory/                    [新]
    __init__.py
    vector_store.py          sqlite-vss 后端 + insert/search
    embedder.py              DeepSeek embedding API + 本地 LRU cache
    schema.py                TradeMemory dataclass
    retriever.py             kNN + reranking + time decay

src/altcoin_agent/agents/                    [新]
    __init__.py
    post_mortem_agent.py     agentic loop, 4 个 tool
    tools/
        __init__.py
        similar_trades.py    query 向量记忆
        kol_track.py         查 KOLHistoryStore
        funding_history.py   查 funding 时序
        market_microstructure.py   查 OI / volume 时序
    prompts.py               system prompt + few-shot examples

src/altcoin_agent/learning_engine.py         [改造 ~60%]
    + RuleStore.update_with_memory_tags
    + run_post_mortem 走新 agentic 路径
    + 旧 closed-set 路径保留为 fallback (LLM 挂时用)
```

### 4.3 TradeMemory schema

```python
@dataclass
class TradeMemory:
    trade_id: str                   # uuid
    symbol: str
    side: str                       # "long" | "short"
    opened_at_ts_ms: int
    closed_at_ts_ms: int
    realized_pnl_usdt: float
    realized_r: float               # leverage-aware

    # 输入特征 (用于 embedding)
    embedding_input_text: str       # "symbol=PEPE, social_velocity=4.2x, " + LLM 写的语义摘要
    embedding: list[float]          # 1024-dim, DeepSeek embedding-v2

    # 元数据 (kNN 后的 reranking 用)
    rule_score_at_entry: float
    social_score_at_entry: float
    llm_score_at_entry: float
    fused_score_at_entry: float
    quadrant_at_entry: str
    phase_at_entry: str
    kol_authors: list[str]
    close_reason: str               # stop_loss / trail / emergency / manual
    memory_tags: list[str]          # agent post-mortem 输出, 如 "meme_pump_bull_div"
```

### 4.4 向量存储后端

```
.kiro/state/memory/trades.db (sqlite + vss extension)

CREATE VIRTUAL TABLE trades_vss USING vss0(embedding(1024));

CREATE TABLE trades_meta (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    pnl_r REAL,
    metadata_json TEXT,
    embedding_input_text TEXT,
    closed_at_ts_ms INTEGER
);

-- 索引
CREATE INDEX idx_trades_symbol_ts ON trades_meta(symbol, closed_at_ts_ms);
```

**降级策略**:
- sqlite-vss 装不上 → 用 `numpy + brute-force cosine`(< 10k 条仍可接受 ~50ms)
- 不强依赖 vss,这一点很重要,**不能让一个 C 扩展挡住整个项目**

### 4.5 Embedder

```python
class DeepSeekEmbedder:
    api: DeepSeek embedding endpoint (OpenAI-compatible /v1/embeddings)
    model: "deepseek-embedding-v2" or 实际模型名(查文档)
    cache: LRU 1024 entries on disk (.kiro/state/memory/embedding_cache.db)
    rate_limit: 同 LLMEngine 共享 TokenBudgetManager

    async def embed(self, text: str) -> list[float]:
        if cache.hit: return cached
        result = await api.embed(text)
        cache.set(text, result)
        return result
```

### 4.6 Retriever

```python
class MemoryRetriever:
    async def search_similar(
        self,
        embedding_query: list[float],
        symbol_filter: str | None = None,    # 同 symbol 优先,但不强制
        k: int = 20,
        time_decay_half_life_days: int = 30,
    ) -> list[ScoredTradeMemory]:
        """
        1. vss kNN top-50
        2. 同 symbol 加权 +20% (但不排除其他 symbol, 因为要泛化)
        3. 时间衰减: weight *= exp(-(now - closed_at) / half_life)
        4. 取 top-k
        """

    def memory_score(self, similar: list[ScoredTradeMemory]) -> float:
        """
        weighted_avg pnl_R of similar trades
        > +0.5R: 学到的强正向
        < -0.5R: 学到的强负向
        否则:中性
        """
```

### 4.7 Agentic Post-Mortem

```python
# agents/post_mortem_agent.py

SYSTEM_PROMPT = """
You are a trading post-mortem agent. After every closed trade, you must:
1. Reason about why it won/lost using available tools.
2. Output structured findings.

Available tools:
- query_similar_past_trades(context: str, k=10)
- get_kol_track_record(author: str, lookback_days=30)
- get_funding_history(symbol: str, ts_range: tuple)
- get_market_microstructure(symbol: str, ts_range: tuple)

Output JSON schema (strict):
{
  "what_worked": str,
  "what_failed": str,
  "feature_attribution": [
    {"feature": str, "weight": float, "confidence": float}
  ],
  "memory_tags": [str],         # e.g., "meme_pump_bull_div", "thin_book_squeeze"
  "actionable_rule": {
    "condition": dict,           # 必须使用现有的 8 个 closed-set features 之一组合
    "action": str,               # "boost_long" | "boost_short" | "veto_long" | "veto_short"
    "confidence": float,
    "min_samples_for_activation": int    # 至少要看到这个组合 N 次才用
  } | null
}
"""


class PostMortemAgent:
    max_tool_calls: int = 6              # 防 agent 死循环烧 token
    max_total_tokens: int = 15000        # 单次 post-mortem 硬上限
    fallback_to_legacy: bool = True      # LLM 挂了走旧 closed-set heuristic

    async def analyze(self, trade: ClosedTrade) -> PostMortemReport:
        # 1. agent loop
        # 2. 工具调用 ≤ max_tool_calls
        # 3. 输出 JSON 解析,失败重试 1 次
        # 4. 还失败 → fallback to legacy
```

### 4.8 与 fuser 的接入

```python
# fuser.py 新增第四路 (memory)
weights = {
    "rule": 0.35,
    "social": 0.35,
    "llm": 0.20,
    "memory": 0.10,    # 起步保守,看效果调
}

# memory 路计算:
async def compute_memory_score(self, signal: FusedSignal) -> float | None:
    text = format_signal_for_embedding(signal)
    embedding = await embedder.embed(text)
    similar = await retriever.search_similar(embedding, symbol_filter=signal.symbol)
    if len(similar) < 5:    # 样本不足
        return None
    return retriever.memory_score(similar)   # 映射到 0~100
```

### 4.9 Token 预算控制

```yaml
# 新增 LLM 预算桶
llm_budget_buckets:
  total_monthly_tokens: 30_000_000     # ~$30 at deepseek-chat 标准价
  buckets:
    judge: 0.50          # 入场前 LLM 判断
    post_mortem: 0.30    # agent post-mortem (这一路最贵)
    embedding: 0.10
    pre_rate: 0.05
    reflection: 0.05

# 各桶超额时的降级
on_bucket_freeze:
  judge: "neutral_verdict"
  post_mortem: "fallback_to_legacy_closed_set"
  embedding: "skip_memory_score_for_this_signal"
  pre_rate: "stop_pre_rating"
  reflection: "skip_reflection_report"
```

### 4.10 测试要求

`tests/test_memory_store_mock.py`:

- [ ] `test_insert_and_retrieve_basic`
- [ ] `test_kNN_returns_top_k_by_cosine`
- [ ] `test_time_decay_lowers_old_trade_weight`
- [ ] `test_symbol_filter_boosts_same_symbol`
- [ ] `test_vss_unavailable_falls_back_to_brute_force`
- [ ] `test_embedding_cache_hit_zero_tokens`
- [ ] `test_concurrent_inserts_thread_safe`

`tests/test_post_mortem_agent_mock.py`:

- [ ] `test_agent_terminates_within_max_tool_calls`
- [ ] `test_agent_total_tokens_under_cap`
- [ ] `test_agent_invalid_json_retries_once`
- [ ] `test_agent_invalid_json_after_retry_falls_back_to_legacy`
- [ ] `test_agent_tool_failure_continues_with_remaining_tools`
- [ ] `test_agent_output_actionable_rule_uses_only_closed_set_features`
- [ ] `test_token_bucket_freeze_falls_back_to_legacy`

`tests/test_fuser_with_memory_mock.py`:

- [ ] `test_memory_score_below_min_samples_returns_none`
- [ ] `test_four_way_blend_when_all_paths_active`
- [ ] `test_fuser_skips_memory_when_embedding_disabled`

### 4.11 已知 bug / 风险 / 应对

| # | 风险 | 应对 |
|---|---|---|
| C1 | sqlite-vss 在 docker 镜像里编译失败 | 用纯 Python `numpy + cosine` fallback;Dockerfile 加可选 build-arg 决定是否装 vss |
| C2 | DeepSeek embedding 模型改名/下线 | embedder 层用 model_name 配置化,挂了走 fallback embedding (用 LLM chat 输出 hex hash 凑个粗糙 embedding,纯保活) |
| C3 | Agent 死循环烧 token | 硬上限 `max_tool_calls=6` + `max_total_tokens=15k`,触顶强制返回 fallback |
| C4 | Agent 输出 JSON schema 不合法 | strict json schema 校验,失败重试 1 次,还失败走 legacy heuristic |
| C5 | 向量记忆冷启动空 | 用 PR #28 的 backtest engine 跑 90 天历史数据预热,产生 1000+ 条 TradeMemory;**这是 PR-C 上线前的硬要求** |
| C6 | LLM 输出的 actionable_rule 用了不在 closed-set 里的特征(如 "social_score_above_80") | 校验:rule.condition 的 key 必须是 8 个 closed-set features 子集;其他直接拒绝写入 dynamic_rules.json |
| C7 | 同一笔 trade 触发多次 post-mortem | trade_id 主键 + UPSERT,重复触发是 idempotent |
| C8 | 月底 token 预算耗尽 | TokenBudgetManager 已有 FREEZE 档;agent 走 legacy fallback,memory_score 还能返回(只用 embedding cache + 历史 trades) |
| C9 | Memory 数据库单文件涨到 10GB+ | `time_decay_half_life_days=30`;每月跑一次清理:删掉 closed_at < 180 天前的 trade(已经被衰减到几乎无权重) |
| C10 | 跨币种泛化得太激进:PEPE 学到的应用到 ETH | symbol_filter 加权 +20%,**不排除** 其他 symbol,但 retriever 的 top-k 里至少 50% 必须是同 symbol 或同 cluster(meme/AI/L1) |
| C11 | Embedding cache 跨 session 不共享 → 重复付费 | cache 持久化到 `.kiro/state/memory/embedding_cache.db`,key=hash(text),value=embedding |
| C12 | post_mortem_agent 改写 dynamic_rules.json 时和 RuleIndex 热加载 race | 复用 PR #4 的 atomic write (tmp + os.replace) 已经有了;不需要新加锁 |

### 4.12 PR-C 完成定义 (DoD)

- [ ] 全部测试矩阵过(4.10 的 17 项)
- [ ] PR-C 合并前必须跑完 90 天历史预热,产生 ≥ 1000 条 TradeMemory(操作员手工确认)
- [ ] `embedding_enabled=false` 时,fuser memory 路被禁用,行为退化到 PR-B
- [ ] `/metrics` 新增 `memory_db_size_bytes` / `memory_query_latency_ms` /
      `agent_tool_calls_total{tool}` / `agent_token_total` / `agent_fallback_total`
- [ ] dashboard 新增 "Memory & Learning" 面板:最近 10 笔 post-mortem 报告 + token 消耗曲线
- [ ] DeepSeek 月费从 PR-C 上线第一天起,每天 dashboard 看一眼,超过 $25/月发 telegram 警告

---

## 5. 跨 PR 共用的工程纪律

### 5.1 不可破坏的契约

每个 PR review 时必须显式说明这 7 项**没有动**:

1. RiskGate 14 闸的次序
2. CCXTExecutor cid 幂等格式 (`alt-{kind}-{hex}`)
3. AccountPersistor schema_version (改了要 bump 并写迁移)
4. TrailingStopFSM monotonic 不变量
5. dynamic_rules.json 是 SoT(.md 自动重生)
6. dashboard token + 默认 127.0.0.1
7. `LIVE_CONFIRM=I_UNDERSTAND` 启动门

### 5.2 测试基线

- PR 合并前 `pytest -q` 必须 ≥ 411 通过(当前基线)
- ruff check src/ scripts/ examples/ 必须 clean
- 新增测试每个必须 docstring 写明捍卫的不变量

### 5.3 默认 OFF 原则

任何新功能引入新行为 → AppConfig 默认 false。
PR 描述里必须有一节 "Default behaviour vs. before this PR",显式说差异。

### 5.4 Dashboard + Metrics 必加

任何新模块都必须:
- 至少 1 个 Prometheus gauge / counter
- 至少 1 个 dashboard 面板字段
- 失败必须有明确 metric `*_failures_total{reason}`

### 5.5 Token 预算红线

DeepSeek 月费 $30 是硬上限:
- TokenBudgetManager 已经有 4 档 (FREE/ECONOMY/EMERGENCY/FREEZE)
- 新增 4 个细分桶 (judge/post_mortem/embedding/pre_rate),独立 freeze
- dashboard 新增"DeepSeek Spend"曲线,每天看

### 5.6 Branch / Commit 纪律

- 每个 PR 单独 branch:`feat/social-discovery-pr-a` / `pr-b` / `pr-c`
- commit message 必须 reference 本计划单的章节:`feat(PR-A): ... refs SOCIAL_FIRST_DISCOVERY_PLAN.md §2.x`
- PR description 第一行必须是 "Implements §2 / §3 / §4 from SOCIAL_FIRST_DISCOVERY_PLAN.md"

---

## 6. 上线流程(30 天 dry-run + paper-trade)

### Day 0~3:PR-A 合并 + dry-run
- `discovery_enabled=true`,但 `social_scorer_enabled=false`(走旧 fuser)
- 观察:discovery pool 大小、各 source 的 score 分布、screener 订阅是否稳定
- **红线**:discovery_failures_total 增长率 > 10/min → 立即 disable,debug

### Day 4~10:PR-B 合并 + 三路 fuser 起步
- `social_scorer_enabled=true`
- `signal_blend_weights = {rule:0.60, social:0.20, llm:0.20}`
- `social_solo_trigger_enabled=false`
- 观察:social_score 分布、KOL hit_rate、circuit_breaker 触发频率
- **红线**:scraper 失败率 > 30% → 检查 cookie / 限流

### Day 11~17:逐步提高 social weight
- 满足以下条件再升:
  - 7 天 dry-run 无 daemon crash
  - circuit_breaker 总开启 < 5 次
  - KOL hit_rate 收敛(min 10 个作者有 ≥10 样本)
- 升到 `signal_blend_weights = {rule:0.45, social:0.35, llm:0.20}`

### Day 18~24:开 social_solo_trigger
- `social_solo_trigger_enabled=true`
- 观察:social_solo_triggers_total / fused_high_priority_total 比值
- **红线**:social_solo 触发的仓位 7 天内胜率 < 40% → 关闭

### Day 25:PR-C 合并(memory + agent)
- 必须先跑完 90 天 backtest 预热 memory db
- `embedding_enabled=true`,`memory_weight=0.10`
- 观察:agent fallback 率(目标 < 5%)、token 消耗(目标 < $25/月)

### Day 30:决策点
- 7 天 paper-trade(testnet)
- 满足:总 PnL > 0、max DD < 6%、agent fallback < 5%、scraper 稳定 → 切实盘 100 USDT 上限

### Day 30+:扩规模
- 实盘 100 USDT 跑 7 天稳了 → 1000 USDT
- 1000 USDT 跑 30 天稳了 → 操作员决定上限

---

## 7. 接手者的 Quick Start (5 分钟)

如果你是新对话刚打开这个文件:

1. **第一步**:读 §0(2 分钟)— 知道意图和预算
2. **第二步**:读 §1.1 + §1.2(1 分钟)— 知道哪些不能动
3. **第三步**:看 §6 上线流程当前停在哪一天(看 git log 和 dashboard)
4. **第四步**:打开当前 PR 对应的章节(§2 / §3 / §4)
5. **第五步**:看那一节的 "测试要求" 和 "已知风险" — 优先解决 risk 列表里没勾的

**不要**:
- 不要重新设计架构(PR #1~#39 已经定调,不能扔)
- 不要质疑"社交先选币"是否对(操作员 §0.1 已经拍板)
- 不要把 LLM 切到 Claude(预算 §0.3 锁死 DeepSeek $30/月)
- 不要引入本地模型(操作员 §0.3 明确拒绝)

---

## 8. 版本和签收

| 版本 | 日期 | 改动 | 签收 |
|---|---|---|---|
| 1.0 | 2026-05-17 | 初版,操作员拍板社交先选币 + DeepSeek $30 + 单账号 cookie + RSS+HTML 双路 | yiling-ui |

---

## 9. Backlog (V2.0 之后再说,本计划不做)

- WashTradingDetector trade_count 在 ccxt.pro 实盘的修复
- compute_leverage 对妖币的反向激励改造
- 真正的 RL 训练环路(actor-critic / Q-learning)
- 跨交易所套利(Binance vs Gate 价差捕捉)
- 链上数据接入(Bitquery / DexScreener 高级 API)
- 社交 sentiment 的 fine-tuning(目前关键词 + LLM,V2 可考虑训自己的小模型)
- 自动调仓位上限(根据当前账户 PnL 曲线动态调 max_concurrent)
