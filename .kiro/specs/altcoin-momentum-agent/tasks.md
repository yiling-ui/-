# Tasks — Altcoin Momentum Agent

> 与 `requirements.md` / `design.md` 配套。把 5 个核心模块拆成**可独立测试**的开发子任务。
>
> 文档版本：v0.1 / Draft

---

## 0. 任务约定

### 0.1 编号规则
- `T-{模块代号}-{序号}`：例如 `T-A-03` 表示 Screener 模块的第 3 个任务。
- 模块代号：`A` 盘面 Screener，`B` 社交 Crawler，`C` AI Inference，`D` 风控/执行，`E` 回测/RL，`X` 跨模块基础设施。

### 0.2 每个任务必须包含的字段
- **Goal**：一句话目标。
- **Inputs**：依赖的上游数据/接口。
- **Outputs**：产出物（代码路径、事件 topic、配置项等）。
- **Acceptance（DoD）**：明确、可验证的完成标准。
- **Test**：单测/集成测试做法。
- **Depends on**：前置任务编号。
- **Estimate**：粗略人日（仅供排期参考）。

### 0.3 通用约束（适用所有任务）
- 代码风格：`ruff` + `black` + `mypy --strict`；
- 测试框架：`pytest` + `pytest-asyncio` + `respx`（mock httpx）+ `freezegun`；
- 覆盖率要求：核心模块 ≥ 80%，Risk Gate ≥ 90%；
- 全部异步代码必须可以在 `pytest-asyncio` 下被驱动；
- 任何对外部服务（交易所、DeepSeek、Twitter）的依赖必须可 mock 注入，禁止单测直连真实服务。

---

## 1. 跨模块基础设施（X）

### T-X-01 项目脚手架
- **Goal**：建立 Python 3.10+ 工程骨架。
- **Outputs**：`pyproject.toml`（poetry 或 uv），`src/altcoin_agent/`，`tests/`，`.editorconfig`，`.pre-commit-config.yaml`，`Makefile`。
- **Acceptance**：`make lint test` 在空骨架上能跑通；CI（GitHub Actions）跑过 lint+test。
- **Test**：CI 绿。
- **Depends on**：—
- **Estimate**：0.5d

### T-X-02 配置层（pydantic-settings + YAML）
- **Goal**：统一配置加载与热更新。
- **Outputs**：`src/altcoin_agent/config.py`（Settings 模型），`config/app.yaml.example`。
- **Acceptance**：缺字段时启动报错并指出路径；支持环境变量 override；支持监听文件变更触发回调。
- **Test**：单测覆盖：缺字段、错类型、环境变量覆盖、热更新回调被触发。
- **Depends on**：T-X-01
- **Estimate**：0.5d

### T-X-03 事件总线封装（Redis Streams）
- **Goal**：统一 producer / consumer / consumer-group / dead-letter 接口。
- **Outputs**：`src/altcoin_agent/bus/{producer,consumer,schema}.py`；topic 命名常量；trace_id 注入。
- **Acceptance**：
  - 支持 at-least-once；
  - 支持消费组、ack、pending 重投；
  - 消息体序列化用 msgpack；
  - 内嵌 backpressure：单 stream 长度 > 阈值时 producer 报警。
- **Test**：用 `fakeredis` 跑：发布-订阅、消费组分发、ack 失败重投、序列化兼容性。
- **Depends on**：T-X-01
- **Estimate**：1d

### T-X-04 持久层（TimescaleDB + Parquet）
- **Goal**：把 raw 行情、特征、信号、订单写入存储，供回测使用。
- **Outputs**：`src/altcoin_agent/persist/`：`timescale_writer.py`、`parquet_writer.py`、迁移脚本（`alembic` 或裸 SQL）。
- **Acceptance**：
  - 自动批量写（每 N 条或每 T ms flush）；
  - Parquet 按 `symbol/date` 分区；
  - 启动时自检 schema 与索引存在。
- **Test**：单测注入假数据，验证 SQL/Parquet 输出；集成测试用 docker-compose 起 TimescaleDB。
- **Depends on**：T-X-03
- **Estimate**：1.5d

### T-X-05 观测：日志 / Metrics / Tracing
- **Goal**：全链路可观测性骨架。
- **Outputs**：`observability/{logger,metrics,tracing}.py`；Prometheus exporter 端口；OpenTelemetry SDK 接入；Grafana dashboard JSON 模板。
- **Acceptance**：
  - 所有事件带 trace_id；
  - `/metrics` 暴露 design.md §8 列出的关键指标；
  - 日志 JSON 格式 + 默认脱敏（API key / token）。
- **Test**：单测 metrics 计数；集成测试拉一个 trace_id 端到端贯通。
- **Depends on**：T-X-01
- **Estimate**：1d

### T-X-06 部署（docker-compose）
- **Goal**：一键拉起 Redis、TimescaleDB、Prometheus、Grafana、Loki、agent 容器。
- **Outputs**：`docker-compose.yml`、`docker/agent.Dockerfile`、`docs/deploy.md`。
- **Acceptance**：陌生工程师 ≤ 30 分钟拉起；带健康检查；带卷持久化。
- **Test**：CI smoke test：起容器 → 调 `/healthz` → 关闭。
- **Depends on**：T-X-03、T-X-04、T-X-05
- **Estimate**：1d

---

## 2. 模块 A — 盘面与链上数据监控中心

### T-A-01 ccxt.pro 多交易所连接器
- **Goal**：抽象 `ExchangeFeed` 协议；实现 Binance / OKX / Gate.io 永续合约。
- **Outputs**：`screener/feeds/{base,binance,okx,gate}.py`。
- **Acceptance**：
  - watchTrades / watchOHLCV(1m,5m) / watchOrderBook(25) / watchFundingRate / watchOpenInterest 全通；
  - 断线 ≤ 3s 自动重连；
  - 每个 feed 独立健康度指标。
- **Test**：用 ccxt 内置 mock / 录制流回放（`vcrpy` 风格）：断线重连、字段映射、序列化输出。
- **Depends on**：T-X-03
- **Estimate**：2d

### T-A-02 实时特征计算（zscore / ATR / OI delta）
- **Goal**：纯 numpy 滚动窗口特征。
- **Outputs**：`screener/features/{rolling,atr,oi.py}`，`signals_schema.py`。
- **Acceptance**：
  - 1k symbol × 60s 窗口下 CPU < 50%（单核）；
  - 与离线 pandas 实现误差 ≤ 1e-9。
- **Test**：与 pandas 黄金值对拍；性能基准 `pytest-benchmark`。
- **Depends on**：T-A-01
- **Estimate**：1.5d

### T-A-03 SMC 检测器（BOS / CHoCH / OB / Liquidity Sweep）
- **Goal**：基于 fractal 的有限状态机。
- **Outputs**：`screener/features/smc.py`；产出事件 `signal.market.smc.*`。
- **Acceptance**：
  - 在已标注的 fixtures（≥ 20 段历史 K 线）上召回 ≥ 0.8、precision ≥ 0.7；
  - 参数（fractal window、wick 比、buffer）可配置。
- **Test**：fixtures 黄金标注比对；增量更新与重算结果一致。
- **Depends on**：T-A-02
- **Estimate**：2.5d

### T-A-04 规则引擎（Volume Spike / Funding Extreme / OI Surge）
- **Goal**：根据特征发布硬规则信号。
- **Outputs**：`screener/rules/`；topic `signal.market.*`。
- **Acceptance**：
  - 阈值热更新（监听 config 变更）；
  - 同一根 K 线只触发一次（去重 key = `{symbol, kline_id, rule}`）。
- **Test**：注入合成 K 线序列，断言事件次数与字段。
- **Depends on**：T-A-02、T-A-03、T-X-02
- **Estimate**：1d

### T-A-05 Screener 端到端集成测试
- **Goal**：从 mock 交易所 WS → 持久化 → 事件总线全链路通。
- **Outputs**：`tests/integration/test_screener_e2e.py`。
- **Acceptance**：3 个交易所并发，10 个 symbol，10 分钟回放跑通；规则信号数量与黄金值一致。
- **Test**：本身就是测试任务。
- **Depends on**：T-A-01..04、T-X-04
- **Estimate**：1d

---

## 3. 模块 B — 社交情绪采集器

### T-B-01 SocialSourceAdapter 协议与限速器
- **Goal**：定义统一适配器协议；全局令牌桶限速。
- **Outputs**：`social/adapters/base.py`，`social/limiter.py`。
- **Acceptance**：
  - 协议含 `stream() -> AsyncIterator[Post]`；
  - 默认 ≥ 1 req/s；可按 source 单独配置；
  - 一键禁用某 adapter。
- **Test**：单测限速触发与释放；多 adapter 并发不互踩。
- **Depends on**：T-X-02、T-X-03
- **Estimate**：0.5d

### T-B-02 Binance Square 适配器
- **Goal**：抓取热门 / 关键词流。
- **Outputs**：`social/adapters/binance_square.py`。
- **Acceptance**：
  - UA 池 + 失败回退；
  - 输出标准 Post；
  - 触发反爬时熔断 5 分钟。
- **Test**：用 `respx` mock HTML/JSON 响应；失败重试与熔断断言。
- **Depends on**：T-B-01
- **Estimate**：1.5d

### T-B-03 Twitter / Nitter KOL 适配器
- **Goal**：基于 KOL 白名单抓推文。
- **Outputs**：`social/adapters/{twitter,nitter}.py`，`config/kol_list.yaml`。
- **Acceptance**：
  - 支持镜像/Nitter 双源 fallback；
  - 输出包含 follower_count（若可得）。
- **Test**：mock 响应测试；空列表/限流路径。
- **Depends on**：T-B-01
- **Estimate**：1.5d

### T-B-04 TickerExtractor（中英文分词 + 词典）
- **Goal**：从原文抽 ticker 列表。
- **Outputs**：`social/extractors/ticker.py`，`config/ticker_aliases.yaml`。
- **Acceptance**：
  - 召回 ≥ 0.9 在标注集（≥ 500 条）；
  - 误抽率 ≤ 0.05。
- **Test**：标注 fixtures 比对；中英文混排、emoji、$XYZ 形式。
- **Depends on**：T-B-01
- **Estimate**：1d

### T-B-05 Surge Detector
- **Goal**：mention count 时间序列异常检测，发布 `signal.social.surge`。
- **Outputs**：`social/surge.py`。
- **Acceptance**：
  - 与 design.md FR-B2 一致；
  - 抗噪：低基数（mention < 10）时不触发。
- **Test**：合成时间序列；金标历史事件回测。
- **Depends on**：T-B-04
- **Estimate**：1d

### T-B-06 Social 端到端集成测试
- **Goal**：模拟两源并发 → 抽取 → surge → 事件总线。
- **Outputs**：`tests/integration/test_social_e2e.py`。
- **Acceptance**：注入合成流，事件数量、字段、限速行为全部断言通过。
- **Depends on**：T-B-02..05、T-X-03
- **Estimate**：0.5d

---

## 4. 模块 C — AI 研判引擎

### T-C-01 DeepSeek 客户端与重试 / 预算
- **Goal**：基于 httpx 异步封装 DeepSeek `chat` / `reasoner`；带超时、重试、token 计数、月预算。
- **Outputs**：`inference/deepseek_client.py`，`inference/budget.py`。
- **Acceptance**：
  - 超时 3s、重试 1 次；
  - token 写 Redis counter，超预算 raise `BudgetExceeded`；
  - 错误类型分类（网络 / 限流 / 4xx / 5xx）并 metric 上报。
- **Test**：`respx` mock：200/429/500/timeout；预算计数；并发节流。
- **Depends on**：T-X-03、T-X-05
- **Estimate**：1d

### T-C-02 Prompt 模板与 JSON 强约束解析
- **Goal**：固定 prompt schema + pydantic 解析 + 修正重试。
- **Outputs**：`inference/prompt.py`，`inference/schema.py`。
- **Acceptance**：
  - 输出契约见 requirements.md FR-C3；
  - 解析失败重试 1 次，二次失败降级为 `noise/score=0`。
- **Test**：注入合法/非法/不完整响应；解析与降级路径全部覆盖。
- **Depends on**：T-C-01
- **Estimate**：1d

### T-C-03 触发器与冷却 / 并发调度
- **Goal**：消费 `signal.market.*` + `signal.social.*` 滑窗交叉，触发 LLM 调用。
- **Outputs**：`inference/trigger.py`，`inference/scheduler.py`。
- **Acceptance**：
  - 单 symbol 冷却 60s；
  - 全局并发 ≤ 4；
  - 触发统计写 metric。
- **Test**：合成事件流（同币种、跨币种、超 90s 窗口），断言触发次数与冷却。
- **Depends on**：T-C-02、T-X-03
- **Estimate**：1d

### T-C-04 Score Fuser（融合分）
- **Goal**：`potential_score_final = w1·rule_score + w2·llm_score`，含降级权重。
- **Outputs**：`inference/fuser.py`，使用 `config/policy_weights.yaml`。
- **Acceptance**：
  - LLM 缺失时自动退化为纯规则；
  - `kol_intent=exit_liquidity` 时融合分上限封顶（默认 70）；
  - 输出 ≥ 85 → 发布 `signal.high_priority`。
- **Test**：参数化覆盖：LLM 在/不在、KOL 出货、规则强弱组合。
- **Depends on**：T-C-02、T-A-04、T-B-05
- **Estimate**：1d

### T-C-05 Inference 端到端集成测试
- **Goal**：mock DeepSeek + 合成事件，验证全链路。
- **Outputs**：`tests/integration/test_inference_e2e.py`。
- **Acceptance**：从 signal → LLM → fuser → high_priority 事件，时延 p95 < 3s（mock 下）。
- **Depends on**：T-C-01..04
- **Estimate**：0.5d

---

## 5. 模块 D — 风控与执行引擎

### T-D-01 Risk Gate 核心
- **Goal**：硬墙；任一 fail 即拒；fail-closed。
- **Outputs**：`risk/gate.py`，`risk/state.py`（持仓 / 当日 PnL 状态）。
- **Acceptance**：覆盖 design §3.4 的 7 项检查；返回结构 `{approved, reason, normalized_intent}`。
- **Test**：参数化 ≥ 30 个用例覆盖每条规则；混沌测试：异常输入不能产生 approve。
- **Depends on**：T-X-02、T-X-03
- **Estimate**：1.5d

### T-D-02 仓位计算（risk parity）
- **Goal**：根据风险敞口、入场价、初始止损算 size。
- **Outputs**：`risk/sizing.py`。
- **Acceptance**：与 design §3.4 公式一致；考虑合约面值、最小下单步进、杠杆上限。
- **Test**：金标用例（手算）≥ 10 组；边界（极小/极大止损距离）。
- **Depends on**：T-D-01
- **Estimate**：0.5d

### T-D-03 Trailing Stop FSM
- **Goal**：状态机：INIT → ARMED → BREAKEVEN → TRAILING → CLOSED；不变量：stop 单调向利。
- **Outputs**：`risk/trailing.py`。
- **Acceptance**：
  - 不变量在所有路径下都成立（基于属性测试 `hypothesis`）；
  - 1R / 2R / ATR-trailing 转换正确。
- **Test**：单测 + `hypothesis` 属性测试随机 tick 序列。
- **Depends on**：T-D-02
- **Estimate**：1.5d

### T-D-04 ccxt 实盘 Executor
- **Goal**：实际下单 / 撤单；限价 IOC、滑点保护、重试。
- **Outputs**：`execution/ccxt_executor.py`。
- **Acceptance**：
  - 滑点 > 0.3% 拒绝；
  - 网络错指数退避；
  - 下单 / 成交事件全部落 `order.*`。
- **Test**：用 ccxt sandbox / mock 适配器；模拟限流、超时、部分成交。
- **Depends on**：T-D-01..03、T-A-01
- **Estimate**：1.5d

### T-D-05 持仓 Reconciler（启动 & 周期）
- **Goal**：交易所持仓为 source of truth；启动全量比对，运行期周期校对。
- **Outputs**：`risk/reconciler.py`。
- **Acceptance**：本地状态与交易所 diff 时报警，按策略修复（默认告警人工介入）。
- **Test**：注入差异场景（多/少/方向反），断言 alert 与修复行为。
- **Depends on**：T-D-04
- **Estimate**：1d

### T-D-06 Risk + Execution 端到端集成测试
- **Goal**：从 `signal.high_priority` → Gate → Executor → Position Mgr → 平仓 全流程。
- **Outputs**：`tests/integration/test_exec_e2e.py`。
- **Acceptance**：覆盖正常成交 / 被拒单 / 滑点超限 / 网络重试 / 触发 trailing stop 平仓。
- **Depends on**：T-D-01..05
- **Estimate**：1d

---

## 6. 模块 E — 回测与强化学习

### T-E-01 Event Replayer
- **Goal**：从 TimescaleDB / Parquet / 社交 JSON 合并出全局有序事件流；可加速倍率。
- **Outputs**：`lab/replayer.py`。
- **Acceptance**：
  - 时间戳全局单调；
  - 与实盘 Bus 接口 100% 兼容（模块代码 0 修改）；
  - 加速倍率 1x ~ 100x 可调。
- **Test**：黄金 fixtures（已知事件序）回放 → 顺序、字段一致。
- **Depends on**：T-X-04
- **Estimate**：1.5d

### T-E-02 Sim Executor（基于 orderbook 深度撮合）
- **Goal**：用历史 L2 快照模拟成交、滑点、未成交。
- **Outputs**：`execution/sim_executor.py`。
- **Acceptance**：
  - 拒绝"完美成交"假设；吃深度算 VWAP；
  - 与实盘 Executor 接口同名同形（Liskov）。
- **Test**：注入已知 orderbook，预期 fill 价 / 数量与手算一致。
- **Depends on**：T-E-01
- **Estimate**：1.5d

### T-E-03 回测 Runner + 报告
- **Goal**：在 backtest mode 下复用全部模块；产出报告。
- **Outputs**：`lab/runner.py`，`lab/reports/`，`reports/{run_id}/index.html`，mlflow 注册。
- **Acceptance**：
  - 输出胜率 / 盈亏比 / MDD / Sharpe / Calmar / 单笔分布 / 特征 PnL 归因；
  - 同一份历史 + 同一权重，多次回测结果可复现（固定种子）。
- **Test**：小规模 fixtures 回测稳定复现；指标值与手算一致。
- **Depends on**：T-E-01、T-E-02、T-A-05、T-C-05、T-D-06
- **Estimate**：2d

### T-E-04 LLM 事后复盘
- **Goal**：对历史妖币启动窗口生成结构化复盘。
- **Outputs**：`lab/postmortem.py`，`data/postmortem/{symbol}_{date}.json`。
- **Acceptance**：
  - 输出含 `trigger_features`、`false_positive_features`、`narrative`；
  - 失败 / 限流时不污染主流程。
- **Test**：mock DeepSeek，校验输出 schema 与字段完整。
- **Depends on**：T-C-01
- **Estimate**：1d

### T-E-05 策略权重更新（贝叶斯加权）
- **Goal**：基于复盘 / 回测命中率更新 `policy_weights.yaml`。
- **Outputs**：`lab/weights_update.py`，权重文件版本化（保留近 10 版）。
- **Acceptance**：
  - 一键回滚到任意历史版本；
  - 权重总和归一；
  - 更新前后 dry-run 报告（diff + 预期影响）。
- **Test**：注入合成命中率序列，与手算后验对拍；回滚动作幂等。
- **Depends on**：T-E-03、T-E-04
- **Estimate**：1d

### T-E-06 回测/复盘流水线编排
- **Goal**：用 Prefect / Makefile 把"拉数据 → 回测 → 复盘 → 更新权重 → 报告"串起来。
- **Outputs**：`lab/pipelines/`、`Makefile` targets：`make backtest`、`make postmortem`、`make update-weights`。
- **Acceptance**：单条命令可跑通一周历史的完整流水。
- **Depends on**：T-E-01..05
- **Estimate**：1d

---

## 7. 端到端 / 上线前任务

### T-Z-01 测试网（Binance Testnet）端到端
- **Goal**：实盘代码路径在测试网完成一次完整闭环。
- **Acceptance**：从 WS 抓取 → 信号 → AI → 下单 → 追踪止损 → 平仓 → 落库 → 报告。
- **Depends on**：所有 A/B/C/D 模块完成
- **Estimate**：1d

### T-Z-02 混沌测试
- **Goal**：注入故障验证降级。
- **Cases**：
  - DeepSeek 全超时；
  - Redis 重启；
  - 交易所 WS 全断 30s；
  - 月预算瞬间打满；
  - Risk Gate 收到畸形 intent。
- **Acceptance**：系统不下错单、不打穿账户；告警全部触达。
- **Depends on**：T-Z-01
- **Estimate**：1d

### T-Z-03 文档与 Runbook
- **Goal**：交付可读文档。
- **Outputs**：`README.md`、`docs/deploy.md`、`docs/runbook.md`（常见故障处置）、`docs/strategy.md`（策略说明 & 调参指南）。
- **Acceptance**：陌生工程师按文档可独立部署 + 调一组参数 + 跑一次回测。
- **Depends on**：T-Z-01
- **Estimate**：1d

---

## 8. 任务依赖图（粗粒度）

```
T-X-01 ──┬─▶ T-X-02 ─┐
         ├─▶ T-X-03 ─┼─▶ T-X-04 ─▶ T-X-06
         └─▶ T-X-05 ─┘

T-X-03 ─▶ T-A-01 ─▶ T-A-02 ─▶ T-A-03 ─▶ T-A-04 ─▶ T-A-05
T-X-03 ─▶ T-B-01 ─▶ T-B-02/03/04 ─▶ T-B-05 ─▶ T-B-06
T-X-03 ─▶ T-C-01 ─▶ T-C-02 ─▶ T-C-03 ─▶ T-C-04 ─▶ T-C-05
                                          ▲
                            T-A-04 / T-B-05 ┘

T-C-04 ─▶ T-D-01 ─▶ T-D-02 ─▶ T-D-03 ─▶ T-D-04 ─▶ T-D-05 ─▶ T-D-06

T-X-04 ─▶ T-E-01 ─▶ T-E-02 ─▶ T-E-03 ─▶ T-E-05
                                       ▲
                              T-C-01 ─▶ T-E-04 ┘
T-E-01..05 ─▶ T-E-06

(全部就绪) ─▶ T-Z-01 ─▶ T-Z-02 ─▶ T-Z-03
```

---

## 9. 排期参考（粗算）

| 阶段 | 内容 | 大致工时 |
|---|---|---|
| Sprint 1 | X 全套 + A 全套 | ~10d |
| Sprint 2 | B 全套 + C 全套 | ~10d |
| Sprint 3 | D 全套 + E-01/02 | ~9d |
| Sprint 4 | E-03..06 + Z 全套 | ~8d |

> 真实排期需结合人手与不可抗力（交易所 API 变更等）评估，文档仅作初版基线。

---

## 10. 风险点（任务侧）

- **T-A-03 SMC 检测**：业内对 OB / Liquidity Sweep 定义不统一，必须以仓库内 fixtures 标注作为唯一事实标准；
- **T-B-02 Binance Square**：抓取稳定性是最大不确定项，提前准备 Nitter/Twitter 备份方案，避免阻塞 Sprint 2；
- **T-C-01 DeepSeek 预算**：开发期容易把 token 烧光，建议默认开"sandbox 模式"返回固定假响应；
- **T-D-04 实盘 Executor**：交易所 sandbox 行为与生产存在差异，必须搭配 T-Z-01 测试网回归；
- **T-E-02 Sim Executor**：orderbook 历史数据量巨大，需要尽早决定存储策略（Parquet 分区 / 抽稀采样）。
