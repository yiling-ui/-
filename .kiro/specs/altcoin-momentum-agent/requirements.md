# Requirements — Altcoin Momentum Agent

> 多模态 AI 交易智能体，用于捕捉山寨币（RAVE / MYX / 类似低市值标的）早期的十倍爆拉或砸盘行情。
>
> 文档版本：v0.1 / Draft
> 维护人：（待填写）
> 关联代码仓库：本仓库根目录

---

## 1. 背景与目标

### 1.1 业务背景
山寨币的"妖盘"启动通常具有以下特征：
- 低流动性合约出现 1m / 5m 级异常成交量；
- 资金费率短时间内极端化（深度负费率 = 空头被逼，深度正费率 = 多头脆弱）；
- 未平仓合约（Open Interest, OI）阶梯式跳升；
- 同步出现链上巨鲸地址异动 / KOL 集中喊单 / 币安广场讨论增速暴涨；
- 价格行为符合 SMC（Smart Money Concept）中的流动性清扫与订单块突破。

人工盯盘无法在 1–3 分钟内同时完成「盘面 + 链上 + 社交 + 语义」四象限交叉验证，本系统目标是把这套闭环自动化。

### 1.2 总体目标
构建一个**事件驱动、低延迟、AI 辅助**的交易智能体，实现：
- **G1**：在妖币启动后 ≤ 60s 内产出可执行信号；
- **G2**：通过多模态融合，把"假突破 / KOL 出货"过滤掉，提升信号 precision；
- **G3**：风控硬约束下，单次最大回撤受控，不出现"一笔打穿账户"事件；
- **G4**：具备历史回放与 RL 权重更新能力，策略可自我演化。

### 1.3 非目标（Out of Scope，V1）
- DEX 链上抓单 / MEV（V2 再考虑）；
- 现货季度套利、跨所搬砖；
- 多用户 SaaS、多账户隔离；
- 自研 LLM / 微调（直接使用 DeepSeek API）；
- 移动端 / 图形界面（V1 仅 CLI + Web Dashboard）。

---

## 2. 角色与利益相关者

| 角色 | 描述 | 关注点 |
|---|---|---|
| 交易员（主用户） | 个人/小团队量化交易者 | 信号质量、回撤、可解释性 |
| 运维（同人） | 自部署、自维护 | 部署简单、报警及时、日志可查 |
| 策略研究员（同人或外协） | 调参、回测、迭代 | 回测真实性、复盘工具、权重表可读 |

V1 假设单用户、单账户、单机部署。

---

## 3. 功能需求

> 验收语句采用 **EARS（Easy Approach to Requirements Syntax）** 风格：
> *When [条件], the system shall [行为].*

### 3.1 模块 A — 盘面与链上数据监控中心 (Market & On-Chain Screener)

**FR-A1 多交易所实时行情接入**
- A1.1 系统应通过 `ccxt.pro` WebSocket 同时接入 Binance / OKX / Gate.io 永续合约的 `tickers / trades / orderbook(L2, 25 档) / funding / openInterest`。
- A1.2 当任一 WebSocket 断线，系统应在 ≤ 3s 内自动重连，并对断线期间的快照通过 REST 补齐。
- A1.3 系统应支持动态加载币种白名单（YAML 热更新），新增/删除币种无需重启。

**FR-A2 异常成交量识别（Volume Spike）**
- *When* 1m/5m K 线成交量 ≥ 过去 N 根（默认 N=60）的均值 + k·σ（默认 k=4） *and* 价格变动方向与放量方向一致，*the system shall* 在 500ms 内发布 `signal.volume_spike` 事件。
- 必须输出特征：`zscore, vol_ratio, side(buy/sell), klines_window`。

**FR-A3 资金费率异常**
- *When* 实时资金费率 ≤ -0.1% / 8h *or* ≥ +0.15% / 8h 持续 ≥ 2 个采样周期，*the system shall* 发布 `signal.funding_extreme` 事件。
- 阈值必须可配置且支持按币种 override。

**FR-A4 OI 激增**
- *When* 5m 窗口内 OI 增长 ≥ 15% *and* 价格波动 ≤ 1%，*the system shall* 标记为「静默建仓」事件 `signal.oi_silent_build`；当价格同向加速时标记为 `signal.oi_breakout`。

**FR-A5 SMC / 价格行为识别**
- A5.1 系统应在 1m/5m 多周期上识别：
  - **Liquidity Sweep**：价格瞬时穿破前 N 根高/低点 ≥ x bps 后回收（wick 长度 ≥ 实体 1.5 倍）；
  - **Order Block (OB)**：上一根反向 K 线的最后一根同向 K 线区间，被突破后视为有效 OB；
  - **BOS (Break of Structure)** / **CHoCH**：基于摆动高低点（fractal）。
- A5.2 每个 SMC 事件须输出：`type, level, timeframe, strength_score(0-1), reference_kline_id`。

### 3.2 模块 B — 社交情绪采集器 (Social Sentiment Crawler)

**FR-B1 数据源**
- 系统应支持以下来源（适配器化，可热插拔）：
  - 币安广场（Binance Square）热门帖子流；
  - 指定 KOL 列表（Twitter/X handle 列表，先用第三方镜像 API，无 Twitter 官方 key 时可降级为 Nitter）；
  - 可选：Telegram 公开群（V1.1）。
- **法务/合规备注**：币安广场抓取存在 ToS 风险，系统设计上必须保持适配器可移除；不得对单一来源 IP 轰炸（默认 ≥ 1 req/s 限速）。

**FR-B2 KOL 讨论热度增速**
- *When* 某 ticker 的提及量在 5m 窗口内 ≥ 过去 1h 中位数 × 5 倍，*the system shall* 发布 `signal.social_surge` 事件，附带文本样本（≤ 20 条）。

**FR-B3 文本归一化**
- 系统应抽取并归一化每条帖子的 `ticker_mentions, sentiment_keyword, author_id, author_follower_count, post_ts, raw_text`。
- 中英文混合文本须正确切词（jieba + 英文 tokenizer）。

### 3.3 模块 C — AI 研判引擎 (AI Inference Engine)

**FR-C1 触发条件**
- *When* Screener 输出的硬规则信号 + Social 输出的 surge 信号在 90s 内同币种交叉，*the system shall* 触发一次 DeepSeek 裁决。
- 单币种裁决冷却时间默认 ≥ 60s，避免重复消费 token。

**FR-C2 输入打包**
- 系统应将以下结构化上下文喂给 DeepSeek（JSON）：
  - 盘面特征：volume zscore、funding、OI 变化、SMC 事件列表；
  - 价格行为：最近 30 根 1m K 线 OHLCV 摘要；
  - 社交：最近 N 条原文 + 作者画像（粉丝数、历史准确率分位数，若有）。

**FR-C3 输出契约**
- DeepSeek 必须返回严格 JSON，字段：
  ```
  {
    "verdict": "pump_genuine" | "pump_distribution" | "noise" | "dump_genuine" | "dump_trap",
    "confidence": 0-1,
    "kol_intent": "frontrun_call" | "exit_liquidity" | "neutral",
    "key_evidence": ["..."],
    "potential_score": 0-100
  }
  ```
- 解析失败、超时（>3s）、限流时，系统应**降级**到纯规则分（不阻塞主循环）。

**FR-C4 爆发潜力指数（融合分）**
- 最终 `potential_score_final = w1·rule_score + w2·llm_potential_score`，权重读取 `policy_weights.yaml`。
- 当 `potential_score_final ≥ 85`，标记为高优信号 `signal.high_priority`。

### 3.4 模块 D — 风控与执行引擎 (Execution & Risk)

**FR-D1 风控硬墙（独立于策略）**
- 所有下单请求必须经过 Risk Gate；以下任一条件触发拒单：
  - 单笔风险敞口 > 账户权益 × `max_risk_per_trade`（默认 1%）；
  - 24h 累计亏损 > 账户权益 × `daily_drawdown_limit`（默认 4%）→ 触发当日熔断；
  - 当前持仓数 ≥ `max_concurrent_positions`（默认 3）；
  - 标的 24h 平均 orderbook 5 档深度 < `min_liquidity_usdt`（避免拍死自己）。

**FR-D2 仓位计算**
- 基于"风险等额（risk parity per trade）"：`size = (equity × risk_pct) / |entry - initial_stop|`，再按合约面值与杠杆换算张数。

**FR-D3 自适应追踪止损**
- 入场后，止损初始放在最近一个 SMC OB 的反向边界 + buffer。
- 浮盈达到 1R 后，止损上移到保本（breakeven）。
- 浮盈达到 2R 后，启用 ATR-trailing：`stop = max(prev_stop, price - n·ATR(14))`，n 默认 2。
- 所有调整必须以"只朝有利方向移动"为不变量。

**FR-D4 下单**
- *When* `signal.high_priority` 通过 Risk Gate，*the system shall* 通过 ccxt 在对应交易所下市价 / 限价单（默认 IOC 限价 + slippage 上限 0.3%）。
- 下单失败必须重试（指数退避，最多 3 次），仍失败则报警并放弃。

### 3.5 模块 E — 回测与强化学习模块 (RL & Backtest Lab)

**FR-E1 历史数据回放**
- 系统应基于落地的 tick / kline / funding / OI / 社交文本，构造**事件回放器**，按真实时间序还原至 Screener 输入端。
- 撮合滑点模型必须基于历史 orderbook 深度（不允许使用固定 bps）。

**FR-E2 LLM 事后复盘**
- 给定一段历史窗口（通常某个妖币启动前后 ±2h），系统应自动调用 DeepSeek 标注："启动前 X 分钟出现哪些可识别特征"。
- 复盘结果落 `data/postmortem/<symbol>_<date>.json`。

**FR-E3 策略权重更新**
- 系统应支持基于一组复盘结果，更新 `policy_weights.yaml` 中各特征权重；
- 算法 V1：贝叶斯加权（每个特征历史命中率作为先验，新样本更新）；V2：可替换为 contextual bandit / PPO。
- 权重更新必须有版本号 + 回滚能力。

**FR-E4 报告**
- 回测须输出：胜率、盈亏比、最大回撤、Sharpe、Calmar、单笔分布、按特征切片的 PnL attribution。

---

## 4. 非功能需求

| 维度 | 要求 |
|---|---|
| **延迟** | Hot path（事件 → 规则信号）p99 < 500ms；LLM 裁决 p95 < 3s；下单端到端 p99 < 1.5s |
| **吞吐** | 同时监控 ≤ 200 个 symbol，事件总线 ≥ 5k msg/s |
| **可用性** | 单机 99%，关键事件持久化，重启 ≤ 60s 恢复订阅 |
| **可观测性** | 全链路 trace_id；Prometheus 指标（信号数/秒、LLM 调用耗时直方图、下单成功率、PnL）；Loki/文件结构化日志 |
| **成本** | LLM 月调用预算可配置（默认 $200/月），超额自动降级为纯规则模式 |
| **安全** | API key 全部走 `.env` + OS keyring；下单 key 与查询 key 必须分离；Risk Gate 不可绕过 |
| **可移植** | Python 3.10+；Docker compose 一键部署；x86 / arm64 通用 |

---

## 5. 约束与假设

- **C1** 编程语言锁定 Python ≥ 3.10（用到 `match`、PEP 604 联合类型）；
- **C2** 行情统一通过 ccxt（含 `ccxt.pro`），不直接对接交易所原始协议；
- **C3** AI 模型仅使用 DeepSeek API（chat / reasoner 双模型，按场景路由）；
- **C4** 默认运行在 1 台 8C/16G 云主机，SSD ≥ 200GB；
- **C5** 不做客户资金托管；账户与 API Key 由用户自持。

---

## 6. 风险与开放问题

| ID | 风险 | 缓解 |
|---|---|---|
| R1 | 币安广场抓取被风控 / 触犯 ToS | 适配器化、UA 轮换、限速、可一键禁用，备用 Nitter / 自建爬虫 |
| R2 | 山寨币真实滑点远大于回测假设 | 撮合引擎强制基于 orderbook 深度模拟；活跃度过滤（FR-D1） |
| R3 | DeepSeek 拒答 / 超时导致信号丢失 | 主链路异步、LLM 仅做加分裁决；规则分独立可下单 |
| R4 | 极端行情下交易所 API 限流 | ccxt 内置限速器 + 指数退避；下单与查询 key 分离 |
| R5 | KOL 利用系统反向收割（喊单出货） | DeepSeek 显式判断 `kol_intent=exit_liquidity` 时，融合分上限封顶 |
| R6 | 单点机器宕机 | V1 接受；V1.1 引入热备 + Redis Streams 复制 |
| R7 | 法律合规 | 用户自行评估所在司法辖区交易合规性；本系统不对外提供 |

**开放问题（需用户决策）**：
- Q1：DeepSeek 月预算上限？默认 $200 是否合适？
- Q2：是否希望 V1 即支持现货爆拉抓取，还是仅永续？
- Q3：Telegram 群抓取是否需要在 V1 落地？
- Q4：风控参数（最大单笔 1%、日内 4%）是否符合实际偏好？

---

## 7. 验收（V1 Done 定义）

- [ ] 5 个模块均通过单模块集成测试（见 `tasks.md`）；
- [ ] 端到端在测试网（Binance Testnet）跑通一次"事件 → AI → 下单 → 追踪止损 → 平仓"完整闭环；
- [ ] 历史回放能完整复现至少 3 个真实妖币（如 RAVE / MYX 等）启动行情，并产出复盘报告；
- [ ] Risk Gate 单测覆盖率 ≥ 90%，且通过混沌测试（注入异常订单、断网、LLM 超时）；
- [ ] 部署文档可让一名陌生工程师在 30 分钟内拉起系统。
