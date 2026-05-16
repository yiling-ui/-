# 机会成本惩罚 + 生产级工程化 — 追加计划

> **本文档是 [QUADRANT_STRATEGY_PLAN.md](./QUADRANT_STRATEGY_PLAN.md) 的并行追加。**
>
> 包含两件事：
> 1. **机会成本惩罚机制**（防止策略婆婆妈妈不开单）
> 2. **生产级工程化清单**（4 个维度 P0~P3 缺口）
>
> 两份计划独立但相互依赖——业务策略（赚钱逻辑）+ 基础设施（能稳定跑下来），实盘启用前**两份必须都完成**。

---

# Part A：机会成本惩罚机制

## A.1 现状（已查阅代码确认）

| 现有机制 | 局限 |
|---|---|
| `learning_engine.run_post_mortem` | **只在开仓后 1h 触发**，从不复盘"被拒绝"的信号 |
| `fuser.learned_overall_reward_cap=1.30` / `penalty_cap=0.70` | 只学已开仓的好坏，不学"该开没开" |
| `compute_event_result(expected_direction=...)` | 半成品，只用于已开仓反向（long 止损→missed_pump），**没接到 RiskGate 拒单流** |
| `audit_log.py decisions.jsonl` | ✅ **已记录所有 approved + rejected 决策**——数据基础已有 |

**完全缺失**：
- 识别"被拒绝但事后大涨"的信号
- 对"过度优柔寡断"的反向训练
- 强制再复盘 / 自适应阈值松绑

## A.2 设计：MissPenaltyEngine（机会成本惩罚引擎）

### A.2.1 核心数据结构

```python
@dataclass
class MissedOpportunity:
    """被拒绝的信号 + 事后实际行情对比"""
    trace_id: str               # 来自 decisions.jsonl
    symbol: str
    rejected_at_ts: int          # RiskGate 拒绝时刻
    rejected_reason: str         # "anti_chase" / "min_liquidity" / 
                                 # "consecutive_loss_cooldown" 等
    rejected_score: float        # 当时的 final_score
    direction: str               # "long" / "short"
    entry_price_if_taken: float  # 当时的 trigger_price

    # 事后回看（24h 后才算）
    realized_max_favorable_pct: float    # 朝预期方向走了多少
    realized_max_adverse_pct: float      # 朝反方向走了多少
    would_have_pnl_pct: float            # 若开仓 1.5% risk + 当时 leverage
    
    # 是否构成"错过的妖币"
    is_missed_pump: bool         # MFE >= 100% AND MAE <= 30% 
    miss_severity: float         # 0.0..1.0，按 MFE 大小分级
    
    # 学习字段
    rejected_reason_bucket: str  # 把 reason 归类
    market_regime: str           # accumulation/ramp/parabolic/blowoff/...
```

### A.2.2 工作流程

```
每天凌晨 02:00 UTC 跑一次（与 P0 工程化的训练 03:00 错开）：

1. 读取 decisions.jsonl 过去 24h 所有 rejected 决策
2. 对每个 reject，从 ccxt fetch 该 symbol 自拒绝时刻起 24h 的 1m K 线
3. 计算 MFE (Max Favorable Excursion) / MAE (Max Adverse Excursion)
4. 判定 is_missed_pump:
   - LONG reject + 24h MFE >= +100% + MAE <= -30% → missed_pump
   - SHORT reject + 24h MFE >= +50%（向下）+ MAE <= +20%（向上）→ missed_dump
5. 写入 missed_opportunities.jsonl
6. 更新 reject_reason 的 penalty 权重（见 A.2.3）
7. 触发 fuser 阈值自适应（见 A.2.4）
```

### A.2.3 拒单理由的奖惩积分

每个拒单理由维护一个**积分表**，从 0 出发，每天调整：

```python
@dataclass
class RejectReasonScore:
    reason: str                  # 如 "anti_chase" / "min_liquidity"
    correct_rejects: int         # 该 reason 拒了，事后该 symbol 真亏 → +1
    missed_pumps: int            # 该 reason 拒了，事后大涨 → -3 (惩罚加倍)
    confidence_score: float      # = (correct - 3*missed) / total
    
    # 自动调整阈值（仅当置信度低时）
    suggested_threshold_adjust_pct: float
    
    # 训练保护：50 个样本以下不调整
    samples_required: int = 50
```

错过的惩罚**是命中的 3 倍**——这是核心激励：
- 拒了一个垃圾信号 = +1 分（次要功劳）
- 错过一个 +200% 妖币 = -3 分（重大失误）

数学含义：要让一个拒单理由保持"正分"，它必须**正确拒绝 75% 以上**的真垃圾信号；如果它的正确率 < 75%，得分变负，触发阈值调整。

### A.2.4 自适应阈值松绑（关键）

每周日 04:00 UTC 跑一次：

```
对每个 reject_reason，如果：
  - samples >= 50
  - confidence_score < -10 （明显不靠谱）
  - 错过的 missed_pumps >= 5

则触发 LOOSEN：
  - anti_chase_max_move_pct: 2.5% → 4.0% (按象限)
  - min_liquidity_usdt: 200,000 → 150,000
  - consecutive_loss_cooldown_sec: 4h → 2h
  - 等等

但**阈值松绑有上限**：每个参数有 hard_max（4 象限计划已定义）
不能无限松绑导致风险失控。

阈值收紧（反向）也存在：
  如果某个 reason 的 missed_pumps < 2 且 correct_rejects > 30
  → 维持当前阈值（不主动收紧，避免过拟合）
```

### A.2.5 强制再复盘机制（你的核心需求）

**触发条件**：连续 7 天内：
- `missed_pumps >= 3` AND `actual_trades < 2` → 系统进入"反思模式"

**反思模式的具体动作**：

1. **暂停新开仓 24h**（除非 A 象限 + final_score >= 95）
2. **强制 LLM 复盘**：把过去 7 天的 missed_opportunities.jsonl 喂给 DeepSeek，要求生成报告：
   ```
   "你过去 7 天错过了 N 个妖币（详情：...），实际只开了 M 单。
    分析：(1) 哪些拒绝理由最常错？(2) 阈值是否过严？
    (3) 给出具体的参数调整建议（不超过 5 条）"
   ```
3. **报告写到 `.kiro/state/reflection_reports/YYYYMMDD.md`**
4. **Telegram 推送给操作员**：标题 "策略反思报告：过度保守"
5. **操作员人工确认后**才允许调整 production_rules.json
   （避免 LLM 幻觉直接改实盘参数）

### A.2.6 token 预算约束（衔接你之前的硬要求）

| 操作 | token 消耗 | 频率 |
|---|---|---|
| 每日 missed_opportunity 检测 | 0（纯规则）| 每天 |
| 拒单理由积分更新 | 0（纯规则）| 每天 |
| 阈值自适应 | 0（纯规则）| 每周 |
| **强制再复盘 LLM 报告** | ~3,000 tokens × 7 天 | 仅在反思模式触发时 |

每月最多 ~12,000 tokens（约 0.24% 月预算），**几乎免费**。

## A.3 实施步骤（Phase A）

| Step | 任务 | 工作量 | 验收 |
|---|---|---|---|
| A.1 | 写 `risk/miss_penalty_engine.py`，复用 audit_log 数据 | 1.5 天 | unit test：fake decisions.jsonl + fake K 线 → 正确判定 missed_pump |
| A.2 | 写 `risk/reject_reason_scorer.py`，维护积分表 | 0.5 天 | unit test：模拟 100 个 reject + 行情 → 积分正确 |
| A.3 | 写 `risk/threshold_auto_tuner.py`，按积分调阈值 | 1 天 | unit test：低分 reason → 阈值松绑 + hard_max 不超 |
| A.4 | 写反思模式：`risk/reflection_mode.py` | 1 天 | integration test：触发 → 暂停 + LLM 报告 |
| A.5 | 接到 main.py 的 daily/weekly cron | 0.5 天 | dry-run 跑 7 天看输出 |
| A.6 | Telegram 推送反思报告 | 0.5 天 | 收到测试消息 |

**总计 5 天**。

---

# Part B：生产级工程化清单

> 基于审计的 4 个维度，加上代码原文核对的客观结论。

## B.1 物理执行层（维度 1）

| 缺口 | 现状 | 致命性 | 工作量 | 修复方案 |
|---|---|---|---|---|
| **clientOrderId 幂等性** | **完全没有**（grep 0 命中）| 🔴 致命 | 1 天 | 每个 order 生成 UUID，传给 ccxt `params={"newClientOrderId": uuid}`，重试时复用同一 UUID，交易所自动去重 |
| **Market entry 重试** | 没有（只有 stop placement 有）| 🔴 致命 | 0.5 天 | 加 `place_entry_retries=2` + 指数退避，与 idempotency 联动 |
| **5xx / 502 / 504 区分** | 全部 raise | 🟠 严重 | 0.5 天 | 按 ccxt error class 分类：retriable (NetworkError/ExchangeNotAvailable) → 退避重试；non-retriable (InsufficientFunds/InvalidOrder) → 立即放弃 |
| **API ban 处理** | enableRateLimit 只防自己打爆 | 🟠 严重 | 0.5 天 | 监听 `-1003 Too many requests` → 进入 60s 冷静期，期间不发任何请求 |
| **Partial fill** | ✅ 已修（min_fill_ratio=0.95）| 已完成 | — | 见 P2 audit |
| **Partial fill at 边界** (0.94) | ✅ 用 `actual_size = filled` | 已完成 | — | 第三轮 audit #3 |

**关键代码定位**：
- `executor.py:90` market_order 调用 → 加 retry wrapper
- `ccxt_adapter.py:113` market_order 实现 → 接 newClientOrderId
- `executor.py:130` 已有 stop retries → 复用模式到 entry

## B.2 状态管理（维度 2）

| 缺口 | 现状 | 致命性 | 工作量 | 修复方案 |
|---|---|---|---|---|
| **重启恢复** | ✅ AccountPersistor + Reconciler 已有 | 已完成 | — | — |
| **持久化频率不够** | save() 只在 3 个特定点调用 | 🟠 严重 | 1 天 | 把 save() 改为事件驱动：每次 `set_cooldown` / `register_loss` / `daily_stoploss_hits++` 自动 save |
| **JSON 文件 vs WAL** | 当前是 JSON 单文件 | 🟡 中等 | 3-5 天 | 接 SQLite (single-file embedded WAL) 或 Redis (生产推荐)。SQLite 优先：零运维，原生 ACID |
| **WAL append-only 模式** | 没有 | 🟡 中等 | 1 天 | 在 SQLite 之上写 `INSERT INTO state_log (ts, key, value)` 而不是 UPDATE，崩溃恢复时回放 |
| **多进程安全** | 没有锁 | 🟢 低（单进程）| — | 仅当切多进程时才需要 |

**实施建议**：
- 阶段 1：把 save() 改事件驱动（1 天，立即得 80% 价值）
- 阶段 2：底层换 SQLite（3 天，完整 WAL 语义）

## B.3 延迟与性能（维度 3）

| 缺口 | 现状 | 致命性 | 工作量 | 修复方案 |
|---|---|---|---|---|
| **LLM 1-3 秒阻塞核心矛盾** | LLM 异步但实际不影响实盘决策 | 🔴 **项目最大伤口** | 5 天 | 实施 **LLM Pre-Rate**：对每个高分 candidate 提前 30s 触发 LLM 推理，verdict 缓存 5min。妖币爆发时 RiskGate 直接读缓存 0ms |
| **GIL / asyncio 抖动** | 单进程 asyncio | 🟠 严重 | 7+ 天 | 切多进程：WS I/O 一个进程 + 决策一个进程，shm 共享。**性价比低**，建议先 LLM Pre-Rate |
| **Event loop lag 监控** | 没有 | 🟢 低 | 0.5 天 | 加 `asyncio.events._get_event_loop_policy()._get_running_loop().time()` 周期采样 |

**LLM Pre-Rate 详细方案**：

```python
class LLMPreRater:
    """对所有 score >= 60 的 candidate 提前推理，结果缓存。"""
    
    cache: dict[str, AIVerdict]  # symbol -> verdict, TTL 5min
    
    async def maybe_pre_rate(self, candidate: SignalEvent):
        """背景任务，每 5min 跑一次：
        1. 取 score top 20 candidates
        2. 检查缓存：没有 / 过期 → 走 LLM
        3. 写回缓存
        token 消耗：top 20 × 12 次/小时 × 24h = 5760 次/天 → 极贵
        优化：仅对 quadrant A/B + score >= 70 → 削到 ~500 次/天
        """
    
    def get_cached_verdict(self, symbol: str) -> AIVerdict | None:
        """RiskGate 调用，0ms"""
```

token 预算重新核算：
- 当前 LLM 月预算：5M
- Pre-Rate 每天 ~500 次 × 1500 tokens = 750K/天 → 22.5M/月 ❌ **超出 4.5 倍**

**修正方案**：
1. Pre-Rate 仅限 A 象限 → ~50 次/天 → 75K/天 → 2.25M/月 ✅
2. 缓存 TTL 拉长到 15 分钟 → ~17 次/天 → 25K/天 → 750K/月 ✅
3. 同 symbol + 同 phase → 合并成一次（去重）→ 进一步压低

实施时再迭代。

## B.4 可观测性（维度 4）

| 缺口 | 现状 | 致命性 | 工作量 | 修复方案 |
|---|---|---|---|---|
| **Prometheus 指标 < 10 个** | 仅 5-6 个 gauge | 🟠 严重 | 2 天 | 加 30+ 指标：histogram (WS lag p50/p95/p99, order latency, LLM latency)、counter (rejects by reason, partial fills, retries) |
| **结构化日志** | logger.info 文本 | 🟡 中等 | 1 天 | 改 JSON 日志 + trace_id 串联整条决策链 |
| **OpenTelemetry trace** | 没有 | 🟡 中等 | 3 天 | 全链路：WS tick → screener → fuser → LLM → gate → executor，单个 trace 看清耗时分布 |
| **死信队列 (DLQ)** | 失败信号丢弃 | 🟡 中等 | 1 天 | 接到 .kiro/state/dlq/ JSONL，定期 LLM 复盘 |
| **Tick 级回测** | 没有（只有 1m K 线 post-mortem）| 🔴 致命（决定能否上实盘）| 10+ 天 | 见 B.5 |
| **代码穿透** | 实盘和回测不是同一套代码 | 🔴 致命 | 重构 IO 抽象层 5 天 | 见 B.5 |

## B.5 真正的回测引擎（最大单项工程）

> 这是**距离生产级最远的一块**，也是决定项目可否上实盘的硬门槛。

### 设计原则

```
✅ 实盘 daemon 和回测使用同一套 RiskGate / Sizer / TrailingFSM 代码
✅ 唯一差异在 IO 层：实盘 ccxt → 回测 HistoricalDataAdapter
✅ 回测的 PnL 必须用真实滑点公式（taker fee 0.04% + market impact）
✅ 同一信号在实盘和回测的 PnL 差异 ≤ 5%（验收标准）
```

### 关键模块

| 模块 | 作用 | 工作量 |
|---|---|---|
| `backtest/data_adapter.py` | 替代 ccxt：从 parquet 缓存读历史 K 线、orderbook、funding | 2 天 |
| `backtest/matching_engine.py` | 模拟撮合：market order → 用 1m K 线的 high/low 推断 fill price + 滑点 | 3 天 |
| `backtest/runner.py` | 串起 daemon + matching engine | 2 天 |
| `backtest/slippage_model.py` | 滑点模型：fill_price = mark_price × (1 + impact_pct + spread/2)，impact_pct 与 size/depth 成正比 | 2 天 |
| `backtest/walk_forward.py` | 滚动训练 + 验证 | 1 天 |

### 滑点模型校准（关键）

```python
def estimate_slippage(
    side: Side, size: float, top_depth_usdt: float, 
    realized_vol_pct: float
) -> float:
    """返回滑点百分比 (0.001 = 10bps)"""
    # 经验公式（需用历史成交数据校准）
    base_spread = 0.0005  # 5bps for top altcoins, 50bps for shitcoins
    market_impact = (size * mark_price / top_depth_usdt) ** 0.5 * 0.01
    vol_premium = realized_vol_pct * 0.02  # 高 vol 时滑点 ↑
    return base_spread + market_impact + vol_premium
```

**校准方法**：
1. 实盘运行 30 天积累 fill data
2. 把每笔实盘订单的 (size, depth, vol, actual_slippage) 写到 `.kiro/state/slippage_observations.jsonl`
3. 用 sklearn LinearRegression 拟合公式参数
4. 公式更新后回测的 PnL 与实盘的 PnL 偏差应缩到 ±2%

## B.6 实施步骤总览（Phase B）

```
Phase B.1: P0 致命缺口 (3 天)
  ├─ B.1.1: clientOrderId 幂等性
  ├─ B.1.2: market entry 重试 + 5xx 区分
  └─ B.1.3: 状态持久化事件驱动

Phase B.2: 监控 + 日志 (3 天)
  ├─ B.2.1: Prometheus 30+ 指标
  ├─ B.2.2: 结构化 JSON 日志 + trace_id
  └─ B.2.3: 死信队列

Phase B.3: 状态层升级 (3 天)
  └─ B.3.1: SQLite WAL 替代 JSON

Phase B.4: 回测引擎 (10 天) ← 最大块
  ├─ B.4.1: data_adapter             ✅ src/altcoin_agent/backtest/data_adapter.py
  ├─ B.4.2: matching_engine + slippage  ✅ matching_engine.py + slippage_model.py
  ├─ B.4.3: runner + walk_forward    ✅ runner.py (Phase 1-3) + walk_forward.py
  └─ B.4.4: 实盘/回测一致性验证      ✅ test_phase_4_e2e_mock.py：MatchingEngine 实现 ExchangeAdapter Protocol，e2e 跑 trainer 全链路

Phase B.5: LLM Pre-Rate (5 天)
  ├─ B.5.1: cache 实现                ✅ src/altcoin_agent/llm/cache.py (Phase 1-3 落地)
  ├─ B.5.2: 后台 pre-rate worker       ✅ src/altcoin_agent/llm/pre_rater.py — A 象限 + score>=70 队列模式，预算锁死自动丢弃
  └─ B.5.3: token budget 集成          ✅ ai_engine.LLMEngine 增加 cache + budget_manager 参数；命中 0 token，FREEZE 模式合成 neutral verdict

Phase B.6: OpenTelemetry trace (3 天)
  └─ B.6.1: 全链路 instrumentation

Phase B.7: 多进程切分 (7 天，可选)
  └─ 仅当 GIL 抖动确实成为瓶颈才做

总计：30-40 天（不含可选项）
```

---

# Part C：两份计划的执行顺序

```
                    ┌─────────────────────────────────┐
                    │ 当前位置（plan 分支）            │
                    └────────────────┬────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ Phase 0: 已完成（PR #23 P2 安全修复）          │
        └────────────────────────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ Phase B.1: P0 致命缺口（3 天）                 │ ← 必须最先做
        │   clientOrderId + retry + 持久化事件驱动        │
        └────────────────────────────────────────────────┘
                                     ▼
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
        ┌───────────────────────┐    ┌───────────────────────────┐
        │ Phase A:               │    │ Phase B.2 + B.3:           │
        │ 机会成本惩罚（5 天）   │    │ 监控 + SQLite WAL（6 天）  │
        │  并行执行              │    │  并行执行                  │
        └───────────────────────┘    └───────────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ QUADRANT_STRATEGY_PLAN Phase 1-3                │
        │ 框架 + 历史数据 + PumpPhaseFSM（约 7 天）       │
        └────────────────────────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ Phase B.4: 回测引擎（10 天）                    │ ← 最关键
        │ 所有上层功能都基于回测引擎验证                   │
        └────────────────────────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ Phase B.5 + QUADRANT Phase 4-5                  │
        │ LLM Pre-Rate + 训练系统 + 实盘接入（约 10 天）  │
        └────────────────────────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ 30 天 dry-run 验证                              │
        │ + Phase B.6 OpenTelemetry trace                 │
        └────────────────────────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ 操作员手动改 dry_run: false → 实盘启用          │
        └────────────────────────────────────────────────┘

            总计约 40-50 个工作日
```

---

# Part D：关键原则（贯穿所有 phase）

1. **Token 预算硬约束**：5M/月不变，Phase A 增加 ~12K，Phase B.5 LLM Pre-Rate 必须缩到 < 1M/月
2. **80% 把握度门槛**：QUADRANT_STRATEGY_PLAN 里的 production_rules 标准不放松
3. **机会成本惩罚 ≠ 鼓励冒进**：阈值松绑有 hard_max，松到极限仍守得住风控
4. **每个 phase commit + push**：不合并 main，新会话能从 git log 看出进度
5. **每个 phase 跑测试**：基线测试通过率 100%，不破坏既有 354 tests

---

# Part E：新会话开始的指引

> 如果你刚 checkout 到 `plan/quadrant-strategy-and-self-learning` 分支：

1. 读 `PLAN_README.md`（入口）
2. 读 `QUADRANT_STRATEGY_PLAN.md`（业务策略）
3. **读本文档**（机会惩罚 + 工程化）
4. 按 Part C 顺序执行
5. 每完成一个 phase：
   - 运行 `python3.12 -m pytest tests/ -q`（应 354+ passed）
   - commit + push
   - 在对应文档打勾

---

**文档版本**：v1.0  
**创建日期**：2026-05-16  
**分支**：`plan/quadrant-strategy-and-self-learning`  
**前置 PR**：#23（P2 安全修复，已 push 未合并）  
**关联文档**：[QUADRANT_STRATEGY_PLAN.md](./QUADRANT_STRATEGY_PLAN.md)
