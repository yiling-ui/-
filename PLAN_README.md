# 📋 工作计划入口（New Session Entry Point）

**这个分支不是要合并的代码分支，是一份工作计划。**

如果你是新对话刚 checkout 到这个分支：

## 第 1 步：读完整计划（按顺序）
```
.kiro/plan/QUADRANT_STRATEGY_PLAN.md          ← 业务策略（四象限 + 自学习）
.kiro/plan/MISS_PENALTY_AND_PRODUCTION_PLAN.md ← 机会惩罚 + 工程化升级
```

## 第 2 步：理解上下文
- 这个仓库是一个 altcoin 妖币交易 agent
- 之前的修复：P2 安全修复（PR #23，已 push 未合并）
- 操作员决定：
  1. 放弃硬指标 5000% 收益，改为按妖币画像差异化打法
  2. **加入机会成本惩罚**：错过一个 +200% 妖币 = -3 分（命中的 3 倍）
  3. **强制反思模式**：连续 7 天错 3 个机会但只开 < 2 单 → 暂停 + LLM 复盘
  4. **生产级工程化**：clientOrderId 幂等性、SQLite WAL、Prometheus 30+ 指标、真 tick 级回测引擎
- 核心要求：**80% 训练把握度才开单 + 严控 LLM token 预算（5M/月）**

## 第 3 步：执行计划

按 `MISS_PENALTY_AND_PRODUCTION_PLAN.md` Part C 的执行顺序图：

```
Phase 0: 已完成（PR #23）
  ▼
Phase B.1: P0 致命缺口（3 天）— 必须最先做
  clientOrderId 幂等性 + market entry retry + 持久化事件驱动
  ▼
Phase A (5 天) ⊕ Phase B.2+B.3 (6 天)  — 并行执行
  机会惩罚引擎     ⊕  监控 + SQLite WAL
  ▼
QUADRANT_STRATEGY_PLAN Phase 1-3（7 天）
  框架 + 历史数据 + PumpPhaseFSM
  ▼
Phase B.4: 回测引擎（10 天）— 最关键
  ▼
Phase B.5 + QUADRANT Phase 4-5（10 天）
  LLM Pre-Rate + 训练系统 + 实盘接入
  ▼
30 天 dry-run + Phase B.6 OpenTelemetry
  ▼
操作员手动改 dry_run: false → 实盘启用
```

**总计约 40-50 个工作日。**

## 第 4 步：保持原则
- ❌ 不要合并到 main
- ❌ 不要无限制调 LLM
- ❌ 不要让策略婆婆妈妈不开单（机会惩罚机制会发现并触发反思）
- ✅ 每个 Phase 完成后 commit + push
- ✅ 训练 token 消耗必须 < 200K，实盘信号 < 3M/月，Pre-Rate < 1M/月
- ✅ 80% win_rate + 30 samples + 3 个月验证 才能晋升 production rule
- ✅ 错过一个 +200% 妖币 = -3 分（命中的 3 倍），积累到一定程度自动松绑阈值
- ✅ 实盘和回测必须用同一套代码（IO 层抽象差异）
- ✅ clientOrderId 幂等性是 P0，不做不能上实盘

## 关键文件速查

| 文件 | 作用 |
|---|---|
| `.kiro/plan/QUADRANT_STRATEGY_PLAN.md` | 业务策略（四象限 + 自学习） |
| `.kiro/plan/MISS_PENALTY_AND_PRODUCTION_PLAN.md` | 机会惩罚 + 生产级工程化 |
| `examples/sim_pump_cycle.py` | 之前的简化模拟（参考） |
| `tests/test_safety_hardening_mock.py` | P2 修复的测试（基线 354 个）|
| `src/altcoin_agent/` | 现有代码 |

---

**当前阶段**：计划已制定，等待执行。  
**下一步**：从 Phase 1 开始。
