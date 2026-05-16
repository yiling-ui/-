# 📋 工作计划入口（New Session Entry Point）

**这个分支不是要合并的代码分支，是一份工作计划。**

如果你是新对话刚 checkout 到这个分支：

## 第 1 步：读完整计划
```
.kiro/plan/QUADRANT_STRATEGY_PLAN.md
```

## 第 2 步：理解上下文
- 这个仓库是一个 altcoin 妖币交易 agent
- 之前的修复：P2 安全修复（PR #23，已 push 未合并）
- 操作员决定：放弃硬指标 5000% 收益，改为按妖币画像差异化打法
- 核心要求：**80% 训练把握度才开单 + 严控 LLM token 预算（5M/月）**

## 第 3 步：执行计划
按 `QUADRANT_STRATEGY_PLAN.md` 第七节"实施计划"的 5 个 Phase 顺序执行：

1. Phase 1: 框架搭建（1-2 天）
2. Phase 2: 历史数据回填（3-4 天）
3. Phase 3: 相位识别 + 回测引擎（5-7 天）
4. Phase 4: 训练系统（8-12 天）
5. Phase 5: 接入实盘 + Token 预算（13-15 天）

## 第 4 步：保持原则
- ❌ 不要合并到 main
- ❌ 不要无限制调 LLM
- ✅ 每个 Phase 完成后 commit + push
- ✅ 训练 token 消耗必须 < 200K，实盘信号 < 3M/月
- ✅ 80% win_rate + 30 samples + 3 个月验证 才能晋升 production rule

## 关键文件速查

| 文件 | 作用 |
|---|---|
| `.kiro/plan/QUADRANT_STRATEGY_PLAN.md` | 完整计划（读这个）|
| `examples/sim_pump_cycle.py` | 之前的简化模拟（参考）|
| `tests/test_safety_hardening_mock.py` | P2 修复的测试（基线）|
| `src/altcoin_agent/` | 现有代码 |

---

**当前阶段**：计划已制定，等待执行。  
**下一步**：从 Phase 1 开始。
