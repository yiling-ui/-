# 四象限差异化策略 + 自学习引擎 — 工作计划

> **本文档是新会话的入口点。** 如果你刚 checkout 到 `plan/quadrant-strategy-and-self-learning` 分支，从这里开始读。
>
> 目标：把 altcoin agent 从 "硬指标追求 5000% 收益" 改造成 "**按妖币画像差异化打法 + 持续自学习 + token 预算严控**"。
>
> 上下文（前置对话已完成）：
> 1. P2 audit (#13~#18) 已修：audit log rotation / RegimeFilterConfig 校验 / cluster key 校验 / kill switch 硬化（PR #23）
> 2. 模拟运行已做：500U 在妖币周期上 +42%，但因为没接 DeepSeek + 广场社交 + learning，失真严重
> 3. 操作员明确放弃硬指标 5000%，改为"按币画像分级、风险分级、复利"

---

## 一、最终目标（不可变）

| 维度 | 目标 |
|---|---|
| **每个妖币周期收益** | 高质量妖币 **+200~600%**（A 象限），普通拉盘 **+50~150%**（B/C 象限），垃圾币 **0%（不开仓）** |
| **风险约束** | 单日最大回撤 ≤ 12%（A 象限）/ 8%（B）/ 6%（C/D）|
| **训练把握度门槛** | **置信度 ≥ 80% 才允许开单**，否则降级为观察者模式（dry-run 记日志，不下单）|
| **Token 预算** | DeepSeek 月度上限 ≤ 5M tokens，**单次推理优先用缓存/规则，仅在必要时才走 LLM** |
| **复利路径** | 一个月内捕捉 2-3 个妖币周期，账户预期 **月度 +50~150%**，回撤可控 |

---

## 二、整体架构（最终态）

```
                    ┌──────────────────────────────────────┐
                    │  Binance / OKX / Gate ccxt.pro WS    │
                    └──────────────────┬───────────────────┘
                                       ▼
              ┌────────────────────────────────────────────┐
              │ Screener (existing) — 7 detectors          │
              │   volume / funding / OI / sweep / wash...  │
              └──────────────────┬─────────────────────────┘
                                 ▼
       ┌──────────────────────────────────────────────────────┐
       │ ★ NEW: SymbolProfile (risk/symbol_profile.py)        │
       │   每个 symbol 一份画像，6h 重算一次：                │
       │   • quadrant: A | B | C | D                          │
       │   • social_score (历史 + 当前)                       │
       │   • liquidity_score                                  │
       │   • historical_win_rate                              │
       │   • last_pump_ts, scam_score                         │
       │   • confidence_threshold (0.0..1.0)                  │
       │   存档：.kiro/state/symbol_profiles.json              │
       └──────────────────┬───────────────────────────────────┘
                          ▼
       ┌──────────────────────────────────────────────────────┐
       │ ★ NEW: PumpPhaseFSM (risk/pump_phase.py)             │
       │   实时识别 7 个相位（accum/ramp/parabolic/blowoff_top │
       │   /crash/bleed/dead），每根 K 推进一次               │
       │   每个相位转换触发不同动作：                         │
       │     RAMP→PARABOLIC: 触发 rolling 加仓                 │
       │     PARABOLIC→BLOWOFF: trailing 收紧到 ±2% ATR        │
       │     BLOWOFF→CRASH: 自动反手 SHORT (若象限允许)        │
       └──────────────────┬───────────────────────────────────┘
                          ▼
       ┌──────────────────────────────────────────────────────┐
       │ ★ NEW: ConfidenceGate (risk/confidence_gate.py)      │
       │   综合 (rule_score, llm_score, profile, phase, learned)│
       │   → confidence ∈ [0.0, 1.0]                          │
       │   < 0.80 → 不开单（observer mode 仅记日志）           │
       │   ≥ 0.80 → 进入 RiskGate / PositionSizer              │
       └──────────────────┬───────────────────────────────────┘
                          ▼
              ┌─────────────────────────────────────────────┐
              │ Existing: RiskGate + PositionSizer + Executor│
              │   被改为接收 (profile, phase, confidence)    │
              │   按象限给不同 risk_pct / leverage / atr_mult │
              └─────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────┐
       │ ★ NEW: Backtester + Trainer (后台进程)                │
       │   • backtest/historical_loader.py                    │
       │     - 历史数据来源：ccxt fetch_ohlcv 1m/5m/1d         │
       │     - 时间范围：过去 3 年                             │
       │     - 缓存到 .kiro/state/backtest_cache/              │
       │   • backtest/runner.py                               │
       │     - 离线驱动 Screener + PumpPhaseFSM + Fuser + Gate │
       │     - 不调 LLM，用 mock_llm + 缓存的历史 verdict       │
       │   • training/trainer.py                              │
       │     - 滚动 walk-forward：每月窗口训练 → 下月验证       │
       │     - 输出：dynamic_rules.json 的更新                 │
       │     - 收敛指标：win_rate, sharpe, max_dd              │
       │     - 训练程度判定：连续 3 个月 win_rate ≥ 0.80 才生效 │
       │     存档：.kiro/state/training/                       │
       └──────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────┐
       │ ★ NEW: TokenBudgetManager (llm/token_budget.py)      │
       │   月度预算：5,000,000 tokens                          │
       │   每次 LLM 调用前检查：                                │
       │   • 缓存命中？（同 symbol + 同 phase + 同 social_hash） │
       │     → skip LLM，复用 verdict（节约 100%）             │
       │   • 高 confidence 规则命中？                           │
       │     → skip LLM，用规则的 verdict                      │
       │   • 月度预算剩余 < 20%？                               │
       │     → 仅对 A 象限调用 LLM，BCD 全部用规则             │
       │   • 月度预算剩余 < 5%？                                │
       │     → 完全停 LLM，纯规则模式                           │
       │   存档：.kiro/state/token_usage.json                  │
       └──────────────────────────────────────────────────────┘
```

---

## 三、四象限策略矩阵（核心）

按 (社交热度, 量能/流动性) 分四象限，每个象限不同打法：

| 维度 | **A 高质量妖币** | **B 抱团妖币** | **C 庄拉妖币** | **D 砸盘币** |
|---:|---|---|---|---|
| 识别条件 | social≥70 AND liq≥70 | social≥70 AND liq<70 | social<70 AND liq≥70 | social<70 AND liq<70 |
| `max_risk_per_trade` | **2.5%** | 1.5% | 1.0% | 0.5% / skip |
| `max_leverage_long` | 15x | 10x | 8x | 5x |
| `max_leverage_short` | 10x | 8x | 5x | 5x |
| `rolling_enabled` | ✅ | ✅ | ❌ | ❌ |
| `rolling_max_legs` | 4 | 2 | 1 | 0 |
| `trailing atr_mult` | 2.5 | 1.5 | 1.0 | 0.5 |
| `anti_chase_max_move` | 6% | 4% | 2.5% | 2.5% |
| `breakeven_at_r` | 1.5 | 1.0 | 0.7 | 0.5 |
| `daily_drawdown_limit` | 12% | 8% | 6% | 6% |
| **顶部反手 SHORT** | ✅ + 重仓 | ✅ + 中仓 | ❌（庄控盘风险）| ✅ + 小仓 |
| **置信度门槛** | 0.80 | 0.85 | 0.85 | 0.90（D 极少开单）|
| **LLM 调用频次** | 高频 | 中频 | 中频 | **低频（仅历史规则）** |

---

## 四、PumpPhaseFSM 相位识别规则（可直接编码）

```
ACCUMULATION
  • 30 天 vol z-score < 1.0
  • 价格在 ±5% 区间震荡
  • 动作：观察，不开仓

RAMP
  • vol z-score >= 3.0
  • 价格 24h 涨幅 30% ~ 200%
  • 动作：开 LONG（如果象限允许）

PARABOLIC
  • vol z-score >= 6.0 AND 6h 涨幅 >= 100%
  • 动作：rolling 加仓（A/B 象限）；trailing 提前收到 1.5×ATR

BLOWOFF_TOP
  • 1d K 出现 upper_wick > 2 × body
  • close 回落到 1d 中位以下
  • 动作：trailing 立刻收到 ±2% ATR；准备反手信号

CRASH
  • 1h 跌幅 > 30% OR 5m 出现 lower_wick > 5%
  • 动作：多仓全平 + 自动 SHORT（A/B/D 象限允许）

BLEED
  • 日 vol 衰减
  • 价格阶梯式下跌
  • 动作：trailing 收紧；不开新仓

DEAD
  • 14 天 vol 持续低位
  • 动作：标记 symbol 进入冷却
```

每根 1m kline 推进一次状态机，状态变化触发回调到 RiskGate / TrailingFSM。

---

## 五、训练系统（关键）

### 5.1 训练数据来源

| 数据类型 | 来源 | 时间跨度 | 频率 |
|---|---|---|---|
| K 线 (1m/5m/1d) | ccxt `fetch_ohlcv`（binance/okx）| 过去 **3 年** | 一次性下载 + 缓存 |
| Funding rate | ccxt `fetch_funding_rate_history` | 过去 3 年 | 一次性下载 |
| OI 历史 | binance/okx REST | 过去 1 年（OI 历史 API 限制）| 一次性下载 |
| 社交快照 | 现在开始抓取 + 存档 | 从今天开始 | 每 6h 一次 |
| 历史 PnL | 现有 `decisions.jsonl` | 系统运行后 | 实时 |

**估算总数据量**：
- ccxt 单 symbol 1m × 3 年 = 1,576,800 条；100 个 symbol = 1.5 亿条 K 线
- 单 K 线 ~50 字节 → 7.5 GB（gzip 后约 2 GB）
- **缓存到本地，不重复下载**

### 5.2 训练循环（walk-forward）

```
第 1 个月：训练（learn rules）
第 2 个月：验证（dry-run，不下单，记录预测 vs 实际）
第 3 个月：滚动 → 第 1 个月的训练结果 + 第 2 个月的实际表现 → 重新训练

反复迭代直到：
  连续 3 个月 win_rate ≥ 0.80 AND sharpe ≥ 1.5
  → 输出 production_rules.json
  → 允许该规则在实盘开单（confidence ≥ 0.80）
```

### 5.3 训练程度判定（80% 把握度的具体含义）

每条学到的规则附带：
```json
{
  "rule_id": "social_post_count_high_kol_intent_call_in_RAMP_phase",
  "samples": 47,
  "wins": 38,
  "losses": 9,
  "win_rate": 0.808,
  "avg_pnl_pct": 0.18,
  "sharpe": 1.62,
  "first_observed": "2024-08-15",
  "last_observed": "2026-05-10",
  "validation_months_passed": 3,
  "confidence": 0.85,
  "production_ready": true
}
```

只有 `production_ready: true` 的规则才进 `production_rules.json`。
其他规则进 `candidate_rules.json`（仅观察，不影响开单）。

**80% 门槛的硬约束**：
- `samples >= 30`（统计显著性）
- `win_rate >= 0.80`
- `validation_months_passed >= 3`
- `sharpe >= 1.5`

任何一条不满足 → `production_ready: false` → 不允许影响实盘决策。

### 5.4 持续训练（线上运行后）

每天凌晨 03:00 UTC 跑一次：
1. 拉取昨天的所有 audit log
2. 把昨天的实际 PnL 反馈到 candidate_rules
3. 重算每条规则的 win_rate / sharpe
4. 满足晋升条件的 → 进 production_rules.json
5. 连续 30 天 win_rate < 0.50 的 production rule → 降级回 candidate

输出每日训练报告到 `.kiro/state/training/daily_report_YYYYMMDD.md`。

---

## 六、Token 预算严控（你的硬要求）

### 6.1 月度预算

| 用途 | 月度上限 | 占比 |
|---|---|---|
| 实盘信号 LLM 推理 | 3,000,000 | 60% |
| 训练用历史复盘 | 1,500,000 | 30% |
| 紧急/调试 | 500,000 | 10% |
| **合计** | **5,000,000 tokens/月** | 100% |

### 6.2 节约措施（按节省比例排序）

| 措施 | 节省比例 | 实现位置 |
|---|---|---|
| **缓存命中**（同 symbol + 同 phase + 同 social_hash 12h 内）| 60-80% | `llm/cache.py`（新）|
| **规则优先**（高 confidence rule 命中 → skip LLM）| 30-50% | `fuser.py` 现有 learned_rules |
| **批处理**（多个 symbol 合并成一个 prompt 推理）| 40% | `ai_engine.py` 改 |
| **prompt 压缩**（移除冗余字段，用 token-efficient JSON schema）| 20% | 现有 `_USER_PROMPT_TPL` 重写 |
| **降级策略**（预算剩余 < 20% → 仅 A 象限走 LLM）| 动态 | `TokenBudgetManager` |
| **历史复盘用便宜模型**（deepseek-chat → 训练用 deepseek-reasoner-cheap）| 50% | `llm_provider.py` 加 model 参数 |
| **离线训练复用历史 verdict 缓存**（训练时不重新调 LLM）| 95%（训练阶段）| `backtest/runner.py` |

### 6.3 TokenBudgetManager 行为

```python
class TokenBudgetManager:
    def can_call_llm(self, symbol: str, profile: SymbolProfile) -> bool:
        used_pct = self.month_used / self.month_budget
        
        if used_pct < 0.50:
            return True  # 自由模式
        elif used_pct < 0.80:
            # 节约模式：仅 A/B 象限
            return profile.quadrant in ("A", "B")
        elif used_pct < 0.95:
            # 紧急模式：仅 A 象限 + 高分信号
            return profile.quadrant == "A" and signal.rule_score >= 70
        else:
            # 冻结模式：完全停 LLM
            return False
    
    def estimate_tokens(self, prompt: str) -> int:
        # 字符数 / 3.5（中英混合的经验比）
        return int(len(prompt) / 3.5)
    
    def record_usage(self, tokens: int) -> None:
        self.month_used += tokens
        self.persist()
        if self.month_used > self.month_budget * 0.8:
            logger.warning("Token budget at %.0f%%, switching to economy mode",
                          used_pct * 100)
```

### 6.4 训练阶段的特殊处理

训练时**绝对不调实时 LLM**。改为：
1. 第一次回测：对每个历史信号，**用规则算出 mock verdict**（基于该信号当时的 funding/OI/social 等观测值，按预设规则映射到 confidence 0-1）
2. 训练验证：跑完一遍后，挑出 **关键转折点**（高 PnL 的信号 + 大亏的信号），仅这些点走真实 LLM 推理（约 200-500 个/月）
3. 把 LLM 真实 verdict 作为 ground truth 回填到 mock_verdict_cache，下次训练直接用

预估训练阶段月度消耗：**< 100,000 tokens**（远低于预算）。

---

## 七、实施计划（5 个阶段）

### Phase 1: 框架搭建（第 1-2 天）

- [x] 创建模块目录骨架
  - `src/altcoin_agent/risk/symbol_profile.py`
  - `src/altcoin_agent/risk/pump_phase.py`
  - `src/altcoin_agent/risk/confidence_gate.py`
  - `src/altcoin_agent/llm/token_budget.py`
  - `src/altcoin_agent/llm/cache.py`
  - `src/altcoin_agent/backtest/historical_loader.py`
  - `src/altcoin_agent/backtest/runner.py`
  - `src/altcoin_agent/training/trainer.py`
  - `src/altcoin_agent/training/rules_promoter.py`
- [x] 写 `dataclass` 骨架（SymbolProfile / PumpPhase enum / ConfidenceVerdict / TokenBudgetState）
- [x] 写最小可运行的 mock 实现（returns hardcoded values）
- [x] 单元测试：每个 dataclass 序列化 + 反序列化

**验收**：`pytest tests/` 全部通过，新模块 import 不报错。**已通过 (614 passed, 92 new tests).**

### Phase 2: 历史数据回填（第 3-4 天）

- [x] 实现 `historical_loader.py`
  - ccxt fetch_ohlcv 拉 100 个 symbol 过去 3 年的 1m K 线
  - 限速：每分钟 < 1200 次请求（binance 上限）— 通过 `inter_call_sleep_sec` 节流
  - 增量缓存：已下载的不重复（`force=False` 跳过已有月份）
  - 输出：`.kiro/state/backtest_cache/{exchange}/{symbol}/{tf}/{YYYY}/{MM}.json`（v1 用 JSON，避免 pyarrow 依赖）
- [ ] 实现 `funding_history_loader.py` — Phase 4 再做
- [ ] 写一个 CLI：`python -m altcoin_agent.backtest.historical_loader --symbols PEPE,WIF,TRUMP --years 3` — Phase 4 再做

**验收**：能下载 + 加载 PEPE 过去 3 年的 1m K 线（约 150 万根）。**Phase 1-3 内只验证了 mock fetcher + 月份分片缓存；100-symbol × 3y 的真实拉取留给操作员触发。**

### Phase 3: 相位识别 + 回测引擎（第 5-7 天）

- [x] 实现 `pump_phase.py` 状态机
- [ ] 写 `phase_tagger.py` CLI：输入 symbol + 时间范围，输出每根 K 的 phase 标签 — Phase 4 再做
- [ ] 用历史数据验证：手动选 3 个已知妖币（PEPE 2024.5、WIF 2024.3、TRUMP 2025.1），对比代码输出的相位序列与人工判断的相位 — Phase 4 验证
- [x] 实现 `backtest/runner.py`：离线驱动 PumpPhaseFSM 输出每根 K 的 phase 标签（Screener + Fuser + Gate 的串接留给 Phase B.4 完整回测引擎）
- [ ] 输出：每个 symbol 的 backtest report（trades, PnL curve, max_dd, sharpe）— Phase B.4

**验收**：在 PEPE 2024.5 那次妖币行情上，回测出 +200~600% 收益（A 象限预期）。**Phase 1-3 仅验证了合成 pump cycle 的相位识别正确（RAMP→PARABOLIC→BLOWOFF→CRASH 全部命中）；真实数据回测留给 Phase B.4 + Phase 4。**

### Phase 4: 训练系统（第 8-12 天）

- [x] 实现 `training/trainer.py`：walk-forward 训练 — `walkforward_trainer.py` 落地（`trainer.py` 保留为 daily-cycle wrapper）
- [x] 实现 `training/rules_promoter.py`：80% 门槛判定 + 晋升 / 降级 — Phase 1-3 已落地
- [x] 新增 `training/rule_miner.py`：从 (features, pnl) 观察派生 LearnedRule（按 quadrant/phase/score 桶分组）
- [x] e2e 集成（B.4.4）：MatchingEngine → TradeObservation → WalkforwardTrainer → RulesPromoter，全链路 token=0
- [ ] 跑过去 3 年的全量训练 — 需要操作员触发 `historical_loader.download` 拉真实 K 线，留给 Phase 5
- [ ] 输出 `production_rules.json` + 每月训练报告 — Phase 5（实盘 dry-run 阶段）
- [ ] 实现持续训练 cron job（每日 03:00 UTC）— Phase 5（与 main.py 接入一起做）

**验收**：
- 训练完成后，`production_rules.json` 至少包含 10 条规则 — 留给 Phase 5 真实数据跑完
- 每条规则的 `samples >= 30, win_rate >= 0.80, validation_months_passed >= 3` — 已通过 unit + e2e mock test 验证机制正确
- 训练总 token 消耗 < 200,000 — 当前实现 = **0 tokens**（纯规则桶分，e2e test 锁死）

### Phase 5: 接入实盘 + Token 预算（第 13-15 天）

- [x] 实现 `TokenBudgetManager`，挂到 `ai_engine.py` 调用前 — `LLMEngine.judge` 增加可选 `budget_manager` 参数；模式分级 + quadrant + score 联合 gating
- [x] 实现 `llm/cache.py`，挂到 `llm_provider.py` 内部 — `LLMEngine.judge` 增加可选 `cache` + `phase` 参数；命中 0 token，未命中写回
- [x] 实现 `llm/pre_rater.py` 后台 worker — A 象限 + score>=70 才入队，最大队列 64，预算锁死后丢弃
- [x] `risk/quadrant_factory.py`：把 `SymbolProfile` 的 quadrant 翻译成 `PositionSizer + RiskGateConfig + TrailingStopFSM`（不动 `sizing.py` / `gate.py` / `trailing.py` 本体，零回归风险）
- [x] `main.py` 接入：`AppConfig` 加 10 个 cfg 字段（默认全 OFF）+ App 加 3 个 slot + 启动/关闭生命周期 + 11 个 integration test
- [ ] 修改 `ai_engine.py`：批处理多 symbol 合并 prompt — 推迟（命中率 60-80% 后批处理边际收益小）
- [ ] 修改 `risk/gate.py`：接收 `pump_phase` + `confidence`，< 0.80 直接拒 — 推迟到 Phase 5.1，等 main.py 实际跑起来再判断是否需要
- [ ] 修改 `risk/trailing.py`：接收 `pump_phase`，BLOWOFF_TOP 时立即收紧 — 同上
- [ ] 修改 `fuser.py`：每个象限不同 high_priority 阈值 — 同上
- [ ] dry-run 30 天验证

**验收**：
- dry-run 30 天，预测准确率 ≥ 80%
- Token 消耗 < 4,000,000（预算 80%）
- 模拟收益曲线显示月度 +50~150%，单日回撤 ≤ 12%

---

## 八、关键文件 / 路径速查

```
.kiro/
├── plan/
│   └── QUADRANT_STRATEGY_PLAN.md     ← 本文件，新会话入口
├── state/
│   ├── backtest_cache/{symbol}/      ← 历史 K 线缓存（gitignore）
│   ├── social_history/{symbol}/      ← 社交快照（gitignore）
│   ├── symbol_profiles.json          ← 每个 symbol 的画像（gitignore）
│   ├── token_usage.json              ← 月度 token 用量
│   ├── production_rules.json         ← 通过 80% 门槛的规则
│   ├── candidate_rules.json          ← 仍在训练中的规则
│   └── training/
│       ├── daily_report_YYYYMMDD.md
│       └── walkforward_report.md
└── steering/
    └── dynamic_rules.md              ← 现有，由 production_rules.json 自动生成
```

```
src/altcoin_agent/
├── risk/
│   ├── symbol_profile.py     ← 新
│   ├── pump_phase.py         ← 新
│   ├── confidence_gate.py    ← 新
│   ├── sizing.py             ← 改：接收 SymbolProfile
│   ├── gate.py               ← 改：接收 pump_phase + confidence
│   └── trailing.py           ← 改：接收 pump_phase
├── llm/
│   ├── token_budget.py       ← 新
│   └── cache.py              ← 新
├── backtest/
│   ├── historical_loader.py  ← 新
│   └── runner.py             ← 新
├── training/
│   ├── trainer.py            ← 新
│   └── rules_promoter.py     ← 新
├── fuser.py                  ← 改：象限阈值
└── ai_engine.py              ← 改：批处理 + token budget 挂钩
```

---

## 九、新会话开始的指引

> 如果你刚 checkout 到 `plan/quadrant-strategy-and-self-learning` 分支，按下面流程接手：

1. **读本文档全文**（本文件）
2. **读上一个 PR 修复的内容**（`git log --oneline -10` 看最近改动）
3. **运行现有测试确认基线没坏**：
   ```bash
   python3.12 -m pytest tests/ -q
   # 应该 354 passed
   ```
4. **从 Phase 1 开始**，逐项实现并打勾
5. **每完成一个 Phase**：
   - 跑测试
   - commit + push（不要合并到 main）
   - 在本文件的对应 checkbox 上打勾
6. **Token 预算永远是硬约束**：每写一处 LLM 调用前先想"能不能用缓存 / 规则代替"

---

## 十、关键决策点（执行时遇到再选）

下列决策**不要在写代码前预判**，等遇到时根据数据决定：

1. **历史数据 100 个 symbol 选哪些？** — 等 Phase 2 时按"过去 3 年成交额 top 100 永续合约"自动筛
2. **PumpPhaseFSM 的阈值（vol z-score=3, 涨幅=30% 等）该用多少？** — Phase 3 在 PEPE/WIF/TRUMP 三个样本上反向调
3. **80% 门槛是否过严？** — Phase 4 跑完后看 production_rules 数量，若 < 5 条则放宽到 75%（一次性允许）
4. **Token 预算 5M 够不够？** — Phase 5 dry-run 后看实际消耗，超额时优先砍训练用（30%）

---

## 十一、风险声明

本计划是**纯 dry-run 训练 + 离线学习**，不涉及实盘资金调动。所有改动都在分支上，不合并 main。

实盘启用必须满足：
- [ ] 训练完成且 production_rules ≥ 5 条
- [ ] 30 天 dry-run win_rate ≥ 0.80
- [ ] 操作员手动改 `app.yaml` 把 `dry_run: true` 改成 `false`

---

**文档版本**：v1.0  
**创建日期**：2026-05-16  
**分支**：`plan/quadrant-strategy-and-self-learning`  
**前置 PR**：#23（P2 安全修复，已 push 未合并）
