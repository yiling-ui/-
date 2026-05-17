# 🚀 一键部署 + 配置说明

> 部署本身见仓库根的 `deploy.sh` / `deploy.ps1`（Linux/Mac 用 sh，Windows 用 ps1）。  
> 本文档讲**部署完成之后怎么按时间线打开各种 flag**。

---

## ⚙️ Phase 5 / R3 / R6 配置 flag 一览

> 所有 flag **默认 OFF**（除了 4 个安全保险）。这是**有意的**：第一次部署的人不应该被一堆开关淹没。

### 🟢 默认开（不用动）

| flag in `app.yaml` | 干嘛的 | 为啥默认开 |
|---|---|---|
| `kill_switch_enabled` | 摸 `.kiro/state/HALT` 文件就停所有交易 | 安全闸 |
| `account_persistence_enabled` | 重启不丢账户状态 | 数据安全 |
| `decision_audit_log_enabled` | 每个决策写一行 JSON 到 `logs/decisions.jsonl` | 合规 + 复盘 |
| `regime_filter_enabled` | BTC 跌得快阻止 LONG / 涨太猛阻止 SHORT | 防被市场 beta 砸 |

### 🟡 第一周关，**等真实数据**再开

| flag | 干嘛的 | 何时开 | 怎么验证 |
|---|---|---|---|
| `llm_cache_enabled` | LLM 同 (symbol, phase) 12h 内复用 verdict | Day 8 | `/metrics` 看 `llm_cache_hit_ratio` ≥ 0.5 |
| `metrics_enabled` | Prometheus 30+ 指标 | Day 1（推荐立即开） | `curl /metrics \| grep altcoin` |
| `structured_logging_enabled` | 日志变 JSON + trace_id | Day 1（推荐立即开） | `docker compose logs \| head -1` 是 JSON |
| `dlq_enabled` | 死信队列：失败决策落盘 | Day 1（推荐立即开） | `.kiro/state/dlq/main.jsonl` 不为空 |

### 🟠 等训练 / 数据攒够 1-2 周再开

| flag | 干嘛的 | 何时开 | 前置条件 |
|---|---|---|---|
| `production_rules_enabled` | 周期性 reload `production_rules.json` 给 fuser/gate 用 (R3) | Day 15 | 跑过 1 次 `run_walkforward_trainer.py` |
| `miss_penalty_enabled` | 错过 +200% 妖币 = -3 分 + reflection 模式 | Day 22 | 至少 7 天 `decisions.jsonl` |
| `cluster_cap_enabled` | 防 PEPE+WIF+FLOKI 同时 SHORT | Day 8 | 先在 `app.yaml` 填 `cluster_map: { PEPE: meme, WIF: meme, ... }` |

### 🔵 R6 — phase + confidence 路由

| flag / 配置 | 干嘛的 | 推荐设置 |
|---|---|---|
| `RiskGateConfig.allowed_entry_phases` | 限制只在某些 phase 进新单 | 推荐：`{accumulation, ramp, parabolic}`（拒绝 CRASH/BLEED/DEAD） |
| `RiskGateConfig.phase_min_confidence` | 每个 phase 单独一个 confidence 门槛 | 例：`{ramp: 0.70, parabolic: 0.85, blowoff_top: 0.95}` |
| `FuserConfig.phase_threshold_overrides` | 每个 phase 不同的 high_priority_threshold | 例：`{accumulation: 80, ramp: 85, parabolic: 92, blowoff_top: 95}` |
| `TrailingStopFSM.blowoff_atr_tighten` | BLOWOFF_TOP 时把 ATR 倍数 ×0.5（更紧 trail） | 默认 0.5 即可 |
| `TrailingStopFSM.crash_atr_tighten` | CRASH/BLEED/DEAD 时把 ATR 倍数 ×0.3 | 默认 0.3 即可 |

> R6 的所有参数都是**可选传入**，不传就完全走 v1.0 行为。第一周不要管它们；第二周开始按需调。

### 🔴 高级：token 优化（用了能省 30-60% LLM 费用）

| flag | 干嘛的 | 何时开 | 注意 |
|---|---|---|---|
| `llm_budget_manager_enabled` | quadrant + 评分分级预算（FREE/ECONOMY/EMERGENCY/FREEZE） | 跟 cache 一起开 | 单开它而不开 cache 反而费 token |
| `llm_pre_rate_enabled` | 后台 worker 提前给 A 象限 + score≥70 的高潜力币打分 | Day 15+ | 默认 `llm_pre_rate_min_score=70` 别调低 |

### ⚫ 实盘最后一闸（**决不能默认开**）

| flag | 含义 | 操作员手动改 |
|---|---|---|
| `dry_run` | true=不下单，false=真下单 | Day 30 后改 false |
| `live_confirm` | 必须等于 `"I_UNDERSTAND"` 才能切实盘 | 切实盘前手填 |
| `paper_trade` | 真 ccxt 调用但 testnet | dry_run=false 时另一种选择 |

---

## 📊 按时间线最简化的 `app.yaml` 示例

### 第 1 周（最保守）

```yaml
dry_run: true
metrics_enabled: true
structured_logging_enabled: true
dlq_enabled: true
kill_switch_enabled: true
account_persistence_enabled: true
decision_audit_log_enabled: true
regime_filter_enabled: true
```

### 第 2 周末（加 cache + 训练 reload）

```yaml
# 上面所有 + 下面新增
llm_cache_enabled: true
production_rules_enabled: true
production_rules_dir: ".kiro/state/training"
```

### 第 3 周末（加机会惩罚）

```yaml
# 上面所有 + 下面新增
miss_penalty_enabled: true
miss_penalty_state_dir: ".kiro/state/miss_penalty"
```

### 第 4 周末（加预算管理 + pre-rate + R6 phase 路由）

```yaml
# 上面所有 + 下面新增
llm_budget_manager_enabled: true
llm_pre_rate_enabled: true
llm_pre_rate_min_score: 70.0
# R6 phase 路由示例
risk:
  allowed_entry_phases: ["accumulation", "ramp", "parabolic"]
  phase_min_confidence:
    ramp: 0.70
    parabolic: 0.85
    blowoff_top: 0.95
fuser:
  phase_threshold_overrides:
    accumulation: 80
    ramp: 85
    parabolic: 92
    blowoff_top: 95
```

### 第 30 天（切实盘）

```yaml
dry_run: false
paper_trade: false
live_confirm: "I_UNDERSTAND"
initial_equity_usdt: 100.0   # 第一周只用 100 USDT
```

---

## 🛠️ 工具脚本一览

| 脚本 | 干嘛 | 调用例 |
|---|---|---|
| `scripts/fetch_history.py` | 拉历史 OHLCV / funding / OI（R1 + R2） | `python scripts/fetch_history.py --symbols PEPE/USDT:USDT --days 90 --with-funding --with-oi` |
| `scripts/run_walkforward_trainer.py` | 训练 → 写 production_rules.json（R3） | `python scripts/run_walkforward_trainer.py --observations obs.jsonl --state-dir .kiro/state/training` |
| `scripts/discover_events.py` | 自动找历史 pump/dump events | 见 PLAN_README |
| `scripts/backtest_30d.py` | 跑 post-mortem 攒 dynamic_rules | 见 PLAN_README |

---

## 📖 完整 30 天上线流程

详见 [`docs/PRODUCTION_CHECKLIST.md`](docs/PRODUCTION_CHECKLIST.md)。
