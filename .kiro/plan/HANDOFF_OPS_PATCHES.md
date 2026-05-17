# 🔄 交接备忘 — feat/operational-patches

> 当前对话已经把第一轮 PR #34 的 4 项原始要求 + 后续 review 找出的 3 个 blocker + 用户新增的"反手安全机制"全部完成。

---

## 📍 当前状态

**分支**: `feat/operational-patches`（已 push 到 GitHub）
**基线分支**: `plan-and-tasks-START-HERE`（已合并）
**测试**: **978 passed** ✅（baseline 929 + PR #34 一轮新增 29 + 本次 review 后新增 20）
**已 push 到 main**: 否，等操作员决定

---

## ✅ 全部已完成

### Phase 1 — 大体量保护 (操作员要求 ①)
- `RiskGateConfig.max_notional_vs_depth_pct` 闸门 #10b
- 应用到 entry + rolling 加仓两条路径
- **review 后修订**：改为 **side-aware**：long 只看 ask depth，short 只看 bid depth。
  legacy float depth 调用回退到 `halved_sum` (减半启发)，由 `RiskGateConfig.depth_cap_side_aware` 控制
- `CCXTExchangeAdapter.fetch_top_depth_by_side()` 新增；旧的 `fetch_top_depth_usdt` 内部调用新的，行为完全 backward-compat
- `_handle_high_priority_impl` 在 fetch summed depth 之后再尽力获取 by-side（best-effort，失败不阻塞下单）
- **+8 测试**（depth cap 4 + side-aware helper 2 + 之前的整体 4，共计 4 新增 + 4 旧）

### Phase 2 — 提现自动识别 (操作员要求 ②)
- 新模块 `risk/withdrawal_detector.py`
- `AccountState.adjust_equity_baseline()` —— 等比缩放 starting_equity，不污染 daily_drawdown
- `CCXTExchangeAdapter.fetch_total_usdt_balance()` 支持 binance / gate / 通用 ccxt 回退
- **review 后修订**：detector 新增 `_last_rollover_date_utc` 跟踪，每次 poll 比对 `account.last_rollover_date_utc`，跨 UTC 日时只重置自己的快照、不触发 phantom flow event。新增 `rollover_resync` action
- **+13 测试**（11 原有 + 2 新增：rollover 不再 phantom，rollover 同时遇到真实 withdrawal 仍能识别）

### Phase 3 — 可视化 + TG 双向命令 (操作员要求 ③)
- Dashboard PnL Chart.js 曲线 + `/api/pnl-curve` + `/api/external-flows`
- `DashboardState.equity_curve` 1440 点环形缓冲（~24h）
- 双向 TG 命令：`/status` `/equity` `/positions` `/pnl` `/halt` `/resume` `/help`
- **review 后修订**：
  - `AccountState.resume()` 镜像 `halt()` 走 `_notify_change` —— `/resume` 现在跨重启对称持久化
  - `record_equity_snapshot` 的 `import time` 提到模块级
  - `_parse_int_list` 容错：`telegram_commands_allowed_chat_ids: [123, abc]` 不再炸 boot
  - `Notifier.flow()` 新通道：deposits/withdrawals 用 🏦 BANK 图标，不再 🚨 ERROR
- **+4 测试**（2 dashboard 原有 + 2 新增 resume 行为）+ **+10 telegram_commands 原有**

### Phase 4 — 多交易所主用 binance + gate, 其他预留 (操作员要求 ④)
- ✅ 配置层完成：`exchanges: [binance]` 默认；`gateio/okx/bybit/bitget/kucoin/mexc/bingx` 通过 ccxt.pro 即开即用
- ✅ 代码层完成：`ccxt_adapter.py` 已有 venue-specific 分支
- ⚠️ 多交易所**并行**未做（操作员明确不需要）：当前 `cfg.exchanges[0]` 单交易所。若未来要并行，需要 `App._adapter` 改成 `dict[str, ExchangeAdapter]`、`Reconciler/Executor` 加 venue_name 路由 —— 工时 3-4 天

### Phase 5 — 反手安全机制 ReversalGuard (操作员要求 ⑤，新增)
- 新模块 `risk/reversal_guard.py`
- 三个动作：`approve` / `defer_close_and_watch` / `veto`
- 默认 **flat-and-watch**：持仓被反向信号触发时，先平仓观望、不立刻反手
- 反手必须满足（"合适时机"）：
  1. 不在 cooldown
  2. 距上次 close ≥ `min_seconds_since_close` (默认 30s)
  3. 最近 wick window 的 Parkinson range < `wick_threshold_pct` (默认 4% over 60s) —— 即没插针
  4. 新信号 final_score ≥ `min_reversal_final_score` (默认 7.5)
- 任一失败 → veto 并设 cooldown，避免 thrashing
- 主路径：`_handle_high_priority_impl` 在 gate.evaluate 之前调用，veto/defer 时 `_reverse_guard_flatten` 用 `reduce_only` 市价单平掉旧仓 + 取消 stop + 设 cooldown
- 收尾：`_on_position_close` 调 `note_close()` 让 guard 知道刚刚 close 过
- **+7 测试**（disabled / no-flip / open-flip-defers / wick-veto / too-soon-defers / weak-score-veto / clean-approve / cooldown-short-circuit，共 8）

---

## 📂 文件改动清单（合并版，整个分支）

```
src/altcoin_agent/risk/gate.py            +depth-aware notional cap (闸 #10b) + side-aware helper
src/altcoin_agent/risk/state.py           +adjust_equity_baseline() + resume()
src/altcoin_agent/risk/withdrawal_detector.py     NEW + rollover-aware
src/altcoin_agent/risk/ccxt_adapter.py    +fetch_total_usdt_balance() + fetch_top_depth_by_side()
src/altcoin_agent/risk/reversal_guard.py  NEW (post-review, ~310 lines)
src/altcoin_agent/dashboard.py            +PnL curve + flows API + Chart.js HTML + import time
src/altcoin_agent/notifier/telegram.py    +Notifier.flow() (Protocol/Null/Telegram)
src/altcoin_agent/notifier/telegram_commands.py   NEW (401 lines)
src/altcoin_agent/main.py                 +AppConfig fields (depth/withdrawal/TG/reversal)
                                          + _parse_int_list helper (YAML tolerance)
                                          + _build_telegram_command_handlers
                                          + _reverse_guard_flatten
                                          + _fetch_top_depth_by_side
                                          + dry-run set_top_depth_by_side
                                          + ReversalGuard wiring + note_close on close
                                          + flow() routing for bank events
config/app.yaml                           +primary/reserved venues docs
                                          +depth/withdrawal/TG/reversal sections

tests/test_risk_mock.py                   +4 tests (depth cap, original)
tests/test_withdrawal_detector_mock.py    NEW (11 tests)
tests/test_dashboard_mock.py              +4 tests (PnL curve + flows)
tests/test_telegram_commands_mock.py      NEW (10 tests)
tests/test_post_review_patches_mock.py    NEW (20 tests, post-review)
```

---

## 🟡 仍未做（不阻塞合并 / 上线）

1. **main.py wiring 集成测试**——3 个新 worker (equity_snapshot / withdrawal_detector / telegram_command) 的 end-to-end startup/shutdown 测试。所有内部行为各自单测覆盖，但端到端验证 + ReversalGuard wiring 的整链路覆盖度更稳
2. **Chart.js CDN 本地化**——dashboard 当前从 jsdelivr CDN 加载 chart.js；air-gapped 主机会泄漏 DNS。建议把 `chart.umd.min.js` (~133KB) 放进 `src/altcoin_agent/static/` 由 aiohttp 自己 serve
3. **多交易所并行**——见 Phase 4 末尾。操作员已明确不需要

---

## 🚀 接手 / 推送命令

```bash
git fetch origin
git checkout feat/operational-patches
python3.12 -m pytest tests/ -q                 # 期望 978 passed
git push origin feat/operational-patches
# PR #34 已存在；这次 push 会把 2 个新 commit 加上去
```

---

**文档版本**：v2.0（review-后定稿）
**最后 commit (本批)**：见 `git log --oneline -5`
