# 🛡️ 生产上线 checklist

> 这份文档是给操作员看的，按顺序执行。每一步做完就在自己的笔记本上打勾，跑下一步。

---

## 第 ① 阶段：上线前（一次性，约 2-3 小时）

### A. 环境检查

- [ ] 云服务器选**香港 / 新加坡 / 东京**（避免国内 IP 被币安风控）
- [ ] 服务器配置 ≥ **2核 4G**，磁盘 ≥ **40G**（30 天日志 + SQLite WAL）
- [ ] Docker Desktop 或 Linux Docker 已安装并运行：`docker info` 没报错
- [ ] 本仓库 `feat/finalize-r1-r6` 分支已 clone 到 `~/altcoin-agent`
- [ ] 跑一次 `./deploy.sh`（或 `deploy.ps1`），看到 "Deploy successful"

### B. Key 安全配置

- [ ] LLM key 已在厂家后台**设置月度消费上限**（推荐 50 元）
- [ ] 币安 API Key 用的是 **Futures Testnet**，不是主网
  - https://testnet.binancefuture.com/
- [ ] Testnet API Key **限制 IP 白名单**为你云服务器的固定 IP
- [ ] Telegram Bot Token + Chat ID 已填（可选，但强烈推荐手机推送）
- [ ] **`.env` 文件已加进 `.gitignore`**（仓库默认已配，确认下别 commit）
- [ ] 至少有一份 `.env` 备份在另一台机器上（云服务器挂了能恢复）

### C. 历史数据预热（Phase 4 训练前置）

```bash
# 在云服务器上 cron 之前手动跑一次确认
python scripts/fetch_history.py \
    --symbols PEPE/USDT:USDT,WIF/USDT:USDT,TRUMP/USDT:USDT \
    --timeframe 1m --days 90 \
    --with-funding --with-oi \
    --max-rate 10 --capacity 20
```

- [ ] 看到日志 `bars written: ~129600` (90天×1440分钟×3币)
- [ ] `.kiro/state/backtest_cache/binance/PEPE_USDT_USDT/1m/2024/12.json` 等文件已生成
- [ ] funding 数据：`.kiro/state/backtest_cache/binance/PEPE_USDT_USDT/funding/...` 不为空
- [ ] OI 数据：`.kiro/state/backtest_cache/binance/PEPE_USDT_USDT/openInterest/...` 不为空

### D. 第一次训练（产出 production_rules.json）

```bash
# 假设你已经用 backtest_30d 或 matching engine 攒了 obs.jsonl
python scripts/run_walkforward_trainer.py \
    --observations obs.jsonl \
    --state-dir .kiro/state/training \
    --train-days 30 --validate-days 30 --step-days 30
```

- [ ] 退出码 0
- [ ] `.kiro/state/training/production_rules.json` 存在
- [ ] `.kiro/state/training/last_training_run.json` 存在并能读
- [ ] 看 `last_training_run.json` 里 `rules_promoted_now` ≥ 1（如果 = 0 说明数据不够，再攒几周）

### E. 单元测试基线（每次部署都要跑）

```bash
docker compose run --rm altcoin-agent python -m pytest tests/ -q
```

- [ ] `816 passed` 或更多，**0 failed**
- [ ] 没看到 `error` 关键字（warning 可以忽略）

---

## 第 ② 阶段：启动 dry-run（30 天观察期开始）

### F. 配置 `app.yaml` 安全档（推荐第一周用这套）

复制粘贴到 `config/app.yaml`：

```yaml
# 模式
dry_run: true                    # 第一周必须 true
paper_trade: false
live_confirm: ""                 # 留空就是 dry-run

# 观察期最关键开关
miss_penalty_enabled: false      # 第一周关，等数据攒够再开
production_rules_enabled: false  # 第一次训练完后开
llm_cache_enabled: false         # 第二周再开（第一周看真实 cache 命中率）
llm_budget_manager_enabled: false
llm_pre_rate_enabled: false      # A 象限 + score>=70 才用，第一周关

# 安全保险（强烈建议保持开）
kill_switch_enabled: true
account_persistence_enabled: true
decision_audit_log_enabled: true
metrics_enabled: true
structured_logging_enabled: true
dlq_enabled: true

# 第一周观察的 symbols（少一点稳一点）
symbols:
  - PEPE/USDT:USDT
  - WIF/USDT:USDT
```

- [ ] `app.yaml` 已配置如上
- [ ] `docker compose up -d --build`（重启）
- [ ] `docker compose logs -f` 看到 "Altcoin Agent V1.0 starting (mode=dry_run)"
- [ ] `curl http://localhost:8080/healthz` 返回 200
- [ ] `curl http://localhost:8080/metrics | grep altcoin` 能看到 ~30 个指标

### G. 第一天的人工验证（部署后 1 小时内）

- [ ] 看 `docker compose logs --tail=200`，没有 `ERROR` 级日志
- [ ] dashboard 打开 `http://localhost:8080/dashboard` 输入 token 能看到面板
- [ ] 至少有 1 个 SignalEvent 被处理（见 `screener_alive: true`）
- [ ] Telegram 收到了 "agent started" 推送（如果配了）

---

## 第 ③ 阶段：30 天观察期（每天打卡）

### H. 每日检查（建议固定时间，比如晚上 9 点）

把下面这张表存 Excel，**每天填一行**：

| 日期 | uptime_hr | signals_total | high_prio | dry_run_orders | rejections | miss_pen_count | llm_tokens | errors | 备注 |
|---|---|---|---|---|---|---|---|---|---|

每天跑这一行命令拿数据：

```bash
curl -s http://localhost:8080/api/state | python -m json.tool > today_$(date +%Y%m%d).json
curl -s http://localhost:8080/metrics | grep -E "(llm_tokens|miss_penalty|high_priority|reject)" \
    > metrics_$(date +%Y%m%d).txt
```

### I. 观察期里程碑（按周开关）

| 第几天 | 该做什么 | 怎么做 |
|---|---|---|
| Day 1-7 | 只观察，**不改任何 flag** | 攒原始数据 |
| Day 8 | 把 `llm_cache_enabled: true` 打开 | 编辑 `app.yaml` → `docker compose restart` |
| Day 8-14 | 看 cache 命中率 | `curl /metrics \| grep llm_cache_hit_ratio` 应 ≥ 0.5 |
| Day 15 | 把 `production_rules_enabled: true` 打开（如果第一阶段 D 训练成功） | 同上 |
| Day 15-21 | 看 R6 phase + confidence 是否正在工作 | dashboard 的 risk_decisions 应包含 `phase_not_allowed` / `confidence_below_floor` 这类 reason |
| Day 22 | 把 `miss_penalty_enabled: true` 打开 | 同上，需要至少 7 天 audit log |
| Day 22-28 | 看 reject_reason_scorer 输出 | dashboard 的 `reject_reasons` 应有分数 |
| Day 29 | 跑第二次训练 + 看效果 | `python scripts/run_walkforward_trainer.py ...` |
| Day 30 | **决策：是否切实盘** | 见下面"切实盘 checklist" |

### J. 红线（任何一个出现立刻停）

如果 30 天里**任意一天**出现以下情况，**立刻 `docker compose down` 暂停并复盘**：

- [ ] `last_error` 持续 > 1 小时不消失
- [ ] LLM token 月消耗已经超过 3M
- [ ] `daily_drawdown_pct` 超过 6%（哪怕是模拟账户）
- [ ] reflection_mode 触发了第 2 次（第 1 次正常，连续触发说明策略婆妈）
- [ ] kill_switch 自己触发了（除了你手动碰文件）
- [ ] Telegram 推送 ERROR 频率 > 3 条/小时

---

## 第 ④ 阶段：切实盘（30 天观察期之后，**绝不提前**）

### K. 上实盘前最后一道闸

- [ ] 30 天 dry-run 全部完成，没有红线
- [ ] 至少 **2 次** `run_walkforward_trainer.py` 跑出 `production_rules`
- [ ] miss_penalty 的反思模式从未 **连续两次** 触发
- [ ] 你已经验证手动按 kill switch 能 1 秒内停掉所有交易：
  ```bash
  touch .kiro/state/HALT
  # 看 dashboard 是否立刻显示 halted
  rm .kiro/state/HALT
  ```
- [ ] 你已经准备好**首笔实盘只用 100 USDT**（不是全仓）

### L. 切实盘的步骤（**操作员手动**）

```bash
# 1. 备份当前状态（万一要回滚）
cp -r .kiro/state .kiro/state.backup.$(date +%Y%m%d)

# 2. 切配置
nano .env
# 把 BINANCE_TESTNET=true 改成 BINANCE_TESTNET=false
# 重新填入主网 API Key + Secret（**只勾"现货+合约交易",不勾提币**）

nano config/app.yaml
# dry_run: false
# paper_trade: false
# live_confirm: "I_UNDERSTAND"   <-- 必须这一行
# initial_equity_usdt: 100.0      <-- 第一周只用 100 USDT

# 3. 重启
docker compose down
docker compose up -d --build

# 4. 看日志确认进入 live 模式
docker compose logs -f | grep -i "live\|mode="
# 应看到 "live trading mode enabled"
```

- [ ] 第一笔实盘下单后，10 分钟内手机查看 Binance APP 确认订单真实存在
- [ ] 第一周每天上线 dashboard 至少 3 次
- [ ] 每天结束截图 dashboard 留档

---

## 🚨 应急停止流程（出问题随时用）

```bash
# 方案 A：紧急停所有交易（保留持仓不平）
touch .kiro/state/HALT
# 解除：rm .kiro/state/HALT

# 方案 B：完全停 daemon（不平仓但停止新单）
docker compose down

# 方案 C：核武器（**只在交易所被攻击时用**）
# 1. 立刻去 binance 后台 revoke API key
# 2. docker compose down
# 3. 把所有持仓人工平掉
```

---

## 📞 关键文件路径速查

| 想看什么 | 看哪 |
|---|---|
| 实时日志 | `docker compose logs -f` |
| 健康状态 | `curl http://localhost:8080/healthz` |
| Prometheus 指标 | `curl http://localhost:8080/metrics` |
| 决策审计 | `logs/decisions.jsonl` |
| 死信队列 | `.kiro/state/dlq/main.jsonl` |
| 账户快照 | `.kiro/state/account.json` 或 `account.sqlite3` |
| 已学规则 | `.kiro/steering/dynamic_rules.json` |
| 训练产物 | `.kiro/state/training/production_rules.json` |
| miss penalty 报告 | `.kiro/state/reflection_reports/*.md` |

---

## ✅ 完成标志

如果你能在最后画上这些勾，恭喜你跑完了一个完整的训练-观察-上线周期：

- [ ] 30 天 dry-run 完成，没碰任何红线
- [ ] production_rules.json 至少更新过 2 次
- [ ] 实盘第一周只用 100 USDT 跑完，赔不超过 10 USDT
- [ ] 操作员（你）每天看 dashboard 至少一次
- [ ] 至少备份过一次 `.kiro/state/`

到这里**整个项目从代码到运维都闭环**了。
