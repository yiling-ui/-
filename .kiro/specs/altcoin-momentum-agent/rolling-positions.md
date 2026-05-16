# Rolling Positions (滚仓) — 设计文档

**Status**: PR-A merged — 数据模型 + 控制器 + 测试已实现
**Author**: agent
**Last updated**: 2026-05-16
**Targets**: V1.1（Bug #1/2/3 合并后的下一个里程碑）

---

## 0. 一句话总结

把**已经盈利仓位的浮盈**作为新仓位的保证金，在同一标的同一方向上**逐级加仓**；
所有的加仓都**仍然走 Risk Gate**，止损共享同一条 trailing stop（永远只朝有利方向移动），
日内 DD / 3-strike / 全局 halt 仍然适用，**不会被滚仓机制绕过**。

操作员可以在**仪表盘开关**或**Telegram 命令** 实时打开/关闭/调参，
默认 **OFF**（保守），打开后任何一次失败都自动回退到 OFF。

---

## 1. 为什么需要滚仓 / 它解决什么问题

V1.0 的逻辑是 **"一次入场 → 一次离场"**。这对 妖币momentum 策略有两个明显短板：

1. **盈利没有杠杆化**。一个 +5R 的趋势行情和一个 +1R 的小行情，
   利润比例只是 5×，但保证金占用一样。资金效率低。
2. **TrailingStop 单条腿**。单条腿离场后浮盈被一次性锁定，
   下一个 setup 才能重新部署资金。错过了趋势第二波。

滚仓就是把"已经盈利的部分"再投入到同一方向。

> 注意：滚仓 ≠ 加大单次入场风险。每次"加仓腿"自身的最大亏损
> 仍然受 risk_per_trade(1.5%) 限制，但**风险来源是浮盈**而不是本金，
> 所以从账户净值视角看，"实际承担的本金风险"始终被钉在 1R = 1.5%。

---

## 2. 滚仓与现有风控框架的关系

V1.0 的 9 道 Risk Gate（state.py + gate.py）必须**全部继续生效**：

| Gate | 滚仓时是否仍生效 | 备注 |
|---|---|---|
| 1. global_trading_halted | ✅ 生效 | 操作员一键全停同时停滚仓 |
| 2. reconciliation_complete | ✅ 生效 | 启动期不滚 |
| 3. daily_drawdown_pct ≥ 6% | ✅ 生效 | 当日 DD 超限不滚（关键） |
| 4. daily_stoploss_hits ≥ 3 | ✅ 生效 | |
| 5. symbol cooldown | ⚠️ **改造** | 加仓不应被同 symbol cooldown 拒（见 §3.5） |
| 6. consecutive_losses | ✅ 生效 | |
| 7. max_concurrent_positions | ⚠️ **改造** | 加仓不算新 position；按 symbol 数量计 |
| 8. min_liquidity | ✅ 生效 | 加仓时实时再查一次 top5 depth |
| 9. SR-1 dynamic slippage | ✅ 生效 | 用 trigger=current_stop, current=mark |

**SR-2 hard stop pairing 也照常**：每一条加仓腿成交后，
立即更新（cancel+place）整个 position 的总 size 的单条 STOP_MARKET。

---

## 3. 设计

### 3.1 新增的可配置参数

加到 `AppConfig`：

```python
# 滚仓总开关。默认 False（保守）。可以在仪表盘 / Telegram 实时切换。
rolling_enabled: bool = False

# 在哪些 R 倍触发加仓。每次只触发一次（去重）。例：[1.5, 3.0, 5.0]
# 表示当浮盈达到 +1.5R 时第一次加仓，+3R 时第二次，+5R 时第三次。
rolling_trigger_r_levels: tuple[float, ...] = (1.5, 3.0, 5.0)

# 每次加仓使用浮盈的比例。例：0.5 表示用一半浮盈作为新仓的保证金。
rolling_unrealized_pnl_ratio: float = 0.5

# 加仓的初始止损距离（按当前价的百分比）。比 5% 更紧，因为基础腿
# 已经为整个 position 提供了保护。例：0.025 = 2.5%
rolling_leg_stop_pct: float = 0.025

# 单个 symbol 的总加仓腿数上限（防止无限加仓）。基础腿不计入。
rolling_max_legs_per_symbol: int = 3

# 加仓后必须至少多久后才能再次加仓（秒）。防止价格快速震荡反复触发。
rolling_min_interval_sec: int = 60

# 一旦发生任何加仓失败（下单失败 / 止损失败），自动关闭滚仓直到下次手工恢复。
rolling_auto_disable_on_failure: bool = True
```

YAML 配置位置：`config/app.yaml` 的 `rolling:` 段。

### 3.2 数据模型扩展

`Position` 不变（仍是单一聚合视图），但内部增加 `legs`：

```python
@dataclass
class PositionLeg:
    """A single executed entry within a rolling position."""
    leg_id: int                    # 0 = original, 1+ = rolled legs
    side: Side
    size: float                    # base units
    entry_price: float
    entry_ts_ms: int
    margin_source: str             # "initial" | "rolled_unrealized"

@dataclass
class Position:
    # ... 现有字段不动 ...
    legs: list[PositionLeg] = field(default_factory=list)

    @property
    def avg_entry_price(self) -> float:
        """Size-weighted average entry of all legs."""
        if not self.legs:
            return self.entry_price
        total_size = sum(L.size for L in self.legs)
        return sum(L.size * L.entry_price for L in self.legs) / total_size

    @property
    def total_size(self) -> float:
        return sum(L.size for L in self.legs) if self.legs else self.size
```

### 3.3 TrailingFSM 状态扩展

现有：`INIT → ARMED → BREAKEVEN → TRAILING → TARGET_REACHED`。

新增**侧向**事件（不改变主状态机），由 `RollingController` 监听：

```
ROLL_TRIGGER  emitted on each tick when:
    state in {BREAKEVEN, TRAILING}
  AND realized_r ≥ next un-fired threshold in cfg.rolling_trigger_r_levels
  AND now - last_roll_ts > rolling_min_interval_sec
  AND len(position.legs) - 1 < rolling_max_legs_per_symbol
  AND cfg.rolling_enabled  (live-toggleable)
```

**关键不变量**：
- 主状态机不感知 ROLL_TRIGGER。它继续按 BREAKEVEN/TRAILING 计算"下一个 stop 位置"。
- 加仓后，`Position.total_size` 增大，trailing 的 `tighten_hard_stop()` 自动用新 size 替换 STOP_MARKET。

### 3.4 新组件：`RollingController`

新文件：`src/altcoin_agent/risk/rolling.py`

```python
@dataclass
class RollingConfig:
    enabled: bool = False
    trigger_r_levels: tuple[float, ...] = (1.5, 3.0, 5.0)
    unrealized_pnl_ratio: float = 0.5
    leg_stop_pct: float = 0.025
    max_legs_per_symbol: int = 3
    min_interval_sec: int = 60
    auto_disable_on_failure: bool = True


@dataclass
class RollingController:
    cfg: RollingConfig
    sizer: PositionSizer
    gate: RiskGate
    executor: CCXTExecutor
    account: AccountState
    health: HealthState
    notifier: Notifier
    quote_provider: Callable[[str], Awaitable[float]]
    # per-symbol bookkeeping
    _last_roll_ts: dict[str, int] = field(default_factory=dict)
    _fired_levels: dict[str, set[float]] = field(default_factory=dict)

    async def maybe_roll(self, position: Position) -> bool:
        """Called from the trailing worker on every kline tick.

        Returns True if a leg was actually added.

        Fail-closed semantics: any exception in the gate, the sizing,
        or the executor.add_to_position path → the controller logs,
        notifies, and (if auto_disable_on_failure) flips ``cfg.enabled
        = False`` so subsequent ticks skip until ops re-enables.
        """
        ...

    def reset_for_symbol(self, symbol: str) -> None:
        """Called by App._on_position_close when the original position
        closes — clears _fired_levels and _last_roll_ts so a fresh
        position next time starts clean."""
        ...
```

### 3.5 RiskGate 改造

`RiskGate.evaluate(...)` 现有逻辑保持不变，但增加一个新方法 `evaluate_rolling()`：

```python
def evaluate_rolling(
    self,
    *,
    parent: Position,             # 原仓位
    proposed_size: float,         # 新加仓 size（已经按 §3.6 算好）
    proposed_stop: float,
    account: AccountState,
    current_price: float,
    top5_depth_usdt: float,
    realized_vol_pct: float,
    now_ms: int | None = None,
) -> RiskDecision:
    """Like evaluate(), but for an additional leg on an EXISTING position.

    Differences from evaluate():
      * Skips check #5 (symbol cooldown) — adding to a winning position
        is the opposite of "open after a loss".
      * Skips check #7 (max_concurrent_positions) — adding doesn't
        increase the symbol count.
      * Still enforces daily DD, 3-strike, halt, reconciliation,
        liquidity, slippage, leverage cap.
      * Sizing is bounded by both (a) the parent position's leverage
        and (b) ``account.equity_usdt - sum_existing_notionals``
        so total exposure ≤ equity * max_leverage.
"""
```

`max_concurrent_positions` 仍按"不同 symbol 个数"算（现在 V1.0 的实现已经是这样）。

### 3.6 Sizing 数学（具体公式）

设：
- `entry` = 原仓位的 avg_entry_price
- `mark` = 当前 mark price
- `r_unit` = abs(entry - initial_stop) ＝ 1R（USDT/unit）
- `total_size` = 原仓位所有腿的 size 之和
- `unrealized_pnl_usdt` = (mark - entry) * total_size （LONG；SHORT 相反）

新加仓腿的预算：

```python
# (a) 用浮盈作为风险预算
roll_risk_budget = unrealized_pnl_usdt * cfg.rolling_unrealized_pnl_ratio

# (b) 新腿的 stop_distance
new_stop_distance = mark * cfg.rolling_leg_stop_pct

# (c) risk-parity sizing（与 PositionSizer.compute_size 一致的公式）
new_leg_size = roll_risk_budget / new_stop_distance

# (d) 杠杆 / 流动性 / 总敞口约束
new_leg_notional = new_leg_size * mark
existing_notional = total_size * mark
max_total_notional = account.equity_usdt * leverage
new_leg_notional = min(
    new_leg_notional,
    max_total_notional - existing_notional,    # 总敞口不超账户*杠杆
)
new_leg_size = new_leg_notional / mark

# (e) 单腿最小 notional 约束（避免下单被交易所拒）
if new_leg_notional < sizer.min_notional_usdt:
    return None    # 跳过本次滚仓
```

### 3.7 Stop 管理（单 stop, 多 leg）

**核心不变量**：每个 symbol 在交易所只挂**一条 STOP_MARKET**。
不为加仓腿单独挂 stop。

加仓后：
1. trailing_fsm 下一个 tick 用新的 `total_size` 计算 stop 位置（位置不变；逻辑不变）。
2. `executor.tighten_hard_stop(position, new_stop)` 内部 cancel 旧 stop → place 新 stop（size = `total_size`）。
3. **如果 cancel/place 失败**，落入现有的"恢复旧 stop"分支（v1.0 已经实现）。
4. 加仓本身不会触发 stop 调整，**它只是放大 size**。trailing 自然会在下一个 tick 把 stop 收紧。

**额外保护**：加仓腿的 entry_price 已经远离 stop（因为 stop 还在原仓位的 breakeven 或更紧）。
即便加仓后立即反向，**最坏情况是新加仓腿的浮盈被吃掉**，原仓位仍在保本。

### 3.8 与 Bug #3 (daily rollover) 的交互

`maybe_roll_over_day()` 在跨 UTC 日时重置 `realized_pnl_today_usdt = 0`。
**这不影响**滚仓判定（滚仓只看 `unrealized_pnl_usdt`，是仓位级别的，不受 daily 重置影响）。

但跨日时如果当时仓位**仍是开仓状态**：
- `RollingController._last_roll_ts` / `_fired_levels` 不重置（这些是仓位生命周期级别的）。
- daily DD 重新归零，所以新一天可以继续滚仓。
- ✅ 正确行为。

### 3.9 与 Bug #1 (PositionWatcher) 的交互

当 STOP_MARKET 触发时，**所有腿一起平仓**（因为它们共享一条 stop）。
`_on_position_close` 已经会：
- 计算实现 PnL（用 `total_size`）
- 更新 `realized_pnl_today_usdt` / `daily_stoploss_hits` / `consecutive_losses`
- 调用 `RollingController.reset_for_symbol(symbol)`

所有腿被合并视作一笔交易。

---

## 4. 操作员控制

### 4.1 仪表盘新增滚仓面板

在 `dashboard.py` 现有 HTML 末尾加一个面板：

```
┌─ Rolling Positions (滚仓) ─────────────────────┐
│  [●] Enabled    [○] Disabled                 │
│                                                │
│  Trigger R levels:    [ 1.5, 3.0, 5.0 ]      │
│  Unrealized ratio:    [——●——————]  50%       │
│  Leg stop %:          [——●———]  2.5%          │
│  Max legs/symbol:     [ 3 ▼ ]                 │
│  Min interval (sec):  [ 60 ]                  │
│                                                │
│  [ Apply ]                                     │
│                                                │
│  Status: ENABLED  ·  3 legs added today        │
│  Last roll: BTC/USDT 14:32 +1.5R → +0.3 BTC   │
└────────────────────────────────────────────────┘
```

新 API：
- `POST /api/rolling/toggle  {enabled: bool}`
- `POST /api/rolling/config  {trigger_r_levels: [...], unrealized_pnl_ratio: 0.5, ...}`
- `GET  /api/rolling/status` → `{enabled, fired_levels: {...}, last_roll_ts, total_legs_today}`

**安全**：
- 仪表盘已被定位为"内部 only"。本次仍然不增加认证（保持现有 README 风险声明）。
- 但所有 POST 端点必须做基本的 input 校验（trigger levels ∈ [0.5, 10], ratio ∈ [0.1, 0.9]）。
- 任何配置改动 → 写入 `logs/rolling_audit.jsonl`（一行 JSON / 改动）。

### 4.2 Telegram 命令

`telegram.py` 现在只发出站消息，不接 webhook。我们**新增一个 long-poll worker** 监听 `/getUpdates`，
解析以下命令（仅来自 `TG_CHAT_ID` 配置的那个 chat，其他 chat 一律忽略）：

| 命令 | 行为 |
|---|---|
| `/roll status` | 返回当前 enabled、所有 trigger levels、最近 5 次加仓 |
| `/roll on` | enabled=True |
| `/roll off` | enabled=False |
| `/roll set ratio=0.5 leg_stop=0.025 max_legs=3` | 改配置 |
| `/roll levels 1.5 3.0 5.0` | 改 trigger_r_levels |
| `/roll auto_disable on/off` | 切换失败自动关闭 |

**实现**：在 `notifier/telegram.py` 增加 `class TelegramCommandLoop` async worker，
和 dashboard 共用同一个 `RollingController` 实例。所有命令通过它的 `apply_config()` /
`set_enabled()` 方法生效，所以仪表盘改和 TG 改是一份代码。

**鉴权**：只有 `TG_ADMIN_USER_IDS`（环境变量逗号分隔）里的 user 发的命令才生效。
其他人发命令直接静默丢弃 + 日志。

### 4.3 配置持久化

为了重启后保持操作员的最后选择：

- 配置改动写入 `logs/rolling_state.json`（path 来自 `cfg.rolling_state_path`）
- 启动时 `App.run()` 读取这个文件并应用到 `RollingController.cfg`
- 文件不存在 → 用 `AppConfig.rolling_*` 的默认值
- 文件读取失败 → 警告日志 + 继续用默认（fail-open，因为这是软配置）

---

## 5. 失败模式 (Failure Modes)

| 故障 | 现象 | 处理 |
|---|---|---|
| 加仓 market_order 被交易所拒（min_notional 等） | RiskDecision OK 但 executor.market_order 异常 | 记录失败 / TG 通知 / `auto_disable_on_failure → enabled=False` |
| 加仓成功但 stop 替换失败 | 新仓裸奔 | 现有恢复旧 stop 流程；如还失败 → 触发 emergency_close 整个 position |
| 加仓后立刻反向 | 新加仓腿浮亏 | trailing stop 已经收紧到 ≥ breakeven，整个 position 仍在保本 |
| 配置被恶意改成 `ratio=0.99`（过激） | 风险变高但仍受 leverage cap 限制 | 校验输入 + 改动需写 audit log |
| 跨 UTC 日，DD 在加仓的瞬间触发 | 新加仓腿入场后 DD>6% | 加仓腿瞬间被下一个 tick 的 risk_gate 不让加新；现有 stop 仍管平仓 |
| Telegram 命令被冒名 | 非 admin 发 `/roll on` | 静默丢弃 + WARN log |
| `rolling_state.json` 损坏 | 配置乱 | 读取失败 → fall back 到 `AppConfig.rolling_*` 默认 |

---

## 6. 测试计划

**Unit (新文件 `tests/test_rolling_mock.py`)**：

1. `test_rolling_disabled_does_nothing` — `cfg.enabled=False` → maybe_roll 永远 return False
2. `test_first_roll_at_1_5R` — 浮盈刚到 1.5R 时触发；之后 2R 不再触发（去重）
3. `test_size_uses_unrealized_pnl_ratio` — sizing 公式
4. `test_size_capped_by_total_leverage` — 加仓不会突破 equity*leverage
5. `test_min_interval_blocks_double_roll` — 60s 内只能加一次
6. `test_max_legs_caps` — 第 4 次（max_legs=3）不再触发
7. `test_daily_dd_breaker_blocks_roll` — DD>6% 时拒绝加仓
8. `test_failure_disables_when_auto_disable_on` — sizing 抛 → enabled 自动 False
9. `test_failure_keeps_enabled_when_auto_disable_off` — auto_disable=False 时保持开
10. `test_position_avg_entry_price_after_roll` — 多腿后 avg_entry 计算正确
11. `test_trailing_stop_uses_total_size_after_roll` — stop 替换用 total_size
12. `test_reset_for_symbol_clears_state` — 仓位关闭后 _fired_levels 清空

**Integration**：

- `test_bug_rolling_full_lifecycle` — 在 dry-run app 里：开仓 → 价格涨到 +1.5R →
  `RollingController.maybe_roll()` 被 trailing worker 调用 → 加仓成交 → adapter 同时只挂一条 stop
  → 价格继续涨到 +3R → 再加 → 价格回落 stop → 全部腿一起平仓 → PnL 正确
- `test_dashboard_toggle_endpoint` — POST `/api/rolling/toggle` → 配置生效
- `test_telegram_command_only_admin` — 非 admin chat 发 `/roll on` 被丢弃

**Mocked TG long-poll**：用 `respx` 模拟 `getUpdates`，验证命令解析。

---

## 7. 分阶段交付

我建议分 3 个 PR 推：

### PR-A: 数据模型 + 核心控制器（不接前端，纯后端）
- `Position.legs` + `avg_entry_price` / `total_size`
- `RollingController` + `RollingConfig`
- `RiskGate.evaluate_rolling`
- `PositionSizer.compute_roll_size`
- `CCXTExecutor.add_to_position`（核心：调用 market_order + 更新 stop size）
- 12 个 unit + 1 个 integration 测试
- **完全可用**：通过 `cfg.rolling_enabled=True` (env 或 yaml) 即可手工开启

### PR-B: 仪表盘 UI + REST 端点
- 新面板 + 3 个 API endpoint
- `rolling_state.json` 持久化
- input validation + audit log

### PR-C: Telegram 命令 + admin 鉴权
- `TelegramCommandLoop` long-poll worker
- 命令解析 + admin 校验
- TG 通知现在每次加仓发一张卡

每个 PR 通过的测试不少于 6 个，与现有 198 个保持向后兼容。

---

## 8. 不在本次范围内（可以后续做）

- **跨 symbol 滚仓**（盈利的 BTC 单 → 开 ETH 新仓）。复杂度太高，先做单 symbol 同向。
- **金字塔减仓**（trailing 缩小 size 而不是 stop）。和现有 trailing FSM 思路冲突，留作 V1.2。
- **滚仓 Telegram inline keyboard**（点按钮改配置）。先用文本命令，按钮 V1.2。
- **滚仓后单独 post-mortem**（每条腿独立学习）。复杂；先把腿合并到原仓位的 post-mortem 里。
- **持久化历史腿到 SQLite**（用于回测）。先靠 `recent_orders` ring buffer。

---

## 9. 估时

- PR-A：约 1.5 天（含测试）
- PR-B：约 0.5 天
- PR-C：约 0.5 天

总计 ~2.5 天，分 3 个 PR 推送，每个独立可合并。

---

## 10. 决策点 — 等你确认

请在以下选项里选定，我再开 PR-A：

1. **触发 R 阶梯**：默认 `(1.5, 3.0, 5.0)` 三档，可改成 `(1.0, 2.0, 4.0)` 更激进，
   或 `(2.0, 4.0)` 两档保守。建议从默认开始。

2. **每次浮盈用量**：默认 50%。建议保守起步用 30%。

3. **加仓腿止损宽度**：默认 2.5%（比基础腿 5% 紧）。建议保持。

4. **单 symbol 最大腿数**：默认 3 条加仓腿（合计最多 4 个 entry）。建议保持。

5. **Telegram 命令长 poll 间隔**：默认 30s。建议保持。

6. **`rolling_enabled` 默认值**：建议 **OFF**，操作员手动开启。

7. **是否开 PR-A 时**：你说"开始"我就动手；保持默认参数。

任何一项你想覆写，告诉我即可。
