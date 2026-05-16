"""reflection_mode.py — forced strategy review when the agent goes silent (Phase A.4).

Background
----------
The 4-quadrant strategy and the +1 / -3 reject scoring are not, on
their own, sufficient. A pathological failure mode is "the gate is
too tight, we miss every pump, but the operator never notices because
the daemon happily keeps logging rejections". The reflection mode
solves this with a hard-coded liveness check:

    Within the last ``window_sec`` (default 7 days):
        * ``missed_pumps >= miss_threshold`` (default 3)   AND
        * ``actual_trades_opened < trade_threshold`` (default 2)
    -> the strategy is over-conservative, force a review.

When a review is forced:

    1. Suspend new entries for ``suspension_sec`` (default 24h),
       except when an A-quadrant signal arrives with
       ``final_score >= a_quadrant_score_floor`` (95).
    2. Generate a markdown report aggregating the missed opportunities.
       Optionally include a short LLM-written analysis (callable
       injected at construction time so tests don't need a live LLM
       and the production token-budget can wrap the call).
    3. Persist the report to ``.kiro/state/reflection_reports/``.
    4. Notify the operator (Telegram callback injected at construction).
    5. Wait for the operator's manual ``acknowledge()`` before the
       suggested threshold tweaks are written into
       ``threshold_overrides.json`` -- the tuner from A.3 only emits
       proposals; reflection mode is the human-in-the-loop checkpoint.

State machine
-------------

::

    IDLE  ── trigger ────────────►  SUSPENDED + REPORT WRITTEN
      ▲                                  │
      │  acknowledge() OR                │
      │  suspension expires              │
      └──────────────────────────────────┘

We deliberately don't try to "restart" automatically when the next
weekly score table looks better. That leaves the operator in charge
of saying "we're done reflecting" -- no implicit auto-tuning back to
production traffic.

Concurrency
-----------
This module is purely synchronous on the read/decide path. The
``generate_report`` coroutine is the only async call (because it may
talk to an LLM). The full daemon loop will invoke ``maybe_trigger``
inside the daily-cron worker and only spin up the async report when
``maybe_trigger`` returns a non-None decision.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from altcoin_agent.risk.miss_penalty_engine import MissedOpportunity
from altcoin_agent.risk.reject_reason_scorer import RejectReasonScore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------- #


# (prompt) -> assistant-message text. Async so production can wrap
# token-budget / rate-limit / retry. Tests inject a synchronous lambda
# wrapped in ``async def`` -- see :func:`stub_llm_caller`.
LLMCaller = Callable[[str], Awaitable[str]]

# Notify callback: ``payload`` is a small dict the Telegram notifier
# already understands (see notifier/telegram.py). Async; never raises.
Notifier = Callable[[dict], Awaitable[None]]


@dataclass
class ReflectionConfig:
    """All trigger / suspension knobs in one place."""

    # Trigger window.
    window_sec: int = 7 * 24 * 3600
    miss_threshold: int = 3
    trade_threshold: int = 2

    # Suspension behaviour after trigger.
    suspension_sec: int = 24 * 3600
    a_quadrant_bypass_score: float = 95.0   # final_score floor that beats suspension

    # Persistence.
    state_path: str = ".kiro/state/miss_penalty/reflection_state.json"
    reports_dir: str = ".kiro/state/reflection_reports"

    # Cooldown between reflection reports so we don't spam the operator
    # if multiple consecutive cron runs see the same trigger conditions.
    min_seconds_between_reports: int = 24 * 3600


@dataclass
class ReflectionState:
    """What the daemon needs to remember across restarts."""

    suspended_until_ts_ms: int = 0
    last_report_ts_ms: int = 0
    last_report_path: str = ""
    last_trigger_summary: str = ""
    acknowledged: bool = True   # True == nothing pending
    pending_report_id: str = "" # "" == no report awaiting ack

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TriggerDecision:
    """Result of :meth:`ReflectionModeController.maybe_trigger`."""

    triggered: bool
    reason: str
    missed_pumps_in_window: int
    trades_in_window: int
    window_sec: int


@dataclass
class ReflectionReport:
    """A single review report, on disk + in memory."""

    report_id: str
    generated_at_ts_ms: int
    window_start_ts_ms: int
    window_end_ts_ms: int
    missed_pumps_in_window: int
    trades_in_window: int
    top_missed_opportunities: list[dict]
    bucket_breakdown: dict[str, int]
    suggested_threshold_changes: list[dict]
    llm_analysis: str
    markdown_path: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------- #


class ReflectionModeController:
    """Glues missed opportunities + trade count + LLM + notifier."""

    def __init__(
        self,
        *,
        config: ReflectionConfig | None = None,
        llm_caller: LLMCaller | None = None,
        notifier: Notifier | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.cfg = config or ReflectionConfig()
        self._llm = llm_caller
        self._notifier = notifier
        self._clock = clock or time.time
        self._state: ReflectionState | None = None  # lazily loaded

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_suspended(self, *, now_ms: int | None = None) -> bool:
        """True when a recent reflection trigger asks the gate to
        block new trades. Auto-expires after ``suspension_sec``."""
        now_ms = now_ms or int(self._clock() * 1000)
        st = self._load_state()
        if st.suspended_until_ts_ms <= 0:
            return False
        return now_ms < st.suspended_until_ts_ms

    def can_bypass_suspension(self, *, final_score: float) -> bool:
        """A-quadrant bypass: when reflection has paused the daemon but
        an obviously-strong signal arrives, the gate may still let it
        through. Calibration matches operator's instruction
        (``MISS_PENALTY_AND_PRODUCTION_PLAN.md§A.2.5``)."""
        return final_score >= self.cfg.a_quadrant_bypass_score

    def maybe_trigger(
        self,
        *,
        missed: Iterable[MissedOpportunity],
        actual_trades_in_window: int,
        now_ms: int | None = None,
    ) -> TriggerDecision:
        """Pure synchronous decision: should we enter reflection mode?

        Caller still has to invoke :meth:`generate_report` if the
        decision is positive -- this method does NOT mutate state
        beyond reading the existing reflection_state.json.
        """
        now_ms = now_ms or int(self._clock() * 1000)
        cutoff_ms = now_ms - self.cfg.window_sec * 1000
        misses_in_window = sum(
            1 for m in missed
            if m.is_missed_pump and m.rejected_at_ts_ms >= cutoff_ms
        )
        triggered = (
            misses_in_window >= self.cfg.miss_threshold
            and actual_trades_in_window < self.cfg.trade_threshold
        )

        # Cooldown guard so the daily cron doesn't write a report every
        # run while the operator is still digesting the last one.
        if triggered:
            st = self._load_state()
            since_last = now_ms - st.last_report_ts_ms
            if (
                st.last_report_ts_ms > 0
                and since_last < self.cfg.min_seconds_between_reports * 1000
            ):
                return TriggerDecision(
                    triggered=False,
                    reason="report_cooldown_active",
                    missed_pumps_in_window=misses_in_window,
                    trades_in_window=actual_trades_in_window,
                    window_sec=self.cfg.window_sec,
                )

        return TriggerDecision(
            triggered=triggered,
            reason=(
                "ok_active_strategy"
                if not triggered
                else f"misses={misses_in_window}>=" \
                     f"{self.cfg.miss_threshold}_AND_trades=" \
                     f"{actual_trades_in_window}<{self.cfg.trade_threshold}"
            ),
            missed_pumps_in_window=misses_in_window,
            trades_in_window=actual_trades_in_window,
            window_sec=self.cfg.window_sec,
        )

    async def generate_report(
        self,
        *,
        decision: TriggerDecision,
        missed: Iterable[MissedOpportunity],
        scores: dict[str, RejectReasonScore] | None = None,
        suggested_overrides: list[dict] | None = None,
        now_ms: int | None = None,
    ) -> ReflectionReport:
        """Build the markdown report, persist it, notify operator,
        flip suspension on. Idempotent within ``min_seconds_between_
        reports`` -- the trigger guard handles that.

        ``scores`` and ``suggested_overrides`` are optional: when
        provided they're embedded in the markdown so the operator can
        see what the auto-tuner would have done before saying yes.
        """
        if not decision.triggered:
            raise ValueError(
                "generate_report invoked with a non-triggered decision",
            )

        now_ms = now_ms or int(self._clock() * 1000)
        cutoff_ms = now_ms - self.cfg.window_sec * 1000
        window_misses = [
            m for m in missed
            if m.is_missed_pump and m.rejected_at_ts_ms >= cutoff_ms
        ]
        # Sort by severity DESC so the worst misses lead the markdown.
        window_misses.sort(key=lambda m: m.miss_severity, reverse=True)

        bucket_breakdown: dict[str, int] = {}
        for m in window_misses:
            bucket_breakdown[m.rejected_reason_bucket] = (
                bucket_breakdown.get(m.rejected_reason_bucket, 0) + 1
            )

        # LLM-generated narrative. Optional; falls back to a templated
        # summary when llm_caller is None or raises.
        analysis = await self._safe_llm_analysis(
            window_misses=window_misses,
            scores=scores or {},
            decision=decision,
        )

        report_id = f"reflection_{now_ms}"
        markdown = self._render_markdown(
            report_id=report_id,
            decision=decision,
            now_ms=now_ms,
            cutoff_ms=cutoff_ms,
            window_misses=window_misses,
            bucket_breakdown=bucket_breakdown,
            scores=scores or {},
            suggested_overrides=suggested_overrides or [],
            analysis=analysis,
        )
        markdown_path = self._persist_markdown(report_id, markdown)

        report = ReflectionReport(
            report_id=report_id,
            generated_at_ts_ms=now_ms,
            window_start_ts_ms=cutoff_ms,
            window_end_ts_ms=now_ms,
            missed_pumps_in_window=decision.missed_pumps_in_window,
            trades_in_window=decision.trades_in_window,
            top_missed_opportunities=[
                m.to_dict() for m in window_misses[:10]
            ],
            bucket_breakdown=bucket_breakdown,
            suggested_threshold_changes=suggested_overrides or [],
            llm_analysis=analysis,
            markdown_path=str(markdown_path),
        )

        # Flip state: suspension on, ack pending.
        self._update_state(
            ReflectionState(
                suspended_until_ts_ms=now_ms + self.cfg.suspension_sec * 1000,
                last_report_ts_ms=now_ms,
                last_report_path=str(markdown_path),
                last_trigger_summary=decision.reason,
                acknowledged=False,
                pending_report_id=report_id,
            ),
        )

        # Telegram. Never propagate failures.
        if self._notifier is not None:
            try:
                await self._notifier({
                    "kind": "reflection_report",
                    "title": "策略反思报告：过度保守",
                    "report_id": report_id,
                    "markdown_path": str(markdown_path),
                    "missed_pumps": decision.missed_pumps_in_window,
                    "trades": decision.trades_in_window,
                    "window_days": self.cfg.window_sec // (24 * 3600),
                })
            except Exception as e:
                logger.warning(
                    "ReflectionMode: notifier failed (swallowed): %s", e,
                )

        return report

    def acknowledge(self, *, report_id: str | None = None) -> bool:
        """Operator confirms they've reviewed the report. Clears the
        pending flag so the auto-tuner is allowed to publish overrides.

        ``report_id`` is checked when provided so a stale ack from an
        earlier report doesn't clear a fresh one. Returns True on
        successful ack, False when ``report_id`` doesn't match.
        """
        st = self._load_state()
        if report_id is not None and st.pending_report_id != report_id:
            return False
        st.acknowledged = True
        st.pending_report_id = ""
        st.suspended_until_ts_ms = 0  # clear suspension on ack
        self._update_state(st)
        return True

    def has_pending_review(self) -> bool:
        st = self._load_state()
        return not st.acknowledged and bool(st.pending_report_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _safe_llm_analysis(
        self,
        *,
        window_misses: list[MissedOpportunity],
        scores: dict[str, RejectReasonScore],
        decision: TriggerDecision,
    ) -> str:
        if self._llm is None:
            return self._fallback_analysis(window_misses, scores, decision)
        try:
            prompt = self._build_llm_prompt(
                window_misses=window_misses,
                scores=scores,
                decision=decision,
            )
            text = await self._llm(prompt)
            text = (text or "").strip()
            return text or self._fallback_analysis(
                window_misses, scores, decision,
            )
        except Exception as e:
            logger.warning(
                "ReflectionMode: LLM analysis failed (%s); using fallback", e,
            )
            return self._fallback_analysis(
                window_misses, scores, decision,
            )

    def _build_llm_prompt(
        self,
        *,
        window_misses: list[MissedOpportunity],
        scores: dict[str, RejectReasonScore],
        decision: TriggerDecision,
    ) -> str:
        # Keep prompt under ~3K tokens (operator's budget). We send only
        # top-10 misses + top-5 score buckets, no raw decisions.jsonl.
        top = window_misses[:10]
        miss_lines = [
            (
                f"- symbol={m.symbol} dir={m.direction} "
                f"reject_reason={m.rejected_reason_bucket} "
                f"MFE={m.realized_max_favorable_pct:+.2%} "
                f"MAE={m.realized_max_adverse_pct:+.2%} "
                f"severity={m.miss_severity:.2f}"
            )
            for m in top
        ]
        score_lines = [
            (
                f"- {s.reason}: confidence={s.confidence_score:+.0f} "
                f"correct={s.correct_rejects} missed={s.missed_pumps}"
            )
            for s in sorted(
                scores.values(), key=lambda s: s.confidence_score,
            )[:5]
        ]
        return (
            "你是 altcoin 交易 agent 的策略复盘官。"
            f"过去 {self.cfg.window_sec // 86400} 天内系统错过了 "
            f"{decision.missed_pumps_in_window} 个 +200% 妖币机会，"
            f"但只开了 {decision.trades_in_window} 个仓位 —— "
            "这是过度保守的强信号。\n\n"
            "错过的机会 (top 10):\n" + "\n".join(miss_lines) + "\n\n"
            "拒单理由积分 (worst 5):\n" + "\n".join(score_lines) + "\n\n"
            "请输出一份 markdown 报告，包含三部分：\n"
            "1) 哪些拒绝理由最常错？(列出最多 3 条)\n"
            "2) 阈值是否过严？(yes/no + 一句话原因)\n"
            "3) 给出最多 5 条具体的参数调整建议，每条形如:\n"
            "   `param_name: current_value -> proposed_value (rationale)`\n\n"
            "硬约束：建议不能突破 4 象限策略的 hard_max\n"
            "(anti_chase_max_move_pct <= 6%, min_liquidity_usdt >= 100k,"
            " consecutive_loss_cooldown_sec >= 1h)。\n"
            "总长度不超过 800 字。"
        )

    def _fallback_analysis(
        self,
        window_misses: list[MissedOpportunity],
        scores: dict[str, RejectReasonScore],
        decision: TriggerDecision,
    ) -> str:
        """Deterministic summary when no LLM is available.

        Lists the top miss reasons + biggest single misses with no
        editorialising. Useful in dry-run / token-frozen mode.
        """
        lines = []
        lines.append(
            f"自动复盘：过去 {self.cfg.window_sec // 86400} 天 "
            f"missed_pumps={decision.missed_pumps_in_window}, "
            f"trades={decision.trades_in_window}.",
        )
        # Top 3 buckets.
        buckets: dict[str, int] = {}
        for m in window_misses:
            buckets[m.rejected_reason_bucket] = (
                buckets.get(m.rejected_reason_bucket, 0) + 1
            )
        if buckets:
            top3 = sorted(buckets.items(), key=lambda kv: -kv[1])[:3]
            lines.append("最常错的拒绝理由：")
            for name, count in top3:
                lines.append(f"  - {name}: {count} 次")
        # Worst 3 individual misses.
        if window_misses:
            worst = sorted(
                window_misses, key=lambda m: -m.miss_severity,
            )[:3]
            lines.append("严重程度最高的 3 个错过：")
            for m in worst:
                lines.append(
                    f"  - {m.symbol} ({m.direction}) "
                    f"MFE={m.realized_max_favorable_pct:+.0%} "
                    f"reason={m.rejected_reason_bucket}",
                )
        lines.append(
            "未配置 LLM analysis；以上为基于规则的占位摘要。",
        )
        return "\n".join(lines)

    def _render_markdown(
        self,
        *,
        report_id: str,
        decision: TriggerDecision,
        now_ms: int,
        cutoff_ms: int,
        window_misses: list[MissedOpportunity],
        bucket_breakdown: dict[str, int],
        scores: dict[str, RejectReasonScore],
        suggested_overrides: list[dict],
        analysis: str,
    ) -> str:
        # Operators want a glanceable header + drill-down sections.
        out = []
        out.append(f"# 策略反思报告 — {report_id}")
        out.append("")
        out.append(
            f"**触发条件**：过去 {self.cfg.window_sec // 86400} 天内 "
            f"missed_pumps={decision.missed_pumps_in_window} ≥ "
            f"{self.cfg.miss_threshold}, trades="
            f"{decision.trades_in_window} < {self.cfg.trade_threshold}",
        )
        out.append("")
        out.append(
            f"**暂停时长**：{self.cfg.suspension_sec // 3600} 小时 "
            f"(A 象限 final_score >= "
            f"{self.cfg.a_quadrant_bypass_score} 仍可开单)",
        )
        out.append("")
        out.append("## 1. 错过机会分布 (按 reject_reason)")
        out.append("")
        if bucket_breakdown:
            out.append("| reject_reason | 次数 |")
            out.append("|---|---|")
            for name, count in sorted(
                bucket_breakdown.items(), key=lambda kv: -kv[1],
            ):
                out.append(f"| {name} | {count} |")
        else:
            out.append("_(无)_")
        out.append("")
        out.append("## 2. 严重程度 top 10 错过")
        out.append("")
        if window_misses:
            out.append(
                "| symbol | direction | MFE | MAE | severity | reason |",
            )
            out.append("|---|---|---|---|---|---|")
            for m in window_misses[:10]:
                out.append(
                    f"| {m.symbol} | {m.direction} | "
                    f"{m.realized_max_favorable_pct:+.0%} | "
                    f"{m.realized_max_adverse_pct:+.0%} | "
                    f"{m.miss_severity:.2f} | "
                    f"{m.rejected_reason_bucket} |",
                )
        else:
            out.append("_(无)_")
        out.append("")
        out.append("## 3. 拒单理由积分 (confidence)")
        out.append("")
        if scores:
            out.append("| reason | confidence | correct | missed | win_rate |")
            out.append("|---|---|---|---|---|")
            for s in sorted(
                scores.values(),
                key=lambda s: (s.confidence_score, -s.total_audited),
            ):
                out.append(
                    f"| {s.reason} | {s.confidence_score:+.0f} | "
                    f"{s.correct_rejects} | {s.missed_pumps} | "
                    f"{s.win_rate:.0%} |",
                )
        else:
            out.append("_(无)_")
        out.append("")
        out.append("## 4. 自动调参建议 (待操作员确认)")
        out.append("")
        if suggested_overrides:
            out.append("| param | current | proposed | reason |")
            out.append("|---|---|---|---|")
            for ov in suggested_overrides:
                out.append(
                    f"| {ov.get('param_path', '?')} | "
                    f"{ov.get('previous', '?')} | "
                    f"{ov.get('new', '?')} | "
                    f"{ov.get('reason', '?')} |",
                )
        else:
            out.append(
                "_(无 — 所有 reject_reason 都未达到调参样本/置信度门槛)_",
            )
        out.append("")
        out.append("## 5. 复盘分析")
        out.append("")
        out.append(analysis)
        out.append("")
        out.append("---")
        out.append("")
        out.append(
            "**操作员动作**：审阅以上内容，"
            "调用 `ReflectionModeController.acknowledge(report_id=...)` "
            "或在仪表盘点击 'Acknowledge'。"
            "未确认前 dry_run 自动延长暂停。",
        )
        out.append("")
        return "\n".join(out)

    def _persist_markdown(self, report_id: str, content: str) -> Path:
        out_dir = Path(self.cfg.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{report_id}.md"
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.warning(
                "ReflectionMode: markdown persist failed: %s", e,
            )
        return path

    def _state_path(self) -> Path:
        return Path(self.cfg.state_path)

    def _load_state(self) -> ReflectionState:
        if self._state is not None:
            return self._state
        path = self._state_path()
        if not path.exists():
            self._state = ReflectionState()
            return self._state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            clean = {
                k: v for k, v in data.items()
                if k in ReflectionState.__dataclass_fields__
            }
            self._state = ReflectionState(**clean)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "ReflectionMode: state load failed (%s); fresh start", e,
            )
            self._state = ReflectionState()
        return self._state

    def _update_state(self, st: ReflectionState) -> None:
        self._state = st
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(st.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(
                "ReflectionMode: state persist failed: %s", e,
            )


# --------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------- #


def stub_llm_caller(text: str) -> LLMCaller:
    """Return an :type:`LLMCaller` that always yields ``text``.

    Used by tests that don't want to rig up a full DeepSeek mock --
    we only care that the controller threads the result into the
    markdown.
    """

    async def _call(_prompt: str) -> str:
        return text

    return _call
