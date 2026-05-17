"""main.py — V1.0 production daemon entry point.

Responsibilities:
    * Load .env / YAML config.
    * Wire screener -> asyncio.Queue -> fuser -> risk_gate -> executor.
    * Run reconciler at startup (SR-2). Until reconciliation completes the
      gate refuses every signal.
    * Maintain a TrailingStopFSM per open position, fed by closed klines
      and live ATR.
    * Expose /healthz, /dashboard (HTML) and /api/* JSON on aiohttp.
    * SIGINT / SIGTERM -> graceful shutdown:
        1. Stop accepting new screener events.
        2. Drain the queue with a bounded timeout.
        3. Cancel exchange WS subscriptions, close clients.
    * Three operating modes:
        --dry-run        (default): no orders, just structured logging.
        --paper-trade    : real ccxt orders against testnet/sandbox.
        live (no flag, set in .env): real ccxt orders against mainnet.
    * Optional Telegram notifier (TG_ENABLED=true).
    * Optional dashboard (DASHBOARD_ENABLED=true, default true).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from altcoin_agent.ai_engine import DeepSeekEngine
from altcoin_agent.dashboard import DashboardState, make_dashboard_app
from altcoin_agent.fuser import Direction, FusedSignal, FuserConfig, ScoreFuser
from altcoin_agent.learning_engine import RuleStore
from altcoin_agent.llm.cache import LLMCache
from altcoin_agent.llm.pre_rater import LLMPreRater
from altcoin_agent.llm.token_budget import TokenBudgetManager
from altcoin_agent.notifier import Notifier, build_default_notifier
from altcoin_agent.observability import (
    DeadLetterQueue,
    DLQEntry,
    bind_trace_id,
    build_default_registry,
)
from altcoin_agent.observability.metrics import DefaultMetrics
from altcoin_agent.observability.structured_log import (
    configure_structured_logging,
)
from altcoin_agent.observability.tracing import (
    configure_tracing,
    shutdown_tracing,
    start_span,
)
from altcoin_agent.pipeline import (
    CandidateGate,
    DelayedPostMortemScheduler,
    LLMConsultor,
    RecentSignalsCache,
    cookie_jar_from_env,
    proxy_config_from_env,
)
from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk import (
    AccountPersistor,
    AccountState,
    ATRCalculator,
    CCXTExchangeAdapter,
    CCXTExecutor,
    ClusterCapConfig,
    ClusterMap,
    DecisionAuditLog,
    ExchangeAdapter,
    KillSwitchConfig,
    KillSwitchWatcher,
    Position,
    PositionSizer,
    PositionWatcher,
    Reconciler,
    RegimeFilter,
    RegimeFilterConfig,
    RiskDecision,
    RiskGate,
    RiskGateConfig,
    RollingConfig,
    RollingController,
    Side,
    SQLiteAccountStore,
    TrailingState,
    TrailingStopFSM,
    build_ccxt_adapter,
)
from altcoin_agent.risk.miss_penalty_engine import (
    MissPenaltyConfig,
    MissPenaltyEngine,
    make_ccxt_kline_fetcher,
)
from altcoin_agent.risk.reflection_mode import (
    ReflectionConfig,
    ReflectionModeController,
)
from altcoin_agent.risk.reject_reason_scorer import (
    RejectReasonScorer,
    RejectReasonScorerConfig,
)
from altcoin_agent.risk.threshold_auto_tuner import (
    ThresholdAutoTuner,
    ThresholdAutoTunerConfig,
)
from altcoin_agent.training.production_rules_loader import (
    ProductionRulesLoader,
)
from altcoin_agent.screener import (
    FundingSnapshot,
    Kline,
    OISnapshot,
    Screener,
    SignalEvent,
)
from altcoin_agent.social.historical_analyzer import (
    HistoricalAnalyzer,
    HistoricalAnalyzerConfig,
    KOLHistoryStore,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class AppConfig:
    exchanges: list[str] = field(default_factory=lambda: ["binance"])
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT"])
    timeframes: tuple[str, ...] = ("1m", "5m")
    healthz_port: int = 8080
    graceful_timeout_sec: float = 30.0
    initial_equity_usdt: float = 10_000.0
    dry_run: bool = True
    paper_trade: bool = False
    hedge_mode: bool = False
    min_liquidity_usdt: float = 200_000.0
    dashboard_enabled: bool = True
    # Dashboard exposes positions/orders/rules and MUST NOT share the
    # public 0.0.0.0 healthz bind. Defaults: dashboard on a separate
    # port and bound to loopback. Operators that want remote access
    # should either SSH-tunnel (``ssh -L``) or set ``dashboard_token``
    # AND change ``dashboard_bind`` to a routable interface; the
    # daemon will refuse to start on a non-loopback bind without a
    # token (see ``App.run``). Bug C1 fix.
    dashboard_bind: str = "127.0.0.1"
    # 0 = "follow healthz_port + 1" (resolved at run time). Lets tests
    # that only set ``healthz_port`` keep their two sites on adjacent
    # ports without colliding across the suite.
    dashboard_port: int = 0
    dashboard_token: str = ""
    # Healthz can stay on 0.0.0.0 because it returns no operational
    # data; container probes need it reachable.
    healthz_bind: str = "0.0.0.0"
    # Path to dynamic_rules.json. The dashboard always needs a concrete path,
    # while the LLM/post-mortem path resolution will fall through env vars or
    # the fuser default when the user leaves this on the default value.
    dynamic_rules_path: str = ".kiro/steering/dynamic_rules.json"
    # LLM consult tuning
    llm_consult_cooldown_sec: int = 300   # per-symbol min spacing between consults
    llm_queue_max: int = 256
    # Online post-mortem (self-evolution)
    post_mortem_delay_sec: int = 3600     # 1h after open
    # Position-watcher (close lifecycle)
    position_watcher_poll_sec: float = 5.0
    position_watcher_miss_threshold: int = 2
    # Bug #3 fix: trading-day rollover.
    # ``rollover_anchor_utc_hour`` (0-23) defines when one trading day
    # ends and the next begins. Default 0 == midnight UTC, the
    # convention used by Binance reporting and most prop desks.
    # ``rollover_poll_sec`` is how often the background ticker checks
    # for a day flip; small enough to fire within ~minute of the
    # boundary, large enough to be free.
    rollover_anchor_utc_hour: int = 0
    rollover_poll_sec: float = 60.0

    # ------------------------------------------------------------------ #
    # Rolling positions (滚仓 / pyramid-add).
    #
    # Default OFF: enabling this is a deliberate operator decision
    # because it adds same-side exposure to a winning position and
    # therefore both the upside AND the path-dependent downside scale
    # with each new leg. See .kiro/specs/.../rolling-positions.md for
    # the full risk model.
    # ------------------------------------------------------------------ #
    rolling_enabled: bool = False
    rolling_trigger_r_levels: tuple[float, ...] = (1.5, 3.0, 5.0)
    rolling_unrealized_pnl_ratio: float = 0.5
    rolling_leg_stop_pct: float = 0.025
    rolling_max_legs_per_symbol: int = 3
    rolling_min_interval_sec: int = 60
    rolling_auto_disable_on_failure: bool = True
    rolling_require_strategy_min_score: float = 85.0
    rolling_require_min_rule_score: float = 35.0

    # ------------------------------------------------------------------ #
    # Low-latency hot-path knobs (for high-volatility altcoins).
    #
    # ``anti_chase_*`` and ``vol_kill_*`` configure the price-tape gate;
    # see ``altcoin_agent.price_tape.PriceTapeConfig`` for the full
    # rationale. Defaults are calibrated for typical Binance perp
    # altcoin pump-and-dump behaviour — refuse to enter when the move
    # has already run away from us, and refuse to enter while the tape
    # is in a 60-second whipsaw cascade.
    #
    # ``telegram_fire_and_forget`` decouples Telegram I/O from the
    # order placement hot path: when True, ``notifier.signal`` is
    # spawned as a background task instead of awaited inline, saving
    # 100–800 ms of HTTPS RTT before the venue order goes out.
    #
    # ``use_uvloop`` swaps in uvloop's event-loop policy at startup
    # when available. Typically a 30–60% throughput improvement on the
    # async hot path; safe to leave on if uvloop is installed.
    # ------------------------------------------------------------------ #
    anti_chase_window_ms: int = 30_000
    anti_chase_max_move_pct: float = 0.025
    vol_kill_window_ms: int = 60_000
    vol_kill_range_pct: float = 0.08
    price_tape_max_samples: int = 5_000
    telegram_fire_and_forget: bool = True
    use_uvloop: bool = True

    # Bug C4 fix: dry-run only. When the PriceTape doesn't have enough
    # samples yet to compute realized vol (cold start, mocked screener,
    # offline test), use this conservative default for sizing instead of
    # rejecting the order. Calibrated for typical altcoin vol around
    # half the vol-kill cap (3-4%), an order of magnitude above
    # BTC-grade and safe to size against. LIVE mode never falls back —
    # cold tape there is fail-closed.
    dry_run_fallback_vol_pct: float = 0.04

    # ------------------------------------------------------------------ #
    # Audit (third pass) #1: safety modules wiring.
    #
    # PR #21 introduced AccountPersistor / RegimeFilter / ClusterMap /
    # KillSwitchWatcher / DecisionAuditLog as standalone modules but
    # never wired them into ``App.run`` — they were dead code at
    # runtime. The third audit pass flagged this as the most
    # important finding. The knobs below let operators turn each
    # capability on individually and tune the parameters; defaults are
    # chosen to match the per-module intent (regime filter on,
    # cluster cap on with one-per-cluster default, persistence on,
    # audit log on, kill switch on with sentinel file path).
    # ------------------------------------------------------------------ #
    # AccountState persistence (audit #12)
    account_persistence_enabled: bool = True
    account_persistence_path: str = ".kiro/state/account.json"
    # BTC market-regime gate (audit #10)
    regime_filter_enabled: bool = True
    regime_reference_symbol: str = "BTC/USDT:USDT"
    regime_btc_window_ms: int = 60 * 60 * 1000      # 1h
    regime_btc_drop_block_long_pct: float = 0.03    # 3% drop -> block LONG
    regime_btc_rip_block_short_pct: float = 0.05    # 5% rip -> block SHORT
    regime_min_samples: int = 10
    # Symbol cluster cap (audit #11)
    # Default OFF: most operators don't yet have a meaningful
    # ``cluster_map`` populated, and a default-on cap with empty map
    # would put every symbol in the ``other`` bucket, making the
    # second concurrent position always rejected. Once an operator
    # configures ``cluster_map`` in app.yaml they can flip this on.
    cluster_cap_enabled: bool = False
    cluster_max_per_cluster: int = 1
    # Map of base-token -> cluster-name. Empty by default; everything
    # falls into ``other``. Operators populate this in app.yaml so
    # PEPE/WIF/FLOKI all map to ``meme`` and don't compete for the same
    # cap as BTC/ETH/SOL.
    cluster_map: dict[str, str] = field(default_factory=dict)
    # Kill switch (audit #25)
    kill_switch_enabled: bool = True
    kill_switch_path: str = ".kiro/state/HALT"
    kill_switch_poll_sec: float = 2.0
    # Decision audit log (audit #28)
    decision_audit_log_enabled: bool = True
    decision_audit_log_path: str = "logs/decisions.jsonl"

    # ------------------------------------------------------------------ #
    # Phase B.2 — observability (metrics + structured logs + DLQ).
    #
    # ``metrics_enabled`` toggles the Phase B.2.1 :class:`MetricsRegistry`.
    # When True, ``/metrics`` returns BOTH the legacy hand-rolled gauges
    # AND the full registry render (~30 metrics including histograms
    # for order/LLM/persistence latency and counters with reject-reason
    # / partial-fill labels).
    #
    # ``structured_logging_enabled`` swaps the root logger formatter to
    # JSON and binds a ``trace_id`` per high-priority decision so the
    # ~10-hop pipeline is grep-correlatable (screener -> fuser -> LLM ->
    # gate -> executor -> trailing).
    #
    # ``dlq_enabled`` writes one JSONL line per unrecoverable failure
    # (executor exceptions, quote/depth/vol unavailable, naked
    # emergency-close, stop-replace failures) under
    # ``dlq_path``. Failures are rotated like ``decision_audit_log_path``.
    #
    # All three default OFF so existing dry-run / integration tests
    # keep their byte-for-byte log shapes; operators flip them on in
    # ``app.yaml`` for production. Phase B.2 doesn't modify any
    # behaviour visible to the trading hot path -- just adds
    # side-channel observability.
    # ------------------------------------------------------------------ #
    metrics_enabled: bool = False
    structured_logging_enabled: bool = False
    dlq_enabled: bool = False
    dlq_path: str = ".kiro/state/dlq/main.jsonl"

    # ------------------------------------------------------------------ #
    # Phase B.3 — state persistence backend.
    #
    # ``persistence_backend`` selects between the original JSON
    # ``AccountPersistor`` (default, byte-for-byte unchanged) and the
    # Phase B.3.1 :class:`SQLiteAccountStore`. The SQLite backend
    # writes the same snapshot fields PLUS an append-only
    # ``state_log`` table for postmortem reconstruction across days.
    # ``account_persistence_sqlite_path`` is the SQLite file (WAL
    # journal sits next to it).
    # ------------------------------------------------------------------ #
    persistence_backend: str = "json"  # "json" | "sqlite"
    account_persistence_sqlite_path: str = ".kiro/state/account.sqlite3"

    # ------------------------------------------------------------------ #
    # Phase A — opportunity-cost penalty pipeline.
    #
    # The miss-penalty worker runs once per UTC day and (a) audits the
    # last 48h of rejected decisions to find missed pumps, (b)
    # recomputes the +1 / -3 reject-reason scores. The threshold tuner
    # runs once per UTC week (Sunday) and emits override proposals.
    # The reflection-mode controller is checked daily after the
    # audit; when it fires the daemon enters a 24h suspension window
    # and Telegram pings the operator with a markdown report.
    #
    # Defaults align with MISS_PENALTY_AND_PRODUCTION_PLAN.md§A.2 —
    # 7-day window, miss>=3 AND trades<2 -> reflection.
    #
    # Default OFF: while the audit log isn't yet a week old, the cron
    # would always classify recent rejections as ``insufficient_data``
    # and never cross the trigger threshold. Operators flip this on
    # in app.yaml after a week of dry-run rejections has accumulated.
    # ------------------------------------------------------------------ #
    miss_penalty_enabled: bool = False
    miss_penalty_state_dir: str = ".kiro/state/miss_penalty"
    miss_penalty_run_at_utc_hour: int = 2  # 02:00 UTC, off-peak
    miss_penalty_poll_sec: float = 5 * 60.0  # check every 5 min
    miss_penalty_lookback_hours: int = 48
    # 24h forward window before a rejection becomes "auditable".
    miss_penalty_forward_window_sec: int = 24 * 3600
    # Reflection-mode trigger.
    reflection_window_days: int = 7
    reflection_miss_threshold: int = 3
    reflection_trade_threshold: int = 2
    reflection_suspension_hours: int = 24
    # A-quadrant bypass: signals with final_score >= this STILL go
    # through even when reflection is suspending the daemon.
    reflection_a_quadrant_bypass_score: float = 95.0
    reflection_reports_dir: str = ".kiro/state/reflection_reports"
    # When True, the reflection report calls into the existing
    # DeepSeek engine (re-using its token budget). When False, only
    # the deterministic fallback summary is written -- safer for
    # first-week dry-runs where token budgets aren't dialled in yet.
    reflection_use_llm: bool = False
    # Threshold tuner runs only on the UTC weekday matching this number.
    # 6 == Sunday (Python's date.weekday(): Mon=0..Sun=6). Daily audit
    # still runs every day; only the proposal step is gated.
    threshold_tuner_run_on_weekday: int = 6

    # ------------------------------------------------------------------ #
    # Phase 5 — LLM cache + tier-aware budget + pre-rate worker.
    #
    # All three are opt-in (default OFF) so a fresh deployment behaves
    # byte-for-byte like v1.0. When ``llm_cache_enabled`` is True the
    # cache is wired into ``LLMEngine.judge`` keyed by
    # (symbol, phase, social_hash); cache hits return the prior
    # verdict in 0ms with zero tokens consumed. When
    # ``llm_budget_manager_enabled`` is True the engine consults the
    # tier-aware ``TokenBudgetManager`` (FREE/ECONOMY/EMERGENCY/FREEZE)
    # before paying for a call. When ``llm_pre_rate_enabled`` is True
    # the daemon spawns a background worker that proactively calls the
    # engine for high-priority A-quadrant candidates so the hot path
    # always hits the cache.
    #
    # Operators wanting full Phase 5 behaviour should set all three to
    # True. Setting only the cache (no budget manager) gives the
    # latency win without the tier policy. Setting only the budget
    # manager (no cache) is wasteful (every miss pays full network
    # latency); we don't recommend that combination.
    # ------------------------------------------------------------------ #
    llm_cache_enabled: bool = False
    llm_cache_path: str = ".kiro/state/llm_cache.json"
    llm_cache_max_entries: int = 1024
    llm_cache_ttl_sec: int = 12 * 3600
    llm_budget_manager_enabled: bool = False
    llm_budget_state_path: str = ".kiro/state/token_usage.json"
    llm_budget_monthly_tokens: int = 5_000_000
    llm_pre_rate_enabled: bool = False
    # Per the plan's revised math, only A-quadrant + score >= 70
    # keep token spend under 2.25M/month (vs naive 22M/month).
    llm_pre_rate_min_score: float = 70.0
    llm_pre_rate_queue_max: int = 64

    # ------------------------------------------------------------------ #
    # Phase B.6 sister deliverable — KOL historical hit-rate analyzer.
    #
    # The fuser already routes ``exit_liquidity`` LLM verdicts through a
    # confidence-graded HARD VETO / SOFT CAP path; without history every
    # KOL gets the same weight regardless of past accuracy. When
    # ``kol_history_enabled`` is True the daemon constructs a
    # :class:`KOLHistoryStore` (atomic JSON) plus a wrapping
    # :class:`HistoricalAnalyzer`, plumbs it into the fuser, and the
    # delayed post-mortem scheduler records one observation per cited
    # author 1h after each entry — same data the rule learner sees, so
    # the two memories converge consistently.
    #
    # Default OFF: the store starts empty, so until the operator
    # bootstraps it (via ``scripts/rebuild_kol_history.py`` or organic
    # accumulation) every author would land in the
    # ``insufficient_samples`` branch and the analyzer would no-op
    # anyway. Operators flip it on AFTER seeding the file.
    # ------------------------------------------------------------------ #
    kol_history_enabled: bool = False
    kol_history_path: str = ".kiro/state/social/kol_history.json"
    kol_history_min_samples: int = 10
    kol_history_strong_bound: float = 0.65
    kol_history_weak_bound: float = 0.40
    kol_history_conf_lift_max: float = 0.20
    kol_history_conf_drop_max: float = 0.20

    # ------------------------------------------------------------------ #
    # Phase B.6 — OpenTelemetry distributed-trace exporter.
    #
    # ``tracing_enabled`` toggles the
    # :mod:`altcoin_agent.observability.tracing` exporter. When False
    # (default) every span helper is a cheap no-op, and the daemon's
    # behaviour is byte-for-byte identical to v1.0. When True the
    # daemon initialises an OTel TracerProvider with a Resource carrying
    # ``service.name`` and ``deployment.environment``, attaches a
    # BatchSpanProcessor backed by an OTLP/gRPC exporter (when
    # ``tracing_otlp_endpoint`` is set) and/or a synchronous Console
    # exporter (``tracing_console=true``), and copies each span's
    # trace-id into the existing ``structured_log`` contextvar so a
    # ``grep trace_id`` correlates JSON logs with the exporter's
    # spans without operator guesswork.
    #
    # Operators that flip ``tracing_enabled=true`` MUST install the
    # optional extra ``pip install -e '.[otel]'``; without it the
    # tracer logs once at WARNING and stays disabled (never crashes).
    # ------------------------------------------------------------------ #
    tracing_enabled: bool = False
    tracing_service_name: str = "altcoin-agent"
    tracing_otlp_endpoint: str = ""
    tracing_otlp_insecure: bool = True
    tracing_console: bool = False
    tracing_sampler_ratio: float = 1.0
    tracing_environment: str = "production"

    # ------------------------------------------------------------------ #
    # R3 — production_rules.json hot reload.
    #
    # The walk-forward trainer (``scripts/run_walkforward_trainer.py``)
    # writes ``production_rules.json`` into ``production_rules_dir``
    # whenever a rule clears the 80% gate. The live daemon polls the
    # file's mtime every ``production_rules_reload_interval_sec`` and
    # rebuilds an in-memory snapshot on change. Subscribers (the
    # fuser / risk gate / dashboard) consult the snapshot through
    # ``ProductionRulesLoader.lookup_by_features`` — no restart
    # required.
    #
    # Default OFF: a fresh deployment has no production rules yet and
    # turning the worker on before the first training cycle just
    # logs "file not found" warnings. Operators flip this on in
    # ``app.yaml`` after the first ``run_walkforward_trainer.py``
    # run produces the file.
    # ------------------------------------------------------------------ #
    production_rules_enabled: bool = False
    production_rules_dir: str = ".kiro/state/training"
    production_rules_reload_interval_sec: float = 600.0  # 10 min

    @classmethod
    def from_file(cls, path: str) -> AppConfig:
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyyaml is required to load YAML config") from e
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(
            exchanges=d.get("exchanges", ["binance"]),
            symbols=d.get("symbols", ["BTC/USDT:USDT"]),
            timeframes=tuple(d.get("timeframes", ["1m", "5m"])),
            healthz_port=int(d.get("healthz_port", 8080)),
            graceful_timeout_sec=float(d.get("graceful_timeout_sec", 30)),
            initial_equity_usdt=float(d.get("initial_equity_usdt", 10_000)),
            dry_run=bool(d.get("dry_run", True)),
            paper_trade=bool(d.get("paper_trade", False)),
            hedge_mode=bool(d.get("hedge_mode", False)),
            min_liquidity_usdt=float(d.get("min_liquidity_usdt", 200_000)),
            dashboard_enabled=bool(d.get("dashboard_enabled", True)),
            dashboard_bind=str(d.get("dashboard_bind", "127.0.0.1")),
            dashboard_port=int(d.get("dashboard_port", 0)),
            dashboard_token=str(d.get("dashboard_token", "")),
            healthz_bind=str(d.get("healthz_bind", "0.0.0.0")),
            dynamic_rules_path=str(d.get(
                "dynamic_rules_path", ".kiro/steering/dynamic_rules.json")),
            llm_consult_cooldown_sec=int(d.get("llm_consult_cooldown_sec", 300)),
            llm_queue_max=int(d.get("llm_queue_max", 256)),
            post_mortem_delay_sec=int(d.get("post_mortem_delay_sec", 3600)),
            position_watcher_poll_sec=float(
                d.get("position_watcher_poll_sec", 5.0),
            ),
            position_watcher_miss_threshold=int(
                d.get("position_watcher_miss_threshold", 2),
            ),
            rollover_anchor_utc_hour=int(
                d.get("rollover_anchor_utc_hour", 0),
            ),
            rollover_poll_sec=float(
                d.get("rollover_poll_sec", 60.0),
            ),
            rolling_enabled=bool(d.get("rolling_enabled", False)),
            rolling_trigger_r_levels=tuple(
                float(x) for x in d.get(
                    "rolling_trigger_r_levels", (1.5, 3.0, 5.0)
                )
            ),
            rolling_unrealized_pnl_ratio=float(
                d.get("rolling_unrealized_pnl_ratio", 0.5),
            ),
            rolling_leg_stop_pct=float(
                d.get("rolling_leg_stop_pct", 0.025),
            ),
            rolling_max_legs_per_symbol=int(
                d.get("rolling_max_legs_per_symbol", 3),
            ),
            rolling_min_interval_sec=int(
                d.get("rolling_min_interval_sec", 60),
            ),
            rolling_auto_disable_on_failure=bool(
                d.get("rolling_auto_disable_on_failure", True),
            ),
            rolling_require_strategy_min_score=float(
                d.get("rolling_require_strategy_min_score", 85.0),
            ),
            rolling_require_min_rule_score=float(
                d.get("rolling_require_min_rule_score", 35.0),
            ),
            anti_chase_window_ms=int(d.get("anti_chase_window_ms", 30_000)),
            anti_chase_max_move_pct=float(
                d.get("anti_chase_max_move_pct", 0.025)
            ),
            vol_kill_window_ms=int(d.get("vol_kill_window_ms", 60_000)),
            vol_kill_range_pct=float(d.get("vol_kill_range_pct", 0.08)),
            price_tape_max_samples=int(d.get("price_tape_max_samples", 5_000)),
            telegram_fire_and_forget=bool(
                d.get("telegram_fire_and_forget", True)
            ),
            use_uvloop=bool(d.get("use_uvloop", True)),
            # Audit (third pass) #1: safety wiring.
            account_persistence_enabled=bool(
                d.get("account_persistence_enabled", True),
            ),
            account_persistence_path=str(
                d.get(
                    "account_persistence_path", ".kiro/state/account.json",
                ),
            ),
            regime_filter_enabled=bool(
                d.get("regime_filter_enabled", True),
            ),
            regime_reference_symbol=str(
                d.get("regime_reference_symbol", "BTC/USDT:USDT"),
            ),
            regime_btc_window_ms=int(
                d.get("regime_btc_window_ms", 60 * 60 * 1000),
            ),
            regime_btc_drop_block_long_pct=float(
                d.get("regime_btc_drop_block_long_pct", 0.03),
            ),
            regime_btc_rip_block_short_pct=float(
                d.get("regime_btc_rip_block_short_pct", 0.05),
            ),
            regime_min_samples=int(
                d.get("regime_min_samples", 10),
            ),
            cluster_cap_enabled=bool(
                d.get("cluster_cap_enabled", False),
            ),
            cluster_max_per_cluster=int(
                d.get("cluster_max_per_cluster", 1),
            ),
            cluster_map={
                str(k): str(v)
                for k, v in (d.get("cluster_map") or {}).items()
            },
            kill_switch_enabled=bool(
                d.get("kill_switch_enabled", True),
            ),
            kill_switch_path=str(
                d.get("kill_switch_path", ".kiro/state/HALT"),
            ),
            kill_switch_poll_sec=float(
                d.get("kill_switch_poll_sec", 2.0),
            ),
            decision_audit_log_enabled=bool(
                d.get("decision_audit_log_enabled", True),
            ),
            decision_audit_log_path=str(
                d.get("decision_audit_log_path", "logs/decisions.jsonl"),
            ),
            # Phase B.2 — observability.
            metrics_enabled=bool(d.get("metrics_enabled", False)),
            structured_logging_enabled=bool(
                d.get("structured_logging_enabled", False),
            ),
            dlq_enabled=bool(d.get("dlq_enabled", False)),
            dlq_path=str(d.get("dlq_path", ".kiro/state/dlq/main.jsonl")),
            # Phase B.3 — persistence backend.
            persistence_backend=str(
                d.get("persistence_backend", "json"),
            ),
            account_persistence_sqlite_path=str(
                d.get(
                    "account_persistence_sqlite_path",
                    ".kiro/state/account.sqlite3",
                ),
            ),
            # Phase A — miss-penalty pipeline.
            miss_penalty_enabled=bool(d.get("miss_penalty_enabled", False)),
            miss_penalty_state_dir=str(
                d.get("miss_penalty_state_dir", ".kiro/state/miss_penalty"),
            ),
            miss_penalty_run_at_utc_hour=int(
                d.get("miss_penalty_run_at_utc_hour", 2),
            ),
            miss_penalty_poll_sec=float(
                d.get("miss_penalty_poll_sec", 5 * 60.0),
            ),
            miss_penalty_lookback_hours=int(
                d.get("miss_penalty_lookback_hours", 48),
            ),
            miss_penalty_forward_window_sec=int(
                d.get("miss_penalty_forward_window_sec", 24 * 3600),
            ),
            reflection_window_days=int(
                d.get("reflection_window_days", 7),
            ),
            reflection_miss_threshold=int(
                d.get("reflection_miss_threshold", 3),
            ),
            reflection_trade_threshold=int(
                d.get("reflection_trade_threshold", 2),
            ),
            reflection_suspension_hours=int(
                d.get("reflection_suspension_hours", 24),
            ),
            reflection_a_quadrant_bypass_score=float(
                d.get("reflection_a_quadrant_bypass_score", 95.0),
            ),
            reflection_reports_dir=str(
                d.get(
                    "reflection_reports_dir",
                    ".kiro/state/reflection_reports",
                ),
            ),
            reflection_use_llm=bool(d.get("reflection_use_llm", False)),
            threshold_tuner_run_on_weekday=int(
                d.get("threshold_tuner_run_on_weekday", 6),
            ),
            # Phase 5 — LLM cache + budget manager + pre-rater.
            llm_cache_enabled=bool(d.get("llm_cache_enabled", False)),
            llm_cache_path=str(
                d.get("llm_cache_path", ".kiro/state/llm_cache.json"),
            ),
            llm_cache_max_entries=int(d.get("llm_cache_max_entries", 1024)),
            llm_cache_ttl_sec=int(d.get("llm_cache_ttl_sec", 12 * 3600)),
            llm_budget_manager_enabled=bool(
                d.get("llm_budget_manager_enabled", False),
            ),
            llm_budget_state_path=str(
                d.get(
                    "llm_budget_state_path", ".kiro/state/token_usage.json",
                ),
            ),
            llm_budget_monthly_tokens=int(
                d.get("llm_budget_monthly_tokens", 5_000_000),
            ),
            llm_pre_rate_enabled=bool(d.get("llm_pre_rate_enabled", False)),
            llm_pre_rate_min_score=float(
                d.get("llm_pre_rate_min_score", 70.0),
            ),
            llm_pre_rate_queue_max=int(
                d.get("llm_pre_rate_queue_max", 64),
            ),
            # Phase B.6 sister deliverable — KOL historical analyzer.
            kol_history_enabled=bool(d.get("kol_history_enabled", False)),
            kol_history_path=str(
                d.get("kol_history_path", ".kiro/state/social/kol_history.json"),
            ),
            kol_history_min_samples=int(
                d.get("kol_history_min_samples", 10),
            ),
            kol_history_strong_bound=float(
                d.get("kol_history_strong_bound", 0.65),
            ),
            kol_history_weak_bound=float(
                d.get("kol_history_weak_bound", 0.40),
            ),
            kol_history_conf_lift_max=float(
                d.get("kol_history_conf_lift_max", 0.20),
            ),
            kol_history_conf_drop_max=float(
                d.get("kol_history_conf_drop_max", 0.20),
            ),
            # Phase B.6 — OpenTelemetry tracer.
            tracing_enabled=bool(d.get("tracing_enabled", False)),
            tracing_service_name=str(
                d.get("tracing_service_name", "altcoin-agent"),
            ),
            tracing_otlp_endpoint=str(
                d.get("tracing_otlp_endpoint", ""),
            ),
            tracing_otlp_insecure=bool(
                d.get("tracing_otlp_insecure", True),
            ),
            tracing_console=bool(d.get("tracing_console", False)),
            tracing_sampler_ratio=float(
                d.get("tracing_sampler_ratio", 1.0),
            ),
            tracing_environment=str(
                d.get("tracing_environment", "production"),
            ),
            # R3 — production_rules.json hot reload.
            production_rules_enabled=bool(
                d.get("production_rules_enabled", False),
            ),
            production_rules_dir=str(
                d.get("production_rules_dir", ".kiro/state/training"),
            ),
            production_rules_reload_interval_sec=float(
                d.get("production_rules_reload_interval_sec", 600.0),
            ),
        )


# --------------------------------------------------------------------- #
# Health server
# --------------------------------------------------------------------- #


@dataclass
class HealthState:
    started_at: float = 0.0
    fuser_alive: bool = True
    screener_alive: bool = True
    reconciliation_complete: bool = False
    last_signal_ts: float = 0.0
    high_priority_count: int = 0
    rule_event_count: int = 0
    open_positions: int = 0
    orders_placed: int = 0
    orders_rejected: int = 0
    closed_positions: int = 0
    last_close_ts: float = 0.0
    llm_consults: int = 0
    llm_consults_skipped: int = 0
    post_mortems_scheduled: int = 0
    last_error: str | None = None


async def make_health_app(
    state: HealthState,
    *,
    metrics_registry: Any = None,
) -> web.Application:
    """Build the healthz + /metrics aiohttp app.

    Phase B.2.1 hook: when ``metrics_registry`` is a
    :class:`altcoin_agent.observability.MetricsRegistry`, its rendered
    Prometheus text is appended after the legacy hand-rolled gauges so
    existing test-shape assertions (``altcoin_agent_up 1.0``,
    ``altcoin_agent_high_priority_count 7``) keep matching while the
    operator gets the full Phase B.2 metric set in the SAME scrape.
    """
    async def healthz(_request: web.Request) -> web.Response:
        ok = (
            state.fuser_alive
            and state.screener_alive
            and state.reconciliation_complete
        )
        body = {
            "status": "ok" if ok else "degraded",
            "uptime_sec": round(time.time() - state.started_at, 1),
            "fuser_alive": state.fuser_alive,
            "screener_alive": state.screener_alive,
            "reconciliation_complete": state.reconciliation_complete,
            "high_priority_count": state.high_priority_count,
            "rule_event_count": state.rule_event_count,
            "open_positions": state.open_positions,
            "orders_placed": state.orders_placed,
            "orders_rejected": state.orders_rejected,
            "closed_positions": state.closed_positions,
            "last_close_ts": state.last_close_ts,
            "llm_consults": state.llm_consults,
            "llm_consults_skipped": state.llm_consults_skipped,
            "post_mortems_scheduled": state.post_mortems_scheduled,
            "last_signal_ts": state.last_signal_ts,
            "last_error": state.last_error,
        }
        return web.json_response(body, status=200 if ok else 503)

    async def metrics(_request: web.Request) -> web.Response:
        # Audit #23: minimal Prometheus text-format exporter. We bind
        # the same fields we already publish via /healthz so operators
        # can plot SLOs without bringing in a heavy client library.
        # All metrics are gauges (counters that only increase are also
        # valid gauges); no labels for V1 simplicity. Follow-up PR can
        # add per-symbol labels once the cardinality budget is set.
        lines: list[str] = []

        def gauge(name: str, value: float, help_text: str) -> None:
            lines.append(f"# HELP altcoin_agent_{name} {help_text}")
            lines.append(f"# TYPE altcoin_agent_{name} gauge")
            lines.append(f"altcoin_agent_{name} {value}")

        gauge("up", 1.0 if (
            state.fuser_alive and state.screener_alive
            and state.reconciliation_complete
        ) else 0.0, "1 if all subsystems alive AND reconciled")
        gauge("uptime_sec",
              round(time.time() - state.started_at, 1),
              "Seconds since boot")
        gauge("fuser_alive", 1.0 if state.fuser_alive else 0.0,
              "1 if the fuser worker is running")
        gauge("screener_alive", 1.0 if state.screener_alive else 0.0,
              "1 if the screener worker is running")
        gauge("reconciliation_complete",
              1.0 if state.reconciliation_complete else 0.0,
              "1 if startup reconciler completed successfully")
        gauge("high_priority_count", state.high_priority_count,
              "Total high-priority FusedSignals emitted")
        gauge("rule_event_count", state.rule_event_count,
              "Total raw screener events seen")
        gauge("open_positions", state.open_positions,
              "Currently open positions")
        gauge("orders_placed", state.orders_placed,
              "Total entry orders placed")
        gauge("orders_rejected", state.orders_rejected,
              "Total entries rejected by the gate or executor")
        gauge("closed_positions", state.closed_positions,
              "Total positions closed")
        gauge("last_close_ts", state.last_close_ts,
              "Wall-clock ts of most-recent close")
        gauge("llm_consults", state.llm_consults,
              "Total LLM consults that produced a verdict")
        gauge("llm_consults_skipped", state.llm_consults_skipped,
              "Total LLM consults skipped (no engine, dropped, error)")
        gauge("post_mortems_scheduled", state.post_mortems_scheduled,
              "Total post-mortem learning passes scheduled after open")
        gauge("last_signal_ts", state.last_signal_ts,
              "Wall-clock ts of most-recent screener event")
        text = "\n".join(lines) + "\n"
        # Phase B.2.1: append the rich registry's Prometheus payload.
        # Render failures must NEVER block the health endpoint, so we
        # swallow exceptions and emit a comment line instead.
        if metrics_registry is not None:
            try:
                extra = metrics_registry.render()
                if extra:
                    text += extra
            except Exception as e:  # pragma: no cover - defensive
                text += f"# RENDER_ERROR registry: {e}\n"
        return web.Response(
            text=text,
            content_type="text/plain",
            charset="utf-8",
        )

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/metrics", metrics)
    return app


# --------------------------------------------------------------------- #
# Dry-run exchange adapter (no network)
# --------------------------------------------------------------------- #


class DryRunExchangeAdapter:
    """Mimics ``ExchangeAdapter`` but logs everything instead of placing
    real orders. Used when ``--dry-run`` is set.

    Tracks a tiny in-memory ``_open_positions`` map so the
    :class:`PositionWatcher` can observe simulated entries and reduce-only
    market closes the same way it observes real exchange state. Tests can
    drive ``simulate_close()`` to mimic the exchange firing a STOP_MARKET.
    """

    def __init__(self) -> None:
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self.leverages: list[tuple[str, float]] = []
        self._n = 0
        # symbol -> {"side": "long"/"short", "size": float, "entryPrice": float}
        self._open_positions: dict[str, dict[str, Any]] = {}
        # Test/integration helper for the dynamic-slippage live-quote path
        # (Bug #2). When set, ``fetch_ticker_price`` returns this value;
        # otherwise it raises so the gate falls into the fail-closed path
        # (matching live-mode behaviour where a missing quote is fatal for
        # SR-1, not a silent passthrough).
        self._mark_prices: dict[str, float] = {}
        # Bug C4 fix: parallel to ``_mark_prices`` for top-5 depth in
        # USDT. Dry-run defaults to "infinite" depth so the SR-2
        # liquidity gate doesn't reject every signal in test/dry mode;
        # tests can override per-symbol with ``set_top_depth``.
        self._top_depths: dict[str, float] = {}
        self._default_top_depth_usdt: float = 1_000_000_000.0

    def _id(self) -> str:
        self._n += 1
        return f"dryrun-{self._n}"

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False,
                           client_order_id=None):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "price": price, "reduce_only": reduce_only,
               "average": price or 0.0,
               "client_order_id": client_order_id}
        self.market_orders.append(rec)
        # Maintain a fake position book so PositionWatcher sees consistent
        # state. An entry is the order whose side matches the position's
        # eventual side; a reduce_only market order closes it.
        if reduce_only:
            self._open_positions.pop(symbol, None)
        else:
            pos_side = "long" if side.value == "long" else "short"
            self._open_positions[symbol] = {
                "symbol": symbol, "side": pos_side,
                "contracts": float(size),
                "entryPrice": float(price or 0.0),
            }
        logger.info("[DRY-RUN] MARKET %s %s %s @ %s reduce=%s coid=%s",
                    side.value.upper(), size, symbol, price, reduce_only,
                    client_order_id)
        return rec

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True,
                               client_order_id=None):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "stop_price": stop_price, "reduce_only": reduce_only,
               "client_order_id": client_order_id}
        self.stop_orders.append(rec)
        logger.info("[DRY-RUN] STOP-MARKET %s %s %s @ %s coid=%s",
                    side.value.upper(), size, symbol, stop_price, client_order_id)
        return rec

    async def cancel_order(self, order_id, symbol):  # noqa: ANN001
        self.cancelled.append(order_id)
        logger.info("[DRY-RUN] CANCEL %s on %s", order_id, symbol)
        return {"id": order_id, "status": "cancelled"}

    async def set_leverage(self, symbol, leverage):  # noqa: ANN001
        self.leverages.append((symbol, leverage))
        logger.info("[DRY-RUN] SET LEVERAGE %s on %s", leverage, symbol)
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_positions(self) -> list[dict]:
        return [dict(p) for p in self._open_positions.values()]

    async def fetch_open_orders(self) -> list[dict]:
        return []

    async def fetch_ticker_price(self, symbol: str) -> float:
        """Return the test-supplied mark price, or raise if none was set.

        Tests drive this via ``set_mark_price`` to exercise SR-1's adverse-
        slip check. Raising on missing data is the deliberate, conservative
        choice: it makes the dry-run path behave the same way live mode
        will when the venue is unreachable -- the gate fail-closes with
        ``quote_unavailable`` instead of silently re-using the trigger.
        """
        if symbol in self._mark_prices:
            return self._mark_prices[symbol]
        raise RuntimeError(f"DryRun: no mark price set for {symbol}")

    # Test helper: pretend the exchange returns this mark price.
    def set_mark_price(self, symbol: str, price: float) -> None:
        self._mark_prices[symbol] = float(price)

    async def fetch_top_depth_usdt(self, symbol: str, *, levels: int = 5) -> float:
        """Return the test-supplied top-N depth, or the default infinite.

        Bug C4 fix companion of ``fetch_ticker_price``. Tests can pin a
        finite depth via ``set_top_depth`` to exercise the SR-2
        liquidity gate; otherwise dry-run sees "infinite" depth so it
        doesn't reject everything.
        """
        del levels  # we don't slice in dry-run
        if symbol in self._top_depths:
            return self._top_depths[symbol]
        return self._default_top_depth_usdt

    def set_top_depth(self, symbol: str, depth_usdt: float) -> None:
        self._top_depths[symbol] = float(depth_usdt)

    # Test helper: pretend the exchange-side STOP_MARKET fired.
    def simulate_close(self, symbol: str) -> None:
        self._open_positions.pop(symbol, None)


assert isinstance(DryRunExchangeAdapter(), ExchangeAdapter), (
    "DryRunExchangeAdapter does not satisfy ExchangeAdapter protocol"
)


# --------------------------------------------------------------------- #
# Live adapter factory
# --------------------------------------------------------------------- #


def _build_live_adapter(cfg: AppConfig) -> CCXTExchangeAdapter | None:
    """Build a ccxt-backed adapter from environment variables."""
    exchange = cfg.exchanges[0] if cfg.exchanges else "binance"
    key_var = f"{exchange.upper()}_API_KEY"
    sec_var = f"{exchange.upper()}_API_SECRET"
    pass_var = f"{exchange.upper()}_API_PASSPHRASE"
    testnet_var = f"{exchange.upper()}_TESTNET"

    api_key = os.getenv(key_var, "")
    api_secret = os.getenv(sec_var, "")
    api_passphrase = os.getenv(pass_var) or None
    testnet = os.getenv(testnet_var, "true").lower() in ("1", "true", "yes")

    if not api_key or not api_secret:
        logger.error("Missing %s / %s in environment; cannot build live adapter.",
                     key_var, sec_var)
        return None
    try:
        return build_ccxt_adapter(
            exchange_name=exchange,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            testnet=testnet or cfg.paper_trade,
            hedge_mode=cfg.hedge_mode,
        )
    except Exception as e:
        logger.error("Failed to build live adapter: %s", e)
        return None


# --------------------------------------------------------------------- #
# Trailing controller
# --------------------------------------------------------------------- #


@dataclass
class _Tracked:
    position: Position
    state: TrailingState = TrailingState.INIT


@dataclass
class TrailingController:
    """Per-symbol trailing FSM state machine.

    Optionally also drives the rolling-positions controller on each
    kline tick: after the FSM is given the chance to tighten the stop,
    the rolling controller is consulted with the same live bar and may
    add a new same-side leg if all gates pass. The rolling step is a
    pure no-op when ``rolling`` is None or its ``cfg.enabled`` is False.

    Audit (third pass) #2 fix: the rolling controller's depth + vol
    inputs used to be hardcoded class attributes (200k USDT depth,
    5% realised vol). That made the rolling SR-2 liquidity check a
    permanent no-op (the threshold compared to itself) and let the
    sizing path use BTC-grade vol on PEPE-grade alts. We now accept
    optional async ``depth_provider`` / ``vol_provider`` callbacks
    that the App wires to the same paths used by the entry hot path
    (``CCXTExchangeAdapter.fetch_top_depth_usdt`` /
    ``PriceTape.realized_vol_pct``). When None, we fall back to the
    legacy fields so existing tests keep passing.
    """

    fsm: TrailingStopFSM
    atr: ATRCalculator
    executor: CCXTExecutor
    account: AccountState
    health: HealthState
    rolling: RollingController | None = None
    rolling_top5_depth_usdt: float = 200_000.0
    rolling_realized_vol_pct: float = 0.05
    # Optional async hooks. When set, evaluated *per kline tick* so
    # the rolling gate sees the same live values the entry path used.
    rolling_depth_provider: Callable[[str], Awaitable[float]] | None = None
    rolling_vol_provider: Callable[[str], Awaitable[float | None]] | None = None
    rolling_dry_run_fallback_vol_pct: float = 0.04
    _by_symbol: dict[str, _Tracked] = field(default_factory=dict)

    def attach(self, position: Position) -> None:
        self._by_symbol[position.symbol] = _Tracked(position=position)

    def detach(self, symbol: str) -> None:
        self._by_symbol.pop(symbol, None)

    async def on_kline(self, exchange: str, symbol: str, bar: Kline) -> None:
        atr = self.atr.update(exchange, symbol, bar)
        tracked = self._by_symbol.get(symbol)
        if tracked is None or tracked.position.closed:
            return
        next_state, new_stop, reason = self.fsm.tick(
            position=tracked.position,
            current_price=bar.close,
            atr=atr,
            current_state=tracked.state,
        )
        tracked.state = next_state
        if new_stop is not None:
            ok = await self.executor.tighten_hard_stop(tracked.position, new_stop)
            if ok:
                logger.info("trailing %s: %s -> stop %s (atr=%.5f)",
                            symbol, reason, new_stop, atr)
            else:
                # Audit #15: tighten failed. The executor's
                # ``tighten_hard_stop`` returns False in two materially
                # different cases:
                #   (a) replace failed but the OLD stop is back on the
                #       book — the position is still protected, just at
                #       a wider stop than the FSM wanted. We log a
                #       warning and continue.
                #   (b) replace failed AND the restore failed. The
                #       position is **naked** (no resting stop) and
                #       ``stop_order_id`` is None. SR-2 fail-closed
                #       posture demands we close it now rather than
                #       wait for the next bar; we emergency-close at
                #       market and let the position-watcher fire the
                #       close callback on its next poll.
                if tracked.position.stop_order_id is None:
                    logger.critical(
                        "trailing %s: tighten FAILED and restore FAILED "
                        "— position is NAKED, emergency-closing now",
                        symbol,
                    )
                    self.health.last_error = (
                        f"trailing naked emergency-close {symbol}"
                    )
                    try:
                        await self.executor.adapter.market_order(
                            symbol=symbol,
                            side=tracked.position.side.opposite,
                            size=tracked.position.total_size,
                            price=bar.close,
                            reduce_only=True,
                        )
                        tracked.position.closed = True
                        # Audit (third pass) #10: free up the concurrency
                        # slot and engage the same 4h cooldown the
                        # executor uses on stop-failure paths. Without
                        # this, the symbol would still occupy a slot in
                        # ``account.open_positions`` for up to
                        # ``position_watcher_poll_sec`` (default 5s) —
                        # long enough for a fresh high-priority signal
                        # on the same symbol to be allowed in.
                        # ``_on_position_close`` will see the entry was
                        # already removed and just runs its bookkeeping.
                        self.account.open_positions.pop(symbol, None)
                        self.account.set_cooldown(
                            symbol,
                            self.executor.stop_failure_cooldown_sec,
                            int(bar.ts),
                        )
                    except Exception as e:
                        logger.critical(
                            "EMERGENCY CLOSE on naked trailing failed "
                            "for %s: %s — manual intervention required",
                            symbol, e,
                        )
                else:
                    logger.warning(
                        "trailing %s: tighten FAILED (%s); old stop "
                        "restored, position still protected",
                        symbol, reason,
                    )
                    self.health.last_error = (
                        f"trailing tighten restored on {symbol}"
                    )

        # Rolling-positions evaluation. We run it AFTER the trailing tick
        # so that whatever the FSM just did to the stop is the baseline
        # the rolling controller's gate sees. Any failure is logged but
        # never escapes -- the trailing path is the safety-critical one
        # and must not be blocked by the (optional) rolling path.
        if self.rolling is not None and self.rolling.cfg.enabled:
            # Audit (third pass) #2: pull live depth + vol per tick when
            # providers are wired (production); fall back to legacy class
            # attributes otherwise (existing tests). Failures in the
            # providers degrade to legacy values rather than blocking
            # the safety-critical trailing tick on a flaky network.
            top5_depth_usdt = self.rolling_top5_depth_usdt
            realized_vol_pct = self.rolling_realized_vol_pct
            if self.rolling_depth_provider is not None:
                try:
                    top5_depth_usdt = float(
                        await self.rolling_depth_provider(symbol)
                    )
                except Exception as e:
                    logger.warning(
                        "rolling depth_provider failed for %s (%s); "
                        "skipping rolling tick this bar", symbol, e,
                    )
                    return
            if self.rolling_vol_provider is not None:
                try:
                    vol = await self.rolling_vol_provider(symbol)
                except Exception as e:
                    logger.warning(
                        "rolling vol_provider failed for %s (%s); "
                        "skipping rolling tick this bar", symbol, e,
                    )
                    return
                if vol is None or vol <= 0:
                    # Cold tape: refuse to size a leg on a guess.
                    # In dry-run we use the same conservative fallback
                    # the entry path uses so end-to-end tests can run
                    # without a live screener.
                    realized_vol_pct = (
                        self.rolling_dry_run_fallback_vol_pct
                    )
                else:
                    realized_vol_pct = float(vol)
            try:
                decision = await self.rolling.maybe_roll(
                    position=tracked.position,
                    account=self.account,
                    top5_depth_usdt=top5_depth_usdt,
                    realized_vol_pct=realized_vol_pct,
                    now_ms=int(bar.ts),
                )
                if decision.fired:
                    logger.info(
                        "rolling %s: leg added at R=%.2f size=%.4f notional=%.2f",
                        symbol,
                        decision.next_threshold_r or 0.0,
                        decision.new_leg_size or 0.0,
                        decision.new_leg_notional or 0.0,
                    )
                elif decision.reason not in (
                    "disabled", "no_next_threshold", "min_interval_active",
                    "no_unrealised_pnl",
                ):
                    # Other reasons (gate rejected, strategy changed, etc.)
                    # are interesting enough to log at debug.
                    logger.debug("rolling %s: skipped (%s)", symbol, decision.reason)
            except Exception as e:
                logger.exception("rolling on_kline %s failed: %s", symbol, e)
                self.health.last_error = f"rolling:{type(e).__name__}"


# --------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------- #


@dataclass
class App:
    cfg: AppConfig
    state: HealthState = field(default_factory=HealthState)
    dashboard: DashboardState = field(default_factory=DashboardState)
    notifier: Notifier | None = None
    # Bug #2 fix hook: production code uses the adapter's
    # ``fetch_ticker_price``; tests can inject a fixed price (or an
    # exception) to exercise SR-1 without a network round-trip. When None,
    # ``_get_live_quote`` falls back to ``adapter.fetch_ticker_price``.
    quote_provider: Callable[[str], Awaitable[float]] | None = None
    # Bug C4 fix hook: production calls the adapter's
    # ``fetch_top_depth_usdt``; tests can inject a fixed depth (or an
    # exception) without a network round-trip.
    depth_provider: Callable[[str], Awaitable[float]] | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: list[asyncio.Task] = field(default_factory=list)
    # Audit (third pass) #4: fire-and-forget tasks (e.g. Telegram
    # notifications spawned from ``fused_sink`` when
    # ``cfg.telegram_fire_and_forget=True``) MUST hold a strong reference
    # somewhere or CPython 3.11+ may garbage-collect them mid-execution
    # (the asyncio loop only holds weak refs). We track them in a set
    # and wire ``add_done_callback(self._bg_tasks.discard)`` so completed
    # tasks are reaped automatically. Shutdown awaits / cancels the
    # remaining ones so we don't lose pending notifications on SIGTERM.
    _bg_tasks: set[asyncio.Task] = field(default_factory=set)
    _runner: web.AppRunner | None = None
    _dashboard_runner: web.AppRunner | None = None
    _screener: Screener | None = None
    _adapter: ExchangeAdapter | None = None
    _llm_engine: DeepSeekEngine | None = None
    _post_mortem: DelayedPostMortemScheduler | None = None
    _llm_consultor: LLMConsultor | None = None
    _rolling: RollingController | None = None
    _price_tape: PriceTape | None = None
    # Audit (third pass) #1: wiring slots for the safety modules
    # introduced by PR #21. Each is None unless ``App.run`` decides to
    # construct it based on cfg.* flags. Tests can pre-populate these
    # to inject mocks before calling run().
    _persistor: AccountPersistor | None = None
    _sqlite_store: SQLiteAccountStore | None = None
    _regime_filter: RegimeFilter | None = None
    _cluster_map: ClusterMap | None = None
    _cluster_cap_cfg: ClusterCapConfig | None = None
    _kill_switch: KillSwitchWatcher | None = None
    _decision_audit_log: DecisionAuditLog | None = None
    # Phase B.2 — observability slots (None = feature off).
    _metrics: DefaultMetrics | None = None
    _dlq: DeadLetterQueue | None = None
    # Phase A — opportunity-cost penalty pipeline. Built by ``run`` when
    # ``cfg.miss_penalty_enabled`` is True; ``_handle_high_priority``
    # consults ``_reflection`` to honour the suspension window.
    _miss_penalty: MissPenaltyEngine | None = None
    _reject_scorer: RejectReasonScorer | None = None
    _threshold_tuner: ThresholdAutoTuner | None = None
    _reflection: ReflectionModeController | None = None
    # Phase 5 — LLM cache / budget manager / pre-rate worker. Each is
    # constructed in ``run`` only when the matching cfg flag is True;
    # otherwise the daemon's behaviour is byte-for-byte identical to
    # v1.0. Tests can pre-populate any of these slots before calling
    # ``run`` to inject mocks.
    _llm_cache: LLMCache | None = None
    _token_budget_manager: TokenBudgetManager | None = None
    _llm_pre_rater: LLMPreRater | None = None
    # Phase B.6 sister deliverable — KOL history slots. Both None when
    # cfg.kol_history_enabled is False; ``run`` constructs them and
    # plumbs the analyzer into both ScoreFuser and
    # DelayedPostMortemScheduler.
    _kol_history_store: KOLHistoryStore | None = None
    _kol_analyzer: HistoricalAnalyzer | None = None
    # R3 — production_rules.json hot loader. None = feature off.
    # Wired in ``run`` when ``cfg.production_rules_enabled`` is True;
    # the background ``production_rules_reload_worker`` polls it on
    # ``cfg.production_rules_reload_interval_sec``. Subscribers
    # (R6: fuser / risk gate quadrant lookups) will read the loader
    # directly so they always see the latest trainer output without
    # a daemon restart.
    _production_rules_loader: ProductionRulesLoader | None = None

    async def run(self) -> None:
        self.state.started_at = time.time()
        mode = self._mode_label()
        logger.info("Altcoin Agent V1.0 starting (mode=%s)", mode)

        if self.notifier is None:
            self.notifier = build_default_notifier()
        logger.info("Notifier: %s", self.notifier.name)

        # ----- Phase B.2 — observability bootstrap -----
        # Order matters: structured logging first so subsequent INFO
        # lines emitted from the rest of run() are JSON; then metrics
        # registry (consumed by the metrics endpoint and observed at
        # mutation hooks); then DLQ. All three are opt-in; when off
        # the daemon's behaviour is byte-for-byte identical to v1.0.
        if self.cfg.structured_logging_enabled:
            try:
                configure_structured_logging(service="altcoin-agent")
                logger.info(
                    "structured logging enabled (JSON formatter active)",
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "structured logging setup failed (swallowed): %s", e,
                )

        # Phase B.6: OpenTelemetry tracer. Initialised second so the
        # provider's first span (the structured log "started" line)
        # already lands on the JSON formatter. configure_tracing
        # itself does NOT raise when OTel is missing — it logs a
        # warning and returns a disabled tracer.
        if self.cfg.tracing_enabled:
            try:
                configure_tracing(
                    service_name=self.cfg.tracing_service_name,
                    otlp_endpoint=(
                        self.cfg.tracing_otlp_endpoint or None
                    ),
                    otlp_insecure=self.cfg.tracing_otlp_insecure,
                    console_exporter=self.cfg.tracing_console,
                    sampler_arg=self.cfg.tracing_sampler_ratio,
                    extra_resource_attrs={
                        "deployment.environment": self.cfg.tracing_environment,
                        "altcoin_agent.mode": mode,
                    },
                )
                logger.info(
                    "OpenTelemetry tracer enabled (service=%s, "
                    "endpoint=%s, console=%s, sampler=%.2f)",
                    self.cfg.tracing_service_name,
                    self.cfg.tracing_otlp_endpoint or "<none>",
                    self.cfg.tracing_console,
                    self.cfg.tracing_sampler_ratio,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "tracing setup failed (swallowed): %s", e,
                )

        if self.cfg.metrics_enabled and self._metrics is None:
            try:
                self._metrics = build_default_registry()
                logger.info(
                    "metrics registry enabled (~30 metrics under "
                    "altcoin_agent_*)",
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "metrics registry setup failed (swallowed): %s", e,
                )
                self._metrics = None

        if self.cfg.dlq_enabled and self._dlq is None:
            try:
                self._dlq = DeadLetterQueue(path=Path(self.cfg.dlq_path))
                logger.info("DLQ enabled at %s", self.cfg.dlq_path)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "DLQ setup failed (swallowed): %s", e,
                )
                self._dlq = None

        # ----- adapter selection -----
        if self.cfg.dry_run:
            self._adapter = DryRunExchangeAdapter()
        else:
            adapter = _build_live_adapter(self.cfg)
            if adapter is None:
                logger.error("Live mode requires valid API keys. Aborting.")
                raise SystemExit(2)
            self._adapter = adapter

        executor = CCXTExecutor(adapter=self._adapter,
                                exchange_name=self.cfg.exchanges[0])
        sizer = PositionSizer()
        gate = RiskGate(sizer, RiskGateConfig(
            min_liquidity_usdt=self.cfg.min_liquidity_usdt,
        ))
        atr = ATRCalculator()
        # Anti-chase / vol-kill price tape (low-latency hot-path defence).
        # Fed by the screener's kline wrapper below and consulted by the
        # risk gate before any networked check. See price_tape.py for
        # the rationale.
        self._price_tape = PriceTape(
            cfg=PriceTapeConfig(
                anti_chase_window_ms=self.cfg.anti_chase_window_ms,
                anti_chase_max_move_pct=self.cfg.anti_chase_max_move_pct,
                vol_kill_window_ms=self.cfg.vol_kill_window_ms,
                vol_kill_range_pct=self.cfg.vol_kill_range_pct,
                max_samples_per_symbol=self.cfg.price_tape_max_samples,
            )
        )
        account = AccountState(
            equity_usdt=self.cfg.initial_equity_usdt,
            starting_equity_today_usdt=self.cfg.initial_equity_usdt,
            rollover_anchor_utc_hour=self.cfg.rollover_anchor_utc_hour,
        )
        # Audit (third pass) #1: AccountState persistence (audit #12).
        # Restore from disk BEFORE the first ``maybe_roll_over_day`` call
        # so that the day stamp loaded from disk is what drives the
        # rollover decision (a 24h-restart should *not* re-stamp the
        # equity baseline at the new equity level — that would erase
        # yesterday's drawdown). When restore fails, ``account`` keeps
        # its constructor defaults and we proceed cleanly.
        #
        # Phase B.3.1: ``cfg.persistence_backend`` selects between the
        # JSON-file ``AccountPersistor`` and the SQLite-WAL
        # ``SQLiteAccountStore``. Both expose ``save(account)`` /
        # ``restore_into(account)`` with identical semantics so the
        # change-listener hook below is backend-agnostic. When
        # ``backend == "sqlite"`` the SQLite store is wired to BOTH
        # ``self._sqlite_store`` (for tests + history queries) AND
        # ``self._persistor`` (single var for the change listener).
        if self.cfg.account_persistence_enabled and self._persistor is None:
            backend = (self.cfg.persistence_backend or "json").lower()
            if backend == "sqlite":
                if self._sqlite_store is None:
                    self._sqlite_store = SQLiteAccountStore(
                        path=Path(self.cfg.account_persistence_sqlite_path),
                    )
                self._persistor = self._sqlite_store  # type: ignore[assignment]
                logger.info(
                    "AccountState persistence backend: sqlite (path=%s)",
                    self.cfg.account_persistence_sqlite_path,
                )
            else:
                self._persistor = AccountPersistor(
                    path=Path(self.cfg.account_persistence_path),
                )
                if backend != "json":
                    logger.warning(
                        "unknown persistence_backend=%r; falling back to json",
                        self.cfg.persistence_backend,
                    )
        if self._persistor is not None:
            restored = self._persistor.restore_into(account)
            if restored:
                logger.info(
                    "AccountState restored: equity=%.2f, "
                    "today_pnl=%.2f, stops_today=%d, halted=%s",
                    account.equity_usdt,
                    account.realized_pnl_today_usdt,
                    account.daily_stoploss_hits,
                    account.global_trading_halted,
                )
            else:
                logger.info(
                    "AccountState persistence: no prior snapshot "
                    "(starting fresh)",
                )
            # Phase B.1.3: hook the persistor into AccountState's
            # change-notification channel. From this point forward
            # every ``set_cooldown`` / ``halt`` / ``record_pnl`` /
            # ``maybe_roll_over_day`` call snapshots the new state
            # to disk synchronously. The mutators all live on the
            # AccountState itself so adapters / executors / kill-
            # switch / future modules don't need to know about the
            # persistor — they just call the mutator.
            account.register_change_listener(
                lambda a, _p=self._persistor: _p.save(a),
            )

        # Bug #3 fix: stamp the boot day so the first real flip resets
        # daily counters cleanly. Without this, ``maybe_roll_over_day``
        # called from any path (the worker, the hot path) would treat
        # boot as a "first ever stamp" and never detect day 1 -> day 2.
        account.maybe_roll_over_day()
        # Persist the (possibly rollover-touched) snapshot now so a
        # later crash before any trade still recovers correctly.
        if self._persistor is not None:
            self._persistor.save(account)

        self.dashboard.health = self.state
        self.dashboard.account = account
        self.dashboard.rules_path = Path(self.cfg.dynamic_rules_path)

        # Audit (third pass) #1: build the gate-side safety modules.
        # All of them are optional kwargs to RiskGate.evaluate — when
        # None the gate behaves exactly as before, so existing tests
        # keep passing without modification. Tests can pre-populate
        # ``self._regime_filter`` etc. before calling run() to inject
        # custom configurations.
        if self.cfg.regime_filter_enabled and self._regime_filter is None:
            self._regime_filter = RegimeFilter(
                cfg=RegimeFilterConfig(
                    reference_symbol=self.cfg.regime_reference_symbol,
                    btc_window_ms=self.cfg.regime_btc_window_ms,
                    btc_drop_block_long_pct=(
                        self.cfg.regime_btc_drop_block_long_pct
                    ),
                    btc_rip_block_short_pct=(
                        self.cfg.regime_btc_rip_block_short_pct
                    ),
                    min_samples=self.cfg.regime_min_samples,
                ),
            )
        if self.cfg.cluster_cap_enabled and self._cluster_map is None:
            self._cluster_map = ClusterMap(
                explicit={
                    str(k).upper(): str(v)
                    for k, v in self.cfg.cluster_map.items()
                },
            )
            self._cluster_cap_cfg = ClusterCapConfig(
                enabled=True,
                max_per_cluster=self.cfg.cluster_max_per_cluster,
            )
        if self.cfg.decision_audit_log_enabled and self._decision_audit_log is None:
            self._decision_audit_log = DecisionAuditLog(
                path=Path(self.cfg.decision_audit_log_path),
            )

        # Phase A — miss-penalty + reflection-mode pipeline. Default
        # OFF; operator opts in via ``miss_penalty_enabled`` once the
        # decision-log file has at least 24h of rejections to audit.
        # All four objects share the same state directory so the
        # operator only has to back up one folder.
        if self.cfg.miss_penalty_enabled and self._reflection is None:
            self._wire_miss_penalty_pipeline()

        # ----- reconciler (SR-2) -----
        rec_report = await Reconciler(
            exchange_name=self.cfg.exchanges[0], adapter=self._adapter,
        ).run(account)
        logger.info("Reconciler: %s", rec_report)
        self.state.reconciliation_complete = account.reconciliation_complete

        trailing = TrailingController(
            fsm=TrailingStopFSM(), atr=atr, executor=executor,
            account=account, health=self.state,
        )

        # Position-watcher: detects exchange-side closures (STOP_MARKET fired
        # or manual close) so we can update PnL, daily DD, consec losses,
        # detach trailing, and notify Telegram. Without this, ``Position.closed``
        # is never set to True and the system keeps acting on a phantom
        # position. (Bug #1.)
        async def _watcher_close_cb(position: Position, reason: str) -> None:
            await self._on_position_close(
                position=position,
                reason=reason,
                trailing=trailing,
                account=account,
            )

        position_watcher = PositionWatcher(
            adapter=self._adapter,
            account=account,
            on_close=_watcher_close_cb,
            poll_interval_sec=self.cfg.position_watcher_poll_sec,
            miss_threshold=self.cfg.position_watcher_miss_threshold,
        )

        # ----- queue + components -----
        signal_q: asyncio.Queue[SignalEvent] = asyncio.Queue(maxsize=1000)
        kline_q: asyncio.Queue[tuple[str, str, Kline]] = asyncio.Queue(maxsize=1000)
        llm_q: asyncio.Queue[SignalEvent] = asyncio.Queue(maxsize=self.cfg.llm_queue_max)

        # Per-symbol rolling cache of rule events / funding / OI for SMC
        # context reconstruction. window_sec mirrors the fuser default so the
        # LLM sees the same horizon the fuser is currently scoring on.
        recent_cache = RecentSignalsCache(window_sec=FuserConfig().window_sec)
        candidate_gate = CandidateGate(
            cooldown_sec=self.cfg.llm_consult_cooldown_sec,
        )

        async def screener_sink(ev: SignalEvent) -> None:
            self.state.last_signal_ts = time.time()
            self.state.rule_event_count += 1
            recent_cache.add_signal(ev)
            with suppress(asyncio.QueueFull):
                signal_q.put_nowait(ev)
            # Fan-out to LLM consult queue when this event is worth the
            # token spend. Failures here NEVER block rule scoring.
            if candidate_gate.should_consult(ev):
                try:
                    llm_q.put_nowait(ev)
                    candidate_gate.mark_consulted(ev.symbol, ev.ts)
                except asyncio.QueueFull:
                    self.state.llm_consults_skipped += 1

        self._screener = Screener(
            exchanges=self.cfg.exchanges,
            symbols=self.cfg.symbols,
            sink=screener_sink,
            timeframes=self.cfg.timeframes,
        )
        original_on_kline = self._screener.on_kline
        original_on_funding = self._screener.on_funding
        original_on_oi = self._screener.on_oi

        async def on_kline_wrapper(exchange: str, symbol: str, bar: Kline) -> None:
            # Feed the price tape on every WS kline update (including
            # intra-bar). This is the data source for the anti-chase /
            # vol-kill gates; missing it would make those gates no-ops.
            if self._price_tape is not None:
                self._price_tape.observe(symbol, bar.close, int(bar.ts))
            # Audit (third pass) #1: feed the regime filter from the
            # same stream. ``observe`` ignores non-reference symbols, so
            # this fans out for free across whatever symbol set the
            # screener is following.
            if self._regime_filter is not None:
                self._regime_filter.observe(symbol, bar.close, int(bar.ts))
            with suppress(asyncio.QueueFull):
                kline_q.put_nowait((exchange, symbol, bar))
            await original_on_kline(exchange, symbol, bar)

        async def on_funding_wrapper(exchange: str, snap: FundingSnapshot) -> None:
            recent_cache.add_funding(snap)
            await original_on_funding(exchange, snap)

        async def on_oi_wrapper(exchange: str, snap: OISnapshot) -> None:
            recent_cache.add_oi(snap)
            await original_on_oi(exchange, snap)

        self._screener.on_kline = on_kline_wrapper      # type: ignore[method-assign]
        self._screener.on_funding = on_funding_wrapper  # type: ignore[method-assign]
        self._screener.on_oi = on_oi_wrapper            # type: ignore[method-assign]

        # ----- fused-signal handling -----
        async def fused_sink(sig: FusedSignal) -> None:
            self.state.high_priority_count += 1
            payload = sig.as_dict()
            payload["ts_ms"] = int(time.time() * 1000)
            self.dashboard.push_signal(payload)
            logger.info("HIGH PRIORITY: %s", payload)
            # Telegram notification is a 100–800 ms HTTPS round-trip.
            # In altcoin pump scenarios that's enough time for the
            # mark to drift past the SR-1 cap. When
            # ``cfg.telegram_fire_and_forget`` is True (default) we
            # spawn the notify as a background task so the order
            # placement starts immediately. Failures are swallowed by
            # the notifier itself.
            if self.cfg.telegram_fire_and_forget:
                # Audit (third pass) #4: keep a strong reference so
                # CPython 3.11+ can't GC the task mid-flight.
                bg = asyncio.create_task(
                    self._safe_notify_signal(payload),
                    name="notify_signal_bg",
                )
                self._bg_tasks.add(bg)
                bg.add_done_callback(self._bg_tasks.discard)
            else:
                await self.notifier.signal(payload)
            await self._handle_high_priority(
                sig=sig, gate=gate, executor=executor,
                trailing=trailing, account=account,
            )

        # Phase B.6 sister deliverable — KOL historical hit-rate analyzer.
        # Built BEFORE the fuser so the fuser can take a reference to it.
        # The store is shared with the post-mortem scheduler below so
        # every closed position feeds back into the same counters.
        if self.cfg.kol_history_enabled and self._kol_analyzer is None:
            try:
                if self._kol_history_store is None:
                    self._kol_history_store = KOLHistoryStore(
                        path=Path(self.cfg.kol_history_path),
                    )
                self._kol_analyzer = HistoricalAnalyzer(
                    store=self._kol_history_store,
                    config=HistoricalAnalyzerConfig(
                        min_samples=self.cfg.kol_history_min_samples,
                        strong_bound=self.cfg.kol_history_strong_bound,
                        weak_bound=self.cfg.kol_history_weak_bound,
                        conf_lift_max=self.cfg.kol_history_conf_lift_max,
                        conf_drop_max=self.cfg.kol_history_conf_drop_max,
                    ),
                )
                logger.info(
                    "KOL history analyzer enabled (path=%s, "
                    "%d authors loaded, min_samples=%d)",
                    self.cfg.kol_history_path,
                    len(self._kol_history_store),
                    self.cfg.kol_history_min_samples,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "KOL history analyzer setup failed (swallowed): %s", e,
                )
                self._kol_history_store = None
                self._kol_analyzer = None

        # Resolve dynamic rules path: cfg override -> env -> fuser default.
        fuser_cfg_kwargs: dict[str, Any] = {}
        if self.cfg.dynamic_rules_path:
            fuser_cfg_kwargs["dynamic_rules_path"] = Path(
                self.cfg.dynamic_rules_path,
            )
        fuser = ScoreFuser(
            sink=fused_sink,
            config=FuserConfig(**fuser_cfg_kwargs),
            historical_analyzer=self._kol_analyzer,
        )

        # ----- Rolling-positions controller (optional) -----
        # Default OFF: rolling adds same-side exposure to a winning
        # position, scaling both upside and the path-dependent downside
        # with each new leg. It must be turned on deliberately.
        # When enabled, it is driven from the trailing worker on every
        # kline tick: see TrailingController.on_kline.
        if self.cfg.rolling_enabled:
            rolling_cfg = RollingConfig(
                enabled=True,
                trigger_r_levels=self.cfg.rolling_trigger_r_levels,
                unrealized_pnl_ratio=self.cfg.rolling_unrealized_pnl_ratio,
                leg_stop_pct=self.cfg.rolling_leg_stop_pct,
                max_legs_per_symbol=self.cfg.rolling_max_legs_per_symbol,
                min_interval_sec=self.cfg.rolling_min_interval_sec,
                auto_disable_on_failure=(
                    self.cfg.rolling_auto_disable_on_failure
                ),
                require_strategy_min_score=(
                    self.cfg.rolling_require_strategy_min_score
                ),
                require_min_rule_score=(
                    self.cfg.rolling_require_min_rule_score
                ),
            )

            async def _rolling_quote(symbol: str) -> float:
                # Reuse the same Bug #2 fail-closed live-quote path the
                # entry hot path uses: tests can swap in
                # ``app.quote_provider``, production uses the adapter.
                return await self._get_live_quote(symbol)

            async def _notify_roll(payload: dict[str, Any]) -> None:
                # Push to dashboard + notifier; failures do not propagate.
                with suppress(Exception):
                    self.dashboard.push_signal({
                        **payload, "kind": "rolling_leg_added",
                    })
                with suppress(Exception):
                    if self.notifier is not None:
                        await self.notifier.signal({
                            "kind": "rolling_leg_added", **payload,
                        })

            async def _notify_roll_error(
                msg: str, payload: dict[str, Any] | None = None,
            ) -> None:
                with suppress(Exception):
                    if self.notifier is not None:
                        await self.notifier.error(msg, payload=payload)
                self.state.last_error = msg

            self._rolling = RollingController(
                cfg=rolling_cfg,
                sizer=sizer,
                gate=gate,
                executor=executor,
                fuser=fuser,
                quote_provider=_rolling_quote,
                notify_roll=_notify_roll,
                notify_error=_notify_roll_error,
            )
            trailing.rolling = self._rolling
            # Audit (third pass) #2: wire the same live depth + vol
            # providers the entry hot path uses so the rolling SR-2
            # liquidity gate isn't comparing the threshold to itself
            # and the sizing path doesn't apply BTC-grade vol to alts.
            # The dry-run vol fallback mirrors ``cfg.dry_run_fallback_vol_pct``
            # so dry-run end-to-end tests stay deterministic.
            async def _rolling_depth(sym: str) -> float:
                return await self._fetch_top_depth_usdt(sym)

            async def _rolling_vol(sym: str) -> float | None:
                if self._price_tape is None:
                    return None
                return self._price_tape.realized_vol_pct(
                    symbol=sym,
                    window_ms=self.cfg.vol_kill_window_ms,
                )

            trailing.rolling_depth_provider = _rolling_depth
            trailing.rolling_vol_provider = _rolling_vol
            trailing.rolling_dry_run_fallback_vol_pct = (
                self.cfg.dry_run_fallback_vol_pct
            )
            # Keep the legacy fields populated as a last-resort fallback
            # if a provider raises. ``min_liquidity_usdt`` is the gate
            # threshold so the legacy fallback is at least sane (it'll
            # let the rolling gate's own SR-2 check enforce the bar)
            # rather than the previous identity-equal placeholder.
            trailing.rolling_top5_depth_usdt = self.cfg.min_liquidity_usdt
            logger.info(
                "Rolling positions ENABLED: trigger_r=%s ratio=%.2f "
                "leg_stop=%.3f max_legs=%d",
                rolling_cfg.trigger_r_levels,
                rolling_cfg.unrealized_pnl_ratio,
                rolling_cfg.leg_stop_pct,
                rolling_cfg.max_legs_per_symbol,
            )
        else:
            self._rolling = None
            logger.info("Rolling positions disabled (cfg.rolling_enabled=False)")

        # ----- LLM engine + online learning loop -----
        # The engine reads DEEPSEEK_API_KEY from env. If unset, we don't even
        # build it; the LLM worker will short-circuit and the system stays
        # in rule-only mode (a documented degraded mode).
        if os.getenv("DEEPSEEK_API_KEY"):
            self._llm_engine = DeepSeekEngine()
            logger.info("DeepSeek engine initialised (model=%s)",
                        self._llm_engine.model)
        else:
            self._llm_engine = None
            logger.info("DEEPSEEK_API_KEY not set; running in rule-only mode "
                        "(LLM consults and post-mortems disabled).")

        # ----- Phase 5: cache + budget manager + pre-rate worker -----
        # All three are opt-in. Cache and budget manager attach as
        # optional fields on the existing engine so legacy callers
        # that didn't pass ``cache=`` or ``budget_manager=`` keep
        # working unchanged. The pre-rater spawns a separate task in
        # ``_start_background_tasks`` further down; here we just
        # construct it.
        if self._llm_engine is not None and self.cfg.llm_cache_enabled \
                and self._llm_cache is None:
            try:
                self._llm_cache = LLMCache(
                    max_entries=self.cfg.llm_cache_max_entries,
                    default_ttl_sec=self.cfg.llm_cache_ttl_sec,
                    state_path=self.cfg.llm_cache_path,
                )
                self._llm_engine.cache = self._llm_cache
                logger.info(
                    "LLMCache enabled: ttl=%ds max=%d path=%s "
                    "(60-80%% token saving expected)",
                    self.cfg.llm_cache_ttl_sec,
                    self.cfg.llm_cache_max_entries,
                    self.cfg.llm_cache_path,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "LLMCache setup failed (swallowed): %s", e,
                )
                self._llm_cache = None

        if self._llm_engine is not None \
                and self.cfg.llm_budget_manager_enabled \
                and self._token_budget_manager is None:
            try:
                self._token_budget_manager = TokenBudgetManager(
                    monthly_budget=self.cfg.llm_budget_monthly_tokens,
                    state_path=self.cfg.llm_budget_state_path,
                )
                self._llm_engine.budget_manager = self._token_budget_manager
                logger.info(
                    "TokenBudgetManager enabled: monthly=%d "
                    "(mode=%s, used=%d/%d)",
                    self.cfg.llm_budget_monthly_tokens,
                    self._token_budget_manager.mode().value,
                    self._token_budget_manager.state.used,
                    self._token_budget_manager.state.budget,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "TokenBudgetManager setup failed (swallowed): %s", e,
                )
                self._token_budget_manager = None

        # The pre-rater needs both the engine AND the cache (it calls
        # judge() expecting cache writes). Without the cache, the
        # worker just burns tokens on every candidate; gate it on
        # both flags so misconfigurations don't waste budget.
        if (
            self._llm_engine is not None
            and self.cfg.llm_pre_rate_enabled
            and self._llm_cache is not None
            and self._llm_pre_rater is None
        ):
            try:
                self._llm_pre_rater = LLMPreRater(
                    engine=self._llm_engine,
                    cache=self._llm_cache,
                    budget_manager=self._token_budget_manager,
                    prerate_min_score=self.cfg.llm_pre_rate_min_score,
                    queue_maxsize=self.cfg.llm_pre_rate_queue_max,
                )
                logger.info(
                    "LLMPreRater enabled: min_score=%.1f queue_max=%d "
                    "(A-quadrant only; expected ~2.25M tokens/month)",
                    self.cfg.llm_pre_rate_min_score,
                    self.cfg.llm_pre_rate_queue_max,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "LLMPreRater setup failed (swallowed): %s", e,
                )
                self._llm_pre_rater = None
        elif self.cfg.llm_pre_rate_enabled and self._llm_cache is None:
            logger.warning(
                "LLM pre-rate enabled but cache disabled; skipping "
                "pre-rater (cache is required to avoid token waste)",
            )

        # RuleStore: shared by ScoreFuser (read via dynamic_rules.json mtime
        # reload) and the post-mortem scheduler (write side). The path
        # follows the same precedence as RuleIndex so both ends always see
        # the same file.
        rules_json_path = (
            self.cfg.dynamic_rules_path
            or os.getenv("DYNAMIC_RULES_JSON")
            or str(fuser.rule_index.json_path)
        )
        rule_store = RuleStore(json_path=rules_json_path)

        self._post_mortem = DelayedPostMortemScheduler(
            store=rule_store,
            engine=self._llm_engine,
            delay_sec=self.cfg.post_mortem_delay_sec,
            historical_analyzer=self._kol_analyzer,
        )

        if self._llm_engine is not None:
            self._llm_consultor = LLMConsultor(
                engine=self._llm_engine,
                fuser=fuser,
                cache=recent_cache,
                cookies=cookie_jar_from_env(),
                proxy=proxy_config_from_env(),
            )
        else:
            self._llm_consultor = None

        # ----- HTTP server (healthz + dashboard, separate apps) -----
        # Healthz must stay reachable from container probes (0.0.0.0 by
        # default) but it MUST NOT carry the dashboard or the JSON APIs:
        # those expose positions, orders, and learnt rules. We therefore
        # bind two separate aiohttp sites:
        #
        #   * healthz_app  -> healthz_bind:healthz_port (default 0.0.0.0:8080)
        #     returns only liveness/uptime; no operational data.
        #
        #   * dashboard_app -> dashboard_bind:dashboard_port (default
        #     127.0.0.1:8081). When ``dashboard_token`` is set, every
        #     request must carry ``X-Auth-Token: <token>``.
        #
        # Bug C1 fail-closed: refuse to bind the dashboard on a
        # non-loopback address without a token. Operators that need
        # remote access SSH-tunnel to 127.0.0.1:8081 or set a token in
        # their .env file.
        health_app = await make_health_app(
            self.state,
            metrics_registry=self._metrics.registry if self._metrics else None,
        )
        self._runner = web.AppRunner(health_app)
        await self._runner.setup()
        health_site = web.TCPSite(
            self._runner, self.cfg.healthz_bind, self.cfg.healthz_port,
        )
        await health_site.start()
        logger.info("Health endpoint live: http://%s:%d/healthz",
                    self.cfg.healthz_bind, self.cfg.healthz_port)

        if self.cfg.dashboard_enabled:
            token = (self.cfg.dashboard_token
                     or os.getenv("DASHBOARD_TOKEN", "")).strip() or None
            bind = self.cfg.dashboard_bind
            is_loopback = bind in ("127.0.0.1", "localhost", "::1")
            if not is_loopback and token is None:
                # Fail-closed: do NOT publish positions/rules to the
                # network without auth. Operators see the message and
                # either tunnel to loopback or set DASHBOARD_TOKEN.
                raise SystemExit(
                    "Dashboard refuses to bind on a non-loopback address "
                    f"({bind!r}) without a token. Set DASHBOARD_TOKEN= in "
                    "the environment, or leave dashboard_bind=127.0.0.1 "
                    "and use an SSH tunnel."
                )
            dashboard_app = make_dashboard_app(
                self.dashboard, mode_label=mode, auth_token=token,
            )
            self._dashboard_runner = web.AppRunner(dashboard_app)
            await self._dashboard_runner.setup()
            dash_port = (self.cfg.dashboard_port
                         or self.cfg.healthz_port + 1)
            dash_site = web.TCPSite(
                self._dashboard_runner, bind, dash_port,
            )
            await dash_site.start()
            logger.info(
                "Dashboard live:        http://%s:%d/dashboard  (auth=%s)",
                bind, dash_port,
                "token" if token else "none",
            )

        # ----- workers -----
        async def fuse_worker() -> None:
            try:
                while not self._stop_event.is_set():
                    try:
                        ev = await asyncio.wait_for(signal_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        await fuser.on_rule_signal(ev)
                    except Exception as e:
                        logger.exception("fuser failed on event: %s", e)
                        self.state.last_error = f"fuser:{type(e).__name__}"
                        with suppress(Exception):
                            await self.notifier.error(
                                f"fuser failed: {e}", payload={"event": ev.as_dict()},
                            )
            except asyncio.CancelledError:
                pass
            finally:
                self.state.fuser_alive = False

        async def trailing_worker() -> None:
            try:
                while not self._stop_event.is_set():
                    try:
                        exchange, symbol, bar = await asyncio.wait_for(
                            kline_q.get(), timeout=1.0,
                        )
                    except asyncio.TimeoutError:
                        continue
                    try:
                        await trailing.on_kline(exchange, symbol, bar)
                    except Exception as e:
                        logger.exception("trailing failed on bar: %s", e)
            except asyncio.CancelledError:
                pass

        async def screener_worker() -> None:
            try:
                await self._screener.run()  # type: ignore[union-attr]
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("screener failed: %s", e)
                self.state.last_error = f"screener:{type(e).__name__}"
                with suppress(Exception):
                    await self.notifier.error(f"screener failed: {e}")
            finally:
                self.state.screener_alive = False

        async def llm_worker() -> None:
            """Drain `llm_q`, consult the LLM, feed verdicts into the fuser.

            Runs only when an engine is configured. Each consult is wrapped
            in suppress(Exception) so a misbehaving social/LLM call cannot
            poison the trading bus.
            """
            if self._llm_consultor is None:
                # No engine -> drain forever to keep the queue from filling.
                while not self._stop_event.is_set():
                    try:
                        ev = await asyncio.wait_for(llm_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    self.state.llm_consults_skipped += 1
                    del ev
                return
            try:
                while not self._stop_event.is_set():
                    try:
                        ev = await asyncio.wait_for(llm_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        verdict = await self._llm_consultor.consult(ev)
                        if verdict is not None:
                            self.state.llm_consults += 1
                        else:
                            self.state.llm_consults_skipped += 1
                    except Exception as e:
                        logger.exception("llm_worker consult failed: %s", e)
                        self.state.llm_consults_skipped += 1
                        self.state.last_error = f"llm:{type(e).__name__}"
            except asyncio.CancelledError:
                pass

        async def position_watcher_worker() -> None:
            try:
                await position_watcher.run(self._stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("position_watcher failed: %s", e)
                self.state.last_error = f"position_watcher:{type(e).__name__}"

        async def daily_rollover_worker() -> None:
            """Bug #3 fix: poll for the UTC-day flip and reset daily-scoped
            counters when it happens. Without this worker the daily
            drawdown breaker and 3-strike rule would be one-way latches:
            once tripped, they would never re-arm.

            Polls every ``rollover_poll_sec``; the actual reset is
            idempotent so missing a beat just delays the reset by one
            tick. We also call the same primitive from the hot path
            (``_handle_high_priority``) for defence-in-depth in case
            this worker is suspended (e.g., long GC pause)."""
            try:
                while not self._stop_event.is_set():
                    try:
                        if account.maybe_roll_over_day():
                            logger.info(
                                "Daily rollover: starting_equity=%.2f, "
                                "previous_pnl=%.2f, stops_yesterday=%d",
                                account.starting_equity_today_usdt,
                                account.realized_pnl_today_usdt,
                                account.daily_stoploss_hits,
                            )
                            # Audit (third pass) #1: persist the new
                            # day's stamp so a restart in the first
                            # minute of UTC midnight doesn't replay
                            # the rollover.
                            if self._persistor is not None:
                                with suppress(Exception):
                                    self._persistor.save(account)
                            with suppress(Exception):
                                await self.notifier.error(
                                    "daily rollover applied",
                                    payload={
                                        "trading_day": account.last_rollover_date_utc,
                                        "starting_equity_usdt": (
                                            account.starting_equity_today_usdt
                                        ),
                                    },
                                )
                    except Exception as e:
                        logger.exception("daily_rollover failed: %s", e)
                        self.state.last_error = (
                            f"daily_rollover:{type(e).__name__}"
                        )
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=self.cfg.rollover_poll_sec,
                        )
                    except asyncio.TimeoutError:
                        continue
            except asyncio.CancelledError:
                pass

        self._tasks.append(asyncio.create_task(fuse_worker(), name="fuse_worker"))
        self._tasks.append(asyncio.create_task(trailing_worker(), name="trailing_worker"))
        self._tasks.append(asyncio.create_task(screener_worker(), name="screener_worker"))
        self._tasks.append(asyncio.create_task(llm_worker(), name="llm_worker"))
        self._tasks.append(asyncio.create_task(
            position_watcher_worker(), name="position_watcher_worker",
        ))
        self._tasks.append(asyncio.create_task(
            daily_rollover_worker(), name="daily_rollover_worker",
        ))

        # Phase A — daily miss-penalty audit + reflection trigger.
        # Default OFF (cfg.miss_penalty_enabled) so existing dry-run
        # tests don't grow a new background task they didn't ask for.
        if self._reflection is not None:
            self._tasks.append(asyncio.create_task(
                self._miss_penalty_worker(account),
                name="miss_penalty_worker",
            ))

        # Audit (third pass) #1 + #25: KillSwitchWatcher. Only built
        # when enabled. Touches the sentinel file => account.halt(reason)
        # so the gate's existing global-halt check refuses every entry
        # (LIVE & paper). Removing the file releases the halt only if
        # it came from the kill switch (manual halts stay sticky).
        if self.cfg.kill_switch_enabled and self._kill_switch is None:
            async def _ks_notify_halt(reason: str) -> None:
                with suppress(Exception):
                    if self.notifier is not None:
                        await self.notifier.error(
                            f"KILL SWITCH ENGAGED: {reason}",
                            payload={"halt": True},
                        )

            async def _ks_notify_release(reason: str) -> None:
                with suppress(Exception):
                    if self.notifier is not None:
                        await self.notifier.error(
                            f"KILL SWITCH RELEASED: {reason}",
                            payload={"halt": False},
                        )

            self._kill_switch = KillSwitchWatcher(
                cfg=KillSwitchConfig(
                    enabled=True,
                    path=Path(self.cfg.kill_switch_path),
                    poll_sec=self.cfg.kill_switch_poll_sec,
                ),
                account=account,
                on_halt=_ks_notify_halt,
                on_release=_ks_notify_release,
            )
            ks = self._kill_switch

            async def kill_switch_worker() -> None:
                try:
                    await ks.run(self._stop_event)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("kill_switch_worker failed: %s", e)
                    self.state.last_error = (
                        f"kill_switch:{type(e).__name__}"
                    )

            self._tasks.append(asyncio.create_task(
                kill_switch_worker(), name="kill_switch_worker",
            ))

        # Phase 5 — start the LLM pre-rate background worker. It runs
        # for as long as the daemon does; ``_shutdown`` calls
        # ``stop()`` on the rater so the queue drains cleanly. The
        # rater is its own task lifetime (it manages an asyncio.Queue
        # internally), so we don't append it to ``self._tasks`` —
        # ``stop()`` is the canonical lifecycle hook.
        if self._llm_pre_rater is not None:
            try:
                await self._llm_pre_rater.start()
                logger.info(
                    "LLMPreRater background worker started",
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "LLMPreRater start failed (swallowed): %s", e,
                )

        # R3 — production_rules.json hot reload worker.
        #
        # Built only when the operator opts in. We construct the
        # loader here (just-in-time) so unit tests that bypass
        # ``run`` don't need the trainer state dir to exist. The
        # worker runs forever; ``_stop_event`` cancels it during
        # graceful shutdown.
        if (
            self.cfg.production_rules_enabled
            and self._production_rules_loader is None
        ):
            self._production_rules_loader = ProductionRulesLoader(
                path=os.path.join(
                    self.cfg.production_rules_dir,
                    "production_rules.json",
                ),
                min_check_interval_sec=(
                    self.cfg.production_rules_reload_interval_sec
                ),
            )
            # First read at startup so ``rules()`` is non-empty
            # without waiting one polling interval.
            with suppress(Exception):
                self._production_rules_loader.force_reload()
            logger.info(
                "ProductionRulesLoader: %s rules at boot from %s",
                len(self._production_rules_loader),
                self._production_rules_loader.path,
            )
        if self._production_rules_loader is not None:
            self._tasks.append(asyncio.create_task(
                self._production_rules_reload_worker(),
                name="production_rules_reload_worker",
            ))

        await self._stop_event.wait()
        await self._shutdown()

    async def _safe_notify_signal(self, payload: dict[str, Any]) -> None:
        """Telegram-notify the high-priority signal without raising.

        Used as a fire-and-forget background task from ``fused_sink``
        when ``cfg.telegram_fire_and_forget`` is True (default). Any
        exception is logged and swallowed; we never want a failed
        notification to crash the order pipeline."""
        try:
            await self.notifier.signal(payload)
        except Exception as e:
            logger.warning("notify_signal background task failed: %s", e)

    # ------------------------------------------------------------------ #
    # Phase B.2 — reject-counter / DLQ helpers.
    #
    # Both side-channel observability paths are conditional: when
    # ``self._metrics`` / ``self._dlq`` are None the calls are
    # cheap no-ops and the daemon's behaviour is byte-for-byte
    # identical to v1.0. We deliberately bucket reasons into a small
    # set of canonical labels to prevent Prometheus label-cardinality
    # explosions from per-symbol or per-error-class strings.
    # ------------------------------------------------------------------ #

    # Canonical reason labels for the orders_rejected_total counter.
    # The free-form reason string still goes to the DLQ payload, but
    # the metric carries a coarse bucket so cardinality stays bounded.
    _REJECT_REASON_BUCKETS = (
        "anti_chase", "vol_kill", "min_liquidity",
        "consecutive_loss_cooldown", "daily_drawdown",
        "max_concurrent_positions", "regime_filter", "cluster_cap",
        "kill_switch", "reflection_mode_suspended",
        "quote_unavailable", "depth_unavailable", "vol_unavailable",
        "executor_exception",
    )

    @classmethod
    def _bucket_reject_reason(cls, raw_reason: str) -> str:
        """Map a free-form reject reason to one of the canonical
        buckets. Falls back to ``"other"`` so the cardinality is
        bounded at len(_REJECT_REASON_BUCKETS) + 1.
        """
        if not raw_reason:
            return "other"
        lower = raw_reason.lower()
        for bucket in cls._REJECT_REASON_BUCKETS:
            if bucket in lower:
                return bucket
        return "other"

    def _record_rejection(
        self,
        *,
        symbol: str,
        reason: str,
        kind: str = "gate_reject",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Bump the rejection counter and (optionally) write to DLQ.

        Idempotent + side-effect-only: the caller has already done
        ``self.state.orders_rejected += 1`` and pushed to the
        dashboard ring buffer; this method only adds the Phase B.2
        side-channel observations. Always swallows exceptions so a
        flaky disk / metrics bug never blocks the trading hot path.
        """
        if self._metrics is not None:
            try:
                bucket = self._bucket_reject_reason(reason)
                self._metrics.orders_rejected_total.inc(
                    labels={"reason": bucket},
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "metrics orders_rejected_total inc failed: %s", e,
                )
        if self._dlq is not None:
            try:
                self._dlq.put(DLQEntry(
                    kind=kind,
                    symbol=symbol,
                    reason=reason,
                    payload=payload or {},
                ))
                if self._metrics is not None:
                    with suppress(Exception):
                        self._metrics.dlq_writes_total.inc(
                            labels={"kind": kind},
                        )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("DLQ write failed: %s", e)

    def _record_order_placed(
        self,
        *,
        symbol: str,
        side: str,
    ) -> None:
        """Bump the per-symbol/side ``orders_placed_total`` counter.

        Cardinality safeguard: ``Counter.inc`` already rejects new
        label combinations beyond ``max_label_cardinality=500``, so a
        screener mis-config that floods the counter with thousands of
        symbols downgrades to "no new series", not a crash.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.orders_placed_total.inc(
                labels={"symbol": symbol, "side": side},
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "metrics orders_placed_total inc failed: %s", e,
            )

    def _sync_account_gauges(self, account: AccountState) -> None:
        """Refresh the gauges that mirror ``AccountState``.

        Called after every state mutation that the operator might want
        plotted (open positions, halt, drawdown). Never raises.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.open_positions.set(len(account.open_positions))
            self._metrics.halt_engaged.set(
                1.0 if account.global_trading_halted else 0.0,
            )
            self._metrics.cooldown_symbols.set(
                len(account.cooldown_until_ts_ms),
            )
            self._metrics.daily_stoploss_hits.set(
                float(account.daily_stoploss_hits),
            )
            self._metrics.daily_drawdown_pct.set(
                float(account.daily_drawdown_pct),
            )
            if account.consecutive_losses:
                self._metrics.consecutive_losses_max.set(
                    float(max(account.consecutive_losses.values())),
                )
            else:
                self._metrics.consecutive_losses_max.set(0.0)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("metrics sync_account_gauges failed: %s", e)


    # ------------------------------------------------------------------ #
    # Phase A — miss-penalty + reflection-mode wiring
    # ------------------------------------------------------------------ #

    def _wire_miss_penalty_pipeline(self) -> None:
        """Construct the four miss-penalty actors when enabled.

        Called once from ``run`` after the decision audit log has been
        materialised. The kline-fetcher is plumbed lazily (it needs
        the live ccxt client which is only available on the
        :class:`CCXTExchangeAdapter` -- in dry-run we use an empty
        no-op fetcher so the audit logs ``insufficient_data`` rather
        than crashing).
        """
        state_dir = self.cfg.miss_penalty_state_dir
        decisions_log = Path(self.cfg.decision_audit_log_path)
        missed_path = Path(state_dir) / "missed_opportunities.jsonl"

        # Audit engine.
        miss_cfg = MissPenaltyConfig(
            forward_window_sec=self.cfg.miss_penalty_forward_window_sec,
            state_dir=state_dir,
        )
        self._miss_penalty = MissPenaltyEngine(
            decisions_log_path=decisions_log,
            kline_fetcher=self._build_kline_fetcher(),
            config=miss_cfg,
        )

        # Reject-reason scorer.
        self._reject_scorer = RejectReasonScorer(
            decisions_log_path=decisions_log,
            missed_opportunities_path=missed_path,
            config=RejectReasonScorerConfig(
                state_path=str(
                    Path(state_dir) / "reject_reason_scores.json",
                ),
            ),
        )

        # Threshold tuner.
        self._threshold_tuner = ThresholdAutoTuner(
            scorer=self._reject_scorer,
            config=ThresholdAutoTunerConfig(
                overrides_state_path=str(
                    Path(state_dir) / "threshold_overrides.json",
                ),
            ),
        )

        # Reflection controller. The Telegram callback re-uses the
        # existing ``error`` channel so the operator gets the alert
        # through the same chat as kill-switch + rollover events.
        async def _telegram_callback(payload: dict[str, Any]) -> None:
            with suppress(Exception):
                if self.notifier is not None:
                    await self.notifier.error(
                        payload.get("title", "策略反思报告"),
                        payload=payload,
                    )

        self._reflection = ReflectionModeController(
            config=ReflectionConfig(
                window_sec=self.cfg.reflection_window_days * 24 * 3600,
                miss_threshold=self.cfg.reflection_miss_threshold,
                trade_threshold=self.cfg.reflection_trade_threshold,
                suspension_sec=self.cfg.reflection_suspension_hours * 3600,
                a_quadrant_bypass_score=(
                    self.cfg.reflection_a_quadrant_bypass_score
                ),
                state_path=str(
                    Path(state_dir) / "reflection_state.json",
                ),
                reports_dir=self.cfg.reflection_reports_dir,
            ),
            llm_caller=self._build_reflection_llm_caller(),
            notifier=_telegram_callback,
        )

    def _build_kline_fetcher(self):
        """Pick the right :type:`KlineFetcher` for the active adapter.

        For ``CCXTExchangeAdapter`` we re-use the same client the
        executor talks to (rate-limit / proxy / auth shared). For the
        in-process ``DryRunExchangeAdapter`` we return a no-op fetcher
        so the audit silently records ``insufficient_data`` -- a real
        backtest path would inject a fixture fetcher via
        ``self._miss_penalty = MissPenaltyEngine(...)`` before run().
        """
        adapter = self._adapter
        client = getattr(adapter, "client", None)
        if client is not None and hasattr(client, "fetch_ohlcv"):
            return make_ccxt_kline_fetcher(client)

        async def _empty(_symbol: str, _since: int, _until: int):
            return []

        return _empty

    def _build_reflection_llm_caller(self):
        """Return an :type:`LLMCaller` that calls into the existing
        DeepSeek engine, or None when the operator hasn't opted in.

        Token budget: a single reflection prompt is ~3K tokens. With
        the trigger cooldown of 24h, max usage is ~90K/month --
        within the operator's training token budget per
        ``QUADRANT_STRATEGY_PLAN.md§6.1``.
        """
        if not self.cfg.reflection_use_llm:
            return None
        engine = self._llm_engine
        if engine is None:
            return None

        async def _call(prompt: str) -> str:
            try:
                return await engine.chat_text(prompt, timeout=30.0)
            except AttributeError:
                # Older engines without chat_text; fall back to the
                # deterministic summary.
                return ""

        return _call

    async def _miss_penalty_worker(self, account: AccountState) -> None:
        """Daily cron: audit yesterday's rejections, score reasons,
        check reflection trigger; weekly: emit threshold overrides.

        Modelled on ``daily_rollover_worker``: poll every
        ``miss_penalty_poll_sec`` and fire when the UTC hour matches
        ``miss_penalty_run_at_utc_hour`` AND the run has not yet
        completed for the current UTC date.
        """
        from datetime import datetime, timezone

        if (
            self._miss_penalty is None
            or self._reject_scorer is None
            or self._threshold_tuner is None
            or self._reflection is None
        ):
            return

        last_run_path = (
            Path(self.cfg.miss_penalty_state_dir) / "_last_run.txt"
        )
        last_run_path.parent.mkdir(parents=True, exist_ok=True)

        def _read_last_run_date() -> str:
            try:
                return last_run_path.read_text(encoding="utf-8").strip()
            except OSError:
                return ""

        def _write_last_run_date(date_str: str) -> None:
            try:
                last_run_path.write_text(date_str, encoding="utf-8")
            except OSError as e:
                logger.warning("miss_penalty: last_run write failed: %s", e)

        try:
            while not self._stop_event.is_set():
                try:
                    now_utc = datetime.now(timezone.utc)
                    today_str = now_utc.strftime("%Y-%m-%d")
                    last_str = _read_last_run_date()
                    should_run = (
                        last_str != today_str
                        and now_utc.hour >= self.cfg.miss_penalty_run_at_utc_hour
                    )
                    if should_run:
                        await self._run_miss_penalty_pass(
                            account=account, now_utc=now_utc,
                        )
                        _write_last_run_date(today_str)
                except Exception as e:
                    logger.exception("miss_penalty_worker failed: %s", e)
                    self.state.last_error = (
                        f"miss_penalty:{type(e).__name__}"
                    )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.cfg.miss_penalty_poll_sec,
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    async def _production_rules_reload_worker(self) -> None:
        """R3: poll ``production_rules.json`` and refresh the in-memory snapshot.

        Mtime-throttled inside :class:`ProductionRulesLoader`, so this
        worker can poll on a very tight schedule without re-parsing
        the file every iteration -- the JSON is only re-read when its
        mtime advances. We still wait for the configured interval
        between checks so a busy filesystem doesn't see ~10 stat calls
        per second.

        Failure modes:
          * Missing file: loader logs once and clears the snapshot.
          * Malformed JSON: loader keeps the previous snapshot and
            warns; subsequent calls retry on each mtime change.
          * Cancelled: graceful exit on shutdown.
        """
        loader = self._production_rules_loader
        if loader is None:
            return
        try:
            while not self._stop_event.is_set():
                try:
                    loader.maybe_reload()
                except Exception as e:  # pragma: no cover -- defensive
                    logger.exception(
                        "production_rules_reload_worker iteration failed: %s",
                        e,
                    )
                    self.state.last_error = (
                        f"production_rules_reload:{type(e).__name__}"
                    )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.cfg.production_rules_reload_interval_sec,
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
    async def _run_miss_penalty_pass(
        self, *, account: AccountState, now_utc,
    ) -> None:
        """Single audit + score + (weekly) tune + reflection check.

        Pulled out of the worker so tests can call it directly without
        spinning up a real cron loop.
        """
        if (
            self._miss_penalty is None
            or self._reject_scorer is None
            or self._threshold_tuner is None
            or self._reflection is None
        ):
            return

        logger.info(
            "miss_penalty: starting daily pass at %s", now_utc.isoformat(),
        )

        # 1) Audit rejections, persist missed_opportunities.jsonl.
        new_missed = await self._miss_penalty.run_audit(
            lookback_hours=self.cfg.miss_penalty_lookback_hours,
        )
        logger.info("miss_penalty: %d new audit rows", len(new_missed))

        # 2) Recompute reject-reason scores.
        scores = self._reject_scorer.recompute()

        # 3) Weekly: produce threshold-override proposals.
        suggested = []
        if now_utc.weekday() == self.cfg.threshold_tuner_run_on_weekday:
            overrides = self._threshold_tuner.tune(scores=scores)
            suggested = [
                ov.to_dict()
                for ov in overrides.values()
                if ov.direction_label == "loosen"
            ]
            logger.info(
                "miss_penalty: %d threshold proposals on weekday=%d",
                len(suggested), now_utc.weekday(),
            )

        # 4) Reflection-mode trigger check.
        all_missed = self._miss_penalty.load_recent_missed()
        decision = self._reflection.maybe_trigger(
            missed=all_missed,
            actual_trades_in_window=self._count_recent_trades(
                window_sec=self.cfg.reflection_window_days * 24 * 3600,
            ),
        )
        if decision.triggered:
            logger.warning(
                "miss_penalty: reflection mode TRIGGERED -- %s",
                decision.reason,
            )
            await self._reflection.generate_report(
                decision=decision,
                missed=all_missed,
                scores=scores,
                suggested_overrides=suggested,
            )
        else:
            logger.info(
                "miss_penalty: reflection check ok (%s)", decision.reason,
            )

    def _count_recent_trades(self, *, window_sec: int) -> int:
        """Count approved decisions in the last ``window_sec`` seconds.

        We re-walk the audit log instead of maintaining an in-memory
        counter so a daemon restart doesn't reset the count
        mid-window. Cost is amortised: the worker only runs once per
        day and the audit log is bounded by the rotation policy
        (``audit_log.py``).
        """
        from altcoin_agent.risk.miss_penalty_engine import (
            iter_decisions_with_rotations,
        )

        cutoff = time.time() - window_sec
        count = 0
        for rec in iter_decisions_with_rotations(
            Path(self.cfg.decision_audit_log_path), since_ts=cutoff,
        ):
            if rec.get("approved", False):
                count += 1
        return count

    async def _get_live_quote(self, symbol: str) -> float:
        """Return a fresh mark/last price for ``symbol``.

        Bug #2 fix: ``_handle_high_priority`` used to pass the
        ``signal.trigger_price`` as both the trigger AND the
        ``current_price`` to ``RiskGate.evaluate``, which made the SR-1
        adverse-slippage check a no-op (it was comparing the trigger to
        itself). We now sample a live price here and let the gate compare
        the two -- the slip threshold finally has teeth.

        Resolution order:
          1. ``self.quote_provider`` (test hook / custom integration);
          2. ``adapter.fetch_ticker_price`` (live ccxt + dry-run helper);
          3. raise -- the caller fail-closes the order with
             ``quote_unavailable`` rather than silently bypassing SR-1.
        """
        if self.quote_provider is not None:
            return float(await self.quote_provider(symbol))
        adapter = self._adapter
        fetcher = getattr(adapter, "fetch_ticker_price", None)
        if fetcher is None:
            raise RuntimeError("adapter has no fetch_ticker_price")
        return float(await fetcher(symbol))

    async def _fetch_top_depth_usdt(self, symbol: str) -> float:
        """Return top-5 order-book depth in USDT for ``symbol``.

        Bug C4 fix companion of ``_get_live_quote``. Resolution order:
          1. ``self.depth_provider`` (test hook);
          2. ``adapter.fetch_top_depth_usdt`` (live ccxt + dry-run helper);
          3. raise -- caller fail-closes with ``depth_unavailable`` rather
             than substituting ``cfg.min_liquidity_usdt`` and turning SR-2
             into a no-op.
        """
        if self.depth_provider is not None:
            return float(await self.depth_provider(symbol))
        adapter = self._adapter
        fetcher = getattr(adapter, "fetch_top_depth_usdt", None)
        if fetcher is None:
            raise RuntimeError("adapter has no fetch_top_depth_usdt")
        return float(await fetcher(symbol))

    async def _handle_high_priority(
        self,
        *,
        sig: FusedSignal,
        gate: RiskGate,
        executor: CCXTExecutor,
        trailing: TrailingController,
        account: AccountState,
    ) -> None:
        """Translate a high-priority FusedSignal into an order if Risk Gate
        approves. Attaches a trailing tracker to the new position.

        Phase B.6 wraps the entire decision in a ``decision_pipeline``
        OpenTelemetry span so an operator browsing Jaeger / Tempo can
        see every nested operation (live-quote fetch, gate, executor,
        notifier) under one root. ``start_span`` is a no-op when
        tracing is disabled, so the cost on the disabled path is one
        Python context-manager entry per high-priority signal —
        sub-microsecond.
        """
        if sig.direction == Direction.NEUTRAL:
            return
        if sig.trigger_price is None or sig.trigger_price <= 0:
            logger.warning("missing trigger_price on %s; skipping order", sig.symbol)
            return

        # Phase B.2.2: bind a trace id for the duration of this
        # decision. ``str(sig.ts)`` is what the audit log uses too,
        # so a single grep on ``trace_id`` pulls every JSON line +
        # every audit row associated with this signal. Bound at the
        # top so the reflection-suspension path also gets it.
        bind_trace_id(str(sig.ts))

        # Phase B.6: open the root span for this decision. We use the
        # context-manager form so any unhandled exception propagates
        # back to the queue worker AND lands as an exception event
        # on the span. Attributes here are the small, bounded set
        # most useful for span filtering — symbol, direction,
        # final_score, signal_ts. Per-hop spans (gate, executor)
        # carry their own attributes so we don't pollute the parent.
        with start_span(
            "decision_pipeline",
            attributes={
                "altcoin_agent.symbol": sig.symbol,
                "altcoin_agent.exchange": sig.exchange,
                "altcoin_agent.direction": sig.direction.value,
                "altcoin_agent.final_score": float(sig.final_score),
                "altcoin_agent.rule_score": float(sig.rule_score),
                "altcoin_agent.signal_ts": int(sig.ts),
            },
        ):
            await self._handle_high_priority_impl(
                sig=sig, gate=gate, executor=executor,
                trailing=trailing, account=account,
            )

    async def _handle_high_priority_impl(
        self,
        *,
        sig: FusedSignal,
        gate: RiskGate,
        executor: CCXTExecutor,
        trailing: TrailingController,
        account: AccountState,
    ) -> None:
        """The actual decision-pipeline body. Split out from
        ``_handle_high_priority`` so the public entry point can wrap
        the whole flow in a single OTel span without forcing a
        100-line indentation change. Behaviour is byte-for-byte
        identical to the V1.0 inlined code.
        """
        # Phase B.2.1: count high-priority signals by direction. This
        # is one of the gauges/counters the operator monitors to see
        # "is the screener emitting at all?".
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.high_priority_signals_total.inc(
                    labels={"direction": sig.direction.value},
                )

        # Phase A — reflection mode suspension.
        #
        # When the controller has paused the daemon (because the last
        # 7 days saw >=3 missed pumps and <2 trades), only A-quadrant
        # signals (final_score >= a_quadrant_bypass_score) get through.
        # We log the rejection in the audit so the operator can see
        # what got blocked while reflecting; everything below the
        # bypass score never reaches the gate.
        if (
            self._reflection is not None
            and self._reflection.is_suspended()
            and not self._reflection.can_bypass_suspension(
                final_score=sig.final_score,
            )
        ):
            self.state.orders_rejected += 1
            reason = (
                f"reflection_mode_suspended:final_score={sig.final_score:.1f}"
                f"<{self.cfg.reflection_a_quadrant_bypass_score:.1f}"
            )
            self._record_rejection(
                symbol=sig.symbol, reason=reason,
                kind="reflection_mode_suspended",
            )
            if self._decision_audit_log is not None:
                with suppress(Exception):
                    self._decision_audit_log.record_decision(
                        trace_id=getattr(sig, "trace_id", None),
                        symbol=sig.symbol,
                        signal_kind=getattr(sig, "kind", "unknown"),
                        rule_score=sig.rule_score,
                        final_score=sig.final_score,
                        direction=sig.direction.value,
                        approved=False,
                        reason=reason,
                        leverage=None, size=None, notional_usdt=None,
                        current_price=sig.trigger_price,
                        top5_depth_usdt=None, realized_vol_pct=None,
                        initial_stop=None, max_slippage_used=None,
                    )
            logger.info(
                "Reflection mode suspended; rejecting %s (%s)",
                sig.symbol, reason,
            )
            return

        # Bug #3 fix (defence-in-depth): the background rollover worker
        # is the primary trigger, but the hot path also calls this so a
        # delayed worker (long GC, scheduler stall) cannot leave the
        # daily-drawdown breaker latched on a fresh trading day.
        # Done before the live-quote fetch so a stale-day account fails
        # out without a network round-trip.
        if account.maybe_roll_over_day():
            logger.info(
                "Hot-path daily rollover applied (worker was late): "
                "starting_equity=%.2f",
                account.starting_equity_today_usdt,
            )

        # Bug #2 fix: pull a *live* quote from the venue/adapter and feed
        # that to the gate as ``current_price``. If we can't get one, abort
        # the order -- this is the same fail-closed posture the rest of the
        # gate uses (SR-2 reconciliation, SR-1 slippage cap, etc.).
        try:
            current_price = await self._get_live_quote(sig.symbol)
        except Exception as e:
            logger.warning(
                "live-quote unavailable for %s (%s); aborting order",
                sig.symbol, e,
            )
            self.state.orders_rejected += 1
            self.state.last_error = f"quote_unavailable:{type(e).__name__}"
            rej = {
                "ts": int(time.time() * 1000),
                "symbol": sig.symbol,
                "reason": f"quote_unavailable:{type(e).__name__}",
            }
            self.dashboard.push_rejection(rej)
            self._record_rejection(
                symbol=sig.symbol,
                reason=rej["reason"],
                kind="quote_unavailable",
                payload={"signal": sig.as_dict()},
            )
            with suppress(Exception):
                await self.notifier.rejected(rej)
            return

        if sig.direction == Direction.LONG:
            initial_stop = sig.trigger_price * 0.95
        else:
            initial_stop = sig.trigger_price * 1.05

        # Bug C4 fix: the previous version hard-coded
        # ``top5_depth_usdt = cfg.min_liquidity_usdt`` (i.e. "exactly the
        # floor", which makes the SR-2 liquidity gate a no-op) and
        # ``realized_vol_pct = 0.05`` (BTC-grade vol applied to symbols
        # like PEPE that routinely print 30% intraday). We now pull both
        # values from real sources:
        #
        #   * top-5 depth from the adapter's order book (sum of price*size
        #     across both sides, levels=5).
        #   * realized vol from the live PriceTape Parkinson estimator over
        #     the last 60s.
        #
        # If either source can't produce a value (cold start, adapter
        # error, no ticks yet), we fail-closed: skip the order. This is
        # symmetric with the live-quote handling above and prevents the
        # gate from being lied to.
        try:
            top5_depth_usdt = await self._fetch_top_depth_usdt(sig.symbol)
        except Exception as e:
            logger.warning(
                "depth unavailable for %s (%s); aborting order",
                sig.symbol, e,
            )
            self.state.orders_rejected += 1
            self.state.last_error = f"depth_unavailable:{type(e).__name__}"
            rej = {
                "ts": int(time.time() * 1000),
                "symbol": sig.symbol,
                "reason": f"depth_unavailable:{type(e).__name__}",
            }
            self.dashboard.push_rejection(rej)
            self._record_rejection(
                symbol=sig.symbol,
                reason=rej["reason"],
                kind="depth_unavailable",
                payload={"signal": sig.as_dict()},
            )
            with suppress(Exception):
                await self.notifier.rejected(rej)
            return

        realized_vol_pct: float | None = None
        if self._price_tape is not None:
            realized_vol_pct = self._price_tape.realized_vol_pct(
                symbol=sig.symbol,
                window_ms=self.cfg.vol_kill_window_ms,
            )
        if realized_vol_pct is None or realized_vol_pct <= 0:
            # Cold tape: not enough live ticks yet. In LIVE mode this is
            # fail-closed (do not size on a guess). In dry-run it's
            # tolerable to fall back to a conservative default so the
            # rest of the pipeline can be exercised end-to-end without a
            # live screener — but we still log so a dry-run that's
            # "cold for hours" is visible.
            if self.cfg.dry_run:
                realized_vol_pct = self.cfg.dry_run_fallback_vol_pct
                logger.warning(
                    "realized vol unavailable for %s (cold tape); "
                    "dry-run fallback vol=%.4f",
                    sig.symbol, realized_vol_pct,
                )
            else:
                logger.warning(
                    "realized vol unavailable for %s (cold tape); aborting order",
                    sig.symbol,
                )
                self.state.orders_rejected += 1
                self.state.last_error = "vol_unavailable:cold_tape"
                rej = {
                    "ts": int(time.time() * 1000),
                    "symbol": sig.symbol,
                    "reason": "vol_unavailable:cold_tape",
                }
                self.dashboard.push_rejection(rej)
                self._record_rejection(
                    symbol=sig.symbol,
                    reason=rej["reason"],
                    kind="vol_unavailable",
                    payload={"signal": sig.as_dict()},
                )
                with suppress(Exception):
                    await self.notifier.rejected(rej)
                return

        # Phase B.6: wrap gate.evaluate in its own span so dashboards
        # can spot a slow gate (e.g. SR-1 falling back to a degraded
        # quote provider) at a glance.
        with start_span(
            "risk_gate.evaluate",
            attributes={
                "altcoin_agent.symbol": sig.symbol,
                "altcoin_agent.current_price": float(current_price),
                "altcoin_agent.top5_depth_usdt": float(top5_depth_usdt),
                "altcoin_agent.realized_vol_pct": float(realized_vol_pct),
            },
        ) as gate_span:
            decision: RiskDecision = gate.evaluate(
                signal=sig,
                account=account,
                current_price=current_price,
                top5_depth_usdt=top5_depth_usdt,
                realized_vol_pct=realized_vol_pct,
                initial_stop=initial_stop,
                price_tape=self._price_tape,
                # Audit (third pass) #1: pass the wired safety gates.
                # When None (operator opted out via cfg), RiskGate skips
                # them — same back-compat shape PR #21 already established.
                regime_filter=self._regime_filter,
                cluster_map=self._cluster_map,
                cluster_cap_cfg=self._cluster_cap_cfg,
            )
            # Annotate the span with the outcome so a span filter on
            # ``approved=false`` returns the population we want.
            with suppress(Exception):
                gate_span.set_attribute(
                    "altcoin_agent.approved", bool(decision.approved),
                )
                gate_span.set_attribute(
                    "altcoin_agent.reject_reason", str(decision.reason),
                )

        # Audit (third pass) #1: every gate decision goes to the audit
        # log (approved or rejected). This is the only place we can
        # reconstruct *why* a trade fired (or didn't). Failures inside
        # ``record_decision`` are swallowed by the log itself.
        if self._decision_audit_log is not None:
            with suppress(Exception):
                self._decision_audit_log.record_decision(
                    trace_id=str(sig.ts),
                    symbol=sig.symbol,
                    signal_kind=",".join(
                        s.kind.value for s in sig.rule_signals
                    ) or "fused",
                    rule_score=float(sig.rule_score),
                    final_score=float(sig.final_score),
                    direction=sig.direction.value,
                    approved=bool(decision.approved),
                    reason=str(decision.reason),
                    leverage=decision.leverage,
                    size=decision.size,
                    notional_usdt=decision.notional_usdt,
                    current_price=float(current_price),
                    top5_depth_usdt=float(top5_depth_usdt),
                    realized_vol_pct=float(realized_vol_pct),
                    initial_stop=float(initial_stop),
                    max_slippage_used=decision.max_slippage_used,
                )

        if not decision.approved:
            logger.info("Risk Gate REJECT %s: %s", sig.symbol, decision.reason)
            self.state.orders_rejected += 1
            rej = {"ts": int(time.time() * 1000),
                   "symbol": sig.symbol, "reason": decision.reason}
            self.dashboard.push_rejection(rej)
            self._record_rejection(
                symbol=sig.symbol,
                reason=str(decision.reason),
                kind="gate_reject",
            )
            with suppress(Exception):
                await self.notifier.rejected(rej)
            return
        try:
            # Phase B.6: per-position-open span. Wrapping just
            # executor.open keeps the span boundary tight around the
            # network calls (set_leverage + market_order +
            # place_stop_order) — operators can correlate slow venue
            # round-trips here without picking through the parent
            # decision_pipeline span.
            with start_span(
                "executor.open",
                attributes={
                    "altcoin_agent.symbol": sig.symbol,
                    "altcoin_agent.direction": sig.direction.value,
                    "altcoin_agent.size": float(decision.size or 0.0),
                    "altcoin_agent.leverage": float(decision.leverage or 0.0),
                    "altcoin_agent.notional_usdt": float(
                        decision.notional_usdt or 0.0,
                    ),
                },
                kind="client",
            ):
                position = await executor.open(
                    symbol=sig.symbol,
                    decision=decision,
                    current_price=current_price,
                    account=account,
                    trace_id=str(sig.ts),
                )
            self.state.orders_placed += 1
            self.state.open_positions = len(account.open_positions)
            self._record_order_placed(
                symbol=sig.symbol, side=position.side.value,
            )
            self._sync_account_gauges(account)
            trailing.attach(position)
            logger.info(
                "OPENED %s %s size=%.4f lev=%.2f stop=%.6f",
                position.side.value, position.symbol, position.size,
                position.leverage, position.current_stop,
            )
            opened_payload = {
                "ts": int(time.time() * 1000),
                "symbol": position.symbol,
                "side": position.side.value,
                "size": position.size,
                "leverage": position.leverage,
                "entry_price": position.entry_price,
                "initial_stop": position.initial_stop,
                "type": "MARKET+STOP_MARKET",
                "stop_price": position.current_stop,
            }
            self.dashboard.push_order(opened_payload)
            with suppress(Exception):
                await self.notifier.opened(opened_payload)
            # Close the self-evolution loop: schedule a post-mortem 1h
            # after entry so the rules learned from this trade flow back
            # into the fuser via dynamic_rules.json.
            #
            # Bug #2 fix: pass ``entry_ts_ms`` (now, the moment we opened)
            # and ``expected_direction`` derived from the position side.
            # The post-mortem will then slice [entry-4h, entry+1h], extract
            # features strictly from the pre-entry segment, and evaluate
            # the realized move strictly post-entry — so we learn what
            # predicted what we actually got, not what predicted some
            # arbitrary extremum in the lookback window.
            if self._post_mortem is not None:
                entry_ts_ms = int(time.time() * 1000)
                target_ts_ms = entry_ts_ms + (
                    self._post_mortem.delay_sec * 1000
                )
                expected_direction = (
                    "pump" if position.side == Side.LONG else "dump"
                )
                # Phase B.6: forward the cited KOL authors + the
                # verdict's kol_intent so the analyzer can score them
                # 1h from now using the same realised direction the
                # rule learner sees. Both fields default to None when
                # the analyzer is not wired -> scheduler skips the
                # KOL-history branch automatically.
                kol_intent_for_pm: str | None = None
                if sig.llm_verdict is not None and sig.llm_verdict.kol_intent in (
                    "frontrun_call", "exit_liquidity",
                ):
                    kol_intent_for_pm = sig.llm_verdict.kol_intent
                self._post_mortem.schedule(
                    symbol=sig.symbol,
                    target_ts_ms=target_ts_ms,
                    entry_ts_ms=entry_ts_ms,
                    expected_direction=expected_direction,
                    kol_authors=sig.kol_authors or None,
                    kol_intent=kol_intent_for_pm,
                )
                self.state.post_mortems_scheduled += 1
        except Exception as e:
            logger.exception("Executor failed for %s: %s", sig.symbol, e)
            self.state.last_error = f"executor:{type(e).__name__}"
            self.state.orders_rejected += 1
            self._record_rejection(
                symbol=sig.symbol,
                reason=f"executor_exception:{type(e).__name__}:{e}",
                kind="executor_exception",
                payload={
                    "signal": sig.as_dict(),
                    "exc_type": type(e).__name__,
                },
            )
            with suppress(Exception):
                await self.notifier.error(f"executor failed for {sig.symbol}: {e}")

    async def _on_position_close(
        self,
        *,
        position: Position,
        reason: str,
        trailing: TrailingController,
        account: AccountState,
    ) -> None:
        """Run all the housekeeping that becomes due once a position is
        confirmed closed exchange-side.

        Steps:
            1. Estimate realized PnL using the position's current_stop as
               the most likely fill price (exchange-side STOP_MARKET fired)
               and the position's leverage.
            2. Update ``account.realized_pnl_today_usdt`` and
               ``account.equity_usdt`` so the daily-drawdown circuit
               breaker can actually fire.
            3. Bump ``account.daily_stoploss_hits`` if the close was a loss
               (so the 3-strike circuit breaker can engage).
            4. Bump consecutive_losses for the symbol on a loss; reset on a
               win — feeds the per-symbol cooldown.
            5. Detach the trailing FSM tracker (no more cancel/replace on a
               ghost position).
            6. Reset rolling-controller state for this symbol so the next
               position on the same symbol starts with a clean ladder.
            7. Push to dashboard, notify Telegram.

        Multi-leg correctness:
            ``position.entry_price`` is leg 0's fill price; ``position.size``
            is the legacy single-leg size. After a roll, the truth lives in
            ``position.legs`` and is exposed via ``avg_entry_price`` /
            ``total_size``. We use those here so the realised PnL credited
            to the daily ledger reflects what actually closed -- otherwise
            a 1.0 -> 1.10 leg-0 + 1.10 -> 1.10 leg-1 close would credit
            extra phantom PnL (size_legacy * leg_0_delta) instead of the
            true (total_size * weighted_delta).

        Failure modes are all logged and swallowed: by the time we get
        here the exchange has already done the close, our job is purely
        to record it.
        """
        symbol = position.symbol
        # Best-effort fill price: the resting stop is what the exchange
        # most likely filled at. Live integrations can later replace this
        # with a real fetch_my_trades lookup — for now we record the
        # expected stop fill so the daily-DD math is *directionally*
        # correct rather than zero (the previous behaviour).
        fill_price = position.current_stop
        r_unit = position.r_unit

        # Multi-leg PnL: use weighted-avg entry and aggregate size so a
        # rolled position's PnL reflects the whole book, not just leg 0.
        # For V1.0 single-leg positions these properties degrade to the
        # legacy ``entry_price`` / ``size`` so behaviour is unchanged.
        avg_entry = position.avg_entry_price
        total_size = position.total_size

        if position.side == Side.LONG:
            price_delta = fill_price - avg_entry
        else:
            price_delta = avg_entry - fill_price
        realized_pnl_usdt = price_delta * total_size
        realized_r = (price_delta / r_unit) if r_unit > 0 else 0.0
        is_loss = realized_pnl_usdt < 0

        # 2-4) account-level rollups + per-symbol consec-loss accounting.
        # Phase B.1.3: delegated to AccountState.record_pnl so the new
        # event-driven persistence hook fires automatically. Behaviour
        # is byte-for-byte identical to the previous inlined block:
        #   realized_pnl_today_usdt += pnl
        #   equity_usdt              += pnl
        #   if loss: daily_stoploss_hits += 1; consec_losses[sym] += 1
        #   else:    consec_losses.pop(sym)
        account.record_pnl(symbol, realized_pnl_usdt, is_loss=is_loss)

        # 5) detach trailing tracker
        trailing.detach(symbol)

        # 6) Reset rolling controller's per-symbol bookkeeping so the
        #    next position on this symbol starts with an empty
        #    fired-thresholds set. Without this, residual state from a
        #    prior position (e.g. {1.5} R already fired) would suppress
        #    the first roll on the new one. Safe no-op when the
        #    controller hasn't been wired (cfg.rolling.enabled=False).
        rolling = getattr(self, "_rolling", None)
        if rolling is not None:
            with suppress(Exception):
                rolling.reset_for_symbol(symbol)

        # 7) Phase B.1.3 (was: Audit third-pass #1): the post-close
        # snapshot is now driven automatically by ``record_pnl`` /
        # ``set_cooldown`` / ``halt`` via AccountState's change
        # listener. We keep this explicit save as a defence-in-
        # depth flush in case future code mutates additional state
        # between ``record_pnl`` and the end of this handler — the
        # listener's save() is idempotent and write-cost is <1ms.
        if self._persistor is not None:
            with suppress(Exception):
                self._persistor.save(account)

        # Note: ``account.open_positions.pop`` and ``position.closed=True``
        # are already done by PositionWatcher before this callback runs;
        # we don't redo them here.

        self.state.closed_positions += 1
        self.state.last_close_ts = time.time()
        self.state.open_positions = len(account.open_positions)

        legs_info = (
            f" legs={len(position.legs)}"
            if position.legs and len(position.legs) > 1
            else ""
        )
        logger.info(
            "CLOSED %s %s size=%.4f avg_entry=%.6f fill=%.6f "
            "pnl=%.4f R=%.2f reason=%s%s",
            position.side.value, symbol, total_size,
            avg_entry, fill_price,
            realized_pnl_usdt, realized_r, reason, legs_info,
        )

        closed_payload = {
            "ts": int(time.time() * 1000),
            "symbol": symbol,
            "side": position.side.value,
            "size": total_size,
            "entry_price": avg_entry,
            "fill_price": fill_price,
            "realized_pnl_usdt": round(realized_pnl_usdt, 6),
            "realized_r": round(realized_r, 4),
            "reason": reason,
            "num_legs": len(position.legs) if position.legs else 1,
        }
        with suppress(Exception):
            self.dashboard.push_close(closed_payload)
        if self.notifier is not None:
            with suppress(Exception):
                await self.notifier.closed(closed_payload)

    def _mode_label(self) -> str:
        if self.cfg.dry_run:
            return "DRY-RUN"
        if self.cfg.paper_trade:
            return "PAPER-TRADE (testnet)"
        return "LIVE"

    async def _shutdown(self) -> None:
        logger.info(
            "Shutdown initiated. Graceful timeout=%.1fs",
            self.cfg.graceful_timeout_sec,
        )
        if self._screener is not None:
            self._screener.stop()
        # Cancel any pending post-mortem tasks so we don't block on a 1h sleep.
        if self._post_mortem is not None:
            with suppress(Exception):
                await self._post_mortem.shutdown()
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=self.cfg.graceful_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("Graceful timeout hit; cancelling outstanding tasks.")
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        # Audit (third pass) #4: drain in-flight fire-and-forget tasks
        # (Telegram notifications etc.) so we don't drop messages on
        # SIGTERM. Cancel any that are still running after a short
        # grace period — we already gave them the full
        # graceful_timeout_sec via the workers above; another 2s is
        # enough for HTTPS round-trips to finish.
        if self._bg_tasks:
            pending = list(self._bg_tasks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Background task drain timeout; cancelling %d "
                    "fire-and-forget task(s)", len(pending),
                )
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        if self._runner is not None:
            with suppress(Exception):
                await self._runner.cleanup()
        if self._dashboard_runner is not None:
            with suppress(Exception):
                await self._dashboard_runner.cleanup()
        # Phase 5 — stop the pre-rater BEFORE closing the engine so
        # any in-flight rating finishes its provider call cleanly.
        # ``stop()`` is idempotent and bounded by an internal timeout
        # so a stuck worker can't block shutdown indefinitely.
        if self._llm_pre_rater is not None:
            with suppress(Exception):
                await self._llm_pre_rater.stop()
        # Close DeepSeek client if it was built.
        if self._llm_engine is not None:
            with suppress(Exception):
                await self._llm_engine.aclose()
        # Close ccxt client if any
        client = getattr(self._adapter, "client", None)
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                with suppress(Exception):
                    await close()
        if self.notifier is not None:
            with suppress(Exception):
                await self.notifier.aclose()
        # Phase B.6: flush + drop the OTel TracerProvider so any
        # pending spans get exported. ``shutdown_tracing`` is a no-op
        # when tracing was never configured or OTel is missing.
        with suppress(Exception):
            shutdown_tracing()
        logger.info("Shutdown complete.")

    def request_stop(self) -> None:
        if not self._stop_event.is_set():
            logger.info("Stop signal received.")
            self._stop_event.set()


# --------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------- #


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


# Bug C5 fix: V1.0 live-mode confirmation gate.
#
# ``config/app.yaml`` documents that this build is not yet certified for
# unattended live trading. The previous code allowed any operator with a
# ``DRY_RUN=0`` env var to flip to live (typo, copy-pasted CI script,
# stale shell) and the daemon would happily start placing real orders.
#
# We require an explicit, non-default acknowledgement (the literal
# string ``I_UNDERSTAND``) before live mode is allowed, and we leave a
# loud audit log. ``paper_trade=True`` (testnet) is exempt because no
# real funds are at stake. Tests bypass the gate by either keeping
# ``dry_run=True`` (the default) or constructing ``App`` directly.
LIVE_CONFIRM_TOKEN = "I_UNDERSTAND"
LIVE_CONFIRM_ENV = "LIVE_CONFIRM"


def enforce_live_mode_confirmation(cfg: AppConfig) -> None:
    """Refuse to launch in live mode without the operator acknowledgement.

    Raises ``SystemExit`` (exit code 3) when ``cfg.dry_run`` is False,
    ``cfg.paper_trade`` is False, and ``LIVE_CONFIRM`` is not set to
    ``I_UNDERSTAND`` in the environment. Otherwise returns silently.

    The exception case is recorded with a critical-level log so the
    operator can see exactly which knob is missing.
    """
    if cfg.dry_run or cfg.paper_trade:
        return
    confirm = os.getenv(LIVE_CONFIRM_ENV, "").strip()
    if confirm == LIVE_CONFIRM_TOKEN:
        logger.warning(
            "LIVE MODE ACKNOWLEDGED via %s=%s — real orders will be placed",
            LIVE_CONFIRM_ENV, LIVE_CONFIRM_TOKEN,
        )
        return
    logger.critical(
        "Refusing to start in LIVE mode without %s=%s. Either set the "
        "env var to acknowledge real-money trading, or run with "
        "DRY_RUN=1 / --dry-run / PAPER_TRADE=1 / --paper-trade.",
        LIVE_CONFIRM_ENV, LIVE_CONFIRM_TOKEN,
    )
    raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Altcoin Agent V1.0 daemon")
    parser.add_argument("--config", default="config/app.yaml",
                        help="path to YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="never place real orders, only log")
    parser.add_argument("--paper-trade", action="store_true",
                        help="real ccxt orders against testnet/sandbox")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)

    if os.path.exists(args.config):
        cfg = AppConfig.from_file(args.config)
    else:
        logger.warning("Config %s not found; using defaults.", args.config)
        cfg = AppConfig()

    if args.dry_run or os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes"):
        cfg.dry_run = True
    if args.paper_trade or os.getenv("PAPER_TRADE", "").lower() in ("1", "true", "yes"):
        cfg.paper_trade = True
        cfg.dry_run = False
    if os.getenv("DASHBOARD_ENABLED", "true").lower() in ("0", "false", "no"):
        cfg.dashboard_enabled = False

    # Bug C5 fix: live-mode gate. ``app.yaml`` documents that V1.0
    # refuses to start in live mode without an explicit operator
    # acknowledgement; the previous code allowed it silently. Now we
    # enforce it here, BEFORE the event loop is constructed, so an
    # accidental ``DRY_RUN=0`` in a CI shell can't reach the venue.
    enforce_live_mode_confirmation(cfg)

    # uvloop: 30–60% throughput improvement on the asyncio hot path.
    # Optional dependency; fall back to the stdlib loop if not
    # installed. Disable via cfg.use_uvloop=False (e.g. for Windows or
    # reproducibility-critical tests).
    if cfg.use_uvloop:
        try:
            import uvloop  # type: ignore[import-not-found]
            uvloop.install()
            logger.info("uvloop event loop installed")
        except ImportError:
            logger.info("uvloop not installed; using stdlib asyncio loop")

    app = App(cfg=cfg)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handler(sig_num: int) -> None:
        logger.info("Received signal: %s", sig_num)
        app.request_stop()

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig_name, lambda s=sig_name: _handler(s))

    try:
        loop.run_until_complete(app.run())
    finally:
        loop.close()
        logger.info("Event loop closed. Exiting.")


if __name__ == "__main__":
    main()
