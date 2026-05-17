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
from altcoin_agent.notifier import Notifier, build_default_notifier
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
    TrailingState,
    TrailingStopFSM,
    build_ccxt_adapter,
)
from altcoin_agent.screener import (
    FundingSnapshot,
    Kline,
    OISnapshot,
    Screener,
    SignalEvent,
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
    # Position-watcher (close lifecycle).
    # TICKET-008: previously defaulted to 5.0s × miss_threshold=2 = 10s
    # before a close was confirmed. With the unified RetryPolicy
    # (TICKET-005) the executor's reduce_only emergency close now
    # also kicks the watcher synchronously (see
    # ``TrailingController._emergency_close_naked``), so the poll
    # cadence drives only the *passive* path. 1.5s × 2 = 3s
    # debounce, comfortably above ccxt's typical fetch_positions
    # latency on a healthy network.
    position_watcher_poll_sec: float = 1.5
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
                d.get("position_watcher_poll_sec", 1.5),
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
    post_mortems_recorded: int = 0
    last_error: str | None = None
    # TICKET-016 — operational health counters surfaced on
    # ``/healthz`` (JSON), ``/metrics`` (Prometheus text), and the
    # dashboard's ``/api/state``. Each is a leading indicator for a
    # specific outage class so an on-call operator can triage in
    # seconds:
    #   * ``persistor_save_failures``: persistence layer is dropping
    #     writes — the daily-DD breaker may not survive a restart.
    #     Surfaced from ``AccountPersistor.consecutive_save_failures``
    #     by ``_sync_persistor_health`` after every save site.
    #   * ``position_watcher_lag_sec``: time since last successful
    #     ``fetch_positions``. Spikes here precede phantom positions.
    #     Read live in ``healthz``/``metrics`` from
    #     ``PositionWatcher.last_poll_wall_ts``.
    #   * ``llm_degraded_count``: number of LLM consult outcomes
    #     that returned a synthetic neutral verdict (timeout, parse
    #     error, budget exhaustion, total-budget deadline). Bumped
    #     by ``LLMEngine.judge``'s degradation path.
    #   * ``emergency_close_count``: rate of executor / trailing
    #     emergency closes. A sustained climb usually means a
    #     specific venue is unhappy.
    #   * ``stop_replace_failure_count``: per-tighten failures on
    #     the trailing path — a leading indicator for
    #     ``emergency_close_count``.
    persistor_save_failures: int = 0
    position_watcher_lag_sec: float = 0.0
    llm_degraded_count: int = 0
    emergency_close_count: int = 0
    stop_replace_failure_count: int = 0


async def make_health_app(
    state: HealthState,
    *,
    refresh_metrics: Callable[[], None] | None = None,
) -> web.Application:
    """Build the /healthz + /metrics aiohttp app.

    ``refresh_metrics`` is an optional zero-arg sync hook the App wires
    to refresh dynamic gauges (TICKET-016: persistor save failures,
    position-watcher lag) right before each scrape. The hook MUST
    not raise — failures are caught here so a bad refresh hook
    cannot 500 the probe.
    """
    async def _refresh_safe() -> None:
        if refresh_metrics is None:
            return
        try:
            refresh_metrics()
        except Exception as e:  # noqa: BLE001
            logger.warning("healthz refresh hook failed: %s", e)

    async def healthz(_request: web.Request) -> web.Response:
        await _refresh_safe()
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
            "post_mortems_recorded": state.post_mortems_recorded,
            "last_signal_ts": state.last_signal_ts,
            "last_error": state.last_error,
            # TICKET-016 health metrics.
            "persistor_save_failures": state.persistor_save_failures,
            "position_watcher_lag_sec": round(
                state.position_watcher_lag_sec, 2,
            ),
            "llm_degraded_count": state.llm_degraded_count,
            "emergency_close_count": state.emergency_close_count,
            "stop_replace_failure_count": state.stop_replace_failure_count,
        }
        return web.json_response(body, status=200 if ok else 503)

    async def metrics(_request: web.Request) -> web.Response:
        # Audit #23: minimal Prometheus text-format exporter. We bind
        # the same fields we already publish via /healthz so operators
        # can plot SLOs without bringing in a heavy client library.
        # All metrics are gauges (counters that only increase are also
        # valid gauges); no labels for V1 simplicity. Follow-up PR can
        # add per-symbol labels once the cardinality budget is set.
        await _refresh_safe()
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
        gauge("post_mortems_recorded", state.post_mortems_recorded,
              "Total post-mortem learning passes recorded after a real close")
        gauge("last_signal_ts", state.last_signal_ts,
              "Wall-clock ts of most-recent screener event")
        # TICKET-016 — operational health.
        gauge("persistor_save_failures", state.persistor_save_failures,
              "Consecutive AccountPersistor.save failures (0 == healthy)")
        gauge("position_watcher_lag_sec", state.position_watcher_lag_sec,
              "Seconds since last successful PositionWatcher poll")
        gauge("llm_degraded_count", state.llm_degraded_count,
              "Total LLM consults that returned a degraded neutral verdict")
        gauge("emergency_close_count", state.emergency_close_count,
              "Total executor / trailing emergency closes")
        gauge("stop_replace_failure_count", state.stop_replace_failure_count,
              "Total trailing tighten_hard_stop failures")
        return web.Response(
            text="\n".join(lines) + "\n",
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
        # TICKET-004: per-symbol fill-trade list for the
        # ``_on_position_close`` real-VWAP path. Empty by default;
        # tests pin via ``set_fill_trades``.
        self._my_trades: dict[str, list[dict[str, Any]]] = {}

    def _id(self) -> str:
        self._n += 1
        return f"dryrun-{self._n}"

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False, client_order_id=None):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid,
               "client_order_id": client_order_id,
               "symbol": symbol, "side": side.value, "size": size,
               "amount": size, "filled": size, "remaining": 0.0,
               "status": "closed",
               "price": price, "reduce_only": reduce_only,
               "average": price or 0.0}
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
        logger.info("[DRY-RUN] MARKET %s %s %s @ %s reduce=%s cid=%s",
                    side.value.upper(), size, symbol, price, reduce_only,
                    client_order_id)
        return rec

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True, *, client_order_id=None):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid,
               "client_order_id": client_order_id,
               "symbol": symbol, "side": side.value, "size": size,
               "amount": size, "filled": 0.0, "remaining": size,
               "status": "open",
               "stop_price": stop_price, "reduce_only": reduce_only,
               "average": 0.0, "price": 0.0}
        self.stop_orders.append(rec)
        logger.info("[DRY-RUN] STOP-MARKET %s %s %s @ %s cid=%s",
                    side.value.upper(), size, symbol, stop_price,
                    client_order_id)
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

    # ------------------- TICKET-004: fill-trade lookup -------------------
    # The dry-run adapter doesn't have a real trade feed, but exposing the
    # method (returning the empty list by default) lets ``App._on_position_close``
    # exercise the production code path. Tests can pre-populate per-symbol
    # via ``set_fill_trades``.

    async def fetch_my_trades(
        self, *, symbol: str, since_ms: int,
        client_order_id: str | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        del limit
        rows = self._my_trades.get(symbol, [])
        out: list[dict[str, Any]] = []
        for r in rows:
            if r.get("timestamp", 0) < since_ms:
                continue
            if client_order_id and r.get("client_order_id") not in (
                client_order_id, None,
            ):
                continue
            out.append(dict(r))
        return out

    def set_fill_trades(
        self, symbol: str, trades: list[dict[str, Any]],
    ) -> None:
        """Test helper: pin a list of fill trades for ``symbol``.

        Each trade is a dict with at least ``timestamp`` (ms),
        ``price``, ``amount`` and ``side``. Optional ``client_order_id``
        lets tests filter by parent order.
        """
        self._my_trades[symbol] = list(trades)


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
    # TICKET-009: when a tighten/restore chain fully fails the
    # position is naked. Rather than emergency-close on the very
    # next bar (which conflates "transient venue 502" with "really
    # naked"), we record the wall-clock timestamp the position went
    # naked. Subsequent kline ticks first try to recover by calling
    # ``tighten_hard_stop`` again (which will retry the place via
    # the adapter's RetryPolicy); only after ``naked_grace_sec``
    # have elapsed without a successful re-attach do we
    # emergency-close. ``None`` == not naked.
    naked_since_ts_ms: int | None = None


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
    # TICKET-009: degrade-then-emergency-close window. When the
    # tighten chain fully fails (replace AND restore) we mark the
    # position naked and keep retrying via the FSM tick. Only after
    # this many seconds without a successful re-attach do we
    # emergency-close. Default 30s = roughly 30 bars at 1m, long
    # enough to ride out a transient 502 / DDoS protection burst
    # but short enough to clamp on a real outage.
    naked_grace_sec: float = 30.0
    # TICKET-008: PositionWatcher we kick after every reduce_only
    # close so the close handler runs RIGHT NOW instead of waiting
    # up to ``poll_interval_sec``.
    position_watcher: PositionWatcher | None = None
    # TICKET-016: counter the dashboard exposes so operators can
    # see the rate of naked-position emergency closes at a glance
    # (the rate of stop_replace_failure_count is a leading
    # indicator for venue/health issues).
    _by_symbol: dict[str, _Tracked] = field(default_factory=dict)

    def attach(self, position: Position) -> None:
        self._by_symbol[position.symbol] = _Tracked(position=position)

    def detach(self, symbol: str) -> None:
        self._by_symbol.pop(symbol, None)

    async def _emergency_close_naked(
        self, *, tracked: _Tracked, bar: Kline,
    ) -> None:
        """TICKET-009 helper: emergency-close a position whose stop is
        confirmed-gone (``stop_order_id is None``) and whose grace
        window has elapsed.

        Steps mirror what the previous in-line block did: market
        reduce_only, mark the position closed, free the concurrency
        slot, engage the symbol cooldown, kick the watcher so its
        close callback runs immediately, and bump the dashboard
        counters. Failures inside the close are logged but do not
        propagate (we already concluded the position must close;
        raising would leave the daemon spinning on a stale entry).
        """
        symbol = tracked.position.symbol
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
            tracked.naked_since_ts_ms = None
            self.account.open_positions.pop(symbol, None)
            self.account.set_cooldown(
                symbol,
                self.executor.stop_failure_cooldown_sec,
                int(bar.ts),
            )
            # TICKET-016 metric.
            self.health.emergency_close_count += 1
            # TICKET-008: nudge the watcher so the close handler
            # fires THIS poll instead of next interval.
            if self.position_watcher is not None:
                self.position_watcher.hint_close_reason(
                    symbol, "emergency_close_trailing_naked",
                )
                self.position_watcher.kick()
        except Exception as e:
            logger.critical(
                "EMERGENCY CLOSE on naked trailing failed "
                "for %s: %s — manual intervention required",
                symbol, e,
            )

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
                # TICKET-009: a successful tighten clears any prior
                # naked state — we have a stop on the book again.
                tracked.naked_since_ts_ms = None
            else:
                # Audit #15 / TICKET-009: tighten failed. Two cases:
                #   (a) replace failed but the OLD stop is back on the
                #       book — the position is still protected, just
                #       at a wider stop than the FSM wanted. We log a
                #       warning, clear any prior naked state (the
                #       position is no longer naked), and continue.
                #   (b) replace failed AND the restore failed. The
                #       position is **naked** (no resting stop) and
                #       ``stop_order_id`` is None.
                #
                # Pre-TICKET-009 the case-(b) handler emergency-closed
                # immediately on every bar. With the unified
                # RetryPolicy now wrapping every adapter call
                # (TICKET-005), a single 502 already gets multiple
                # retries inside the adapter — so the executor's False
                # is now a much rarer signal that something is really
                # wrong. But it is STILL not "guaranteed wrong":
                # the operator wants us to give the network one more
                # chance. We therefore degrade gracefully:
                #   * record ``naked_since_ts_ms`` on first detection;
                #   * keep ticking the FSM, which will keep calling
                #     ``tighten_hard_stop`` (which the adapter's
                #     policy handles); on success we clear the flag.
                #   * Only after ``naked_grace_sec`` SECONDS have
                #     elapsed without recovery do we emergency-close.
                if tracked.position.stop_order_id is None:
                    self.health.stop_replace_failure_count += 1
                    if tracked.naked_since_ts_ms is None:
                        tracked.naked_since_ts_ms = int(bar.ts)
                        logger.error(
                            "trailing %s: tighten FAILED and restore "
                            "FAILED — position is NAKED at ts=%s; "
                            "grace window=%.1fs before emergency-close",
                            symbol, bar.ts, self.naked_grace_sec,
                        )
                        self.health.last_error = (
                            f"trailing naked grace {symbol}"
                        )
                    elapsed_sec = (
                        int(bar.ts) - tracked.naked_since_ts_ms
                    ) / 1000.0
                    if elapsed_sec >= self.naked_grace_sec:
                        logger.critical(
                            "trailing %s: NAKED grace expired "
                            "(%.1fs >= %.1fs) — emergency-closing now",
                            symbol, elapsed_sec, self.naked_grace_sec,
                        )
                        await self._emergency_close_naked(
                            tracked=tracked, bar=bar,
                        )
                else:
                    # Case (a): old stop is still on the book.
                    tracked.naked_since_ts_ms = None
                    logger.warning(
                        "trailing %s: tighten FAILED (%s); old stop "
                        "restored, position still protected",
                        symbol, reason,
                    )
                    self.health.last_error = (
                        f"trailing tighten restored on {symbol}"
                    )
                    self.health.stop_replace_failure_count += 1

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
    _regime_filter: RegimeFilter | None = None
    _cluster_map: ClusterMap | None = None
    _cluster_cap_cfg: ClusterCapConfig | None = None
    _kill_switch: KillSwitchWatcher | None = None
    _decision_audit_log: DecisionAuditLog | None = None

    async def run(self) -> None:
        self.state.started_at = time.time()
        mode = self._mode_label()
        logger.info("Altcoin Agent V1.0 starting (mode=%s)", mode)

        # TICKET-003: LIVE/PAPER mode requires persistence. Without it
        # an OOM kill at -5% mid-day silently re-arms the daily-DD
        # breaker at 0%, so the next session can lose ANOTHER 6%
        # before the breaker fires (=11% combined drawdown vs the
        # configured 6% cap). We refuse to start if the operator
        # disabled it for real-money modes.
        if (
            not self.cfg.dry_run
            and not self.cfg.account_persistence_enabled
        ):
            logger.critical(
                "Refusing to start in non-dry-run mode with "
                "account_persistence_enabled=False. The daily-DD breaker "
                "and consecutive-loss tracking would silently re-arm at "
                "zero on every restart. Either set "
                "account_persistence_enabled=True or run with --dry-run.",
            )
            raise SystemExit(4)

        if self.notifier is None:
            self.notifier = build_default_notifier()
        logger.info("Notifier: %s", self.notifier.name)

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
        if self.cfg.account_persistence_enabled and self._persistor is None:
            self._persistor = AccountPersistor(
                path=Path(self.cfg.account_persistence_path),
            )
        if self._persistor is not None:
            restored = self._persistor.restore_into(account)
            # TICKET-003: corrupt snapshot -> fail-closed boot. The
            # persistor has already set ``account.account_state_corrupt``
            # and ``account.halt(reason="account_state_corrupt")`` so
            # even if the operator force-clears one, the other catches
            # it. Refusing to start beats silently re-arming.
            if account.account_state_corrupt:
                logger.critical(
                    "ACCOUNT STATE FILE IS CORRUPT (%s). Refusing to "
                    "start. Inspect the file: if PnL state can be "
                    "manually reconstructed, repair it; otherwise "
                    "delete it after explicitly reconciling against "
                    "the venue's reporting tools.",
                    self._persistor.path,
                )
                raise SystemExit(5)
            if restored:
                logger.info(
                    "AccountState restored from %s: equity=%.2f, "
                    "today_pnl=%.2f, stops_today=%d, halted=%s, "
                    "open_positions=%d",
                    self._persistor.path,
                    account.equity_usdt,
                    account.realized_pnl_today_usdt,
                    account.daily_stoploss_hits,
                    account.global_trading_halted,
                    len(account.open_positions),
                )
            else:
                logger.info(
                    "AccountState persistence: no prior snapshot at %s "
                    "(starting fresh)", self._persistor.path,
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
        # TICKET-008: let the trailing controller kick the watcher
        # the moment it issues a reduce_only close, so the close
        # callback fires on the very next poll instead of waiting up
        # to ``poll_interval_sec``.
        trailing.position_watcher = position_watcher

        # TICKET-004: when the executor issues an emergency close
        # (partial-fill cleanup, stop-replacement failure, naked-position
        # close), it forwards a reason hint to the watcher so the
        # subsequent close event is labelled accurately ("emergency_close_*"
        # vs the generic default "exchange_close_detected"). The hint is
        # consumed on first use; future close events on the same symbol
        # fall back to the default.
        executor.on_emergency_close = position_watcher.hint_close_reason

        # TICKET-004: when the executor issues an emergency close
        # (partial-fill cleanup, stop-replacement failure, naked-position
        # close), it forwards a reason hint to the watcher so the
        # subsequent close event is labelled accurately ("emergency_close_*"
        # vs the generic default "exchange_close_detected"). The hint is
        # consumed on first use; future close events on the same symbol
        # fall back to the default.
        executor.on_emergency_close = position_watcher.hint_close_reason

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

        # Resolve dynamic rules path: cfg override -> env -> fuser default.
        fuser_cfg_kwargs: dict[str, Any] = {}
        if self.cfg.dynamic_rules_path:
            fuser_cfg_kwargs["dynamic_rules_path"] = Path(
                self.cfg.dynamic_rules_path,
            )
        fuser = ScoreFuser(sink=fused_sink, config=FuserConfig(**fuser_cfg_kwargs))

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
                # TICKET-007: same safety modules ``_handle_high_priority``
                # forwards to ``gate.evaluate``. Each is None when the
                # operator opted out via cfg.* flags, in which case
                # ``evaluate_rolling`` no-ops the corresponding gate —
                # identical fail-open semantics to the entry path.
                price_tape=self._price_tape,
                regime_filter=self._regime_filter,
                cluster_map=self._cluster_map,
                cluster_cap_cfg=self._cluster_cap_cfg,
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

        # TICKET-016 metric refresh: pulled at scrape time so the
        # ``/healthz`` and ``/metrics`` snapshots reflect the live
        # state of the persistor and the watcher. The hook never
        # touches the trading bus and is best-effort (failures are
        # caught inside ``make_health_app``).
        def _refresh_health_metrics() -> None:
            if self._persistor is not None:
                self.state.persistor_save_failures = (
                    self._persistor.consecutive_save_failures
                )
            pw = position_watcher
            if pw is not None and pw.last_poll_wall_ts > 0:
                self.state.position_watcher_lag_sec = max(
                    0.0, time.time() - pw.last_poll_wall_ts,
                )
            # TICKET-011/016: surface engine-level degradation count.
            if self._llm_engine is not None:
                self.state.llm_degraded_count = (
                    self._llm_engine.degraded_count
                )

        health_app = await make_health_app(
            self.state, refresh_metrics=_refresh_health_metrics,
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
        approves. Attaches a trailing tracker to the new position."""
        if sig.direction == Direction.NEUTRAL:
            return
        if sig.trigger_price is None or sig.trigger_price <= 0:
            logger.warning("missing trigger_price on %s; skipping order", sig.symbol)
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
                with suppress(Exception):
                    await self.notifier.rejected(rej)
                return

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
            with suppress(Exception):
                await self.notifier.rejected(rej)
            return
        try:
            position = await executor.open(
                symbol=sig.symbol,
                decision=decision,
                current_price=current_price,
                account=account,
                trace_id=str(sig.ts),
            )
            self.state.orders_placed += 1
            self.state.open_positions = len(account.open_positions)
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
            # Audit-fix Req #4: the self-evolution loop is now CLOSE-event
            # driven. ``_on_position_close`` calls ``self._post_mortem.record(...)``
            # with the actual realized PnL / fill price the moment a real
            # exchange-side close is detected by ``PositionWatcher``. We
            # NO LONGER schedule a fixed-time post-mortem at open: doing
            # so would file a learning event ~1h after open regardless of
            # whether the trade had been stopped out 5 minutes in, which
            # poisons ``dynamic_rules.json`` with synthetic market-slice
            # outcomes that have nothing to do with the realised trade.
        except Exception as e:
            logger.exception("Executor failed for %s: %s", sig.symbol, e)
            self.state.last_error = f"executor:{type(e).__name__}"
            self.state.orders_rejected += 1
            with suppress(Exception):
                await self.notifier.error(f"executor failed for {sig.symbol}: {e}")

    async def _lookup_real_fill_price(self, position: Position) -> float:
        """TICKET-004: VWAP across reduce_only fills since the position opened.

        Resolution order:
          1. ``adapter.fetch_my_trades(symbol, since_ms=opened_at_ts_ms,
             client_order_id=position.stop_client_order_id)`` if available.
             We prefer the stop's cid because the *close* fills come from
             whichever order actually closed the position (the resting
             stop on a stop-out, or an executor-issued emergency
             ``market_order(reduce_only=True)`` that DOES NOT carry
             the stop's cid — see fallback below).
          2. Same call with ``client_order_id=None`` to capture any
             reduce_only trade since open. Filtered to side opposite the
             position (``buy`` for SHORT close, ``sell`` for LONG close)
             and ``reduce_only=True`` if the venue echoes that hint.
          3. Fall back to ``position.current_stop`` and log so a missing
             fetch_my_trades surface is visible in production.

        Returns the VWAP (USDT-price). Never raises — close handling is
        purely bookkeeping; an erroring fill-price lookup must not abort
        the close handler.
        """
        adapter = self._adapter
        if adapter is None or not hasattr(adapter, "fetch_my_trades"):
            return position.current_stop
        try:
            trades = await adapter.fetch_my_trades(  # type: ignore[union-attr]
                symbol=position.symbol,
                since_ms=position.opened_at_ts_ms,
                client_order_id=position.stop_client_order_id,
                limit=100,
            )
            if not trades:
                # Fallback: close may have come from an emergency market
                # order whose cid we did not persist. Try the unfiltered
                # call and pick reduce_only trades on the closing side.
                trades = await adapter.fetch_my_trades(  # type: ignore[union-attr]
                    symbol=position.symbol,
                    since_ms=position.opened_at_ts_ms,
                    client_order_id=None,
                    limit=100,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "_lookup_real_fill_price(%s): fetch_my_trades failed (%s) "
                "— falling back to current_stop",
                position.symbol, e,
            )
            return position.current_stop
        # Filter: we want the trades that REDUCED the position. For a
        # LONG close that's side=="sell"; for a SHORT close it's
        # side=="buy". Some venues don't echo a reduce_only flag on the
        # trade object so we don't filter on it.
        closing_side = "sell" if position.side == Side.LONG else "buy"
        relevant = [
            t for t in trades
            if str(t.get("side") or "").lower() == closing_side
        ]
        if not relevant:
            logger.info(
                "_lookup_real_fill_price(%s): no closing-side trades found "
                "since open; falling back to current_stop",
                position.symbol,
            )
            return position.current_stop
        total_qty = 0.0
        total_cost = 0.0
        for t in relevant:
            try:
                qty = abs(float(t.get("amount") or 0.0))
                price = float(t.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or price <= 0:
                continue
            total_qty += qty
            total_cost += qty * price
        if total_qty <= 0:
            logger.info(
                "_lookup_real_fill_price(%s): degenerate trade rows; "
                "falling back to current_stop", position.symbol,
            )
            return position.current_stop
        return total_cost / total_qty

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
        # TICKET-004: real fill price.
        # Pre-fix this used ``position.current_stop`` directly — that's
        # the *expected* stop fill, which decouples ``realized_pnl_today_usdt``
        # from what actually settled. Manual closes, liquidations, and
        # emergency closes all reported a fake stop-equal fill.
        # We now ask the venue for the actual fills since
        # ``opened_at_ts_ms``, filtered to reduce_only trades for our
        # cid where possible, and compute a size-weighted VWAP. When
        # the adapter doesn't have ``fetch_my_trades`` (legacy mock,
        # cold cache) we fall back to ``current_stop`` and log so the
        # degradation is visible.
        fill_price = await self._lookup_real_fill_price(position)
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

        # 2) account-level rollups
        account.realized_pnl_today_usdt += realized_pnl_usdt
        account.equity_usdt += realized_pnl_usdt

        # 3 + 4) loss accounting
        if is_loss:
            account.daily_stoploss_hits += 1
            account.consecutive_losses[symbol] = (
                account.consecutive_losses.get(symbol, 0) + 1
            )
        else:
            account.consecutive_losses.pop(symbol, None)

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

        # 7) Audit (third pass) #1: persist the post-close snapshot so a
        # crash between this close and the next one doesn't lose the
        # daily PnL credit / stoploss counter / consec-loss bump.
        if self._persistor is not None:
            with suppress(Exception):
                self._persistor.save(account)

        # Note: ``account.open_positions.pop`` and ``position.closed=True``
        # are already done by PositionWatcher before this callback runs;
        # we don't redo them here.

        self.state.closed_positions += 1
        self.state.last_close_ts = time.time()
        self.state.open_positions = len(account.open_positions)
        # TICKET-016: count emergency closes so the dashboard surfaces
        # the rate. ``reason`` is the string the watcher reported,
        # which is either the executor's hinted ``emergency_close_*``
        # bucket (TICKET-004) or the generic ``exchange_close_detected``
        # for the common stop-out path. Anything starting with
        # ``emergency_close_`` is operator-actionable.
        if isinstance(reason, str) and reason.startswith("emergency_close_"):
            self.state.emergency_close_count += 1

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

        # Audit-fix Req #4: file the learning event NOW, off the real
        # close, with the realised PnL we just computed. This replaces
        # the previous open-time fixed-1h timer that would update
        # ``dynamic_rules.json`` with synthetic market-slice outcomes
        # disconnected from the actual trade. Failures are best-effort:
        # the learning loop must never poison the trading loop.
        if self._post_mortem is not None:
            with suppress(Exception):
                self._post_mortem.record(
                    symbol=symbol,
                    entry_ts_ms=position.opened_at_ts_ms,
                    close_ts_ms=int(time.time() * 1000),
                    side=position.side.value,
                    entry_price=avg_entry,
                    fill_price=fill_price,
                    realized_pnl_usdt=realized_pnl_usdt,
                    realized_r=realized_r,
                    close_reason=reason,
                    # TICKET-010: feed the position's leverage so
                    # ``magnitude_pct`` reflects realised R, not raw
                    # price delta. Without this a 5x trade that gained
                    # 1.5% on price (= 7.5% on equity) was being
                    # bucketed as ``pos_small`` alongside a 1.5%
                    # unleveraged blip.
                    leverage=position.leverage,
                )
                self.state.post_mortems_recorded += 1

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
