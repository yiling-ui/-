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
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from altcoin_agent.ai_engine import DeepSeekEngine
from altcoin_agent.dashboard import DashboardState, install_dashboard
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
from altcoin_agent.risk import (
    AccountState,
    ATRCalculator,
    CCXTExchangeAdapter,
    CCXTExecutor,
    ExchangeAdapter,
    Position,
    PositionSizer,
    Reconciler,
    RiskDecision,
    RiskGate,
    RiskGateConfig,
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
    # Path to dynamic_rules.json. The dashboard always needs a concrete path,
    # while the LLM/post-mortem path resolution will fall through env vars or
    # the fuser default when the user leaves this on the default value.
    dynamic_rules_path: str = ".kiro/steering/dynamic_rules.json"
    # LLM consult tuning
    llm_consult_cooldown_sec: int = 300   # per-symbol min spacing between consults
    llm_queue_max: int = 256
    # Online post-mortem (self-evolution)
    post_mortem_delay_sec: int = 3600     # 1h after open

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
            dynamic_rules_path=str(d.get(
                "dynamic_rules_path", ".kiro/steering/dynamic_rules.json")),
            llm_consult_cooldown_sec=int(d.get("llm_consult_cooldown_sec", 300)),
            llm_queue_max=int(d.get("llm_queue_max", 256)),
            post_mortem_delay_sec=int(d.get("post_mortem_delay_sec", 3600)),
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
    llm_consults: int = 0
    llm_consults_skipped: int = 0
    post_mortems_scheduled: int = 0
    last_error: str | None = None


async def make_health_app(state: HealthState) -> web.Application:
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
            "llm_consults": state.llm_consults,
            "llm_consults_skipped": state.llm_consults_skipped,
            "post_mortems_scheduled": state.post_mortems_scheduled,
            "last_signal_ts": state.last_signal_ts,
            "last_error": state.last_error,
        }
        return web.json_response(body, status=200 if ok else 503)

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    return app


# --------------------------------------------------------------------- #
# Dry-run exchange adapter (no network)
# --------------------------------------------------------------------- #


class DryRunExchangeAdapter:
    """Mimics ``ExchangeAdapter`` but logs everything instead of placing
    real orders. Used when ``--dry-run`` is set."""

    def __init__(self) -> None:
        self.market_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.cancelled: list[str] = []
        self.leverages: list[tuple[str, float]] = []
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"dryrun-{self._n}"

    async def market_order(self, symbol, side, size, *, price=None, reduce_only=False):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "price": price, "reduce_only": reduce_only,
               "average": price or 0.0}
        self.market_orders.append(rec)
        logger.info("[DRY-RUN] MARKET %s %s %s @ %s reduce=%s",
                    side.value.upper(), size, symbol, price, reduce_only)
        return rec

    async def place_stop_order(self, symbol, side, size, stop_price, reduce_only=True):  # noqa: ANN001
        oid = self._id()
        rec = {"id": oid, "symbol": symbol, "side": side.value, "size": size,
               "stop_price": stop_price, "reduce_only": reduce_only}
        self.stop_orders.append(rec)
        logger.info("[DRY-RUN] STOP-MARKET %s %s %s @ %s",
                    side.value.upper(), size, symbol, stop_price)
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
        return []

    async def fetch_open_orders(self) -> list[dict]:
        return []


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
    """Per-symbol trailing FSM state machine."""

    fsm: TrailingStopFSM
    atr: ATRCalculator
    executor: CCXTExecutor
    account: AccountState
    health: HealthState
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
        if new_stop is None:
            return
        ok = await self.executor.tighten_hard_stop(tracked.position, new_stop)
        if ok:
            logger.info("trailing %s: %s -> stop %s (atr=%.5f)",
                        symbol, reason, new_stop, atr)
        else:
            logger.warning("trailing %s: tighten FAILED (%s); position may be naked",
                           symbol, reason)
            self.health.last_error = f"trailing tighten failed on {symbol}"


# --------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------- #


@dataclass
class App:
    cfg: AppConfig
    state: HealthState = field(default_factory=HealthState)
    dashboard: DashboardState = field(default_factory=DashboardState)
    notifier: Notifier | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: list[asyncio.Task] = field(default_factory=list)
    _runner: web.AppRunner | None = None
    _screener: Screener | None = None
    _adapter: ExchangeAdapter | None = None
    _llm_engine: DeepSeekEngine | None = None
    _post_mortem: DelayedPostMortemScheduler | None = None
    _llm_consultor: LLMConsultor | None = None

    async def run(self) -> None:
        self.state.started_at = time.time()
        mode = self._mode_label()
        logger.info("Altcoin Agent V1.0 starting (mode=%s)", mode)

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
        account = AccountState(
            equity_usdt=self.cfg.initial_equity_usdt,
            starting_equity_today_usdt=self.cfg.initial_equity_usdt,
        )

        self.dashboard.health = self.state
        self.dashboard.account = account
        self.dashboard.rules_path = Path(self.cfg.dynamic_rules_path)

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

        # ----- HTTP server (healthz + dashboard) -----
        health_app = await make_health_app(self.state)
        if self.cfg.dashboard_enabled:
            install_dashboard(health_app, self.dashboard, mode_label=mode)
        self._runner = web.AppRunner(health_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.cfg.healthz_port)
        await site.start()
        logger.info("Health endpoint live: http://0.0.0.0:%d/healthz",
                    self.cfg.healthz_port)
        if self.cfg.dashboard_enabled:
            logger.info("Dashboard live:        http://0.0.0.0:%d/dashboard",
                        self.cfg.healthz_port)

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

        self._tasks.append(asyncio.create_task(fuse_worker(), name="fuse_worker"))
        self._tasks.append(asyncio.create_task(trailing_worker(), name="trailing_worker"))
        self._tasks.append(asyncio.create_task(screener_worker(), name="screener_worker"))
        self._tasks.append(asyncio.create_task(llm_worker(), name="llm_worker"))

        await self._stop_event.wait()
        await self._shutdown()

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

        if sig.direction == Direction.LONG:
            initial_stop = sig.trigger_price * 0.95
        else:
            initial_stop = sig.trigger_price * 1.05

        decision: RiskDecision = gate.evaluate(
            signal=sig,
            account=account,
            current_price=sig.trigger_price,
            top5_depth_usdt=self.cfg.min_liquidity_usdt,
            realized_vol_pct=0.05,
            initial_stop=initial_stop,
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
                current_price=sig.trigger_price,
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
            # Close the self-evolution loop: schedule a post-mortem 1h
            # after entry so the rules learned from this trade flow back
            # into the fuser via dynamic_rules.json.
            if self._post_mortem is not None:
                target_ts_ms = int(time.time() * 1000) + (
                    self._post_mortem.delay_sec * 1000
                )
                self._post_mortem.schedule(
                    symbol=sig.symbol, target_ts_ms=target_ts_ms,
                )
                self.state.post_mortems_scheduled += 1
        except Exception as e:
            logger.exception("Executor failed for %s: %s", sig.symbol, e)
            self.state.last_error = f"executor:{type(e).__name__}"
            self.state.orders_rejected += 1
            with suppress(Exception):
                await self.notifier.error(f"executor failed for {sig.symbol}: {e}")

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
        if self._runner is not None:
            with suppress(Exception):
                await self._runner.cleanup()
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
