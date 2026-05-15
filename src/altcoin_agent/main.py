"""main.py — V1.0 production daemon entry point.

Responsibilities:
    * Load .env / YAML config.
    * Wire screener -> asyncio.Queue -> fuser -> (dry-run logging, for now).
    * Run reconciler at startup (SR-2). Refuses non-dry-run mode in V1.0
      because no live exchange adapter is yet wired in this entry point.
    * Expose /healthz on aiohttp for Docker healthcheck.
    * SIGINT / SIGTERM -> graceful shutdown:
        1. Stop accepting new screener events.
        2. Drain the queue with a bounded timeout.
        3. Cancel exchange WS subscriptions, close clients.
    * --dry-run: no real orders, only structured logging.

V1 keeps it simple: asyncio.Queue, single process, no Redis. Multi-process
scaling is documented in design.md but not implemented here.
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

from aiohttp import web

from altcoin_agent.fuser import FusedSignal, FuserConfig, ScoreFuser
from altcoin_agent.risk.executor import ExchangeAdapter
from altcoin_agent.risk.reconciler import Reconciler
from altcoin_agent.risk.state import AccountState, Side
from altcoin_agent.screener import Screener, SignalEvent

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
            "last_signal_ts": state.last_signal_ts,
        }
        return web.json_response(body, status=200 if ok else 503)

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    return app


# --------------------------------------------------------------------- #
# Dry-run exchange adapter
# --------------------------------------------------------------------- #


class DryRunExchangeAdapter:
    """Mimics the ExchangeAdapter Protocol but logs everything instead of
    placing real orders. Used when --dry-run is set."""

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
        return []  # dry-run starts clean

    async def fetch_open_orders(self) -> list[dict]:
        return []


# Sanity check the protocol fit at import time (helps catch drift early).
assert isinstance(DryRunExchangeAdapter(), ExchangeAdapter), (
    "DryRunExchangeAdapter does not satisfy ExchangeAdapter protocol"
)
# Side import only used for the protocol assertion above.
_ = Side


# --------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------- #


@dataclass
class App:
    cfg: AppConfig
    state: HealthState = field(default_factory=HealthState)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: list[asyncio.Task] = field(default_factory=list)
    _runner: web.AppRunner | None = None
    _screener: Screener | None = None

    async def run(self) -> None:
        self.state.started_at = time.time()
        logger.info("Altcoin Agent V1.0 starting (dry_run=%s)", self.cfg.dry_run)

        # ----- queue + components -----
        signal_q: asyncio.Queue[SignalEvent] = asyncio.Queue(maxsize=1000)

        async def screener_sink(ev: SignalEvent) -> None:
            self.state.last_signal_ts = time.time()
            self.state.rule_event_count += 1
            with suppress(asyncio.QueueFull):
                signal_q.put_nowait(ev)

        self._screener = Screener(
            exchanges=self.cfg.exchanges,
            symbols=self.cfg.symbols,
            sink=screener_sink,
            timeframes=self.cfg.timeframes,
        )

        async def fused_sink(sig: FusedSignal) -> None:
            self.state.high_priority_count += 1
            logger.info("HIGH PRIORITY: %s", sig.as_dict())
            if not self.cfg.dry_run:
                # In a future PR, this is where we'd call:
                #   decision = risk_gate.evaluate(...)
                #   if decision.approved: await executor.open(...)
                logger.warning(
                    "live trading entry point not wired in V1.0 main.py; "
                    "no order placed despite dry_run=False",
                )

        fuser = ScoreFuser(sink=fused_sink, config=FuserConfig())

        # ----- reconciler (SR-2) -----
        account = AccountState(equity_usdt=self.cfg.initial_equity_usdt)
        if self.cfg.dry_run:
            adapter = DryRunExchangeAdapter()
            rec_report = await Reconciler(
                exchange_name=self.cfg.exchanges[0], adapter=adapter,
            ).run(account)
            logger.info("Reconciler: %s", rec_report)
        else:
            logger.error(
                "Live mode requested but no live exchange adapter is wired in "
                "main.py for V1.0. Run with DRY_RUN=true while we add "
                "exchange-specific adapters.",
            )
            raise SystemExit(2)
        self.state.reconciliation_complete = account.reconciliation_complete

        # ----- health server -----
        health_app = await make_health_app(self.state)
        self._runner = web.AppRunner(health_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.cfg.healthz_port)
        await site.start()
        logger.info("Health endpoint live: http://0.0.0.0:%d/healthz",
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
            except asyncio.CancelledError:
                pass
            finally:
                self.state.fuser_alive = False

        async def screener_worker() -> None:
            try:
                await self._screener.run()  # type: ignore[union-attr]
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("screener failed: %s", e)
            finally:
                self.state.screener_alive = False

        self._tasks.append(asyncio.create_task(fuse_worker(), name="fuse_worker"))
        self._tasks.append(asyncio.create_task(screener_worker(), name="screener_worker"))

        await self._stop_event.wait()
        await self._shutdown()

    async def _shutdown(self) -> None:
        logger.info(
            "Shutdown initiated. Graceful timeout=%.1fs",
            self.cfg.graceful_timeout_sec,
        )
        if self._screener is not None:
            self._screener.stop()
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
