"""metrics.py — Prometheus text-format metrics registry.

Phase B.2.1 of ``MISS_PENALTY_AND_PRODUCTION_PLAN.md``: the existing
``/metrics`` exporter in ``main.make_health_app`` published ~14 simple
gauges, far short of the 30+ metrics the operator needs to plot SLOs
and per-reason reject rates.

Design choices
--------------
* **Stdlib only.** We deliberately do NOT depend on the upstream
  ``prometheus_client`` library because:
    1. it pulls in twisted/asyncio shims we don't otherwise need;
    2. its multi-process collector wants a tmpfs/file-handle
       handshake that doesn't fit our single-process daemon;
    3. the wire format is small enough to emit directly, and tests
       can then assert the exact bytes without reaching into the
       library's internals.
* **Snapshot-on-render.** Each metric tracks its raw observations and
  only allocates the per-bucket cumulative arrays when ``render`` is
  called (≤ once per scrape). Keeps the hot path branch-free.
* **Bounded label cardinality.** ``Counter.inc`` accepts a small dict
  of label → value pairs. We refuse to register more than
  ``max_label_cardinality`` distinct combinations per metric (default
  500) to protect Prometheus from the kind of explosion you get from
  per-symbol × per-reason × per-direction labels on an altcoin
  screener.
* **Thread-safe.** All mutation goes through a single
  ``threading.Lock`` per metric. The trading loop is single-threaded
  asyncio, but the aiohttp metrics handler runs in its own coroutine
  and tests use ``asyncio.gather`` to fire concurrent observations.
* **Failure-tolerant render.** If a metric raises during render the
  rendered text contains a ``# RENDER_ERROR`` comment for that family
  but the rest of the registry still serialises — never blocks the
  health endpoint.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Standard latency buckets (seconds). Cover the realistic envelope:
# a 1ms WS round-trip up to a 60s pathological order-place latency.
DEFAULT_LATENCY_BUCKETS_SEC: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

# WS lag and event-loop lag are typically sub-second; finer-grained
# buckets prevent everything from collapsing into the bottom bucket.
DEFAULT_FAST_BUCKETS_SEC: tuple[float, ...] = (
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05,
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
)

# Token-budget / size buckets in raw count.
DEFAULT_TOKEN_BUCKETS: tuple[float, ...] = (
    100, 500, 1_000, 2_500, 5_000, 10_000,
    25_000, 50_000, 100_000, 250_000,
)

_VALID_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_VALID_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_metric_name(name: str) -> str:
    if not _VALID_NAME.fullmatch(name):
        raise ValueError(f"invalid Prometheus metric name: {name!r}")
    return name


def _validate_label(label: str) -> str:
    if not _VALID_LABEL.fullmatch(label):
        raise ValueError(f"invalid Prometheus label name: {label!r}")
    return label


def _escape_label_value(value: str) -> str:
    """Escape a label value per Prometheus exposition rules."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Stable hashable key for a label dict."""
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _format_labels(labels_key: tuple[tuple[str, str], ...]) -> str:
    """Render a labels-key tuple as ``{k="v",k2="v2"}`` (or empty).

    Always sorts the labels alphabetically so the rendered text is
    stable across scrapes regardless of the order the caller declared
    them in. Prometheus dashboards rely on this stability when diffing
    label sets across two scrapes.
    """
    if not labels_key:
        return ""
    parts = ",".join(
        f'{k}="{_escape_label_value(v)}"'
        for k, v in sorted(labels_key, key=lambda kv: kv[0])
    )
    return "{" + parts + "}"


def _format_value(v: float) -> str:
    """Prometheus value formatter: NaN, +Inf, -Inf, otherwise repr()."""
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


# --------------------------------------------------------------------- #
# Metric base + concrete classes
# --------------------------------------------------------------------- #


@dataclass
class _MetricBase:
    name: str
    help_text: str
    label_names: tuple[str, ...] = ()
    max_label_cardinality: int = 500
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        _validate_metric_name(self.name)
        for ln in self.label_names:
            _validate_label(ln)

    def _normalise_labels(
        self, labels: dict[str, str] | None,
    ) -> tuple[tuple[str, str], ...]:
        """Validate caller-supplied labels match the declared schema."""
        if not self.label_names:
            if labels:
                raise ValueError(
                    f"metric {self.name!r} declares no labels but got "
                    f"{sorted(labels)!r}"
                )
            return ()
        labels = labels or {}
        missing = set(self.label_names) - set(labels)
        extra = set(labels) - set(self.label_names)
        if missing or extra:
            raise ValueError(
                f"metric {self.name!r} label mismatch: "
                f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
            )
        return tuple(
            (ln, str(labels[ln])) for ln in self.label_names
        )


class Counter(_MetricBase):
    """Monotonically-increasing counter."""

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: Iterable[str] = (),
        max_label_cardinality: int = 500,
    ) -> None:
        super().__init__(
            name=name,
            help_text=help_text,
            label_names=tuple(label_names),
            max_label_cardinality=max_label_cardinality,
        )
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(
        self,
        amount: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        if amount < 0:
            raise ValueError(
                f"counter {self.name!r} can't decrement (amount={amount})"
            )
        key = self._normalise_labels(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_label_cardinality:
                logger.warning(
                    "metric %r refusing new label combo (cardinality=%d, "
                    "max=%d): %r",
                    self.name, len(self._values),
                    self.max_label_cardinality, dict(key),
                )
                return
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def value(self, labels: dict[str, str] | None = None) -> float:
        key = self._normalise_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> list[str]:
        out = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            items = list(self._values.items())
        if not items:
            # Prometheus convention: emit a zero sample for the no-label
            # case so dashboards don't complain about "missing series".
            if not self.label_names:
                out.append(f"{self.name} 0")
            return out
        for key, val in sorted(items):
            out.append(f"{self.name}{_format_labels(key)} {_format_value(val)}")
        return out


class Gauge(_MetricBase):
    """Arbitrary up/down value (current state)."""

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: Iterable[str] = (),
        max_label_cardinality: int = 500,
    ) -> None:
        super().__init__(
            name=name,
            help_text=help_text,
            label_names=tuple(label_names),
            max_label_cardinality=max_label_cardinality,
        )
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = self._normalise_labels(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_label_cardinality:
                logger.warning(
                    "metric %r refusing new label combo (cardinality=%d, "
                    "max=%d): %r",
                    self.name, len(self._values),
                    self.max_label_cardinality, dict(key),
                )
                return
            self._values[key] = float(value)

    def inc(
        self,
        amount: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = self._normalise_labels(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_label_cardinality:
                return
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(
        self,
        amount: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.inc(-amount, labels=labels)

    def value(self, labels: dict[str, str] | None = None) -> float:
        key = self._normalise_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> list[str]:
        out = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            items = list(self._values.items())
        if not items:
            if not self.label_names:
                out.append(f"{self.name} 0")
            return out
        for key, val in sorted(items):
            out.append(f"{self.name}{_format_labels(key)} {_format_value(val)}")
        return out


class Histogram(_MetricBase):
    """Histogram with cumulative buckets, ``_sum`` and ``_count`` series.

    Buckets must be sorted ascending and finite. The ``+Inf`` bucket is
    appended automatically. Observations land in the first bucket whose
    upper bound is >= the observation; cumulative semantics are
    rendered.
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Iterable[float] = DEFAULT_LATENCY_BUCKETS_SEC,
        label_names: Iterable[str] = (),
        max_label_cardinality: int = 500,
    ) -> None:
        super().__init__(
            name=name,
            help_text=help_text,
            label_names=tuple(label_names),
            max_label_cardinality=max_label_cardinality,
        )
        bs = tuple(float(b) for b in buckets)
        if not bs:
            raise ValueError(f"histogram {name!r} needs at least one bucket")
        prev = -math.inf
        for b in bs:
            if not math.isfinite(b):
                raise ValueError(
                    f"histogram {name!r} bucket must be finite (got {b})"
                )
            if b <= prev:
                raise ValueError(
                    f"histogram {name!r} buckets must be strictly "
                    f"ascending (got {bs})"
                )
            prev = b
        self.buckets: tuple[float, ...] = bs
        # key -> [bucket_count_0, ..., bucket_count_N, +inf_count, sum, n]
        # We store per-bucket non-cumulative counts, then accumulate at
        # render. Cheaper to update on the hot path.
        self._buckets: dict[tuple[tuple[str, str], ...], list[float]] = {}

    def _new_row(self) -> list[float]:
        # bucket counts (one per bucket + one for +Inf), sum, count
        return [0.0] * (len(self.buckets) + 1) + [0.0, 0.0]

    def observe(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        if math.isnan(value):
            return
        key = self._normalise_labels(labels)
        with self._lock:
            row = self._buckets.get(key)
            if row is None:
                if len(self._buckets) >= self.max_label_cardinality:
                    logger.warning(
                        "metric %r refusing new label combo (cardinality=%d)",
                        self.name, self.max_label_cardinality,
                    )
                    return
                row = self._new_row()
                self._buckets[key] = row
            placed = False
            for i, ub in enumerate(self.buckets):
                if value <= ub:
                    row[i] += 1.0
                    placed = True
                    break
            if not placed:
                row[len(self.buckets)] += 1.0
            row[-2] += float(value)
            row[-1] += 1.0

    def time(self, labels: dict[str, str] | None = None) -> _Timer:
        """Context manager that observes elapsed seconds on exit.

        Usage::

            with metrics.order_latency.time():
                await adapter.market_order(...)
        """
        return _Timer(self, labels)

    def stats(
        self, labels: dict[str, str] | None = None,
    ) -> tuple[float, float]:
        """Return (sum, count) for tests."""
        key = self._normalise_labels(labels)
        with self._lock:
            row = self._buckets.get(key)
            if row is None:
                return (0.0, 0.0)
            return (row[-2], row[-1])

    def render(self) -> list[str]:
        out = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            items = list(self._buckets.items())
        if not items:
            return out
        for key, row in sorted(items):
            cum = 0.0
            for i, ub in enumerate(self.buckets):
                cum += row[i]
                bucket_labels = dict(key) | {"le": _format_value(ub)}
                key_b = tuple(sorted(bucket_labels.items()))
                out.append(
                    f"{self.name}_bucket{_format_labels(key_b)} "
                    f"{_format_value(cum)}"
                )
            cum += row[len(self.buckets)]
            inf_labels = dict(key) | {"le": "+Inf"}
            key_inf = tuple(sorted(inf_labels.items()))
            out.append(
                f"{self.name}_bucket{_format_labels(key_inf)} "
                f"{_format_value(cum)}"
            )
            out.append(
                f"{self.name}_sum{_format_labels(key)} "
                f"{_format_value(row[-2])}"
            )
            out.append(
                f"{self.name}_count{_format_labels(key)} "
                f"{_format_value(row[-1])}"
            )
        return out


@dataclass
class _Timer:
    """Tiny context manager returned by ``Histogram.time``."""

    histogram: Histogram
    labels: dict[str, str] | None = None
    _started: float = 0.0

    def __enter__(self) -> _Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        elapsed = max(0.0, time.perf_counter() - self._started)
        try:
            self.histogram.observe(elapsed, labels=self.labels)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "metric %r observe(%.6f) failed: %s",
                self.histogram.name, elapsed, e,
            )


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


@dataclass
class MetricsRegistry:
    """Holds many metrics and renders them as one Prometheus payload.

    Tests can pass ``namespace="altcoin_agent"`` to keep all metric
    names rooted under one prefix. ``register`` rejects duplicate names
    so a typo at startup blows up loudly instead of silently shadowing.
    """

    namespace: str = ""
    _metrics: dict[str, _MetricBase] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _full_name(self, name: str) -> str:
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name

    def register(self, metric: _MetricBase) -> _MetricBase:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(
                    f"metric {metric.name!r} already registered"
                )
            self._metrics[metric.name] = metric
        return metric

    def counter(
        self,
        name: str,
        help_text: str,
        label_names: Iterable[str] = (),
    ) -> Counter:
        c = Counter(
            name=self._full_name(name),
            help_text=help_text,
            label_names=tuple(label_names),
        )
        return self.register(c)  # type: ignore[return-value]

    def gauge(
        self,
        name: str,
        help_text: str,
        label_names: Iterable[str] = (),
    ) -> Gauge:
        g = Gauge(
            name=self._full_name(name),
            help_text=help_text,
            label_names=tuple(label_names),
        )
        return self.register(g)  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        help_text: str,
        buckets: Iterable[float] = DEFAULT_LATENCY_BUCKETS_SEC,
        label_names: Iterable[str] = (),
    ) -> Histogram:
        h = Histogram(
            name=self._full_name(name),
            help_text=help_text,
            buckets=tuple(buckets),
            label_names=tuple(label_names),
        )
        return self.register(h)  # type: ignore[return-value]

    def get(self, name: str) -> _MetricBase | None:
        return self._metrics.get(self._full_name(name))

    def render(self) -> str:
        """Return the full Prometheus exposition for all metrics."""
        with self._lock:
            ordered = sorted(self._metrics.items())
        lines: list[str] = []
        for _, m in ordered:
            try:
                lines.extend(m.render())
            except Exception as e:  # pragma: no cover - defensive
                lines.append(f"# RENDER_ERROR {m.name}: {e}")
                logger.warning(
                    "metric %r render failed: %s", m.name, e,
                )
        return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------- #
# Default registry preset
# --------------------------------------------------------------------- #


@dataclass
class DefaultMetrics:
    """The ~30 metrics Phase B.2.1 prescribes, attached to one registry.

    Built once at startup; passed by reference into the components that
    actually observe values. Each metric's name is namespaced under
    ``altcoin_agent_`` automatically so the prefix shows up in
    Prometheus regardless of whether the registry is constructed with
    or without an explicit namespace argument.
    """

    registry: MetricsRegistry

    # ---- system ----
    up: Gauge = field(init=False)
    uptime_sec: Gauge = field(init=False)
    fuser_alive: Gauge = field(init=False)
    screener_alive: Gauge = field(init=False)
    reconciliation_complete: Gauge = field(init=False)

    # ---- pipeline counters ----
    rule_events_total: Counter = field(init=False)
    high_priority_signals_total: Counter = field(init=False)
    orders_placed_total: Counter = field(init=False)
    orders_rejected_total: Counter = field(init=False)
    closed_positions_total: Counter = field(init=False)
    open_positions: Gauge = field(init=False)

    # ---- order execution ----
    order_retries_total: Counter = field(init=False)
    partial_fills_total: Counter = field(init=False)
    stop_replace_failures_total: Counter = field(init=False)
    naked_emergency_close_total: Counter = field(init=False)

    # ---- LLM ----
    llm_consults_total: Counter = field(init=False)
    llm_consults_skipped_total: Counter = field(init=False)
    llm_tokens_consumed_total: Counter = field(init=False)
    llm_budget_used_pct: Gauge = field(init=False)

    # ---- risk + safety ----
    halt_engaged: Gauge = field(init=False)
    cooldown_symbols: Gauge = field(init=False)
    consecutive_losses_max: Gauge = field(init=False)
    daily_drawdown_pct: Gauge = field(init=False)
    daily_stoploss_hits: Gauge = field(init=False)

    # ---- miss-penalty + reflection ----
    missed_pumps_total: Counter = field(init=False)
    reflection_mode_active: Gauge = field(init=False)

    # ---- DLQ ----
    dlq_writes_total: Counter = field(init=False)
    dlq_size: Gauge = field(init=False)

    # ---- histograms (latency / lag) ----
    ws_lag_seconds: Histogram = field(init=False)
    event_loop_lag_seconds: Histogram = field(init=False)
    fuser_latency_seconds: Histogram = field(init=False)
    risk_gate_latency_seconds: Histogram = field(init=False)
    order_latency_seconds: Histogram = field(init=False)
    llm_latency_seconds: Histogram = field(init=False)
    persistence_save_seconds: Histogram = field(init=False)

    def __post_init__(self) -> None:
        r = self.registry
        # System
        self.up = r.gauge("up", "1 if all subsystems alive AND reconciled")
        self.uptime_sec = r.gauge("uptime_sec", "Seconds since boot")
        self.fuser_alive = r.gauge(
            "fuser_alive", "1 if the fuser worker is running",
        )
        self.screener_alive = r.gauge(
            "screener_alive", "1 if the screener worker is running",
        )
        self.reconciliation_complete = r.gauge(
            "reconciliation_complete",
            "1 if startup reconciler completed successfully",
        )
        # Pipeline counters
        self.rule_events_total = r.counter(
            "rule_events_total",
            "Raw screener events seen",
            label_names=("kind",),
        )
        self.high_priority_signals_total = r.counter(
            "high_priority_signals_total",
            "Fused signals that crossed the high-priority threshold",
            label_names=("direction",),
        )
        self.orders_placed_total = r.counter(
            "orders_placed_total",
            "Entry orders placed (post-gate)",
            label_names=("symbol", "side"),
        )
        self.orders_rejected_total = r.counter(
            "orders_rejected_total",
            "Entries rejected by the gate or executor",
            label_names=("reason",),
        )
        self.closed_positions_total = r.counter(
            "closed_positions_total",
            "Positions closed (any reason)",
            label_names=("reason",),
        )
        self.open_positions = r.gauge(
            "open_positions", "Currently open positions",
        )
        # Order execution
        self.order_retries_total = r.counter(
            "order_retries_total",
            "Order placement retries (entry/stop/cancel)",
            label_names=("op", "outcome"),
        )
        self.partial_fills_total = r.counter(
            "partial_fills_total",
            "Market entry partial-fill events",
        )
        self.stop_replace_failures_total = r.counter(
            "stop_replace_failures_total",
            "Trailing stop tighten/replace failures",
            label_names=("restored",),
        )
        self.naked_emergency_close_total = r.counter(
            "naked_emergency_close_total",
            "Emergency market closes triggered when no resting stop existed",
        )
        # LLM
        self.llm_consults_total = r.counter(
            "llm_consults_total",
            "LLM consults that produced a verdict",
        )
        self.llm_consults_skipped_total = r.counter(
            "llm_consults_skipped_total",
            "LLM consults skipped or failed",
            label_names=("reason",),
        )
        self.llm_tokens_consumed_total = r.counter(
            "llm_tokens_consumed_total",
            "Estimated LLM tokens consumed (signal + training)",
            label_names=("kind",),
        )
        self.llm_budget_used_pct = r.gauge(
            "llm_budget_used_pct",
            "Fraction of monthly token budget used (0..1)",
        )
        # Risk + safety
        self.halt_engaged = r.gauge(
            "halt_engaged", "1 if account.global_trading_halted",
        )
        self.cooldown_symbols = r.gauge(
            "cooldown_symbols",
            "Number of symbols currently in per-symbol cooldown",
        )
        self.consecutive_losses_max = r.gauge(
            "consecutive_losses_max",
            "Max per-symbol consecutive-loss streak",
        )
        self.daily_drawdown_pct = r.gauge(
            "daily_drawdown_pct", "Today's drawdown as a fraction (0..1)",
        )
        self.daily_stoploss_hits = r.gauge(
            "daily_stoploss_hits", "Stoploss fills today",
        )
        # Miss-penalty + reflection
        self.missed_pumps_total = r.counter(
            "missed_pumps_total",
            "Audited rejections that turned out to be missed pumps",
            label_names=("reason",),
        )
        self.reflection_mode_active = r.gauge(
            "reflection_mode_active",
            "1 if reflection-mode suspension window is active",
        )
        # DLQ
        self.dlq_writes_total = r.counter(
            "dlq_writes_total",
            "Entries written to the dead-letter queue",
            label_names=("kind",),
        )
        self.dlq_size = r.gauge(
            "dlq_size", "Approximate row count of the active DLQ file",
        )
        # Histograms
        self.ws_lag_seconds = r.histogram(
            "ws_lag_seconds",
            "Wall-clock lag between WS event timestamp and local arrival",
            buckets=DEFAULT_FAST_BUCKETS_SEC,
        )
        self.event_loop_lag_seconds = r.histogram(
            "event_loop_lag_seconds",
            "Sampled asyncio event-loop lag",
            buckets=DEFAULT_FAST_BUCKETS_SEC,
        )
        self.fuser_latency_seconds = r.histogram(
            "fuser_latency_seconds",
            "ScoreFuser.on_rule_signal end-to-end latency",
            buckets=DEFAULT_FAST_BUCKETS_SEC,
        )
        self.risk_gate_latency_seconds = r.histogram(
            "risk_gate_latency_seconds",
            "RiskGate.evaluate latency",
            buckets=DEFAULT_FAST_BUCKETS_SEC,
        )
        self.order_latency_seconds = r.histogram(
            "order_latency_seconds",
            "Adapter market_order / place_stop_order latency",
            label_names=("op",),
        )
        self.llm_latency_seconds = r.histogram(
            "llm_latency_seconds",
            "DeepSeek round-trip latency",
        )
        self.persistence_save_seconds = r.histogram(
            "persistence_save_seconds",
            "AccountPersistor.save latency",
            buckets=DEFAULT_FAST_BUCKETS_SEC,
        )


def build_default_registry(namespace: str = "altcoin_agent") -> DefaultMetrics:
    """Construct a fresh registry with the full Phase B.2.1 metric set."""
    return DefaultMetrics(registry=MetricsRegistry(namespace=namespace))
