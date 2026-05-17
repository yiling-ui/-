"""tracing.py — OpenTelemetry distributed-trace helper (Phase B.6).

Phase B.6 of ``MISS_PENALTY_AND_PRODUCTION_PLAN.md``: a single decision
takes ~10 hops through the daemon (screener → fuser → LLM → gate →
executor → trailing). Phase B.2.2 already gave each decision a short
``trace_id`` plumbed through ``structured_log`` so an operator can
``grep`` JSON logs for one decision; this module adds the second half
of the picture — a real distributed trace exporter so the same hops
show up as nested spans in any OTel-compatible backend (Jaeger,
Tempo, OTLP collector → Honeycomb / DataDog / cloud-native APMs).

Design constraints
------------------

* **OTel is opt-in via the ``[otel]`` extra.** The module must import
  cleanly, run cleanly, and pass ALL tests when ``opentelemetry`` is
  not installed. Production operators that flip ``tracing_enabled``
  in ``app.yaml`` are responsible for shipping the extra alongside.
* **Trace-id correlation with Phase B.2.2.** When a real span is
  active, we copy its OTel trace-id (a 16-byte hex) into the existing
  short ``structured_log`` contextvar so every JSON log line emitted
  while inside a span carries the same hex prefix you'll see in the
  span exporter — operators can correlate logs and spans without
  guesswork.
* **Stdlib-only no-op.** When OTel is missing OR ``configure_tracing``
  was never called, every helper returns a ``_NullSpan`` whose
  ``set_attribute`` / ``record_exception`` / ``set_status`` are
  cheap no-ops. The trading hot path takes one extra Python call
  per span which is sub-microsecond.
* **Lifecycle is explicit.** ``configure_tracing`` initialises the
  global :class:`Tracer`; ``shutdown_tracing`` flushes the SDK
  exporter at daemon shutdown so in-flight spans aren't lost on
  SIGTERM.
* **No global mutation surprises.** The module owns one
  ``_GLOBAL_TRACER`` slot; tests reset it via ``reset_tracing`` so
  fixtures don't leak between cases.

Public surface
--------------

::

    from altcoin_agent.observability.tracing import (
        Tracer,
        configure_tracing,
        get_tracer,
        reset_tracing,
        shutdown_tracing,
        traced,
    )

The functions on the module level are convenience helpers that look
up :func:`get_tracer()` and call its method — most call sites use
those rather than holding a reference.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from altcoin_agent.observability.structured_log import bind_trace_id

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Optional OTel import
# --------------------------------------------------------------------- #
#
# We deliberately do NOT raise on ImportError. ``OTEL_AVAILABLE`` flips
# True only when every symbol the runtime path uses is importable; if
# any one of them is missing we degrade to the no-op tracer.

OTEL_AVAILABLE: bool = False
_otel_trace = None  # filled in below when available
_otel_status = None
_otel_status_code = None

try:  # pragma: no cover - import guard
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status as _otel_status
    from opentelemetry.trace import StatusCode as _otel_status_code

    OTEL_AVAILABLE = True
except ImportError:
    _otel_trace = None
    _otel_status = None
    _otel_status_code = None


# Type returned by ``Tracer.start_span``. Either an OTel ``Span`` (when
# enabled) OR our :class:`_NullSpan` (when disabled). Both expose the
# subset of the Span API the daemon relies on (set_attribute,
# record_exception, set_status). Typed loosely to keep call sites
# clean.
SpanLike = Any

T = TypeVar("T")


SpanKind = Literal["internal", "producer", "consumer", "client", "server"]


# --------------------------------------------------------------------- #
# No-op span (used both when OTel is missing AND when configure_tracing
# hasn't been called).
# --------------------------------------------------------------------- #


@dataclass
class _NullSpan:
    """Cheap stand-in for an OTel ``Span``.

    Implements ``set_attribute`` / ``record_exception`` / ``set_status``
    as no-ops so call sites don't need to branch on tracer availability.
    """

    name: str = ""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        return None

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        return None

    def record_exception(self, exc: BaseException, attributes: dict[str, Any] | None = None) -> None:
        return None

    def set_status(self, status: Any, description: str | None = None) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def is_recording(self) -> bool:
        return False

    # OTel's Span exposes ``end()``; a NullSpan never needs it but we
    # implement it anyway so a call site that explicitly ends a span
    # works in both modes.
    def end(self) -> None:
        return None


# --------------------------------------------------------------------- #
# Tracer
# --------------------------------------------------------------------- #


@dataclass
class Tracer:
    """Wrapper around an OTel ``Tracer`` with a graceful no-op fallback.

    The wrapper holds a reference to the OTel ``TracerProvider`` so we
    can flush + shut down at daemon exit. ``service_name`` is recorded
    as the ``Resource`` attribute every span carries — operators
    filter on it to separate the trading daemon's spans from any
    sidecar (metrics scraper, dashboard) running in the same OTel
    collector.
    """

    service_name: str = "altcoin-agent"
    enabled: bool = False
    _otel_tracer: Any = None
    _otel_provider: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ----------------------------------------------------------------- #
    # Span construction
    # ----------------------------------------------------------------- #

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind = "internal",
    ) -> Iterator[SpanLike]:
        """Context manager that opens a span for the duration of the
        ``with`` block.

        When the tracer is enabled the OTel SDK creates a real span,
        binds the resulting trace-id (in compact form) into our
        ``structured_log`` contextvar so any JSON log line emitted
        inside the block carries the same hex correlation key, and
        runs the block. Exceptions raised inside the block are
        captured on the span and re-raised so the caller sees the
        same exception flow as without tracing.

        When the tracer is disabled (no OTel installed OR
        ``configure_tracing`` not called) we yield a :class:`_NullSpan`
        so call sites don't need to branch on availability.
        """
        if not self.enabled or self._otel_tracer is None:
            null = _NullSpan(name=name)
            if attributes:
                null.set_attributes(attributes)
            yield null
            return

        otel_kind = _kind_to_otel(kind)
        # OTel's start_as_current_span returns a context manager that
        # also sets the span as the active one in the current
        # context — exactly the propagation we want for nested
        # ``with`` blocks across hops.
        with self._otel_tracer.start_as_current_span(
            name,
            kind=otel_kind,
            attributes=_clean_attrs(attributes) if attributes else None,
        ) as span:
            # Synchronise our short structured-log trace-id with the
            # OTel trace-id (rightmost 12 hex chars; OTel's id is 32
            # chars total, our short form is the trailing 12). This
            # keeps the log line's ``trace_id`` field stable even
            # when nested spans push/pop on the OTel side.
            try:
                ctx = span.get_span_context()
                if ctx is not None and ctx.trace_id != 0:
                    short = format(ctx.trace_id, "032x")[-12:]
                    bind_trace_id(short)
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                yield span
            except Exception as e:
                # Capture exception details on the span, then re-raise.
                try:
                    span.record_exception(e)
                    if _otel_status is not None and _otel_status_code is not None:
                        span.set_status(
                            _otel_status(_otel_status_code.ERROR, str(e)),
                        )
                except Exception:  # pragma: no cover - defensive
                    pass
                raise

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #

    def shutdown(self) -> None:
        """Flush + shut down the OTel SDK exporter.

        Safe to call multiple times; idempotent. Returns silently
        when tracing is disabled or never initialised.
        """
        with self._lock:
            provider = self._otel_provider
            self._otel_provider = None
            self._otel_tracer = None
            self.enabled = False
        if provider is not None:
            try:
                # ``shutdown`` blocks until the in-flight export
                # batches are drained; the SDK ships a 30s default
                # timeout that is plenty for a graceful shutdown.
                provider.shutdown()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Tracer.shutdown: provider shutdown failed: %s", e,
                )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _kind_to_otel(kind: SpanKind) -> Any:
    """Translate our short ``SpanKind`` strings into the OTel enum."""
    if not OTEL_AVAILABLE or _otel_trace is None:
        return None
    # OTel's enum lives at trace.SpanKind.{INTERNAL, CLIENT, SERVER, ...}.
    mapping = {
        "internal": _otel_trace.SpanKind.INTERNAL,
        "producer": _otel_trace.SpanKind.PRODUCER,
        "consumer": _otel_trace.SpanKind.CONSUMER,
        "client": _otel_trace.SpanKind.CLIENT,
        "server": _otel_trace.SpanKind.SERVER,
    }
    return mapping.get(kind, _otel_trace.SpanKind.INTERNAL)


def _clean_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """OTel rejects values it can't serialise (lists of mixed types,
    custom objects). Fall back to ``str()`` for anything unusual so a
    debug attribute can never crash the span.

    Ints, floats, bools, and strings pass through verbatim. Lists of
    those primitive types are kept as-is. Everything else is
    string-coerced.
    """
    cleaned: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            cleaned[str(k)] = v
            continue
        if isinstance(v, (list, tuple)):
            # OTel only allows homogeneous sequences of primitives.
            primitives = [
                x for x in v if isinstance(x, (str, bool, int, float))
            ]
            if primitives and len(primitives) == len(list(v)):
                cleaned[str(k)] = list(primitives)
                continue
            cleaned[str(k)] = ",".join(str(x) for x in v)
            continue
        # Everything else: best-effort string coercion.
        try:
            cleaned[str(k)] = str(v)
        except Exception:  # pragma: no cover - defensive
            cleaned[str(k)] = repr(v)
    return cleaned


# --------------------------------------------------------------------- #
# Module-level singleton + initialiser
# --------------------------------------------------------------------- #


_GLOBAL_TRACER: Tracer = Tracer()  # noqa: E305 (module-level singleton)


def configure_tracing(
    *,
    service_name: str = "altcoin-agent",
    otlp_endpoint: str | None = None,
    otlp_insecure: bool = True,
    console_exporter: bool = False,
    sampler_arg: float = 1.0,
    extra_resource_attrs: dict[str, str] | None = None,
) -> Tracer:
    """Initialise the global :class:`Tracer`.

    Returns the initialised tracer (also accessible via
    :func:`get_tracer`). Idempotent: a second call shuts the previous
    SDK provider down before re-initialising — useful for tests that
    swap exporters between cases.

    When OTel isn't installed, this function logs once at WARNING and
    returns a disabled tracer; the daemon continues to operate, just
    without distributed traces.

    Parameters
    ----------
    service_name
        The ``Resource``'s ``service.name`` attribute. Defaults to
        ``altcoin-agent``; operators running multiple instances
        (paper-trade alongside live) should override.
    otlp_endpoint
        gRPC endpoint of the OTLP collector (e.g. ``otel-collector:4317``).
        ``None`` ⇒ no OTLP exporter is attached, which is what tests
        and the ``console_exporter=True`` path want.
    otlp_insecure
        gRPC ``insecure=`` flag. Default True for local-collector
        deployments; flip to False when shipping to a TLS-protected
        endpoint.
    console_exporter
        When True, attaches a synchronous console exporter that prints
        each span on stdout. Useful for tests + the operator's
        first-time-bring-up sanity check (``OTEL_CONSOLE=1``).
    sampler_arg
        ParentBasedTraceIdRatio sampler ratio in [0.0, 1.0]. Defaults
        to 1.0 — every span is recorded. Operators on a free OTel
        backend with a quota should drop this to 0.05–0.1.
    extra_resource_attrs
        Additional ``Resource`` attributes (e.g. ``deployment.env``,
        ``service.instance.id``). Strings only.
    """
    global _GLOBAL_TRACER

    if not OTEL_AVAILABLE:
        logger.warning(
            "configure_tracing: opentelemetry not installed; tracing "
            "stays disabled. Install with `pip install -e '.[otel]'` "
            "to enable.",
        )
        _GLOBAL_TRACER = Tracer(service_name=service_name, enabled=False)
        return _GLOBAL_TRACER

    # Lazy imports — only reached when OTEL_AVAILABLE is True. Keeping
    # them inside the function means importing this module never
    # forces the SDK + exporter dance.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import (
        ParentBased,
        TraceIdRatioBased,
    )

    # Tear down any previous provider so a re-configure flushes
    # in-flight spans before swapping exporters.
    if _GLOBAL_TRACER._otel_provider is not None:
        try:
            _GLOBAL_TRACER.shutdown()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "configure_tracing: previous shutdown raised: %s", e,
            )

    resource_attrs: dict[str, Any] = {"service.name": service_name}
    if extra_resource_attrs:
        for k, v in extra_resource_attrs.items():
            resource_attrs[str(k)] = str(v)
    resource = Resource.create(resource_attrs)

    sampler = ParentBased(TraceIdRatioBased(max(0.0, min(1.0, sampler_arg))))
    provider = TracerProvider(resource=resource, sampler=sampler)

    # OTLP exporter wiring. The gRPC exporter is robust enough that we
    # don't add the HTTP variant; operators that need HTTP can wrap
    # this module via the same TracerProvider API.
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            otlp = OTLPSpanExporter(
                endpoint=otlp_endpoint, insecure=otlp_insecure,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp))
            logger.info(
                "OTel: OTLP exporter wired (endpoint=%s, insecure=%s)",
                otlp_endpoint, otlp_insecure,
            )
        except Exception as e:  # pragma: no cover - exporter wiring
            logger.warning(
                "configure_tracing: OTLP wiring failed (%s); "
                "continuing with whatever console exporter was set", e,
            )

    if console_exporter:
        # Synchronous so tests reading captured stdout see the span
        # output without sleeping.
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel: console exporter wired")

    # ``trace.set_tracer_provider`` registers our provider as the global
    # one so any third-party libraries that read ``trace.get_tracer``
    # also see it. Idempotent: OTel itself coalesces repeat sets.
    if _otel_trace is not None:
        _otel_trace.set_tracer_provider(provider)
    otel_tracer = provider.get_tracer(service_name)

    _GLOBAL_TRACER = Tracer(
        service_name=service_name,
        enabled=True,
        _otel_tracer=otel_tracer,
        _otel_provider=provider,
    )
    logger.info(
        "configure_tracing: enabled (service=%s, sampler=%.2f)",
        service_name, sampler_arg,
    )
    return _GLOBAL_TRACER


def get_tracer() -> Tracer:
    """Return the active :class:`Tracer`.

    Always returns a tracer — disabled when configure_tracing hasn't
    run, fully-armed otherwise. Call sites can therefore do::

        with get_tracer().start_span("foo"):
            ...

    without an upfront None-check.
    """
    return _GLOBAL_TRACER


def reset_tracing() -> None:
    """Tear down the global tracer + replace it with a fresh disabled
    instance. Used by tests so a per-case ``configure_tracing`` call
    starts from a clean slate."""
    global _GLOBAL_TRACER
    try:
        _GLOBAL_TRACER.shutdown()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("reset_tracing: shutdown raised: %s", e)
    _GLOBAL_TRACER = Tracer()


def shutdown_tracing() -> None:
    """Flush + shutdown the global tracer. Wired into ``App._shutdown``
    so SIGTERM doesn't drop in-flight spans."""
    try:
        _GLOBAL_TRACER.shutdown()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("shutdown_tracing failed (swallowed): %s", e)


# --------------------------------------------------------------------- #
# Convenience module-level wrappers (no need to call get_tracer())
# --------------------------------------------------------------------- #


@contextmanager
def start_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind = "internal",
) -> Iterator[SpanLike]:
    """Module-level alias of :meth:`Tracer.start_span` operating on the
    global tracer.

    Most call sites should reach for this instead of ``get_tracer()``;
    it keeps imports tight at the call site.
    """
    with _GLOBAL_TRACER.start_span(
        name, attributes=attributes, kind=kind,
    ) as span:
        yield span


# --------------------------------------------------------------------- #
# Decorator: @traced("name", attributes=...)
# --------------------------------------------------------------------- #


def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind = "internal",
) -> Callable[[Callable[..., T | Awaitable[T]]], Callable[..., T | Awaitable[T]]]:
    """Decorator that wraps a sync or async function in a span.

    When the global tracer is disabled the decorator is a near-no-op
    (one extra context-manager entry/exit) so it's cheap to leave on
    in production. The span name defaults to ``f"{module}.{qualname}"``
    when not supplied.

    Async functions are detected via ``inspect.iscoroutinefunction``
    so the decorator returns an async wrapper that awaits the call
    inside the span. Generators / async generators are NOT supported
    — wrap them by hand if needed (the closing semantics for
    generator spans are subtle and out of scope for V1).
    """

    def decorator(fn: Callable[..., T | Awaitable[T]]) -> Callable[..., T | Awaitable[T]]:
        span_name = name or _default_span_name(fn)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _GLOBAL_TRACER.start_span(
                    span_name, attributes=attributes, kind=kind,
                ):
                    return await fn(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _GLOBAL_TRACER.start_span(
                span_name, attributes=attributes, kind=kind,
            ):
                return fn(*args, **kwargs)
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _default_span_name(fn: Callable[..., Any]) -> str:
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "fn")
    if module:
        return f"{module}.{qualname}"
    return qualname


# --------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------- #


__all__ = [
    "OTEL_AVAILABLE",
    "SpanKind",
    "Tracer",
    "configure_tracing",
    "get_tracer",
    "reset_tracing",
    "shutdown_tracing",
    "start_span",
    "traced",
]
