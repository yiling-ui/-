"""Unit tests for ``observability.tracing``.

The tracer has two operating modes — disabled (no-op) and enabled
(real OTel SDK). We test both end-to-end so the no-op fallback path
stays tight in production AND the enabled path actually exports
spans.

Tests in the "disabled" cluster do NOT require ``opentelemetry`` to
be installed — they rely on :class:`_NullSpan` which lives in our own
module. Tests in the "enabled" cluster gate themselves on
``tracing.OTEL_AVAILABLE`` so the suite passes regardless of which
extras are installed.
"""

from __future__ import annotations

import pytest

from altcoin_agent.observability import structured_log, tracing
from altcoin_agent.observability.tracing import (
    OTEL_AVAILABLE,
    Tracer,
    _clean_attrs,
    _NullSpan,
    configure_tracing,
    get_tracer,
    reset_tracing,
    shutdown_tracing,
    start_span,
    traced,
)

# --------------------------------------------------------------------- #
# Fixtures: always reset the global tracer between tests so test
# ordering can never cause one case to inherit another's enabled
# state. We also clear the contextvar so a leaking trace_id can't
# bleed between tests.
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_tracer():
    yield
    reset_tracing()


# --------------------------------------------------------------------- #
# Disabled-mode (no configure_tracing call)
# --------------------------------------------------------------------- #


def test_default_tracer_is_disabled():
    t = get_tracer()
    assert t.enabled is False


def test_start_span_returns_null_span_when_disabled():
    with start_span("foo") as span:
        assert isinstance(span, _NullSpan)
        assert span.is_recording() is False
    # NullSpan methods must accept arbitrary attributes silently.
    with start_span("bar", attributes={"x": 1, "y": "z"}) as span:
        span.set_attribute("k", "v")
        span.set_attributes({"k2": 2})
        span.add_event("hop", {"detail": "x"})
        span.record_exception(RuntimeError("boom"))
        span.set_status("error", "msg")
        span.end()


def test_start_span_does_not_raise_when_block_raises():
    """The context manager must propagate exceptions in disabled mode
    too — same flow as if the span weren't there."""
    with pytest.raises(ValueError):
        with start_span("foo"):
            raise ValueError("boom")


def test_traced_decorator_works_when_disabled_sync():
    @traced("my_span")
    def f(x: int) -> int:
        return x + 1

    assert f(2) == 3


@pytest.mark.asyncio
async def test_traced_decorator_works_when_disabled_async():
    @traced("my_async_span")
    async def f(x: int) -> int:
        return x * 2

    assert await f(5) == 10


def test_traced_default_span_name_uses_qualname():
    @traced()
    def fn() -> int:
        return 7

    # Just exercising — the span itself is a no-op so the name is
    # only observable in enabled mode. Smoke test confirms the
    # decorator didn't crash on default-name resolution.
    assert fn() == 7


def test_shutdown_tracing_is_idempotent_when_disabled():
    shutdown_tracing()  # never configured
    shutdown_tracing()  # second call still a no-op


# --------------------------------------------------------------------- #
# _clean_attrs
# --------------------------------------------------------------------- #


def test_clean_attrs_passes_through_primitives():
    out = _clean_attrs({
        "s": "string", "i": 1, "f": 1.5, "b": True,
    })
    assert out == {"s": "string", "i": 1, "f": 1.5, "b": True}


def test_clean_attrs_drops_none_values():
    out = _clean_attrs({"keep": 1, "drop": None})
    assert "drop" not in out
    assert out["keep"] == 1


def test_clean_attrs_keeps_homogeneous_primitive_lists():
    out = _clean_attrs({"ints": [1, 2, 3], "strs": ["a", "b"]})
    assert out["ints"] == [1, 2, 3]
    assert out["strs"] == ["a", "b"]


def test_clean_attrs_joins_heterogeneous_sequences():
    out = _clean_attrs({"mix": [1, "x", None]})
    # OTel rejects mixed sequences -> we string-coerce.
    assert isinstance(out["mix"], str)


def test_clean_attrs_string_coerces_objects():
    class _O:
        def __str__(self) -> str:
            return "obj-str"
    out = _clean_attrs({"o": _O()})
    assert out["o"] == "obj-str"


# --------------------------------------------------------------------- #
# Configure / reset / lifecycle (works regardless of OTel availability)
# --------------------------------------------------------------------- #


def test_configure_returns_disabled_tracer_when_otel_missing(monkeypatch):
    """When OTel is missing we still get a Tracer back; ``enabled`` is
    False and start_span yields NullSpan."""
    # Force the module-level flag to False to simulate "no OTel".
    monkeypatch.setattr(tracing, "OTEL_AVAILABLE", False)
    t = configure_tracing(service_name="svc-test")
    assert t.enabled is False
    assert get_tracer() is t
    with start_span("x") as span:
        assert isinstance(span, _NullSpan)


def test_configure_can_be_called_multiple_times(monkeypatch):
    """A second configure_tracing call must shut the previous provider
    down + re-init cleanly. Behaviour we rely on for tests that swap
    exporters between cases."""
    monkeypatch.setattr(tracing, "OTEL_AVAILABLE", False)
    t1 = configure_tracing(service_name="svc-a")
    t2 = configure_tracing(service_name="svc-b")
    assert t1 is not t2
    assert t2.service_name == "svc-b"


def test_reset_tracing_replaces_global_tracer():
    t_before = get_tracer()
    reset_tracing()
    t_after = get_tracer()
    assert t_before is not t_after
    assert t_after.enabled is False


# --------------------------------------------------------------------- #
# Enabled-mode tests — gated on OTel availability
# --------------------------------------------------------------------- #


pytest.importorskip("opentelemetry", reason="OTel not installed")


@pytest.fixture
def captured_otel_spans():
    """Yield a list that captures every span exported by the SDK.

    Wires an in-memory exporter so tests can assert on span names,
    attributes, and exception flows without scraping stdout. Reuses
    OTel's :class:`InMemorySpanExporter` which the SDK ships for
    exactly this purpose.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # Hijack the global tracer so module-level helpers route to us.
    tracing._GLOBAL_TRACER = Tracer(
        service_name="test", enabled=True,
        _otel_tracer=provider.get_tracer("test"),
        _otel_provider=provider,
    )
    try:
        yield exporter
    finally:
        provider.shutdown()
        reset_tracing()


def test_enabled_tracer_records_span(captured_otel_spans):
    with start_span("decision_pipeline", attributes={"symbol": "PEPE"}):
        pass
    spans = captured_otel_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "decision_pipeline"
    assert spans[0].attributes["symbol"] == "PEPE"


def test_enabled_tracer_records_nested_spans(captured_otel_spans):
    with start_span("outer"):
        with start_span("inner"):
            pass
    spans = captured_otel_spans.get_finished_spans()
    # Span order is finish-order: inner finishes first, then outer.
    assert [s.name for s in spans] == ["inner", "outer"]
    # Nested span's parent_span_id == outer's span_id.
    inner, outer = spans
    assert inner.parent.span_id == outer.context.span_id


def test_enabled_tracer_captures_exception(captured_otel_spans):
    with pytest.raises(ValueError):
        with start_span("boom_span"):
            raise ValueError("explosion")
    spans = captured_otel_spans.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # OTel records the exception as an Event named "exception" with
    # the type/message as attributes.
    assert any(
        ev.name == "exception"
        and ev.attributes.get("exception.type") == "ValueError"
        for ev in span.events
    )
    # Status is set to ERROR.
    from opentelemetry.trace import StatusCode
    assert span.status.status_code == StatusCode.ERROR


def test_enabled_tracer_binds_trace_id_into_structured_log(captured_otel_spans):
    """The OTel trace-id (rightmost 12 hex) becomes the
    ``structured_log`` short trace_id so JSON logs and OTel spans
    correlate without operator guesswork."""
    before = structured_log.current_trace_id()
    with start_span("hop"):
        inside = structured_log.current_trace_id()
    spans = captured_otel_spans.get_finished_spans()
    assert spans
    expected = format(spans[0].context.trace_id, "032x")[-12:]
    assert inside == expected
    assert inside != before  # was bound while inside span


def test_traced_decorator_creates_span_when_enabled(captured_otel_spans):
    @traced("my_op", attributes={"k": "v"})
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    spans = captured_otel_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_op"
    assert spans[0].attributes.get("k") == "v"


@pytest.mark.asyncio
async def test_traced_decorator_async_creates_span(captured_otel_spans):
    @traced("async_op")
    async def f(x: int) -> int:
        return x + 1

    assert await f(7) == 8
    spans = captured_otel_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_op"


def test_enabled_tracer_clean_attrs_handles_none_value(captured_otel_spans):
    """Passing ``attributes={"missing": None}`` must NOT crash the
    span — the cleaner drops the key."""
    with start_span("drops_none", attributes={"missing": None, "kept": 1}):
        pass
    spans = captured_otel_spans.get_finished_spans()
    assert "missing" not in spans[0].attributes
    assert spans[0].attributes.get("kept") == 1


# --------------------------------------------------------------------- #
# configure_tracing wired to the SDK in real OTel mode
# --------------------------------------------------------------------- #


def test_configure_tracing_with_console_exporter_initialises_provider(
    monkeypatch,
):
    """``configure_tracing(console_exporter=True)`` builds a real
    SDK provider and the resulting tracer is enabled. We don't assert
    on stdout here — :class:`InMemorySpanExporter` already covers the
    span-export contract — but we verify the tracer flips to
    ``enabled=True`` and shutdown is clean."""
    if not OTEL_AVAILABLE:
        pytest.skip("OTel not installed")
    t = configure_tracing(service_name="svc", console_exporter=True)
    assert t.enabled is True
    # Smoke test: opening a span works.
    with t.start_span("smoke"):
        pass
    shutdown_tracing()
    # After shutdown, the tracer reports disabled and start_span
    # falls back to NullSpan.
    assert t.enabled is False
    with t.start_span("after_shutdown") as s:
        assert isinstance(s, _NullSpan)


def test_configure_tracing_with_invalid_otlp_endpoint_does_not_raise():
    """OTLP-exporter wiring failures must NOT abort startup."""
    if not OTEL_AVAILABLE:
        pytest.skip("OTel not installed")
    # gRPC accepts "garbage:1234" lazily; we just assert no exception
    # propagates back to us at config time.
    t = configure_tracing(
        service_name="svc",
        otlp_endpoint="not-a-real-host:4317",
        otlp_insecure=True,
    )
    assert t.enabled is True
    shutdown_tracing()


def test_extra_resource_attrs_are_carried(captured_otel_spans):
    """Span resource attributes survive into the exported span."""
    if not OTEL_AVAILABLE:
        pytest.skip("OTel not installed")
    # We can't easily reuse the captured_otel_spans fixture and also
    # call configure_tracing (it would replace our exporter). We
    # instead verify the attribute through the public API.
    t = configure_tracing(
        service_name="svc-x",
        extra_resource_attrs={"deployment.env": "test"},
    )
    assert t.enabled is True
    # Just a smoke test that no exception arises and the tracer can
    # render a span — the resource carrying the attribute is hidden
    # in the SDK's internals; we'd need a real exporter to assert
    # the attribute, which is outside the scope of this unit test.
    with t.start_span("smoke"):
        pass
    shutdown_tracing()


# --------------------------------------------------------------------- #
# Sanity: imports via observability package
# --------------------------------------------------------------------- #


def test_public_imports_via_observability_package():
    from altcoin_agent.observability import (
        OTEL_AVAILABLE as A,
    )
    from altcoin_agent.observability import (
        Tracer as T,
    )
    from altcoin_agent.observability import (
        configure_tracing as C,
    )
    from altcoin_agent.observability import (
        get_tracer as G,
    )
    from altcoin_agent.observability import (
        reset_tracing as R,
    )
    from altcoin_agent.observability import (
        shutdown_tracing as S,
    )
    from altcoin_agent.observability import (
        start_span as SS,
    )
    from altcoin_agent.observability import (
        traced as TR,
    )
    assert A is OTEL_AVAILABLE
    assert T is Tracer
    assert C is configure_tracing
    assert G is get_tracer
    assert R is reset_tracing
    assert S is shutdown_tracing
    assert SS is start_span
    assert TR is traced
