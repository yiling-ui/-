"""observability — Phase B.2 monitoring & logging primitives.

Three sibling modules that the daemon and risk layer can import freely:

* ``metrics``      — :class:`MetricsRegistry` with histogram/counter/gauge
                     primitives that render Prometheus text format.
* ``structured_log``  — :class:`StructuredLogger` adapter + ``trace_id``
                     context variable for end-to-end tracing of one
                     decision through the pipeline.
* ``dlq``          — :class:`DeadLetterQueue` for unrecoverable signal /
                     order failures (file-rotated JSONL, swallowed
                     errors).

Phase B.2 in ``MISS_PENALTY_AND_PRODUCTION_PLAN.md`` calls out the gaps
this package closes:

* Only 5–6 simple gauges existed in ``main.make_health_app`` — far short
  of the 30+ metrics the operator needs to plot SLOs and per-reason
  reject rates.
* No structured logging — text logs make it impossible to correlate the
  ~10 hops a single screener event takes through the pipeline.
* No dead-letter queue — failed signals are silently dropped after
  bumping ``orders_rejected``, leaving no replay/postmortem trail.

The modules are deliberately stdlib-only (no prometheus_client / no
structlog) so the package stays light and the wire format is well
understood when reviewing test fixtures.
"""

from __future__ import annotations

from altcoin_agent.observability.dlq import (
    DeadLetterQueue,
    DLQEntry,
)
from altcoin_agent.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    build_default_registry,
)
from altcoin_agent.observability.structured_log import (
    StructuredLogger,
    bind_trace_id,
    current_trace_id,
    new_trace_id,
)

__all__ = [
    "Counter",
    "DeadLetterQueue",
    "DLQEntry",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "StructuredLogger",
    "bind_trace_id",
    "build_default_registry",
    "current_trace_id",
    "new_trace_id",
]
