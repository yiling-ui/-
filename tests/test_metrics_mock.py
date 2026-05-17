"""tests/test_metrics_mock.py — Phase B.2.1 metrics registry tests.

Covers:

* Counter / Gauge / Histogram primitives — inc / set / observe + the
  no-label degenerate path.
* Label cardinality cap (refuses new combos beyond the limit).
* Prometheus text-format rendering — comments, type, label escaping,
  cumulative bucket semantics.
* DefaultMetrics preset shape — at least 30 metric families exist
  under the ``altcoin_agent_*`` namespace.
* Histogram.time() context manager observes elapsed time.

The module is stdlib-only so we don't need to mock the upstream
``prometheus_client``; we assert directly on the rendered text.
"""

from __future__ import annotations

import time

import pytest

from altcoin_agent.observability.metrics import (
    Counter,
    DefaultMetrics,
    Gauge,
    Histogram,
    MetricsRegistry,
    build_default_registry,
)

# --------------------------------------------------------------------- #
# Counter
# --------------------------------------------------------------------- #


def test_counter_increments_no_label() -> None:
    c = Counter("foo_total", "test counter")
    c.inc()
    c.inc(2.5)
    assert c.value() == 3.5
    text = "\n".join(c.render())
    assert "# TYPE foo_total counter" in text
    # 3.5 -> "3.5" (non-integer float)
    assert "foo_total 3.5" in text


def test_counter_zero_emitted_when_no_labels_and_no_obs() -> None:
    c = Counter("foo_total", "test counter")
    text = "\n".join(c.render())
    # No-label counters with no observations still show 0 so dashboards
    # don't complain about missing series.
    assert "foo_total 0" in text


def test_counter_rejects_negative_increment() -> None:
    c = Counter("foo_total", "test")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_counter_with_labels_renders_each_combination() -> None:
    c = Counter("foo_total", "test", label_names=("symbol", "side"))
    c.inc(labels={"symbol": "PEPE", "side": "long"})
    c.inc(labels={"symbol": "PEPE", "side": "long"})
    c.inc(labels={"symbol": "WIF", "side": "short"})
    rendered = c.render()
    text = "\n".join(rendered)
    assert 'foo_total{side="long",symbol="PEPE"} 2' in text
    assert 'foo_total{side="short",symbol="WIF"} 1' in text


def test_counter_label_mismatch_raises() -> None:
    c = Counter("foo_total", "test", label_names=("symbol",))
    with pytest.raises(ValueError):
        c.inc(labels={"symbol": "X", "extra": "Y"})
    with pytest.raises(ValueError):
        c.inc(labels={})


def test_counter_cardinality_cap_silently_drops_overflow() -> None:
    c = Counter(
        "foo_total", "test",
        label_names=("symbol",),
        max_label_cardinality=3,
    )
    for sym in ("A", "B", "C"):
        c.inc(labels={"symbol": sym})
    # 4th distinct symbol must be dropped, but increments to existing
    # symbols continue to work.
    c.inc(labels={"symbol": "D"})
    c.inc(labels={"symbol": "A"})
    assert c.value(labels={"symbol": "A"}) == 2.0
    assert c.value(labels={"symbol": "D"}) == 0.0


# --------------------------------------------------------------------- #
# Gauge
# --------------------------------------------------------------------- #


def test_gauge_set_inc_dec() -> None:
    g = Gauge("foo", "test gauge")
    g.set(5.0)
    g.inc()
    g.dec(2)
    assert g.value() == 4.0
    text = "\n".join(g.render())
    assert "# TYPE foo gauge" in text
    assert "foo 4" in text


def test_gauge_with_labels_render() -> None:
    g = Gauge("active_workers", "test", label_names=("role",))
    g.set(3, labels={"role": "fuser"})
    g.set(1, labels={"role": "screener"})
    text = "\n".join(g.render())
    assert 'active_workers{role="fuser"} 3' in text
    assert 'active_workers{role="screener"} 1' in text


# --------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------- #


def test_histogram_observations_render_cumulative_buckets() -> None:
    h = Histogram("latency_seconds", "test", buckets=(0.01, 0.1, 1.0))
    for v in (0.005, 0.05, 0.5, 5.0):
        h.observe(v)
    rendered = "\n".join(h.render())
    # Cumulative semantics: bucket 0.01 has 1; 0.1 has 2; 1 has 3; +Inf has 4.
    assert 'latency_seconds_bucket{le="0.01"} 1' in rendered
    assert 'latency_seconds_bucket{le="0.1"} 2' in rendered
    assert 'latency_seconds_bucket{le="1"} 3' in rendered
    assert 'latency_seconds_bucket{le="+Inf"} 4' in rendered
    # Sum is the raw sum; count is the number of observations.
    assert "latency_seconds_sum" in rendered
    assert "latency_seconds_count 4" in rendered


def test_histogram_rejects_unsorted_or_infinite_buckets() -> None:
    with pytest.raises(ValueError):
        Histogram("bad", "test", buckets=(1.0, 0.5))
    with pytest.raises(ValueError):
        Histogram("bad", "test", buckets=(float("inf"),))


def test_histogram_time_context_manager_observes_elapsed() -> None:
    h = Histogram("op_seconds", "test", buckets=(0.001, 1.0))
    with h.time():
        time.sleep(0.002)
    s, n = h.stats()
    assert n == 1
    assert s >= 0.001  # sleep can be slightly less but always > resolution
    assert s < 1.0


def test_histogram_with_labels() -> None:
    h = Histogram(
        "op_seconds", "test", buckets=(0.1, 1.0),
        label_names=("op",),
    )
    h.observe(0.05, labels={"op": "market"})
    h.observe(0.5, labels={"op": "stop"})
    text = "\n".join(h.render())
    assert 'op_seconds_bucket{le="0.1",op="market"} 1' in text
    assert 'op_seconds_bucket{le="1",op="stop"} 1' in text


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


def test_registry_namespace_prefix() -> None:
    r = MetricsRegistry(namespace="altcoin_agent")
    c = r.counter("orders_total", "test")
    assert c.name == "altcoin_agent_orders_total"
    text = r.render()
    assert "# HELP altcoin_agent_orders_total test" in text


def test_registry_rejects_duplicate() -> None:
    r = MetricsRegistry(namespace="ns")
    r.counter("x", "test")
    with pytest.raises(ValueError):
        r.counter("x", "test")


def test_registry_invalid_metric_name_raises() -> None:
    r = MetricsRegistry()
    with pytest.raises(ValueError):
        r.counter("not a valid name", "test")


def test_registry_invalid_label_name_raises() -> None:
    r = MetricsRegistry()
    with pytest.raises(ValueError):
        r.counter("foo_total", "test", label_names=("with space",))


def test_registry_render_groups_metrics_in_sorted_order() -> None:
    r = MetricsRegistry(namespace="ns")
    r.counter("z_total", "test")
    r.counter("a_total", "test")
    text = r.render()
    a_pos = text.index("ns_a_total")
    z_pos = text.index("ns_z_total")
    assert a_pos < z_pos


# --------------------------------------------------------------------- #
# DefaultMetrics preset
# --------------------------------------------------------------------- #


def test_default_registry_has_at_least_30_metrics() -> None:
    dm: DefaultMetrics = build_default_registry()
    text = dm.registry.render()
    # Each metric family contributes at least one ``# HELP`` line; that
    # makes counting them reliable across counter/gauge/histogram types.
    helps = [ln for ln in text.splitlines() if ln.startswith("# HELP ")]
    assert len(helps) >= 30, helps


def test_default_registry_metric_names_namespaced() -> None:
    dm = build_default_registry()
    # Observe one value on each histogram so the sample lines render.
    dm.order_latency_seconds.observe(0.01, labels={"op": "market"})
    dm.llm_latency_seconds.observe(0.5)
    text = dm.registry.render()
    # Sample a few that the operator monitors most heavily.
    for needle in (
        "altcoin_agent_up",
        "altcoin_agent_orders_placed_total",
        "altcoin_agent_orders_rejected_total",
        "altcoin_agent_order_latency_seconds_bucket",
        "altcoin_agent_llm_latency_seconds_bucket",
        "altcoin_agent_dlq_writes_total",
        "altcoin_agent_halt_engaged",
        "altcoin_agent_daily_drawdown_pct",
    ):
        assert needle in text, f"missing metric: {needle}"


def test_default_registry_observation_round_trip() -> None:
    dm = build_default_registry()
    dm.up.set(1.0)
    dm.orders_placed_total.inc(
        labels={"symbol": "PEPE/USDT:USDT", "side": "long"},
    )
    dm.orders_rejected_total.inc(labels={"reason": "anti_chase"})
    dm.order_latency_seconds.observe(
        0.012, labels={"op": "market"},
    )
    text = dm.registry.render()
    assert "altcoin_agent_up 1" in text
    assert (
        'altcoin_agent_orders_placed_total{side="long",'
        'symbol="PEPE/USDT:USDT"} 1'
    ) in text
    assert (
        'altcoin_agent_orders_rejected_total{reason="anti_chase"} 1'
    ) in text


# --------------------------------------------------------------------- #
# Label-value escaping
# --------------------------------------------------------------------- #


def test_label_value_escapes_quotes_and_backslash() -> None:
    c = Counter("foo_total", "test", label_names=("reason",))
    c.inc(labels={"reason": 'with " quote and \\ slash'})
    text = "\n".join(c.render())
    assert r'reason="with \" quote and \\ slash"' in text
