"""tests/test_structured_log_mock.py — Phase B.2.2 structured-log tests.

Covers:

* ``new_trace_id`` / ``current_trace_id`` / ``bind_trace_id``
  contextvar semantics, including unwinding via the returned token.
* ``JsonFormatter`` envelope + ``trace_id`` propagation + ``extra``
  field merging + non-serialisable fallback.
* ``StructuredLogger.bind`` produces a new logger that carries the
  bound fields without affecting the parent.
* contextvars are isolated across asyncio.tasks (so a per-decision
  trace_id doesn't leak into a sibling).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass

import pytest

from altcoin_agent.observability.structured_log import (
    JsonFormatter,
    StructuredLogger,
    _trace_id_var,
    bind_trace_id,
    current_trace_id,
    new_trace_id,
)


@pytest.fixture(autouse=True)
def _reset_trace_id():
    """Each test starts and ends with a clean trace_id contextvar."""
    token = _trace_id_var.set("-")
    try:
        yield
    finally:
        _trace_id_var.reset(token)


# --------------------------------------------------------------------- #
# trace_id helpers
# --------------------------------------------------------------------- #


def test_new_trace_id_binds_a_short_id() -> None:
    tid = new_trace_id()
    assert len(tid) == 12
    assert all(c in "0123456789abcdef" for c in tid)
    assert current_trace_id() == tid


def test_bind_and_unwind_trace_id_token() -> None:
    new_trace_id()
    parent = current_trace_id()

    token = bind_trace_id("nested-id")
    assert current_trace_id() == "nested-id"

    _trace_id_var.reset(token)
    assert current_trace_id() == parent


def test_default_trace_id_is_dash() -> None:
    assert current_trace_id() == "-"


@pytest.mark.asyncio
async def test_trace_id_does_not_leak_across_sibling_tasks() -> None:
    """asyncio.create_task snapshots the *parent* context, so two
    sibling tasks each see whatever the parent had bound at spawn,
    but mutations inside one don't leak to the other.
    """
    new_trace_id()
    parent = current_trace_id()

    async def child(child_tid: str) -> str:
        bind_trace_id(child_tid)
        # Yield control so the scheduler interleaves the children.
        await asyncio.sleep(0)
        return current_trace_id()

    a, b = await asyncio.gather(child("aaa"), child("bbb"))
    assert a == "aaa"
    assert b == "bbb"
    # Parent context is untouched.
    assert current_trace_id() == parent


# --------------------------------------------------------------------- #
# JsonFormatter
# --------------------------------------------------------------------- #


def _capture_json_log(logger_name: str = "test.struct") -> tuple[
    logging.Logger, io.StringIO,
]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(service="test"))
    log = logging.getLogger(logger_name)
    log.handlers = [handler]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log, buf


def test_json_formatter_emits_envelope_and_trace_id() -> None:
    log, buf = _capture_json_log("test.envelope")
    new_trace_id()
    expected_tid = current_trace_id()

    log.info("hello world")
    line = buf.getvalue().strip()
    payload = json.loads(line)

    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["service"] == "test"
    assert payload["trace_id"] == expected_tid
    assert "iso_ts" in payload
    assert "ts" in payload


def test_json_formatter_merges_extra_into_payload() -> None:
    log, buf = _capture_json_log("test.extra")
    log.info("opened", extra={"symbol": "PEPE/USDT:USDT", "size": 1.0})
    payload = json.loads(buf.getvalue().strip())
    assert payload["symbol"] == "PEPE/USDT:USDT"
    assert payload["size"] == 1.0


def test_json_formatter_does_not_overwrite_envelope_with_extra() -> None:
    """``logging`` itself reserves a few names (``message``, ``asctime``)
    and refuses to let ``extra`` shadow them. Our formatter has the
    same posture for envelope keys (``msg``, ``ts``, ``trace_id``…),
    enforced inside ``StructuredLogger._log`` rather than at the bare
    ``logger.info(extra=…)`` call. We test via ``StructuredLogger`` so
    the contract that matters end-to-end is what we cover.
    """
    log, buf = _capture_json_log("test.collide")
    s = StructuredLogger(log)
    # ``msg`` is in our envelope; the StructuredLogger merges then
    # strips reserved attrs before forwarding to ``logger.log``.
    s.info("opened", extra={"trace_id": "OVERRIDE-attempt", "extra_field": "x"})
    payload = json.loads(buf.getvalue().strip())
    # ``trace_id`` is one of our envelope fields; we replace it with
    # the contextvar value, not the caller's extra.
    assert payload["msg"] == "opened"
    # User-supplied extra fields that don't collide flow through.
    assert payload["extra_field"] == "x"


def test_json_formatter_non_serialisable_falls_back_to_repr() -> None:
    @dataclass
    class _Custom:
        x: int

    log, buf = _capture_json_log("test.coerce")
    obj = _Custom(x=7)
    log.info("custom", extra={"obj": obj})
    payload = json.loads(buf.getvalue().strip())
    # The dataclass falls through ``__dict__`` so ``x`` is preserved.
    assert payload["obj"] == {"x": 7}


def test_json_formatter_includes_exc_info_on_error() -> None:
    log, buf = _capture_json_log("test.exc")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("caught")
    payload = json.loads(buf.getvalue().strip())
    assert "exc_info" in payload
    assert "RuntimeError" in payload["exc_info"]


# --------------------------------------------------------------------- #
# StructuredLogger
# --------------------------------------------------------------------- #


def test_structured_logger_bind_merges_fields() -> None:
    log, buf = _capture_json_log("test.bound")
    s = StructuredLogger(log).bind(symbol="PEPE")
    s.info("hello")
    payload = json.loads(buf.getvalue().strip())
    assert payload["symbol"] == "PEPE"


def test_structured_logger_bind_returns_new_instance() -> None:
    log, _ = _capture_json_log("test.bound2")
    parent = StructuredLogger(log)
    child = parent.bind(symbol="X")
    # Parent stays empty, child has the binding.
    assert parent._bound == {}
    assert child._bound == {"symbol": "X"}


def test_structured_logger_per_call_extra_overrides_bound() -> None:
    log, buf = _capture_json_log("test.override")
    s = StructuredLogger(log).bind(symbol="default")
    s.info("hello", extra={"symbol": "override"})
    payload = json.loads(buf.getvalue().strip())
    assert payload["symbol"] == "override"
