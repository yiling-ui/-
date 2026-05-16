"""structured_log.py — JSON structured logging + trace_id propagation.

Phase B.2.2 of ``MISS_PENALTY_AND_PRODUCTION_PLAN.md``: text logs make
it impossible to correlate the ~10 hops a single screener event takes
through the pipeline (screener -> fuser -> LLM -> gate -> executor ->
trailing). We need every line emitted from any of those stages to be
JSON-serialisable AND tagged with the same ``trace_id`` so an operator
can ``grep '"trace_id": "abc123"'`` to retrace one decision.

Design choices
--------------
* **Pure stdlib.** No structlog dep. We attach a custom
  :class:`logging.Formatter` to the root handler that emits one
  JSON object per line.
* **contextvars-based.** ``trace_id`` lives in a ``ContextVar`` so it
  flows naturally across ``await`` boundaries inside one asyncio task
  without leaking across tasks (each ``asyncio.create_task`` snapshots
  the current context). Tests can call ``new_trace_id()`` /
  ``bind_trace_id("manual-id")`` to drive the field deterministically.
* **Lossy on serialisation failure.** If a record carries something
  not JSON-serialisable we fall back to ``repr()`` rather than raise —
  a logger MUST NOT fail the trading loop.
* **Backwards-compatible.** ``configure_structured_logging`` is opt-in
  from ``main.App.run`` based on a config flag; existing tests that
  read plain-text log lines from caplog still see them via the
  in-memory handler that pytest installs (caplog uses its own
  formatter, not ours).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from typing import Any

# ContextVar used by every component to find the current trace id. The
# default ("-") is what shows up in logs emitted from outside any
# trace-bound context (e.g. startup boilerplate, the screener's
# top-level run loop). Tests that want to assert "this came from inside
# a high-priority decision" check for a non-"-" trace id.
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "altcoin_agent_trace_id", default="-",
)

# Reserved attributes that ``logging.LogRecord`` always exposes; we
# strip these from ``record.__dict__`` so the JSON payload only carries
# the user's bound fields plus our standard envelope.
_STD_RECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message",
    "module", "msecs", "msg", "name", "pathname", "process",
    "processName", "relativeCreated", "stack_info", "thread",
    "threadName", "taskName",
})


def new_trace_id() -> str:
    """Generate and bind a fresh short trace id (12 hex chars).

    Short ids keep log lines compact while still being collision-free
    over the lifetime of one trading day (~10^7 events vs. 16^12 space).
    """
    tid = uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def bind_trace_id(trace_id: str) -> contextvars.Token:
    """Bind ``trace_id`` for the current context. Returns a token that
    can be passed to ``_trace_id_var.reset(token)`` for nested unwinding.
    Tests prefer ``new_trace_id`` / direct mutation; production code
    inside ``_handle_high_priority`` calls this with the audit-log id.
    """
    return _trace_id_var.set(str(trace_id))


def current_trace_id() -> str:
    """Return the trace id currently bound, or ``"-"`` when unbound."""
    return _trace_id_var.get()


# --------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------- #


class JsonFormatter(logging.Formatter):
    """Render every record as a single-line JSON object.

    The base envelope is::

        {"ts": <epoch_seconds>, "level": "INFO", "logger": "name",
         "msg": "...", "trace_id": "...", "module": "...",
         "line": 123}

    Any extra fields the caller bound via ``logger.info(..., extra={...})``
    are shallow-merged on top of the envelope. Non-serialisable values
    fall back to ``repr()``.
    """

    def __init__(self, *, service: str = "altcoin-agent") -> None:
        super().__init__()
        self.service = service
        self.hostname = os.uname().nodename if hasattr(os, "uname") else ""

    def format(self, record: logging.LogRecord) -> str:
        try:
            payload: dict[str, Any] = {
                "ts": record.created,
                "iso_ts": _iso(record.created),
                "level": record.levelname,
                "logger": record.name,
                "service": self.service,
                "module": record.module,
                "line": record.lineno,
                "msg": record.getMessage(),
                "trace_id": current_trace_id(),
            }
            if self.hostname:
                payload["host"] = self.hostname
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            # Caller-bound ``extra`` lives directly on record.__dict__.
            for k, v in record.__dict__.items():
                if k in _STD_RECORD_ATTRS or k.startswith("_"):
                    continue
                if k in payload:
                    # Don't overwrite our envelope fields.
                    continue
                payload[k] = v
            return json.dumps(payload, default=_json_default, sort_keys=False)
        except Exception as e:  # pragma: no cover - defensive
            # Logger must never crash the trading loop. Fall back to a
            # minimal envelope so the operator still sees *something*.
            return json.dumps({
                "ts": time.time(),
                "level": "ERROR",
                "logger": "structured_log",
                "msg": f"format_error: {e!r}",
                "raw": repr(record.msg),
            })


def _iso(epoch: float) -> str:
    """Render an epoch second as an ISO-8601 UTC timestamp."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    """Last-ditch coercion for non-JSON-serialisable values.

    We try ``__dict__`` (covers most dataclasses), then a few well-known
    types, then ``repr``. Never raises.
    """
    try:
        if hasattr(obj, "as_dict") and callable(obj.as_dict):
            return obj.as_dict()
        if hasattr(obj, "__dict__") and obj.__dict__:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    except Exception:
        pass
    return repr(obj)


# --------------------------------------------------------------------- #
# StructuredLogger adapter
# --------------------------------------------------------------------- #


class StructuredLogger:
    """Thin wrapper that lets call sites attach structured fields.

    We deliberately do NOT subclass ``LoggerAdapter`` because the adapter
    forces every record through one extra-dict; here we want each call
    to merge its own ``extra`` cleanly so the trace_id and per-call
    fields don't fight for the same dict.
    """

    def __init__(
        self,
        logger: logging.Logger,
        bound: dict[str, Any] | None = None,
    ) -> None:
        self._logger = logger
        self._bound = dict(bound) if bound else {}

    def bind(self, **fields: Any) -> StructuredLogger:
        merged = {**self._bound, **fields}
        return StructuredLogger(self._logger, bound=merged)

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", None) or {}
        # Preserve the caller's ``extra`` semantics — bound fields lose
        # to per-call ``extra`` so a caller can override e.g. ``symbol``
        # for a single line.
        merged = {**self._bound, **extra}
        # ``logging`` reserves a few keys (e.g. ``message``) — drop any
        # collisions defensively.
        for reserved in _STD_RECORD_ATTRS:
            merged.pop(reserved, None)
        self._logger.log(level, msg, *args, extra=merged, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)


# --------------------------------------------------------------------- #
# Configuration entry point
# --------------------------------------------------------------------- #


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    service: str = "altcoin-agent",
    replace_handlers: bool = True,
) -> JsonFormatter:
    """Install :class:`JsonFormatter` on the root logger.

    Idempotent: a second call with the same parameters replaces the
    formatter on the existing handler rather than adding new ones.
    Tests that need plain-text caplog output skip this.
    """
    root = logging.getLogger()
    fmt = JsonFormatter(service=service)
    if replace_handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)
    root.setLevel(level)
    return fmt
