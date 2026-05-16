"""miss_penalty_engine.py — opportunity-cost penalty engine (Phase A.1).

Background
----------
Today the agent only learns from positions it actually opened
(``learning_engine.run_post_mortem`` is invoked 1h after entry). It has
no idea whether a *rejected* signal would have made money. If
``RiskGate`` rejects a +200% pump for ``anti_chase`` the daemon stays
quiet, the rule keeps tripping forever, and the operator never sees
"too conservative" feedback.

This engine reads the existing ``logs/decisions.jsonl`` audit log and,
24h after each rejection, asks "did the symbol moon anyway?". When it
did, the rejection is recorded as a *missed opportunity*. Downstream
modules (``RejectReasonScorer`` in A.2 and ``ThresholdAutoTuner`` in
A.3) consume the missed-opportunity stream to decide whether the
rejection rule is too tight.

Why "opportunity cost is 3x the realized-PnL cost"
--------------------------------------------------
Per the operator's calibration in
``MISS_PENALTY_AND_PRODUCTION_PLAN.md§A.2.3``: missing one +200% pump
is three times worse than catching one normal trade. The 3x multiplier
is applied by ``RejectReasonScorer``, not here — this module just
labels candidate misses.

Tests use injected K-line fetchers so we never call ccxt at unit-test
time. Production wires up an ``async`` ccxt-backed fetcher via the
:func:`make_ccxt_kline_fetcher` factory.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------- #


# (ts_ms, open, high, low, close, volume) — matches ccxt fetch_ohlcv shape.
Kline = tuple[int, float, float, float, float, float]


# Async fetcher: (symbol, since_ms, until_ms) -> list of klines (1m).
# We keep it async because production goes through ccxt.async_support;
# tests inject a sync function wrapped in ``async def`` — see
# :func:`fixed_klines_fetcher` for the pattern.
KlineFetcher = Callable[[str, int, int], Awaitable[list[Kline]]]


@dataclass
class MissedOpportunity:
    """One audited rejection + its 24h post-mortem result.

    Fields match the spec table in
    ``MISS_PENALTY_AND_PRODUCTION_PLAN.md§A.2.1``. ``is_missed_pump``
    is the binary label used by downstream scorers; ``miss_severity``
    is a 0..1 sliding scale used for weight tuning when (eventually)
    we want to penalise huge misses (e.g. +500%) more than borderline
    misses (+100%).
    """

    trace_id: str
    symbol: str
    rejected_at_ts_ms: int
    rejected_reason: str
    rejected_reason_bucket: str
    rejected_score: float
    direction: str  # "long" / "short"
    entry_price_if_taken: float

    # Post-mortem (24h forward window).
    realized_max_favorable_pct: float = 0.0   # signed % toward thesis
    realized_max_adverse_pct: float = 0.0     # signed % away from thesis
    would_have_pnl_pct: float = 0.0           # synthetic PnL if 1.5% risk
    bars_observed: int = 0

    # Labels.
    is_missed_pump: bool = False
    miss_severity: float = 0.0                # 0..1
    insufficient_data: bool = False           # True when bars_observed < min

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MissedOpportunity:
        # Keep forward-compat: drop unknown keys, default missing ones.
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)


@dataclass
class MissPenaltyConfig:
    """Tuning knobs.

    All thresholds are calibrated for the typical "altcoin spike"
    profile — pumps run 100-500% in 24h, dumps 30-70%. They're
    deliberately sharp so we don't classify normal +30% chop as
    "missed pumps".
    """

    # Forward window after a rejection.
    forward_window_sec: int = 24 * 3600

    # LONG miss thresholds.
    long_min_mfe_pct: float = 1.00       # MFE >= +100% toward upside
    long_max_mae_pct: float = 0.30       # MAE drawdown <= 30% (didn't blow up first)

    # SHORT miss thresholds (drawdown is the move TOWARD short thesis).
    short_min_mfe_pct: float = 0.50      # |MFE| >= 50% downward
    short_max_mae_pct: float = 0.20      # MAE upward bounce <= 20%

    # Synthetic PnL params (used to score how much we left on the table).
    assumed_risk_pct: float = 0.015      # 1.5% of equity
    assumed_leverage: float = 10.0       # representative for altcoin perp

    # Quality floors.
    min_bars_for_label: int = 60         # 1h of 1m bars; below this skip
    min_kline_age_sec: int = 60 * 60     # only audit rejects older than 1h
                                          # (don't peek at unfinished moves)

    # When the audit window must be fully closed before labelling.
    require_window_closed: bool = True

    # Output paths (relative to workspace root).
    state_dir: str = ".kiro/state/miss_penalty"
    output_filename: str = "missed_opportunities.jsonl"

    # Reject-reason bucketing.
    # The audit log writes things like
    #   "slippage_too_high:0.0341>0.0212@lev=10.00"
    # which would explode into thousands of unique buckets. We strip
    # the ``:trailing-data`` so the scorer accumulates samples per
    # rule, not per (rule, exact-price-tuple).
    reason_bucket_separator: str = ":"


# --------------------------------------------------------------------- #
# Audit-log reader
# --------------------------------------------------------------------- #


def iter_decisions_jsonl(
    path: Path | str, *, since_ts: float | None = None,
) -> Iterator[dict]:
    """Yield each parsed JSON record from a decisions.jsonl file.

    Robust to: empty file, trailing newline, partial last line (writer
    crash mid-write), garbage bytes (logged + skipped). Optionally
    filters by ``ts >= since_ts``.

    We deliberately re-open the file each call rather than keep an
    fd open: the audit log self-rotates (see
    ``audit_log.DecisionAuditLog``) and a long-held fd would point at
    an unlinked inode after rotation.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("iter_decisions_jsonl: read failed: %s", e)
        return
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            # Treat partial last line as recoverable.
            logger.debug(
                "iter_decisions_jsonl: skip malformed line %d: %s", i, e,
            )
            continue
        if not isinstance(rec, dict):
            continue
        if since_ts is not None:
            ts = rec.get("ts")
            if isinstance(ts, (int, float)) and ts < since_ts:
                continue
        yield rec


def iter_decisions_with_rotations(
    base_path: Path | str, *, since_ts: float | None = None,
) -> Iterator[dict]:
    """Like :func:`iter_decisions_jsonl` but also reads the rotated files
    ``<base_path>.1``..``.N``.

    Order is **oldest first**: rotation N -> rotation 1 -> active. When
    the audit window crosses a rotation we'd otherwise lose half the
    rejections.
    """
    base = Path(base_path)
    parent = base.parent if base.parent != Path("") else Path(".")
    name = base.name
    # Discover backups by listing — don't rely on a fixed N.
    if parent.exists():
        rotations: list[tuple[int, Path]] = []
        for child in parent.iterdir():
            if child.name.startswith(name + "."):
                suffix = child.name[len(name) + 1 :]
                try:
                    n = int(suffix)
                    rotations.append((n, child))
                except ValueError:
                    continue
        # Largest N == oldest, so descend.
        rotations.sort(key=lambda t: t[0], reverse=True)
        for _, p in rotations:
            yield from iter_decisions_jsonl(p, since_ts=since_ts)
    yield from iter_decisions_jsonl(base, since_ts=since_ts)


# --------------------------------------------------------------------- #
# MFE / MAE
# --------------------------------------------------------------------- #


def compute_mfe_mae(
    bars: Iterable[Kline], reference_price: float, direction: str,
) -> tuple[float, float, int]:
    """Walk the bar window once, return (MFE, MAE, count).

    Both numbers are signed *relative to the trader's thesis*:

    * For LONG (``direction == "long"``):
        MFE = max((high - ref) / ref) over the window  -> >= 0
        MAE = min((low  - ref) / ref) over the window  -> <= 0
    * For SHORT (``direction == "short"``):
        MFE = max((ref - low ) / ref) over the window  -> >= 0
        MAE = min((ref - high) / ref) over the window  -> <= 0

    This lets the caller treat MFE the same way regardless of
    direction: "how far did it run in our favour", "how far against".

    Returns (0.0, 0.0, 0) on empty bars or non-positive reference.
    """
    bars = list(bars)
    if not bars or reference_price <= 0:
        return 0.0, 0.0, 0

    direction = direction.lower()
    if direction not in ("long", "short"):
        # Treat unknown as long for safety; caller validates upstream.
        direction = "long"

    mfe = 0.0
    mae = 0.0
    for _ts, _o, high, low, _c, _v in bars:
        if direction == "long":
            up = (high - reference_price) / reference_price
            dn = (low - reference_price) / reference_price
            if up > mfe:
                mfe = up
            if dn < mae:
                mae = dn
        else:  # short
            # "favorable" = price went DOWN.
            up = (reference_price - low) / reference_price
            dn = (reference_price - high) / reference_price
            if up > mfe:
                mfe = up
            if dn < mae:
                mae = dn
    return mfe, mae, len(bars)


# --------------------------------------------------------------------- #
# Reason bucketing
# --------------------------------------------------------------------- #


def bucket_reject_reason(raw: str, *, separator: str = ":") -> str:
    """Strip the trailing ``:value`` from a RiskGate reason.

    Examples
    --------
    >>> bucket_reject_reason("slippage_too_high:0.0341>0.0212@lev=10.00")
    'slippage_too_high'
    >>> bucket_reject_reason("anti_chase:0.034")
    'anti_chase'
    >>> bucket_reject_reason("ok")
    'ok'

    Audit reasons sometimes contain *substring* details after the
    separator (live trace data, leverage). We keep only the rule-name
    prefix because that's the unit the scorer accumulates over.
    """
    if not raw:
        return "unknown"
    # Some reasons embed multiple separators (``signal_blocked:foo:bar``).
    # We only strip the FIRST occurrence so ``signal_blocked:foo`` and
    # ``signal_blocked:bar`` collapse into ``signal_blocked`` together.
    idx = raw.find(separator)
    if idx <= 0:
        return raw
    return raw[:idx]


# --------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------- #


class MissPenaltyEngine:
    """Read the audit log, audit each rejection 24h later, persist labels."""

    def __init__(
        self,
        *,
        decisions_log_path: Path | str,
        kline_fetcher: KlineFetcher,
        config: MissPenaltyConfig | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.decisions_log_path = Path(decisions_log_path)
        self.kline_fetcher = kline_fetcher
        self.cfg = config or MissPenaltyConfig()
        self._clock = clock
        # Track which trace ids we've already labelled so re-runs
        # within the same day are idempotent. The set is hydrated from
        # disk on first call to :meth:`run_audit`.
        self._labelled_trace_ids: set[str] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run_audit(
        self,
        *,
        lookback_hours: int = 48,
    ) -> list[MissedOpportunity]:
        """Audit all rejections that are old enough to be labelled.

        ``lookback_hours`` controls how far back into the audit log we
        scan. Default 48h covers the 24h forward window + 24h slack so
        the same rejection isn't audited twice across day boundaries
        without us tracking it in the dedup set.

        Returns the new ``MissedOpportunity`` records created during
        this run (already persisted).
        """
        now = self._clock()
        oldest_audit_ts = now - lookback_hours * 3600
        forward = self.cfg.forward_window_sec
        min_age = max(self.cfg.min_kline_age_sec, forward)

        labelled = self._load_labelled_trace_ids()
        new_records: list[MissedOpportunity] = []

        for rec in iter_decisions_with_rotations(
            self.decisions_log_path, since_ts=oldest_audit_ts,
        ):
            if rec.get("approved", False):
                # Approved trades are post-mortem'd by ``learning_engine``
                # 1h after entry. They are not "missed opportunities".
                continue
            ts = rec.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            # Window-not-closed yet?
            if self.cfg.require_window_closed and (now - ts) < min_age:
                continue

            trace_id = str(rec.get("trace_id") or "")
            if not trace_id:
                # Without a trace id we can't dedup — skip rather than
                # double-count.
                continue
            if trace_id in labelled:
                continue

            symbol = str(rec.get("symbol") or "").strip()
            direction = str(rec.get("direction") or "").lower()
            if not symbol or direction not in ("long", "short"):
                continue
            entry_price = _coerce_float(rec.get("current_price"))
            if entry_price is None or entry_price <= 0:
                continue

            try:
                opp = await self._audit_one(rec, ts=float(ts))
            except Exception as e:  # never block the audit loop
                logger.warning(
                    "MissPenaltyEngine: audit failed for %s: %s",
                    trace_id, e,
                )
                continue

            self._persist(opp)
            labelled.add(trace_id)
            new_records.append(opp)

        return new_records

    def load_recent_missed(
        self, *, since_ts: float | None = None,
    ) -> list[MissedOpportunity]:
        """Load already-persisted missed opportunities (for downstream
        consumers like :class:`RejectReasonScorer` and the reflection
        mode trigger)."""
        out: list[MissedOpportunity] = []
        out_path = self._output_path()
        if not out_path.exists():
            return out
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            if since_ts is not None:
                ts = d.get("rejected_at_ts_ms")
                if isinstance(ts, (int, float)) and (ts / 1000.0) < since_ts:
                    continue
            try:
                out.append(MissedOpportunity.from_dict(d))
            except TypeError:
                # Forward-compat: drop rows that don't deserialize.
                continue
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _audit_one(self, rec: dict, *, ts: float) -> MissedOpportunity:
        symbol = str(rec["symbol"])
        direction = str(rec["direction"]).lower()
        entry_price = float(rec["current_price"])
        rejected_reason = str(rec.get("reason") or "unknown")
        bucket = bucket_reject_reason(
            rejected_reason, separator=self.cfg.reason_bucket_separator,
        )

        rejected_at_ts_ms = int(ts * 1000)
        until_ms = rejected_at_ts_ms + self.cfg.forward_window_sec * 1000

        bars = await self.kline_fetcher(symbol, rejected_at_ts_ms, until_ms)

        mfe, mae, n_bars = compute_mfe_mae(
            bars, reference_price=entry_price, direction=direction,
        )

        opp = MissedOpportunity(
            trace_id=str(rec.get("trace_id") or ""),
            symbol=symbol,
            rejected_at_ts_ms=rejected_at_ts_ms,
            rejected_reason=rejected_reason,
            rejected_reason_bucket=bucket,
            rejected_score=float(rec.get("final_score") or 0.0),
            direction=direction,
            entry_price_if_taken=entry_price,
            realized_max_favorable_pct=mfe,
            realized_max_adverse_pct=mae,
            bars_observed=n_bars,
        )

        if n_bars < self.cfg.min_bars_for_label:
            opp.insufficient_data = True
            return opp

        opp.is_missed_pump = self._is_missed(opp)
        opp.miss_severity = self._severity(opp)
        opp.would_have_pnl_pct = self._synthetic_pnl(opp)
        return opp

    def _is_missed(self, opp: MissedOpportunity) -> bool:
        cfg = self.cfg
        if opp.direction == "long":
            return (
                opp.realized_max_favorable_pct >= cfg.long_min_mfe_pct
                and abs(opp.realized_max_adverse_pct) <= cfg.long_max_mae_pct
            )
        return (
            opp.realized_max_favorable_pct >= cfg.short_min_mfe_pct
            and abs(opp.realized_max_adverse_pct) <= cfg.short_max_mae_pct
        )

    def _severity(self, opp: MissedOpportunity) -> float:
        """0..1 sliding scale.

        Anchor: 100% MFE on a long => 0.5; 300% MFE => 1.0. Capped.
        For shorts the anchor is 50% / 150%. The ratios match the
        ``long_min_mfe_pct`` / ``short_min_mfe_pct`` thresholds.
        """
        if not opp.is_missed_pump:
            return 0.0
        anchor = (
            self.cfg.long_min_mfe_pct
            if opp.direction == "long"
            else self.cfg.short_min_mfe_pct
        )
        # Map [anchor, 3*anchor] -> [0.5, 1.0]; below anchor -> 0.0
        # (won't trigger because is_missed_pump already False);
        # above 3*anchor -> 1.0.
        x = opp.realized_max_favorable_pct
        if x <= anchor:
            return 0.5
        ratio = (x - anchor) / (2 * anchor)
        return float(min(1.0, 0.5 + 0.5 * ratio))

    def _synthetic_pnl(self, opp: MissedOpportunity) -> float:
        """Estimate the PnL we'd have made on a 1.5% risk position.

        Simplified model: stop placed at MAE distance, take at 50% of
        MFE (to reflect realistic exit slippage and trailing). PnL =
        (taken_move - fees) * leverage * risk_per_lev_unit. We don't
        try to be precise — the consumer (``RejectReasonScorer``) only
        uses sign + ordering, not the absolute value.
        """
        # Fees ~= 0.04% taker * 2 = 0.08%; slippage ~= 0.05%.
        cost = 0.0013
        taken = max(0.0, 0.5 * opp.realized_max_favorable_pct - cost)
        # PnL on equity = leverage * (taken - cost) * risk_pct (approx).
        return float(
            taken * self.cfg.assumed_leverage * self.cfg.assumed_risk_pct
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _output_path(self) -> Path:
        return Path(self.cfg.state_dir) / self.cfg.output_filename

    def _persist(self, opp: MissedOpportunity) -> None:
        out_path = self._output_path()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(opp.to_dict(), default=str) + "\n")
        except OSError as e:
            logger.warning(
                "MissPenaltyEngine: persist failed (swallowed): %s", e,
            )

    def _load_labelled_trace_ids(self) -> set[str]:
        if self._labelled_trace_ids is not None:
            return self._labelled_trace_ids
        out: set[str] = set()
        path = self._output_path()
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid = rec.get("trace_id")
                    if isinstance(tid, str) and tid:
                        out.add(tid)
            except OSError as e:
                logger.warning(
                    "MissPenaltyEngine: dedup-load failed: %s", e,
                )
        self._labelled_trace_ids = out
        return out


# --------------------------------------------------------------------- #
# ccxt-backed kline fetcher (production)
# --------------------------------------------------------------------- #


def make_ccxt_kline_fetcher(exchange) -> KlineFetcher:  # type: ignore[no-untyped-def]
    """Return a :type:`KlineFetcher` backed by a ccxt async exchange.

    ``exchange`` should be a ``ccxt.async_support`` exchange instance
    that the daemon already manages (we do NOT take ownership of the
    lifecycle here; the caller is responsible for ``close()``).

    Production wiring lives in ``main.py``: we reuse the same
    exchange handle that the executor uses, so authentication, proxy
    config, and rate-limit budget are shared.
    """

    async def _fetch(symbol: str, since_ms: int, until_ms: int) -> list[Kline]:
        # ccxt fetch_ohlcv takes ``since`` in ms, returns oldest-first.
        # 1m timeframe x 24h = 1440 bars, well under the 1500 limit on
        # most venues — fetch in one call.
        try:
            limit = max(
                100,
                min(1500, int((until_ms - since_ms) / 60_000) + 5),
            )
            raw = await exchange.fetch_ohlcv(
                symbol, timeframe="1m", since=since_ms, limit=limit,
            )
        except Exception as e:
            logger.warning(
                "ccxt fetch_ohlcv(%s) failed: %s", symbol, e,
            )
            return []
        out: list[Kline] = []
        for row in raw:
            if not row or len(row) < 6:
                continue
            ts = int(row[0])
            if ts < since_ms or ts > until_ms:
                continue
            out.append(
                (ts, float(row[1]), float(row[2]), float(row[3]),
                 float(row[4]), float(row[5])),
            )
        return out

    return _fetch


def fixed_klines_fetcher(
    bars_by_symbol: dict[str, list[Kline]],
) -> KlineFetcher:
    """Return a :type:`KlineFetcher` that yields canned data — for tests."""

    async def _fetch(symbol: str, since_ms: int, until_ms: int) -> list[Kline]:
        return [
            b for b in bars_by_symbol.get(symbol, [])
            if since_ms <= b[0] <= until_ms
        ]

    return _fetch


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _coerce_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f
