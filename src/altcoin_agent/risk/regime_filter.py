"""regime_filter.py — BTC/ETH market-regime gate (audit #10).

Background
----------
Altcoin pump strategies have asymmetric expected value across BTC
regimes. When BTC is in a fast drawdown the entire alt complex
correlates to ~1.0 and any LONG signal — even one with a perfect rule
score — is fighting market beta. The audit specifically called this
out as the second-most-likely cause of a -30% week-one drawdown.

Behaviour
---------
``RegimeFilter`` ingests BTC kline closes from the same screener stream
that already drives the price tape, and exposes:

  * ``roc_pct(window_ms)`` — rate-of-change over the requested window.
  * ``allow_direction(direction)`` — returns (allowed, reason).

When BTC has fallen by more than ``btc_drop_block_long_pct`` over
``btc_window_ms``:
  * LONG signals are blocked.
  * SHORT signals are unaffected (alts crashing along with BTC is
    exactly when shorts should run).

When BTC has risen by more than ``btc_rip_block_short_pct``:
  * SHORT signals are blocked (don't fight a strong bid).
  * LONG signals are unaffected.

Cold start (< ``min_samples`` ticks) is fail-OPEN: a freshly booted
daemon should not refuse every signal just because it hasn't yet seen
a full window of BTC bars. This mirrors the PriceTape design.

Config defaults are intentionally conservative (-3% in 1h to block
LONG; +5% in 1h to block SHORT). They are operator-tunable in
``app.yaml``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegimeFilterConfig:
    enabled: bool = True
    # Reference symbol — the stream the screener already pulls (default
    # config has BTC/USDT:USDT). Operators that don't trade BTC perp can
    # point this at any symbol whose closes are reliably observed.
    reference_symbol: str = "BTC/USDT:USDT"
    # Windows + thresholds.
    btc_window_ms: int = 60 * 60 * 1000             # 1h
    btc_drop_block_long_pct: float = 0.03           # 3% drop -> block LONG
    btc_rip_block_short_pct: float = 0.05           # 5% rip -> block SHORT
    # Min samples in window before the gate engages. Below this we are
    # cold and fail open (don't accidentally veto everything on boot).
    min_samples: int = 10
    # Hard cap on memory; ~1 sample/sec for an hour = 3600. We keep
    # twice that as headroom for noisy intra-bar updates.
    max_samples: int = 8_000
    # Audit P2 #14: the smallest kline cadence the operator's screener
    # is configured to push. If ``btc_window_ms`` is too short to fit
    # ``min_samples`` *bars at this cadence*, the gate would be stuck
    # in cold-tape forever. We default to 60s to match the 1m kline
    # the rest of the daemon uses; operators that wire 5s/10s mark
    # streams should override this so the validation reflects reality.
    expected_sample_interval_ms: int = 60_000

    def __post_init__(self) -> None:
        """Validate the config so a misconfigured value doesn't silently
        render the regime gate inert (or, just as bad, always-on for
        a degenerate tape).

        We raise ``ValueError`` for hard-impossible values (zero/negative
        windows or sample counts) and for the specific case the audit
        flagged: a window so short that it can't hold ``min_samples``
        bars at the expected cadence. A warning-level log is emitted
        for soft-suspicious values (negative thresholds) so the
        operator notices on boot without the daemon crashing if they
        deliberately set those to zero to disable a side.
        """
        if self.btc_window_ms <= 0:
            raise ValueError(
                f"RegimeFilterConfig.btc_window_ms must be > 0; "
                f"got {self.btc_window_ms}"
            )
        if self.min_samples <= 0:
            raise ValueError(
                f"RegimeFilterConfig.min_samples must be > 0; "
                f"got {self.min_samples}"
            )
        if self.max_samples < self.min_samples:
            raise ValueError(
                f"RegimeFilterConfig.max_samples ({self.max_samples}) must be "
                f">= min_samples ({self.min_samples})"
            )
        if self.expected_sample_interval_ms <= 0:
            raise ValueError(
                "RegimeFilterConfig.expected_sample_interval_ms must be > 0; "
                f"got {self.expected_sample_interval_ms}"
            )
        # Audit P2 #14: the killer combo — a 5-minute window that
        # cannot possibly hold the default 10 samples of 1-minute
        # bars. Without this check the filter looks "on" in the
        # config dump but ``allow_direction`` is permanently
        # cold-tape => forever fail-open.
        #
        # Capacity = floor(window / interval) + 1 because a stream
        # at exactly ``interval`` cadence can fit samples at
        # t=0, interval, 2*interval, ..., k*interval for k =
        # floor(window/interval), giving k+1 samples within
        # ``[0, window]``. Using just ``floor`` would over-reject
        # boundary configs (e.g. ``window=60s, interval=60s,
        # min_samples=2`` is fine: samples at t=0 and t=60s are 2
        # samples in a 60s window).
        capacity = (self.btc_window_ms // self.expected_sample_interval_ms) + 1
        if capacity < self.min_samples:
            raise ValueError(
                "RegimeFilterConfig: btc_window_ms="
                f"{self.btc_window_ms}ms can hold at most {capacity} "
                f"samples at expected_sample_interval_ms="
                f"{self.expected_sample_interval_ms}ms, but "
                f"min_samples={self.min_samples}. The gate would be "
                "permanently cold-tape and never engage. Either widen "
                "the window, lower min_samples, or override "
                "expected_sample_interval_ms to match a faster stream."
            )
        # Soft checks — log only, don't crash, since "0 = disable
        # this side" is a reasonable operator intent.
        if self.btc_drop_block_long_pct < 0 or self.btc_rip_block_short_pct < 0:
            logger.warning(
                "RegimeFilterConfig has negative threshold(s): "
                "drop_block_long=%.4f rip_block_short=%.4f — these are "
                "interpreted as fractions, not percentages, and "
                "negative values invert the gate. Double-check your "
                "config.",
                self.btc_drop_block_long_pct,
                self.btc_rip_block_short_pct,
            )


@dataclass
class RegimeFilter:
    cfg: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)
    _samples: deque[tuple[int, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.cfg.max_samples)

    # --------------------- ingestion --------------------- #

    def observe(self, symbol: str, close: float, ts_ms: int) -> None:
        """Record a (ts, close) sample for the reference symbol.

        Non-reference symbols are silently ignored so the same stream
        wrapper that already feeds the price tape can fan-out to us
        without extra filtering at the call site.

        Audit (third pass) #11: ccxt.pro WS reconnects can replay the
        last few bars, producing timestamps that go backwards inside
        the deque. ``roc_pct`` then computes ``(latest - oldest)/oldest``
        with ``oldest`` not actually being the oldest, returning a
        garbage rate. We drop any sample whose ts is not strictly newer
        than the most recent one. Equality is also rejected so the
        deque stays strictly monotone — that lets ``roc_pct``'s
        head-trim loop be unambiguous.
        """
        if not self.cfg.enabled:
            return
        if symbol != self.cfg.reference_symbol:
            return
        if close <= 0 or ts_ms <= 0:
            return
        ts = int(ts_ms)
        if self._samples and ts <= self._samples[-1][0]:
            return  # late or duplicate sample; ignore
        self._samples.append((ts, float(close)))

    # --------------------- queries --------------------- #

    def roc_pct(self, now_ms: int) -> float | None:
        """Returns BTC rate-of-change over ``cfg.btc_window_ms``.

        ``None`` means "cold tape — don't make a regime call from this".
        Otherwise positive == BTC up, negative == BTC down.
        """
        if not self._samples:
            return None
        cutoff = now_ms - self.cfg.btc_window_ms
        # Drop samples older than the window from the head.
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        if len(self._samples) < self.cfg.min_samples:
            return None
        oldest = self._samples[0][1]
        latest = self._samples[-1][1]
        if oldest <= 0:
            return None
        return (latest - oldest) / oldest

    def allow_direction(
        self, direction: str, now_ms: int,
    ) -> tuple[bool, str]:
        """Decide whether the trade direction is compatible with the regime.

        ``direction`` is the FusedSignal direction string ("long" / "short" /
        "neutral"). Returns ``(True, "")`` to allow, ``(False, reason)`` to
        reject.

        Defensive defaults:
          * filter disabled  -> allow.
          * cold tape        -> allow (fail-open at boot).
          * non-directional  -> allow (the gate will reject on its own).
        """
        if not self.cfg.enabled:
            return True, ""
        if direction not in ("long", "short"):
            return True, ""
        roc = self.roc_pct(now_ms)
        if roc is None:
            return True, ""
        if direction == "long" and roc <= -self.cfg.btc_drop_block_long_pct:
            return False, (
                f"btc_regime_block_long:roc={roc:+.4f}<="
                f"{-self.cfg.btc_drop_block_long_pct:+.4f}"
            )
        if direction == "short" and roc >= self.cfg.btc_rip_block_short_pct:
            return False, (
                f"btc_regime_block_short:roc={roc:+.4f}>="
                f"{self.cfg.btc_rip_block_short_pct:+.4f}"
            )
        return True, ""

    # --------------------- maintenance --------------------- #

    def reset(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)
