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
        """
        if not self.cfg.enabled:
            return
        if symbol != self.cfg.reference_symbol:
            return
        if close <= 0 or ts_ms <= 0:
            return
        self._samples.append((int(ts_ms), float(close)))

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
