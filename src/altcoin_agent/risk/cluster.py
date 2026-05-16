"""cluster.py — symbol-cluster cap support (audit #11).

Background
----------
``max_concurrent_positions=3`` doesn't prevent the daemon from being
SHORT ``PEPE / WIF / FLOKI`` simultaneously — three meme coins with a
30-day correlation north of 0.85 ARE THE SAME TRADE. The audit
flagged this as a "3x risk in one bet" footgun.

Design
------
We keep the design simple: each symbol is mapped to a cluster name
(``meme`` / ``ai`` / ``layer1`` / ``defi`` / ``other`` ...). The
operator sets ``max_per_cluster`` (default 1) in ``app.yaml``; the
RiskGate's concurrency check additionally rejects when the proposed
trade would push that cluster over the cap.

The mapping itself is config-driven (``cluster_map`` in ``app.yaml``)
because the universe of altcoins changes weekly. A symbol with no
mapping is treated as cluster ``other`` and capped together with the
rest of the unclassified set. This keeps the rule simple AND ensures
that brand-new symbols (which the operator hasn't yet had time to
classify) inherit a defensive default.

Examples
--------
>>> cm = ClusterMap({"PEPE": "meme", "FLOKI": "meme", "WIF": "meme"})
>>> cm.cluster_of("PEPE/USDT:USDT")
'meme'
>>> cm.cluster_of("BTC/USDT:USDT")
'other'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BASE_RE = re.compile(r"^([A-Z0-9]+)")


def _base_token(symbol: str) -> str:
    """Extract a base token suitable for cluster lookup.

    Handles both ``PEPEUSDT``, ``PEPE/USDT:USDT``, and ``1000PEPE``
    style names. Returns uppercase.
    """
    if not symbol:
        return ""
    head = symbol.split("/")[0].split(":")[0].upper()
    # Strip USDT/USD/USDC suffix when the venue concatenates rather
    # than slashes (e.g. "PEPEUSDT").
    for suffix in ("USDT", "USDC", "USD", "BUSD", "PERP"):
        if head.endswith(suffix) and len(head) > len(suffix):
            head = head[: -len(suffix)]
            break
    # Drop leading "1000" multiplier prefixes used by Binance for
    # micro-cap memes (e.g. "1000PEPE").
    if head.startswith("1000") and len(head) > 4:
        head = head[4:]
    m = _BASE_RE.match(head)
    return m.group(1) if m else head


@dataclass
class ClusterMap:
    """Symbol -> cluster lookup with sensible defaults."""

    explicit: dict[str, str] = field(default_factory=dict)
    default_cluster: str = "other"

    def cluster_of(self, symbol: str) -> str:
        if not symbol:
            return self.default_cluster
        base = _base_token(symbol)
        # Explicit mapping wins both for full symbol and bare base token.
        if symbol.upper() in self.explicit:
            return self.explicit[symbol.upper()]
        if base in self.explicit:
            return self.explicit[base]
        return self.default_cluster

    def cluster_counts(
        self, open_symbols: list[str],
    ) -> dict[str, int]:
        """Tally currently-open symbols by cluster."""
        counts: dict[str, int] = {}
        for sym in open_symbols:
            c = self.cluster_of(sym)
            counts[c] = counts.get(c, 0) + 1
        return counts


@dataclass
class ClusterCapConfig:
    enabled: bool = True
    max_per_cluster: int = 1


def cap_breached(
    *,
    proposed_symbol: str,
    open_symbols: list[str],
    cluster_map: ClusterMap,
    cap_cfg: ClusterCapConfig,
) -> tuple[bool, str]:
    """Pure helper used by RiskGate.

    Returns ``(True, reason)`` if opening a position on
    ``proposed_symbol`` would push its cluster past the cap. The
    cluster of ``proposed_symbol`` is included in the count so the
    check is consistent regardless of whether the symbol is already
    in ``open_symbols``.
    """
    if not cap_cfg.enabled:
        return False, ""
    proposed_cluster = cluster_map.cluster_of(proposed_symbol)
    counts = cluster_map.cluster_counts(open_symbols)
    # If the proposed symbol is not already counted, add 1.
    if proposed_symbol not in open_symbols:
        counts[proposed_cluster] = counts.get(proposed_cluster, 0) + 1
    current = counts.get(proposed_cluster, 0)
    if current > cap_cfg.max_per_cluster:
        return True, (
            f"cluster_cap:{proposed_cluster}:{current}>"
            f"{cap_cfg.max_per_cluster}"
        )
    return False, ""
