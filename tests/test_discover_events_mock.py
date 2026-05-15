"""Tests for the discover_events.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load discover_events.py as a module (it lives in scripts/, not in the package)
_script = Path(__file__).resolve().parent.parent / "scripts" / "discover_events.py"
_spec = importlib.util.spec_from_file_location("discover_events", _script)
discover = importlib.util.module_from_spec(_spec)
sys.modules["discover_events"] = discover
_spec.loader.exec_module(discover)  # type: ignore[union-attr]


def test_to_okx_inst_id_swap_form() -> None:
    assert discover._to_okx_inst_id("RAVEUSDT") == "RAVE-USDT-SWAP"
    assert discover._to_okx_inst_id("MYX/USDT:USDT") == "MYX-USDT-SWAP"
    assert discover._to_okx_inst_id("PEPE-USDT-SWAP") == "PEPE-USDT-SWAP"


def test_events_from_candles_finds_pumps_and_dumps() -> None:
    candles = [
        # (ts, open, high, low, close)
        (1, 1.00, 1.30, 0.99, 1.25),  # +30% pump
        (2, 1.25, 1.26, 1.24, 1.25),  # ~0% noise
        (3, 1.25, 1.26, 0.85, 0.90),  # -32% dump
    ]
    out = discover._events_from_candles("RAVEUSDT", candles, min_move_pct=0.15)
    assert len(out) == 2
    assert out[0]["direction"] == "pump"
    assert out[0]["magnitude_pct"] >= 0.30
    assert out[1]["direction"] == "dump"
    assert out[1]["magnitude_pct"] >= 0.30


def test_events_from_candles_threshold_filter() -> None:
    candles = [(1, 1.00, 1.05, 0.95, 1.02)]   # only +5%/-5%
    out = discover._events_from_candles("X", candles, min_move_pct=0.10)
    assert out == []
