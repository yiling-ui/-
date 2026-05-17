"""Tests for ProductionRulesLoader + run_walkforward_trainer CLI."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from altcoin_agent.training.production_rules_loader import (
    ProductionRulesLoader,
)
from altcoin_agent.training.rules_promoter import LearnedRule


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _write_rules_file(path: Path, rules: list[LearnedRule]) -> None:
    payload = {
        "version": 1,
        "saved_at_ts": int(time.time()),
        "rules": [r.as_dict() for r in rules],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _rule(rule_id: str, win_rate: float = 0.85, samples: int = 50,
          extras: dict | None = None) -> LearnedRule:
    return LearnedRule(
        rule_id=rule_id,
        samples=samples,
        wins=int(samples * win_rate),
        losses=int(samples * (1 - win_rate)),
        win_rate=win_rate,
        avg_pnl_pct=0.05,
        sharpe=1.8,
        validation_months_passed=3,
        production_ready=True,
        extras=extras or {},
    )


# --------------------------------------------------------------------- #
# Loader mtime + throttle behaviour
# --------------------------------------------------------------------- #


def test_loader_reads_file_on_first_call(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1"), _rule("r2", win_rate=0.92)])

    loader = ProductionRulesLoader(
        path=str(p), min_check_interval_sec=0.0,
    )
    changed = loader.maybe_reload()
    assert changed is True
    assert len(loader) == 2
    by_id = {r.rule_id for r in loader.rules()}
    assert by_id == {"r1", "r2"}


def test_loader_throttled_repeat_does_not_reread(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])
    loader = ProductionRulesLoader(
        path=str(p), min_check_interval_sec=999.0,
    )
    assert loader.maybe_reload() is True
    # Mutate file (different rule), but throttle still in effect.
    _write_rules_file(p, [_rule("r2")])
    assert loader.maybe_reload() is False
    # Snapshot is unchanged.
    assert {r.rule_id for r in loader.rules()} == {"r1"}


def test_loader_force_reload_bypasses_throttle(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])
    loader = ProductionRulesLoader(
        path=str(p), min_check_interval_sec=999.0,
    )
    loader.force_reload()
    _write_rules_file(p, [_rule("r2")])
    # Bump mtime explicitly so a same-second re-write is observable.
    os.utime(p, (time.time() + 1, time.time() + 1))
    loader.force_reload()
    assert {r.rule_id for r in loader.rules()} == {"r2"}


def test_loader_keeps_snapshot_on_malformed_json(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])
    loader = ProductionRulesLoader(path=str(p), min_check_interval_sec=0.0)
    loader.maybe_reload()
    # Corrupt the file.
    p.write_text("{ this is not json")
    os.utime(p, (time.time() + 1, time.time() + 1))
    # Reload returns False (no successful reload) but old snapshot
    # survives.
    loader.maybe_reload()
    assert {r.rule_id for r in loader.rules()} == {"r1"}


def test_loader_handles_missing_file(tmp_path):
    p = tmp_path / "nope.json"
    loader = ProductionRulesLoader(path=str(p), min_check_interval_sec=0.0)
    assert loader.maybe_reload() is False
    assert loader.rules() == []


def test_loader_clears_when_file_disappears(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])
    loader = ProductionRulesLoader(path=str(p), min_check_interval_sec=0.0)
    loader.maybe_reload()
    assert loader.rules()
    p.unlink()
    loader.maybe_reload()
    assert loader.rules() == []


def test_loader_lookup_by_features(tmp_path):
    p = tmp_path / "production_rules.json"
    rules = [
        _rule(
            "r_quad_phase=A_ramp",
            extras={"feature_keys": ["quadrant", "phase"], "value": "A/ramp"},
        ),
        _rule(
            "r_quad_phase=B_parabolic",
            extras={"feature_keys": ["quadrant", "phase"], "value": "B/parabolic"},
        ),
    ]
    _write_rules_file(p, rules)
    loader = ProductionRulesLoader(path=str(p), min_check_interval_sec=0.0)
    loader.maybe_reload()

    found = loader.lookup_by_features(
        feature_keys=["quadrant", "phase"], value_key="A/ramp",
    )
    assert found is not None
    assert found.rule_id == "r_quad_phase=A_ramp"

    missing = loader.lookup_by_features(
        feature_keys=["quadrant", "phase"], value_key="C/dead",
    )
    assert missing is None


def test_loader_on_reload_callback_fires(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])
    seen: list[int] = []
    loader = ProductionRulesLoader(
        path=str(p),
        min_check_interval_sec=0.0,
        on_reload=lambda rules: seen.append(len(rules)),
    )
    loader.maybe_reload()
    assert seen == [1]
    # No mtime change -> no callback.
    loader.maybe_reload()
    assert seen == [1]
    # New rule + bumped mtime -> callback fires again.
    _write_rules_file(p, [_rule("r1"), _rule("r2")])
    os.utime(p, (time.time() + 2, time.time() + 2))
    loader.maybe_reload()
    assert seen == [1, 2]


def test_loader_callback_exception_swallowed(tmp_path):
    p = tmp_path / "production_rules.json"
    _write_rules_file(p, [_rule("r1")])

    def boom(_rules):
        raise RuntimeError("intentional")

    loader = ProductionRulesLoader(
        path=str(p), min_check_interval_sec=0.0, on_reload=boom,
    )
    # Loader must not propagate the callback's exception.
    loader.maybe_reload()
    assert loader.rules()  # snapshot still populated


# --------------------------------------------------------------------- #
# CLI smoke (replay-only path) — exercises run_walkforward_trainer end-to-end
# --------------------------------------------------------------------- #


def test_run_walkforward_trainer_cli_replay_only(tmp_path):
    obs_path = tmp_path / "obs.jsonl"
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Build observations: 30 wins, 5 losses with the same feature combo.
    lines = []
    base_ts = 1_700_000_000_000
    for i in range(30):
        lines.append(json.dumps({
            "ts_ms": base_ts + i * 86_400_000,
            "symbol": "PEPE/USDT:USDT",
            "pnl_pct": 0.10,
            "features": {"quadrant": "A", "phase": "ramp"},
        }))
    for i in range(5):
        lines.append(json.dumps({
            "ts_ms": base_ts + (30 + i) * 86_400_000,
            "symbol": "PEPE/USDT:USDT",
            "pnl_pct": -0.05,
            "features": {"quadrant": "A", "phase": "ramp"},
        }))
    obs_path.write_text("\n".join(lines))

    # Import the CLI module lazily.
    from scripts import run_walkforward_trainer

    rc = run_walkforward_trainer.run([
        "--observations", str(obs_path),
        "--state-dir", str(state_dir),
        "--replay-only",
        # Lower validation gate so the test produces a promotion.
        "--min-validation-months", "0",
    ])
    assert rc == 0
    # Either production OR candidate file (or both) must exist.
    prod = state_dir / "production_rules.json"
    cand = state_dir / "candidate_rules.json"
    assert prod.exists() or cand.exists()
