"""
demo_learning.py — closed-loop demo of the Learning Engine.

Runs a synthetic DUMP scenario (textbook distribution-then-waterfall) end
to end:

    1. Build a 4h pre-event slice in memory (no network).
    2. Compute features + realized result.
    3. Pick prognostic features (DeepSeek if DEEPSEEK_API_KEY is set,
       deterministic fallback otherwise).
    4. Update the persistent rule store; regenerate
       .kiro/steering/dynamic_rules.md.
    5. Print everything so you can see exactly what was learned.

Then runs a second synthetic event of the SAME shape — to demonstrate
that the rule's `hits/total` accumulates instead of duplicating.

Run:
    python examples/demo_learning.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from altcoin_agent.ai_engine import DeepSeekEngine
from altcoin_agent.learning_engine import (
    RuleStore,
    run_post_mortem,
    synthesize_dump_slice,
)


def banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def kv(label: str, value: object) -> None:
    print(f"    {label:.<32}{value}")


async def main() -> None:
    # Pathlib calls here are cheap and synchronous; using anyio/trio
    # would be overkill for a one-shot demo script.
    project_root = Path(__file__).resolve().parents[1]  # noqa: ASYNC240
    steering_dir = project_root / ".kiro" / "steering"
    json_path = steering_dir / "dynamic_rules.json"
    md_path = steering_dir / "dynamic_rules.md"

    # Start clean for the demo so the printed file is fully attributable
    # to this run. (In real use, the store accumulates across sessions.)
    if json_path.exists():       # noqa: ASYNC240
        json_path.unlink()       # noqa: ASYNC240
    if md_path.exists():         # noqa: ASYNC240
        md_path.unlink()         # noqa: ASYNC240

    store = RuleStore(json_path=json_path, md_path=md_path)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    engine: DeepSeekEngine | None = None
    if api_key:
        engine = DeepSeekEngine(api_key=api_key)
        print("\n[info] DEEPSEEK_API_KEY found — calling REAL DeepSeek.")
    else:
        print("\n[info] DEEPSEEK_API_KEY not set — using deterministic fallback.")
        print("       (Set DEEPSEEK_API_KEY=sk-... to enable LLM picks.)")

    banner("EVENT 1 — synthetic $RAVE distribution-then-dump")

    target_ts_ms = 1_700_000_000_000
    slc = synthesize_dump_slice(symbol="RAVEUSDT", target_ts_ms=target_ts_ms)
    kv("slice klines", len(slc.klines_1m))
    kv("slice funding samples", len(slc.funding_rates))
    kv("slice OI samples", len(slc.open_interest))

    run = await run_post_mortem(
        symbol="RAVEUSDT",
        target_ts_ms=target_ts_ms,
        engine=engine,
        rule_store=store,
        slice_override=slc,
    )

    print()
    print("  --- Realized event ---")
    if run.result:
        kv("side", run.result.side.value)
        kv("magnitude", f"{run.result.magnitude_pct*100:.2f}%")
        kv("minutes_to_extreme", run.result.minutes_to_extreme)

    print()
    print("  --- Candidate features (8 max) ---")
    for f in run.features:
        kv(
            f.name,
            f"value={f.value:+.6f}  bucket={f.bucket}  hint={f.direction_hint}",
        )

    print()
    print("  --- Post-mortem verdict ---")
    kv("source", "fallback (no LLM)" if run.used_fallback else "DeepSeek")
    if run.verdict:
        for p in run.verdict.picks:
            kv(f"pick #{p.rank}", f"{p.feature_name} — {p.rationale[:80]}")
        kv("summary", run.verdict.summary[:120])

    print()
    print("  --- Rules updated ---")
    for r in run.touched_rules:
        kv(r.rule_id, f"hits/total={r.hits}/{r.total}  hit_rate={r.hit_rate:.2%}")

    banner("EVENT 2 — same shape (to demonstrate Bayesian accumulation)")
    target_ts_ms_2 = target_ts_ms + 7 * 24 * 3600 * 1000  # +1 week
    slc2 = synthesize_dump_slice(symbol="MYXUSDT", target_ts_ms=target_ts_ms_2)
    run2 = await run_post_mortem(
        symbol="MYXUSDT",
        target_ts_ms=target_ts_ms_2,
        engine=engine,
        rule_store=store,
        slice_override=slc2,
    )
    print()
    print("  --- Rules after second event ---")
    for r in run2.touched_rules:
        kv(r.rule_id, f"hits/total={r.hits}/{r.total}  hit_rate={r.hit_rate:.2%}")

    banner("PERSISTED FILES")
    kv("json_path", json_path)
    kv("md_path", md_path)
    kv("json size (bytes)", json_path.stat().st_size if json_path.exists() else 0)
    kv("md size (bytes)", md_path.stat().st_size if md_path.exists() else 0)

    print()
    print("  --- dynamic_rules.json contents (truncated) ---")
    if json_path.exists():
        body = json.loads(json_path.read_text())
        for r in body.get("rules", [])[:5]:
            print(
                f"    {r['feature_name']}|{r['bucket']}|{r['side']}: "
                f"hits/total={r['hits']}/{r['total']}"
            )
        print(f"    ... ({len(body.get('rules', []))} rules total)")

    print()
    print("  --- dynamic_rules.md (full file) ---")
    print()
    if md_path.exists():
        for line in md_path.read_text().splitlines():
            print(f"      {line}")

    banner("LOOP CLOSED")
    print("    The Fuser and AI Engine will read .kiro/steering/dynamic_rules.md")
    print("    on their next run. Strategy has evolved.")


if __name__ == "__main__":
    asyncio.run(main())
