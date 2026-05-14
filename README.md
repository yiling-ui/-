# Altcoin Momentum Agent

Multimodal AI trading agent that hunts early 10x pumps / waterfall dumps on
low-cap altcoins (RAVE, MYX, etc.) by fusing:

1. Real-time market & on-chain features (volume, funding, OI, SMC liquidity sweeps)
2. Social sentiment (Binance Square, KOL feeds)
3. DeepSeek LLM judgement on whether KOLs are front-running or distributing

See [`.kiro/specs/altcoin-momentum-agent/`](./.kiro/specs/altcoin-momentum-agent)
for the full requirements / design / task spec.

## Status

| Module | Status |
|---|---|
| A. Market & On-Chain Screener | initial implementation in `src/altcoin_agent/screener.py` |
| B. Social Sentiment Crawler | spec only |
| C. AI Inference Engine (DeepSeek) | initial implementation in `src/altcoin_agent/ai_engine.py` |
| D. Risk Gate + Executor | spec only |
| E. Backtest & RL Lab | spec only |

## Quickstart (local dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# run unit tests (no network required)
pytest -q

# run a fully-mocked demo of the screener
python examples/demo_mock.py
```

## Using the AI engine

```python
import os, asyncio
from altcoin_agent import DeepSeekEngine, SMCContext, SocialPost

os.environ["DEEPSEEK_API_KEY"] = "sk-..."

async def main():
    async with DeepSeekEngine(model="deepseek-chat") as eng:
        verdict = await eng.judge(
            symbol="RAVEUSDT",
            exchange="binance",
            funding_rate=-0.0025,
            funding_deviation_z=-3.5,
            smc=SMCContext(volume_spike={"zscore": 7.2, "side": "buy"}),
            posts=[SocialPost(author="@whale", follower_count=120_000,
                              text="$RAVE loading", ts=1)],
        )
        print(verdict.model_dump_json(indent=2))

asyncio.run(main())
```

The engine returns a strict `AIVerdict`:

```json
{
  "intent": "pump",
  "confidence_score": 88,
  "reason": "Negative funding, OI silent build, sweep above equal highs.",
  "kol_intent": "frontrun_call",
  "key_evidence": ["funding -0.25%", "OI +22% in 5m", "sweep wick 2.1x body"]
}
```

Failure modes (timeout / invalid JSON / budget) all degrade to
`{"intent":"neutral","confidence_score":0,"reason":"degraded: ..."}` instead
of raising — the upstream score-fuser then weights LLM at 0.
