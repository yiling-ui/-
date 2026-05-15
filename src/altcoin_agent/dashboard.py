"""dashboard.py — Web dashboard for live monitoring + backtest results.

Mounted on the same aiohttp app as ``/healthz``. No frontend framework —
just a single self-contained HTML page that polls JSON endpoints every
5 seconds.

Routes:

    GET /dashboard           - HTML page
    GET /api/state           - HealthState + recent counters
    GET /api/signals         - last N high-priority signals
    GET /api/positions       - currently-open positions
    GET /api/orders          - last N orders sent (dry-run or live)
    GET /api/rules           - current dynamic_rules.json snapshot
    GET /api/backtest-rules  - same as /api/rules but easier name in UI
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """In-memory ring buffers populated by main.py callbacks."""

    health: Any = None      # HealthState
    account: Any = None     # AccountState
    rules_path: Path | None = None
    recent_signals: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50),
    )
    recent_orders: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50),
    )
    recent_rejections: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50),
    )

    def push_signal(self, payload: dict[str, Any]) -> None:
        self.recent_signals.append(payload)

    def push_order(self, payload: dict[str, Any]) -> None:
        self.recent_orders.append(payload)

    def push_rejection(self, payload: dict[str, Any]) -> None:
        self.recent_rejections.append(payload)


# --------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------- #


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Altcoin Agent V1.0 — Dashboard</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}
  header{background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}
  header h1{margin:0;font-size:16px;font-weight:600}
  header .pill{padding:4px 10px;border-radius:12px;font-size:12px}
  .ok{background:#1f6f3a;color:#fff}
  .deg{background:#a14a00;color:#fff}
  main{padding:14px;max-width:1300px;margin:0 auto;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
  .card h2{margin:0 0 8px;font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}
  .kv b{color:#8b949e;font-weight:500}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #21262d;white-space:nowrap}
  th{color:#8b949e;font-weight:500}
  tr:hover td{background:#1c2128}
  .long{color:#3fb950}
  .short{color:#f85149}
  .neutral{color:#8b949e}
  .err{color:#f85149}
  footer{padding:10px;text-align:center;font-size:11px;color:#6e7681}
</style>
</head>
<body>
<header>
  <h1>Altcoin Agent V1.0 — Live Dashboard</h1>
  <span id="status" class="pill">…</span>
</header>
<main>
  <div class="card">
    <h2>System</h2>
    <div id="state" class="kv"></div>
  </div>
  <div class="card">
    <h2>Open Positions</h2>
    <table id="positions"><thead><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>Stop</th><th>Lev</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card">
    <h2>Recent Signals (last 50)</h2>
    <table id="signals"><thead><tr><th>Time</th><th>Sym</th><th>Dir</th><th>Score</th><th>Trigger</th><th>Notes</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card">
    <h2>Recent Orders (last 50)</h2>
    <table id="orders"><thead><tr><th>Time</th><th>Sym</th><th>Side</th><th>Size</th><th>Type</th><th>Stop</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Dynamic Rules (top 20 by hit rate)</h2>
    <table id="rules"><thead><tr><th>Feature</th><th>Bucket</th><th>Side</th><th>Hits/Total</th><th>Hit Rate</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Recent Rejections (last 50)</h2>
    <table id="rejections"><thead><tr><th>Time</th><th>Sym</th><th>Reason</th></tr></thead><tbody></tbody></table>
  </div>
</main>
<footer>auto-refresh every 5s · /healthz · /api/state · /api/signals · /api/positions · /api/orders · /api/rules</footer>
<script>
function fmtTime(ms){if(!ms)return "-";const d=new Date(ms);return d.toISOString().slice(11,19)+"Z"}
function row(parent,cells,cls){const tr=document.createElement("tr");for(const c of cells){const td=document.createElement("td");td.textContent=c==null?"-":String(c);if(cls&&c===cells[2])td.className=cls;tr.appendChild(td)}parent.appendChild(tr)}
async function refresh(){
  try{
    const [s,sigs,pos,ords,rules,rejs]=await Promise.all([
      fetch("/api/state").then(r=>r.json()),
      fetch("/api/signals").then(r=>r.json()),
      fetch("/api/positions").then(r=>r.json()),
      fetch("/api/orders").then(r=>r.json()),
      fetch("/api/rules").then(r=>r.json()),
      fetch("/api/rejections").then(r=>r.json()),
    ]);
    const status=document.getElementById("status");
    status.textContent=s.status||"?";
    status.className="pill "+(s.status==="ok"?"ok":"deg");
    const k=document.getElementById("state");k.innerHTML="";
    for(const [name,val] of [
      ["Mode", s.mode||"-"],
      ["Uptime (sec)", s.uptime_sec],
      ["Reconciliation", s.reconciliation_complete],
      ["Rule events seen", s.rule_event_count],
      ["High-priority signals", s.high_priority_count],
      ["Orders placed", s.orders_placed],
      ["Orders rejected", s.orders_rejected],
      ["Open positions", s.open_positions],
      ["Last signal ts", fmtTime((s.last_signal_ts||0)*1000)],
      ["Last error", s.last_error||"-"],
    ]){
      const b=document.createElement("b");b.textContent=name;
      const v=document.createElement("span");v.textContent=val;
      if(name==="Last error" && val && val!=="-")v.className="err";
      k.appendChild(b);k.appendChild(v);
    }
    const sigBody=document.querySelector("#signals tbody");sigBody.innerHTML="";
    for(const x of sigs.slice().reverse()){
      const cls=x.direction==="long"?"long":x.direction==="short"?"short":"neutral";
      row(sigBody,[fmtTime(x.ts),x.symbol,x.direction,x.final_score,x.trigger_price,(x.notes||[]).slice(-1)[0]||""],cls);
    }
    const posBody=document.querySelector("#positions tbody");posBody.innerHTML="";
    for(const x of pos)row(posBody,[x.symbol,x.side,x.size,x.entry_price,x.current_stop,x.leverage],x.side==="long"?"long":"short");
    const ordBody=document.querySelector("#orders tbody");ordBody.innerHTML="";
    for(const x of ords.slice().reverse())row(ordBody,[fmtTime(x.ts),x.symbol,x.side,x.size,x.type,x.stop_price||""]);
    const rulesBody=document.querySelector("#rules tbody");rulesBody.innerHTML="";
    for(const r of rules.top){
      const pct=(r.hit_rate*100).toFixed(1)+"%";
      row(rulesBody,[r.feature_name,r.bucket,r.side,`${r.hits}/${r.total}`,pct],r.side==="pump"?"long":"short");
    }
    const rejBody=document.querySelector("#rejections tbody");rejBody.innerHTML="";
    for(const x of rejs.slice().reverse())row(rejBody,[fmtTime(x.ts),x.symbol,x.reason]);
  }catch(e){console.error(e);}
}
refresh();setInterval(refresh,5000);
</script>
</body>
</html>"""


# --------------------------------------------------------------------- #
# Route handlers
# --------------------------------------------------------------------- #


def _build_routes(state: DashboardState, mode_label: str) -> list[web.RouteDef]:
    async def page(_req: web.Request) -> web.Response:
        return web.Response(text=_HTML, content_type="text/html")

    async def api_state(_req: web.Request) -> web.Response:
        h = state.health
        body: dict[str, Any] = {"mode": mode_label}
        if h is not None:
            body.update({
                "status": "ok" if (
                    getattr(h, "fuser_alive", False)
                    and getattr(h, "screener_alive", False)
                    and getattr(h, "reconciliation_complete", False)
                ) else "degraded",
                "uptime_sec": _safe_uptime(h),
                "reconciliation_complete": getattr(h, "reconciliation_complete", False),
                "rule_event_count": getattr(h, "rule_event_count", 0),
                "high_priority_count": getattr(h, "high_priority_count", 0),
                "orders_placed": getattr(h, "orders_placed", 0),
                "orders_rejected": getattr(h, "orders_rejected", 0),
                "open_positions": getattr(h, "open_positions", 0),
                "last_signal_ts": getattr(h, "last_signal_ts", 0.0),
                "last_error": getattr(h, "last_error", None),
            })
        return web.json_response(body)

    async def api_signals(_req: web.Request) -> web.Response:
        return web.json_response(list(state.recent_signals))

    async def api_orders(_req: web.Request) -> web.Response:
        return web.json_response(list(state.recent_orders))

    async def api_rejections(_req: web.Request) -> web.Response:
        return web.json_response(list(state.recent_rejections))

    async def api_positions(_req: web.Request) -> web.Response:
        rows: list[dict[str, Any]] = []
        if state.account is not None:
            for p in getattr(state.account, "open_positions", {}).values():
                rows.append({
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "current_stop": p.current_stop,
                    "leverage": p.leverage,
                })
        return web.json_response(rows)

    async def api_rules(_req: web.Request) -> web.Response:
        if state.rules_path is None or not state.rules_path.exists():
            return web.json_response({"top": [], "archived": [], "total": 0})
        try:
            data = json.loads(state.rules_path.read_text())
        except Exception as e:
            return web.json_response(
                {"top": [], "archived": [], "total": 0, "error": str(e)},
                status=200,
            )
        rules = data.get("rules", [])
        annotated: list[dict[str, Any]] = []
        for r in rules:
            hits = int(r.get("hits", 0))
            total = int(r.get("total", 0))
            hit_rate = (hits + 1) / (total + 2)
            annotated.append({**r, "hit_rate": hit_rate})
        annotated.sort(key=lambda r: (-r["hit_rate"], -r["total"]))
        return web.json_response({
            "top": annotated[:20],
            "archived": [r for r in annotated[20:] if r["hit_rate"] < 0.5],
            "total": len(annotated),
        })

    return [
        web.get("/dashboard", page),
        web.get("/api/state", api_state),
        web.get("/api/signals", api_signals),
        web.get("/api/orders", api_orders),
        web.get("/api/rejections", api_rejections),
        web.get("/api/positions", api_positions),
        web.get("/api/rules", api_rules),
    ]


def _safe_uptime(h: Any) -> float:
    started = getattr(h, "started_at", 0.0) or 0.0
    if started == 0:
        return 0.0
    import time
    return round(time.time() - started, 1)


def install_dashboard(
    app: web.Application,
    state: DashboardState,
    mode_label: str = "DRY-RUN",
) -> None:
    """Install dashboard routes on an existing aiohttp app."""
    app.add_routes(_build_routes(state, mode_label))
