"""
The paper-trading dashboard — two endpoints, no dependencies.
--------------------------------------------------------------

`GET /api/status` is the whole contract: one JSON document describing every
algo and the portfolio. `GET /` is a static page that polls it.

Served by aiohttp on the engine's own event loop, so there is no second
process to keep alive and no IPC. The handler calls exactly one method on the
board — a pure query — and never awaits the engine, mutates a book, or takes a
lock; StatusBoard's synchronous status() is what makes that safe.

The page is deliberately self-contained: no CDN, no framework, no build step.
It works with no network beyond localhost, and there is no third-party script
in the path of a page that displays trading state.
"""

from typing import Any

from aiohttp import web


# ── Page ───────────────────────────────────────────────────────────────

class DashboardPage:
    def html(self) -> str:
        return _PAGE


# ── Endpoints ──────────────────────────────────────────────────────────

class PageEndpoint:
    def __init__(self, page: DashboardPage) -> None:
        self._page = page

    async def handle(self, request: web.Request) -> web.Response:
        return web.Response(text=self._page.html(), content_type="text/html")


class StatusEndpoint:
    def __init__(self, board: Any) -> None:
        self._board = board

    async def handle(self, request: web.Request) -> web.Response:
        return web.json_response(self._board.payload())


# ── Application ────────────────────────────────────────────────────────
# Split from the site so tests can drive the routes without binding a port.

class DashboardApp:
    def __init__(self, board: Any) -> None:
        self._board = board

    def application(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", PageEndpoint(DashboardPage()).handle)
        app.router.add_get("/api/status", StatusEndpoint(self._board).handle)
        return app


class DashboardSite:
    def __init__(self, app: DashboardApp, host: str, port: int) -> None:
        self._app  = app
        self._host = host
        self._port = port
        self._runner: Any = None

    # Returns once the socket is listening — it does not block the caller, so
    # the engine's loops start right after.
    async def start(self) -> None:
        runner = web.AppRunner(self._app.application())
        await runner.setup()
        await web.TCPSite(runner, self._host, self._port).start()
        self._runner = runner

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


# ── Markup ─────────────────────────────────────────────────────────────

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Paper Trading</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --line: #262d36; --text: #e6edf3;
    --dim: #8b949e; --up: #3fb950; --down: #f85149; --warn: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  header { padding: 16px 20px; border-bottom: 1px solid var(--line);
           display: flex; flex-wrap: wrap; gap: 24px; align-items: baseline; }
  h1 { font-size: 15px; margin: 0; letter-spacing: .08em; text-transform: uppercase; }
  .meta { color: var(--dim); font-size: 12px; }
  .cards { display: flex; flex-wrap: wrap; gap: 1px; background: var(--line);
           border-bottom: 1px solid var(--line); }
  .card { flex: 1 1 150px; background: var(--panel); padding: 12px 16px; }
  .card .k { color: var(--dim); font-size: 11px; text-transform: uppercase;
             letter-spacing: .06em; }
  .card .v { font-size: 20px; margin-top: 4px; }
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 1100px; }
  th, td { padding: 8px 12px; text-align: right; white-space: nowrap;
           border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-size: 11px; text-transform: uppercase;
       letter-spacing: .06em; text-align: right; position: sticky; top: 0;
       background: var(--bg); }
  th:first-child, td:first-child, th.l, td.l { text-align: left; }
  .up { color: var(--up); } .down { color: var(--down); }
  .pill { font-size: 11px; padding: 2px 8px; border-radius: 10px;
          border: 1px solid currentColor; }
  .pill.ok { color: var(--up); } .pill.err { color: var(--down); }
  .pill.risk { color: var(--warn); }
  .err-row td { color: var(--down); font-size: 12px; text-align: left; }
  footer { padding: 12px 20px; color: var(--dim); font-size: 12px; }
</style>
<header>
  <h1>Paper Trading</h1>
  <span class="meta">next tick <b id="next">-</b></span>
  <span class="meta">price age <b id="age">-</b></span>
  <span class="meta">updated <b id="gen">-</b></span>
</header>
<div class="cards" id="cards"></div>
<div class="wrap">
  <table>
    <thead><tr>
      <th class="l">Algo</th><th class="l">Venue</th><th class="l">Pair</th>
      <th class="l">Candle</th><th class="l">Status</th><th class="l">Last</th>
      <th>Mark</th><th>Position</th><th>Entry</th><th>Liq</th>
      <th>Unreal</th><th>Unreal %</th>
      <th>Balance</th><th>Equity</th><th>Realized</th>
      <th>Fees</th><th>Interest</th>
      <th>Trades</th><th>Win %</th><th>MaxDD</th><th>Return</th><th>Ann.</th>
      <th>RSI</th><th>MACD</th><th>Score</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>
<footer>Simulated books only &mdash; no orders are ever placed.</footer>
<script>
// Exchange error bodies reach this page verbatim and are often HTML, so
// nothing untrusted is interpolated into innerHTML unescaped.
const esc = s => String(s === null || s === undefined ? "" : s).replace(
  /[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;",
                     '"': "&quot;", "'": "&#39;"}[c]));
const n = (v, d = 2) => (v === null || v === undefined) ? "-" : Number(v).toFixed(d);
const pct = v => (v === null || v === undefined) ? "-" : (Number(v) * 100).toFixed(2) + "%";
const cls = v => Number(v) > 0 ? "up" : (Number(v) < 0 ? "down" : "");
const dur = s => {
  if (s === null || s === undefined || s < 0) return "-";
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return m > 0 ? m + "m " + r + "s" : r + "s";
};

function cards(p) {
  const items = [
    ["Equity", n(p.equity), cls(p.equity - p.starting_balance)],
    ["Return", pct(p.total_return), cls(p.total_return)],
    ["Realized", n(p.realized_pnl), cls(p.realized_pnl)],
    ["Unrealized", n(p.unrealized_pnl), cls(p.unrealized_pnl)],
    ["Win rate", pct(p.win_rate), ""],
    ["Algos OK", p.algos_ok + " / " + (p.algos_ok + p.algos_errored),
      p.algos_errored ? "down" : ""],
  ];
  document.getElementById("cards").innerHTML = items.map(
    ([k, v, c]) => `<div class="card"><div class="k">${k}</div>
                    <div class="v ${c}">${v}</div></div>`).join("");
}

function rows(algos) {
  const out = [];
  for (const a of algos) {
    const p = a.position;
    // Warn while the mark is within 5% of the liquidation price; a short with
    // no borrow reports an infinite (null over JSON) liquidation price.
    // A 1x long has no liquidation level (IsolatedMargin reports 0.0), and an
    // unborrowed short has none either (reported as absent) — neither is a
    // price to display.
    const liq = p && p.liquidation_price ? p.liquidation_price : null;
    const near = liq !== null && liq > 0 &&
      Math.abs(a.mark_price - liq) / liq < 0.05;
    out.push(`<tr>
      <td class="l">${esc(a.name)}</td><td class="l">${esc(a.exchange)}</td>
      <td class="l">${esc(a.pair)}</td>
      <td class="l">${esc(a.granularity)}</td>
      <td class="l"><span class="pill ${a.running ? "ok" : "err"}">
        ${a.running ? "RUNNING" : "ERROR"}</span>
        ${near ? ' <span class="pill risk">AT RISK</span>' : ""}</td>
      <td class="l">${esc(a.last_action)}</td>
      <td>${n(a.mark_price, 4)}</td>
      <td>${p ? esc(p.direction) + " " + n(p.size, 6) : "flat"}</td>
      <td>${p ? n(p.entry_price, 4) : "-"}</td>
      <td>${liq !== null ? n(liq, 4) : "-"}</td>
      <td class="${cls(a.unrealized_pnl)}">${n(a.unrealized_pnl)}</td>
      <td class="${cls(a.unrealized_pnl)}">${p ? pct(p.unrealized_return) : "-"}</td>
      <td>${n(a.balance)}</td>
      <td>${n(a.equity)}</td>
      <td class="${cls(a.realized_pnl)}">${n(a.realized_pnl)}</td>
      <td>${n(a.fee_paid)}</td>
      <td>${n(a.interest_paid)}</td>
      <td>${a.trades}</td><td>${pct(a.win_rate)}</td>
      <td>${pct(a.max_drawdown)}</td>
      <td class="${cls(a.total_return)}">${pct(a.total_return)}</td>
      <td class="${cls(a.annualized_yield)}">${pct(a.annualized_yield)}</td>
      <td>${n(a.rsi, 1)}</td><td>${n(a.macd, 4)}</td><td>${n(a.signal_score, 3)}</td>
    </tr>`);
    if (a.error) {
      out.push(`<tr class="err-row"><td colspan="25">${esc(a.name)}: ${esc(a.error)}</td></tr>`);
    }
  }
  document.getElementById("rows").innerHTML = out.join("");
}

async function refresh() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("next").textContent = dur(d.next_tick_in);
    document.getElementById("age").textContent = dur(d.seconds_since_price);
    document.getElementById("gen").textContent = d.generated_at;
    cards(d.portfolio);
    rows(d.algos);
  } catch (e) {
    document.getElementById("gen").textContent = "unreachable";
  }
}
refresh();
setInterval(refresh, 5000);
</script>
"""
