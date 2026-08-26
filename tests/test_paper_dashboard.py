import re

import pytest
from aiohttp.test_utils import TestClient, TestServer

from coinbase.ga.paper_dashboard import DashboardApp, DashboardPage


# ── Test doubles ─────────────────────────────────────────────────────

class _RecordingBoard:
    """Stands in for StatusBoard — records that only payload() was ever called.

    It deliberately implements no tick()/mark(): a handler that tried to drive
    the engine would raise AttributeError rather than quietly working.
    """

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls    = 0

    def payload(self) -> dict:
        self.calls += 1
        return self._payload


def _payload() -> dict:
    return {
        "started_at": "2026-08-26T00:00:00+00:00",
        "generated_at": "2026-08-26T12:00:00+00:00",
        "next_tick_in": 120.0,
        "seconds_since_price": 5.0,
        "portfolio": {
            "starting_balance": 1000.0, "equity": 1100.0, "realized_pnl": 50.0,
            "unrealized_pnl": 50.0, "total_return": 0.1, "win_rate": 0.5,
            "algos_ok": 1, "algos_errored": 0,
        },
        "algos": [{
            "name": "btc", "exchange": "coinbase", "pair": "BTC-USDC",
            "granularity": "THIRTY_MINUTE", "running": True, "error": None,
            "last_tick_at": None, "last_candle_start": 1800, "last_action": "BUY",
            "starting_balance": 1000.0, "balance": 1000.0, "mark_price": 110.0,
            "equity": 1100.0, "realized_pnl": 0.0, "unrealized_pnl": 100.0,
            "trades": 2, "wins": 1, "rsi": 55.0, "macd": 0.25,
            "signal_score": 0.5, "fee_paid": 0.0, "position": None,
            "win_rate": 0.5, "total_return": 0.1,
            "annualized_yield": 0.2, "max_drawdown": 0.05,
        }],
    }


@pytest.fixture
async def client():
    board = _RecordingBoard(_payload())
    async with TestClient(TestServer(DashboardApp(board).application())) as test_client:
        test_client.board = board
        yield test_client


# ── /api/status ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_status_endpoint_serves_the_boards_payload_as_json(client):
    response = await client.get("/api/status")

    assert response.status == 200
    assert response.content_type == "application/json"
    body = await response.json()
    assert body["algos"][0]["name"] == "btc"
    assert body["portfolio"]["equity"] == pytest.approx(1100.0)


@pytest.mark.asyncio
async def test_the_handler_only_queries_the_board_and_never_drives_it(client):
    await client.get("/api/status")

    # Command-query separation: serving a page must not advance the engine.
    assert client.board.calls == 1


# ── / ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_page_is_served_as_html(client):
    response = await client.get("/")

    assert response.status == 200
    assert response.content_type == "text/html"


@pytest.mark.asyncio
async def test_the_page_loads_nothing_from_an_external_host(client):
    # No CDN, no framework: the page must work offline, and no third-party
    # script should sit in the path of one displaying trading state.
    body = await (await client.get("/")).text()

    assert not re.search(r'(src|href)\s*=\s*["\']https?://', body)


def test_the_page_polls_the_status_endpoint():
    assert "/api/status" in DashboardPage().html()
