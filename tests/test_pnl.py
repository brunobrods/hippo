import pytest

from coinbase.coinbase_adapter import CoinbaseError
from coinbase.pnl import (
    EntryPrice,
    Holding,
    OpenPositions,
    PnlLine,
    PortfolioPnl,
    PositionPnl,
    SpotPrice,
)


# ── Test double ────────────────────────────────────────────────────────

class FakeAdapter:
    """Stands in for CoinbaseAdapter — only the three methods PnL touches."""

    def __init__(
        self,
        accounts: list[dict],
        fills: dict[str, list[dict]],
        bids: dict[str, float],
    ) -> None:
        self._accounts = accounts
        self._fills    = fills
        self._bids     = bids

    async def get_accounts(self, limit: int = 250) -> list[dict]:
        return self._accounts

    async def get_fills(
        self,
        order_id: str = "",
        product_id: str = "",
        limit: int = 250,
        max_pages: int = 20,
    ) -> list[dict]:
        return self._fills.get(product_id, [])

    async def get_best_bid_ask(self, *product_ids: str) -> dict:
        pid = product_ids[0]
        if pid not in self._bids:
            return {"pricebooks": [{"bids": [], "asks": []}]}
        return {"pricebooks": [{"bids": [{"price": str(self._bids[pid])}], "asks": []}]}


def _account(currency: str, available: str, hold: str = "0") -> dict:
    return {
        "currency":          currency,
        "available_balance": {"value": available, "currency": currency},
        "hold":              {"value": hold, "currency": currency},
    }


def _fill(side: str, price: str, size: str, when: str,
          commission: str = "0", size_in_quote: bool = False) -> dict:
    return {
        "side":          side,
        "price":         price,
        "size":          size,
        "commission":    commission,
        "size_in_quote": size_in_quote,
        "trade_time":    when,
    }


# ── Holding ────────────────────────────────────────────────────────────

def test_holding_size_sums_available_and_hold():
    holding = Holding(_account("FET", "100.5", "9.5"))
    assert holding.currency() == "FET"
    assert holding.size() == 110.0
    assert not holding.is_empty()


def test_holding_is_empty_on_zero():
    assert Holding(_account("BTC", "0", "0")).is_empty()


# ── OpenPositions ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_positions_skips_quote_and_empty():
    adapter = FakeAdapter(
        accounts=[
            _account("USDC", "5000"),   # quote — excluded
            _account("FET", "100"),     # kept
            _account("BTC", "0"),       # empty — excluded
        ],
        fills={},
        bids={},
    )
    positions = await OpenPositions(adapter).all()
    assert [p.currency() for p in positions] == ["FET"]


# ── EntryPrice ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_entry_price_average_cost_with_a_sell():
    fills = [
        _fill("BUY",  "1.00", "100", "2026-01-01T00:00:00Z", commission="0.5"),
        _fill("BUY",  "2.00", "100", "2026-02-01T00:00:00Z", commission="1.0"),
        _fill("SELL", "3.00", "50",  "2026-03-01T00:00:00Z"),
    ]
    entry = EntryPrice(FakeAdapter([], {"FET-USDC": fills}, {}), "FET-USDC")
    # cost after buys = 100.5 + 201.0 = 301.5 over 200 units -> avg 1.5075
    # sell 50 at that avg leaves cost 226.125 over 150 -> avg unchanged
    assert await entry.average() == pytest.approx(1.5075)


@pytest.mark.asyncio
async def test_entry_price_first_filled_at_is_earliest_buy():
    fills = [
        _fill("BUY", "2.00", "1", "2026-02-01T00:00:00Z"),
        _fill("BUY", "1.00", "1", "2026-01-01T00:00:00Z"),
    ]
    entry = EntryPrice(FakeAdapter([], {"FET-USDC": fills}, {}), "FET-USDC")
    bought = await entry.first_filled_at()
    assert bought.year == 2026 and bought.month == 1


@pytest.mark.asyncio
async def test_entry_price_handles_quote_denominated_size():
    # market buy: size is $200 of quote at price 2.0 -> 100 base units
    fills = [_fill("BUY", "2.00", "200", "2026-01-01T00:00:00Z", size_in_quote=True)]
    entry = EntryPrice(FakeAdapter([], {"FET-USDC": fills}, {}), "FET-USDC")
    assert await entry.average() == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_entry_price_raises_without_buys():
    entry = EntryPrice(FakeAdapter([], {"FET-USDC": []}, {}), "FET-USDC")
    with pytest.raises(CoinbaseError):
        await entry.average()


# ── SpotPrice ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spot_price_reads_best_bid():
    spot = SpotPrice(FakeAdapter([], {}, {"FET-USDC": 4.0}), "FET-USDC")
    assert await spot.bid() == 4.0


@pytest.mark.asyncio
async def test_spot_price_raises_without_bid():
    spot = SpotPrice(FakeAdapter([], {}, {}), "FET-USDC")
    with pytest.raises(CoinbaseError):
        await spot.bid()


# ── PnlLine ────────────────────────────────────────────────────────────

def test_pnl_line_computes_gain():
    line = PnlLine(currency="FET", size=150.0, entry=1.5075, price=4.0)
    assert line.pnl() == pytest.approx(150.0 * 4.0 - 150.0 * 1.5075)


def test_pnl_line_error_row_carries_message():
    line = PnlLine(currency="FET", error="boom")
    assert line.pnl() == 0.0
    assert "ERROR: boom" in line.as_row()


# ── PositionPnl / PortfolioPnl ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_position_pnl_line_marks_to_market():
    fills = [_fill("BUY", "1.00", "150", "2026-01-01T00:00:00Z")]
    adapter = FakeAdapter([], {"FET-USDC": fills}, {"FET-USDC": 4.0})
    line = await PositionPnl(adapter, Holding(_account("FET", "150"))).line()
    assert line.pnl() == pytest.approx(150.0 * 4.0 - 150.0 * 1.0)


@pytest.mark.asyncio
async def test_portfolio_report_totals_and_isolates_errors():
    adapter = FakeAdapter(
        accounts=[
            _account("USDC", "1000"),               # excluded
            _account("FET", "150"),                 # priced -> +450
            _account("XYZ", "10"),                  # no bid -> error line
        ],
        fills={
            "FET-USDC": [_fill("BUY", "1.00", "150", "2026-01-01T00:00:00Z")],
            "XYZ-USDC": [_fill("BUY", "1.00", "10", "2026-01-01T00:00:00Z")],
        },
        bids={"FET-USDC": 4.0},                     # XYZ-USDC missing on purpose
    )
    report = await PortfolioPnl(adapter).report()
    assert report.total() == pytest.approx(450.0)   # error line contributes 0
    table = report.as_table()
    assert "FET" in table and "ERROR" in table