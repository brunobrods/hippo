"""
Unrealized PnL for open spot positions, expressed in the quote currency (USDC).
--------------------------------------------------------------------------------

An "open position" is any non-quote wallet with a non-zero balance. For each one
we derive an average entry price from the account's own fill history (average-cost
method), mark it against the current best bid, and express the gain/loss in USDC.

Responsibilities are split across small objects:

    Holding        — one wallet balance (currency + size)
    OpenPositions  — pulls all non-quote, non-empty Holdings from the account
    EntryPrice     — average entry price + first-buy time, from a product's fills
    SpotPrice      — current best bid for a product
    PositionPnl    — marks one Holding to market → a PnlLine
    PnlLine        — one position's numbers; knows how to render its own row
    PnlReport      — the collection of lines; renders the table and the total
    PortfolioPnl   — the entry point: account in, PnlReport out

Usage:
    async with CoinbaseAdapter(api_key, api_secret) as adapter:
        report = await PortfolioPnl(adapter).report()
        print(report.as_table())

Cost basis is average-cost: BUY fills add to the running cost and size; SELL fills
reduce both at the current average. The number reported is therefore the unrealized
PnL on the size you still hold, not lifetime realized PnL.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from coinbase.coinbase_adapter import CoinbaseAdapter, CoinbaseError

logger = logging.getLogger(__name__)

QUOTE = "USDC"


# ── Holding ────────────────────────────────────────────────────────────

class Holding:
    """One wallet balance from the accounts endpoint."""

    def __init__(self, account: dict) -> None:
        self._account = account

    def currency(self) -> str:
        return self._account["currency"]

    def size(self) -> float:
        return self._amount("available_balance") + self._amount("hold")

    def is_empty(self) -> bool:
        return self.size() <= 0.0

    def _amount(self, field: str) -> float:
        return float(self._account.get(field, {}).get("value", 0.0) or 0.0)


# ── OpenPositions ──────────────────────────────────────────────────────

class OpenPositions:
    """Every non-quote wallet in the account that still holds something."""

    def __init__(self, adapter: CoinbaseAdapter, quote: str = QUOTE) -> None:
        self._adapter = adapter
        self._quote   = quote

    async def all(self) -> list[Holding]:
        accounts = await self._adapter.get_accounts(limit=250)
        return [
            holding
            for holding in (Holding(a) for a in accounts)
            if holding.currency() != self._quote and not holding.is_empty()
        ]


# ── EntryPrice ─────────────────────────────────────────────────────────

class EntryPrice:
    """Average-cost entry price and first-buy time for one product's fills."""

    def __init__(self, adapter: CoinbaseAdapter, product_id: str) -> None:
        self._adapter    = adapter
        self._product_id = product_id
        self._cache: Optional[list[dict]] = None

    async def average(self) -> float:
        cost, size, _ = await self._basis()
        if size <= 0.0:
            raise CoinbaseError(
                0, {"product_id": self._product_id},
                f"No open buy history for {self._product_id}",
            )
        return cost / size

    async def first_filled_at(self) -> datetime:
        _, _, first = await self._basis()
        if first is None:
            raise CoinbaseError(
                0, {"product_id": self._product_id},
                f"No fills for {self._product_id}",
            )
        return first

    async def _fills(self) -> list[dict]:
        if self._cache is None:
            self._cache = await self._adapter.get_fills(product_id=self._product_id)
        return self._cache

    async def _basis(self) -> tuple[float, float, Optional[datetime]]:
        fills = sorted(await self._fills(), key=lambda f: f["trade_time"])
        cost  = 0.0
        size  = 0.0
        first: Optional[datetime] = None
        for fill in fills:
            price = float(fill["price"])
            base  = self._base_size(fill, price)
            fee   = float(fill.get("commission", 0.0) or 0.0)
            if fill["side"] == "BUY":
                cost += price * base + fee
                size += base
                if first is None:
                    first = self._time(fill)
            elif size > 0.0:
                average = cost / size
                sold    = min(base, size)
                cost   -= average * sold
                size   -= sold
        return cost, size, first

    @staticmethod
    def _base_size(fill: dict, price: float) -> float:
        raw = float(fill["size"])
        if fill.get("size_in_quote") and price > 0.0:
            return raw / price
        return raw

    @staticmethod
    def _time(fill: dict) -> datetime:
        return datetime.fromisoformat(fill["trade_time"].replace("Z", "+00:00"))


# ── SpotPrice ──────────────────────────────────────────────────────────

class SpotPrice:
    """Current best bid — the realistic exit price for a held position."""

    def __init__(self, adapter: CoinbaseAdapter, product_id: str) -> None:
        self._adapter    = adapter
        self._product_id = product_id

    async def bid(self) -> float:
        book = await self._adapter.get_best_bid_ask(self._product_id)
        bids = book["pricebooks"][0]["bids"]
        if not bids:
            raise CoinbaseError(
                0, book, f"No bid available for {self._product_id}"
            )
        return float(bids[0]["price"])


# ── PositionPnl ────────────────────────────────────────────────────────

class PositionPnl:
    """Marks a single Holding to market and returns its PnlLine."""

    def __init__(
        self,
        adapter: CoinbaseAdapter,
        holding: Holding,
        quote: str = QUOTE,
    ) -> None:
        self._adapter = adapter
        self._holding = holding
        self._quote   = quote

    async def line(self) -> "PnlLine":
        product = f"{self._holding.currency()}-{self._quote}"
        entry   = EntryPrice(self._adapter, product)
        spot    = SpotPrice(self._adapter, product)
        return PnlLine(
            currency = self._holding.currency(),
            size     = self._holding.size(),
            entry    = await entry.average(),
            price    = await spot.bid(),
            bought   = await entry.first_filled_at(),
        )


# ── PnlLine ────────────────────────────────────────────────────────────

class PnlLine:
    """One position's numbers, in the quote currency. Renders its own row."""

    def __init__(
        self,
        currency: str,
        size: float = 0.0,
        entry: float = 0.0,
        price: float = 0.0,
        bought: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        self._currency = currency
        self._size     = size
        self._entry    = entry
        self._price    = price
        self._bought   = bought
        self._error    = error

    def pnl(self) -> float:
        return self._value() - self._cost()

    def as_row(self) -> str:
        if self._error:
            return f"{self._currency:<8}  ERROR: {self._error}"
        bought = self._bought.strftime("%Y-%m-%d") if self._bought else "—"
        return (
            f"{self._currency:<8} {self._size:>16.6f} "
            f"{self._entry:>14.6f} {self._price:>14.6f} "
            f"{self.pnl():>+14.2f} {self._pct():>+8.1f}%  {bought}"
        )

    def _cost(self) -> float:
        return self._entry * self._size

    def _value(self) -> float:
        return self._price * self._size

    def _pct(self) -> float:
        cost = self._cost()
        if cost == 0.0:
            return 0.0
        return self.pnl() / cost * 100.0


# ── PnlReport ──────────────────────────────────────────────────────────

class PnlReport:
    """A collection of PnlLines that knows how to total and tabulate itself."""

    def __init__(self, lines: list[PnlLine], quote: str = QUOTE) -> None:
        self._lines = lines
        self._quote = quote

    def total(self) -> float:
        return sum(line.pnl() for line in self._lines)

    def as_table(self) -> str:
        header = (
            f"{'Asset':<8} {'Size':>16} {'Entry':>14} {'Price':>14} "
            f"{f'PnL({self._quote})':>14} {'PnL%':>9}  {'Since':<10}"
        )
        rule  = "─" * len(header)
        rows  = [line.as_row() for line in self._lines] or ["(no open positions)"]
        total = f"{'TOTAL':<8} {'':>16} {'':>14} {'':>14} {self.total():>+14.2f}"
        return "\n".join([header, rule, *rows, rule, total])


# ── PortfolioPnl ───────────────────────────────────────────────────────

class PortfolioPnl:
    """
    Entry point: an authenticated account (CoinbaseAdapter) in, a PnlReport out.

    Pulls every open non-quote position and marks each to market concurrently.
    A position whose price or fill history cannot be resolved becomes an error
    line rather than sinking the whole report.
    """

    def __init__(self, adapter: CoinbaseAdapter, quote: str = QUOTE) -> None:
        self._adapter = adapter
        self._quote   = quote

    async def report(self) -> PnlReport:
        holdings = await OpenPositions(self._adapter, self._quote).all()
        lines    = await asyncio.gather(*(self._line(h) for h in holdings))
        return PnlReport(list(lines), self._quote)

    async def _line(self, holding: Holding) -> PnlLine:
        try:
            return await PositionPnl(self._adapter, holding, self._quote).line()
        except Exception as exc:
            return PnlLine(currency=holding.currency(), error=str(exc))


# ── Smoke test ─────────────────────────────────────────────────────────
# Run:  python coinbase/pnl.py
# Reads live balances and fills — requires a key with trade:read (or read_write).

async def _main() -> None:
    from coinbase.credentials import api_key, api_secret

    async with CoinbaseAdapter(api_key, api_secret) as adapter:
        report = await PortfolioPnl(adapter).report()
        print(report.as_table())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_main())
