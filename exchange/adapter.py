"""
Exchange-neutral contracts shared by every adapter.
----------------------------------------------------

The GA pipeline (candle fetch, indicator frame, backtest, live/paper runs) is
written against these Protocols rather than against a concrete adapter, so the
same trained strategy can be driven by Coinbase or Binance.

Canonical formats — every adapter accepts and returns these, translating to its
own exchange's dialect internally:

    product_id   "BTC-USDC"      base-quote, dash separated
    granularity  "SIX_HOUR"      see market_scanner.GRANULARITY_SECONDS
    start / end  UNIX seconds
    candle       {"start", "open", "high", "low", "close", "volume"} — all str
    account      {"currency", "available_balance": {"value", "currency"}, ...}
                 plus an optional "product_id" when the balance is scoped to a
                 single market, as isolated margin balances are.
    product      {"product_id", "base_currency", "quote_currency", "status",
                  "tradable", "can_long", "can_short", "quote_increment",
                  "base_increment", "base_min_size", "min_market_funds",
                  "volume_24h_quote", "price_change_24h_pct"}
                 Increments stay strings — they are handed back to the venue,
                 which rejects floats. The two derived statistics are floats:
                 nothing sends them anywhere, and "volume_24h_quote" is always
                 denominated in the QUOTE currency, so pairs are comparable.
"""

from typing import Any, Protocol


# ── Errors ─────────────────────────────────────────────────────────────

class ExchangeError(Exception):
    def __init__(self, status: int, raw: Any, message: str = "") -> None:
        msg = message or str(raw)
        super().__init__(f"[HTTP {status}] {msg} | raw={raw}")
        self.status = status
        self.raw    = raw


# ── Contracts ──────────────────────────────────────────────────────────

class CandleSource(Protocol):
    async def get_product_candles(
        self, product_id: str, start: int, end: int, granularity: str,
    ) -> list[dict]: ...

    # Per-request candle ceiling — ChunkedTimeRange splits a window by it.
    # Coinbase caps at 300, Binance at 1000.
    def max_candles_per_request(self) -> int: ...

    # Discriminates otherwise-identical data between venues — the candle cache
    # is keyed by it, so a BTC-USDC window fetched from one exchange is never
    # served to the other.
    def name(self) -> str: ...


class AccountSource(Protocol):
    async def get_accounts(self, limit: int = 50) -> list[dict]: ...


# Split out rather than folded into ExchangeAdapter so a caller that only needs
# the universe — the pair screener — can type against this alone, and its test
# doubles need not grow candle and account methods they never use.
class ProductCatalog(Protocol):
    async def list_products(self) -> list[dict]: ...

    # Maker and taker rates for THIS account, in basis points per side. Read
    # from the venue rather than assumed: both exchanges discount by 30-day
    # volume (and Binance again for paying fees in BNB), so a hardcoded base
    # tier is wrong for anyone who has already traded.
    async def fee_rates(self) -> tuple[float, float]: ...


class ExchangeAdapter(CandleSource, AccountSource, ProductCatalog, Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    # Every consumer drives an adapter as `async with ... as adapter`, so the
    # contract has to include the dunders, not just connect/close.
    async def __aenter__(self) -> "ExchangeAdapter": ...

    async def __aexit__(self, *args: Any) -> None: ...

    async def get_product(self, product_id: str) -> dict: ...

    async def get_best_bid_ask(self, *product_ids: str) -> dict: ...
