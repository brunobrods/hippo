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


class ExchangeAdapter(CandleSource, AccountSource, Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    # Every consumer drives an adapter as `async with ... as adapter`, so the
    # contract has to include the dunders, not just connect/close.
    async def __aenter__(self) -> "ExchangeAdapter": ...

    async def __aexit__(self, *args: Any) -> None: ...

    async def get_product(self, product_id: str) -> dict: ...

    async def get_best_bid_ask(self, *product_ids: str) -> dict: ...
