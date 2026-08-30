import asyncio
import logging
import time
from typing import Optional

import aiohttp
import pandas as pd

from coinbase.ga.market_data_processor import (
    AccountBalance,
    HistoricalMarketData,
    IndicatorPeriods,
    MarketBasket,
)
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.trading_strategy import Decision, Ledger, Strategy
from exchange.adapter import ExchangeAdapter, ExchangeError

logger = logging.getLogger(__name__)


# ── Live row ─────────────────────────────────────────────────────────────
# Reuses HistoricalMarketData over a trailing window so the row has the same
# shape (close, norm_sma_short, norm_rsi, ...) a Strategy expects — but the
# min-max normalization is scaled against this trailing window, not the
# original training window's range. A GaStrategy trained elsewhere will see
# a different normalization reference live than it did during backtesting.

class LiveMarketRow:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        pair: str,
        granularity: str,
        periods: IndicatorPeriods,
        normalized_columns: tuple[str, ...],
        lookback_candles: int = 200,
        limit: Optional[asyncio.Semaphore] = None,
        index_pairs: tuple[str, ...] = (),
        index_period: int = 30,
    ) -> None:
        self._adapter            = adapter
        self._pair               = pair
        self._granularity        = granularity
        self._periods            = periods
        self._normalized_columns = normalized_columns
        self._lookback_candles   = lookback_candles
        self._limit              = limit
        self._index_pairs        = index_pairs
        self._index_period       = index_period

    def pair(self) -> str:
        return self._pair

    def granularity(self) -> str:
        return self._granularity

    async def frame(self) -> pd.DataFrame:
        end   = int(time.time())
        start = end - self._lookback_candles * GRANULARITY_SECONDS[self._granularity]
        # cache_dir=None: this window slides every call, so disk caching would
        # never hit and would just accumulate one throwaway file per tick.
        # The basket is fetched over the same sliding window for the same
        # reason, and only when a genome was actually trained against one.
        index_returns = None
        if self._index_pairs:
            index_returns = await MarketBasket(
                self._adapter, self._index_pairs, self._granularity, start, end,
                cache_dir=None, limit=self._limit,
            ).returns()
        return await HistoricalMarketData(
            self._adapter, self._pair, self._granularity, start, end,
            self._periods, self._normalized_columns, cache_dir=None,
            limit=self._limit,
            index_returns=index_returns, index_period=self._index_period,
        ).dataframe()

    async def latest(self) -> dict[str, float]:
        return (await self.frame()).iloc[-1].to_dict()


# ── Closed-candle row ────────────────────────────────────────────────────
# LiveMarketRow's last row is whatever the exchange returned most recently,
# which mid-interval is the CURRENT, still-forming candle — its "close" is
# just the price right now, and its high/low are partial. A Backtest only ever
# sees completed candles, so anything meant to reproduce backtest behaviour has
# to drop that row and decide on the last one that actually closed.

class ClosedMarketRow:
    def __init__(self, rows: LiveMarketRow) -> None:
        self._rows = rows

    def pair(self) -> str:
        return self._rows.pair()

    async def latest(self) -> dict[str, float]:
        frame  = await self._rows.frame()
        closed = frame[frame["timestamp"] < self.current_candle_start()]
        if closed.empty:
            raise ValueError(
                f"no completed {self._rows.granularity()} candle for "
                f"{self._rows.pair()} in the fetched window"
            )
        return closed.iloc[-1].to_dict()

    # Start of the candle still forming right now; anything at or after this
    # timestamp is incomplete.
    def current_candle_start(self) -> int:
        seconds = GRANULARITY_SECONDS[self._rows.granularity()]
        return int(time.time()) // seconds * seconds


# ── Retrying row ─────────────────────────────────────────────────────────
# An unattended loop must survive the exchange being briefly unavailable. Only
# transient failures are retried: a 429 or a 5xx, or the connection dropping.
# A 4xx other than 429 is a real rejection, and ClosedMarketRow's ValueError
# ("no completed candle yet") is a real state — retrying either just delays the
# same answer, so both propagate immediately.

class RetriedMarketRow:
    def __init__(self, rows: ClosedMarketRow, attempts: int = 3, base_delay: float = 2.0) -> None:
        self._rows       = rows
        self._attempts   = attempts
        self._base_delay = base_delay

    def pair(self) -> str:
        return self._rows.pair()

    async def latest(self) -> dict[str, float]:
        for attempt in range(self._attempts):
            try:
                return await self._rows.latest()
            except (ExchangeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if not self._transient(exc) or attempt == self._attempts - 1:
                    raise
                delay = self._base_delay * (2 ** attempt)
                logger.warning(
                    "%s fetch failed (%s), retrying in %.1fs", self._rows.pair(), exc, delay,
                )
                await asyncio.sleep(delay)
        # Unreachable: the loop either returns or raises on its last attempt.
        raise ValueError(f"no attempts made for {self._rows.pair()}")

    @staticmethod
    def _transient(exc: Exception) -> bool:
        if isinstance(exc, ExchangeError):
            return exc.status == 429 or exc.status >= 500
        return True


# ── Live trading loop ────────────────────────────────────────────────────

class LiveTradingRun:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        market_row: LiveMarketRow,
        strategy: Strategy,
        quote_currency: str,
    ) -> None:
        self._adapter        = adapter
        self._market_row     = market_row
        self._strategy       = strategy
        self._quote_currency = quote_currency
        # Ledger's balance side goes unused here — only its position tracking is;
        # the real balance is fetched fresh from the account on every tick.
        self._ledger         = Ledger(0.0)

    # Decides what the strategy would do right now — it does not place a real
    # order. Wiring a Decision to actual adapter order execution is a
    # deliberately separate step (neither exchange offers a sandbox for these
    # endpoints, so a mistake here trades real funds).
    async def on_timer(self) -> Decision:
        # Scoped to the traded pair: on Binance isolated margin the quote
        # balance lives in that pair's own wallet, not an account-wide one.
        row, balance = await asyncio.gather(
            self._market_row.latest(),
            AccountBalance(
                self._adapter, self._quote_currency, self._market_row.pair(),
            ).available(),
        )
        # Same order as Backtest.run(): a position carried in from the previous
        # tick is liquidation-checked against this candle's range before a new
        # decision is taken, so a live short obeys the same 1x isolated margin
        # model it was trained and scored under.
        self._ledger.liquidate(row["high"], row["low"])
        decision = self._strategy.decide(row, self._ledger.position(), balance)
        self._ledger.apply(decision, row["close"])
        return decision


# ── Paper trading loop ───────────────────────────────────────────────────
# Same shape as LiveTradingRun, but sizes/fills against a simulated Ledger
# instead of the real account balance, and never touches order placement —
# for watching a trained strategy's live decisions before it risks capital.

class PaperTradingRun:
    def __init__(
        self,
        market_row: LiveMarketRow,
        strategy:   Strategy,
        ledger:     Ledger,
    ) -> None:
        self._market_row = market_row
        self._strategy   = strategy
        self._ledger     = ledger
        self._last_price: Optional[float] = None

    async def on_timer(self) -> Decision:
        row = await self._market_row.latest()
        self._ledger.liquidate(row["high"], row["low"])
        decision = self._strategy.decide(row, self._ledger.position(), self._ledger.balance())
        self._ledger.apply(decision, row["close"])
        self._last_price = row["close"]
        return decision

    def ledger(self) -> Ledger:
        return self._ledger

    def last_price(self) -> float:
        if self._last_price is None:
            raise ValueError("PaperTradingRun has not ticked yet")
        return self._last_price
