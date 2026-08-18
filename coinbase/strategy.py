import asyncio
import time
from typing import Optional

from coinbase.coinbase_adapter import CoinbaseAdapter
from coinbase.ga.market_data_processor import AccountBalance, HistoricalMarketData, IndicatorPeriods
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.trading_strategy import Decision, Ledger, Strategy


# ── Live row ─────────────────────────────────────────────────────────────
# Reuses HistoricalMarketData over a trailing window so the row has the same
# shape (close, norm_sma_short, norm_rsi, ...) a Strategy expects — but the
# min-max normalization is scaled against this trailing window, not the
# original training window's range. A GaStrategy trained elsewhere will see
# a different normalization reference live than it did during backtesting.

class LiveMarketRow:
    def __init__(
        self,
        adapter: CoinbaseAdapter,
        pair: str,
        granularity: str,
        periods: IndicatorPeriods,
        normalized_columns: tuple[str, ...],
        lookback_candles: int = 200,
    ) -> None:
        self._adapter            = adapter
        self._pair               = pair
        self._granularity        = granularity
        self._periods            = periods
        self._normalized_columns = normalized_columns
        self._lookback_candles   = lookback_candles

    async def latest(self) -> dict[str, float]:
        end   = int(time.time())
        start = end - self._lookback_candles * GRANULARITY_SECONDS[self._granularity]
        frame = await HistoricalMarketData(
            self._adapter, self._pair, self._granularity, start, end, self._periods, self._normalized_columns,
        ).dataframe()
        return frame.iloc[-1].to_dict()


# ── Live trading loop ────────────────────────────────────────────────────

class LiveTradingRun:
    def __init__(
        self,
        adapter: CoinbaseAdapter,
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
    # order. Wiring a Decision to actual CoinbaseAdapter order execution is a
    # deliberately separate step (this API has no sandbox to test against).
    async def on_timer(self) -> Decision:
        row, balance = await asyncio.gather(
            self._market_row.latest(),
            AccountBalance(self._adapter, self._quote_currency).available(),
        )
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
        row      = await self._market_row.latest()
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
