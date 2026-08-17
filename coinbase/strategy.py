import asyncio
import time
from typing import Optional

from coinbase.coinbase_adapter import CoinbaseAdapter
from coinbase.ga.market_data_processor import AccountBalance, HistoricalMarketData, IndicatorPeriods
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.trading_strategy import Action, Decision, Position, Strategy


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
        self._position: Optional[Position] = None

    # Decides what the strategy would do right now — it does not place a real
    # order. Wiring a Decision to actual CoinbaseAdapter order execution is a
    # deliberately separate step (this API has no sandbox to test against).
    async def on_timer(self) -> Decision:
        row, balance = await asyncio.gather(
            self._market_row.latest(),
            AccountBalance(self._adapter, self._quote_currency).available(),
        )
        decision = self._strategy.decide(row, self._position, balance)
        self._update_position(decision, row["close"])
        return decision

    def _update_position(self, decision: Decision, price: float) -> None:
        if decision.action is Action.BUY and self._position is None:
            self._position = Position(price, decision.size)
        elif decision.action is Action.SELL and self._position is not None:
            self._position = None
