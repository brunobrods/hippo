from typing import Optional

import pytest

from coinbase.ga.market_data_processor import IndicatorPeriods
from coinbase.strategy import LiveMarketRow, LiveTradingRun
from coinbase.trading_strategy import Action, Decision, Position


# ── Test doubles ─────────────────────────────────────────────────────

class FakeAdapter:
    def __init__(self, candles: list[dict], accounts: list[dict]) -> None:
        self._candles  = candles
        self._accounts = accounts

    async def get_product_candles(self, product_id: str, start: int, end: int, granularity: str) -> list[dict]:
        return self._candles

    async def get_accounts(self, limit: int = 250) -> list[dict]:
        return self._accounts


class _FixedActionStrategy:
    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions
        self._calls   = 0
        self.received_positions: list[Optional[Position]] = []

    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        self.received_positions.append(position)
        action = self._actions[self._calls]
        self._calls += 1
        size = (balance * 0.10) / row["close"] if action is Action.BUY else 0.0
        return Decision(action, size)


def _candle(start: int, close: float) -> dict:
    return {"start": str(start), "close": str(close), "high": str(close), "low": str(close), "volume": "1.0"}


def _rising_candles(n: int, step: int = 3600) -> list[dict]:
    return [_candle(i * step, close=100.0 + i) for i in range(n)]


def _account(currency: str, available: str) -> dict:
    return {"currency": currency, "available_balance": {"value": available, "currency": currency}}


# ── LiveMarketRow ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_market_row_latest_has_close_and_normalized_columns():
    adapter = FakeAdapter(candles=_rising_candles(80), accounts=[])
    row = await LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods()).latest()
    assert "close" in row
    assert "norm_sma_short" in row
    assert "norm_rsi" in row


# ── LiveTradingRun ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_trading_run_returns_strategys_decision():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[_account("USDC", "1000")])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods())
    strategy   = _FixedActionStrategy([Action.BUY])
    run        = LiveTradingRun(adapter, market_row, strategy, quote_currency="USDC")

    decision = await run.on_timer()
    assert decision.action is Action.BUY
    assert decision.size > 0.0


@pytest.mark.asyncio
async def test_live_trading_run_tracks_position_across_ticks():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[_account("USDC", "1000")])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods())
    strategy   = _FixedActionStrategy([Action.BUY, Action.SELL, Action.BUY])
    run        = LiveTradingRun(adapter, market_row, strategy, quote_currency="USDC")

    await run.on_timer()
    assert strategy.received_positions[0] is None  # flat on the first tick

    await run.on_timer()
    assert strategy.received_positions[1] is not None  # position tracked after the BUY

    await run.on_timer()
    assert strategy.received_positions[2] is None  # flat again after the SELL
