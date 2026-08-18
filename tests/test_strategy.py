from typing import Optional

import pytest

from coinbase.ga.market_data_processor import IndicatorPeriods
from coinbase.strategy import LiveMarketRow, LiveTradingRun, PaperTradingRun
from coinbase.trading_strategy import Action, Decision, Ledger, Position


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


_NORMALIZED_COLUMNS = ("sma_short", "sma_long", "sma_extra", "rsi", "macd")


# ── LiveMarketRow ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_market_row_latest_has_close_and_normalized_columns():
    adapter = FakeAdapter(candles=_rising_candles(80), accounts=[])
    row = await LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS).latest()
    assert "close" in row
    assert "norm_sma_short" in row
    assert "norm_rsi" in row


# ── LiveTradingRun ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_trading_run_returns_strategys_decision():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[_account("USDC", "1000")])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.BUY])
    run        = LiveTradingRun(adapter, market_row, strategy, quote_currency="USDC")

    decision = await run.on_timer()
    assert decision.action is Action.BUY
    assert decision.size > 0.0


@pytest.mark.asyncio
async def test_live_trading_run_tracks_position_across_ticks():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[_account("USDC", "1000")])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.BUY, Action.SELL, Action.BUY])
    run        = LiveTradingRun(adapter, market_row, strategy, quote_currency="USDC")

    await run.on_timer()
    assert strategy.received_positions[0] is None  # flat on the first tick

    await run.on_timer()
    assert strategy.received_positions[1] is not None  # position tracked after the BUY

    await run.on_timer()
    assert strategy.received_positions[2] is None  # flat again after the SELL


# ── PaperTradingRun ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_paper_trading_run_sizes_against_the_simulated_ledger_not_the_account():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[])  # no account data needed
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.BUY])
    run        = PaperTradingRun(market_row, strategy, Ledger(1000.0))

    decision = await run.on_timer()
    assert decision.action is Action.BUY
    assert decision.size == pytest.approx((1000.0 * 0.10) / 179.0)  # sized off the ledger's 1000.0, not any account


@pytest.mark.asyncio
async def test_paper_trading_run_never_calls_the_adapters_order_endpoints():
    # FakeAdapter deliberately implements no order-placement methods; if
    # PaperTradingRun ever tried to place a real order this would raise
    # AttributeError instead of completing.
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.BUY, Action.SELL])
    run        = PaperTradingRun(market_row, strategy, Ledger(1000.0))

    first  = await run.on_timer()
    second = await run.on_timer()
    assert (first.action, second.action) == (Action.BUY, Action.SELL)


@pytest.mark.asyncio
async def test_paper_trading_run_updates_the_ledger_across_ticks():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.BUY, Action.SELL])
    ledger     = Ledger(1000.0)
    run        = PaperTradingRun(market_row, strategy, ledger)

    await run.on_timer()
    assert ledger.position() is not None

    await run.on_timer()
    assert ledger.position() is None
    assert len(ledger.trades()) == 1  # FakeAdapter returns the same candles every tick, so this
                                       # round-trips at an unchanged price -- zero profit is expected


@pytest.mark.asyncio
async def test_paper_trading_run_exposes_the_last_observed_price():
    adapter    = FakeAdapter(candles=_rising_candles(80), accounts=[])
    market_row = LiveMarketRow(adapter, "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    strategy   = _FixedActionStrategy([Action.HOLD])
    run        = PaperTradingRun(market_row, strategy, Ledger(1000.0))

    await run.on_timer()
    assert run.last_price() == pytest.approx(179.0)  # close of the last of 80 rising candles (100..179)


def test_paper_trading_run_last_price_raises_before_the_first_tick():
    market_row = LiveMarketRow(FakeAdapter([], []), "BTC-USDC", "ONE_HOUR", IndicatorPeriods(), _NORMALIZED_COLUMNS)
    run = PaperTradingRun(market_row, _FixedActionStrategy([]), Ledger(1000.0))
    with pytest.raises(ValueError):
        run.last_price()
