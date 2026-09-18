import inspect

import pandas as pd
import pytest

from coinbase.ga import strategy_evaluator
from coinbase.ga.ga_engine import Genome
from coinbase.ga.market_data_processor import (
    IndicatorFrame,
    IndicatorPeriods,
    MarketDataConfig,
)
from coinbase.ga.strategy_evaluator import (
    StrategyConfig,
    StrategyConfigFile,
    StrategyEvaluator,
    ValidatedStrategyConfig,
)
from coinbase.trading_strategy import (
    Action,
    AtrDistance,
    Backtest,
    Decision,
    FixedDistance,
    MarketRows,
)


# Buys the first candle and then holds forever, so the exit is decided entirely
# by the resting order and the candle ranges the test supplies.
class _BuysOnceThenHolds:
    def __init__(self, size: float = 1.0) -> None:
        self._size = size

    def decide(self, row: dict[str, float], position, balance: float) -> Decision:
        if position is None:
            return Decision(Action.BUY, self._size)
        return Decision(Action.HOLD)


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["close", "high", "low", "atr_pct"],
    ).assign(timestamp=lambda f: range(0, 3600 * len(f), 3600))


# ── The target reads the entry candle's volatility ───────────────────

def test_the_target_rests_at_a_multiple_of_the_entry_candles_atr():
    # atr_pct 2% on the entry candle, multiple 3 -> a target 6% above 100.
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 106.0, 100.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(6.0)   # filled at 106, not the 100 close


def test_a_candle_short_of_the_target_leaves_the_position_open():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 105.9, 100.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(3.0),
    ).run()

    # Only the end-of-window unwind closed it, at its own entry price.
    assert result.trades()[0].profit() == pytest.approx(0.0)


# The order rests where it was placed. Recomputing it per candle would let a
# target retreat from a price the market had already reached — a cancel and
# replace, which nothing here models.
def test_the_target_does_not_move_when_volatility_does():
    # Entry at 2% (target 106), then volatility doubles. A target that tracked
    # the current candle would sit at 112 and never fill on this high.
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 104.0, 100.0, 0.04),
        (100.0, 107.0, 100.0, 0.04),
    ])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(6.0)


# The entry candle's own range happened before the order was placed, so it
# cannot fill against it — the same rule the fixed target already followed.
def test_the_entry_candle_cannot_fill_the_order_it_placed():
    frame  = _frame([(100.0, 200.0, 100.0, 0.02), (100.0, 100.0, 100.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(0.0)


def test_a_zero_multiple_rests_no_order_at_all():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 500.0, 100.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(0.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(0.0)


# A frame without the column would otherwise price every target at zero, which
# reads as "no take-profit" — a configured knob silently doing nothing.
# The one case that would break if the target were ever hoisted out of the loop
# or cached on the strategy: each position reads its OWN entry candle.
def test_a_second_position_gets_its_own_target():
    # Entry at 2% ATR -> target 106, filled on candle 1. The book re-enters on
    # that same candle's close, where ATR is 5% -> target 115, which candle 3's
    # high of 120 reaches. A stale 6% target would have booked 6.0 twice.
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 106.0, 100.0, 0.05),
        (100.0, 100.0, 100.0, 0.05),
        (100.0, 120.0, 100.0, 0.05),
    ])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, take_profit=AtrDistance(3.0),
    ).run()

    # The third trade is the re-entry the window cut short, unwound at its own
    # entry price; the two that the resting order closed are what this asserts.
    profits = [trade.profit() for trade in result.trades()]
    assert profits[:2] == [pytest.approx(6.0), pytest.approx(15.0)]


def test_a_frame_without_atr_pct_raises_rather_than_resting_nothing():
    with pytest.raises(ValueError, match="atr_pct"):
        AtrDistance(3.0).fraction({"close": 100.0})


# NaN compares False against every threshold, so it would rest an order that
# can never fill — the same silence a missing column is raised for.
def test_a_nan_atr_raises_rather_than_resting_nothing():
    with pytest.raises(ValueError, match="NaN"):
        AtrDistance(3.0).fraction({"atr_pct": float("nan")})


def test_a_fixed_target_ignores_the_row_entirely():
    assert FixedDistance(0.05).fraction({"atr_pct": 0.99}) == pytest.approx(0.05)


# ── The column ───────────────────────────────────────────────────────

def _candles(count: int) -> list[dict[str, float]]:
    return [
        {
            "start":  i * 3600,
            "close":  100.0 + i,
            "high":   101.0 + i,
            "low":     99.0 + i,
            "volume": 10.0,
        }
        for i in range(count)
    ]


def test_the_indicator_frame_carries_atr_pct():
    frame = IndicatorFrame(_candles(120), IndicatorPeriods()).dataframe
    assert "atr_pct" in frame.columns
    assert (frame["atr_pct"] > 0.0).all()


# The column is free only if it costs no rows. ATR's warm-up is 14 against
# sma_extra_period's 50, so the frame is exactly as long as it was before —
# every recorded run's window is unchanged.
def test_adding_atr_costs_no_rows():
    candles = _candles(120)
    default = IndicatorFrame(candles, IndicatorPeriods()).dataframe
    assert len(default) == len(candles) - (IndicatorPeriods().sma_extra_period - 1)


# Directly: the frame with the column is exactly as long as one whose ATR
# warm-up is a single candle, so the column itself drops nothing.
def test_the_column_drops_no_rows_of_its_own():
    candles = _candles(120)
    assert (
        len(IndicatorFrame(candles, IndicatorPeriods()).dataframe)
        == len(IndicatorFrame(candles, IndicatorPeriods(atr_period=1)).dataframe)
    )


# Which only holds while the ATR warm-up hides behind a scored one. Longer, and
# the window would silently shorten — so the config refuses it.
def test_an_atr_period_longer_than_every_scored_indicator_is_rejected():
    raw = {"strategy": {"indicators": {
        "sma_short_period": 9, "sma_long_period": 20, "sma_extra_period": 50,
        "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "atr_period": 120,
    }}}
    with pytest.raises(ValueError, match="atr_period"):
        MarketDataConfig(raw).periods()


def test_an_atr_period_inside_the_existing_warm_up_is_accepted():
    raw = {"strategy": {"indicators": {
        "sma_short_period": 9, "sma_long_period": 20, "sma_extra_period": 50,
        "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "atr_period": 50,
    }}}
    assert MarketDataConfig(raw).periods().atr_period == 50


# ── Config ───────────────────────────────────────────────────────────

def _section(**overrides: object) -> dict[str, object]:
    section = {
        "position_size_pct": 0.6,
        "buy_threshold":     0.6,
        "sell_threshold":    0.4,
        "starting_balance":  1000.0,
    }
    section.update(overrides)
    return {"strategy": section}


def test_the_multiple_is_read_from_config():
    config = StrategyConfigFile(_section(take_profit_atr_mult=2.5)).config()
    assert config.take_profit_atr_mult == pytest.approx(2.5)


def test_it_defaults_to_off_so_an_existing_config_is_unchanged():
    assert StrategyConfigFile(_section()).config().take_profit_atr_mult == 0.0


def test_a_negative_multiple_is_rejected():
    with pytest.raises(ValueError, match="take_profit_atr_mult"):
        ValidatedStrategyConfig(
            StrategyConfigFile(_section(take_profit_atr_mult=-1.0)).config(),
        ).config()


# One position rests ONE order. Preferring either silently would score a
# strategy nobody configured.
def test_setting_both_targets_is_rejected():
    with pytest.raises(ValueError, match="rests ONE order"):
        ValidatedStrategyConfig(
            StrategyConfigFile(
                _section(take_profit_pct=0.05, take_profit_atr_mult=3.0),
            ).config(),
        ).config()


# ── Wiring ───────────────────────────────────────────────────────────
# The knob is worth nothing if the evaluator does not hand it to the backtest,
# and a genome scored without its target is not the strategy anyone configured.

_KEYS = ("sma_short", "sma_long", "sma_extra", "rsi", "macd")


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":      [0, 864000, 1728000, 2592000],
        "close":          [100.0, 100.0, 120.0, 120.0],
        "atr_pct":        [0.02, 0.02, 0.02, 0.02],
        "norm_sma_short": [0.0, 1.0, 1.0, 0.0],
        "norm_sma_long":  [0.0, 0.0, 0.0, 0.0],
        "norm_sma_extra": [0.0, 0.0, 0.0, 0.0],
        "norm_rsi":       [0.0, 0.0, 0.0, 0.0],
        "norm_macd":      [0.0, 0.0, 0.0, 0.0],
    })


def _target_seen_by_backtest(monkeypatch, **overrides) -> object:
    return _distances_seen_by_backtest(monkeypatch, **overrides)["take_profit"]


# By NAME, not by position: Backtest now takes two distances, and an argument
# added later would otherwise leave these tests quietly asserting about
# whichever one happened to land last.
def _distances_seen_by_backtest(monkeypatch, **overrides) -> dict[str, object]:
    seen: list[dict[str, object]] = []
    real = strategy_evaluator.Backtest

    class RecordingBacktest(real):
        def __init__(self, *args: object, **kwargs: object) -> None:
            bound = inspect.signature(real.__init__).bind(self, *args, **kwargs)
            bound.apply_defaults()
            seen.append(dict(bound.arguments))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(strategy_evaluator, "Backtest", RecordingBacktest)
    config    = StrategyConfig(
        position_size_pct=0.10, buy_threshold=0.6, sell_threshold=0.4,
        starting_balance=1000.0, **overrides,
    )
    evaluator = StrategyEvaluator(_scored_frame(), config, _KEYS)
    evaluator.fitness(Genome(dict.fromkeys(_KEYS, 0.0) | {"sma_short": 1.0}))
    return seen[0]


def test_the_evaluator_hands_the_backtest_a_volatility_scaled_target(monkeypatch):
    target = _target_seen_by_backtest(monkeypatch, take_profit_atr_mult=3.0)
    assert isinstance(target, AtrDistance)
    assert target.fraction({"atr_pct": 0.02}) == pytest.approx(0.06)


def test_the_evaluator_still_hands_it_a_fixed_target_when_that_is_what_is_set(monkeypatch):
    target = _target_seen_by_backtest(monkeypatch, take_profit_pct=0.05)
    assert isinstance(target, FixedDistance)
    assert target.fraction({}) == pytest.approx(0.05)


def test_a_config_setting_neither_rests_nothing(monkeypatch):
    assert _target_seen_by_backtest(monkeypatch).fraction({}) == 0.0


def test_either_target_alone_is_accepted():
    fixed = ValidatedStrategyConfig(
        StrategyConfigFile(_section(take_profit_pct=0.05)).config(),
    ).config()
    scaled = ValidatedStrategyConfig(
        StrategyConfigFile(_section(take_profit_atr_mult=3.0)).config(),
    ).config()

    assert (fixed.take_profit_pct, fixed.take_profit_atr_mult) == (0.05, 0.0)
    assert (scaled.take_profit_pct, scaled.take_profit_atr_mult) == (0.0, 3.0)


def test_a_negative_fixed_target_is_rejected_too():
    with pytest.raises(ValueError, match="take_profit_pct"):
        ValidatedStrategyConfig(
            StrategyConfigFile(_section(take_profit_pct=-0.05)).config(),
        ).config()
