from typing import Optional

import pandas as pd
import pytest

from coinbase.trading_strategy import (
    Action,
    Backtest,
    Decision,
    Direction,
    IsolatedMargin,
    Ledger,
    Position,
    Trade,
)


# ── Test double ──────────────────────────────────────────────────────
# Mirrors GaStrategy's buy-above/sell-below/hold-between logic, but driven by
# a scripted list of scores rather than computed weights — lets Backtest's
# engine behavior be tested independent of any specific Strategy.

class _ScriptedStrategy:
    def __init__(self, scores: list[float], buy_threshold: float, sell_threshold: float, position_size_pct: float) -> None:
        self._scores            = scores
        self._buy_threshold      = buy_threshold
        self._sell_threshold     = sell_threshold
        self._position_size_pct = position_size_pct
        self._calls             = 0

    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        score = self._scores[self._calls]
        self._calls += 1
        if position is None and score > self._buy_threshold:
            return Decision(Action.BUY, (balance * self._position_size_pct) / row["close"])
        if position is not None and score < self._sell_threshold:
            return Decision(Action.SELL)
        return Decision(Action.HOLD)


def _strategy(scores: list[float]) -> _ScriptedStrategy:
    return _ScriptedStrategy(scores, buy_threshold=0.6, sell_threshold=0.4, position_size_pct=0.10)


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


# ── Trade / Position ─────────────────────────────────────────────────

def test_trade_profit_is_gross_no_fees():
    trade = Trade(entry_price=100.0, exit_price=110.0, size=2.0)
    assert trade.profit() == pytest.approx(20.0)


def test_position_exposes_entry_price_and_size():
    position = Position(entry_price=50.0, size=4.0)
    assert position.entry_price() == pytest.approx(50.0)
    assert position.size() == pytest.approx(4.0)


def test_position_closed_and_unrealized_agree_at_same_price():
    position = Position(entry_price=50.0, size=4.0)
    assert position.unrealized(60.0) == pytest.approx(position.closed(60.0).profit())


def test_position_defaults_to_long():
    assert Position(entry_price=50.0, size=1.0).direction() is Direction.LONG


# ── Short positions ──────────────────────────────────────────────────

def test_short_trade_profits_when_price_falls():
    trade = Trade(entry_price=100.0, exit_price=80.0, size=2.0, direction=Direction.SHORT)
    assert trade.profit() == pytest.approx(40.0)


def test_short_trade_loses_when_price_rises():
    trade = Trade(entry_price=100.0, exit_price=120.0, size=2.0, direction=Direction.SHORT)
    assert trade.profit() == pytest.approx(-40.0)


def test_short_position_unrealized_mirrors_the_long_case():
    short = Position(entry_price=100.0, size=2.0, direction=Direction.SHORT)
    long_ = Position(entry_price=100.0, size=2.0, direction=Direction.LONG)
    assert short.unrealized(80.0) == pytest.approx(-long_.unrealized(80.0))


def test_unrealized_return_is_positive_for_a_winning_position_either_way():
    short = Position(entry_price=100.0, size=2.0, direction=Direction.SHORT)
    long_ = Position(entry_price=100.0, size=2.0, direction=Direction.LONG)
    assert short.unrealized_return(80.0) == pytest.approx(0.20)   # price fell 20% -> short is up
    assert long_.unrealized_return(120.0) == pytest.approx(0.20)  # price rose 20% -> long is up


# ── IsolatedMargin (1x) ──────────────────────────────────────────────

def test_short_liquidates_at_double_the_entry_price():
    margin = IsolatedMargin(Position(entry_price=100.0, size=1.0, direction=Direction.SHORT))
    assert margin.liquidation_price() == pytest.approx(200.0)
    assert margin.breached_by(high=200.0, low=90.0) is True
    assert margin.breached_by(high=199.9, low=90.0) is False


def test_long_is_never_liquidated_at_1x():
    margin = IsolatedMargin(Position(entry_price=100.0, size=1.0, direction=Direction.LONG))
    assert margin.liquidation_price() == pytest.approx(0.0)
    assert margin.breached_by(high=100.0, low=0.01) is False


def test_ledger_liquidates_a_short_at_its_liquidation_price_not_the_candle_high():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SHORT, size=2.0), price=100.0)
    ledger.liquidate(high=250.0, low=95.0)  # wick far beyond the 200.0 liquidation
    assert ledger.position() is None
    assert ledger.trades()[0].profit() == pytest.approx(-200.0)  # (100-200)*2, not (100-250)*2
    assert ledger.balance() == pytest.approx(800.0)


def test_ledger_liquidate_leaves_an_unbreached_position_open():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SHORT, size=2.0), price=100.0)
    ledger.liquidate(high=150.0, low=90.0)
    assert ledger.position() is not None
    assert ledger.trades() == []


# ── Ledger: short lifecycle ──────────────────────────────────────────

def test_ledger_short_then_cover_realizes_profit_on_a_fall():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SHORT, size=2.0), price=100.0)
    assert ledger.position().direction() is Direction.SHORT
    assert ledger.equity(90.0) == pytest.approx(1020.0)  # unrealized gain as price falls
    ledger.apply(Decision(Action.COVER), price=90.0)
    assert ledger.position() is None
    assert ledger.balance() == pytest.approx(1020.0)


def test_ledger_sell_does_not_close_a_short_and_cover_does_not_close_a_long():
    short = Ledger(1000.0)
    short.apply(Decision(Action.SHORT, size=1.0), price=100.0)
    short.apply(Decision(Action.SELL), price=90.0)
    assert short.position() is not None  # SELL only closes longs

    long_ = Ledger(1000.0)
    long_.apply(Decision(Action.BUY, size=1.0), price=100.0)
    long_.apply(Decision(Action.COVER), price=110.0)
    assert long_.position() is not None  # COVER only closes shorts


def test_ledger_short_while_already_positioned_is_a_no_op():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.BUY, size=1.0), price=100.0)
    ledger.apply(Decision(Action.SHORT, size=5.0), price=100.0)
    assert ledger.position().direction() is Direction.LONG
    assert ledger.position().size() == pytest.approx(1.0)


# ── Decision ─────────────────────────────────────────────────────────

def test_decision_size_defaults_to_zero():
    assert Decision(Action.HOLD).size == pytest.approx(0.0)


# ── Ledger ───────────────────────────────────────────────────────────

def test_ledger_starts_flat_at_the_starting_balance():
    ledger = Ledger(1000.0)
    assert ledger.balance() == pytest.approx(1000.0)
    assert ledger.position() is None
    assert ledger.equity(100.0) == pytest.approx(1000.0)


def test_ledger_apply_buy_opens_a_position_without_touching_balance():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.BUY, size=2.0), price=100.0)
    assert ledger.balance() == pytest.approx(1000.0)
    assert ledger.position().entry_price() == pytest.approx(100.0)
    assert ledger.equity(110.0) == pytest.approx(1020.0)  # unrealized gain only


def test_ledger_apply_sell_closes_the_position_and_realizes_profit():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.BUY, size=2.0), price=100.0)
    ledger.apply(Decision(Action.SELL), price=110.0)
    assert ledger.position() is None
    assert ledger.balance() == pytest.approx(1020.0)
    assert len(ledger.trades()) == 1


def test_ledger_apply_buy_while_already_in_a_position_is_a_no_op():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.BUY, size=2.0), price=100.0)
    ledger.apply(Decision(Action.BUY, size=5.0), price=105.0)
    assert ledger.position().entry_price() == pytest.approx(100.0)  # unchanged


def test_ledger_apply_sell_while_flat_is_a_no_op():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SELL), price=100.0)
    assert ledger.balance() == pytest.approx(1000.0)
    assert ledger.trades() == []


def test_ledger_force_close_realizes_an_open_position():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.BUY, size=2.0), price=100.0)
    ledger.force_close(110.0)
    assert ledger.position() is None
    assert ledger.balance() == pytest.approx(1020.0)


def test_ledger_force_close_while_flat_is_a_no_op():
    ledger = Ledger(1000.0)
    ledger.force_close(100.0)
    assert ledger.balance() == pytest.approx(1000.0)
    assert ledger.trades() == []


# ── Backtest ─────────────────────────────────────────────────────────

def test_backtest_opens_and_closes_a_single_position_on_threshold_crossing():
    frame    = _frame([100.0, 100.0, 120.0, 120.0])
    strategy = _strategy([0.5, 0.7, 0.7, 0.3])  # flat, buy, hold, sell
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((120.0 - 100.0) / 100.0 * 100.0)  # 10% of 1000 @100 -> 1 unit


def test_backtest_never_opens_a_second_overlapping_position():
    frame    = _frame([100.0, 100.0, 100.0, 100.0])
    strategy = _strategy([0.7, 0.7, 0.7, 0.7])  # stays above buy_threshold throughout
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    assert len(result.trades()) == 1  # never re-buys; only trade is the forced close at the end


def test_backtest_unwinds_an_open_position_at_entry_price_by_default():
    frame    = _frame([100.0, 110.0])
    strategy = _strategy([0.7, 0.7])  # buys and never sells
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx(0.0)  # still "in flight", not judged a win or a loss


def test_backtest_force_closes_at_the_final_market_price_when_unwind_at_entry_price_is_false():
    frame    = _frame([100.0, 110.0])
    strategy = _strategy([0.7, 0.7])  # buys and never sells
    result   = Backtest(frame, strategy, starting_balance=1000.0, unwind_at_entry_price=False).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((110.0 - 100.0) / 100.0 * 100.0)


def test_backtest_holds_position_between_thresholds_without_selling():
    frame    = _frame([100.0, 100.0, 100.0])
    strategy = _strategy([0.7, 0.5, 0.5])  # buy, then sits in the hold band
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    assert len(result.trades()) == 1  # only force-closed at the end, not sold mid-run


def test_backtest_position_size_compounds_on_current_balance():
    # trade 1: buy 1000*0.10/100 = 1.0 unit @100, sell @200 -> +100 profit, balance 1000 -> 1100
    # trade 2 must size off the *updated* 1100 balance (0.55 units @200), not the original 1000 (which
    # would size 0.5 units) -> the two hypotheses diverge to 55.0 vs 50.0 profit on the @300 exit
    frame    = _frame([100.0, 200.0, 200.0, 300.0])
    strategy = _strategy([0.7, 0.3, 0.7, 0.3])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 2
    assert trades[0].profit() == pytest.approx(100.0)
    assert trades[1].profit() == pytest.approx(55.0)


def test_backtest_equity_curve_tracks_unrealized_gains_while_in_position():
    frame    = _frame([100.0, 150.0])
    strategy = _strategy([0.7, 0.7])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    curve    = result.equity_curve()
    assert curve[0] == pytest.approx(1000.0)  # just bought, no move yet
    assert curve[1] == pytest.approx(1000.0 + (150.0 - 100.0) / 100.0 * 100.0)


# Emits a scripted action per candle, sizing opens off the current balance.
class _ScriptedDirectionalStrategy:
    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions
        self._calls   = 0

    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        action = self._actions[self._calls]
        self._calls += 1
        if action in (Action.BUY, Action.SHORT):
            return Decision(action, (balance * 0.10) / row["close"])
        return Decision(action)


def _ohlc_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows]}
    )


def test_backtest_short_profits_as_price_falls():
    frame    = _ohlc_frame([(100.0, 100.0, 100.0), (80.0, 100.0, 80.0), (80.0, 80.0, 80.0)])
    strategy = _ScriptedDirectionalStrategy([Action.SHORT, Action.HOLD, Action.COVER])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].direction() is Direction.SHORT
    assert trades[0].profit() == pytest.approx(20.0)  # 1 unit shorted @100, covered @80


def test_backtest_liquidates_a_short_before_taking_that_candles_decision():
    # the position is opened on candle 0; candle 1 wicks above 2x entry and must
    # liquidate it, so the scripted COVER on candle 2 finds nothing to close
    frame    = _ohlc_frame([(100.0, 100.0, 100.0), (150.0, 210.0, 140.0), (90.0, 150.0, 90.0)])
    strategy = _ScriptedDirectionalStrategy([Action.SHORT, Action.HOLD, Action.COVER])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx(-100.0)  # 1 unit @100 closed at 200, not at 210 or 90


def test_backtest_does_not_liquidate_a_short_on_the_candle_that_opened_it():
    # candle 0's own high already exceeds 2x entry, but that range happened
    # before the close the position is opened at
    frame    = _ohlc_frame([(100.0, 500.0, 100.0), (95.0, 100.0, 90.0)])
    strategy = _ScriptedDirectionalStrategy([Action.SHORT, Action.HOLD])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    assert result.trades()[0].profit() == pytest.approx(0.0)  # unwound at entry, never liquidated


def test_backtest_empty_when_signal_never_crosses_buy_threshold():
    frame    = _frame([100.0, 100.0, 100.0])
    strategy = _strategy([0.5, 0.5, 0.5])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    assert result.trades() == []
    assert result.gross_profit() == 0.0
