from typing import Optional

import pandas as pd
import pytest

from coinbase.trading_strategy import Action, Backtest, Decision, Position, Trade


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


# ── Decision ─────────────────────────────────────────────────────────

def test_decision_size_defaults_to_zero():
    assert Decision(Action.HOLD).size == pytest.approx(0.0)


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


def test_backtest_force_closes_an_open_position_at_the_final_close():
    frame    = _frame([100.0, 110.0])
    strategy = _strategy([0.7, 0.7])  # buys and never sells
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
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


def test_backtest_empty_when_signal_never_crosses_buy_threshold():
    frame    = _frame([100.0, 100.0, 100.0])
    strategy = _strategy([0.5, 0.5, 0.5])
    result   = Backtest(frame, strategy, starting_balance=1000.0).run()
    assert result.trades() == []
    assert result.gross_profit() == 0.0
