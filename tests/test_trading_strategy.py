from typing import Optional

import pandas as pd
import math

import pytest

from coinbase.trading_strategy import (
    Action,
    Backtest,
    BasisPointFee,
    BorrowInterest,
    ConfiguredBorrowRate,
    ConfiguredFees,
    Decision,
    Direction,
    HourlyBasisPointRate,
    IsolatedMargin,
    Ledger,
    MarketRows,
    NoBorrowRate,
    NoFees,
    Position,
    TakeProfit,
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


# ── IsolatedMargin ───────────────────────────────────────────────────

def test_short_liquidation_is_collateral_plus_proceeds_over_the_margin_level():
    # 100 collateral, 1 unit shorted at 100 -> (100 + 100) / (1.05 * 1)
    margin = IsolatedMargin(
        Position(entry_price=100.0, size=1.0, direction=Direction.SHORT), collateral=100.0,
    )
    assert margin.liquidation_price() == pytest.approx(190.476190, rel=1e-6)
    assert margin.breached_by(high=191.0, low=90.0) is True
    assert margin.breached_by(high=190.0, low=90.0) is False


def test_a_full_size_short_liquidates_near_but_below_double_the_entry():
    # The old flat rule said exactly 2x. A short whose notional is the entire
    # wallet is the ONE case it approximated, and even there it was generous:
    # the level is really hit at 2x / 1.05.
    margin = IsolatedMargin(
        Position(entry_price=100.0, size=1.0, direction=Direction.SHORT), collateral=100.0,
    )
    assert margin.liquidation_price() < 200.0
    assert margin.liquidation_price() == pytest.approx(200.0 / 1.05, rel=1e-9)


def test_more_collateral_pushes_the_liquidation_price_further_away():
    def liquidation(collateral: float) -> float:
        return IsolatedMargin(
            Position(entry_price=100.0, size=1.0, direction=Direction.SHORT), collateral,
        ).liquidation_price()

    assert liquidation(100.0) < liquidation(200.0) < liquidation(1000.0)


def test_a_sixty_percent_sized_short_liquidates_around_two_and_a_half_times_entry():
    # position_size_pct 0.60 against a 1000 balance: 6 units at 100 = 600 notional.
    margin = IsolatedMargin(
        Position(entry_price=100.0, size=6.0, direction=Direction.SHORT), collateral=1000.0,
    )
    # (1000 + 600) / (1.05 * 6) = 253.97 -> the flat 2x rule fired at 200.0
    assert margin.liquidation_price() == pytest.approx(253.968254, rel=1e-6)


def test_matches_the_live_binance_liquidation_price():
    # Measured on BTCUSDT: 0.00019022 BTC borrowed at entry 78266.0 against
    # quote assets of 57.92954995 reported liquidatePrice 290022.60.
    size, entry = 0.00019022, 78266.0
    margin = IsolatedMargin(
        Position(entry_price=entry, size=size, direction=Direction.SHORT),
        collateral=57.92954995 - size * entry,   # collateral before the sale proceeds
    )
    assert margin.liquidation_price() == pytest.approx(290022.60, rel=1e-4)


def test_a_zero_size_short_borrows_nothing_and_is_never_liquidated():
    margin = IsolatedMargin(
        Position(entry_price=100.0, size=0.0, direction=Direction.SHORT), collateral=1000.0,
    )
    assert margin.liquidation_price() == math.inf
    assert margin.breached_by(high=1e12, low=0.0) is False


def test_long_is_never_liquidated_because_it_borrows_nothing():
    margin = IsolatedMargin(
        Position(entry_price=100.0, size=1.0, direction=Direction.LONG), collateral=100.0,
    )
    assert margin.liquidation_price() == pytest.approx(0.0)
    assert margin.breached_by(high=100.0, low=0.01) is False


def test_ledger_liquidates_a_short_at_its_liquidation_price_not_the_candle_high():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SHORT, size=2.0), price=100.0)
    # (1000 + 200) / (1.05 * 2) = 571.428...
    ledger.liquidate(high=600.0, low=95.0)
    assert ledger.position() is None
    assert ledger.trades()[0].profit() == pytest.approx((100.0 - 1200.0 / 2.1) * 2.0)
    assert ledger.balance() == pytest.approx(1000.0 + (100.0 - 1200.0 / 2.1) * 2.0)


def test_ledger_liquidate_leaves_an_unbreached_position_open():
    ledger = Ledger(1000.0)
    ledger.apply(Decision(Action.SHORT, size=2.0), price=100.0)
    ledger.liquidate(high=150.0, low=90.0)
    assert ledger.position() is not None
    assert ledger.trades() == []


def test_ledger_liquidation_uses_the_balance_backing_the_position():
    # Same short, different wallets: the thinner one is liquidated by a wick
    # that the richer one survives.
    thin = Ledger(100.0)
    rich = Ledger(10000.0)
    for ledger in (thin, rich):
        ledger.apply(Decision(Action.SHORT, size=1.0), price=100.0)
        ledger.liquidate(high=400.0, low=95.0)

    assert thin.position() is None       # liquidates at (100+100)/1.05 = 190.5
    assert rich.position() is not None   # liquidates at (10000+100)/1.05 = 9619.0


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


# ── MarketRows ───────────────────────────────────────────────────────

def test_market_rows_exposes_the_frame_as_records():
    rows = MarketRows(_frame([100.0, 120.0]))
    assert [row["close"] for row in rows.records] == [100.0, 120.0]


def test_market_rows_converts_only_once():
    rows  = MarketRows(_frame([100.0, 120.0]))
    first = rows.records
    assert rows.records is first


def test_market_rows_of_an_empty_frame_has_no_records():
    assert MarketRows(_frame([])).records == []


# ── Backtest ─────────────────────────────────────────────────────────

def test_backtest_opens_and_closes_a_single_position_on_threshold_crossing():
    frame    = _frame([100.0, 100.0, 120.0, 120.0])
    strategy = _strategy([0.5, 0.7, 0.7, 0.3])  # flat, buy, hold, sell
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((120.0 - 100.0) / 100.0 * 100.0)  # 10% of 1000 @100 -> 1 unit


def test_backtest_never_opens_a_second_overlapping_position():
    frame    = _frame([100.0, 100.0, 100.0, 100.0])
    strategy = _strategy([0.7, 0.7, 0.7, 0.7])  # stays above buy_threshold throughout
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    assert len(result.trades()) == 1  # never re-buys; only trade is the forced close at the end


def test_backtest_unwinds_an_open_position_at_entry_price_by_default():
    frame    = _frame([100.0, 110.0])
    strategy = _strategy([0.7, 0.7])  # buys and never sells
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx(0.0)  # still "in flight", not judged a win or a loss


def test_backtest_force_closes_at_the_final_market_price_when_unwind_at_entry_price_is_false():
    frame    = _frame([100.0, 110.0])
    strategy = _strategy([0.7, 0.7])  # buys and never sells
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0, unwind_at_entry_price=False).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((110.0 - 100.0) / 100.0 * 100.0)


def test_backtest_holds_position_between_thresholds_without_selling():
    frame    = _frame([100.0, 100.0, 100.0])
    strategy = _strategy([0.7, 0.5, 0.5])  # buy, then sits in the hold band
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    assert len(result.trades()) == 1  # only force-closed at the end, not sold mid-run


def test_backtest_position_size_compounds_on_current_balance():
    # trade 1: buy 1000*0.10/100 = 1.0 unit @100, sell @200 -> +100 profit, balance 1000 -> 1100
    # trade 2 must size off the *updated* 1100 balance (0.55 units @200), not the original 1000 (which
    # would size 0.5 units) -> the two hypotheses diverge to 55.0 vs 50.0 profit on the @300 exit
    frame    = _frame([100.0, 200.0, 200.0, 300.0])
    strategy = _strategy([0.7, 0.3, 0.7, 0.3])
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 2
    assert trades[0].profit() == pytest.approx(100.0)
    assert trades[1].profit() == pytest.approx(55.0)


def test_backtest_equity_curve_tracks_unrealized_gains_while_in_position():
    frame    = _frame([100.0, 150.0])
    strategy = _strategy([0.7, 0.7])
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
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
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    assert trades[0].direction() is Direction.SHORT
    assert trades[0].profit() == pytest.approx(20.0)  # 1 unit shorted @100, covered @80


def test_backtest_liquidates_a_short_before_taking_that_candles_decision():
    # the position is opened on candle 0; candle 1 wicks past the liquidation
    # price and must close it, so the scripted COVER on candle 2 finds nothing.
    # 1 unit @100 against a 1000 balance liquidates at (1000+100)/1.05 = 1047.6
    frame    = _ohlc_frame([(100.0, 100.0, 100.0), (150.0, 1100.0, 140.0), (90.0, 150.0, 90.0)])
    strategy = _ScriptedDirectionalStrategy([Action.SHORT, Action.HOLD, Action.COVER])
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    trades   = result.trades()
    assert len(trades) == 1
    # closed at the liquidation price, not at the 1100.0 wick or the 90.0 close
    assert trades[0].profit() == pytest.approx(100.0 - 1100.0 / 1.05)


def test_backtest_does_not_liquidate_a_short_on_the_candle_that_opened_it():
    # candle 0's own high already exceeds the liquidation price, but that range
    # happened before the close the position is opened at
    frame    = _ohlc_frame([(100.0, 5000.0, 100.0), (95.0, 100.0, 90.0)])
    strategy = _ScriptedDirectionalStrategy([Action.SHORT, Action.HOLD])
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    assert result.trades()[0].profit() == pytest.approx(0.0)  # unwound at entry, never liquidated


def test_backtest_empty_when_signal_never_crosses_buy_threshold():
    frame    = _frame([100.0, 100.0, 100.0])
    strategy = _strategy([0.5, 0.5, 0.5])
    result   = Backtest(MarketRows(frame), strategy, starting_balance=1000.0).run()
    assert result.trades() == []
    assert result.gross_profit() == 0.0


# ── Costs ────────────────────────────────────────────────────────────

def test_no_fees_charges_nothing():
    assert NoFees().charge(1000.0) == 0.0


def test_a_basis_point_fee_charges_its_share_of_the_notional():
    assert BasisPointFee(10.0).charge(1000.0) == pytest.approx(1.0)


def test_an_hourly_borrow_rate_prorates_by_the_time_held():
    rate = HourlyBasisPointRate(10.0)
    assert rate.interest(1000.0, 3600.0) == pytest.approx(1.0)
    assert rate.interest(1000.0, 1800.0) == pytest.approx(0.5)


def test_a_zero_rate_configures_the_no_op_object():
    # Not an arithmetic no-op: a run without costs takes the same code path it
    # took before costs existed, so its recorded numbers reproduce exactly.
    assert isinstance(ConfiguredFees(0.0).schedule(), NoFees)
    assert isinstance(ConfiguredBorrowRate(0.0).rate(), NoBorrowRate)
    assert isinstance(ConfiguredFees(10.0).schedule(), BasisPointFee)
    assert isinstance(ConfiguredBorrowRate(10.0).rate(), HourlyBasisPointRate)


def test_a_long_accrues_no_borrow_interest():
    # Longs are sent with sideEffectType NO_SIDE_EFFECT — nothing is borrowed,
    # so nothing accrues however long the position is held.
    position = Position(100.0, 2.0, Direction.LONG, entry_timestamp=1000.0)
    assert BorrowInterest(position, 1000.0 + 7200.0, HourlyBasisPointRate(10.0)).amount() == 0.0


def test_a_short_accrues_interest_on_what_it_borrowed():
    position = Position(100.0, 2.0, Direction.SHORT, entry_timestamp=1000.0)
    # 200 borrowed at entry, 10 bps an hour, held two hours.
    assert BorrowInterest(
        position, 1000.0 + 7200.0, HourlyBasisPointRate(10.0),
    ).amount() == pytest.approx(0.4)


def test_a_short_with_no_recorded_entry_time_accrues_nothing():
    # A book restored from a state file written before entry times existed:
    # accruing from the epoch would charge it decades of interest.
    position = Position(100.0, 2.0, Direction.SHORT, entry_timestamp=0.0)
    assert BorrowInterest(
        position, 1_700_000_000.0, HourlyBasisPointRate(10.0),
    ).amount() == 0.0


def test_the_ledger_charges_a_fee_on_both_legs():
    ledger = Ledger(1000.0, fees=BasisPointFee(10.0))
    ledger.apply(Decision(Action.BUY, 1.0), 100.0)
    ledger.apply(Decision(Action.SELL), 110.0)

    trade = ledger.trades()[0]
    assert trade.profit()     == pytest.approx(10.0)    # gross is untouched
    assert trade.fee()        == pytest.approx(0.21)    # 0.10 in, 0.11 out
    assert trade.net_profit() == pytest.approx(9.79)
    assert ledger.balance()   == pytest.approx(1009.79)
    assert ledger.charged()   == pytest.approx(0.21)


def test_the_ledger_charges_a_short_for_the_time_it_was_held():
    ledger = Ledger(1000.0, borrow=HourlyBasisPointRate(10.0))
    ledger.apply(Decision(Action.SHORT, 2.0), 100.0, 1000.0)
    ledger.apply(Decision(Action.COVER), 90.0, 1000.0 + 7200.0)

    trade = ledger.trades()[0]
    assert trade.profit()     == pytest.approx(20.0)
    assert trade.interest()   == pytest.approx(0.4)
    assert trade.net_profit() == pytest.approx(19.6)
    assert ledger.balance()   == pytest.approx(1019.6)


def test_the_ledger_reports_fees_and_interest_apart():
    # A report that merges them cannot answer which cost is eating the book —
    # the split is the whole point of charging the two separately.
    ledger = Ledger(1000.0, fees=BasisPointFee(10.0), borrow=HourlyBasisPointRate(10.0))
    ledger.apply(Decision(Action.SHORT, 2.0), 100.0, 1000.0)
    ledger.apply(Decision(Action.COVER), 90.0, 1000.0 + 7200.0)

    assert ledger.fees_charged()     == pytest.approx(0.38)   # 0.20 in @100, 0.18 out @90
    assert ledger.interest_charged() == pytest.approx(0.4)    # 200 borrowed, 10 bps/h, 2h
    assert ledger.charged()          == pytest.approx(0.78)
    assert ledger.balance()          == pytest.approx(1019.22)


def test_a_liquidated_position_still_pays_its_exit_fee():
    # The exchange does not waive taker on a forced close.
    ledger = Ledger(1000.0, fees=BasisPointFee(10.0))
    ledger.apply(Decision(Action.SHORT, 2.0), 100.0, 1000.0)
    ledger.liquidate(10_000.0, 9_000.0, 1000.0 + 3600.0)

    trade = ledger.trades()[0]
    assert ledger.position() is None
    assert trade.fee() > 0.1                            # the entry leg alone is 0.10
    assert ledger.charged() == pytest.approx(trade.fee())


def test_an_entry_fee_survives_a_book_restored_between_the_two_legs():
    # A scheduled paper run opens on one invocation and closes on another, with
    # the position reloaded from disk in between — the closing Trade still has
    # to report what the whole round trip cost.
    opening = Ledger(1000.0, fees=BasisPointFee(10.0))
    opening.apply(Decision(Action.BUY, 1.0), 100.0, 1000.0)

    closing = Ledger(opening.balance(), opening.position(), BasisPointFee(10.0))
    closing.apply(Decision(Action.SELL), 110.0, 1000.0 + 3600.0)

    trade = closing.trades()[0]
    assert trade.fee()       == pytest.approx(0.21)     # both legs
    assert closing.charged() == pytest.approx(0.11)     # only this one left THIS balance
    assert closing.balance() == pytest.approx(1009.79)


def test_a_backtest_without_costs_reports_net_equal_to_gross():
    # The regression guard for every run already recorded: at the 0.0 defaults
    # the arithmetic is exactly what it was before costs existed.
    rows   = MarketRows(_frame([100.0, 110.0, 100.0]))
    result = Backtest(rows, _strategy([0.7, 0.3, 0.5]), 1000.0).run()

    assert result.gross_profit()  == pytest.approx(10.0)
    assert result.net_profit()    == pytest.approx(result.gross_profit())
    assert result.fees_paid()     == 0.0
    assert result.interest_paid() == 0.0


def test_a_backtest_charges_fees_without_disturbing_gross():
    rows   = MarketRows(_frame([100.0, 110.0, 100.0]))
    result = Backtest(
        rows, _strategy([0.7, 0.3, 0.5]), 1000.0, True, BasisPointFee(10.0),
    ).run()

    assert result.gross_profit() == pytest.approx(10.0)
    assert result.fees_paid()    == pytest.approx(0.21)
    assert result.net_profit()   == pytest.approx(9.79)


def test_the_end_of_window_unwind_pays_its_exit_leg():
    # unwind_at_entry_price prices the forced close at entry, so the position is
    # neither credited nor penalized for where the window cut off — but it is
    # still charged what holding it actually cost.
    rows   = MarketRows(_frame([100.0, 105.0]))
    result = Backtest(
        rows, _strategy([0.7, 0.5]), 1000.0, True, BasisPointFee(10.0),
    ).run()

    trade = result.trades()[0]
    assert trade.profit()      == pytest.approx(0.0)
    assert trade.fee()         == pytest.approx(0.2)    # 100 in, 100 out, 1 unit
    assert result.net_profit() == pytest.approx(-0.2)

# ── TakeProfit ───────────────────────────────────────────────────────
# A post-only limit resting on the opposite side from entry, filled by the
# market reaching it rather than by a decision on some later close.

# Opens one short on the first candle and then never acts again, so the test
# controls the exit entirely through the candle's range.
class _OpensOneShort:
    def __init__(self) -> None:
        self._opened = False

    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        if position is None and not self._opened:
            self._opened = True
            return Decision(Action.SHORT, (balance * 0.10) / row["close"])
        return Decision(Action.HOLD)

def test_a_long_target_sits_above_entry_and_a_short_target_below():
    long_exit  = TakeProfit(Position(100.0, 1.0, Direction.LONG), 0.03)
    short_exit = TakeProfit(Position(100.0, 1.0, Direction.SHORT), 0.03)
    assert long_exit.price() == pytest.approx(103.0)
    assert short_exit.price() == pytest.approx(97.0)


def test_a_long_fills_on_the_candle_high_not_its_close():
    exit_order = TakeProfit(Position(100.0, 1.0, Direction.LONG), 0.03)
    assert exit_order.reached_by(high=104.0, low=99.0) is True     # spiked through
    assert exit_order.reached_by(high=102.0, low=99.0) is False    # never got there


def test_a_short_fills_on_the_candle_low():
    exit_order = TakeProfit(Position(100.0, 1.0, Direction.SHORT), 0.03)
    assert exit_order.reached_by(high=101.0, low=96.0) is True
    assert exit_order.reached_by(high=101.0, low=98.0) is False


def test_a_zero_target_never_fills():
    exit_order = TakeProfit(Position(100.0, 1.0, Direction.LONG), 0.0)
    assert exit_order.reached_by(high=1e9, low=0.0) is False


# The whole point of the feature: the wick is captured, not the close.
def test_a_spike_that_closes_flat_is_still_booked_at_the_target():
    frame = pd.DataFrame({
        "timestamp":      [0, 3600, 7200],
        "close":          [100.0, 100.0, 100.0],   # closes flat throughout
        "high":           [100.0, 108.0, 100.0],   # but candle 1 spiked +8%
        "low":            [100.0, 100.0, 100.0],
        "norm_sma_short": [1.0, 0.5, 0.5],         # buys on candle 0, then holds
    })
    plain = Backtest(MarketRows(frame), _strategy([0.9, 0.5, 0.5]), 1000.0).run()
    resting = Backtest(
        MarketRows(frame), _strategy([0.9, 0.5, 0.5]), 1000.0, take_profit_pct=0.03,
    ).run()
    # size = 1000 * 0.10 / 100 = 1.0, filled at 103.0 rather than the 100.0 close
    assert plain.gross_profit() == pytest.approx(0.0)      # closes flat, books nothing
    assert resting.trades()[0].profit() == pytest.approx(3.0)


# The trap this feature lives or dies on. A candle that reaches BOTH the
# take-profit and the liquidation gives no way to know which came first, and
# OHLC never will. Assuming the favourable one invents money, so the adverse
# one has to win — Backtest runs liquidate() before the resting exit.
def test_a_candle_reaching_both_the_target_and_the_liquidation_takes_the_loss():
    # A short at 100 with 1000 collateral liquidates well above entry; the
    # candle spikes through that AND dips to the take-profit in the same bar.
    frame = pd.DataFrame({
        "timestamp":      [0, 3600],
        "close":          [100.0, 100.0],
        "high":           [100.0, 100000.0],   # blows through the liquidation
        "low":            [100.0, 50.0],       # and also reaches the target
        "norm_sma_short": [0.0, 0.5],          # opens a short on candle 0
    })
    result = Backtest(MarketRows(frame), _OpensOneShort(), 1000.0, take_profit_pct=0.03).run()
    assert len(result.trades()) == 1
    assert result.trades()[0].profit() < 0.0   # the liquidation won the tie, not the target
