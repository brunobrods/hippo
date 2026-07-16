import pandas as pd
import pytest

from coinbase.ga.ga_engine import Genome
from coinbase.ga.strategy_evaluator import (
    Backtest,
    OpenPosition,
    SignalScores,
    StrategyConfig,
    StrategyConfigFile,
    StrategyEvaluator,
    Trade,
)


def _config(**overrides) -> StrategyConfig:
    defaults = dict(
        position_size_pct=0.10,
        buy_threshold=0.6,
        sell_threshold=0.4,
        starting_balance=1000.0,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


# ── StrategyConfigFile ───────────────────────────────────────────────

def test_strategy_config_file_reads_section():
    raw = {
        "strategy": {
            "position_size_pct": 0.2,
            "buy_threshold": 0.7,
            "sell_threshold": 0.3,
            "starting_balance": 500.0,
            "indicators": {},
        }
    }
    config = StrategyConfigFile(raw).config()
    assert config.position_size_pct == 0.2
    assert config.buy_threshold == 0.7
    assert config.sell_threshold == 0.3
    assert config.starting_balance == 500.0


# ── SignalScores ─────────────────────────────────────────────────────

def test_signal_scores_computes_weighted_sum():
    frame = pd.DataFrame({
        "norm_sma_short": [1.0, 0.0],
        "norm_sma_long":  [0.0, 1.0],
        "norm_sma_extra": [0.0, 0.0],
        "norm_rsi":       [0.0, 0.0],
        "norm_macd":      [0.0, 0.0],
    })
    genome = Genome({
        "sma_short": 0.5, "sma_long": 0.5, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0,
    })
    scores = SignalScores(frame, genome).series
    assert scores.tolist() == pytest.approx([0.5, 0.5])


# ── Trade / OpenPosition ─────────────────────────────────────────────

def test_trade_profit_is_gross_no_fees():
    trade = Trade(entry_price=100.0, exit_price=110.0, size=2.0)
    assert trade.profit() == pytest.approx(20.0)


def test_open_position_closed_and_unrealized_agree_at_same_price():
    position = OpenPosition(entry_price=50.0, size=4.0)
    assert position.unrealized(60.0) == pytest.approx(position.closed(60.0).profit())


# ── Backtest ─────────────────────────────────────────────────────────

def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def test_backtest_opens_and_closes_a_single_position_on_threshold_crossing():
    frame  = _frame([100.0, 100.0, 120.0, 120.0])
    scores = pd.Series([0.5, 0.7, 0.7, 0.3])  # flat, buy, hold, sell
    result = Backtest(frame, scores, _config()).run()
    trades = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((120.0 - 100.0) / 100.0 * 100.0)  # 10% of 1000 @100 -> 1 unit


def test_backtest_never_opens_a_second_overlapping_position():
    frame  = _frame([100.0, 100.0, 100.0, 100.0])
    scores = pd.Series([0.7, 0.7, 0.7, 0.7])  # stays above buy_threshold throughout
    result = Backtest(frame, scores, _config()).run()
    assert len(result.trades()) == 1  # never re-buys; only trade is the forced close at the end


def test_backtest_force_closes_an_open_position_at_the_final_close():
    frame  = _frame([100.0, 110.0])
    scores = pd.Series([0.7, 0.7])  # buys and never sells
    result = Backtest(frame, scores, _config()).run()
    trades = result.trades()
    assert len(trades) == 1
    assert trades[0].profit() == pytest.approx((110.0 - 100.0) / 100.0 * 100.0)


def test_backtest_holds_position_between_thresholds_without_selling():
    frame  = _frame([100.0, 100.0, 100.0])
    scores = pd.Series([0.7, 0.5, 0.5])  # buy, then sits in the hold band
    result = Backtest(frame, scores, _config()).run()
    assert len(result.trades()) == 1  # only force-closed at the end, not sold mid-run


def test_backtest_position_size_compounds_on_current_balance():
    # trade 1: buy 1000*0.10/100 = 1.0 unit @100, sell @200 -> +100 profit, balance 1000 -> 1100
    # trade 2 must size off the *updated* 1100 balance (0.55 units @200), not the original 1000 (which
    # would size 0.5 units) -> the two hypotheses diverge to 55.0 vs 50.0 profit on the @300 exit
    frame  = _frame([100.0, 200.0, 200.0, 300.0])
    scores = pd.Series([0.7, 0.3, 0.7, 0.3])
    result = Backtest(frame, scores, _config()).run()
    trades = result.trades()
    assert len(trades) == 2
    assert trades[0].profit() == pytest.approx(100.0)
    assert trades[1].profit() == pytest.approx(55.0)


def test_backtest_equity_curve_tracks_unrealized_gains_while_in_position():
    frame  = _frame([100.0, 150.0])
    scores = pd.Series([0.7, 0.7])
    result = Backtest(frame, scores, _config()).run()
    curve = result.equity_curve()
    assert curve[0] == pytest.approx(1000.0)  # just bought, no move yet
    assert curve[1] == pytest.approx(1000.0 + (150.0 - 100.0) / 100.0 * 100.0)


def test_backtest_empty_when_signal_never_crosses_buy_threshold():
    frame  = _frame([100.0, 100.0, 100.0])
    scores = pd.Series([0.5, 0.5, 0.5])
    result = Backtest(frame, scores, _config()).run()
    assert result.trades() == []
    assert result.gross_profit() == 0.0


# ── StrategyEvaluator ────────────────────────────────────────────────

def test_strategy_evaluator_fitness_matches_backtest_gross_profit():
    frame = pd.DataFrame({
        "close":          [100.0, 100.0, 120.0, 120.0],
        "norm_sma_short": [0.0, 1.0, 1.0, 0.0],
        "norm_sma_long":  [0.0, 0.0, 0.0, 0.0],
        "norm_sma_extra": [0.0, 0.0, 0.0, 0.0],
        "norm_rsi":       [0.0, 0.0, 0.0, 0.0],
        "norm_macd":      [0.0, 0.0, 0.0, 0.0],
    })
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config())
    assert evaluator.fitness(genome) == pytest.approx(evaluator.result(genome).gross_profit())
    assert evaluator.fitness(genome) == pytest.approx(20.0)  # 10% of 1000 @100 -> 1 unit, +20 on the move
