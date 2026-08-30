import pandas as pd
import pytest

import coinbase.ga.strategy_evaluator as strategy_evaluator
from coinbase.ga.ga_engine import Genome
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    AnnualizedYield,
    GaStrategy,
    StrategyConfig,
    StrategyConfigFile,
    StrategyEvaluator,
    ValidatedStrategyConfig,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.trading_strategy import Action, Direction, MarketRows, Position

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Keys are always passed explicitly (never a module default) so these tests
# exercise GaStrategy/StrategyEvaluator's own logic, independent of whatever
# indicator set config.yaml currently configures the GA to weigh.
_KEYS = ("sma_short", "sma_long", "sma_extra", "rsi", "macd")


def _config(**overrides) -> StrategyConfig:
    defaults = dict(
        position_size_pct=0.10,
        buy_threshold=0.6,
        sell_threshold=0.4,
        starting_balance=1000.0,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _row(close: float, sma_short: float) -> dict[str, float]:
    return {
        "close": close,
        "norm_sma_short": sma_short,
        "norm_sma_long": 0.0,
        "norm_sma_extra": 0.0,
        "norm_rsi": 0.0,
        "norm_macd": 0.0,
    }


def _all_weight_on_sma_short() -> Genome:
    return Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})


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
    assert config.unwind_at_entry_price is True  # defaults on when absent from config.yaml


def test_strategy_config_file_reads_unwind_at_entry_price_when_present():
    raw = {
        "strategy": {
            "position_size_pct": 0.2,
            "buy_threshold": 0.7,
            "sell_threshold": 0.3,
            "starting_balance": 500.0,
            "indicators": {},
            "unwind_at_entry_price": False,
        }
    }
    assert StrategyConfigFile(raw).config().unwind_at_entry_price is False


# ── ValidatedStrategyConfig ──────────────────────────────────────────

def test_validated_strategy_config_passes_a_sane_config_through():
    config = _shorting_config()
    assert ValidatedStrategyConfig(config).config() is config


def test_validated_strategy_config_rejects_leverage_above_1x():
    # above 1.0 a liquidated short loses more than the whole account, which
    # drives total return below -100% and makes AnnualizedYield complex
    with pytest.raises(ValueError, match="1x isolated"):
        ValidatedStrategyConfig(_config(position_size_pct=1.5)).config()


def test_validated_strategy_config_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="1x isolated"):
        ValidatedStrategyConfig(_config(position_size_pct=0.0)).config()


def test_validated_strategy_config_allows_exactly_1x():
    ValidatedStrategyConfig(_config(position_size_pct=1.0)).config()


def test_validated_strategy_config_rejects_a_sell_threshold_above_buy():
    with pytest.raises(ValueError, match="sell_threshold"):
        ValidatedStrategyConfig(_config(buy_threshold=0.4, sell_threshold=0.6)).config()


def test_validated_strategy_config_rejects_short_band_overlapping_the_long_band():
    # buy sits at/below short_entry (0.25), so the long branch would win inside
    # the short band; sell stays under buy so only this violation is in play
    with pytest.raises(ValueError, match="short_entry_threshold"):
        ValidatedStrategyConfig(_shorting_config(buy_threshold=0.2, sell_threshold=0.15)).config()


def test_validated_strategy_config_ignores_short_bands_when_shorting_is_off():
    # nonsense short thresholds are harmless while allow_short is False
    ValidatedStrategyConfig(_config(short_entry_threshold=0.9, short_exit_threshold=0.1)).config()


# ── WeightKeysConfig ─────────────────────────────────────────────────

def test_weight_keys_config_reads_section():
    raw = {"strategy": {"weight_keys": ["sma_short", "rsi"]}}
    assert WeightKeysConfig(raw).keys() == ("sma_short", "rsi")


# ── ValidatedWeightKeys ──────────────────────────────────────────────

def test_validated_weight_keys_passes_through_a_subset():
    keys = ValidatedWeightKeys(("sma_short", "rsi"), ("sma_short", "sma_long", "rsi")).keys()
    assert keys == ("sma_short", "rsi")


def test_validated_weight_keys_rejects_a_key_missing_from_normalized_columns():
    with pytest.raises(ValueError, match="macd"):
        ValidatedWeightKeys(("sma_short", "macd"), ("sma_short",)).keys()


# ── GaStrategy ────────────────────────────────────────────────────────

def test_ga_strategy_buys_when_flat_and_score_above_buy_threshold():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.8), position=None, balance=1000.0)
    assert decision.action is Action.BUY
    assert decision.size == pytest.approx(1.0)  # 10% of 1000 / 100


def test_ga_strategy_holds_when_flat_and_score_between_thresholds():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.5), position=None, balance=1000.0)
    assert decision.action is Action.HOLD


def test_ga_strategy_never_re_buys_while_already_positioned():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.9), position=Position(90.0, 1.0), balance=1000.0)
    assert decision.action is Action.HOLD


def test_ga_strategy_sells_when_positioned_and_score_below_sell_threshold():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.2), position=Position(90.0, 1.0), balance=1000.0)
    assert decision.action is Action.SELL


def test_ga_strategy_holds_position_when_score_between_thresholds():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.5), position=Position(90.0, 1.0), balance=1000.0)
    assert decision.action is Action.HOLD


# ── GaStrategy + position_pnl ─────────────────────────────────────────

def _all_weight_on_position_pnl() -> Genome:
    return Genome({"sma_short": 0.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0, POSITION_PNL_KEY: 1.0})


def test_ga_strategy_position_pnl_is_zero_while_flat():
    keys     = _KEYS + (POSITION_PNL_KEY,)
    strategy = GaStrategy(_all_weight_on_position_pnl(), _config(), keys)
    decision = strategy.decide(_row(close=100.0, sma_short=0.0), position=None, balance=1000.0)
    assert decision.action is Action.HOLD  # score is 0.0, not above buy_threshold


def test_ga_strategy_position_pnl_pulls_score_down_on_a_loss_and_triggers_sell():
    keys     = _KEYS + (POSITION_PNL_KEY,)
    strategy = GaStrategy(_all_weight_on_position_pnl(), _config(), keys)
    position = Position(entry_price=100.0, size=1.0)
    decision = strategy.decide(_row(close=70.0, sma_short=0.0), position=position, balance=1000.0)
    # unrealized_return = (70-100)/100 = -0.3, weight 1.0 -> score -0.3 < sell_threshold 0.4
    assert decision.action is Action.SELL


def test_ga_strategy_position_pnl_holds_a_winning_position():
    keys     = _KEYS + (POSITION_PNL_KEY,)
    strategy = GaStrategy(_all_weight_on_position_pnl(), _config(), keys)
    position = Position(entry_price=100.0, size=1.0)
    decision = strategy.decide(_row(close=150.0, sma_short=0.0), position=position, balance=1000.0)
    # unrealized_return = (150-100)/100 = 0.5, not below sell_threshold 0.4 -> stays in
    assert decision.action is Action.HOLD


def test_ga_strategy_ignores_position_pnl_when_key_not_in_use():
    # confirms no behavior change for callers that don't opt into POSITION_PNL_KEY
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)
    position = Position(entry_price=100.0, size=1.0)
    decision = strategy.decide(_row(close=1.0, sma_short=0.5), position=position, balance=1000.0)
    assert decision.action is Action.HOLD  # a huge unrealized loss on close=1.0 has zero effect


# ── GaStrategy: shorting (three bands) ───────────────────────────────

def _shorting_config(**overrides) -> StrategyConfig:
    return _config(allow_short=True, short_entry_threshold=0.25, short_exit_threshold=0.40, **overrides)


def test_ga_strategy_opens_a_short_when_score_falls_below_short_entry():
    strategy = GaStrategy(_all_weight_on_sma_short(), _shorting_config(), _KEYS)
    decision = strategy.decide(_row(close=100.0, sma_short=0.1), position=None, balance=1000.0)
    assert decision.action is Action.SHORT
    assert decision.size == pytest.approx(1.0)  # 10% of 1000 / 100, same sizing as a long


def test_ga_strategy_stays_flat_in_the_band_between_short_entry_and_buy():
    strategy = GaStrategy(_all_weight_on_sma_short(), _shorting_config(), _KEYS)
    for score in (0.3, 0.5):
        decision = strategy.decide(_row(close=100.0, sma_short=score), position=None, balance=1000.0)
        assert decision.action is Action.HOLD


def test_ga_strategy_never_shorts_when_allow_short_is_off():
    strategy = GaStrategy(_all_weight_on_sma_short(), _config(), _KEYS)  # allow_short defaults False
    decision = strategy.decide(_row(close=100.0, sma_short=0.0), position=None, balance=1000.0)
    assert decision.action is Action.HOLD


def test_ga_strategy_covers_a_short_when_score_rises_above_short_exit():
    strategy = GaStrategy(_all_weight_on_sma_short(), _shorting_config(), _KEYS)
    position = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
    decision = strategy.decide(_row(close=90.0, sma_short=0.5), position=position, balance=1000.0)
    assert decision.action is Action.COVER


def test_ga_strategy_holds_a_short_while_score_stays_below_short_exit():
    strategy = GaStrategy(_all_weight_on_sma_short(), _shorting_config(), _KEYS)
    position = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
    decision = strategy.decide(_row(close=90.0, sma_short=0.2), position=position, balance=1000.0)
    assert decision.action is Action.HOLD


def test_ga_strategy_closes_a_long_rather_than_flipping_straight_to_short():
    # score crosses the whole range in one candle: the long must close first,
    # leaving the short to a later candle from flat.
    strategy = GaStrategy(_all_weight_on_sma_short(), _shorting_config(), _KEYS)
    position = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
    decision = strategy.decide(_row(close=100.0, sma_short=0.0), position=position, balance=1000.0)
    assert decision.action is Action.SELL


def test_ga_strategy_position_pnl_reads_a_winning_short_as_positive():
    keys     = _KEYS + (POSITION_PNL_KEY,)
    strategy = GaStrategy(_all_weight_on_position_pnl(), _shorting_config(), keys)
    position = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
    # price fell 50% -> the short is up 0.5, which is above short_exit 0.40 -> take profit
    decision = strategy.decide(_row(close=50.0, sma_short=0.0), position=position, balance=1000.0)
    assert decision.action is Action.COVER


# ── AnnualizedYield ──────────────────────────────────────────────────

def test_annualized_yield_compounds_a_short_window_up_to_a_year():
    value = AnnualizedYield(gross_profit=100.0, starting_balance=1000.0, duration_seconds=30 * 86400).value()
    assert value == pytest.approx((1.10) ** (SECONDS_PER_YEAR / (30 * 86400)) - 1.0)


def test_annualized_yield_matches_simple_return_over_exactly_one_year():
    value = AnnualizedYield(gross_profit=100.0, starting_balance=1000.0, duration_seconds=SECONDS_PER_YEAR).value()
    assert value == pytest.approx(0.10)


def test_annualized_yield_handles_losses():
    value = AnnualizedYield(gross_profit=-50.0, starting_balance=1000.0, duration_seconds=10 * 86400).value()
    assert value < 0.0


def test_annualized_yield_falls_back_to_simple_return_when_duration_is_zero():
    value = AnnualizedYield(gross_profit=100.0, starting_balance=1000.0, duration_seconds=0.0).value()
    assert value == pytest.approx(0.10)


# ── StrategyEvaluator ────────────────────────────────────────────────

def _frame_with_timestamps(timestamps: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":      timestamps,
        "close":          [100.0, 100.0, 120.0, 120.0],
        "norm_sma_short": [0.0, 1.0, 1.0, 0.0],
        "norm_sma_long":  [0.0, 0.0, 0.0, 0.0],
        "norm_sma_extra": [0.0, 0.0, 0.0, 0.0],
        "norm_rsi":       [0.0, 0.0, 0.0, 0.0],
        "norm_macd":      [0.0, 0.0, 0.0, 0.0],
    })


def test_strategy_evaluator_fitness_is_the_annualized_yield_of_the_backtest():
    frame     = _frame_with_timestamps([0, 864000, 1728000, 2592000])  # 30-day window
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config(), _KEYS)
    result    = evaluator.result(genome)

    assert result.gross_profit() == pytest.approx(20.0)  # 10% of 1000 @100 -> 1 unit, +20 on the move
    assert evaluator.fitness(genome) == pytest.approx(evaluator.annualized_yield(result))
    assert evaluator.fitness(genome) == pytest.approx((1.02) ** (SECONDS_PER_YEAR / 2592000) - 1.0)


def _frame_that_buys_and_never_sells() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":      [0, 864000],
        "close":          [100.0, 150.0],
        "norm_sma_short": [1.0, 1.0],
        "norm_sma_long":  [0.0, 0.0],
        "norm_sma_extra": [0.0, 0.0],
        "norm_rsi":       [0.0, 0.0],
        "norm_macd":      [0.0, 0.0],
    })


def test_strategy_evaluator_result_unwinds_a_still_open_position_at_entry_price_by_default():
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(_frame_that_buys_and_never_sells(), _config(), _KEYS)  # default: unwind_at_entry_price=True
    result    = evaluator.result(genome)
    assert result.gross_profit() == pytest.approx(0.0)  # still "in flight", not realized at the 150.0 close


def test_strategy_evaluator_result_realizes_at_market_price_when_unwind_at_entry_price_is_false():
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(
        _frame_that_buys_and_never_sells(), _config(unwind_at_entry_price=False), _KEYS,
    )
    result = evaluator.result(genome)
    assert result.gross_profit() == pytest.approx(50.0)  # 10% of 1000 @100 -> 1 unit, +50 on the move


def _frame_that_never_trades() -> pd.DataFrame:
    # score sits in the hold band on every row, so no position is ever opened
    return pd.DataFrame({
        "timestamp":      [0, 864000],
        "close":          [100.0, 150.0],
        "norm_sma_short": [0.5, 0.5],
        "norm_sma_long":  [0.0, 0.0],
        "norm_sma_extra": [0.0, 0.0],
        "norm_rsi":       [0.0, 0.0],
        "norm_macd":      [0.0, 0.0],
    })


def test_fitness_ranks_a_no_trade_genome_below_any_losing_strategy():
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(_frame_that_never_trades(), _config(), _KEYS)
    result    = evaluator.result(genome)

    assert result.trades() == []
    # a strategy can lose at most its whole balance, so -1.0 is the worst real yield
    assert evaluator.fitness(genome) < -1.0


def test_no_trade_run_still_reports_an_honest_zero_yield():
    # fitness is a selection score; the performance report must not inherit its penalty
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(_frame_that_never_trades(), _config(), _KEYS)
    assert evaluator.annualized_yield(evaluator.result(genome)) == pytest.approx(0.0)


def test_fitness_still_equals_annualized_yield_when_the_genome_trades():
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(_frame_that_buys_and_never_sells(), _config(), _KEYS)
    result    = evaluator.result(genome)
    assert result.trades() != []
    assert evaluator.fitness(genome) == pytest.approx(evaluator.annualized_yield(result))


def test_strategy_evaluator_duration_falls_back_to_zero_for_a_single_row_frame():
    frame     = _frame_with_timestamps([0, 864000, 1728000, 2592000]).iloc[:1]
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config(), _KEYS)
    result    = evaluator.result(genome)
    assert evaluator.annualized_yield(result) == pytest.approx(result.gross_profit() / _config().starting_balance)


def test_strategy_evaluator_converts_the_window_to_rows_once_for_every_genome(monkeypatch):
    # The GA scores population x generations genomes against ONE window, and
    # Backtest is rebuilt per genome — so the row conversion has to be held by
    # the evaluator or it re-runs thousands of times per training run.
    seen: list[MarketRows] = []
    real_backtest = strategy_evaluator.Backtest

    class RecordingBacktest(real_backtest):
        def __init__(self, rows: MarketRows, *args: object, **kwargs: object) -> None:
            seen.append(rows)
            super().__init__(rows, *args, **kwargs)

    monkeypatch.setattr(strategy_evaluator, "Backtest", RecordingBacktest)

    evaluator = StrategyEvaluator(_frame_with_timestamps([0, 864000, 1728000, 2592000]), _config(), _KEYS)
    for weight in (1.0, 0.5, 0.25):
        evaluator.fitness(Genome(
            {"sma_short": weight, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0},
        ))

    assert len(seen) == 3
    assert all(rows is seen[0] for rows in seen)                  # one MarketRows for every genome
    assert all(rows.records is seen[0].records for rows in seen)  # converted exactly once
