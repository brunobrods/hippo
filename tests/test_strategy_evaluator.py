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
    StudentT,
    TradeSample,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.trading_strategy import Action, Direction, MarketRows, Position, Trade

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


# fitness is deliberately NOT the annualized yield any more: the reported
# metric stays the realized figure so index.csv remains comparable, while the
# selection score prices in how uncertain the trade sample is.
def test_reported_yield_is_still_the_realized_figure():
    frame     = _frame_with_timestamps([0, 864000, 1728000, 2592000])  # 30-day window
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config(), _KEYS)
    result    = evaluator.result(genome)

    assert result.gross_profit() == pytest.approx(20.0)  # 10% of 1000 @100 -> 1 unit, +20 on the move
    assert evaluator.annualized_yield(result) == pytest.approx((1.02) ** (SECONDS_PER_YEAR / 2592000) - 1.0)


def test_zero_confidence_restores_the_historical_fitness():
    frame     = _frame_with_timestamps([0, 864000, 1728000, 2592000])
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config(fitness_confidence=0.0), _KEYS)
    result    = evaluator.result(genome)

    assert evaluator.fitness(genome) == pytest.approx(evaluator.annualized_yield(result))


# The exploit this exists to close: a single lucky trade was annualized into a
# yearly rate, so "trade once and hope" outranked trading consistently.
def test_one_lucky_trade_earns_no_credit():
    frame     = _frame_with_timestamps([0, 864000, 1728000, 2592000])
    genome    = Genome({"sma_short": 1.0, "sma_long": 0.0, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    evaluator = StrategyEvaluator(frame, _config(), _KEYS)
    result    = evaluator.result(genome)

    assert len(result.trades()) == 1
    assert result.gross_profit() > 0.0            # it made money
    assert evaluator.fitness(genome) == pytest.approx(0.0)   # and is credited none of it


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


# ── Signed weights ───────────────────────────────────────────────────
# The score must stay on the [0, ceiling] scale the thresholds are calibrated
# against, or a genome would be penalised merely for using a negative weight.

def _signed_row(**cols) -> dict:
    base = {f"norm_{k}": 0.0 for k in _KEYS}
    base.update({f"norm_{k}": v for k, v in cols.items()})
    base["close"] = 100.0
    return base


def test_an_all_positive_genome_scores_exactly_as_before():
    # The backward-compatibility guarantee: no offset, no change.
    genome = Genome({"sma_short": 0.5, "sma_long": 0.5, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    score  = GaStrategy(genome, _config(), _KEYS).signal_score(_signed_row(sma_short=1.0, sma_long=0.4), None)
    assert score == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)


def test_a_negative_weight_inverts_a_columns_contribution():
    genome   = Genome({"sma_short": 0.5, "sma_long": -0.5, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    strategy = GaStrategy(genome, _config(), _KEYS)
    # sma_long high should now push the score DOWN relative to sma_long low.
    high = strategy.signal_score(_signed_row(sma_short=1.0, sma_long=1.0), None)
    low  = strategy.signal_score(_signed_row(sma_short=1.0, sma_long=0.0), None)
    assert high < low


# Without the offset a signed genome's whole reachable range slides below the
# buy threshold, and the search gets pushed back to the non-negative corner
# this change exists to escape.
def test_the_score_floor_stays_at_zero_for_a_signed_genome():
    genome   = Genome({"sma_short": 0.5, "sma_long": -0.5, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    strategy = GaStrategy(genome, _config(), _KEYS)
    worst    = strategy.signal_score(_signed_row(sma_short=0.0, sma_long=1.0), None)
    best     = strategy.signal_score(_signed_row(sma_short=1.0, sma_long=0.0), None)
    assert worst == pytest.approx(0.0)
    assert best == pytest.approx(1.0)


def test_a_signed_genome_can_still_reach_the_buy_threshold():
    genome   = Genome({"sma_short": 0.7, "sma_long": -0.3, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    strategy = GaStrategy(genome, _config(buy_threshold=0.6), _KEYS)
    best     = strategy.signal_score(_signed_row(sma_short=1.0, sma_long=0.0), None)
    assert best > 0.6
    assert strategy.decide(_signed_row(sma_short=1.0, sma_long=0.0), None, 1000.0).action is Action.BUY


def test_the_ceiling_counts_absolute_weight_mass():
    genome = Genome({"sma_short": 0.7, "sma_long": -0.3, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    assert GaStrategy(genome, _config(), _KEYS).flat_score_ceiling() == pytest.approx(1.0)


def test_the_ceiling_is_unchanged_for_an_all_positive_genome():
    genome = Genome({"sma_short": 0.4, "sma_long": 0.2, "sma_extra": 0.0, "rsi": 0.0, "macd": 0.0})
    assert GaStrategy(genome, _config(), _KEYS).flat_score_ceiling() == pytest.approx(0.6)


# ── StudentT ─────────────────────────────────────────────────────────

# A fixed z-multiplier refuses to punish a two-trade sample; t does.
def test_a_tiny_sample_gets_a_far_larger_multiplier():
    assert StudentT(1).multiplier() > 6.0
    assert StudentT(30).multiplier() < 1.75


def test_the_multiplier_shrinks_monotonically_toward_the_normal():
    values = [StudentT(df).multiplier() for df in (1, 2, 5, 10, 30, 120, 5000)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(1.645)


def test_an_untabulated_size_takes_the_conservative_neighbour():
    # 17 df sits between the 15 and 20 rows; err toward punishing uncertainty.
    assert StudentT(17).multiplier() == StudentT(20).multiplier()


def test_degenerate_degrees_of_freedom_are_treated_as_one():
    assert StudentT(0).multiplier() == StudentT(1).multiplier()


# ── TradeSample ──────────────────────────────────────────────────────

def _trades(*profits: float) -> list:
    # entry 100, size 1 -> exit price carries the profit
    return [Trade(100.0, 100.0 + p, 1.0) for p in profits]


def test_returns_are_profits_as_a_fraction_of_the_starting_balance():
    sample = TradeSample(_trades(10.0, -20.0), starting_balance=1000.0)
    assert sample.returns == pytest.approx([0.01, -0.02])
    assert sample.mean() == pytest.approx(-0.005)


def test_the_standard_error_falls_as_the_sample_grows():
    few  = TradeSample(_trades(10.0, 30.0), 1000.0).standard_error()
    many = TradeSample(_trades(*([10.0, 30.0] * 10)), 1000.0).standard_error()
    assert many < few


def test_a_single_trade_has_no_measurable_standard_error():
    assert TradeSample(_trades(10.0), 1000.0).standard_error() == 0.0


# The hole a bound on observed variance alone leaves: two trades that happen
# to return the same amount have zero observed variance and would take zero
# penalty, ranking level with a thirty-trade record.
def test_identical_trades_still_carry_uncertainty():
    twins = TradeSample(_trades(300.0, 300.0), 1000.0)
    assert twins.standard_error() == 0.0
    assert twins.effective_standard_error() > 0.0


def test_the_uncertainty_floor_falls_as_the_sample_grows():
    few  = TradeSample(_trades(*([20.0] * 4)), 1000.0)
    many = TradeSample(_trades(*([20.0] * 36)), 1000.0)
    assert many.effective_standard_error() < few.effective_standard_error()


def test_a_genuinely_erratic_sample_keeps_its_observed_error():
    erratic = TradeSample(_trades(200.0, -160.0, 180.0, -140.0), 1000.0)
    assert erratic.effective_standard_error() == pytest.approx(erratic.standard_error())


# A single win proves nothing on the upside; its loss is still real.
def test_a_single_winning_trade_is_credited_nothing():
    assert TradeSample(_trades(50.0), 1000.0).lower_bound(1.0) == pytest.approx(0.0)


def test_a_single_losing_trade_still_counts_against_it():
    assert TradeSample(_trades(-50.0), 1000.0).lower_bound(1.0) == pytest.approx(-0.05)


def test_zero_confidence_is_just_the_realized_mean():
    sample = TradeSample(_trades(10.0, 30.0), 1000.0)
    assert sample.lower_bound(0.0) == pytest.approx(sample.mean())


# The core of the fix: two lucky trades must not outrank many consistent ones.
def test_a_two_trade_windfall_loses_to_a_long_consistent_record():
    windfall   = TradeSample(_trades(300.0, 250.0), 1000.0)
    consistent = TradeSample(_trades(*([12.0, 8.0] * 15)), 1000.0)
    assert windfall.mean() > consistent.mean()                       # bigger per trade
    assert windfall.pessimistic_return(1.0) < consistent.pessimistic_return(1.0)


def test_an_erratic_record_loses_to_a_steady_one_of_the_same_mean():
    steady  = TradeSample(_trades(*([20.0] * 12)), 1000.0)
    erratic = TradeSample(_trades(*([200.0, -160.0] * 6)), 1000.0)
    assert erratic.mean() == pytest.approx(steady.mean())
    assert erratic.pessimistic_return(1.0) < steady.pessimistic_return(1.0)


# Below -1.0 AnnualizedYield raises a negative base to a fractional power and
# returns a complex number, which would poison every GA comparison.
def test_the_pessimistic_return_is_floored_at_total_ruin():
    ruinous = TradeSample(_trades(*([900.0, -900.0] * 4)), 1000.0)
    assert ruinous.pessimistic_return(1.0) == pytest.approx(-1.0)


# The penalty has to bite on small samples without gutting a genuinely long,
# consistent record — 60 steady trades keep ~78% of their realized edge.
def test_a_good_long_record_keeps_most_of_its_edge():
    sample   = TradeSample(_trades(*([25.0, 15.0, 20.0] * 20)), 1000.0)
    realized = sample.mean() * sample.count()
    assert sample.pessimistic_return(1.0) > 0.7 * realized
    assert sample.pessimistic_return(1.0) < realized

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
