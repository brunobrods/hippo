import json

import pytest

from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.ga.strategy_output import (
    GaRunLog,
    MaxDrawdown,
    OutputConfigFile,
    PerformanceReport,
    StrategyJson,
    StrategyJsonFile,
    StrategyMetadata,
    TrainedStrategy,
    TrainingPeriod,
)
from coinbase.trading_strategy import BacktestResult, Trade


def _ga_config() -> GaConfig:
    return GaConfig(
        population_size=20,
        generations=5,
        mutation_rate=0.2,
        crossover_rate=0.7,
        tournament_size=3,
        elitism_count=2,
        mutation_sigma=0.15,
        seed=42,
    )


def _strategy_config() -> StrategyConfig:
    return StrategyConfig(
        position_size_pct=0.10,
        buy_threshold=0.6,
        sell_threshold=0.4,
        starting_balance=1000.0,
    )


# ── OutputConfigFile ─────────────────────────────────────────────────

def test_output_config_file_reads_section():
    raw = {"output": {"strategy_filepath": "./best.json", "log_filepath": "./log.txt"}}
    config = OutputConfigFile(raw).config()
    assert config.strategy_filepath == "./best.json"
    assert config.log_filepath == "./log.txt"


# ── TrainingPeriod ───────────────────────────────────────────────────

def test_training_period_as_string():
    period = TrainingPeriod("2024-01-01", "2026-07-01")
    assert period.as_string() == "2024-01-01 to 2026-07-01"


# ── StrategyMetadata ─────────────────────────────────────────────────

def test_strategy_metadata_as_dict():
    metadata = StrategyMetadata(
        pair="BTC-USDC",
        granularity="ONE_HOUR",
        training_period=TrainingPeriod("2024-01-01", "2026-07-01"),
        ga_config=_ga_config(),
        created_at="2026-07-15T10:30:00+00:00",
    )
    as_dict = metadata.as_dict()
    assert as_dict["pair"] == "BTC-USDC"
    assert as_dict["timeframe"] == "ONE_HOUR"
    assert as_dict["training_period"] == "2024-01-01 to 2026-07-01"
    assert as_dict["created_at"] == "2026-07-15T10:30:00+00:00"
    assert as_dict["ga_config"] == {
        "population_size": 20,
        "generations": 5,
        "mutation_rate": 0.2,
        "crossover_rate": 0.7,
        "tournament_size": 3,
        "elitism_count": 2,
        "mutation_sigma": 0.15,
        "seed": 42,
    }


# ── TrainedStrategy ──────────────────────────────────────────────────

def test_trained_strategy_as_dict():
    genome   = Genome({"sma_short": 0.4, "sma_long": 0.6})
    strategy = TrainedStrategy(genome, _strategy_config())
    as_dict  = strategy.as_dict()
    assert as_dict["weights"] == {"sma_short": 0.4, "sma_long": 0.6}
    assert as_dict["hyperparameters"] == {
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "position_size_pct": 0.10,
    }


# ── MaxDrawdown ──────────────────────────────────────────────────────

def test_max_drawdown_fraction_of_peak():
    curve = [100.0, 200.0, 150.0, 50.0, 300.0]
    assert MaxDrawdown(curve).fraction() == pytest.approx(0.75)  # 200 -> 50


def test_max_drawdown_zero_when_curve_never_drops():
    assert MaxDrawdown([100.0, 110.0, 120.0]).fraction() == pytest.approx(0.0)


def test_max_drawdown_empty_curve_is_zero():
    assert MaxDrawdown([]).fraction() == pytest.approx(0.0)


# ── PerformanceReport ────────────────────────────────────────────────

def test_performance_report_as_dict():
    trades = [Trade(100.0, 110.0, 1.0), Trade(100.0, 90.0, 1.0)]  # +10, -10
    result = BacktestResult(trades, equity_curve=[1000.0, 1010.0, 1000.0])
    report = PerformanceReport(result, annualized_yield=0.0).as_dict()
    assert report["gross_profit"] == pytest.approx(0.0)
    assert report["annualized_yield"] == pytest.approx(0.0)
    assert report["total_trades"] == 2
    assert report["win_rate"] == pytest.approx(0.5)
    assert report["avg_profit_per_trade"] == pytest.approx(0.0)
    assert report["max_drawdown"] == pytest.approx((1010.0 - 1000.0) / 1010.0)


def test_performance_report_handles_no_trades():
    result = BacktestResult([], equity_curve=[1000.0])
    report = PerformanceReport(result, annualized_yield=0.0).as_dict()
    assert report["total_trades"] == 0
    assert report["win_rate"] == 0.0
    assert report["avg_profit_per_trade"] == 0.0


# ── StrategyJson / StrategyJsonFile ──────────────────────────────────

def test_strategy_json_round_trips_through_disk(tmp_path):
    genome   = Genome({"sma_short": 1.0, "rsi": 0.0})
    metadata = StrategyMetadata(
        pair="BTC-USDC",
        granularity="ONE_HOUR",
        training_period=TrainingPeriod("2024-01-01", "2026-07-01"),
        ga_config=_ga_config(),
        created_at="2026-07-15T10:30:00+00:00",
    )
    strategy    = TrainedStrategy(genome, _strategy_config())
    performance = PerformanceReport(BacktestResult([Trade(100.0, 110.0, 1.0)], [1000.0, 1010.0]), annualized_yield=0.15)

    filepath = tmp_path / "best_strategy.json"
    StrategyJson(metadata, strategy, performance).save(str(filepath))

    on_disk = json.loads(filepath.read_text())
    assert on_disk["strategy"]["weights"] == {"sma_short": 1.0, "rsi": 0.0}
    assert on_disk["performance"]["gross_profit"] == pytest.approx(10.0)
    assert on_disk["performance"]["annualized_yield"] == pytest.approx(0.15)

    loaded = StrategyJsonFile(str(filepath))
    assert loaded.weights() == {"sma_short": 1.0, "rsi": 0.0}
    assert loaded.hyperparameters() == {
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "position_size_pct": 0.10,
    }


# ── GaRunLog ─────────────────────────────────────────────────────────

def test_ga_run_log_appends_a_line_per_call(tmp_path):
    filepath = tmp_path / "ga_run_log.txt"
    log = GaRunLog(str(filepath))
    log.append(1, best_fitness=1.5, average_fitness=0.8)
    log.append(2, best_fitness=2.0, average_fitness=1.1)

    lines = filepath.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == "1\t1.500000\t0.800000"
    assert lines[1] == "2\t2.000000\t1.100000"
