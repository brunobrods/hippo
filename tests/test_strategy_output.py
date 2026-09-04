import json
import os

import pytest

from coinbase.ga.config import GA_RESULTS_ROOT
from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.ga.strategy_output import (
    DryRunLog,
    FanOutRunLog,
    GaRunLog,
    MaxDrawdown,
    OutputConfigFile,
    ParentDirectory,
    PerformanceReport,
    RunHeader,
    StrategyJson,
    StrategyJsonFile,
    StrategyMetadata,
    TrainedStrategy,
    TrainingPeriod,
)
from coinbase.trading_strategy import Action, BacktestResult, Decision, Trade


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


# ── ParentDirectory ───────────────────────────────────────────────────

def test_parent_directory_creates_missing_nested_directories(tmp_path):
    filepath = tmp_path / "a" / "b" / "c" / "file.txt"
    assert not filepath.parent.exists()
    ParentDirectory(str(filepath)).ensure()
    assert filepath.parent.is_dir()


def test_parent_directory_is_a_no_op_for_a_bare_filename():
    ParentDirectory("file.txt").ensure()  # must not raise


# ── OutputConfigFile ─────────────────────────────────────────────────

def test_output_config_file_reads_section():
    raw = {"output": {
        "strategy_filepath": "./best.json",
        "log_filepath": "./log.txt",
        "dry_run_log_filepath": "./dry_run.txt",
        "experiments_dir": "./experiments",
        "index_filepath": "./experiments/index.csv",
    }}
    config = OutputConfigFile(raw).config()
    assert config.strategy_filepath == "./best.json"
    assert config.log_filepath == "./log.txt"
    assert config.dry_run_log_filepath == "./dry_run.txt"
    assert config.experiments_dir == "./experiments"
    assert config.index_filepath == "./experiments/index.csv"


def test_output_config_file_defaults_dry_run_log_filepath_when_absent():
    raw = {"output": {
        "strategy_filepath": "./best.json",
        "log_filepath": "./log.txt",
        "experiments_dir": "./experiments",
        "index_filepath": "./experiments/index.csv",
    }}
    config = OutputConfigFile(raw).config()
    assert config.dry_run_log_filepath == str(GA_RESULTS_ROOT / "dry_run_log.txt")


def test_output_config_file_defaults_every_path_when_output_section_absent():
    config = OutputConfigFile({}).config()
    assert config.strategy_filepath == str(GA_RESULTS_ROOT / "best_strategy.json")
    assert config.log_filepath == str(GA_RESULTS_ROOT / "ga_run_log.txt")
    assert config.dry_run_log_filepath == str(GA_RESULTS_ROOT / "dry_run_log.txt")
    assert config.experiments_dir == str(GA_RESULTS_ROOT / "experiments")
    assert config.index_filepath == str(GA_RESULTS_ROOT / "experiments" / "index.csv")


def test_output_config_file_defaults_every_path_when_output_section_is_none():
    # a YAML mapping whose every child is commented out parses to None, not {}
    config = OutputConfigFile({"output": None}).config()
    assert config.strategy_filepath == str(GA_RESULTS_ROOT / "best_strategy.json")
    assert config.experiments_dir == str(GA_RESULTS_ROOT / "experiments")


def test_output_config_file_index_filepath_defaults_inside_overridden_experiments_dir():
    raw = {"output": {"experiments_dir": "/custom/experiments"}}
    config = OutputConfigFile(raw).config()
    assert config.index_filepath == os.path.join("/custom/experiments", "index.csv")


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
        "unwind_at_entry_price": True,
        "allow_short": False,
        "short_entry_threshold": 0.25,
        "short_exit_threshold": 0.40,
        "fee_bps": 0.0,
        "borrow_bps_per_hour": 0.0,
        "signal_score_version": 2,
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


def test_performance_report_separates_costs_from_gross():
    trades = [Trade(100.0, 110.0, 1.0, fee=0.2, interest=0.3)]
    result = BacktestResult(trades, equity_curve=[1000.0, 1010.0])
    report = PerformanceReport(result, annualized_yield=0.0).as_dict()
    assert report["gross_profit"]  == pytest.approx(10.0)
    assert report["fees_paid"]     == pytest.approx(0.2)
    assert report["interest_paid"] == pytest.approx(0.3)
    assert report["net_profit"]    == pytest.approx(9.5)


def test_a_trade_that_did_not_pay_for_itself_is_not_a_win():
    # Up on the move, down after costs. Counting it as a win is how a strategy
    # that churns for a spread thinner than its fees looks profitable.
    trades = [Trade(100.0, 101.0, 1.0, fee=2.0)]
    result = BacktestResult(trades, equity_curve=[1000.0, 1001.0])
    report = PerformanceReport(result, annualized_yield=0.0).as_dict()
    assert report["gross_profit"] == pytest.approx(1.0)
    assert report["net_profit"]   == pytest.approx(-1.0)
    assert report["win_rate"]     == 0.0
    # The gross average stays positive — it is the historical field. The net one
    # is what agrees with win_rate.
    assert report["avg_profit_per_trade"]     == pytest.approx(1.0)
    assert report["avg_net_profit_per_trade"] == pytest.approx(-1.0)


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
        "unwind_at_entry_price": True,
        "allow_short": False,
        "short_entry_threshold": 0.25,
        "short_exit_threshold": 0.40,
        "fee_bps": 0.0,
        "borrow_bps_per_hour": 0.0,
        "signal_score_version": 2,
    }


def test_strategy_json_save_creates_missing_parent_directories(tmp_path):
    # the default strategy_filepath now lives under ~/.coinbase/ga/, which
    # (unlike the old cwd-relative default) may not exist yet on first use
    metadata    = StrategyMetadata(
        pair="BTC-USDC", granularity="ONE_HOUR", training_period=TrainingPeriod("2024-01-01", "2024-01-02"),
        ga_config=_ga_config(), created_at="2026-07-15T10:30:00+00:00",
    )
    strategy    = TrainedStrategy(Genome({"sma_short": 1.0}), _strategy_config())
    performance = PerformanceReport(BacktestResult([], [1000.0]), annualized_yield=0.0)

    filepath = tmp_path / "nested" / "dir" / "best_strategy.json"
    StrategyJson(metadata, strategy, performance).save(str(filepath))
    assert filepath.exists()


# ── RunHeader ────────────────────────────────────────────────────────

def _run_header() -> RunHeader:
    return RunHeader(
        started_at="2026-07-21T21:15:03+00:00",
        pair="FET-USDC",
        granularity="TWO_HOUR",
        start_date="2026-01-01",
        end_date="2026-07-01",
        test_split=0.5,
        strategy_config=_strategy_config(),
        ga_config=_ga_config(),
    )


def test_run_header_lines_include_timestamp_window_and_config():
    lines = _run_header().lines()
    assert lines[0] == "=== run 2026-07-21T21:15:03+00:00 ==="
    assert lines[1] == "pair=FET-USDC granularity=TWO_HOUR window=2026-01-01..2026-07-01 test_split=0.5"
    assert lines[2] == (
        "buy_threshold=0.60 sell_threshold=0.40 position_size_pct=0.10 "
        "unwind_at_entry_price=True allow_short=False short_entry=0.25 short_exit=0.40"
    )
    assert lines[3] == (
        "population=20 generations=5 mutation_rate=0.2 crossover_rate=0.7 "
        "tournament_size=3 elitism_count=2 mutation_sigma=0.15 seed=42"
    )
    assert lines[4] == "generation\tbest_fitness\tavg_fitness"


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


def test_ga_run_log_append_creates_missing_parent_directories(tmp_path):
    filepath = tmp_path / "nested" / "dir" / "ga_run_log.txt"
    GaRunLog(str(filepath)).append(1, best_fitness=1.5, average_fitness=0.8)
    assert filepath.exists()


def test_ga_run_log_start_creates_missing_parent_directories(tmp_path):
    filepath = tmp_path / "nested" / "dir" / "ga_run_log.txt"
    GaRunLog(str(filepath)).start(_run_header())
    assert filepath.exists()


def test_ga_run_log_start_writes_header_before_generation_rows(tmp_path):
    filepath = tmp_path / "ga_run_log.txt"
    log = GaRunLog(str(filepath))
    log.start(_run_header())
    log.append(1, best_fitness=1.5, average_fitness=0.8)

    lines = filepath.read_text().splitlines()
    assert lines[:5] == _run_header().lines()
    assert lines[5] == "1\t1.500000\t0.800000"


def test_ga_run_log_start_appends_to_existing_history(tmp_path):
    filepath = tmp_path / "ga_run_log.txt"
    filepath.write_text("1\t0.100000\t0.050000\n")

    log = GaRunLog(str(filepath))
    log.start(_run_header())

    lines = filepath.read_text().splitlines()
    assert lines[0] == "1\t0.100000\t0.050000"
    assert lines[1] == "=== run 2026-07-21T21:15:03+00:00 ==="


def test_fan_out_run_log_writes_to_every_underlying_log(tmp_path):
    path_a, path_b = tmp_path / "a.txt", tmp_path / "b.txt"
    fan_out = FanOutRunLog((GaRunLog(str(path_a)), GaRunLog(str(path_b))))

    fan_out.start(_run_header())
    fan_out.append(1, best_fitness=1.5, average_fitness=0.8)

    for path in (path_a, path_b):
        lines = path.read_text().splitlines()
        assert lines[:5] == _run_header().lines()
        assert lines[5] == "1\t1.500000\t0.800000"


# ── DryRunLog ────────────────────────────────────────────────────────

def test_dry_run_log_appends_a_line_per_tick(tmp_path):
    filepath = tmp_path / "dry_run_log.txt"
    log = DryRunLog(str(filepath))
    log.append("2026-08-18T00:00:00+00:00", Decision(Action.BUY, size=1.5), balance=1000.0, equity=1000.0)
    log.append("2026-08-18T01:00:00+00:00", Decision(Action.HOLD), balance=1000.0, equity=1015.0)

    lines = filepath.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == "2026-08-18T00:00:00+00:00\tBUY\t1.500000\t1000.000000\t1000.000000"
    assert lines[1] == "2026-08-18T01:00:00+00:00\tHOLD\t0.000000\t1000.000000\t1015.000000"


def test_dry_run_log_creates_missing_parent_directories(tmp_path):
    filepath = tmp_path / "nested" / "dir" / "dry_run_log.txt"
    DryRunLog(str(filepath)).append("2026-08-18T00:00:00+00:00", Decision(Action.HOLD), balance=1000.0, equity=1000.0)
    assert filepath.exists()
