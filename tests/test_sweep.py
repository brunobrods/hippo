import csv
import math
import os

import pytest

from coinbase.ga.config import ConfigFile
from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.main import TrainingSummary
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.ga.strategy_output import PerformanceReport, StrategyJson, StrategyMetadata, TrainedStrategy, TrainingPeriod
from coinbase.ga.sweep import ConsoleSweepProgress, DottedPathOverride, Sweep, SweepAxis, SweepConfigFile, SweepPlan, SweepPoint
from coinbase.trading_strategy import BacktestResult


# ── Test double ──────────────────────────────────────────────────────

class FakeAdapter:
    """Stands in for CoinbaseAdapter — only the method market data touches."""

    def __init__(self, candles: list[dict]) -> None:
        self._candles = candles

    async def get_product_candles(self, product_id: str, start: int, end: int, granularity: str) -> list[dict]:
        return self._candles

    def max_candles_per_request(self) -> int:
        return 300

    def name(self) -> str:
        return "coinbase"


def _candle(start: int, close: float) -> dict:
    return {"start": str(start), "close": str(close), "high": str(close), "low": str(close), "volume": "1.0"}


def _oscillating_candles(n: int, step: int = 3600) -> list[dict]:
    return [_candle(i * step, close=100.0 + 10.0 * math.sin(i / 5.0)) for i in range(n)]


def _base_config(output_dir) -> dict:
    return {
        "data": {
            "pair": "BTC-USDC",
            "granularity": "ONE_HOUR",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "test_split": 0.3,
        },
        "market_data": {
            "cache_dir": str(output_dir / "candle_cache"),
            "normalized_columns": ["sma_short", "sma_long", "sma_extra", "rsi", "macd"],
            "delta_columns": ["delta_1", "delta_3", "delta_5", "delta_10"],
        },
        "strategy": {
            "position_size_pct": 0.10,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "starting_balance": 1000.0,
            "indicators": {
                "sma_short_period": 2,
                "sma_long_period": 3,
                "sma_extra_period": 5,
                "rsi_period": 5,
                "macd_fast": 2,
                "macd_slow": 3,
                "macd_signal": 2,
            },
            "weight_keys": ["sma_short", "sma_long", "sma_extra", "rsi", "macd"],
        },
        "genetic_algorithm": {
            "population_size": 6,
            "generations": 2,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "tournament_size": 3,
            "elitism_count": 1,
            "seed": 11,
        },
        "output": {
            "strategy_filepath": str(output_dir / "best_strategy.json"),
            "log_filepath":      str(output_dir / "ga_run_log.txt"),
            "experiments_dir":   str(output_dir / "experiments"),
            "index_filepath":    str(output_dir / "experiments" / "index.csv"),
        },
    }


# ── SweepConfigFile ──────────────────────────────────────────────────

def test_sweep_config_file_reads_base_config_path_and_seeds():
    raw = {
        "base_config": "coinbase/ga/config.yaml",
        "seeds": [1, 2, 3],
        "axes": [{"path": "strategy.buy_threshold", "values": [0.5, 0.6]}],
    }
    config = SweepConfigFile(raw)
    assert config.base_config_path() == "coinbase/ga/config.yaml"
    assert config.seeds() == (1, 2, 3)


def test_sweep_config_file_reads_axes():
    raw = {
        "base_config": "x.yaml",
        "seeds": [1],
        "axes": [
            {"path": "strategy.buy_threshold", "values": [0.5, 0.6]},
            {"path": "genetic_algorithm.mutation_rate", "values": [0.1, 0.2, 0.3]},
        ],
    }
    axes = SweepConfigFile(raw).axes()
    assert axes == (
        SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.6)),
        SweepAxis(path="genetic_algorithm.mutation_rate", values=(0.1, 0.2, 0.3)),
    )


# ── DottedPathOverride ───────────────────────────────────────────────

def test_dotted_path_override_sets_a_nested_value():
    raw_config = {"strategy": {"buy_threshold": 0.6, "sell_threshold": 0.4}}
    result = DottedPathOverride(raw_config, "strategy.buy_threshold", 0.9).applied()
    assert result["strategy"]["buy_threshold"] == 0.9
    assert result["strategy"]["sell_threshold"] == 0.4


def test_dotted_path_override_does_not_mutate_the_original():
    raw_config = {"strategy": {"buy_threshold": 0.6}}
    DottedPathOverride(raw_config, "strategy.buy_threshold", 0.9).applied()
    assert raw_config["strategy"]["buy_threshold"] == 0.6


def test_dotted_path_override_raises_for_an_unknown_path():
    raw_config = {"strategy": {"buy_threshold": 0.6}}
    with pytest.raises(KeyError):
        DottedPathOverride(raw_config, "strategy.nonexistent_key.deeper", 1).applied()


def test_dotted_path_override_treats_a_fully_commented_out_section_as_empty():
    # A YAML mapping whose every child is commented out (e.g. a fully
    # optional `output:` section) parses to None, not {}.
    raw_config = {"output": None}
    result = DottedPathOverride(raw_config, "output.strategy_filepath", "./scratch.json").applied()
    assert result["output"]["strategy_filepath"] == "./scratch.json"


# ── SweepPlan ────────────────────────────────────────────────────────

def _minimal_base(experiments_dir: str = "./experiments") -> dict:
    return {
        "strategy":         {"buy_threshold": 0.6, "sell_threshold": 0.4},
        "genetic_algorithm": {"seed": 1},
        "output":           {"experiments_dir": experiments_dir, "strategy_filepath": "./best_strategy.json"},
    }


def test_sweep_plan_produces_one_point_per_axis_value_per_seed():
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.7)),)
    points = SweepPlan(_minimal_base(), axes, seeds=(1, 2, 3)).points()
    assert len(points) == 2 * 3  # 2 values x 3 seeds


def test_sweep_plan_is_one_factor_at_a_time():
    axes = (
        SweepAxis(path="strategy.buy_threshold", values=(0.9,)),
        SweepAxis(path="strategy.sell_threshold", values=(0.1,)),
    )
    points = SweepPlan(_minimal_base(), axes, seeds=(1,)).points()

    buy_point  = next(p for p in points if p.axis_path == "strategy.buy_threshold")
    sell_point = next(p for p in points if p.axis_path == "strategy.sell_threshold")

    assert buy_point.raw_config["strategy"]["buy_threshold"] == 0.9
    assert buy_point.raw_config["strategy"]["sell_threshold"] == 0.4  # untouched base value

    assert sell_point.raw_config["strategy"]["sell_threshold"] == 0.1
    assert sell_point.raw_config["strategy"]["buy_threshold"] == 0.6  # untouched base value


def test_sweep_plan_overrides_seed_per_point():
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    points = SweepPlan(_minimal_base(), axes, seeds=(7, 8)).points()
    assert {p.raw_config["genetic_algorithm"]["seed"] for p in points} == {7, 8}
    assert {p.seed for p in points} == {7, 8}


def test_sweep_plan_redirects_strategy_filepath_to_a_scratch_file():
    # TrainingRun.train() unconditionally overwrites output.strategy_filepath —
    # a sweep must never let that clobber the base config's canonical file.
    base   = _minimal_base(experiments_dir="/tmp/experiments")
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.7)),)
    points = SweepPlan(base, axes, seeds=(1, 2)).points()

    scratch_dir = os.path.join("/tmp/experiments", "_sweep_scratch")
    for point in points:
        assert point.raw_config["output"]["strategy_filepath"].startswith(scratch_dir)
        assert point.raw_config["output"]["strategy_filepath"] != base["output"]["strategy_filepath"]


# One path PER POINT. Every run writes its best strategy there and reads it
# straight back to verify the reload matches; points running concurrently on a
# shared path would each verify whichever run happened to write last.
def test_every_point_gets_its_own_scratch_file():
    base   = _minimal_base(experiments_dir="/tmp/experiments")
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.7)),)
    points = SweepPlan(base, axes, seeds=(1, 2)).points()

    paths = [point.raw_config["output"]["strategy_filepath"] for point in points]
    assert len(set(paths)) == len(points) == 4


def test_sweep_plan_scratch_path_uses_the_default_when_experiments_dir_absent():
    from coinbase.ga.config import GA_RESULTS_ROOT

    base   = {"strategy": {"buy_threshold": 0.6}, "genetic_algorithm": {"seed": 1}, "output": {}}
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    points = SweepPlan(base, axes, seeds=(1,)).points()

    expected = os.path.join(str(GA_RESULTS_ROOT / "experiments"), "_sweep_scratch", "best_strategy_0.json")
    assert points[0].raw_config["output"]["strategy_filepath"] == expected


def test_sweep_plan_works_against_the_real_shipped_config_yaml():
    # regression test: config.yaml ships with every `output:` key commented
    # out, so raw["output"] is None — SweepPlan must not crash on this.
    base   = ConfigFile("coinbase/ga/config.yaml").raw()
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    points = SweepPlan(base, axes, seeds=(1,)).points()
    assert len(points) == 1
    assert points[0].raw_config["output"]["strategy_filepath"]


def test_sweep_plan_ensure_scratch_directory_creates_the_directory(tmp_path):
    base = _minimal_base(experiments_dir=str(tmp_path / "experiments"))
    axes = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    plan = SweepPlan(base, axes, seeds=(1,))

    scratch_dir = tmp_path / "experiments" / "_sweep_scratch"
    assert not scratch_dir.exists()
    plan.ensure_scratch_directory()
    assert scratch_dir.is_dir()


# ── Sweep (end-to-end, fake adapter) ──────────────────────────────────

@pytest.mark.asyncio
async def test_sweep_runs_one_training_run_per_point_and_records_history(tmp_path):
    adapter = FakeAdapter(_oscillating_candles(60))
    base    = _base_config(tmp_path)
    axes    = (SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.7)),)
    plan    = SweepPlan(base, axes, seeds=(1, 2))
    plan.ensure_scratch_directory()
    points  = plan.points()
    assert len(points) == 4

    summaries = await Sweep(adapter, points, ConsoleSweepProgress(len(points))).run()

    assert len(summaries) == 4
    assert len({summary.run_id() for summary in summaries}) == 4  # every run_id distinct

    index_rows = list(csv.DictReader((tmp_path / "experiments" / "index.csv").read_text().splitlines()))
    assert len(index_rows) == 4
    assert {row["buy_threshold"] for row in index_rows} == {"0.5", "0.7"}
    assert {row["seed"] for row in index_rows} == {"1", "2"}

    # the base config's canonical strategy_filepath must never be touched by a sweep
    assert not (tmp_path / "best_strategy.json").exists()
    assert (tmp_path / "experiments" / "_sweep_scratch" / "best_strategy_0.json").exists()


# ── ConsoleSweepProgress ─────────────────────────────────────────────

def test_console_sweep_progress_prints_point_started_and_finished(capsys):
    progress = ConsoleSweepProgress(total=2)
    progress.point_started(1, SweepPoint("strategy.buy_threshold", 0.5, 1, {}))

    metadata = StrategyMetadata(
        pair="BTC-USDC", granularity="ONE_HOUR", training_period=TrainingPeriod("2024-01-01", "2024-01-02"),
        ga_config=GaConfig(population_size=6, generations=2, mutation_rate=0.2, crossover_rate=0.8,
                            tournament_size=3, elitism_count=1),
        created_at="2026-07-18T00:00:00+00:00",
    )
    strategy = TrainedStrategy(
        Genome({"sma_short": 1.0}),
        StrategyConfig(position_size_pct=0.1, buy_threshold=0.6, sell_threshold=0.4, starting_balance=1000.0),
    )
    performance = PerformanceReport(BacktestResult([], [1000.0]), annualized_yield=0.0)
    summary = TrainingSummary(
        StrategyJson(metadata, strategy, performance), output_path="/tmp/x.json",
        run_id="run-1", round_trip_matches=True,
    )
    progress.point_finished(summary)

    out = capsys.readouterr().out
    assert "[1/2] strategy.buy_threshold=0.5 seed=1" in out
    assert "Saved strategy to /tmp/x.json" in out
