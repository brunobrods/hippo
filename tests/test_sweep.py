import csv
import math
import os

import pytest

from coinbase.ga.config import ConfigFile
from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.main import TrainingSummary
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.ga.strategy_output import PerformanceReport, StrategyJson, StrategyMetadata, TrainedStrategy, TrainingPeriod
from coinbase.ga.sweep import (
    ConsoleSweepProgress,
    DottedPathOverride,
    FixedOverrides,
    Sweep,
    SweepAxis,
    SweepConfigFile,
    SweepPlan,
    SweepPoint,
)
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


def test_sweep_config_file_reads_fixed_overrides_in_order():
    raw = {
        "base_config": "x.yaml",
        "fixed": {"data.exchange": "binance", "data.granularity": "THIRTY_MINUTE"},
        "seeds": [1],
        "axes": [{"path": "data.pair", "values": ["BTC-USDT"]}],
    }
    assert SweepConfigFile(raw).fixed() == (
        ("data.exchange", "binance"),
        ("data.granularity", "THIRTY_MINUTE"),
    )


def test_sweep_config_file_fixed_is_empty_when_the_section_is_absent():
    raw = {"base_config": "x.yaml", "seeds": [1], "axes": [{"path": "a.b", "values": [1]}]}
    assert SweepConfigFile(raw).fixed() == ()


def test_sweep_config_file_fixed_is_empty_when_the_section_is_commented_out():
    # A YAML mapping whose every child is commented out parses to None, not {}.
    raw = {"base_config": "x.yaml", "fixed": None, "seeds": [1], "axes": [{"path": "a.b", "values": [1]}]}
    assert SweepConfigFile(raw).fixed() == ()


# ── FixedOverrides ───────────────────────────────────────────────────

def test_fixed_overrides_applies_every_path():
    base   = {"data": {"exchange": "coinbase", "granularity": "SIX_HOUR", "pair": "BTC-USDC"}}
    result = FixedOverrides(base, (
        ("data.exchange", "binance"),
        ("data.granularity", "THIRTY_MINUTE"),
    )).applied()
    assert result["data"]["exchange"] == "binance"
    assert result["data"]["granularity"] == "THIRTY_MINUTE"
    assert result["data"]["pair"] == "BTC-USDC"  # untouched


def test_fixed_overrides_does_not_mutate_the_original():
    base = {"data": {"exchange": "coinbase"}}
    FixedOverrides(base, (("data.exchange", "binance"),)).applied()
    assert base["data"]["exchange"] == "coinbase"


def test_fixed_overrides_with_nothing_fixed_returns_an_equal_but_separate_config():
    base   = {"data": {"exchange": "coinbase"}}
    result = FixedOverrides(base, ()).applied()
    assert result == base
    result["data"]["exchange"] = "binance"
    assert base["data"]["exchange"] == "coinbase"  # never an alias of the input


def test_fixed_overrides_reach_every_sweep_point():
    # The whole point of `fixed:` — SweepPlan is built from the already-fixed
    # base, so a pinned path shows up on every point without being an axis.
    base   = FixedOverrides(
        {"data": {"granularity": "SIX_HOUR"}, "strategy": {"buy_threshold": 0.6, "sell_threshold": 0.4},
         "genetic_algorithm": {"seed": 1}, "output": {"experiments_dir": "/tmp/e", "strategy_filepath": "/tmp/s.json"}},
        (("data.granularity", "THIRTY_MINUTE"),),
    ).applied()
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5, 0.7)),)
    points = SweepPlan(base, axes, seeds=(1, 2)).points()

    assert len(points) == 4
    assert all(point.raw_config["data"]["granularity"] == "THIRTY_MINUTE" for point in points)


def test_shipped_pair_sweep_configs_pin_exchange_and_granularity():
    # Regression guard: the three arms must differ ONLY in granularity, or the
    # trade-count comparison between them is not attributable to candle size.
    arms = {
        "coinbase/ga/sweep_pairs_30m.yaml": "THIRTY_MINUTE",
        "coinbase/ga/sweep_pairs_2h.yaml":  "TWO_HOUR",
        "coinbase/ga/sweep_pairs_6h.yaml":  "SIX_HOUR",
    }
    seen_pairs = set()
    for path, granularity in arms.items():
        config = SweepConfigFile(ConfigFile(path).raw())
        assert dict(config.fixed()) == {"data.exchange": "binance", "data.granularity": granularity}
        assert config.seeds() == (1, 2, 3)
        axes = config.axes()
        assert len(axes) == 1 and axes[0].path == "data.pair"
        seen_pairs.add(axes[0].values)
    assert len(seen_pairs) == 1  # all three arms sweep the identical pair list


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
    points = SweepPlan(base, axes, seeds=(1, 2), process_id=4242).points()

    expected_scratch_path = os.path.join("/tmp/experiments", "_sweep_scratch", "best_strategy_4242.json")
    for point in points:
        assert point.raw_config["output"]["strategy_filepath"] == expected_scratch_path
        assert point.raw_config["output"]["strategy_filepath"] != base["output"]["strategy_filepath"]


def test_sweep_plan_scratch_path_uses_shared_default_when_experiments_dir_absent():
    from coinbase.ga.config import GA_RESULTS_ROOT

    base   = {"strategy": {"buy_threshold": 0.6}, "genetic_algorithm": {"seed": 1}, "output": {}}
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    points = SweepPlan(base, axes, seeds=(1,), process_id=4242).points()

    expected = os.path.join(
        str(GA_RESULTS_ROOT / "experiments"), "_sweep_scratch", "best_strategy_4242.json",
    )
    assert points[0].raw_config["output"]["strategy_filepath"] == expected


def test_sweep_plan_scratch_path_defaults_to_the_running_process():
    base   = _minimal_base(experiments_dir="/tmp/experiments")
    axes   = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    points = SweepPlan(base, axes, seeds=(1,)).points()
    assert points[0].raw_config["output"]["strategy_filepath"].endswith(
        "best_strategy_%d.json" % os.getpid()
    )


def test_two_concurrent_sweeps_do_not_share_one_scratch_file():
    # TrainingRun writes strategy_filepath then reads it straight back for its
    # round-trip check, so two sweeps sharing the path would verify against
    # each other's strategy — or against a half-written file.
    base = _minimal_base(experiments_dir="/tmp/experiments")
    axes = (SweepAxis(path="strategy.buy_threshold", values=(0.5,)),)
    first  = SweepPlan(base, axes, seeds=(1,), process_id=111).points()
    second = SweepPlan(base, axes, seeds=(1,), process_id=222).points()

    assert (first[0].raw_config["output"]["strategy_filepath"]
            != second[0].raw_config["output"]["strategy_filepath"])


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
    scratch = tmp_path / "experiments" / "_sweep_scratch"
    assert [path.name for path in scratch.iterdir()] == ["best_strategy_%d.json" % os.getpid()]


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
