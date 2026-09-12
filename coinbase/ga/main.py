import asyncio
import math
from typing import Any

from exchange.adapter import ExchangeAdapter
from exchange.selection import ConfiguredExchange
from coinbase.ga.config import ConfigFile
from coinbase.ga.experiment_history import (
    ExperimentDirectory,
    ExperimentIndex,
    ExperimentRecord,
    GitCommitHash,
    ResolvedConfigFile,
    RunId,
)
from coinbase.ga.ga_engine import GaConfigFile, GeneticAlgorithm, Genome
from coinbase.ga.market_data_processor import (
    HistoricalMarketData,
    MarketBasket,
    MarketDataConfig,
    TrainTestSplit,
)
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    SignalDesign,
    StrategyConfigFile,
    StrategyEvaluator,
    ValidatedStrategyConfig,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.ga.strategy_output import (
    FanOutRunLog,
    GaRunLog,
    OutputConfigFile,
    PerformanceReport,
    RunHeader,
    StrategyJson,
    StrategyJsonFile,
    StrategyMetadata,
    TrainedStrategy,
    TrainingPeriod,
    UtcNow,
)


# ── Console progress ─────────────────────────────────────────────────────

class ConsoleGenerationLog:
    def __init__(self, log: GaRunLog) -> None:
        self._log = log

    def start(self, header: RunHeader) -> None:
        self._log.start(header)
        print("\n".join(header.lines()))

    def append(self, generation: int, best_fitness: float, average_fitness: float) -> None:
        self._log.append(generation, best_fitness, average_fitness)
        print(f"generation {generation:>4}  best_fitness {best_fitness:>14.4f}  avg {average_fitness:>14.4f}")


# ── Summary ──────────────────────────────────────────────────────────────

class TrainingSummary:
    def __init__(
        self, strategy_json: StrategyJson, output_path: str, run_id: str, round_trip_matches: bool,
    ) -> None:
        self._strategy_json      = strategy_json
        self._output_path        = output_path
        self._run_id             = run_id
        self._round_trip_matches = round_trip_matches

    def run_id(self) -> str:
        return self._run_id

    def as_text(self) -> str:
        performance = self._strategy_json.as_dict()["performance"]
        return (
            f"Saved strategy to {self._output_path}\n"
            f"Experiment run_id:      {self._run_id}\n"
            f"Test-set gross profit:  {performance['gross_profit']:.2f}\n"
            f"Test-set net profit:    {performance['net_profit']:.2f}"
            f"  (fees {performance['fees_paid']:.2f}, interest {performance['interest_paid']:.2f})\n"
            f"Test-set annualized yield: {performance['annualized_yield']:+.1%}\n"
            f"Total trades:           {performance['total_trades']}\n"
            f"Win rate:               {performance['win_rate']:.1%}\n"
            f"Max drawdown:           {performance['max_drawdown']:.1%}\n"
            f"Reload round-trip:      {'OK' if self._round_trip_matches else 'MISMATCH'}"
        )


# ── Orchestration ────────────────────────────────────────────────────────

class TrainingRun:
    def __init__(self, adapter: ExchangeAdapter, raw_config: dict[str, Any]) -> None:
        self._adapter    = adapter
        self._raw_config = raw_config

    async def train(self) -> TrainingSummary:
        market_config   = MarketDataConfig(self._raw_config)
        window          = market_config.window()
        strategy_config = ValidatedStrategyConfig(StrategyConfigFile(self._raw_config).config()).config()
        ga_config       = GaConfigFile(self._raw_config).config()
        output_config   = OutputConfigFile(self._raw_config).config()
        data            = self._raw_config["data"]

        # fetched early and deliberately not swallowed on failure — fail fast,
        # before the expensive fetch/train below, rather than after a full
        # training run has already saved its results.
        git_commit = await GitCommitHash().value()
        # Same reasoning: the index is only appended to at the very end, so an
        # incompatible one would otherwise surface after a full evolution — and
        # would abort every remaining point of a sweep along with it.
        ExperimentIndex(output_config.index_filepath).ensure_appendable()

        # The basket the index_z column is measured against, fetched once for
        # the whole run. Empty by default, in which case the frame is exactly
        # the one every genome trained so far was scored on.
        index_returns = None
        if market_config.index_pairs():
            index_returns = await MarketBasket(
                self._adapter, market_config.index_pairs(), window.granularity,
                window.start, window.end, market_config.cache_dir(),
            ).returns()

        frame = await HistoricalMarketData(
            self._adapter, window.pair, window.granularity, window.start, window.end,
            market_config.periods(), market_config.columns(), market_config.cache_dir(),
            index_returns=index_returns, index_period=market_config.index_period(),
        ).dataframe()
        split = TrainTestSplit(frame, window.test_split)
        weight_keys = ValidatedWeightKeys(
            WeightKeysConfig(self._raw_config).keys(), market_config.normalized_columns(),
        ).keys()
        keys = weight_keys + (POSITION_PNL_KEY,)

        train_evaluator = StrategyEvaluator(split.train(), strategy_config, keys)
        test_evaluator  = StrategyEvaluator(split.test(), strategy_config, keys)

        started_at = UtcNow().iso()
        run_id     = RunId(started_at).value()
        experiment_dir = ExperimentDirectory(output_config.experiments_dir, run_id)
        experiment_dir.ensure()
        ResolvedConfigFile(experiment_dir.config_path(), self._raw_config).save()

        console_log = ConsoleGenerationLog(FanOutRunLog((
            GaRunLog(output_config.log_filepath),
            GaRunLog(experiment_dir.log_path()),
        )))
        console_log.start(RunHeader(
            started_at      = started_at,
            pair            = window.pair,
            granularity     = window.granularity,
            start_date      = data["start_date"],
            end_date        = data["end_date"],
            test_split      = window.test_split,
            strategy_config = strategy_config,
            ga_config       = ga_config,
        ))
        # The design supplies the scaling, because how a genome's numbers are
        # rescaled is a property of the model rather than of the search: L1
        # means "one unit of conviction spread across the indicators", which is
        # right for a weighted sum and wrong for anything whose parameters are
        # not weights. GeneticAlgorithm defaults to L1, so omitting this trains
        # every future design as if it were linear — silently, and only in the
        # numbers.
        best_genome = GeneticAlgorithm(
            ga_config, keys, SignalDesign(strategy_config.design).scaling(),
        ).evolve(train_evaluator, on_generation=console_log.append)

        metadata = StrategyMetadata(
            pair            = window.pair,
            granularity     = window.granularity,
            training_period = TrainingPeriod(data["start_date"], data["end_date"]),
            ga_config       = ga_config,
            created_at      = UtcNow().iso(),
        )
        strategy      = TrainedStrategy(best_genome, strategy_config)
        test_result   = test_evaluator.result(best_genome)
        test_yield    = test_evaluator.annualized_yield(test_result)
        performance   = PerformanceReport(test_result, test_yield)
        strategy_json = StrategyJson(metadata, strategy, performance)
        strategy_json.save(output_config.strategy_filepath)
        strategy_json.save(experiment_dir.strategy_path())

        ExperimentIndex(output_config.index_filepath).append(ExperimentRecord(
            run_id          = run_id,
            started_at      = started_at,
            git_commit      = git_commit,
            pair            = window.pair,
            granularity     = window.granularity,
            start_date      = data["start_date"],
            end_date        = data["end_date"],
            test_split      = window.test_split,
            ga_config       = ga_config,
            strategy_config = strategy_config,
            performance     = performance.as_dict(),
        ))

        # Compares yield to yield, not fitness to yield: fitness deliberately
        # departs from realized yield for a genome that never trades, so using
        # it here would report a mismatch for a strategy that reloaded fine.
        reloaded        = StrategyJsonFile(output_config.strategy_filepath)
        reloaded_yield  = test_evaluator.annualized_yield(test_evaluator.result(Genome(reloaded.weights())))
        round_trip_matches = math.isclose(reloaded_yield, test_yield, rel_tol=1e-9)

        return TrainingSummary(strategy_json, output_config.strategy_filepath, run_id, round_trip_matches)


# ── Entry point ────────────────────────────────────────────────────────
# Run (as a module, from the repo root — a direct script path fails with
# ModuleNotFoundError: No module named 'coinbase'):
#   python -m coinbase.ga.main
# Trains a GA strategy on live historical data from the exchange named by
# `data.exchange` in config.yaml, evaluates it on the
# held-out test split, saves it to output.strategy_filepath, and verifies
# the saved JSON reloads into an identical backtest result.

async def _main() -> None:
    raw_config = ConfigFile("coinbase/ga/config.yaml").raw()

    async with ConfiguredExchange(raw_config).adapter() as adapter:
        summary = await TrainingRun(adapter, raw_config).train()

    print()
    print(summary.as_text())


if __name__ == "__main__":
    asyncio.run(_main())
