import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from coinbase.ga.config import GA_RESULTS_ROOT
from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.trading_strategy import BacktestResult, Decision


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputConfig:
    strategy_filepath:    str
    log_filepath:         str
    dry_run_log_filepath: str
    experiments_dir:      str
    index_filepath:       str


class OutputConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> OutputConfig:
        # `or {}`: a YAML mapping whose every child is commented out parses to
        # None, not {} — every path here is meant to be optional in that case.
        section          = self._raw.get("output") or {}
        experiments_dir  = section.get("experiments_dir", str(GA_RESULTS_ROOT / "experiments"))
        return OutputConfig(
            strategy_filepath    = section.get("strategy_filepath", str(GA_RESULTS_ROOT / "best_strategy.json")),
            log_filepath         = section.get("log_filepath", str(GA_RESULTS_ROOT / "ga_run_log.txt")),
            dry_run_log_filepath = section.get("dry_run_log_filepath", str(GA_RESULTS_ROOT / "dry_run_log.txt")),
            experiments_dir      = experiments_dir,
            index_filepath       = section.get("index_filepath", os.path.join(experiments_dir, "index.csv")),
        )


# ── Clock ──────────────────────────────────────────────────────────────

class UtcNow:
    def iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


# ── Metadata ───────────────────────────────────────────────────────────

class TrainingPeriod:
    def __init__(self, start_date: str, end_date: str) -> None:
        self._start_date = start_date
        self._end_date   = end_date

    def as_string(self) -> str:
        return f"{self._start_date} to {self._end_date}"


class StrategyMetadata:
    def __init__(
        self,
        pair: str,
        granularity: str,
        training_period: TrainingPeriod,
        ga_config: GaConfig,
        created_at: str,
    ) -> None:
        self._pair            = pair
        self._granularity     = granularity
        self._training_period = training_period
        self._ga_config       = ga_config
        self._created_at      = created_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_at":      self._created_at,
            "pair":            self._pair,
            "timeframe":       self._granularity,
            "training_period": self._training_period.as_string(),
            "ga_config":       dataclasses.asdict(self._ga_config),
        }


# ── Strategy ───────────────────────────────────────────────────────────

class TrainedStrategy:
    def __init__(self, genome: Genome, strategy_config: StrategyConfig) -> None:
        self._genome = genome
        self._config = strategy_config

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self._genome.weights(),
            "hyperparameters": {
                "buy_threshold":         self._config.buy_threshold,
                "sell_threshold":        self._config.sell_threshold,
                "position_size_pct":     self._config.position_size_pct,
                "unwind_at_entry_price": self._config.unwind_at_entry_price,
                "allow_short":           self._config.allow_short,
                "short_entry_threshold": self._config.short_entry_threshold,
                "short_exit_threshold":  self._config.short_exit_threshold,
                # The costs the genome was selected under. Saved for the same
                # reason the thresholds are: a paper run charged at a rate the
                # scoring never assumed is measuring a different strategy, and
                # TrainedStrategyConfig.divergences() can only report a drift
                # in keys that are recorded here.
                "fee_bps":               self._config.fee_bps,
                "borrow_bps_per_hour":   self._config.borrow_bps_per_hour,
                # Which model design produced these weights. Without it a
                # genome is just a bag of numbers, and a loader would score it
                # through whatever design happened to be configured. Absent in
                # anything saved before designs were named, where it reads as
                # "linear" — which is what those runs were.
                "design":                self._config.design,
                # The resting target decides when a position closes, so a genome
                # papered without it exits on signal alone and trades a strategy
                # nobody scored. Measured: at 1% it takes 61 trades where the
                # same genome at 0.0 takes 5.
                "take_profit_pct":       self._config.take_profit_pct,
            },
        }


# ── Performance ────────────────────────────────────────────────────────

class MaxDrawdown:
    def __init__(self, equity_curve: list[float]) -> None:
        self._equity_curve = equity_curve

    def fraction(self) -> float:
        peak      = float("-inf")
        max_drop  = 0.0
        for equity in self._equity_curve:
            peak = max(peak, equity)
            if peak > 0.0:
                max_drop = max(max_drop, (peak - equity) / peak)
        return max_drop


class PerformanceReport:
    def __init__(self, result: BacktestResult, annualized_yield: float) -> None:
        self._result           = result
        self._annualized_yield = annualized_yield

    # gross_profit and avg_profit_per_trade stay gross so every strategy.json
    # written before costs existed still means what it says. net_profit,
    # fees_paid and interest_paid are the new truth, and win_rate counts a trade
    # a win only if it paid for itself — with both rates at 0.0 that is the same
    # count it always was.
    def as_dict(self) -> dict[str, Any]:
        return {
            "gross_profit":         self._result.gross_profit(),
            "net_profit":           self._result.net_profit(),
            "fees_paid":            self._result.fees_paid(),
            "interest_paid":        self._result.interest_paid(),
            "annualized_yield":     self._annualized_yield,
            "total_trades":         self._total_trades(),
            "win_rate":             self._win_rate(),
            "max_drawdown":         MaxDrawdown(self._result.equity_curve()).fraction(),
            "avg_profit_per_trade": self._avg_profit_per_trade(),
            "avg_net_profit_per_trade": self._avg_net_profit_per_trade(),
        }

    def _total_trades(self) -> int:
        return len(self._result.trades())

    def _win_rate(self) -> float:
        trades = self._result.trades()
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade.net_profit() > 0.0)
        return wins / len(trades)

    def _avg_profit_per_trade(self) -> float:
        trades = self._result.trades()
        if not trades:
            return 0.0
        return self._result.gross_profit() / len(trades)

    # win_rate counts net while avg_profit_per_trade stays gross, so a strategy
    # churning for less than its fees reports 0% wins beside a positive average
    # and reads like a bug. This is the number that agrees with win_rate.
    def _avg_net_profit_per_trade(self) -> float:
        trades = self._result.trades()
        if not trades:
            return 0.0
        return self._result.net_profit() / len(trades)


# ── Persistence ─────────────────────────────────────────────────────────

class ParentDirectory:
    # The various default output paths now live under GA_RESULTS_ROOT, which
    # (unlike the old cwd-relative defaults) may not exist yet on first use.
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def ensure(self) -> None:
        directory = os.path.dirname(self._filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)


class StrategyJson:
    def __init__(
        self,
        metadata:    StrategyMetadata,
        strategy:    TrainedStrategy,
        performance: PerformanceReport,
    ) -> None:
        self._metadata    = metadata
        self._strategy    = strategy
        self._performance = performance

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata":    self._metadata.as_dict(),
            "strategy":    self._strategy.as_dict(),
            "performance": self._performance.as_dict(),
        }

    def save(self, filepath: str) -> None:
        ParentDirectory(filepath).ensure()
        with open(filepath, "w") as handle:
            json.dump(self.as_dict(), handle, indent=2)


class StrategyJsonFile:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def raw(self) -> dict[str, Any]:
        with open(self._filepath) as handle:
            return json.load(handle)

    def weights(self) -> dict[str, float]:
        return self.raw()["strategy"]["weights"]

    def hyperparameters(self) -> dict[str, Any]:
        return self.raw()["strategy"]["hyperparameters"]


# ── GA run log ───────────────────────────────────────────────────────────

class RunHeader:
    def __init__(
        self,
        started_at:      str,
        pair:            str,
        granularity:     str,
        start_date:      str,
        end_date:        str,
        test_split:      float,
        strategy_config: StrategyConfig,
        ga_config:       GaConfig,
    ) -> None:
        self._started_at      = started_at
        self._pair            = pair
        self._granularity     = granularity
        self._start_date      = start_date
        self._end_date        = end_date
        self._test_split      = test_split
        self._strategy_config = strategy_config
        self._ga_config       = ga_config

    def lines(self) -> list[str]:
        return [
            f"=== run {self._started_at} ===",
            f"pair={self._pair} granularity={self._granularity} "
            f"window={self._start_date}..{self._end_date} test_split={self._test_split}",
            f"buy_threshold={self._strategy_config.buy_threshold:.2f} "
            f"sell_threshold={self._strategy_config.sell_threshold:.2f} "
            f"position_size_pct={self._strategy_config.position_size_pct:.2f} "
            f"unwind_at_entry_price={self._strategy_config.unwind_at_entry_price} "
            f"allow_short={self._strategy_config.allow_short} "
            f"short_entry={self._strategy_config.short_entry_threshold:.2f} "
            f"short_exit={self._strategy_config.short_exit_threshold:.2f}",
            f"population={self._ga_config.population_size} generations={self._ga_config.generations} "
            f"mutation_rate={self._ga_config.mutation_rate} crossover_rate={self._ga_config.crossover_rate} "
            f"tournament_size={self._ga_config.tournament_size} elitism_count={self._ga_config.elitism_count} "
            f"mutation_sigma={self._ga_config.mutation_sigma} seed={self._ga_config.seed}",
            "generation\tbest_fitness\tavg_fitness",
        ]


class GaRunLog:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def start(self, header: RunHeader) -> None:
        ParentDirectory(self._filepath).ensure()
        with open(self._filepath, "a") as handle:
            handle.write("\n".join(header.lines()) + "\n")

    def append(self, generation: int, best_fitness: float, average_fitness: float) -> None:
        ParentDirectory(self._filepath).ensure()
        with open(self._filepath, "a") as handle:
            handle.write(f"{generation}\t{best_fitness:.6f}\t{average_fitness:.6f}\n")


class FanOutRunLog:
    def __init__(self, logs: tuple[GaRunLog, ...]) -> None:
        self._logs = logs

    def start(self, header: RunHeader) -> None:
        for log in self._logs:
            log.start(header)

    def append(self, generation: int, best_fitness: float, average_fitness: float) -> None:
        for log in self._logs:
            log.append(generation, best_fitness, average_fitness)


# ── Dry-run log ────────────────────────────────────────────────────────

class DryRunLog:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def append(self, timestamp: str, decision: Decision, balance: float, equity: float) -> None:
        ParentDirectory(self._filepath).ensure()
        with open(self._filepath, "a") as handle:
            handle.write(
                f"{timestamp}\t{decision.action.value}\t{decision.size:.6f}\t"
                f"{balance:.6f}\t{equity:.6f}\n"
            )
