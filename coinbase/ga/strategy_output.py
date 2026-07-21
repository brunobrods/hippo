import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.trading_strategy import BacktestResult


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputConfig:
    strategy_filepath: str
    log_filepath:       str


class OutputConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> OutputConfig:
        section = self._raw["output"]
        return OutputConfig(
            strategy_filepath = section["strategy_filepath"],
            log_filepath       = section["log_filepath"],
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
                "buy_threshold":     self._config.buy_threshold,
                "sell_threshold":    self._config.sell_threshold,
                "position_size_pct": self._config.position_size_pct,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "gross_profit":         self._result.gross_profit(),
            "annualized_yield":     self._annualized_yield,
            "total_trades":         self._total_trades(),
            "win_rate":             self._win_rate(),
            "max_drawdown":         MaxDrawdown(self._result.equity_curve()).fraction(),
            "avg_profit_per_trade": self._avg_profit_per_trade(),
        }

    def _total_trades(self) -> int:
        return len(self._result.trades())

    def _win_rate(self) -> float:
        trades = self._result.trades()
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade.profit() > 0.0)
        return wins / len(trades)

    def _avg_profit_per_trade(self) -> float:
        trades = self._result.trades()
        if not trades:
            return 0.0
        return self._result.gross_profit() / len(trades)


# ── Persistence ─────────────────────────────────────────────────────────

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

class GaRunLog:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def append(self, generation: int, best_fitness: float, average_fitness: float) -> None:
        with open(self._filepath, "a") as handle:
            handle.write(f"{generation}\t{best_fitness:.6f}\t{average_fitness:.6f}\n")
