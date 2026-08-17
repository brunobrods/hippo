from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from coinbase.ga.ga_engine import Genome
from coinbase.trading_strategy import Action, Backtest, BacktestResult, Decision, Position


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    position_size_pct: float
    buy_threshold:     float
    sell_threshold:    float
    starting_balance:  float


class StrategyConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> StrategyConfig:
        section = self._raw["strategy"]
        return StrategyConfig(
            position_size_pct = section["position_size_pct"],
            buy_threshold     = section["buy_threshold"],
            sell_threshold    = section["sell_threshold"],
            starting_balance  = section["starting_balance"],
        )


class WeightKeysConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def keys(self) -> tuple[str, ...]:
        return tuple(self._raw["strategy"]["weight_keys"])


# ── GA-driven strategy ───────────────────────────────────────────────────

POSITION_PNL_KEY = "position_pnl"


class GaStrategy:
    def __init__(self, genome: Genome, config: StrategyConfig, keys: tuple[str, ...]) -> None:
        self._genome = genome
        self._config = config
        self._keys   = keys

    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        score = self._signal_score(row, position)
        if position is None and score > self._config.buy_threshold:
            size = (balance * self._config.position_size_pct) / row["close"]
            return Decision(Action.BUY, size)
        if position is not None and score < self._config.sell_threshold:
            return Decision(Action.SELL)
        return Decision(Action.HOLD)

    def _signal_score(self, row: dict[str, float], position: Optional[Position]) -> float:
        total = sum(
            self._genome.weight(key) * row[f"norm_{key}"]
            for key in self._keys if key != POSITION_PNL_KEY
        )
        if POSITION_PNL_KEY in self._keys:
            total += self._genome.weight(POSITION_PNL_KEY) * self._unrealized_return(row, position)
        return total

    def _unrealized_return(self, row: dict[str, float], position: Optional[Position]) -> float:
        if position is None:
            return 0.0
        return (row["close"] - position.entry_price()) / position.entry_price()


# ── Yield ────────────────────────────────────────────────────────────────

class AnnualizedYield:
    _SECONDS_PER_YEAR = 365.25 * 24 * 3600

    def __init__(self, gross_profit: float, starting_balance: float, duration_seconds: float) -> None:
        self._gross_profit     = gross_profit
        self._starting_balance = starting_balance
        self._duration_seconds = duration_seconds

    def value(self) -> float:
        total_return = self._gross_profit / self._starting_balance
        if self._duration_seconds <= 0.0:
            return total_return
        return (1.0 + total_return) ** (self._SECONDS_PER_YEAR / self._duration_seconds) - 1.0


# ── Evaluator ───────────────────────────────────────────────────────────

class StrategyEvaluator:
    def __init__(self, frame: pd.DataFrame, config: StrategyConfig, keys: tuple[str, ...]) -> None:
        self._frame  = frame
        self._config = config
        self._keys   = keys

    def fitness(self, genome: Genome) -> float:
        return self.annualized_yield(self.result(genome))

    def result(self, genome: Genome) -> BacktestResult:
        strategy = GaStrategy(genome, self._config, self._keys)
        return Backtest(self._frame, strategy, self._config.starting_balance).run()

    def annualized_yield(self, result: BacktestResult) -> float:
        return AnnualizedYield(result.gross_profit(), self._config.starting_balance, self._duration_seconds()).value()

    def _duration_seconds(self) -> float:
        if len(self._frame) < 2:
            return 0.0
        return float(self._frame["timestamp"].iloc[-1] - self._frame["timestamp"].iloc[0])
