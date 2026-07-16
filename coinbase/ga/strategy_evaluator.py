import functools
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from coinbase.ga.ga_engine import Genome, WEIGHT_KEYS


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


# ── Signal scoring ─────────────────────────────────────────────────────

class SignalScores:
    def __init__(self, frame: pd.DataFrame, genome: Genome, keys: tuple[str, ...] = WEIGHT_KEYS) -> None:
        self._frame  = frame
        self._genome = genome
        self._keys   = keys

    @functools.cached_property
    def series(self) -> pd.Series:
        total = pd.Series(0.0, index=self._frame.index)
        for key in self._keys:
            total = total + self._genome.weight(key) * self._frame[f"norm_{key}"]
        return total


# ── Trades ──────────────────────────────────────────────────────────────

class Trade:
    def __init__(self, entry_price: float, exit_price: float, size: float) -> None:
        self._entry_price = entry_price
        self._exit_price  = exit_price
        self._size        = size

    def profit(self) -> float:
        return (self._exit_price - self._entry_price) * self._size


class OpenPosition:
    def __init__(self, entry_price: float, size: float) -> None:
        self._entry_price = entry_price
        self._size        = size

    def closed(self, exit_price: float) -> Trade:
        return Trade(self._entry_price, exit_price, self._size)

    def unrealized(self, price: float) -> float:
        return (price - self._entry_price) * self._size


# ── Backtest ────────────────────────────────────────────────────────────

class BacktestResult:
    def __init__(self, trades: list[Trade], equity_curve: list[float]) -> None:
        self._trades       = trades
        self._equity_curve = equity_curve

    def trades(self) -> list[Trade]:
        return list(self._trades)

    def gross_profit(self) -> float:
        return sum(trade.profit() for trade in self._trades)

    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)


class Backtest:
    def __init__(self, frame: pd.DataFrame, signal_scores: pd.Series, config: StrategyConfig) -> None:
        self._frame         = frame
        self._signal_scores = signal_scores
        self._config        = config

    def run(self) -> BacktestResult:
        closes       = self._frame["close"].tolist()
        scores       = self._signal_scores.tolist()
        balance      = self._config.starting_balance
        position: Optional[OpenPosition] = None
        trades:       list[Trade] = []
        equity_curve: list[float] = []

        for price, score in zip(closes, scores):
            if position is None and score > self._config.buy_threshold:
                position = OpenPosition(price, (balance * self._config.position_size_pct) / price)
            elif position is not None and score < self._config.sell_threshold:
                trade    = position.closed(price)
                balance += trade.profit()
                trades.append(trade)
                position = None
            equity_curve.append(balance + (position.unrealized(price) if position is not None else 0.0))

        if position is not None:
            trades.append(position.closed(closes[-1]))

        return BacktestResult(trades, equity_curve)


# ── Evaluator ───────────────────────────────────────────────────────────

class StrategyEvaluator:
    def __init__(self, frame: pd.DataFrame, config: StrategyConfig, keys: tuple[str, ...] = WEIGHT_KEYS) -> None:
        self._frame  = frame
        self._config = config
        self._keys   = keys

    def fitness(self, genome: Genome) -> float:
        return self.result(genome).gross_profit()

    def result(self, genome: Genome) -> BacktestResult:
        scores = SignalScores(self._frame, genome, self._keys).series
        return Backtest(self._frame, scores, self._config).run()
