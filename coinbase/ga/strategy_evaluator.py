from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from coinbase.ga.ga_engine import Genome
from coinbase.trading_strategy import Action, Backtest, BacktestResult, Decision, Direction, Position


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    position_size_pct:     float
    buy_threshold:         float
    sell_threshold:        float
    starting_balance:      float
    unwind_at_entry_price: bool  = True
    # Shorting is off unless a caller opts in: Coinbase Advanced Trade has no
    # short side, so the long-only path stays the default everywhere.
    allow_short:           bool  = False
    short_entry_threshold: float = 0.25  # score below this opens a short
    short_exit_threshold:  float = 0.40  # score above this covers it


class StrategyConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> StrategyConfig:
        section = self._raw["strategy"]
        return StrategyConfig(
            position_size_pct     = section["position_size_pct"],
            buy_threshold         = section["buy_threshold"],
            sell_threshold        = section["sell_threshold"],
            starting_balance      = section["starting_balance"],
            unwind_at_entry_price = section.get("unwind_at_entry_price", True),
            allow_short           = section.get("allow_short", False),
            short_entry_threshold = section.get("short_entry_threshold", 0.25),
            short_exit_threshold  = section.get("short_exit_threshold", 0.40),
        )


class ValidatedStrategyConfig:
    def __init__(self, config: StrategyConfig) -> None:
        self._config = config

    def config(self) -> StrategyConfig:
        for message in self._violations():
            raise ValueError(message)
        return self._config

    def _violations(self) -> list[str]:
        c = self._config
        found = []
        # 1x isolated margin: a position's notional can never exceed the quote
        # balance backing it. Above 1.0 a liquidated short loses more than the
        # whole account, driving total return below -100% — which sends
        # AnnualizedYield into a fractional power of a negative base and returns
        # a complex number that blows up the GA's fitness comparisons.
        if not 0.0 < c.position_size_pct <= 1.0:
            found.append(
                f"strategy.position_size_pct must be in (0, 1] for a 1x isolated "
                f"margin account, got {c.position_size_pct}"
            )
        if c.sell_threshold > c.buy_threshold:
            found.append(
                f"strategy.sell_threshold ({c.sell_threshold}) is above buy_threshold "
                f"({c.buy_threshold}); a long would be closed on the candle it opened"
            )
        if c.allow_short and c.short_entry_threshold >= c.buy_threshold:
            found.append(
                f"strategy.short_entry_threshold ({c.short_entry_threshold}) overlaps "
                f"buy_threshold ({c.buy_threshold}); the long band would always win"
            )
        if c.allow_short and c.short_exit_threshold < c.short_entry_threshold:
            found.append(
                f"strategy.short_exit_threshold ({c.short_exit_threshold}) is below "
                f"short_entry_threshold ({c.short_entry_threshold}); a short would be "
                f"covered on the candle it opened"
            )
        return found


class WeightKeysConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def keys(self) -> tuple[str, ...]:
        return tuple(self._raw["strategy"]["weight_keys"])


class ValidatedWeightKeys:
    def __init__(self, weight_keys: tuple[str, ...], normalized_columns: tuple[str, ...]) -> None:
        self._weight_keys        = weight_keys
        self._normalized_columns = normalized_columns

    def keys(self) -> tuple[str, ...]:
        missing = set(self._weight_keys) - set(self._normalized_columns)
        if missing:
            raise ValueError(
                f"strategy.weight_keys not in market_data.normalized_columns: {sorted(missing)}"
            )
        return self._weight_keys


# ── GA-driven strategy ───────────────────────────────────────────────────

POSITION_PNL_KEY = "position_pnl"


class GaStrategy:
    def __init__(self, genome: Genome, config: StrategyConfig, keys: tuple[str, ...]) -> None:
        self._genome = genome
        self._config = config
        self._keys   = keys

    # Three bands: a high score opens a long, a low score opens a short, and the
    # span between the two exit thresholds is the hold band. A position is only
    # ever opened from flat, so a score that crosses the whole range in one
    # candle closes the current position and leaves the reversal to the next.
    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision:
        score = self.signal_score(row, position)
        if position is None:
            if score > self._config.buy_threshold:
                return Decision(Action.BUY, self._size(balance, row))
            if self._config.allow_short and score < self._config.short_entry_threshold:
                return Decision(Action.SHORT, self._size(balance, row))
            return Decision(Action.HOLD)
        if position.direction() is Direction.LONG and score < self._config.sell_threshold:
            return Decision(Action.SELL)
        if position.direction() is Direction.SHORT and score > self._config.short_exit_threshold:
            return Decision(Action.COVER)
        return Decision(Action.HOLD)

    def _size(self, balance: float, row: dict[str, float]) -> float:
        return (balance * self._config.position_size_pct) / row["close"]

    # Public because it is the number that explains a decision: a monitor
    # showing why a strategy is holding needs the score, not just the action.
    # A pure query — asking for it never changes what decide() would return.
    def signal_score(self, row: dict[str, float], position: Optional[Position]) -> float:
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
        return position.unrealized_return(row["close"])


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
    # A genome that never opens a position realizes nothing, which scored 0.0 and
    # so ranked above every strategy that traded and lost — in a falling market
    # that made inaction the local optimum and the GA converged on it. A genome
    # can lose at most its whole balance, so a real annualized yield can never go
    # below -1.0; scoring no-trade below that floor ranks it last without
    # distorting the arithmetic the way -inf would.
    _NO_TRADE_FITNESS = -2.0

    def __init__(self, frame: pd.DataFrame, config: StrategyConfig, keys: tuple[str, ...]) -> None:
        self._frame  = frame
        self._config = config
        self._keys   = keys

    # Selection score, not a reported metric: annualized_yield() below still
    # reports the honest 0.0 for a no-trade run in the performance report.
    def fitness(self, genome: Genome) -> float:
        result = self.result(genome)
        if not result.trades():
            return self._NO_TRADE_FITNESS
        return self.annualized_yield(result)

    def result(self, genome: Genome) -> BacktestResult:
        strategy = GaStrategy(genome, self._config, self._keys)
        return Backtest(
            self._frame, strategy, self._config.starting_balance, self._config.unwind_at_entry_price,
        ).run()

    def annualized_yield(self, result: BacktestResult) -> float:
        return AnnualizedYield(result.gross_profit(), self._config.starting_balance, self._duration_seconds()).value()

    def _duration_seconds(self) -> float:
        if len(self._frame) < 2:
            return 0.0
        return float(self._frame["timestamp"].iloc[-1] - self._frame["timestamp"].iloc[0])
