import functools
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from coinbase.ga.ga_engine import Genome
from coinbase.trading_strategy import (
    Action,
    Backtest,
    BacktestResult,
    Decision,
    Direction,
    MarketRows,
    Position,
    Trade,
)


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
    # How hard the GA is penalised for an uncertain trade sample. Scales the
    # one-sided 95% Student-t bound on the per-trade edge: 1.0 applies it in
    # full, 0.0 restores the historical behaviour of scoring the realized total
    # with no regard for how few trades produced it.
    fitness_confidence:    float = 1.0
    # Fraction above (long) or below (short) entry at which a post-only limit
    # rests from the moment the position opens. 0.0 leaves every fill at a
    # close, which is what every run so far was scored under.
    take_profit_pct:       float = 0.0


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
            fitness_confidence    = float(section.get("fitness_confidence", 1.0)),
            take_profit_pct       = float(section.get("take_profit_pct", 0.0)),
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
        return total + self._negative_offset()

    # Lifts the score's floor back to zero when a genome carries negative
    # weights.
    #
    # Every norm_* column is in [0, 1] and the weights are L1-normalized, so a
    # genome's reachable span is [-M, P] where M is its negative mass and P its
    # positive mass. The thresholds (0.6 / 0.4 / 0.25) are calibrated against a
    # score that starts at 0, so without this shift a genome would be penalised
    # for using negative weights at all — its whole range would slide below the
    # buy threshold — and the search would be pushed straight back to the
    # non-negative corner this change exists to escape.
    #
    # Exactly 0.0 whenever every weight is non-negative, which is what keeps
    # every previously trained genome scoring identically.
    @functools.cached_property
    def _offset(self) -> float:
        return sum(
            max(-self._genome.weight(key), 0.0)
            for key in self._keys if key != POSITION_PNL_KEY
        )

    def _negative_offset(self) -> float:
        return self._offset

    def _unrealized_return(self, row: dict[str, float], position: Optional[Position]) -> float:
        if position is None:
            return 0.0
        return position.unrealized_return(row["close"])

    # The highest score reachable with no position open. Every norm_* column
    # is in [0, 1] and NormalizedWeights forces the weights to sum to 1.0
    # INCLUDING position_pnl — which contributes exactly 0 when flat — so the
    # ceiling is the weight mass on the indicator keys, not 1.0.
    #
    # It matters: a genome carrying 0.49 on position_pnl tops out at 0.51 and
    # can never cross a 0.6 buy_threshold, so it is structurally short-only.
    # Anything ranking scores across genomes has to measure against this rather
    # than against 1.0, or it ranks on whose position_pnl weight is smallest.
    # Absolute, because signal_score shifts the floor to zero: a genome's
    # reachable span becomes [0, sum of |weight|] over the indicator keys.
    # Identical to the signed sum whenever no weight is negative.
    def flat_score_ceiling(self) -> float:
        return sum(
            abs(self._genome.weight(key))
            for key in self._keys if key != POSITION_PNL_KEY
        )


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


# One-sided 95% Student-t multipliers by degrees of freedom.
#
# A table rather than a formula because scipy is not a dependency, and the
# only values that matter here are the small-sample ones: t(1) = 6.31 against
# t(inf) = 1.64 is exactly the penalty a two-trade sample deserves and a
# fixed z-multiplier refuses to apply.
class StudentT:
    _BY_DF = {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
        6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        12: 1.782, 15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697,
        40: 1.684, 60: 1.671, 120: 1.658,
    }
    _ASYMPTOTIC = 1.645

    def __init__(self, degrees_of_freedom: int) -> None:
        self._df = degrees_of_freedom

    def multiplier(self) -> float:
        if self._df < 1:
            return self._BY_DF[1]
        if self._df in self._BY_DF:
            return self._BY_DF[self._df]
        larger = [df for df in self._BY_DF if df > self._df]
        # Between tabulated rows, take the more conservative (larger) value
        # rather than interpolating — erring toward punishing uncertainty.
        return self._BY_DF[min(larger)] if larger else self._ASYMPTOTIC


# The per-trade returns a backtest produced, and what can honestly be inferred
# from them.
#
# This exists because annualizing a realized total treats however many trades
# happened as a repeatable rate: a genome that took two lucky trades in eleven
# months was scored as if that pace and that luck would continue all year, and
# eight such runs scored above 100% on a median of three trades. Measuring the
# LOWER BOUND on the per-trade edge instead prices the uncertainty in, so a
# small or erratic sample cannot outrank a large consistent one.
class TradeSample:
    def __init__(self, trades: list[Trade], starting_balance: float) -> None:
        self._trades           = trades
        self._starting_balance = starting_balance

    @functools.cached_property
    def returns(self) -> list[float]:
        return [trade.profit() / self._starting_balance for trade in self._trades]

    def count(self) -> int:
        return len(self._trades)

    def mean(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else 0.0

    # The observed standard error, honestly reported — zero when the sample is
    # too small to have one, or when every trade happened to return the same.
    def standard_error(self) -> float:
        n = self.count()
        if n < 2:
            return 0.0
        mean     = self.mean()
        variance = sum((value - mean) ** 2 for value in self.returns) / (n - 1)
        return (variance ** 0.5) / (n ** 0.5)

    # The observed error floored by a prior, and the floor is what makes the
    # sample size bite.
    #
    # Two trades that happen to return the same amount have zero observed
    # variance, so a bound built on the observed error alone applies no penalty
    # at all and ranks them level with a thirty-trade record. Consistency across
    # two observations is not evidence of consistency. The prior says a trade's
    # outcome is at least as uncertain as its own average magnitude — true of
    # essentially any real strategy — which restores the 1/sqrt(n) dependence
    # that small samples deserve.
    def effective_standard_error(self) -> float:
        floor = abs(self.mean()) / (self.count() ** 0.5) if self.count() else 0.0
        return max(self.standard_error(), floor)

    # `scale` of 0.0 is the historical behaviour exactly — the realized mean,
    # with no penalty for how few trades produced it.
    def lower_bound(self, scale: float) -> float:
        if scale <= 0.0:
            return self.mean()
        if self.count() == 1:
            # One trade admits no variance estimate at all. It earns no credit
            # on the upside — a single win proves nothing — while its loss
            # still counts, so inaction dressed up as one trade cannot score.
            return min(self.returns[0], 0.0)
        return self.mean() - scale * StudentT(self.count() - 1).multiplier() * self.effective_standard_error()

    # The whole-window return the lower bound implies, floored at -1.0: below
    # that, AnnualizedYield raises a negative base to a fractional power and
    # returns a complex number. A real book can only lose what it has.
    def pessimistic_return(self, scale: float) -> float:
        return max(self.lower_bound(scale) * self.count(), -1.0)


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

    # One conversion of the window shared by every genome scored against it.
    # Backtest is rebuilt per genome, so this cannot live there — see MarketRows.
    @functools.cached_property
    def _rows(self) -> MarketRows:
        return MarketRows(self._frame)

    # Selection score, not a reported metric: annualized_yield() below still
    # reports the realized figure, so index.csv stays comparable across every
    # run ever recorded even as what the GA optimizes changes.
    #
    # Scored on the lower bound of the per-trade edge rather than the realized
    # total, because annualizing the realized total let a two-trade sample be
    # graded as a yearly rate. A genome now has to earn its yield often enough
    # and consistently enough for the bound to survive.
    def fitness(self, genome: Genome) -> float:
        result = self.result(genome)
        sample = TradeSample(result.trades(), self._config.starting_balance)
        if sample.count() == 0:
            return self._NO_TRADE_FITNESS
        pessimistic = sample.pessimistic_return(self._config.fitness_confidence)
        return AnnualizedYield(
            pessimistic * self._config.starting_balance,
            self._config.starting_balance,
            self._duration_seconds(),
        ).value()

    def result(self, genome: Genome) -> BacktestResult:
        strategy = GaStrategy(genome, self._config, self._keys)
        return Backtest(
            self._rows, strategy, self._config.starting_balance,
            self._config.unwind_at_entry_price, self._config.take_profit_pct,
        ).run()

    def annualized_yield(self, result: BacktestResult) -> float:
        return AnnualizedYield(result.gross_profit(), self._config.starting_balance, self._duration_seconds()).value()

    def _duration_seconds(self) -> float:
        if len(self._frame) < 2:
            return 0.0
        return float(self._frame["timestamp"].iloc[-1] - self._frame["timestamp"].iloc[0])
