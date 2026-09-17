import functools
import math
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd

from coinbase.ga.ga_engine import Genome, GroupedL1Scaling, L1Scaling, WeightScaling
from coinbase.trading_strategy import (
    Action,
    AtrTakeProfit,
    Backtest,
    BacktestResult,
    ConfiguredBorrowRate,
    ConfiguredFees,
    Decision,
    Direction,
    FixedTakeProfit,
    MarketRows,
    Position,
    TakeProfitTarget,
    Trade,
)

# The weighted sum every run so far was trained under. Named here, above the
# config that defaults to it, because a design name is part of a strategy's
# identity — see SignalDesign below.
LINEAR_DESIGN = "linear"

# Two weight vectors in one genome: one scored while flat, one while holding.
# See DualSignal for why the split exists.
DUAL_DESIGN = "dual"

# How a dual genome's keys name their model. "entry:rsi" and "exit:rsi" are two
# independent weights on the same column, and everything that operates on a
# genome as a flat dict stays unaware of the distinction.
GROUP_SEPARATOR = ":"
ENTRY_PREFIX    = "entry"
EXIT_PREFIX     = "exit"


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
    # Trading costs, both zero by default so an existing config reproduces the
    # numbers it always produced. fee_bps is charged per leg on the notional
    # that changed hands; borrow_bps_per_hour accrues only against a short, for
    # the time it is held (a long borrows nothing on either venue).
    fee_bps:               float = 0.0
    borrow_bps_per_hour:   float = 0.0
    # Which model design turns a candle into a score. "linear" is the weighted
    # sum every run so far was trained under, and is the default so an existing
    # config and an existing strategy.json both keep meaning what they meant.
    design:                str = LINEAR_DESIGN
    # Fraction above (long) or below (short) entry at which a post-only limit
    # rests from the moment the position opens. 0.0 leaves every fill at a
    # close, which is what every run so far was scored under.
    take_profit_pct:       float = 0.0
    # The same resting target, expressed in units of the pair's own volatility:
    # the order rests at this multiple of atr_pct away from entry, read on the
    # candle the position opened. 0.0 leaves take_profit_pct in charge.
    #
    # It exists because a fixed percentage is not one parameter but twenty. The
    # best level measured between 0% and beyond 20% depending on the pair, and
    # a level below a pair's own 10th percentile of reachable move fires on
    # nearly every hold — the weights stop mattering — while one above its 90th
    # never fires at all. A multiple of atr_pct sits at the same place in every
    # pair's distribution, so one value means the same thing on all of them.
    #
    # Saved with the genome for the reason take_profit_pct is: the target
    # decides when a position closes, so a genome papered without it trades a
    # strategy nobody scored.
    take_profit_atr_mult:  float = 0.0
    # Which columns the EXIT model of a dual genome scores. Ignored by linear.
    #
    # MUST NAME AT LEAST ONE COLUMN. Left empty, the exit group holds only
    # position_pnl, and GroupedL1Scaling pins a one-key group at exactly 1.0 —
    # so the exit model has no free parameters and the GA cannot train it. That
    # shipped briefly and every long was sold on the candle after it opened.
    #
    # Carried on the config rather than read from config.yaml at load time so
    # it is SAVED WITH THE GENOME: it decides the genome's shape, and a dual
    # genome rebuilt against a different exit_keys would be a different model
    # wearing the same weights. That is the drift TrainedStrategyConfig exists
    # to prevent, and the same gap take_profit_pct had before PR #20.
    exit_keys:             tuple[str, ...] = ()
    # The unrealized move at which the dual exit model's position_pnl term
    # reads 0.88 (ahead) or 0.12 (behind) — see ScaledReturn. Saved with the
    # genome for the same reason exit_keys is: it changes what the model
    # computes, so a genome rebuilt at a different scale is a different model.
    #
    # 0.02 was a guess and it measured badly. At SIX_HOUR a candle moves about
    # 1%, so a 2% half-point puts the exit score across sell_threshold on
    # ordinary noise — the corrected dual exit held a median of TWO candles
    # against linear's forty-eight. Sweep it rather than assuming it.
    exit_pnl_scale:        float = 0.02
    # Which tail of the exit model's own birth distribution counts as "leave".
    # 0.10 means a long closes when the exit score is in the bottom tenth of
    # what that model reads across the window, and a short when it is in the
    # top tenth. Only the dual design uses it — see CalibratedExit.
    exit_quantile:         float = 0.10


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
            fee_bps               = float(section.get("fee_bps", 0.0)),
            borrow_bps_per_hour   = float(section.get("borrow_bps_per_hour", 0.0)),
            design                = str(section.get("design", LINEAR_DESIGN)),
            take_profit_pct       = float(section.get("take_profit_pct", 0.0)),
            take_profit_atr_mult  = float(section.get("take_profit_atr_mult", 0.0)),
            exit_keys             = tuple(section.get("exit_keys") or ()),
            exit_pnl_scale        = float(section.get("exit_pnl_scale", 0.02)),
            exit_quantile         = float(section.get("exit_quantile", 0.10)),
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
        # A negative rate would be read as "no cost" by ConfiguredFees /
        # ConfiguredBorrowRate, so a sign typo would silently score a run free
        # of the very costs it was configured to pay.
        if c.fee_bps < 0.0:
            found.append(f"strategy.fee_bps must not be negative, got {c.fee_bps}")
        if c.borrow_bps_per_hour < 0.0:
            found.append(
                f"strategy.borrow_bps_per_hour must not be negative, got {c.borrow_bps_per_hour}"
            )
        if c.take_profit_atr_mult < 0.0:
            found.append(
                f"strategy.take_profit_atr_mult must not be negative, got "
                f"{c.take_profit_atr_mult}"
            )
        # One position, one resting order. Honouring both would mean two orders
        # on the same side at different prices, and silently preferring one
        # would score a strategy nobody configured.
        if c.take_profit_pct > 0.0 and c.take_profit_atr_mult > 0.0:
            found.append(
                f"strategy.take_profit_pct ({c.take_profit_pct}) and "
                f"strategy.take_profit_atr_mult ({c.take_profit_atr_mult}) are both "
                f"set; a position rests ONE order, so set whichever target you mean "
                f"and leave the other at 0.0"
            )
        if not 0.0 < c.exit_quantile < 0.5:
            found.append(
                f"strategy.exit_quantile must be in (0, 0.5) — it names a tail "
                f"of the exit model's own distribution, got {c.exit_quantile}"
            )
        # A zero or negative scale divides by zero or inverts the term, and the
        # failure would surface as a strategy that never exits rather than as
        # an error.
        if c.exit_pnl_scale <= 0.0:
            found.append(
                f"strategy.exit_pnl_scale must be positive, got {c.exit_pnl_scale}"
            )
        # Checked here so a misspelled design fails before a multi-hour training
        # run, not after it — the same reason ExperimentIndex is checked up front.
        if c.design not in (LINEAR_DESIGN, DUAL_DESIGN):
            found.append(
                f"strategy.design must be one of {LINEAR_DESIGN!r}, "
                f"{DUAL_DESIGN!r}; got {c.design!r}"
            )
        return found


class WeightKeysConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def keys(self) -> tuple[str, ...]:
        return tuple(self._raw["strategy"]["weight_keys"])


# exit_keys lives on StrategyConfig rather than in a config object of its own,
# unlike weight_keys, because it has to be SAVED WITH THE GENOME — see the note
# on the field. Read it from the strategy config everywhere.


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


# ── Signal models ────────────────────────────────────────────────────────
# A design is the FUNCTION that turns a candle into a score. It is deliberately
# separate from the policy that acts on that score (GaStrategy below: the three
# bands, sizing, the thresholds), because those are what every recorded run and
# every paper book are calibrated against, and they should not have to change
# for a new kind of model to exist.
#
# A design also owns how the GA rescales its numbers — see WeightScaling in
# ga_engine — since L1 scaling is a statement about a weighted sum, not about
# search.
#
# LINEAR is the only design today. It is named rather than assumed so that a
# saved strategy.json records what produced it, and so a second design becomes
# an addition rather than an edit.

class SignalModel(Protocol):
    def score(self, row: dict[str, float], position: Optional[Position]) -> float: ...

    # Every design must be able to say how high it can score while flat, because
    # the cross-sectional selector ranks conviction across genomes and cannot do
    # that against a fixed 1.0 — see the note on LinearSignal's implementation.
    def flat_score_ceiling(self) -> float: ...


class LinearSignal:
    def __init__(self, genome: Genome, keys: tuple[str, ...]) -> None:
        self._genome = genome
        self._keys   = keys

    def score(self, row: dict[str, float], position: Optional[Position]) -> float:
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

    # The highest score reachable with no position open. Every norm_* column is
    # in [0, 1] and NormalizedWeights forces the weights to sum to 1.0
    # INCLUDING position_pnl — which contributes exactly 0 when flat — so the
    # ceiling is the weight mass on the indicator keys, not 1.0.
    #
    # It matters: a genome carrying 0.49 on position_pnl tops out at 0.51 and
    # can never cross a 0.6 buy_threshold, so it is structurally short-only.
    # Anything ranking scores across genomes has to measure against this rather
    # than against 1.0, or it ranks on whose position_pnl weight is smallest.
    # Absolute, because score() shifts the floor to zero: a genome's reachable
    # span becomes [0, sum of |weight|] over the indicator keys. Identical to
    # the signed sum whenever no weight is negative.
    def flat_score_ceiling(self) -> float:
        return sum(
            abs(self._genome.weight(key))
            for key in self._keys if key != POSITION_PNL_KEY
        )


# An open position's unrealized return, squashed onto the [0, 1] scale every
# other scored column lives on.
#
# Without this the dual exit model compares a raw fractional return against
# thresholds meant for a min-max column, and the mismatch is not subtle: a
# LONG closes when the score drops under sell_threshold 0.40, so it would need
# to be +40% ahead merely to be held one more candle, while a SHORT needs +45%
# to ever be covered. Measured: every long sold on the candle after it opened.
#
# tanh rather than a clamp because an exit rule cares most about small moves
# and should saturate on large ones — the difference between +1% and +2% ahead
# should move the score, the difference between +40% and +50% should not. HALF
# is the return at which the score reaches roughly 0.88 or 0.12; 2% is about a
# fifth of a typical eight-day move on BTC, so the usable band covers the range
# an exit decision is actually taken over.
#
# Flat P&L maps to exactly 0.5, so a position that has gone nowhere reads
# neutral rather than reading as a reason to leave.
#
# LinearSignal deliberately does NOT use this. Every genome ever trained under
# that design was scored on the raw return, and rescaling it now would silently
# change what all of them mean.
class ScaledReturn:
    def __init__(self, unrealized_return: float, half: float) -> None:
        self._value = unrealized_return
        self._half  = half

    def value(self) -> float:
        return 0.5 + 0.5 * math.tanh(self._value / self._half)


# Two models in one genome: one scored while FLAT, deciding whether to enter,
# and one scored while IN POSITION, deciding whether to leave.
#
# The linear design uses a single weight vector for both jobs, which is where
# its worst structural failure comes from. position_pnl is useless for entry —
# it contributes exactly zero while flat — but under flat L1 the weight it
# carries still comes out of the same budget of 1.0, lowering the highest score
# the entry side can reach. A genome holding 0.48 on position_pnl tops out at
# 0.52 against a buy_threshold of 0.60 and can only ever short. That happened
# live, and TwoSidedModel exists to catch it.
#
# Here each model owns its own budget, normalized separately by
# GroupedL1Scaling, so BOTH scores span the full [0, 1] and the thresholds mean
# what they were calibrated to mean on either side. flat_score_ceiling() is 1.0
# by construction rather than by luck.
#
# Keys are prefixed: "entry:rsi" and "exit:rsi" are two independent weights on
# the same column. Every genome operator — crossover, mutation, backfill —
# works on a flat dict and is untouched by this; only the scaling and this
# model know the groups exist.
#
# The exit model is deliberately allowed to be SMALLER than the entry model.
# Eleven columns measured worse and noisier than two at this data size, so
# doubling a seven-column genome would likely spend the gain on search
# difficulty. An exit rule mostly needs to know its own P&L and whether the
# short-horizon move turned.
class DualSignal:
    def __init__(self, genome: Genome, keys: tuple[str, ...], exit_pnl_scale: float) -> None:
        self._genome         = genome
        self._keys           = keys
        self._exit_pnl_scale = exit_pnl_scale

    def score(self, row: dict[str, float], position: Optional[Position]) -> float:
        if position is None:
            return self._sum(self._entry_keys, row, 0.0) + self._offset(self._entry_keys)
        return (
            self._sum(self._exit_keys, row, self._move(row, position))
            + self._offset(self._exit_keys)
        )

    # The UNDIRECTED price move since entry, not Position.unrealized_return.
    #
    # That distinction decides whether position_pnl acts as a stop loss or as
    # its opposite. unrealized_return is direction-agnostic — positive means
    # "ahead" for a long and a short alike — but the policy's two exit bands
    # are not symmetric: a long closes when the score falls BELOW
    # sell_threshold, a short when it rises ABOVE short_exit_threshold. Feeding
    # both the same "am I ahead" number therefore means opposite things:
    #
    #   long  ahead -> high score -> held      (a winner runs)
    #   short ahead -> high score -> COVERED   (a winner cut)
    #
    # The exit score is a statement about the MARKET, not about the position's
    # profit, so the term that belongs in it is where price has gone. Then a
    # rising price holds a long and covers a short, which is a stop loss on both
    # sides and lets both winners run.
    #
    # LinearSignal keeps unrealized_return. Every genome trained under it was
    # scored that way, and changing it would silently redefine all of them.
    def _move(self, row: dict[str, float], position: Position) -> float:
        signed = position.unrealized_return(row["close"])
        return signed if position.direction() is Direction.LONG else -signed

    # The entry model never sees position_pnl, so its whole weight mass is
    # reachable from flat: 1.0 after grouped scaling.
    def flat_score_ceiling(self) -> float:
        return sum(abs(self._genome.weight(key)) for key in self._entry_keys)

    @functools.cached_property
    def _entry_keys(self) -> tuple[str, ...]:
        return self._group(ENTRY_PREFIX)

    @functools.cached_property
    def _exit_keys(self) -> tuple[str, ...]:
        return self._group(EXIT_PREFIX)

    def _group(self, prefix: str) -> tuple[str, ...]:
        return tuple(k for k in self._keys if k.startswith(f"{prefix}{GROUP_SEPARATOR}"))

    def _sum(self, keys: tuple[str, ...], row: dict[str, float], pnl: float) -> float:
        return sum(
            self._genome.weight(key) * self._reading(key, row, pnl) for key in keys
        )

    def _reading(self, key: str, row: dict[str, float], pnl: float) -> float:
        column = key.split(GROUP_SEPARATOR, 1)[1]
        if column == POSITION_PNL_KEY:
            return ScaledReturn(pnl, self._exit_pnl_scale).value()
        return row[f"norm_{column}"]

    # Lifts a signed model's floor back to zero, per group, for the same reason
    # LinearSignal does it: the thresholds are calibrated against a score that
    # starts at 0, and without the shift a genome would be penalised merely for
    # using a negative weight. Exactly 0.0 when every weight in the group is
    # non-negative, which is what keeps a non-negative genome scoring identically.
    def _offset(self, keys: tuple[str, ...]) -> float:
        return sum(max(-self._genome.weight(key), 0.0) for key in keys)


# Exit thresholds set from the exit model's OWN score distribution, rather than
# from numbers calibrated against a different model.
#
# This exists because splitting entry and exit broke a guarantee the linear
# design gets for free. Under linear, one score decides both: a long opens only
# above buy_threshold 0.60 and closes when THAT SAME score falls to 0.40, so a
# position is born far from its own exit by construction. Measured over 467
# linear entries, exactly 0% were already past their exit threshold on the
# candle they opened.
#
# A dual genome's exit model is a different function, and its value at entry is
# arbitrary — it clusters near 0.5, and 21% to 31% of positions were born
# ALREADY past the threshold that closes them. They were doomed before their
# first candle, which is why the median hold was two candles at every
# exit_pnl_scale tried: the problem is where the exit score STARTS, not how
# fast it moves.
#
# So the thresholds become quantiles of the model's own birth distribution.
# "Exit" then means "unusually bearish for this model" rather than "below 0.40",
# which is what it already means under linear. The distribution is measured with
# the position flat — position_pnl reads exactly 0.5 at zero unrealized move —
# because that is the score every position is born with.
#
# Calibrated per genome and per window, so it has to be recomputed wherever a
# genome is scored. It is saved with the genome for the same reason exit_keys
# and exit_pnl_scale are: a dual genome rehydrated against another genome's
# thresholds is a different strategy wearing the same weights.
class CalibratedExit:
    def __init__(
        self,
        model: SignalModel,
        frame: pd.DataFrame,
        config: StrategyConfig,
    ) -> None:
        self._model  = model
        self._frame  = frame
        self._config = config

    # Unchanged for linear, which needs no calibration and whose every recorded
    # run must keep reproducing its own numbers.
    def config(self) -> StrategyConfig:
        if self._config.design != DUAL_DESIGN:
            return self._config
        low, high = self._quantiles()
        return dataclasses.replace(
            self._config, sell_threshold=low, short_exit_threshold=high,
        )

    @functools.cached_property
    def _birth_scores(self) -> "pd.Series":
        # A position opened at this row's own close has zero unrealized move,
        # so this is the exit score each candle would hand a new position.
        return pd.Series([
            self._model.score(row, Position(row["close"], 1.0, Direction.LONG))
            for row in self._frame.to_dict("records")
        ])

    def _quantiles(self) -> tuple[float, float]:
        scores = self._birth_scores
        low  = float(scores.quantile(self._config.exit_quantile))
        high = float(scores.quantile(1.0 - self._config.exit_quantile))
        # ValidatedStrategyConfig requires sell <= buy and cover >= short_entry,
        # and a degenerate window could produce a flat distribution that breaks
        # either. Clamping keeps a pathological genome scoreable rather than
        # aborting a whole sweep on one point.
        return (
            min(low, self._config.buy_threshold),
            max(high, self._config.short_entry_threshold),
        )


# The one place that maps a design NAME to its objects. Every caller that
# rebuilds a strategy from a saved genome goes through here, so an unknown
# design fails loudly at the single point that knows the list, rather than
# scoring silently through the wrong function.
class SignalDesign:
    def __init__(self, name: str) -> None:
        self._name = name

    # exit_pnl_scale is REQUIRED rather than defaulted. It is ignored by the
    # linear design, but a default here would be silently wrong for any caller
    # that forgot to thread it through — which is exactly how a genome came to
    # be rebuilt against the wrong exit_keys.
    def model(self, genome: Genome, keys: tuple[str, ...], exit_pnl_scale: float) -> SignalModel:
        if self._name == LINEAR_DESIGN:
            return LinearSignal(genome, keys)
        if self._name == DUAL_DESIGN:
            return DualSignal(genome, keys, exit_pnl_scale)
        raise ValueError(self._unknown())

    def scaling(self) -> WeightScaling:
        if self._name == LINEAR_DESIGN:
            return L1Scaling()
        if self._name == DUAL_DESIGN:
            return GroupedL1Scaling(GROUP_SEPARATOR)
        raise ValueError(self._unknown())

    # What a genome of this design is made of. The design owns this because the
    # SHAPE of a genome is part of the model, not of the caller: linear takes
    # one weight per column plus position_pnl, while dual takes two prefixed
    # groups. Every caller that builds a genome's key list goes through here,
    # so adding a design does not mean finding five places that assumed linear.
    def keys(self, weight_keys: tuple[str, ...], exit_keys: tuple[str, ...] = ()) -> tuple[str, ...]:
        if self._name == LINEAR_DESIGN:
            return weight_keys + (POSITION_PNL_KEY,)
        if self._name == DUAL_DESIGN:
            return (
                tuple(f"{ENTRY_PREFIX}{GROUP_SEPARATOR}{k}" for k in weight_keys)
                + tuple(f"{EXIT_PREFIX}{GROUP_SEPARATOR}{k}" for k in exit_keys)
                + (f"{EXIT_PREFIX}{GROUP_SEPARATOR}{POSITION_PNL_KEY}",)
            )
        raise ValueError(self._unknown())

    def _unknown(self) -> str:
        return (
            f"unknown strategy.design {self._name!r}; this build knows "
            f"{LINEAR_DESIGN!r} and {DUAL_DESIGN!r}. A genome trained under a "
            f"design this code does not have would be scored by the wrong "
            f"function."
        )


# A genome that cannot reach its own buy_threshold while flat, and so can only
# ever short.
#
# Every norm_* column is in [0, 1] and L1 scaling makes the weights sum to 1
# INCLUDING position_pnl — which contributes exactly 0 while no position is
# open. So the highest score reachable from flat is the weight mass on the
# indicator keys alone, and a genome carrying 0.48 on position_pnl tops out at
# 0.52 against a buy_threshold of 0.60. It is structurally short-only, and
# nothing about the score says so: it simply never crosses.
#
# This was found live. The papered ETH book had a ceiling of 0.5196 and had
# never been able to open a long in its entire history — not a bad genome, a
# single L1 budget of 1.0 being shared between an entry model and an exit model
# when only the exit half can use position_pnl.
#
# Raised rather than warned, following SignalDesign: a genome that cannot take
# one of the two sides it was scored on is not the strategy anybody chose, and
# a book running it reports on a strategy that was never tested. Training warns
# instead — there a one-sided genome is a measurement, not something about to
# trade.
class TwoSidedModel:
    def __init__(self, model: SignalModel, config: StrategyConfig) -> None:
        self._model  = model
        self._config = config

    def model(self) -> SignalModel:
        ceiling = self._model.flat_score_ceiling()
        if ceiling <= self._config.buy_threshold:
            raise ValueError(self._complaint(ceiling))
        return self._model

    def is_one_sided(self) -> bool:
        return self._model.flat_score_ceiling() <= self._config.buy_threshold

    def _complaint(self, ceiling: float) -> str:
        return (
            f"genome can never open a long: its highest score while flat is "
            f"{ceiling:.4f}, at or below buy_threshold "
            f"{self._config.buy_threshold}. Weight on position_pnl scores only "
            f"once a position is open, so it lowers this ceiling without ever "
            f"raising the score that has to cross it. Retrain, or lower "
            f"buy_threshold below {ceiling:.4f}."
        )


# ── GA-driven strategy ───────────────────────────────────────────────────

class GaStrategy:
    def __init__(self, model: SignalModel, config: StrategyConfig) -> None:
        self._model  = model
        self._config = config

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
        return self._model.score(row, position)


    def flat_score_ceiling(self) -> float:
        return self._model.flat_score_ceiling()


# ── Yield ────────────────────────────────────────────────────────────────

class AnnualizedYield:
    _SECONDS_PER_YEAR = 365.25 * 24 * 3600

    # `profit` rather than `gross_profit`: the caller decides whether costs are
    # already taken out, and StrategyEvaluator now hands it the net figure.
    def __init__(self, profit: float, starting_balance: float, duration_seconds: float) -> None:
        self._profit           = profit
        self._starting_balance = starting_balance
        self._duration_seconds = duration_seconds

    def value(self) -> float:
        # Floored at a total loss. Gross profit could never fall below
        # -starting_balance (position_size_pct <= 1.0 bounds it), but fees and
        # interest are charged ON TOP of the price loss, so a liquidated short
        # that then pays its exit fee and its accrued interest can leave the
        # balance negative. Below -1.0 the base of the power is negative and
        # Python returns a COMPLEX number, which makes every fitness comparison
        # in the GA raise TypeError and kills a multi-hour training run.
        total_return = max(-1.0, self._profit / self._starting_balance)
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

    # Net, so the edge the GA is asked to prove is the one that survives its own
    # costs: a genome churning for a spread thinner than its fees has no edge to
    # bound, however consistent the gross figure looks. With both rates at their
    # 0.0 default this is the gross number it has always been.
    @functools.cached_property
    def returns(self) -> list[float]:
        return [trade.net_profit() / self._starting_balance for trade in self._trades]

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

    # The thresholds this genome is scored under. For linear it is the config
    # unchanged; for dual the exit bands are quantiles of that genome's own
    # exit-score distribution — see CalibratedExit. Public because the training
    # run has to SAVE these alongside the genome: scoring it here under one set
    # of thresholds and papering it under another is the drift
    # TrainedStrategyConfig exists to prevent.
    def calibrated(self, genome: Genome) -> StrategyConfig:
        return CalibratedExit(self._model(genome), self._frame, self._config).config()

    def _model(self, genome: Genome) -> SignalModel:
        return SignalDesign(self._config.design).model(
            genome, self._keys, self._config.exit_pnl_scale,
        )

    def result(self, genome: Genome) -> BacktestResult:
        model    = self._model(genome)
        strategy = GaStrategy(model, self.calibrated(genome))
        return Backtest(
            self._rows, strategy, self._config.starting_balance, self._config.unwind_at_entry_price,
            ConfiguredFees(self._config.fee_bps).schedule(),
            ConfiguredBorrowRate(self._config.borrow_bps_per_hour).rate(),
            self._target(),
        ).run()

    # Which resting order this run places. Validation has already ruled out
    # both being set, so the volatility-scaled one wins when it is present and
    # a config that sets neither gets FixedTakeProfit(0.0) — no resting order,
    # every fill at a close, exactly as before either knob existed.
    def _target(self) -> TakeProfitTarget:
        if self._config.take_profit_atr_mult > 0.0:
            return AtrTakeProfit(self._config.take_profit_atr_mult)
        return FixedTakeProfit(self._config.take_profit_pct)

    # Net, so the GA pays for the trading it does: a genome that churns for a
    # thin edge now scores below one that waits for a wide one. With both rates
    # at their 0.0 default this is identical to the gross figure it used to be.
    def annualized_yield(self, result: BacktestResult) -> float:
        return AnnualizedYield(result.net_profit(), self._config.starting_balance, self._duration_seconds()).value()

    def _duration_seconds(self) -> float:
        if len(self._frame) < 2:
            return 0.0
        return float(self._frame["timestamp"].iloc[-1] - self._frame["timestamp"].iloc[0])
