"""
Cross-sectional pair selection — one book, many candidate pairs.
----------------------------------------------------------------

Every trained genome answers "should I be long, short or flat in MY pair?".
This asks the question one level up: given N genomes each watching its own
pair, which pair is worth holding right now, and in which direction?

One position at a time, taken in whichever pair has the strongest conviction,
held until that pair's own exit threshold fires. No rotation mid-hold — a
round trip costs real basis points and rotating on every flicker of relative
strength spends the edge it is chasing.

Two things here are not in `Backtest`, deliberately:

  Fees are charged unconditionally.  `Ledger` can now charge them too, but
  only when `strategy.fee_bps` is set, and it defaults to 0.0 — so a genome's
  recorded performance is usually still gross. A selector that re-enters on
  every exit trades far more than a single-pair strategy, so the fee drag is
  the difference between a result and a fantasy, and it is not left optional
  here.

  Direction accuracy is measured.  It is the number the screener's
  `required_accuracy` column is a bar for, and the only honest way to ask
  whether the selection has any skill: `direction_accuracy` counts trades
  where price moved the way the genome bet, GROSS of costs. Compare it against
  `required_accuracy` for the same pairs — if it is not comfortably above,
  the strategy does not clear its own cost floor and no amount of sizing fixes
  that.

Known limit, stated because it bounds what a good result here would mean:
`signal_score` is a weighted sum of `norm_*` columns min-max scaled within
each pair's OWN window, so a raw 0.8 on one pair and 0.8 on another are not
the same quantity. `Conviction` normalizes each score against its own
genome's trigger to make them roughly commensurable, but "roughly" is the
honest word. Ranking across pairs is the weakest link in this module.

Usage (run as a module, from the repo root):
    python -m coinbase.ga.pair_selector --from-shortlist
    python -m coinbase.ga.pair_selector --pairs FET-USDT ETH-USDT --fee-bps 10
    python -m coinbase.ga.pair_selector --pairs FET-USDT --start 2025-01-01
"""

import argparse
import asyncio
import functools
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from coinbase.ga.config import GA_RESULTS_ROOT, ConfigFile
from coinbase.ga.experiment_history import ExperimentDirectory
from coinbase.ga.ga_engine import BackfilledGenome, Genome
from coinbase.ga.market_data_processor import (
    HistoricalMarketData,
    IsoDate,
    MarketBasket,
    MarketDataConfig,
)
from coinbase.ga.paper_trading import BasisPointFee, FeeSchedule, NoFees
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    SignalDesign,
    GaStrategy,
    StrategyConfig,
    WeightKeysConfig,
    ValidatedWeightKeys,
)
from coinbase.ga.strategy_output import OutputConfigFile, StrategyJsonFile
from coinbase.ga.paper_trading import TrainedStrategyConfig
from coinbase.trading_strategy import Action, Decision, Direction, Ledger, Position, Trade
from exchange.pool import ExchangePool

logger = logging.getLogger(__name__)


# ── Conviction ─────────────────────────────────────────────────────────

# How far past its own trigger a score sits, as a fraction of the room it had
# left to travel. Raw scores cannot be compared between pairs — each is a sum
# of columns min-max scaled inside that pair's own window, and each genome
# carries its own thresholds — so ranking on the raw number would mostly rank
# on whose normalization happened to run hot. Measuring each score against its
# own genome's bar removes both differences, as far as anything can.
class Conviction:
    # `ceiling` is the genome's own highest reachable flat score, NOT 1.0.
    # Weights sum to 1.0 including position_pnl, which contributes 0 with no
    # position open, so a genome carrying 0.37 there tops out at 0.63. Dividing
    # long excess by (1 - buy_threshold) instead of the real room above the bar
    # made conviction a proxy for whose position_pnl weight was smallest: on
    # the first live run a maximally-bullish AVAX ranked 0.081, losing to any
    # short that breached its trigger by 8%, and 8 of 12 entries were shorts.
    def __init__(self, score: float, config: StrategyConfig, ceiling: float = 1.0) -> None:
        self._score   = score
        self._config  = config
        self._ceiling = ceiling

    def action(self) -> Action:
        if self._score > self._config.buy_threshold:
            return Action.BUY
        if self._config.allow_short and self._score < self._config.short_entry_threshold:
            return Action.SHORT
        return Action.HOLD

    def value(self) -> float:
        action = self.action()
        if action is Action.BUY:
            return self._fraction(self._score - self._config.buy_threshold,
                                  self._ceiling - self._config.buy_threshold)
        # The score floor genuinely is 0 — every norm_* column can be 0 at once
        # — so the short denominator needs no such correction.
        if action is Action.SHORT:
            return self._fraction(self._config.short_entry_threshold - self._score,
                                  self._config.short_entry_threshold)
        return 0.0

    # A genome whose ceiling sits at or below its own buy_threshold cannot ever
    # open a long, whatever the market does. Worth saying out loud rather than
    # letting it silently vanish from one side of the ranking.
    def can_ever_go_long(self) -> bool:
        return self._ceiling > self._config.buy_threshold

    # A threshold pinned at the end of the range leaves no room to be "far
    # past" it; treat any breach as full conviction rather than dividing by ~0.
    @staticmethod
    def _fraction(excess: float, room: float) -> float:
        if room <= 0.0:
            return 1.0
        return min(excess / room, 1.0)


@dataclass(frozen=True)
class Candidate:
    pair:       str
    action:     Action
    conviction: float
    score:      float


class PairRanking:
    def __init__(self, candidates: tuple[Candidate, ...]) -> None:
        self._candidates = candidates

    def ranked(self) -> tuple[Candidate, ...]:
        return tuple(sorted(
            (c for c in self._candidates if c.action is not Action.HOLD),
            key=lambda c: c.conviction, reverse=True,
        ))

    def best(self) -> Optional[Candidate]:
        ranked = self.ranked()
        return ranked[0] if ranked else None


# ── Aligned frames ─────────────────────────────────────────────────────

# Cross-sectional selection needs one clock. Pairs list at different dates and
# drop candles independently, so the index is the INTERSECTION of every pair's
# timestamps — comparing a score computed on Tuesday against one from Monday
# would rank on staleness rather than strength.
class AlignedFrames:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def pairs(self) -> tuple[str, ...]:
        return tuple(self._frames)

    @functools.cached_property
    def timestamps(self) -> tuple[int, ...]:
        if not self._frames:
            return ()
        common: Optional[set[int]] = None
        for frame in self._frames.values():
            stamps = set(int(value) for value in frame["timestamp"])
            common = stamps if common is None else (common & stamps)
        return tuple(sorted(common or ()))

    @functools.cached_property
    def _rows(self) -> dict[str, dict[int, dict[str, float]]]:
        indexed: dict[str, dict[int, dict[str, float]]] = {}
        for pair, frame in self._frames.items():
            indexed[pair] = {
                int(row["timestamp"]): row for row in frame.to_dict("records")
            }
        return indexed

    def row(self, pair: str, timestamp: int) -> dict[str, float]:
        return self._rows[pair][timestamp]


# ── Result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairTrade:
    pair:       str
    trade:      Trade
    fee:        float
    entered_at: int
    exited_at:  int
    # True for a position the window cut short rather than the strategy
    # closing it. Unwound at its own entry price, so its profit is exactly
    # zero — it is not a bet that was won or lost, and counting it either way
    # would be inventing an outcome the data never produced.
    unwound:    bool = False

    # Gross of costs: did price move the way the genome bet? This is what
    # `required_accuracy` from the screener is a threshold on, so it must not
    # be contaminated by fees the way a net win rate is.
    def called_direction(self) -> bool:
        return self.trade.profit() > 0.0

    def resolved(self) -> bool:
        return not self.unwound

    def net_profit(self) -> float:
        return self.trade.profit() - self.fee


class SelectionResult:
    def __init__(
        self,
        trades: tuple[PairTrade, ...],
        equity_curve: list[float],
        starting_balance: float,
    ) -> None:
        self._trades           = trades
        self._equity_curve     = equity_curve
        self._starting_balance = starting_balance

    def trades(self) -> tuple[PairTrade, ...]:
        return self._trades

    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    def gross_profit(self) -> float:
        return sum(trade.trade.profit() for trade in self._trades)

    def fees_paid(self) -> float:
        return sum(trade.fee for trade in self._trades)

    def net_profit(self) -> float:
        return self.gross_profit() - self.fees_paid()

    # Only resolved trades count. An unwound position was closed by the window
    # ending, not by the strategy, so putting it in the denominator scores the
    # calendar rather than the genome — and always as a miss, since it unwinds
    # at exactly break-even.
    def resolved_trades(self) -> tuple[PairTrade, ...]:
        return tuple(trade for trade in self._trades if trade.resolved())

    def direction_accuracy(self) -> float:
        resolved = self.resolved_trades()
        if not resolved:
            return 0.0
        return sum(1 for trade in resolved if trade.called_direction()) / len(resolved)

    def win_rate(self) -> float:
        resolved = self.resolved_trades()
        if not resolved:
            return 0.0
        return sum(1 for trade in resolved if trade.net_profit() > 0.0) / len(resolved)

    def total_return(self) -> float:
        return self.net_profit() / self._starting_balance

    # Which pairs the selector actually chose, and what each one earned. A
    # selector whose entire result comes from one pair has not demonstrated
    # selection — it has demonstrated that one pair went up.
    def attribution(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        rows = [
            {
                "pair":       trade.pair,
                "direction":  trade.trade.direction().name,
                "gross":      trade.trade.profit(),
                "fee":        trade.fee,
                "net":        trade.net_profit(),
                # NaN for an unwound trade so the per-pair accuracy skips it,
                # matching the headline. Counting it as a miss reported
                # LINK-USDT at 0% accuracy for a position the calendar closed.
                "called":     float(trade.called_direction()) if trade.resolved() else float("nan"),
            }
            for trade in self._trades
        ]
        frame = pd.DataFrame(rows)
        summary = frame.groupby("pair").agg(
            trades=("net", "count"),
            net=("net", "sum"),
            gross=("gross", "sum"),
            fees=("fee", "sum"),
            accuracy=("called", "mean"),
        )
        return summary.sort_values("net", ascending=False)


# ── The run ────────────────────────────────────────────────────────────

class SelectionRun:
    def __init__(
        self,
        aligned: AlignedFrames,
        strategies: dict[str, GaStrategy],
        configs: dict[str, StrategyConfig],
        starting_balance: float,
        fees: FeeSchedule,
    ) -> None:
        self._aligned          = aligned
        self._strategies       = strategies
        self._configs          = configs
        self._starting_balance = starting_balance
        self._fees             = fees
        self._result: Optional[SelectionResult] = None

    def run(self) -> None:
        ledger  = Ledger(self._starting_balance)
        trades: list[PairTrade] = []
        equity: list[float]     = []
        held:    Optional[str]  = None
        entered: int            = 0
        settled: int            = 0     # trades already folded into `trades`
        opening: float          = 0.0   # the entry fee this position already paid

        for timestamp in self._aligned.timestamps:
            if held is not None:
                row   = self._aligned.row(held, timestamp)
                price = row["close"]
                size  = ledger.position().size()

                # The held pair lives through its candle's range before any new
                # decision is taken on its close — the ordering Backtest.run
                # uses, so a genome sees here exactly what it saw in training.
                ledger.liquidate(row.get("high", price), row.get("low", price))
                if len(ledger.trades()) > settled:
                    # A liquidation pays no exit fee, matching PaperTick: the
                    # exchange took the position, we did not trade out of it.
                    # The entry fee it already paid still belongs to the trade.
                    trades.append(self._settled(ledger, held, entered, timestamp, 0.0, opening))
                    settled, held = len(ledger.trades()), None
                else:
                    decision = self._strategies[held].decide(row, ledger.position(), ledger.balance())
                    ledger.apply(decision, price)
                    if len(ledger.trades()) > settled:
                        trades.append(self._settled(ledger, held, entered, timestamp, size * price, opening))
                        settled, held = len(ledger.trades()), None

            if held is None:
                held, entered, opening = self._entered(ledger, timestamp)
            equity.append(self._equity(ledger, held, timestamp))

        # An open position at the end unwinds at its own entry price, matching
        # Backtest's unwind_at_entry_price — a window that happens to cut
        # mid-hold should neither reward nor punish the strategy.
        if held is not None and ledger.position() is not None:
            ledger.force_close(ledger.position().entry_price())
            if len(ledger.trades()) > settled:
                last = self._aligned.timestamps[-1]
                trades.append(self._settled(ledger, held, entered, last, 0.0, opening, unwound=True))

        self._result = SelectionResult(tuple(trades), equity, self._starting_balance)

    def result(self) -> SelectionResult:
        if self._result is None:
            raise ValueError("selection run has not been executed")
        return self._result

    # Returns the entry fee alongside the pair, so the trade it opens can carry
    # BOTH legs. Charging it here and recording only the exit on the PairTrade
    # made fees_paid() and net_profit() disagree with the equity curve by every
    # entry fee — and a trade losing less than its entry cost scored as a win.
    def _entered(self, ledger: Ledger, timestamp: int) -> tuple[Optional[str], int, float]:
        # A wiped book must not open anything: a negative balance yields a
        # negative size, whose Trade.profit() has its sign flipped and would
        # profit from losing.
        if ledger.balance() <= 0.0:
            return None, 0, 0.0
        best = PairRanking(self._candidates(timestamp)).best()
        if best is None:
            return None, 0, 0.0

        row = self._aligned.row(best.pair, timestamp)
        # The genome's own decide() produces the action AND the size, so the
        # selector cannot drift from the strategy it is selecting: Conviction
        # only ever ranks, it never decides.
        decision = self._strategies[best.pair].decide(row, None, ledger.balance())
        if decision.action is Action.HOLD:
            return None, 0, 0.0
        ledger.apply(decision, row["close"])
        # Charged on notional rather than on decision.size, for the reason
        # PaperTick gives: a closing decision carries size 0.0, so pricing off
        # it would charge entries only and halve the real cost.
        fee = self._fees.charge(decision.size * row["close"])
        ledger.charge(fee)
        return best.pair, timestamp, fee

    def _candidates(self, timestamp: int) -> tuple[Candidate, ...]:
        candidates: list[Candidate] = []
        for pair in self._aligned.pairs():
            strategy = self._strategies[pair]
            # Scored flat: this is the question "would I open here?", so the
            # position_pnl term must not carry another pair's open position.
            score      = strategy.signal_score(self._aligned.row(pair, timestamp), None)
            conviction = Conviction(score, self._configs[pair], strategy.flat_score_ceiling())
            candidates.append(Candidate(pair, conviction.action(), conviction.value(), score))
        return tuple(candidates)

    def _settled(
        self,
        ledger: Ledger,
        held: str,
        entered: int,
        timestamp: int,
        notional: float,
        entry_fee: float,
        unwound: bool = False,
    ) -> PairTrade:
        exit_fee = self._fees.charge(notional)
        ledger.charge(exit_fee)
        return PairTrade(
            held, ledger.trades()[-1], entry_fee + exit_fee, entered, timestamp, unwound,
        )

    def _equity(self, ledger: Ledger, held: Optional[str], timestamp: int) -> float:
        if held is None:
            return ledger.balance()
        return ledger.equity(self._aligned.row(held, timestamp)["close"])


# ── Benchmarks ─────────────────────────────────────────────────────────

# What the same money would have done with no strategy at all, over the same
# aligned window. Without this a headline return means nothing: a selector that
# returns +60% in a window where simply holding returned +200% has destroyed
# value, and the trade log alone will never say so.
class BuyAndHold:
    def __init__(self, aligned: AlignedFrames) -> None:
        self._aligned = aligned

    @functools.cached_property
    def returns(self) -> dict[str, float]:
        stamps = self._aligned.timestamps
        if not stamps:
            return {}
        held: dict[str, float] = {}
        for pair in self._aligned.pairs():
            first = self._aligned.row(pair, stamps[0])["close"]
            last  = self._aligned.row(pair, stamps[-1])["close"]
            if first > 0.0:
                held[pair] = last / first - 1.0
        return held

    # Equal money in every pair on day one, untouched. The fair benchmark for
    # a selector drawing from this universe — it had the same menu.
    def equal_weight(self) -> float:
        values = list(self.returns.values())
        return sum(values) / len(values) if values else 0.0

    def best_pair(self) -> tuple[str, float]:
        if not self.returns:
            return "", 0.0
        pair = max(self.returns, key=self.returns.get)
        return pair, self.returns[pair]


# The 95% interval on a hit rate from n trades. Printed next to direction
# accuracy because a bare "58%" from 12 trades reads as evidence when it is
# indistinguishable from a coin, and the required_accuracy bar it is being
# compared against sits inside the interval.
class WilsonInterval:
    def __init__(self, successes: int, trials: int, z: float = 1.96) -> None:
        self._successes = successes
        self._trials    = trials
        self._z         = z

    def bounds(self) -> tuple[float, float]:
        n = self._trials
        if n == 0:
            return 0.0, 1.0
        p      = self._successes / n
        z2     = self._z ** 2
        centre = (p + z2 / (2 * n)) / (1 + z2 / n)
        spread = (self._z / (1 + z2 / n)) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
        return max(0.0, centre - spread), min(1.0, centre + spread)


# ── Genome sourcing ────────────────────────────────────────────────────

# Picks each pair's best recorded run out of experiments/index.csv.
#
# This is IN-SAMPLE selection and the results it feeds are optimistic by
# exactly that much: the run chosen is the one that scored best on the very
# data it is then selected on. It is the right default for a first look
# — there is no other automatic way to name a genome per pair — but a result
# worth acting on wants genomes chosen on one window and selected over another.
class BestRunPerPair:
    def __init__(
        self,
        frame: pd.DataFrame,
        pairs: tuple[str, ...],
        metric: str,
        granularity: str,
    ) -> None:
        self._frame       = frame
        self._pairs       = pairs
        self._metric      = metric
        self._granularity = granularity

    def run_ids(self) -> dict[str, str]:
        chosen: dict[str, str] = {}
        for pair in self._pairs:
            # Filtered on granularity as well as pair: every frame here is
            # built at one granularity, and a genome trained on THIRTY_MINUTE
            # candles applied to SIX_HOUR rows is reading indicators that mean
            # something else entirely. The index holds runs at three.
            rows = self._frame[
                (self._frame["pair"] == pair)
                & (self._frame["granularity"] == self._granularity)
            ]
            if rows.empty:
                continue
            chosen[pair] = rows.loc[rows[self._metric].idxmax(), "run_id"]
        return chosen


# Resolves a list of pairs to their trained genomes, shared by the backtest
# and the paper runner so the two can never disagree about which genome drives
# which pair.
class TrainedPairs:
    def __init__(
        self,
        raw_config: dict[str, Any],
        pairs: tuple[str, ...],
        metric: str,
        granularity: str,
    ) -> None:
        self._raw_config  = raw_config
        self._pairs       = pairs
        self._metric      = metric
        self._granularity = granularity

    @functools.cached_property
    def resolved(self) -> dict[str, "TrainedPair"]:
        output      = OutputConfigFile(self._raw_config).config()
        index       = pd.read_csv(output.index_filepath)
        run_ids     = BestRunPerPair(index, self._pairs, self._metric, self._granularity).run_ids()
        market      = MarketDataConfig(self._raw_config)
        weight_keys = ValidatedWeightKeys(
            WeightKeysConfig(self._raw_config).keys(), market.normalized_columns(),
        ).keys() + (POSITION_PNL_KEY,)
        return {
            pair: TrainedPair(
                pair,
                ExperimentDirectory(output.experiments_dir, run_ids[pair]).strategy_path(),
                self._raw_config, weight_keys,
            )
            for pair in self._pairs if pair in run_ids
        }

    # Named rather than silently dropped: a selector quietly running over three
    # of twelve pairs looks like a result, not a misconfiguration.
    def missing(self) -> tuple[str, ...]:
        return tuple(pair for pair in self._pairs if pair not in self.resolved)


class TrainedPair:
    def __init__(
        self,
        pair: str,
        strategy_filepath: str,
        raw_config: dict[str, Any],
        weight_keys: tuple[str, ...],
    ) -> None:
        self._pair              = pair
        self._strategy_filepath = strategy_filepath
        self._raw_config        = raw_config
        self._weight_keys       = weight_keys

    @functools.cached_property
    def _saved(self) -> StrategyJsonFile:
        return StrategyJsonFile(self._strategy_filepath)

    @functools.cached_property
    def config(self) -> StrategyConfig:
        # The genome's own hyperparameters win over config.yaml, matching how
        # paper_trading rehydrates a saved strategy.
        return TrainedStrategyConfig(self._raw_config, self._saved.hyperparameters()).config()

    def strategy(self) -> GaStrategy:
        # A genome saved before a weight key existed carries no weight for it,
        # and Genome.weight raises rather than guessing. Backfilling at zero
        # keeps every pre-existing genome runnable against a config that has
        # since grown a column, scored on exactly what it was trained on.
        backfilled = BackfilledGenome(Genome(self._saved.weights()), self._weight_keys)
        if backfilled.missing():
            logger.warning(
                "%s: genome predates %s — running it with those weighted zero; "
                "retrain to let the GA actually use them",
                self._pair, ", ".join(backfilled.missing()),
            )
        model = SignalDesign(self.config.design).model(
            backfilled.filled(), self._weight_keys,
        )
        return GaStrategy(model, self.config)


# ── Report ─────────────────────────────────────────────────────────────

class SelectionReport:
    def __init__(
        self,
        result: SelectionResult,
        pairs: tuple[str, ...],
        candles: int,
        fee_bps: float,
        starting_balance: float,
        benchmark: BuyAndHold,
    ) -> None:
        self._result           = result
        self._pairs            = pairs
        self._candles          = candles
        self._fee_bps          = fee_bps
        self._starting_balance = starting_balance
        self._benchmark        = benchmark

    def print(self) -> None:
        rule   = "─" * 78
        result = self._result
        print(f"\n{rule}")
        print(f"Cross-sectional selection over {len(self._pairs)} pairs, {self._candles} aligned candles")
        print(f"pairs     {', '.join(self._pairs)}")
        print(f"fees      {self._fee_bps:.2f} bps per side")
        print(rule)
        unwound = len(result.trades()) - len(result.resolved_trades())
        print(
            f"trades              {len(result.trades())}"
            + (f"   ({unwound} unwound by the window ending, excluded from the rates below)"
               if unwound else "")
        )
        print(f"gross profit        {result.gross_profit():>12,.2f}")
        print(f"fees paid           {result.fees_paid():>12,.2f}")
        print(f"net profit          {result.net_profit():>12,.2f}   on {self._starting_balance:,.0f}")
        print(f"total return        {result.total_return() * 100:>11.2f}%")
        print(f"win rate (net)      {result.win_rate() * 100:>11.2f}%")

        resolved  = result.resolved_trades()
        called    = sum(1 for trade in resolved if trade.called_direction())
        low, high = WilsonInterval(called, len(resolved)).bounds()
        print(
            f"direction accuracy  {result.direction_accuracy() * 100:>11.2f}%   "
            f"95% CI [{low * 100:.1f}%, {high * 100:.1f}%]  <- vs req_accuracy"
        )

        self._print_benchmark()

        if not result.attribution().empty:
            print("\nPer-pair attribution:")
            print(result.attribution().to_string())
            self._print_concentration()

        self._print_caveats()

    # The comparison that decides whether any of this was worth doing.
    def _print_benchmark(self) -> None:
        equal            = self._benchmark.equal_weight()
        best_pair, best  = self._benchmark.best_pair()
        strategy         = self._result.total_return()
        print(f"\nvs doing nothing over the same window:")
        print(f"  equal-weight buy and hold   {equal * 100:>11.2f}%")
        print(f"  best single pair held       {best * 100:>11.2f}%   ({best_pair})")
        print(f"  this selector               {strategy * 100:>11.2f}%")
        if strategy <= equal:
            print("  -> the selector LOST to holding every pair equally. Selection")
            print("     subtracted value here; the trades are not the reason for any gain.")

    # A result carried by one pair is not evidence of selection skill, it is
    # evidence that one pair moved. Named explicitly because the total return
    # hides it completely.
    def _print_concentration(self) -> None:
        attribution = self._result.attribution()
        net         = self._result.net_profit()
        top         = attribution["net"].iloc[0]
        if net > 0 and top > 0 and top / net > 0.6:
            share = top / net * 100
            print(
                f"\n  NOTE: {attribution.index[0]} alone accounts for {share:.0f}% of net profit. "
                f"Strip it out and the rest returns {(net - top) / self._starting_balance * 100:.2f}%."
            )

    def _print_caveats(self) -> None:
        result = self._result
        print(f"\n{'─' * 78}")
        print("Read before believing this:")
        if len(result.resolved_trades()) < 30:
            print(f"  · {len(result.resolved_trades())} resolved trades is too few to distinguish")
            print("    skill from luck. Treat the return as noise until the count is in the hundreds.")
        print("  · Genomes were picked as each pair's BEST recorded run, on the same data")
        print("    they are then selected over. That is in-sample twice; expect the live")
        print("    number to be materially worse.")
        print("  · Cross-pair ranking compares scores built from norm_* columns min-max")
        print("    scaled inside each pair's own window. Conviction normalizes against")
        print("    each genome's own trigger, but the comparison stays approximate.")
        print("  · No borrow interest on shorts, no slippage, no partial fills. Every")
        print("    fill is at the candle close, in full.")
        print(f"{'─' * 78}")


# ── Data ───────────────────────────────────────────────────────────────

class SelectorMarketData:
    def __init__(
        self,
        pairs: tuple[str, ...],
        exchange: str,
        raw_config: dict[str, Any],
        start: int,
        end: int,
        max_concurrent_requests: int = 8,
    ) -> None:
        self._pairs      = pairs
        self._exchange   = exchange
        self._raw_config = raw_config
        self._start      = start
        self._end        = end
        self._max        = max_concurrent_requests
        self._frames: Optional[dict[str, pd.DataFrame]] = None

    async def run(self) -> None:
        market = MarketDataConfig(self._raw_config)
        window = market.window()
        async with ExchangePool((self._exchange,), self._max) as pool:
            lane = pool.lane(self._exchange)
            # Fetched once for the whole run and handed to every pair. Without
            # it, a config that lists index_z in normalized_columns produces no
            # such column and NormalizedIndicators dies on KeyError before a
            # single candle is scored.
            index_returns = None
            if market.index_pairs():
                index_returns = await MarketBasket(
                    lane.adapter(), market.index_pairs(), window.granularity,
                    self._start, self._end, market.cache_dir(), lane.limit(),
                ).returns()
            # One shared semaphore across every pair, for the reason
            # HistoricalCandles' own docstring gives.
            frames = await asyncio.gather(*(
                HistoricalMarketData(
                    lane.adapter(), pair, window.granularity, self._start, self._end,
                    market.periods(), market.normalized_columns(),
                    cache_dir=market.cache_dir(), limit=lane.limit(),
                    index_returns=index_returns, index_period=market.index_period(),
                ).dataframe()
                for pair in self._pairs
            ))
        self._frames = dict(zip(self._pairs, frames))

    def frames(self) -> dict[str, pd.DataFrame]:
        if self._frames is None:
            raise ValueError("market data has not been fetched")
        return self._frames


# ── CLI ────────────────────────────────────────────────────────────────

class ShortlistPairs:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def pairs(self) -> tuple[str, ...]:
        with open(self._filepath, encoding="utf-8") as handle:
            return tuple(json.load(handle)["pairs"])


class SelectorArguments:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    @functools.cached_property
    def parsed(self) -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Backtest one book selecting across many pairs' trained genomes",
        )
        p.add_argument("--config", default="coinbase/ga/config.yaml")
        source = p.add_mutually_exclusive_group()
        source.add_argument("--pairs", nargs="+", help="Pairs to select across")
        source.add_argument("--from-shortlist", metavar="EXCHANGE", nargs="?", const="binance",
                            help="Read pairs from the screener's shortlist for this exchange")
        p.add_argument("--exchange", default=None, help="Override data.exchange")
        p.add_argument("--metric", default="annualized_yield", help="Column picking each pair's genome")
        p.add_argument("--fee-bps", type=float, default=10.0, help="Per-side fee (default: 10)")
        p.add_argument("--no-fees", action="store_true", help="Run gross, like Backtest does")
        p.add_argument("--starting-balance", type=float, default=None)
        p.add_argument("--start", default=None, help="ISO date, overrides data.start_date")
        p.add_argument("--end", default=None, help="ISO date, overrides data.end_date")
        p.add_argument("--max-concurrent", type=int, default=8)
        return p.parse_args(self._argv)

    def pairs(self, default_exchange: str) -> tuple[str, ...]:
        args = self.parsed
        if args.pairs:
            return tuple(args.pairs)
        exchange = args.from_shortlist or default_exchange
        return ShortlistPairs(
            str(GA_RESULTS_ROOT / "screener" / f"shortlist_{exchange}_latest.json"),
        ).pairs()

    def fees(self) -> FeeSchedule:
        return NoFees() if self.parsed.no_fees else BasisPointFee(self.parsed.fee_bps)


async def _main(argv: list[str]) -> None:
    # stderr too: warnings carry the same box-drawing and dash characters the
    # report does, and a cp1252 console mangles them there just as readily.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    arguments  = SelectorArguments(argv)
    args       = arguments.parsed
    raw_config = ConfigFile(args.config).raw()
    market     = MarketDataConfig(raw_config)
    window     = market.window()
    exchange   = args.exchange or (raw_config.get("data") or {}).get("exchange", "coinbase")
    pairs      = arguments.pairs(exchange)

    catalogue = TrainedPairs(raw_config, pairs, args.metric, window.granularity)
    trained   = catalogue.resolved
    if catalogue.missing():
        logger.warning(
            "no %s genome for %s — train them first with a data.pair sweep axis",
            window.granularity, ", ".join(catalogue.missing()),
        )
    pairs = tuple(trained)
    if not pairs:
        raise ValueError(
            "none of the requested pairs have a trained genome for "
            f"{window.granularity}. Train them first: add a `data.pair` axis to "
            "sweep.yaml and run `python -m coinbase.ga.sweep`."
        )

    data = SelectorMarketData(
        pairs, exchange, raw_config,
        _IsoOrDefault(args.start, window.start).value(),
        _IsoOrDefault(args.end, window.end).value(),
        args.max_concurrent,
    )
    await data.run()

    aligned = AlignedFrames(data.frames())
    if not aligned.timestamps:
        raise ValueError("no timestamp is shared by every pair — windows do not overlap")

    balance = (
        args.starting_balance if args.starting_balance is not None
        else next(iter(trained.values())).config.starting_balance
    )
    run     = SelectionRun(
        aligned,
        {pair: item.strategy() for pair, item in trained.items()},
        {pair: item.config for pair, item in trained.items()},
        balance,
        arguments.fees(),
    )
    run.run()

    SelectionReport(
        run.result(), pairs, len(aligned.timestamps),
        0.0 if args.no_fees else args.fee_bps, balance, BuyAndHold(aligned),
    ).print()


class _IsoOrDefault:
    def __init__(self, value: Optional[str], fallback: int) -> None:
        self._value    = value
        self._fallback = fallback

    def value(self) -> int:
        if self._value is None:
            return self._fallback
        return IsoDate(self._value).timestamp()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
