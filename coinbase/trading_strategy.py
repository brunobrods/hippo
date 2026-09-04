import functools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

import pandas as pd


# ── Decisions ────────────────────────────────────────────────────────────

class Action(Enum):
    BUY   = "BUY"    # open a long
    SELL  = "SELL"   # close a long
    SHORT = "SHORT"  # open a short
    COVER = "COVER"  # close a short
    HOLD  = "HOLD"


class Direction(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Decision:
    action: Action
    size:   float = 0.0


# ── Costs ────────────────────────────────────────────────────────────────
# What a round trip costs beyond the price move. Two charges, kept apart
# because they behave differently: a fee is taken once per leg on the notional
# that changed hands, while borrow interest accrues with time held and only
# against a short — a long here borrows nothing (Binance orders use
# sideEffectType NO_SIDE_EFFECT), so it accrues none.
#
# Both are charged on notional, never on a Decision: a closing decision
# (SELL/COVER) carries size 0.0 because the Ledger closes whatever is open, so
# pricing off decision.size would charge entries only and halve the real cost.
#
# Approximations worth knowing: interest is priced on the borrowed base marked
# at ENTRY rather than re-marked as price moves, slippage is not modelled at
# all, and Ledger.equity() does not deduct interest accrued but not yet paid —
# an open short reads slightly rich until it closes.

# `maker` defaults to False so every existing caller keeps charging the rate it
# always charged. A schedule that does not distinguish the two sides ignores it.
class FeeSchedule(Protocol):
    def charge(self, notional: float, maker: bool = False) -> float: ...


class NoFees:
    def charge(self, notional: float, maker: bool = False) -> float:
        return 0.0


class BasisPointFee:
    def __init__(self, basis_points: float) -> None:
        self._basis_points = basis_points

    def charge(self, notional: float, maker: bool = False) -> float:
        return notional * self._basis_points / 10_000.0


# Both venues price the two sides of the book differently — Coinbase steeply
# (its base tier takes twice as much from a taker), Binance not at all at base
# tier, where the only gain from resting is the spread rather than the fee.
# A strategy that rests its orders is charged the wrong rate by a flat
# BasisPointFee, in whichever direction the venue happens to differ.
class MakerTakerFee:
    def __init__(self, maker_bps: float, taker_bps: float) -> None:
        self._maker_bps = maker_bps
        self._taker_bps = taker_bps

    def charge(self, notional: float, maker: bool = False) -> float:
        rate = self._maker_bps if maker else self._taker_bps
        return notional * rate / 10_000.0


class BorrowRate(Protocol):
    def interest(self, notional: float, seconds: float) -> float: ...


class NoBorrowRate:
    def interest(self, notional: float, seconds: float) -> float:
        return 0.0


class HourlyBasisPointRate:
    def __init__(self, basis_points: float) -> None:
        self._basis_points = basis_points

    def interest(self, notional: float, seconds: float) -> float:
        return notional * self._basis_points / 10_000.0 * (seconds / 3600.0)


# A rate of zero is the no-op object rather than an arithmetic no-op, so a run
# configured without costs takes exactly the code path it took before costs
# existed and reproduces its numbers bit for bit. Both defaults are 0.0, which
# is what keeps every config.yaml already on disk scoring as it always did.

class ConfiguredFees:
    def __init__(self, basis_points: float) -> None:
        self._basis_points = basis_points

    def schedule(self) -> FeeSchedule:
        if self._basis_points <= 0.0:
            return NoFees()
        return BasisPointFee(self._basis_points)


class ConfiguredBorrowRate:
    def __init__(self, basis_points_per_hour: float) -> None:
        self._basis_points_per_hour = basis_points_per_hour

    def rate(self) -> BorrowRate:
        if self._basis_points_per_hour <= 0.0:
            return NoBorrowRate()
        return HourlyBasisPointRate(self._basis_points_per_hour)


# ── Positions / trades ───────────────────────────────────────────────────

class Trade:
    def __init__(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: Direction = Direction.LONG,
        fee: float = 0.0,
        interest: float = 0.0,
    ) -> None:
        self._entry_price = entry_price
        self._exit_price  = exit_price
        self._size        = size
        self._direction   = direction
        self._fee         = fee
        self._interest    = interest

    # Gross, deliberately: every strategy.json and every experiments/index.csv
    # row written before costs existed is denominated in this number, so it has
    # to keep meaning what it meant. net_profit() is what the GA selects on.
    def profit(self) -> float:
        move = (self._exit_price - self._entry_price) * self._size
        return move if self._direction is Direction.LONG else -move

    def fee(self) -> float:
        return self._fee

    def interest(self) -> float:
        return self._interest

    def cost(self) -> float:
        return self._fee + self._interest

    def net_profit(self) -> float:
        return self.profit() - self.cost()

    def direction(self) -> Direction:
        return self._direction


class Position:
    def __init__(
        self,
        entry_price: float,
        size: float,
        direction: Direction = Direction.LONG,
        entry_timestamp: float = 0.0,
        entry_fee: float = 0.0,
    ) -> None:
        self._entry_price     = entry_price
        self._size            = size
        self._direction       = direction
        self._entry_timestamp = entry_timestamp
        self._entry_fee       = entry_fee

    def entry_price(self) -> float:
        return self._entry_price

    def size(self) -> float:
        return self._size

    def direction(self) -> Direction:
        return self._direction

    def entry_timestamp(self) -> float:
        return self._entry_timestamp

    # What was already paid to open. Carried on the position, not tallied by
    # the Ledger, because the two legs of a paper round trip happen in
    # different processes — the book is restored from disk between them, and
    # the closing Trade still has to report the whole trip's cost.
    def entry_fee(self) -> float:
        return self._entry_fee

    def closed(self, exit_price: float, exit_fee: float = 0.0, interest: float = 0.0) -> Trade:
        return Trade(
            self._entry_price, exit_price, self._size, self._direction,
            self._entry_fee + exit_fee, interest,
        )

    def unrealized(self, price: float) -> float:
        move = (price - self._entry_price) * self._size
        return move if self._direction is Direction.LONG else -move

    # Signed fractional return on the entry price — direction-agnostic, so a
    # strategy can read "how far is this position ahead" without branching.
    def unrealized_return(self, price: float) -> float:
        move = (price - self._entry_price) / self._entry_price
        return move if self._direction is Direction.LONG else -move


# What a position owes for the base it borrowed, over the time it was held. A
# long borrows nothing and so owes nothing. A position whose entry time is
# unknown — a book restored from a state file written before entry times were
# recorded — accrues nothing rather than accruing from the epoch.
class BorrowInterest:
    def __init__(self, position: Position, exit_timestamp: float, rate: BorrowRate) -> None:
        self._position       = position
        self._exit_timestamp = exit_timestamp
        self._rate           = rate

    def amount(self) -> float:
        if self._position.direction() is not Direction.SHORT:
            return 0.0
        entry = self._position.entry_timestamp()
        if entry <= 0.0 or self._exit_timestamp <= entry:
            return 0.0
        notional = self._position.size() * self._position.entry_price()
        return self._rate.interest(notional, self._exit_timestamp - entry)


# ── Margin ───────────────────────────────────────────────────────────────
# Isolated margin, matching how Binance actually liquidates.
#
# A short borrows the base asset and sells it, so the wallet afterwards holds
# its original collateral plus the sale proceeds, against a debt of `size` base
# units marked at the live price:
#
#     margin level = (collateral + size * entry) / (size * price)
#
# Binance closes the position when that ratio decays to LIQUIDATION_MARGIN_LEVEL,
# giving
#
#     liquidation price = (collateral + size * entry) / (level * size)
#
# Both halves were verified live on BTCUSDT: a short of 0.00019022 BTC entered
# at 78,266 against 57.9295 USDT of quote assets reported marginLevel 3.890 and
# liquidatePrice 290,022.60, which puts the trigger at exactly 1.05.
#
# This replaces a flat "shorts liquidate at 2x entry" rule. That rule is only
# right for a short whose notional is the ENTIRE wallet — and even then it is
# slightly generous, since 2x/1.05 = 1.905x is where the level is really hit.
# Below full size it fired far too early: at position_size_pct 0.60 the real
# liquidation is around 2.54x entry, so shorts were being closed at a loss that
# the exchange would have let run.
#
# A long here borrows nothing (Binance orders use sideEffectType NO_SIDE_EFFECT),
# so it carries no debt and no liquidation price at all.

LIQUIDATION_MARGIN_LEVEL = 1.05


class IsolatedMargin:
    def __init__(self, position: Position, collateral: float) -> None:
        self._position   = position
        self._collateral = collateral

    def liquidation_price(self) -> float:
        if self._position.direction() is not Direction.SHORT:
            return 0.0
        size = self._position.size()
        # No borrow means no debt to be liquidated against, at any price.
        if size <= 0.0:
            return math.inf
        proceeds = size * self._position.entry_price()
        return (self._collateral + proceeds) / (LIQUIDATION_MARGIN_LEVEL * size)

    def breached_by(self, high: float, low: float) -> bool:
        if self._position.direction() is Direction.SHORT:
            return high >= self.liquidation_price()
        return low <= self.liquidation_price()


# ── Strategy contract ────────────────────────────────────────────────────

class Strategy(Protocol):
    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision: ...


# ── Ledger ───────────────────────────────────────────────────────────────

class Ledger:
    # `position` restores a book carried in from a previous process — a paper
    # or live run that ticks once per candle rebuilds its open position here.
    # `trades` always starts empty: it records what closed during THIS ledger's
    # life, while realized profit is already folded into `balance`.
    def __init__(
        self,
        balance: float,
        position: Optional[Position] = None,
        fees: FeeSchedule = NoFees(),
        borrow: BorrowRate = NoBorrowRate(),
    ) -> None:
        self._balance  = balance
        self._position = position
        self._fees     = fees
        self._borrow   = borrow
        self._fees_charged     = 0.0
        self._interest_charged = 0.0
        self._trades:   list[Trade] = []

    def apply(self, decision: Decision, price: float, timestamp: float = 0.0) -> None:
        if decision.action is Action.BUY and self._position is None:
            self._open(price, decision.size, Direction.LONG, timestamp)
        elif decision.action is Action.SHORT and self._position is None:
            self._open(price, decision.size, Direction.SHORT, timestamp)
        elif decision.action is Action.SELL and self._holds(Direction.LONG):
            self._close(price, timestamp)
        elif decision.action is Action.COVER and self._holds(Direction.SHORT):
            self._close(price, timestamp)

    # Closes an open position at its liquidation price when the candle's range
    # breached it. Called with the range of the candle the position is carried
    # into, before that candle's own decision is taken. A forced exit pays the
    # same taker fee a voluntary one does — the exchange does not waive it.
    def liquidate(self, high: float, low: float, timestamp: float = 0.0) -> None:
        if self._position is None:
            return
        margin = IsolatedMargin(self._position, self._balance)
        if margin.breached_by(high, low):
            self._close(margin.liquidation_price(), timestamp)

    def force_close(self, price: float, timestamp: float = 0.0) -> None:
        if self._position is not None:
            self._close(price, timestamp)

    # Takes a cost out of the book without touching the position. Trade.profit()
    # deliberately stays gross — every saved genome's recorded performance and
    # every row of experiments/index.csv is denominated in it, so folding fees
    # in there would invalidate comparisons across the whole history. A caller
    # that wants a net book charges on top, here.
    def charge(self, amount: float) -> None:
        self._balance -= amount

    def balance(self) -> float:
        return self._balance

    def position(self) -> Optional[Position]:
        return self._position

    def equity(self, price: float) -> float:
        return self._balance + (self._position.unrealized(price) if self._position is not None else 0.0)

    def trades(self) -> list[Trade]:
        return list(self._trades)

    # What actually left the balance during THIS ledger's life. An entry fee
    # lands here on the candle it was paid rather than being held back until the
    # round trip closes, which is what a per-tick report needs — and why this is
    # not the same as summing trades' costs. Kept apart because a report that
    # merges them cannot show which cost is the one hurting.
    def fees_charged(self) -> float:
        return self._fees_charged

    def interest_charged(self) -> float:
        return self._interest_charged

    def charged(self) -> float:
        return self._fees_charged + self._interest_charged

    def _holds(self, direction: Direction) -> bool:
        return self._position is not None and self._position.direction() is direction

    def _open(self, price: float, size: float, direction: Direction, timestamp: float) -> None:
        fee = self._fees.charge(size * price)
        self._balance      -= fee
        self._fees_charged += fee
        self._position = Position(price, size, direction, timestamp, fee)

    def _close(self, price: float, timestamp: float) -> None:
        exit_fee = self._fees.charge(self._position.size() * price)
        interest = BorrowInterest(self._position, timestamp, self._borrow).amount()
        trade    = self._position.closed(price, exit_fee, interest)
        # The entry fee left the balance when the position opened, so only this
        # leg's costs come out now — trade.cost() covers both legs and would
        # charge the entry twice.
        self._balance          += trade.profit() - exit_fee - interest
        self._fees_charged     += exit_fee
        self._interest_charged += interest
        self._trades.append(trade)
        self._position = None


# ── Backtest ────────────────────────────────────────────────────────────

# The frame in its row-iteration form. Separate from Backtest because a GA
# scores thousands of genomes against ONE window: Backtest is rebuilt per
# genome, so a cache living on it never survived to a second use and
# to_dict("records") re-ran once per genome. Held by the caller instead, the
# conversion happens once per window and every genome reuses it — measured
# 23-26% faster over a 150-genome sample, which at THIRTY_MINUTE (a ~21k-row
# frame, 100 x 51 genomes) is 9.2 -> 7.0 minutes per training run.
class MarketRows:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    @functools.cached_property
    def records(self) -> list[dict[str, float]]:
        return self._frame.to_dict("records")


class BacktestResult:
    def __init__(self, trades: list[Trade], equity_curve: list[float]) -> None:
        self._trades       = trades
        self._equity_curve = equity_curve

    def trades(self) -> list[Trade]:
        return list(self._trades)

    def gross_profit(self) -> float:
        return sum(trade.profit() for trade in self._trades)

    def fees_paid(self) -> float:
        return sum(trade.fee() for trade in self._trades)

    def interest_paid(self) -> float:
        return sum(trade.interest() for trade in self._trades)

    def net_profit(self) -> float:
        return sum(trade.net_profit() for trade in self._trades)

    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)


class Backtest:
    def __init__(
        self,
        rows: MarketRows,
        strategy: Strategy,
        starting_balance: float,
        unwind_at_entry_price: bool = True,
        fees: FeeSchedule = NoFees(),
        borrow: BorrowRate = NoBorrowRate(),
    ) -> None:
        self._rows                  = rows
        self._strategy              = strategy
        self._starting_balance      = starting_balance
        self._unwind_at_entry_price = unwind_at_entry_price
        self._fees                  = fees
        self._borrow                = borrow

    def run(self) -> BacktestResult:
        ledger = Ledger(self._starting_balance, None, self._fees, self._borrow)
        equity_curve: list[float] = []
        records = self._rows.records

        for row in records:
            price     = row["close"]
            timestamp = row.get("timestamp", 0.0)
            # A position carried in from the previous candle lives through this
            # candle's range before any new decision is taken on its close.
            ledger.liquidate(row.get("high", price), row.get("low", price), timestamp)
            decision = self._strategy.decide(row, ledger.position(), ledger.balance())
            ledger.apply(decision, price, timestamp)
            equity_curve.append(ledger.equity(price))

        if records:
            last = records[-1]
            # The window's end is not a trading decision, but the position was
            # still open through it: the exit leg pays its fee and a short pays
            # the interest it accrued, so `unwind_at_entry_price` is now
            # net-zero minus what holding actually cost, not flat.
            ledger.force_close(
                self._final_close_price(ledger, last["close"]), last.get("timestamp", 0.0),
            )
        return BacktestResult(ledger.trades(), equity_curve)

    def _final_close_price(self, ledger: Ledger, market_price: float) -> float:
        position = ledger.position()
        if self._unwind_at_entry_price and position is not None:
            return position.entry_price()
        return market_price
