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


# ── Positions / trades ───────────────────────────────────────────────────

class Trade:
    def __init__(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: Direction = Direction.LONG,
    ) -> None:
        self._entry_price = entry_price
        self._exit_price  = exit_price
        self._size        = size
        self._direction   = direction

    def profit(self) -> float:
        move = (self._exit_price - self._entry_price) * self._size
        return move if self._direction is Direction.LONG else -move

    def direction(self) -> Direction:
        return self._direction


class Position:
    def __init__(
        self,
        entry_price: float,
        size: float,
        direction: Direction = Direction.LONG,
    ) -> None:
        self._entry_price = entry_price
        self._size        = size
        self._direction   = direction

    def entry_price(self) -> float:
        return self._entry_price

    def size(self) -> float:
        return self._size

    def direction(self) -> Direction:
        return self._direction

    def closed(self, exit_price: float) -> Trade:
        return Trade(self._entry_price, exit_price, self._size, self._direction)

    def unrealized(self, price: float) -> float:
        move = (price - self._entry_price) * self._size
        return move if self._direction is Direction.LONG else -move

    # Signed fractional return on the entry price — direction-agnostic, so a
    # strategy can read "how far is this position ahead" without branching.
    def unrealized_return(self, price: float) -> float:
        move = (price - self._entry_price) / self._entry_price
        return move if self._direction is Direction.LONG else -move


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
    def __init__(self, balance: float, position: Optional[Position] = None) -> None:
        self._balance  = balance
        self._position = position
        self._trades:   list[Trade] = []

    def apply(self, decision: Decision, price: float) -> None:
        if decision.action is Action.BUY and self._position is None:
            self._position = Position(price, decision.size, Direction.LONG)
        elif decision.action is Action.SHORT and self._position is None:
            self._position = Position(price, decision.size, Direction.SHORT)
        elif decision.action is Action.SELL and self._holds(Direction.LONG):
            self._close(price)
        elif decision.action is Action.COVER and self._holds(Direction.SHORT):
            self._close(price)

    # Closes an open position at its liquidation price when the candle's range
    # breached it. Called with the range of the candle the position is carried
    # into, before that candle's own decision is taken.
    def liquidate(self, high: float, low: float) -> None:
        if self._position is None:
            return
        margin = IsolatedMargin(self._position, self._balance)
        if margin.breached_by(high, low):
            self._close(margin.liquidation_price())

    def force_close(self, price: float) -> None:
        if self._position is not None:
            self._close(price)

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

    def _holds(self, direction: Direction) -> bool:
        return self._position is not None and self._position.direction() is direction

    def _close(self, price: float) -> None:
        trade = self._position.closed(price)
        self._balance += trade.profit()
        self._trades.append(trade)
        self._position = None


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
    def __init__(
        self,
        frame: pd.DataFrame,
        strategy: Strategy,
        starting_balance: float,
        unwind_at_entry_price: bool = True,
    ) -> None:
        self._frame                 = frame
        self._strategy              = strategy
        self._starting_balance      = starting_balance
        self._unwind_at_entry_price = unwind_at_entry_price

    @functools.cached_property
    def _rows(self) -> list[dict[str, float]]:
        return self._frame.to_dict("records")

    def run(self) -> BacktestResult:
        ledger = Ledger(self._starting_balance)
        equity_curve: list[float] = []

        for row in self._rows:
            price = row["close"]
            # A position carried in from the previous candle lives through this
            # candle's range before any new decision is taken on its close.
            ledger.liquidate(row.get("high", price), row.get("low", price))
            decision = self._strategy.decide(row, ledger.position(), ledger.balance())
            ledger.apply(decision, price)
            equity_curve.append(ledger.equity(price))

        if self._rows:
            ledger.force_close(self._final_close_price(ledger, self._rows[-1]["close"]))
        return BacktestResult(ledger.trades(), equity_curve)

    def _final_close_price(self, ledger: Ledger, market_price: float) -> float:
        position = ledger.position()
        if self._unwind_at_entry_price and position is not None:
            return position.entry_price()
        return market_price
