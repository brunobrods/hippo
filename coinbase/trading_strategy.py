import functools
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

import pandas as pd


# ── Decisions ────────────────────────────────────────────────────────────

class Action(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Decision:
    action: Action
    size:   float = 0.0


# ── Positions / trades ───────────────────────────────────────────────────

class Trade:
    def __init__(self, entry_price: float, exit_price: float, size: float) -> None:
        self._entry_price = entry_price
        self._exit_price  = exit_price
        self._size        = size

    def profit(self) -> float:
        return (self._exit_price - self._entry_price) * self._size


class Position:
    def __init__(self, entry_price: float, size: float) -> None:
        self._entry_price = entry_price
        self._size        = size

    def entry_price(self) -> float:
        return self._entry_price

    def size(self) -> float:
        return self._size

    def closed(self, exit_price: float) -> Trade:
        return Trade(self._entry_price, exit_price, self._size)

    def unrealized(self, price: float) -> float:
        return (price - self._entry_price) * self._size


# ── Strategy contract ────────────────────────────────────────────────────

class Strategy(Protocol):
    def decide(self, row: dict[str, float], position: Optional[Position], balance: float) -> Decision: ...


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
    def __init__(self, frame: pd.DataFrame, strategy: Strategy, starting_balance: float) -> None:
        self._frame            = frame
        self._strategy         = strategy
        self._starting_balance = starting_balance

    @functools.cached_property
    def _rows(self) -> list[dict[str, float]]:
        return self._frame.to_dict("records")

    def run(self) -> BacktestResult:
        balance = self._starting_balance
        position: Optional[Position] = None
        trades:       list[Trade] = []
        equity_curve: list[float] = []

        for row in self._rows:
            decision = self._strategy.decide(row, position, balance)
            price    = row["close"]
            if decision.action is Action.BUY and position is None:
                position = Position(price, decision.size)
            elif decision.action is Action.SELL and position is not None:
                trade    = position.closed(price)
                balance += trade.profit()
                trades.append(trade)
                position = None
            equity_curve.append(balance + (position.unrealized(price) if position is not None else 0.0))

        if position is not None:
            trades.append(position.closed(self._rows[-1]["close"]))

        return BacktestResult(trades, equity_curve)
