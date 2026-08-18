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


# ── Ledger ───────────────────────────────────────────────────────────────

class Ledger:
    def __init__(self, balance: float) -> None:
        self._balance  = balance
        self._position: Optional[Position] = None
        self._trades:   list[Trade] = []

    def apply(self, decision: Decision, price: float) -> None:
        if decision.action is Action.BUY and self._position is None:
            self._position = Position(price, decision.size)
        elif decision.action is Action.SELL and self._position is not None:
            self._close(price)

    def force_close(self, price: float) -> None:
        if self._position is not None:
            self._close(price)

    def balance(self) -> float:
        return self._balance

    def position(self) -> Optional[Position]:
        return self._position

    def equity(self, price: float) -> float:
        return self._balance + (self._position.unrealized(price) if self._position is not None else 0.0)

    def trades(self) -> list[Trade]:
        return list(self._trades)

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
    def __init__(self, frame: pd.DataFrame, strategy: Strategy, starting_balance: float) -> None:
        self._frame            = frame
        self._strategy         = strategy
        self._starting_balance = starting_balance

    @functools.cached_property
    def _rows(self) -> list[dict[str, float]]:
        return self._frame.to_dict("records")

    def run(self) -> BacktestResult:
        ledger = Ledger(self._starting_balance)
        equity_curve: list[float] = []

        for row in self._rows:
            decision = self._strategy.decide(row, ledger.position(), ledger.balance())
            price    = row["close"]
            ledger.apply(decision, price)
            equity_curve.append(ledger.equity(price))

        if self._rows:
            ledger.force_close(self._rows[-1]["close"])
        return BacktestResult(ledger.trades(), equity_curve)
