"""
The read model behind the paper-trading dashboard.
---------------------------------------------------

Everything here is pure: frozen snapshots of what an algo looked like at a
moment, and objects that derive statistics from them. Nothing fetches, nothing
mutates a book, nothing awaits — which is what lets a web handler read engine
state mid-run without locking (see paper_engine.StatusBoard).

NOTE: norm_* columns are deliberately absent from every payload. They are
min-max scaled against each pair's own trailing window, so norm_rsi for one
pair means nothing next to another's. Raw rsi/macd are comparable and are what
gets reported; the normalized values stay strategy input only.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from coinbase.ga.strategy_evaluator import AnnualizedYield
from coinbase.ga.strategy_output import MaxDrawdown


# ── JSON safety ────────────────────────────────────────────────────────
# json.dumps emits a bare `Infinity`/`NaN` for non-finite floats. Neither is
# valid JSON, and JSON.parse rejects the whole document — so one unbounded
# number would blank the entire dashboard rather than one cell. Absent is the
# honest encoding, and the page already renders a missing value as "-".

class _Finite:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> Optional[float]:
        return self._value if math.isfinite(self._value) else None


# ── Snapshots ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PositionView:
    direction:         str
    size:              float
    entry_price:       float
    unrealized:        float
    unrealized_return: float
    liquidation_price: float


@dataclass(frozen=True)
class AlgoStatus:
    name:              str
    exchange:          str
    pair:              str
    granularity:       str
    running:           bool
    error:             Optional[str]
    last_tick_at:      Optional[str]
    last_candle_start: int
    last_action:       str
    starting_balance:  float
    balance:           float
    mark_price:        float
    equity:            float
    position:          Optional[PositionView]
    realized_pnl:      float
    unrealized_pnl:    float
    trades:            int
    wins:              int
    rsi:               float
    macd:              float
    signal_score:      float
    fee_paid:          float
    interest_paid:     float = 0.0


# ── Equity curve ───────────────────────────────────────────────────────
# Sampled on every price refresh, not once per decision, so live drawdown is
# measured against the intraday path rather than four points a day.

class EquityCurve:
    def __init__(self, capacity: int = 20160) -> None:
        self._capacity = capacity
        self._values: list[float] = []

    def record(self, equity: float) -> None:
        self._values.append(equity)
        if len(self._values) > self._capacity:
            del self._values[: len(self._values) - self._capacity]

    def values(self) -> list[float]:
        return list(self._values)


# ── Derived statistics ─────────────────────────────────────────────────

class AlgoPerformance:
    def __init__(self, status: AlgoStatus, curve: EquityCurve, elapsed_seconds: float) -> None:
        self._status          = status
        self._curve           = curve
        self._elapsed_seconds = elapsed_seconds

    def win_rate(self) -> float:
        if self._status.trades == 0:
            return 0.0
        return self._status.wins / self._status.trades

    def total_return(self) -> float:
        if self._status.starting_balance == 0.0:
            return 0.0
        return (self._status.equity - self._status.starting_balance) / self._status.starting_balance

    # Annualizing a short window raises (1 + return) to an enormous power: over
    # one hour the exponent is 8766, which overflows for any return above ~8%,
    # and over a minute it is ~500,000. A total loss makes the base negative.
    # Below a day none of it is a meaningful figure anyway, so report the plain
    # return; past that, an overflow means "unbounded" rather than a number.
    _MIN_ELAPSED_SECONDS = 86400.0

    def annualized_yield(self) -> float:
        if self._status.starting_balance == 0.0:
            return 0.0
        if self._elapsed_seconds < self._MIN_ELAPSED_SECONDS:
            return self.total_return()
        if 1.0 + self.total_return() <= 0.0:
            return -1.0
        try:
            return AnnualizedYield(
                self._status.equity - self._status.starting_balance,
                self._status.starting_balance,
                self._elapsed_seconds,
            ).value()
        except OverflowError:
            return math.inf

    def max_drawdown(self) -> float:
        return MaxDrawdown(self._curve.values()).fraction()

    def as_dict(self) -> dict[str, Any]:
        return {
            "win_rate":         self.win_rate(),
            "total_return":     self.total_return(),
            # Absent rather than a bare `Infinity`, which is not valid JSON.
            "annualized_yield": _Finite(self.annualized_yield()).value(),
            "max_drawdown":     self.max_drawdown(),
        }


class PortfolioStatus:
    def __init__(self, statuses: tuple[AlgoStatus, ...]) -> None:
        self._statuses = statuses

    def equity(self) -> float:
        return sum(status.equity for status in self._statuses)

    def realized_pnl(self) -> float:
        return sum(status.realized_pnl for status in self._statuses)

    def unrealized_pnl(self) -> float:
        return sum(status.unrealized_pnl for status in self._statuses)

    def win_rate(self) -> float:
        trades = sum(status.trades for status in self._statuses)
        if trades == 0:
            return 0.0
        return sum(status.wins for status in self._statuses) / trades

    def as_dict(self) -> dict[str, Any]:
        starting = sum(status.starting_balance for status in self._statuses)
        return {
            "starting_balance": starting,
            "equity":           self.equity(),
            "realized_pnl":     self.realized_pnl(),
            "unrealized_pnl":   self.unrealized_pnl(),
            "total_return":     (self.equity() - starting) / starting if starting else 0.0,
            "win_rate":         self.win_rate(),
            "algos_ok":         sum(1 for status in self._statuses if status.running),
            "algos_errored":    sum(1 for status in self._statuses if not status.running),
        }


# ── Payload ────────────────────────────────────────────────────────────

class StatusPayload:
    def __init__(
        self,
        statuses: tuple[AlgoStatus, ...],
        performances: tuple[AlgoPerformance, ...],
        portfolio: PortfolioStatus,
        started_at: str,
        generated_at: str,
        next_tick_in: float,
        seconds_since_price: float,
    ) -> None:
        self._statuses            = statuses
        self._performances        = performances
        self._portfolio           = portfolio
        self._started_at          = started_at
        self._generated_at        = generated_at
        self._next_tick_in        = next_tick_in
        self._seconds_since_price = seconds_since_price

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at":          self._started_at,
            "generated_at":        self._generated_at,
            "next_tick_in":        self._next_tick_in,
            "seconds_since_price": self._seconds_since_price,
            "portfolio":           self._portfolio.as_dict(),
            "algos": [
                {**self._algo(status), **performance.as_dict()}
                for status, performance in zip(self._statuses, self._performances)
            ],
        }

    @staticmethod
    def _algo(status: AlgoStatus) -> dict[str, Any]:
        return {
            "name":              status.name,
            "exchange":          status.exchange,
            "pair":              status.pair,
            "granularity":       status.granularity,
            "running":           status.running,
            "error":             status.error,
            "last_tick_at":      status.last_tick_at,
            "last_candle_start": status.last_candle_start,
            "last_action":       status.last_action,
            "starting_balance":  status.starting_balance,
            "balance":           status.balance,
            "mark_price":        status.mark_price,
            "equity":            status.equity,
            "realized_pnl":      status.realized_pnl,
            "unrealized_pnl":    status.unrealized_pnl,
            "trades":            status.trades,
            "wins":              status.wins,
            "rsi":               status.rsi,
            "macd":              status.macd,
            "signal_score":      status.signal_score,
            "fee_paid":          status.fee_paid,
            "interest_paid":     status.interest_paid,
            "position":          StatusPayload._position(status.position),
        }

    @staticmethod
    def _position(position: Optional[PositionView]) -> Optional[dict[str, Any]]:
        if position is None:
            return None
        return {
            "direction":         position.direction,
            "size":              position.size,
            "entry_price":       position.entry_price,
            "unrealized":        position.unrealized,
            "unrealized_return": position.unrealized_return,
            # A short with nothing borrowed has no liquidation price at all,
            # which IsolatedMargin reports as infinity.
            "liquidation_price": _Finite(position.liquidation_price).value(),
        }
