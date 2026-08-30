"""
Paper trading — one idempotent tick per invocation.
----------------------------------------------------

Unlike `dry_run.py`, which holds a process open and sleeps between ticks, this
runs once and exits. That makes it schedulable: Windows Task Scheduler (or cron)
fires it, it does at most one candle's work, and it leaves. A reboot, a closed
lid, or a missed window costs nothing — the next invocation picks up where the
last one left off.

Two properties make that safe:

  State on disk.  Balance and any open position are persisted after every tick
  and reloaded on the next, so the book survives restarts. `dry_run.py` builds
  a fresh Ledger each launch and silently restarts flat.

  Idempotent per candle.  A tick records the candle it acted on and refuses to
  act on it twice, so the task can be scheduled far more often than the trading
  granularity. Schedule it every 30 minutes against SIX_HOUR candles and 11 of
  every 12 runs are no-ops. That removes any need to align a local-time
  schedule to UTC candle boundaries across DST, and means a machine that was
  asleep at the boundary simply acts late rather than skipping.

Decisions are taken on the last CLOSED candle (see ClosedMarketRow), matching
what a Backtest sees. Nothing here places an order or reads a real balance:
the ledger is simulated, seeded from strategy.starting_balance.

NOTE: `LiveMarketRow` min-max normalizes indicators against its trailing
window, while the GA trained against the full training window's range, so
norm_* columns do not mean quite the same thing here as they did in scoring.
Known and deliberately deferred — see coinbase/strategy.py.

Config (`paper:` section of config.yaml, every key optional):

    paper:
      exchange: "binance"      # defaults to data.exchange
      pair: "BTC-USDT"         # defaults to data.pair
      state_filepath: "..."    # defaults to ~/.coinbase/ga/paper_state.json

Run (as a module, from the repo root):
    python -m coinbase.ga.paper_trading
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from coinbase.ga.config import GA_RESULTS_ROOT, ConfigFile
from coinbase.ga.ga_engine import BackfilledGenome, Genome
from coinbase.ga.market_data_processor import MarketDataConfig
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    GaStrategy,
    StrategyConfig,
    StrategyConfigFile,
    ValidatedStrategyConfig,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.ga.strategy_output import DryRunLog, OutputConfigFile, ParentDirectory, StrategyJsonFile, UtcNow
from coinbase.strategy import ClosedMarketRow, LiveMarketRow
from coinbase.trading_strategy import Decision, Direction, Ledger, Position, Strategy
from exchange.selection import ConfiguredExchange


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PaperConfig:
    exchange:       str
    pair:           str
    state_filepath: str


class PaperConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> PaperConfig:
        # `or {}`: a section whose every child is commented out parses to None.
        section = self._raw.get("paper") or {}
        data    = self._raw.get("data") or {}
        return PaperConfig(
            exchange       = section.get("exchange", data.get("exchange", "coinbase")),
            pair           = section.get("pair", data["pair"]),
            state_filepath = section.get(
                "state_filepath", str(GA_RESULTS_ROOT / "paper_state.json"),
            ),
        )


# ── Trained hyperparameters ────────────────────────────────────────────
# A saved strategy carries the thresholds it was trained and scored under, and
# those are the ones that make its recorded performance mean anything. Reading
# them from config.yaml instead lets the two drift apart silently — a sweep
# varies thresholds per run, so a genome lifted out of one will usually
# disagree with whatever config.yaml happens to hold now.
#
# config.yaml still supplies what is genuinely a run-time choice
# (starting_balance) and what the frame is built from (indicators,
# weight_keys); the genome's own hyperparameters win over the rest.

class TrainedStrategyConfig:
    def __init__(self, raw_config: dict[str, Any], hyperparameters: dict[str, Any]) -> None:
        self._raw_config      = raw_config
        self._hyperparameters = hyperparameters

    def config(self) -> StrategyConfig:
        section = {**(self._raw_config.get("strategy") or {}), **self._hyperparameters}
        return ValidatedStrategyConfig(StrategyConfigFile({"strategy": section}).config()).config()

    # Keys where the saved strategy and config.yaml disagree — surfaced so a
    # divergence is visible rather than silently resolved.
    def divergences(self) -> dict[str, tuple[Any, Any]]:
        section = self._raw_config.get("strategy") or {}
        return {
            key: (section[key], value)
            for key, value in self._hyperparameters.items()
            if key in section and section[key] != value
        }


# ── State ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PaperState:
    balance:            float
    position:           Optional[Position]
    last_candle_start:  int
    realized_trades:    int
    # Realized profit is folded straight into balance, so the closed trades
    # themselves are not recoverable from a resumed book — a running win count
    # is the only way a win rate survives a restart.
    realized_wins:      int = 0


class PaperStateFile:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def exists(self) -> bool:
        return os.path.exists(self._filepath)

    def read(self) -> PaperState:
        with open(self._filepath, encoding="utf-8") as handle:
            raw = json.load(handle)
        return PaperState(
            balance           = float(raw["balance"]),
            position          = self._position(raw.get("position")),
            last_candle_start = int(raw.get("last_candle_start", 0)),
            realized_trades   = int(raw.get("realized_trades", 0)),
            realized_wins     = int(raw.get("realized_wins", 0)),
        )

    def write(self, state: PaperState, pair: str) -> None:
        ParentDirectory(self._filepath).ensure()
        payload = {
            "pair":              pair,
            "balance":           state.balance,
            "position":          self._serialized(state.position),
            "last_candle_start": state.last_candle_start,
            "realized_trades":   state.realized_trades,
            "realized_wins":     state.realized_wins,
            "updated_at":        UtcNow().iso(),
        }
        # Atomic: a crash mid-write must never leave a truncated book behind.
        tmp_path = f"{self._filepath}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self._filepath)

    @staticmethod
    def _position(raw: Optional[dict[str, Any]]) -> Optional[Position]:
        if not raw:
            return None
        return Position(
            entry_price = float(raw["entry_price"]),
            size        = float(raw["size"]),
            direction   = Direction(raw["direction"]),
        )

    @staticmethod
    def _serialized(position: Optional[Position]) -> Optional[dict[str, Any]]:
        if position is None:
            return None
        return {
            "entry_price": position.entry_price(),
            "size":        position.size(),
            "direction":   position.direction().value,
        }


# Where a tick's book comes from and goes back to. PaperStateFile satisfies it
# by reading and writing disk on every tick; a long-lived engine substitutes an
# in-memory store that only touches disk when it chooses to snapshot.

class StateStore(Protocol):
    def exists(self) -> bool: ...

    def read(self) -> PaperState: ...

    def write(self, state: PaperState, pair: str) -> None: ...


class InitialPaperState:
    def __init__(self, starting_balance: float) -> None:
        self._starting_balance = starting_balance

    def state(self) -> PaperState:
        return PaperState(
            balance           = self._starting_balance,
            position          = None,
            last_candle_start = 0,
            realized_trades   = 0,
            realized_wins     = 0,
        )


# ── Fees ───────────────────────────────────────────────────────────────
# Trade.profit() stays gross: every saved strategy's recorded performance, and
# every row in experiments/index.csv, is denominated in it, so charging fees
# there would silently invalidate comparisons across the whole history.
# A paper run charges on top instead, leaving the backtest arithmetic alone.
#
# Charged on notional, not on a Decision: a closing decision (SELL/COVER)
# carries size 0.0 because the Ledger closes whatever is open, so pricing off
# decision.size would silently charge entries only and halve the real cost.
#
# Approximations worth knowing: a liquidation close is not charged, and
# slippage is not modelled at all.

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


# ── Tick ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TickOutcome:
    acted:        bool
    candle_start: int
    decision:     Optional[Decision]
    balance:      float
    equity:       float
    closed_trades: int
    # The candle row the decision was taken on — carried so a caller can report
    # the indicator values behind a decision without fetching them again.
    row:          dict[str, float] = field(default_factory=dict)
    fee:          float = 0.0
    # The position the decision was taken AGAINST, not the one it produced —
    # what a strategy scored, and so what explains the action it chose.
    position_before: Optional[Position] = None


class PaperTick:
    def __init__(
        self,
        rows: ClosedMarketRow,
        strategy: Strategy,
        state_file: StateStore,
        starting_balance: float,
        fees: FeeSchedule = NoFees(),
    ) -> None:
        self._rows             = rows
        self._strategy         = strategy
        self._state_file       = state_file
        self._starting_balance = starting_balance
        self._fees             = fees

    async def run(self) -> TickOutcome:
        state = (
            self._state_file.read() if self._state_file.exists()
            else InitialPaperState(self._starting_balance).state()
        )
        row          = await self._rows.latest()
        candle_start = int(row["timestamp"])

        if candle_start <= state.last_candle_start:
            return TickOutcome(
                acted=False, candle_start=candle_start, decision=None,
                balance=state.balance, equity=self._equity(state, row["close"]),
                closed_trades=0, row=row, position_before=state.position,
            )

        ledger = Ledger(state.balance, state.position)
        # Same order as Backtest.run(): a position carried in is liquidation
        # checked against this candle's range before a new decision is taken.
        ledger.liquidate(row["high"], row["low"])
        before   = ledger.position()
        decision = self._strategy.decide(row, ledger.position(), ledger.balance())
        ledger.apply(decision, row["close"])

        fee     = self._fees.charge(self._notional(before, ledger.position(), row["close"]))
        balance = ledger.balance() - fee
        self._state_file.write(
            PaperState(
                balance           = balance,
                position          = ledger.position(),
                last_candle_start = candle_start,
                realized_trades   = state.realized_trades + len(ledger.trades()),
                realized_wins     = state.realized_wins + sum(
                    1 for trade in ledger.trades() if trade.profit() > 0.0
                ),
            ),
            self._rows.pair(),
        )
        return TickOutcome(
            acted=True, candle_start=candle_start, decision=decision,
            balance=balance, equity=ledger.equity(row["close"]) - fee,
            closed_trades=len(ledger.trades()), row=row, fee=fee,
            position_before=before,
        )

    # What actually changed hands this tick: the size opened, or the size
    # closed. A tick that neither opened nor closed anything trades nothing.
    @staticmethod
    def _notional(before: Optional[Position], after: Optional[Position], price: float) -> float:
        if before is None and after is not None:
            return after.size() * price
        if before is not None and after is None:
            return before.size() * price
        return 0.0

    @staticmethod
    def _equity(state: PaperState, price: float) -> float:
        if state.position is None:
            return state.balance
        return state.balance + state.position.unrealized(price)


# ── Console reporting ──────────────────────────────────────────────────

class ConsoleTickReport:
    def __init__(self, outcome: TickOutcome, log: DryRunLog, pair: str) -> None:
        self._outcome = outcome
        self._log     = log
        self._pair    = pair

    def emit(self) -> None:
        # Deliberately ASCII: this goes to a log file that Task Scheduler writes
        # under the system codepage and that any tool may read back.
        if not self._outcome.acted:
            print(
                f"no new closed candle (last acted {self._outcome.candle_start}) | "
                f"balance {self._outcome.balance:.2f} equity {self._outcome.equity:.2f}"
            )
            return
        timestamp = UtcNow().iso()
        self._log.append(
            timestamp, self._outcome.decision, self._outcome.balance, self._outcome.equity,
        )
        print(
            f"{timestamp}  {self._pair}  candle {self._outcome.candle_start}  "
            f"{self._outcome.decision.action.value:<5} size {self._outcome.decision.size:>12.6f}  "
            f"balance {self._outcome.balance:>12.2f}  equity {self._outcome.equity:>12.2f}"
            + (f"  closed {self._outcome.closed_trades} trade(s)" if self._outcome.closed_trades else "")
        )


# ── Entry point ────────────────────────────────────────────────────────

async def _main() -> None:
    raw_config      = ConfigFile("coinbase/ga/config.yaml").raw()
    paper_config    = PaperConfigFile(raw_config).config()
    market_config   = MarketDataConfig(raw_config)
    window          = market_config.window()
    output_config   = OutputConfigFile(raw_config).config()
    weight_keys     = ValidatedWeightKeys(
        WeightKeysConfig(raw_config).keys(), market_config.normalized_columns(),
    ).keys()

    saved           = StrategyJsonFile(output_config.strategy_filepath)
    trained         = TrainedStrategyConfig(raw_config, saved.hyperparameters())
    strategy_config = trained.config()

    for key, (from_config, from_strategy) in trained.divergences().items():
        print(f"note: {key} config.yaml={from_config} -> using trained {from_strategy}")

    keys       = weight_keys + (POSITION_PNL_KEY,)
    # A genome saved before a weight key existed carries no weight for it, and
    # Genome.weight raises rather than guessing. Backfilling at zero keeps a
    # book already running against an older genome from breaking the moment
    # config.yaml grows a column.
    backfilled = BackfilledGenome(Genome(saved.weights()), keys)
    for key in backfilled.missing():
        print(f"note: genome predates {key} — running it weighted zero; retrain to use it")

    strategy = GaStrategy(backfilled.filled(), strategy_config, keys)

    # The paper run follows its own market, which need not be the one the
    # genome was trained on — ConfiguredExchange is asked for that exchange
    # rather than data.exchange.
    async with ConfiguredExchange({"data": {"exchange": paper_config.exchange}}).adapter() as adapter:
        rows = ClosedMarketRow(LiveMarketRow(
            adapter, paper_config.pair, window.granularity,
            market_config.periods(), market_config.normalized_columns(),
        ))
        outcome = await PaperTick(
            rows, strategy,
            PaperStateFile(paper_config.state_filepath),
            strategy_config.starting_balance,
        ).run()

    ConsoleTickReport(
        outcome, DryRunLog(output_config.dry_run_log_filepath), paper_config.pair,
    ).emit()


if __name__ == "__main__":
    asyncio.run(_main())
