"""
Multi-pair paper trading — many algos, one process, live dashboard.
--------------------------------------------------------------------

`paper_trading.py` runs one tick for one pair and exits. This holds a process
open and runs N algos, each driving its own trained genome against its own
pair, and serves a dashboard showing what every one of them is doing.

Two loops run concurrently:

  Decision loop.  Wakes on each candle boundary, ticks every algo, journals the
  outcome and snapshots every book. This is the only thing that ever changes a
  book.

  Price loop.  Polls best bid every price_refresh_seconds and marks open
  positions to market, so unrealized PnL on the dashboard is live rather than
  up to a granularity stale. It never mutates a book — a liquidation breached
  between ticks is caught by the next decision tick, which checks the closed
  candle's full range.

Books live in memory (PaperBook) and are flushed to the same per-algo
PaperStateFile format paper_trading.py uses, so a restart resumes open
positions rather than silently starting flat. PaperTick's per-candle
idempotence means a restart mid-interval cannot re-trade a candle already acted
on.

Nothing here places an order or reads a real balance: every ledger is
simulated, seeded from its algo's configured starting_balance. The adapters are
only ever asked for candles and prices.

Run (as a module, from the repo root):
    python -m coinbase.ga.paper_engine
    python -m coinbase.ga.paper_engine --once
"""

import argparse
import asyncio
import functools
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

from coinbase.ga.config import GA_RESULTS_ROOT, ConfigFile
from coinbase.ga.ga_engine import Genome
from coinbase.ga.market_data_processor import MarketDataConfig
from coinbase.ga.paper_metrics import (
    AlgoPerformance,
    AlgoStatus,
    EquityCurve,
    PortfolioStatus,
    PositionView,
    StatusPayload,
)
from coinbase.ga.paper_trading import (
    BasisPointFee,
    FeeSchedule,
    InitialPaperState,
    NoFees,
    PaperState,
    PaperStateFile,
    PaperTick,
    TickOutcome,
    TrainedStrategyConfig,
)
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    GaStrategy,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.ga.strategy_output import DryRunLog, ParentDirectory, StrategyJsonFile, UtcNow
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.strategy import ClosedMarketRow, LiveMarketRow, RetriedMarketRow
from coinbase.trading_strategy import Direction, IsolatedMargin, Position
from exchange.pool import ExchangeLane, ExchangePool

logger = logging.getLogger(__name__)

PAPER_ROOT = GA_RESULTS_ROOT / "paper"


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AlgoConfig:
    name:              str
    exchange:          str
    pair:              str
    granularity:       str
    strategy_filepath: str
    starting_balance:  float
    state_filepath:    str
    log_filepath:      str


@dataclass(frozen=True)
class EngineConfig:
    algos:                   tuple[AlgoConfig, ...]
    granularity:             str
    boundary_offset_seconds: int
    price_refresh_seconds:   int
    max_concurrent_requests: int
    fee_bps:                 float
    journal_filepath:        str
    host:                    str
    port:                    int


class PaperEngineConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def ga_config_path(self) -> str:
        return self._raw.get("ga_config", "coinbase/ga/config.yaml")

    def config(self) -> EngineConfig:
        # `or {}`: a section whose every child is commented out parses to None.
        engine    = self._raw.get("engine") or {}
        dashboard = self._raw.get("dashboard") or {}
        state_dir = engine.get("state_dir") or str(PAPER_ROOT / "state")
        log_dir   = engine.get("log_dir") or str(PAPER_ROOT / "logs")
        return EngineConfig(
            algos                   = self._algos(state_dir, log_dir, engine),
            granularity             = engine.get("granularity", "THIRTY_MINUTE"),
            boundary_offset_seconds = int(engine.get("boundary_offset_seconds", 20)),
            price_refresh_seconds   = int(engine.get("price_refresh_seconds", 45)),
            max_concurrent_requests = int(engine.get("max_concurrent_requests", 8)),
            fee_bps                 = float(engine.get("fee_bps", 0.0)),
            journal_filepath        = os.path.join(log_dir, "journal.tsv"),
            host                    = dashboard.get("host", "127.0.0.1"),
            port                    = int(dashboard.get("port", 8787)),
        )

    def _algos(self, state_dir: str, log_dir: str, engine: dict[str, Any]) -> tuple[AlgoConfig, ...]:
        defaults    = self._raw.get("defaults") or {}
        granularity = engine.get("granularity", "THIRTY_MINUTE")
        algos       = []
        for entry in self._raw.get("algos") or []:
            merged = {**defaults, **entry}
            name   = merged["name"]
            algos.append(AlgoConfig(
                name              = name,
                exchange          = merged.get("exchange", "coinbase"),
                pair              = merged["pair"],
                granularity       = merged.get("granularity", granularity),
                strategy_filepath = os.path.expanduser(merged["strategy_filepath"]),
                starting_balance  = float(merged.get("starting_balance", 10000.0)),
                state_filepath    = os.path.expanduser(
                    merged.get("state_filepath", os.path.join(state_dir, f"{name}.json")),
                ),
                log_filepath      = os.path.expanduser(
                    merged.get("log_filepath", os.path.join(log_dir, name, "decisions.tsv")),
                ),
            ))
        self._reject_duplicates(tuple(algo.name for algo in algos))
        return tuple(algos)

    # Names key the state files, so two algos sharing one would silently share
    # a book and overwrite each other's positions.
    @staticmethod
    def _reject_duplicates(names: tuple[str, ...]) -> None:
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate algo name {name!r} in paper config")
            seen.add(name)


# ── In-memory book ─────────────────────────────────────────────────────
# Satisfies paper_trading.StateStore, so PaperTick drives it unchanged. The
# difference from PaperStateFile is when disk is touched: read once at startup
# to resume, then never again until snapshot() is asked for it.

class PaperBook:
    # `pair` is carried from config rather than only learned from a write: a
    # process that resumes and whose first tick is a no-op never calls write(),
    # and would otherwise snapshot the resumed book back with an empty pair.
    def __init__(self, file: PaperStateFile, starting_balance: float, pair: str) -> None:
        self._file             = file
        self._starting_balance = starting_balance
        self._pair             = pair
        self._state: Optional[PaperState] = None

    def exists(self) -> bool:
        return True

    def read(self) -> PaperState:
        if self._state is None:
            self._state = (
                self._file.read() if self._file.exists()
                else InitialPaperState(self._starting_balance).state()
            )
        return self._state

    def write(self, state: PaperState, pair: str) -> None:
        self._state = state
        self._pair  = pair

    def snapshot(self) -> None:
        if self._state is not None:
            self._file.write(self._state, self._pair)


# ── Clock ──────────────────────────────────────────────────────────────
# Pure arithmetic, no sleeping — so the schedule can be tested without waiting
# for one. `offset_seconds` waits past the boundary before fetching, because a
# candle is not queryable the instant it closes.

class CandleBoundary:
    def __init__(self, granularity: str, offset_seconds: int = 20) -> None:
        self._granularity    = granularity
        self._offset_seconds = offset_seconds

    def next_at(self, now: float) -> float:
        seconds = self.period()
        current = (int(now) // seconds) * seconds
        target  = current + self._offset_seconds
        if target > now:
            return target
        return current + seconds + self._offset_seconds

    def seconds_until(self, now: float) -> float:
        return self.next_at(now) - now

    def period(self) -> int:
        return GRANULARITY_SECONDS[self._granularity]

    # Start of the most recent candle that has actually closed. A candle
    # stamped T covers [T, T+period), so mid-interval the newest closed one
    # began a full period before the candle now forming.
    def closed_candle_start(self, now: float) -> int:
        seconds = self.period()
        return (int(now) // seconds) * seconds - seconds


# ── Live prices ────────────────────────────────────────────────────────
# One request per venue for every pair on it, not one per pair. Deliberately
# not a candle fetch: no indicator frame, no normalization, nothing cached.

class PriceMarks:
    def __init__(self, exchange: str, lane: ExchangeLane, pairs: tuple[str, ...]) -> None:
        self._exchange = exchange
        self._lane     = lane
        self._pairs    = pairs

    def exchange(self) -> str:
        return self._exchange

    # Keyed by (exchange, pair), not pair alone: the same product id trades on
    # both venues at different prices, and merging them by pair would let one
    # venue's book drive the other's unrealized PnL and liquidation warning.
    async def prices(self) -> dict[tuple[str, str], float]:
        books = await self._lane.adapter().get_best_bid_ask(*self._pairs)
        return {
            (self._exchange, book["product_id"]): float(book["bids"][0]["price"])
            for book in books.get("pricebooks", [])
            if book.get("bids") and book.get("product_id")
        }


# ── Algo ───────────────────────────────────────────────────────────────

class PaperAlgo:
    def __init__(
        self,
        config: AlgoConfig,
        rows: RetriedMarketRow,
        strategy: GaStrategy,
        book: PaperBook,
        fees: FeeSchedule,
        curve: EquityCurve,
        log: DryRunLog,
    ) -> None:
        self._config   = config
        self._rows     = rows
        self._strategy = strategy
        self._book     = book
        self._fees     = fees
        self._curve    = curve
        self._log      = log
        self._outcome: Optional[TickOutcome] = None
        self._last_tick_at: Optional[str]    = None
        self._mark_price: float              = 0.0
        self._fee_paid: float                = 0.0

    def config(self) -> AlgoConfig:
        return self._config

    async def tick(self) -> None:
        outcome = await PaperTick(
            self._rows, self._strategy, self._book,
            self._config.starting_balance, self._fees,
        ).run()
        self._outcome      = outcome
        self._last_tick_at = UtcNow().iso()
        if outcome.acted:
            self._fee_paid += outcome.fee
            self._mark_price = outcome.row["close"]
            self._log.append(
                self._last_tick_at, outcome.decision, outcome.balance, outcome.equity,
            )
        self._curve.record(self.status().equity)

    # Mark-to-market only. The book is never touched here.
    def mark(self, price: float) -> None:
        self._mark_price = price
        self._curve.record(self.status().equity)

    def status(self) -> AlgoStatus:
        state    = self._book.read()
        price    = self._mark_price or self._entry_price(state)
        position = self._position_view(state, price)
        return AlgoStatus(
            name              = self._config.name,
            exchange          = self._config.exchange,
            pair              = self._config.pair,
            granularity       = self._config.granularity,
            running           = True,
            error             = None,
            last_tick_at      = self._last_tick_at,
            last_candle_start = state.last_candle_start,
            last_action       = self._last_action(),
            starting_balance  = self._config.starting_balance,
            balance           = state.balance,
            mark_price        = price,
            equity            = state.balance + (position.unrealized if position else 0.0),
            position          = position,
            realized_pnl      = state.balance - self._config.starting_balance,
            unrealized_pnl    = position.unrealized if position else 0.0,
            trades            = state.realized_trades,
            wins              = state.realized_wins,
            rsi               = self._row_value("rsi"),
            macd              = self._row_value("macd"),
            signal_score      = self._signal_score(),
            fee_paid          = self._fee_paid,
        )

    def _last_action(self) -> str:
        if self._outcome is None or self._outcome.decision is None:
            return "-"
        return self._outcome.decision.action.value

    def _row_value(self, key: str) -> float:
        if self._outcome is None:
            return 0.0
        return float(self._outcome.row.get(key, 0.0))

    # Scored against the position the decision was taken with, not the one it
    # produced. GaStrategy weights an open position's unrealized return into
    # the score, so recomputing after a SELL closed the book would report a
    # different number than the one that caused the sell.
    def _signal_score(self) -> float:
        if self._outcome is None or not self._outcome.row:
            return 0.0
        return self._strategy.signal_score(self._outcome.row, self._outcome.position_before)

    @staticmethod
    def _entry_price(state: PaperState) -> float:
        return state.position.entry_price() if state.position is not None else 0.0

    @staticmethod
    def _position_view(state: PaperState, price: float) -> Optional[PositionView]:
        position = state.position
        if position is None:
            return None
        return PositionView(
            direction         = position.direction().value,
            size              = position.size(),
            entry_price       = position.entry_price(),
            unrealized        = position.unrealized(price),
            unrealized_return = position.unrealized_return(price),
            liquidation_price = IsolatedMargin(position, state.balance).liquidation_price(),
        )


# ── Failure isolation ──────────────────────────────────────────────────
# One algo's exchange going down must not stop the others, and must not kill
# the loop. Making failure a value (rather than gathering with
# return_exceptions) keeps the convention used by PortfolioPnl and
# market_scanner: a broken member reports an error field, everything else
# carries on.

class IsolatedAlgo:
    def __init__(self, algo: PaperAlgo) -> None:
        self._algo = algo
        self._error: Optional[str] = None

    def config(self) -> AlgoConfig:
        return self._algo.config()

    async def tick(self) -> None:
        try:
            await self._algo.tick()
            self._error = None
        except Exception as exc:
            self._error = str(exc)
            logger.exception("algo %s failed to tick", self._algo.config().name)

    def mark(self, price: float) -> None:
        try:
            self._algo.mark(price)
        except Exception as exc:
            self._error = str(exc)
            logger.exception("algo %s failed to mark", self._algo.config().name)

    # The read path is isolated too, not just tick/mark: the dashboard builds
    # every algo's status in one pass, so one unreadable book raising here
    # would blank the whole page instead of one row.
    def status(self) -> AlgoStatus:
        try:
            status = self._algo.status()
        except Exception as exc:
            logger.exception("algo %s failed to report status", self._algo.config().name)
            return self._unreadable(str(exc))
        if self._error is None:
            return status
        return replace(status, running=False, error=self._error)

    def _unreadable(self, error: str) -> AlgoStatus:
        entry = self._algo.config()
        return AlgoStatus(
            name=entry.name, exchange=entry.exchange, pair=entry.pair,
            granularity=entry.granularity, running=False, error=error,
            last_tick_at=None, last_candle_start=0, last_action="-",
            starting_balance=entry.starting_balance, balance=0.0, mark_price=0.0,
            equity=0.0, position=None, realized_pnl=0.0, unrealized_pnl=0.0,
            trades=0, wins=0, rsi=0.0, macd=0.0, signal_score=0.0, fee_paid=0.0,
        )


# ── Logging ────────────────────────────────────────────────────────────

class PaperJournal:
    _COLUMNS = (
        "timestamp", "algo", "exchange", "pair", "candle_start", "action",
        "price", "fee", "balance", "equity", "position_side", "position_size",
        "entry_price", "unrealized", "realized", "trades", "error",
    )

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def append(self, timestamp: str, status: AlgoStatus) -> None:
        ParentDirectory(self._filepath).ensure()
        # Header once, on creation — 17 unlabelled columns are unreadable, and
        # appending to an existing file must never insert one mid-stream.
        if not os.path.exists(self._filepath):
            with open(self._filepath, "w", encoding="utf-8") as handle:
                handle.write("\t".join(self._COLUMNS) + "\n")
        position = status.position
        row = (
            timestamp, status.name, status.exchange, status.pair,
            str(status.last_candle_start), status.last_action,
            f"{status.mark_price:.8f}", f"{status.fee_paid:.8f}",
            f"{status.balance:.6f}", f"{status.equity:.6f}",
            position.direction if position else "-",
            f"{position.size:.8f}" if position else "0",
            f"{position.entry_price:.8f}" if position else "0",
            f"{status.unrealized_pnl:.6f}", f"{status.realized_pnl:.6f}",
            str(status.trades), self._cell(status.error or ""),
        )
        with open(self._filepath, "a", encoding="utf-8") as handle:
            handle.write("\t".join(row) + "\n")

    # An ExchangeError carries the raw response body, which for a gateway error
    # is often multi-line HTML. Written raw it would split one record across
    # several lines and silently corrupt the whole file.
    @staticmethod
    def _cell(value: str) -> str:
        return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")[:500]


class BookSnapshot:
    def __init__(self, books: tuple[PaperBook, ...]) -> None:
        self._books = books

    def flush(self) -> None:
        for book in self._books:
            book.snapshot()


# ── Loops ──────────────────────────────────────────────────────────────

class DecisionLoop:
    def __init__(
        self,
        algos: tuple[IsolatedAlgo, ...],
        boundary: CandleBoundary,
        journal: PaperJournal,
        snapshot: BookSnapshot,
        catch_up_seconds: int = 60,
    ) -> None:
        self._algos            = algos
        self._boundary         = boundary
        self._journal          = journal
        self._snapshot         = snapshot
        self._catch_up_seconds = catch_up_seconds

    async def once(self) -> None:
        await asyncio.gather(*(algo.tick() for algo in self._algos))
        timestamp = UtcNow().iso()
        for algo in self._algos:
            self._journal.append(timestamp, algo.status())
        self._snapshot.flush()

    # Sleeps first, then works, recomputing the wait from the clock every time —
    # so a slow tick eats into the gap instead of shifting every later tick, the
    # drift a sleep-after-work loop accumulates.
    async def forever(self) -> None:
        while True:
            await asyncio.sleep(self._boundary.seconds_until(time.time()))
            try:
                await self.once()
                await self.catch_up()
            except Exception:
                logger.exception("decision loop tick failed")

    # An exchange that has not published the just-closed candle by the time the
    # offset elapses would otherwise lose that candle for good: the fetch
    # succeeds, returns the PREVIOUS candle, and PaperTick correctly no-ops on
    # it — no exception, so RetriedMarketRow never sees anything to retry, and
    # the next attempt is a whole period too late. Re-ticking until every algo
    # has the candle (or the period runs out) restores the slack that scheduling
    # more often than the granularity used to provide. Ticks are idempotent, so
    # an algo already caught up just re-reads its book.
    async def catch_up(self) -> None:
        while self._behind(time.time()):
            if self._boundary.seconds_until(time.time()) <= self._catch_up_seconds:
                logger.warning("gave up waiting for the closed candle this period")
                return
            await asyncio.sleep(self._catch_up_seconds)
            await self.once()

    def _behind(self, now: float) -> bool:
        expected = self._boundary.closed_candle_start(now)
        return any(
            algo.status().last_candle_start < expected
            for algo in self._algos
            if algo.status().error is None
        )


class PriceLoop:
    def __init__(
        self,
        marks: tuple[PriceMarks, ...],
        algos: tuple[IsolatedAlgo, ...],
        interval_seconds: int,
    ) -> None:
        self._marks    = marks
        self._algos    = algos
        self._interval = interval_seconds
        self._marked_at: float = 0.0

    def marked_at(self) -> float:
        return self._marked_at

    # One venue being down must not cost every other venue its refresh, so each
    # is gathered independently and a failure is logged rather than raised —
    # the same isolation IsolatedAlgo gives the decision loop.
    async def once(self) -> None:
        books = await asyncio.gather(
            *(mark.prices() for mark in self._marks), return_exceptions=True,
        )
        prices: dict[tuple[str, str], float] = {}
        for mark, book in zip(self._marks, books):
            if isinstance(book, BaseException):
                logger.warning("price refresh failed for %s: %s", mark.exchange(), book)
                continue
            prices.update(book)
        if not prices:
            return
        for algo in self._algos:
            price = prices.get((algo.config().exchange, algo.config().pair))
            if price is not None:
                algo.mark(price)
        self._marked_at = time.time()

    async def forever(self) -> None:
        while True:
            try:
                await self.once()
            except Exception:
                logger.exception("price refresh failed")
            await asyncio.sleep(self._interval)


# ── Read model ─────────────────────────────────────────────────────────
# status() and payload() are synchronous and pure. On a single-threaded event
# loop that means a dashboard request cannot interleave with a tick's await
# points, so it always observes a whole book — never one caught mid-update.
# Do not make these async; the absence of locking depends on it.

class StatusBoard:
    def __init__(
        self,
        algos: tuple[IsolatedAlgo, ...],
        curves: dict[str, EquityCurve],
        boundary: CandleBoundary,
        prices: PriceLoop,
        started_at: float,
    ) -> None:
        self._algos      = algos
        self._curves     = curves
        self._boundary   = boundary
        self._prices     = prices
        self._started_at = started_at

    def payload(self) -> dict[str, Any]:
        now      = time.time()
        statuses = tuple(algo.status() for algo in self._algos)
        elapsed  = now - self._started_at
        return StatusPayload(
            statuses            = statuses,
            performances        = tuple(
                AlgoPerformance(status, self._curves[status.name], elapsed)
                for status in statuses
            ),
            portfolio           = PortfolioStatus(statuses),
            started_at          = datetime.fromtimestamp(
                self._started_at, tz=timezone.utc,
            ).isoformat(),
            generated_at        = UtcNow().iso(),
            next_tick_in        = self._boundary.seconds_until(now),
            seconds_since_price = now - self._prices.marked_at() if self._prices.marked_at() else -1.0,
        ).as_dict()


# ── Assembly ───────────────────────────────────────────────────────────

class PaperEngine:
    def __init__(self, config: EngineConfig, ga_raw: dict[str, Any], pool: ExchangePool) -> None:
        self._config = config
        self._ga_raw = ga_raw
        self._pool   = pool

    @functools.cached_property
    def algos(self) -> tuple[IsolatedAlgo, ...]:
        return tuple(IsolatedAlgo(self._algo(entry)) for entry in self._config.algos)

    # Books and curves are shared between the algos, the snapshotter and the
    # status board, so every caller must see the same instance per name.
    @functools.cached_property
    def curves(self) -> dict[str, EquityCurve]:
        return {entry.name: EquityCurve() for entry in self._config.algos}

    @functools.cached_property
    def books(self) -> dict[str, PaperBook]:
        return {
            entry.name: PaperBook(
                PaperStateFile(entry.state_filepath), entry.starting_balance, entry.pair,
            )
            for entry in self._config.algos
        }

    @functools.cached_property
    def marks(self) -> tuple[PriceMarks, ...]:
        by_exchange: dict[str, list[str]] = {}
        for entry in self._config.algos:
            by_exchange.setdefault(entry.exchange, []).append(entry.pair)
        return tuple(
            PriceMarks(name, self._pool.lane(name), tuple(dict.fromkeys(pairs)))
            for name, pairs in by_exchange.items()
        )

    def _algo(self, entry: AlgoConfig) -> PaperAlgo:
        market  = MarketDataConfig(self._ga_raw)
        lane    = self._pool.lane(entry.exchange)
        saved   = StrategyJsonFile(entry.strategy_filepath)
        trained = TrainedStrategyConfig(self._ga_raw, saved.hyperparameters())
        keys    = ValidatedWeightKeys(
            WeightKeysConfig(self._ga_raw).keys(), market.normalized_columns(),
        ).keys()
        rows = RetriedMarketRow(ClosedMarketRow(LiveMarketRow(
            lane.adapter(), entry.pair, entry.granularity,
            market.periods(), market.columns(), limit=lane.limit(),
        )))
        return PaperAlgo(
            config   = entry,
            rows     = rows,
            strategy = GaStrategy(
                Genome(saved.weights()), trained.config(), keys + (POSITION_PNL_KEY,),
            ),
            book     = self.books[entry.name],
            fees     = self._fees(),
            curve    = self.curves[entry.name],
            log      = DryRunLog(entry.log_filepath),
        )

    def _fees(self) -> FeeSchedule:
        if self._config.fee_bps <= 0.0:
            return NoFees()
        return BasisPointFee(self._config.fee_bps)


# ── Entry point ────────────────────────────────────────────────────────

class EngineArguments:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def parsed(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Multi-pair paper trading engine")
        parser.add_argument("--config", default="coinbase/ga/paper.yaml")
        parser.add_argument("--once", action="store_true", help="one decision round, then exit")
        return parser.parse_args(self._argv)


async def _main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args   = EngineArguments(argv if argv is not None else sys.argv[1:]).parsed()
    file   = PaperEngineConfigFile(ConfigFile(args.config).raw())
    config = file.config()
    ga_raw = ConfigFile(file.ga_config_path()).raw()

    if not config.algos:
        raise ValueError(f"no algos configured in {args.config}")
    # Checked here rather than on first use: GRANULARITY_SECONDS is not read
    # until the loop's first sleep, so a typo would otherwise kill the process
    # a full period after it appeared to start cleanly.
    for granularity in {config.granularity, *(a.granularity for a in config.algos)}:
        if granularity not in GRANULARITY_SECONDS:
            raise ValueError(
                f"unknown granularity {granularity!r} — expected one of "
                f"{', '.join(sorted(GRANULARITY_SECONDS))}"
            )
    # A finer per-algo granularity would only ever be polled at the engine's
    # rate, silently dropping most of its candles.
    for entry in config.algos:
        if GRANULARITY_SECONDS[entry.granularity] < GRANULARITY_SECONDS[config.granularity]:
            raise ValueError(
                f"algo {entry.name!r} runs on {entry.granularity}, finer than the engine's "
                f"{config.granularity} — it would miss candles; lower engine.granularity"
            )

    exchanges = tuple(entry.exchange for entry in config.algos)
    async with ExchangePool(exchanges, config.max_concurrent_requests) as pool:
        engine   = PaperEngine(config, ga_raw, pool)
        algos    = engine.algos
        boundary = CandleBoundary(config.granularity, config.boundary_offset_seconds)
        prices   = PriceLoop(engine.marks, algos, config.price_refresh_seconds)
        snapshot = BookSnapshot(tuple(engine.books.values()))
        decisions = DecisionLoop(algos, boundary, PaperJournal(config.journal_filepath), snapshot)
        board     = StatusBoard(algos, engine.curves, boundary, prices, time.time())

        from coinbase.ga.paper_dashboard import DashboardApp, DashboardSite
        site = DashboardSite(DashboardApp(board), config.host, config.port)
        await site.start()
        print(f"Dashboard: http://{config.host}:{config.port}")
        for entry in config.algos:
            print(f"  {entry.name:<12} {entry.exchange:<9} {entry.pair:<10} {entry.granularity}")

        try:
            # Guarded like the loops themselves: the exchange being briefly
            # unavailable at startup is the same transient failure that would
            # merely be logged a minute later, and must not kill the process.
            try:
                await prices.once()
                await decisions.once()
            except Exception:
                logger.exception("startup tick failed; continuing")
            if not args.once:
                await asyncio.gather(decisions.forever(), prices.forever())
        finally:
            snapshot.flush()
            await site.stop()


if __name__ == "__main__":
    asyncio.run(_main())
