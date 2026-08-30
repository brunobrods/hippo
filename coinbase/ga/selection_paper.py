"""
Paper trading for the cross-sectional selector — ONE tick per invocation.
--------------------------------------------------------------------------

The selector equivalent of `paper_trading.py`: runs a single tick against the
last CLOSED candle and exits, so Windows Task Scheduler can drive it instead
of a process left open. The same two properties make that safe.

  State persists.  Balance, which pair is held, and the open position are
  written to `selection_state.json` after every tick. The held PAIR is the
  part `paper_trading` never had to store — a selector's book is meaningless
  without knowing which market the position is in, and reloading a size and
  an entry price against the wrong pair would mark the book at a completely
  unrelated price.

  Idempotent per candle.  A tick records `last_candle_start` and refuses to
  act on that candle twice, so scheduling it far more often than the
  granularity is free.

Decisions are taken on the last closed candle of every pair, and only on
timestamps every pair shares — comparing a fresh score against a stale one
would rank on staleness rather than strength, exactly as in the backtest.

Usage (run as a module, from the repo root):
    python -m coinbase.ga.selection_paper
    python -m coinbase.ga.selection_paper --pairs FET-USDT ETH-USDT
    python -m coinbase.ga.selection_paper --from-shortlist binance
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from coinbase.ga.config import GA_RESULTS_ROOT, ConfigFile
from coinbase.ga.market_data_processor import MarketDataConfig
from coinbase.ga.pair_selector import (
    Candidate,
    Conviction,
    PairRanking,
    TrainedPair,
    TrainedPairs,
)
from coinbase.ga.paper_trading import BasisPointFee, FeeSchedule, NoFees
from coinbase.ga.strategy_output import ParentDirectory
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.strategy import ClosedMarketRow, LiveMarketRow, RetriedMarketRow
from coinbase.trading_strategy import Action, Direction, Ledger, Position
from exchange.pool import ExchangePool

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelectionState:
    balance:           float
    # The pair the open position is in. Meaningless to omit: a size and an
    # entry price restored against the wrong market would be marked at an
    # unrelated price, and the selector would then decide whether to exit
    # using a different pair's genome entirely.
    held_pair:         Optional[str]
    position:          Optional[Position]
    last_candle_start: int
    realized_trades:   int
    realized_wins:     int = 0


class SelectionStateFile:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def exists(self) -> bool:
        return os.path.exists(self._filepath)

    def read(self) -> SelectionState:
        with open(self._filepath, encoding="utf-8") as handle:
            raw = json.load(handle)
        return SelectionState(
            balance           = float(raw["balance"]),
            held_pair         = raw.get("held_pair"),
            position          = self._position(raw.get("position")),
            last_candle_start = int(raw.get("last_candle_start", 0)),
            realized_trades   = int(raw.get("realized_trades", 0)),
            realized_wins     = int(raw.get("realized_wins", 0)),
        )

    def write(self, state: SelectionState) -> None:
        ParentDirectory(self._filepath).ensure()
        payload = {
            "balance":           state.balance,
            "held_pair":         state.held_pair,
            "position":          self._serialized(state.position),
            "last_candle_start": state.last_candle_start,
            "realized_trades":   state.realized_trades,
            "realized_wins":     state.realized_wins,
        }
        tmp_path = f"{self._filepath}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self._filepath)  # atomic — never a half-written book

    @staticmethod
    def _position(raw: Optional[dict]) -> Optional[Position]:
        if not raw:
            return None
        return Position(
            float(raw["entry_price"]), float(raw["size"]),
            Direction[raw["direction"]],
        )

    @staticmethod
    def _serialized(position: Optional[Position]) -> Optional[dict]:
        if position is None:
            return None
        return {
            "entry_price": position.entry_price(),
            "size":        position.size(),
            "direction":   position.direction().name,
        }


class InitialSelectionState:
    def __init__(self, starting_balance: float) -> None:
        self._starting_balance = starting_balance

    def state(self) -> SelectionState:
        return SelectionState(self._starting_balance, None, None, 0, 0, 0)


# ── Tick ───────────────────────────────────────────────────────────────

# `exited` and `entered` are separate because a single tick can do both: the
# selector closes a position and re-enters the next-best pair on the same
# candle, exactly as the backtest does. One `action` field would report only
# whichever happened last and quietly hide the other leg.
@dataclass(frozen=True)
class SelectionOutcome:
    acted:        bool
    candle_start: int
    held_pair:    Optional[str]
    exited:       Optional[str]
    entered:      Optional[str]
    action:       Action
    balance:      float
    equity:       float
    ranking:      tuple[Candidate, ...]
    note:         str = ""


class SelectionTick:
    def __init__(
        self,
        rows: dict[str, Any],
        trained: dict[str, TrainedPair],
        state_file: SelectionStateFile,
        starting_balance: float,
        fees: FeeSchedule,
    ) -> None:
        self._rows             = rows
        self._trained          = trained
        self._state_file       = state_file
        self._starting_balance = starting_balance
        self._fees             = fees

    async def run(self) -> SelectionOutcome:
        state  = self._state()
        latest = await self._latest()

        # Only timestamps every pair has closed. One lagging pair holds the
        # whole tick rather than letting a stale score into the ranking.
        candle_start = min(int(row["timestamp"]) for row in latest.values())
        if candle_start <= state.last_candle_start:
            return SelectionOutcome(
                False, candle_start, state.held_pair, None, None, Action.HOLD,
                state.balance, self._equity(state, latest),
                (), "already acted on this candle",
            )

        ledger  = Ledger(state.balance, state.position)
        held    = state.held_pair
        action  = Action.HOLD
        exited: Optional[str]  = None
        entered: Optional[str] = None
        ranking: tuple[Candidate, ...] = ()
        before  = len(ledger.trades())

        if held is not None and held in latest:
            row   = latest[held]
            price = row["close"]
            size  = ledger.position().size() if ledger.position() else 0.0
            ledger.liquidate(row.get("high", price), row.get("low", price))
            if len(ledger.trades()) > before:
                # The exchange closed it, not us — no exit fee, matching PaperTick.
                exited, held, action = held, None, Action.HOLD
            else:
                decision = self._trained[held].strategy().decide(
                    row, ledger.position(), ledger.balance(),
                )
                ledger.apply(decision, price)
                if len(ledger.trades()) > before:
                    ledger.charge(self._fees.charge(size * price))
                    exited, held, action = held, None, decision.action

        if held is None and ledger.position() is None:
            ranking = self._ranked(latest)
            best    = PairRanking(ranking).best()
            if best is not None:
                row      = latest[best.pair]
                decision = self._trained[best.pair].strategy().decide(
                    row, None, ledger.balance(),
                )
                if decision.action is not Action.HOLD:
                    ledger.apply(decision, row["close"])
                    ledger.charge(self._fees.charge(decision.size * row["close"]))
                    entered, held, action = best.pair, best.pair, decision.action

        closed = ledger.trades()
        self._state_file.write(SelectionState(
            balance           = ledger.balance(),
            held_pair         = held,
            position          = ledger.position(),
            last_candle_start = candle_start,
            realized_trades   = state.realized_trades + len(closed),
            realized_wins     = state.realized_wins + sum(1 for t in closed if t.profit() > 0),
        ))
        return SelectionOutcome(
            True, candle_start, held, exited, entered, action, ledger.balance(),
            self._equity_of(ledger, held, latest), ranking,
        )

    def _state(self) -> SelectionState:
        if self._state_file.exists():
            return self._state_file.read()
        return InitialSelectionState(self._starting_balance).state()

    async def _latest(self) -> dict[str, dict[str, float]]:
        pairs   = tuple(self._rows)
        results = await asyncio.gather(*(self._rows[pair].latest() for pair in pairs))
        return dict(zip(pairs, results))

    def _ranked(self, latest: dict[str, dict[str, float]]) -> tuple[Candidate, ...]:
        candidates: list[Candidate] = []
        for pair, row in latest.items():
            strategy   = self._trained[pair].strategy()
            score      = strategy.signal_score(row, None)
            conviction = Conviction(score, self._trained[pair].config, strategy.flat_score_ceiling())
            candidates.append(Candidate(pair, conviction.action(), conviction.value(), score))
        return tuple(candidates)

    def _equity(self, state: SelectionState, latest: dict[str, dict[str, float]]) -> float:
        if state.position is None or state.held_pair not in latest:
            return state.balance
        return state.balance + state.position.unrealized(latest[state.held_pair]["close"])

    @staticmethod
    def _equity_of(ledger: Ledger, held: Optional[str], latest: dict[str, dict[str, float]]) -> float:
        if held is None or held not in latest:
            return ledger.balance()
        return ledger.equity(latest[held]["close"])


class ConsoleSelectionReport:
    def __init__(self, outcome: SelectionOutcome, starting_balance: float) -> None:
        self._outcome          = outcome
        self._starting_balance = starting_balance

    def print(self) -> None:
        outcome = self._outcome
        if not outcome.acted:
            print(f"no-op: {outcome.note} (candle {outcome.candle_start})")
            return
        print(f"candle    {outcome.candle_start}")
        if outcome.exited:
            print(f"exited    {outcome.exited}")
        if outcome.entered:
            print(f"entered   {outcome.entered}  ({outcome.action.name})")
        if not outcome.exited and not outcome.entered:
            print("action    no change")
        print(f"holding   {outcome.held_pair or 'flat'}")
        print(f"balance   {outcome.balance:,.2f}")
        print(f"equity    {outcome.equity:,.2f}   "
              f"({(outcome.equity / self._starting_balance - 1) * 100:+.2f}%)")
        if outcome.ranking:
            print("\nranking this candle:")
            for candidate in PairRanking(outcome.ranking).ranked()[:5]:
                print(f"  {candidate.pair:14s} {candidate.action.name:6s} "
                      f"conviction {candidate.conviction:.3f}  score {candidate.score:.3f}")


# ── Entry point ────────────────────────────────────────────────────────

async def _main(argv: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="One paper tick of the cross-sectional selector")
    p.add_argument("--config", default="coinbase/ga/config.yaml")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--pairs", nargs="+")
    source.add_argument("--from-shortlist", metavar="EXCHANGE", nargs="?", const="binance")
    p.add_argument("--exchange", default=None)
    p.add_argument("--metric", default="annualized_yield")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--no-fees", action="store_true")
    p.add_argument("--starting-balance", type=float, default=None)
    p.add_argument("--state", default=None, help="Path to selection_state.json")
    p.add_argument("--max-concurrent", type=int, default=8)
    args = p.parse_args(argv)

    raw_config = ConfigFile(args.config).raw()
    market     = MarketDataConfig(raw_config)
    window     = market.window()
    exchange   = args.exchange or (raw_config.get("data") or {}).get("exchange", "coinbase")

    if args.pairs:
        pairs = tuple(args.pairs)
    else:
        shortlist = GA_RESULTS_ROOT / "screener" / f"shortlist_{args.from_shortlist or exchange}_latest.json"
        with open(shortlist, encoding="utf-8") as handle:
            pairs = tuple(json.load(handle)["pairs"])

    catalogue = TrainedPairs(raw_config, pairs, args.metric, window.granularity)
    trained   = catalogue.resolved
    if catalogue.missing():
        logger.warning("no %s genome for %s — skipping those",
                       window.granularity, ", ".join(catalogue.missing()))
    if not trained:
        raise ValueError(
            "none of the requested pairs have a trained genome for "
            f"{window.granularity}. Train them with a `data.pair` sweep axis first."
        )

    balance    = args.starting_balance or next(iter(trained.values())).config.starting_balance
    state_file = SelectionStateFile(
        args.state or str(GA_RESULTS_ROOT / "selection_state.json"),
    )

    async with ExchangePool((exchange,), args.max_concurrent) as pool:
        lane = pool.lane(exchange)
        rows = {
            pair: RetriedMarketRow(ClosedMarketRow(LiveMarketRow(
                lane.adapter(), pair, window.granularity,
                market.periods(), market.normalized_columns(),
                limit=lane.limit(),
                index_pairs=market.index_pairs(), index_period=market.index_period(),
            )))
            for pair in trained
        }
        outcome = await SelectionTick(
            rows, trained, state_file, balance,
            NoFees() if args.no_fees else BasisPointFee(args.fee_bps),
        ).run()

    ConsoleSelectionReport(outcome, balance).print()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
