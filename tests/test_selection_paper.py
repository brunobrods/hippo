import pandas as pd
import pytest

from coinbase.ga.ga_engine import Genome
from coinbase.ga.paper_trading import BasisPointFee, NoFees
from coinbase.ga.selection_paper import (
    InitialSelectionState,
    SelectionState,
    SelectionStateFile,
    SelectionTick,
)
from coinbase.ga.strategy_evaluator import GaStrategy, StrategyConfig
from coinbase.trading_strategy import Action, Direction, Position


# ── Test doubles ───────────────────────────────────────────────────────

# Stands in for ClosedMarketRow — hands back one scripted closed candle.
class FakeRows:
    def __init__(self, row: dict[str, float]) -> None:
        self._row = row

    async def latest(self) -> dict[str, float]:
        return self._row


# Stands in for TrainedPair — a genome weighted entirely on norm_rsi, so a
# test dictates the score directly through the row.
class FakeTrained:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def strategy(self) -> GaStrategy:
        return GaStrategy(Genome({"rsi": 1.0}), self.config, ("rsi",))


def _config(buy: float = 0.6, sell: float = 0.4, short: bool = False) -> StrategyConfig:
    return StrategyConfig(
        position_size_pct     = 1.0,
        buy_threshold         = buy,
        sell_threshold        = sell,
        starting_balance      = 1000.0,
        allow_short           = short,
        short_entry_threshold = 0.25,
        short_exit_threshold  = 0.45,
    )


def _row(timestamp: int, close: float, score: float) -> dict[str, float]:
    return {"timestamp": timestamp, "close": close, "high": close, "low": close, "norm_rsi": score}


def _tick(rows: dict[str, dict], state_file, fees=None, configs=None) -> SelectionTick:
    configs = configs or {pair: _config() for pair in rows}
    return SelectionTick(
        {pair: FakeRows(row) for pair, row in rows.items()},
        {pair: FakeTrained(config) for pair, config in configs.items()},
        state_file,
        1000.0,
        fees or NoFees(),
    )


# ── State ──────────────────────────────────────────────────────────────

def test_initial_state_is_flat_with_the_starting_balance():
    state = InitialSelectionState(5000.0).state()
    assert state.balance == 5000.0
    assert state.held_pair is None
    assert state.position is None


# The pair is the field paper_trading never had to store. A size and an entry
# price restored against the wrong market would be marked at an unrelated
# price, and exited using a different pair's genome.
def test_state_round_trips_which_pair_the_position_is_in(tmp_path):
    store = SelectionStateFile(str(tmp_path / "selection_state.json"))
    store.write(SelectionState(
        balance=900.0, held_pair="FET-USDT",
        position=Position(1.5, 200.0, Direction.SHORT),
        last_candle_start=1000, realized_trades=3, realized_wins=2,
    ))
    state = store.read()
    assert state.held_pair == "FET-USDT"
    assert state.position.direction() is Direction.SHORT
    assert state.position.entry_price() == 1.5
    assert (state.realized_trades, state.realized_wins) == (3, 2)


def test_state_round_trips_a_flat_book(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    store.write(SelectionState(1000.0, None, None, 500, 0, 0))
    assert store.read().held_pair is None
    assert store.read().position is None


def test_state_file_leaves_no_temporary_file_behind(tmp_path):
    directory = tmp_path / "book"
    SelectionStateFile(str(directory / "s.json")).write(SelectionState(1.0, None, None, 0, 0, 0))
    assert [entry.name for entry in directory.iterdir()] == ["s.json"]


# ── Idempotence ────────────────────────────────────────────────────────

# The whole reason this is schedulable: scheduling it far more often than the
# granularity has to be free.
async def test_a_second_tick_on_the_same_candle_does_nothing(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    rows  = {"A-USDT": _row(1000, 100.0, 0.9)}

    first = await _tick(rows, store).run()
    assert first.acted is True

    second = await _tick(rows, store).run()
    assert second.acted is False
    assert "already acted" in second.note


async def test_a_new_candle_is_acted_on(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    await _tick({"A-USDT": _row(1000, 100.0, 0.9)}, store).run()
    later = await _tick({"A-USDT": _row(2000, 110.0, 0.1)}, store).run()
    assert later.acted is True


# One lagging pair holds the whole tick — comparing a fresh score against a
# stale one would rank on staleness rather than strength.
async def test_the_candle_is_the_oldest_every_pair_has_closed(tmp_path):
    store   = SelectionStateFile(str(tmp_path / "s.json"))
    outcome = await _tick(
        {"A-USDT": _row(2000, 100.0, 0.5), "B-USDT": _row(1000, 100.0, 0.5)}, store,
    ).run()
    assert outcome.candle_start == 1000


# ── Selection ──────────────────────────────────────────────────────────

async def test_the_strongest_pair_is_entered_and_recorded(tmp_path):
    store   = SelectionStateFile(str(tmp_path / "s.json"))
    outcome = await _tick({
        "A-USDT": _row(1000, 100.0, 0.65),
        "B-USDT": _row(1000, 100.0, 0.95),
    }, store).run()
    assert outcome.action is Action.BUY
    assert outcome.held_pair == "B-USDT"
    assert store.read().held_pair == "B-USDT"


async def test_nothing_is_entered_when_no_pair_signals(tmp_path):
    store   = SelectionStateFile(str(tmp_path / "s.json"))
    outcome = await _tick({"A-USDT": _row(1000, 100.0, 0.5)}, store).run()
    assert outcome.held_pair is None
    assert outcome.action is Action.HOLD


async def test_a_held_pair_is_exited_on_its_own_signal_not_a_rivals(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    await _tick({
        "A-USDT": _row(1000, 100.0, 0.9),
        "B-USDT": _row(1000, 100.0, 0.5),
    }, store).run()
    assert store.read().held_pair == "A-USDT"

    # A drops through its sell threshold; B is screaming buy but must not be
    # consulted for the exit. Both legs happen on this candle, as in the
    # backtest, so the outcome has to name each separately.
    outcome = await _tick({
        "A-USDT": _row(2000, 120.0, 0.1),
        "B-USDT": _row(2000, 100.0, 0.99),
    }, store).run()
    assert outcome.exited == "A-USDT"
    assert outcome.entered == "B-USDT"
    assert store.read().realized_trades == 1


async def test_a_realized_win_survives_a_restart(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    await _tick({"A-USDT": _row(1000, 100.0, 0.9)}, store).run()
    await _tick({"A-USDT": _row(2000, 130.0, 0.1)}, store).run()
    state = store.read()
    assert state.realized_trades == 1
    assert state.realized_wins == 1
    assert state.balance > 1000.0


async def test_fees_leave_the_book_on_entry(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    await _tick({"A-USDT": _row(1000, 100.0, 0.9)}, store, fees=BasisPointFee(10.0)).run()
    # 1000 notional at 10 bps = 1.0 charged the moment the position opens.
    assert store.read().balance == pytest.approx(999.0)


async def test_a_resumed_book_keeps_trading_the_pair_it_was_holding(tmp_path):
    store = SelectionStateFile(str(tmp_path / "s.json"))
    store.write(SelectionState(
        balance=500.0, held_pair="A-USDT",
        position=Position(100.0, 5.0, Direction.LONG),
        last_candle_start=900, realized_trades=0, realized_wins=0,
    ))
    outcome = await _tick({
        "A-USDT": _row(1000, 150.0, 0.1),
        "B-USDT": _row(1000, 100.0, 0.9),
    }, store).run()
    # Exits A on A's own signal — it does not skip that and jump into B.
    assert outcome.exited == "A-USDT"
    assert store.read().realized_trades == 1
