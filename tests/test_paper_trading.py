import json
import time

import pandas as pd
import pytest

from coinbase.ga.market_data_processor import IndicatorPeriods
from coinbase.ga.paper_trading import (
    BasisPointFee,
    InitialPaperState,
    MakerTakerFee,
    NoFees,
    PaperConfigFile,
    PaperState,
    PaperStateFile,
    PaperTick,
)
from coinbase.strategy import ClosedMarketRow, LiveMarketRow
from coinbase.trading_strategy import Action, Decision, Direction, Position


# ── Test doubles ─────────────────────────────────────────────────────

class FakeRows:
    """Stands in for ClosedMarketRow — hands back a scripted candle row."""

    def __init__(self, rows: list[dict[str, float]]) -> None:
        self._rows  = rows
        self._calls = 0

    def pair(self) -> str:
        return "BTC-USDT"

    async def latest(self) -> dict[str, float]:
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        return row


class _ScriptedStrategy:
    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions
        self._calls   = 0
        self.seen_positions: list = []
        self.seen_balances:  list = []

    def decide(self, row, position, balance) -> Decision:
        self.seen_positions.append(position)
        self.seen_balances.append(balance)
        action = self._actions[min(self._calls, len(self._actions) - 1)]
        self._calls += 1
        size = (balance * 0.5) / row["close"] if action in (Action.BUY, Action.SHORT) else 0.0
        return Decision(action, size)


def _row(timestamp: int, close: float, high: float = None, low: float = None) -> dict[str, float]:
    return {
        "timestamp": float(timestamp),
        "close": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
    }


# ── PaperConfigFile ──────────────────────────────────────────────────

def test_paper_section_overrides_the_training_market():
    config = PaperConfigFile({
        "data":  {"exchange": "coinbase", "pair": "BTC-USDC"},
        "paper": {"exchange": "binance", "pair": "BTC-USDT"},
    }).config()

    assert (config.exchange, config.pair) == ("binance", "BTC-USDT")


def test_paper_falls_back_to_the_training_market_when_unset():
    config = PaperConfigFile({"data": {"exchange": "binance", "pair": "ETH-USDT"}}).config()

    assert (config.exchange, config.pair) == ("binance", "ETH-USDT")


def test_a_commented_out_paper_section_is_not_an_error():
    # A YAML mapping whose every child is commented out parses to None.
    config = PaperConfigFile({"data": {"pair": "BTC-USDC"}, "paper": None}).config()

    assert config.pair == "BTC-USDC"
    assert config.exchange == "coinbase"


def test_state_filepath_defaults_outside_the_repo():
    config = PaperConfigFile({"data": {"pair": "BTC-USDC"}}).config()

    assert config.state_filepath.endswith("paper_state.json")


# ── PaperStateFile ───────────────────────────────────────────────────

def test_state_round_trips_a_flat_book(tmp_path):
    path = str(tmp_path / "state.json")
    file = PaperStateFile(path)
    file.write(PaperState(balance=1234.5, position=None, last_candle_start=99, realized_trades=3), "BTC-USDT")

    state = file.read()
    assert state.balance == pytest.approx(1234.5)
    assert state.position is None
    assert state.last_candle_start == 99
    assert state.realized_trades == 3


def test_state_round_trips_an_open_short(tmp_path):
    path = str(tmp_path / "state.json")
    file = PaperStateFile(path)
    file.write(
        PaperState(
            balance=1000.0,
            position=Position(entry_price=78000.0, size=0.01, direction=Direction.SHORT),
            last_candle_start=1787616000,
            realized_trades=1,
        ),
        "BTC-USDT",
    )

    position = file.read().position
    assert position.entry_price() == pytest.approx(78000.0)
    assert position.size() == pytest.approx(0.01)
    assert position.direction() is Direction.SHORT


def test_state_file_records_the_pair_it_belongs_to(tmp_path):
    path = str(tmp_path / "state.json")
    PaperStateFile(path).write(
        PaperState(balance=1.0, position=None, last_candle_start=0, realized_trades=0), "ETH-USDT",
    )

    assert json.loads((tmp_path / "state.json").read_text())["pair"] == "ETH-USDT"


def test_state_file_reports_absence_before_the_first_tick(tmp_path):
    assert PaperStateFile(str(tmp_path / "nothing.json")).exists() is False


def test_state_write_leaves_no_temp_file_behind(tmp_path):
    path = str(tmp_path / "state.json")
    PaperStateFile(path).write(
        PaperState(balance=1.0, position=None, last_candle_start=0, realized_trades=0), "BTC-USDT",
    )

    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_initial_state_starts_flat_at_the_configured_balance():
    state = InitialPaperState(10000.0).state()

    assert state.balance == pytest.approx(10000.0)
    assert state.position is None
    assert state.last_candle_start == 0


# ── PaperTick ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_tick_seeds_from_the_starting_balance(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    strategy   = _ScriptedStrategy([Action.HOLD])

    outcome = await PaperTick(FakeRows([_row(100, 50.0)]), strategy, state_file, 10000.0).run()

    assert outcome.acted is True
    assert strategy.seen_balances[0] == pytest.approx(10000.0)
    assert outcome.balance == pytest.approx(10000.0)


@pytest.mark.asyncio
async def test_a_second_tick_on_the_same_candle_does_nothing(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0)])
    strategy   = _ScriptedStrategy([Action.BUY, Action.SELL])

    first  = await PaperTick(rows, strategy, state_file, 1000.0).run()
    second = await PaperTick(rows, strategy, state_file, 1000.0).run()

    assert first.acted is True
    assert second.acted is False
    assert second.decision is None
    # the strategy was consulted exactly once
    assert len(strategy.seen_positions) == 1


@pytest.mark.asyncio
async def test_a_new_candle_acts_again(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0), _row(200, 55.0)])
    strategy   = _ScriptedStrategy([Action.BUY, Action.HOLD])

    await PaperTick(rows, strategy, state_file, 1000.0).run()
    second = await PaperTick(rows, strategy, state_file, 1000.0).run()

    assert second.acted is True
    assert second.candle_start == 200


@pytest.mark.asyncio
async def test_an_open_position_survives_a_restart(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0), _row(200, 55.0)])
    strategy   = _ScriptedStrategy([Action.BUY, Action.HOLD])

    await PaperTick(rows, strategy, state_file, 1000.0).run()
    # A brand new PaperTick, as a scheduled task would build on the next run.
    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()

    reloaded = state_file.read()
    assert reloaded.position is not None
    assert reloaded.position.direction() is Direction.LONG
    assert reloaded.position.entry_price() == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_the_strategy_sees_the_position_carried_in(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0), _row(200, 55.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.BUY]), state_file, 1000.0).run()
    second = _ScriptedStrategy([Action.SELL])
    await PaperTick(rows, second, state_file, 1000.0).run()

    assert second.seen_positions[0] is not None


@pytest.mark.asyncio
async def test_realized_profit_accumulates_across_ticks(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0), _row(200, 60.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.BUY]), state_file, 1000.0).run()
    outcome = await PaperTick(rows, _ScriptedStrategy([Action.SELL]), state_file, 1000.0).run()

    # 10 units bought at 50 (half of 1000), sold at 60 -> +100
    assert outcome.balance == pytest.approx(1100.0)
    assert outcome.closed_trades == 1
    assert state_file.read().realized_trades == 1


@pytest.mark.asyncio
async def test_a_skipped_tick_still_reports_mark_to_market_equity(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.BUY]), state_file, 1000.0).run()
    skipped = await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()

    assert skipped.acted is False
    # 10 units long at 50, marked at 50 -> equity unchanged from balance
    assert skipped.equity == pytest.approx(1000.0)


# ── ClosedMarketRow ──────────────────────────────────────────────────

class _FrameRows:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def pair(self) -> str:
        return "BTC-USDT"

    def granularity(self) -> str:
        return "ONE_HOUR"

    async def frame(self) -> pd.DataFrame:
        return self._frame


@pytest.mark.asyncio
async def test_the_still_forming_candle_is_dropped():
    hour    = 3600
    current = int(time.time()) // hour * hour
    frame   = pd.DataFrame([
        {"timestamp": current - 2 * hour, "close": 1.0, "high": 1.0, "low": 1.0},
        {"timestamp": current - hour,     "close": 2.0, "high": 2.0, "low": 2.0},
        {"timestamp": current,            "close": 3.0, "high": 3.0, "low": 3.0},  # in progress
    ])

    row = await ClosedMarketRow(_FrameRows(frame)).latest()

    assert row["close"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_a_window_with_no_closed_candle_raises():
    hour    = 3600
    current = int(time.time()) // hour * hour
    frame   = pd.DataFrame([{"timestamp": current, "close": 3.0, "high": 3.0, "low": 3.0}])

    with pytest.raises(ValueError, match="no completed"):
        await ClosedMarketRow(_FrameRows(frame)).latest()


def test_the_current_candle_boundary_is_a_multiple_of_the_granularity():
    frame = pd.DataFrame([{"timestamp": 0, "close": 1.0, "high": 1.0, "low": 1.0}])

    boundary = ClosedMarketRow(_FrameRows(frame)).current_candle_start()

    assert boundary % 3600 == 0
    assert boundary <= int(time.time())


# ── TrainedStrategyConfig ────────────────────────────────────────────

def _raw_config(**strategy) -> dict:
    base = {
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "position_size_pct": 0.6,
        "starting_balance": 10000.0,
        "allow_short": True,
        "short_entry_threshold": 0.25,
        "short_exit_threshold": 0.45,
    }
    base.update(strategy)
    return {"strategy": base}


def test_the_saved_genomes_thresholds_win_over_config_yaml():
    from coinbase.ga.paper_trading import TrainedStrategyConfig

    config = TrainedStrategyConfig(_raw_config(), {"short_entry_threshold": 0.3}).config()

    assert config.short_entry_threshold == pytest.approx(0.3)


def test_config_yaml_still_supplies_the_starting_balance():
    from coinbase.ga.paper_trading import TrainedStrategyConfig

    # starting_balance is a run-time choice, not something a genome carries.
    config = TrainedStrategyConfig(
        _raw_config(starting_balance=250.0), {"short_entry_threshold": 0.3},
    ).config()

    assert config.starting_balance == pytest.approx(250.0)


def test_divergences_name_both_sides():
    from coinbase.ga.paper_trading import TrainedStrategyConfig

    diverged = TrainedStrategyConfig(
        _raw_config(), {"short_entry_threshold": 0.3, "buy_threshold": 0.6},
    ).divergences()

    assert diverged == {"short_entry_threshold": (0.25, 0.3)}


def test_no_divergence_when_the_two_agree():
    from coinbase.ga.paper_trading import TrainedStrategyConfig

    assert TrainedStrategyConfig(_raw_config(), {"buy_threshold": 0.6}).divergences() == {}


def test_a_trained_config_that_contradicts_itself_is_rejected():
    from coinbase.ga.paper_trading import TrainedStrategyConfig

    # short_exit below short_entry would cover a short on the candle it opened.
    with pytest.raises(ValueError):
        TrainedStrategyConfig(
            _raw_config(), {"short_entry_threshold": 0.5, "short_exit_threshold": 0.1},
        ).config()


# ── Fee schedules ────────────────────────────────────────────────────

def test_maker_taker_fee_charges_the_rate_for_the_side_it_was_filled_on():
    fees = MakerTakerFee(maker_bps=60.0, taker_bps=120.0)
    assert fees.charge(1000.0, maker=True)  == pytest.approx(6.0)
    assert fees.charge(1000.0, maker=False) == pytest.approx(12.0)


# Binance charges both sides alike at base tier, so resting an order saves the
# spread, not the fee — a distinction worth keeping visible in the model.
def test_maker_taker_fee_can_price_both_sides_the_same():
    fees = MakerTakerFee(maker_bps=10.0, taker_bps=10.0)
    assert fees.charge(1000.0, maker=True) == fees.charge(1000.0, maker=False)


def test_maker_taker_fee_defaults_to_the_taker_rate():
    assert MakerTakerFee(10.0, 120.0).charge(1000.0) == pytest.approx(12.0)


# The existing schedules must keep working unchanged — paper_engine imports both.
def test_flat_schedules_ignore_the_maker_flag():
    assert BasisPointFee(20.0).charge(1000.0, maker=True) == pytest.approx(2.0)
    assert NoFees().charge(1000.0, maker=True) == 0.0


# ── Costs across a restart ───────────────────────────────────────────
# Fees and interest are charged into balance as they are taken, so a book that
# does not carry a running total cannot recover one afterwards. The engine
# restarts at every logon, which is what made a per-process tally report a
# book that had paid its way as one that had traded for free.

def test_costs_round_trip_through_the_state_file(tmp_path):
    file = PaperStateFile(str(tmp_path / "state.json"))
    file.write(
        PaperState(
            balance=1000.0, position=None, last_candle_start=1, realized_trades=2,
            fees_paid=12.5, interest_paid=0.75,
        ),
        "BTC-USDT",
    )

    state = file.read()
    assert state.fees_paid == pytest.approx(12.5)
    assert state.interest_paid == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_fees_accumulate_across_ticks_rather_than_being_replaced(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 50.0), _row(200, 55.0)])
    fees       = BasisPointFee(100.0)

    await PaperTick(rows, _ScriptedStrategy([Action.BUY]), state_file, 1000.0, fees).run()
    after_entry = state_file.read().fees_paid
    # A brand new PaperTick, as the engine builds after a restart.
    await PaperTick(rows, _ScriptedStrategy([Action.SELL]), state_file, 1000.0, fees).run()

    assert after_entry > 0.0
    assert state_file.read().fees_paid > after_entry


def test_an_older_book_recovers_the_fee_its_open_position_paid(tmp_path):
    path = tmp_path / "state.json"
    # Written before fees_paid existed — entry_fee is the one cost still on the
    # book, every earlier round trip having been folded into balance.
    path.write_text(json.dumps({
        "pair": "BTC-USDT",
        "balance": 9994.0,
        "position": {
            "entry_price": 79609.28, "size": 0.075, "direction": "SHORT",
            "entry_timestamp": 1788607800, "entry_fee": 6.0,
        },
        "last_candle_start": 1788908400,
        "realized_trades": 0,
    }))

    state = PaperStateFile(str(path)).read()
    assert state.fees_paid == pytest.approx(6.0)
    assert state.interest_paid == pytest.approx(0.0)


def test_an_older_flat_book_starts_its_tally_at_zero(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "pair": "BTC-USDT", "balance": 9994.0, "position": None,
        "last_candle_start": 1788908400, "realized_trades": 3,
    }))

    assert PaperStateFile(str(path)).read().fees_paid == pytest.approx(0.0)


# An explicit zero must not be mistaken for a missing key and re-seeded from
# the open position — that would revive the entry fee on a book that had
# already recorded paying nothing.
def test_a_recorded_zero_is_not_re_seeded_from_the_position(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "pair": "BTC-USDT",
        "balance": 1000.0,
        "position": {
            "entry_price": 50.0, "size": 1.0, "direction": "LONG",
            "entry_timestamp": 100, "entry_fee": 6.0,
        },
        "last_candle_start": 100,
        "realized_trades": 0,
        "fees_paid": 0.0,
    }))

    assert PaperStateFile(str(path)).read().fees_paid == pytest.approx(0.0)


# ── Drawdown and age across a restart ────────────────────────────────
# An equity curve held in memory is rebuilt at every launch, and the process
# start time is not the book's. Both statistics derived from them reset
# nightly; these numbers are what a book needs to keep instead.

def test_the_high_water_mark_and_drawdown_round_trip(tmp_path):
    file = PaperStateFile(str(tmp_path / "state.json"))
    file.write(
        PaperState(
            balance=900.0, position=None, last_candle_start=1, realized_trades=1,
            equity_peak=1200.0, max_drawdown=0.25, opened_at=1788607800.0,
        ),
        "BTC-USDT",
    )

    state = file.read()
    assert state.equity_peak == pytest.approx(1200.0)
    assert state.max_drawdown == pytest.approx(0.25)
    assert state.opened_at == pytest.approx(1788607800.0)


@pytest.mark.asyncio
async def test_the_worst_drawdown_is_kept_after_equity_recovers(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    # Bought 5 units at 100 with half of 1000; marked at 60, then at 110. The
    # trough is 500 cash + 5 units at 60 = 800, against a 1000 peak.
    rows = FakeRows([_row(100, 100.0), _row(200, 60.0), _row(300, 110.0)])

    for action in (Action.BUY, Action.HOLD, Action.HOLD):
        await PaperTick(rows, _ScriptedStrategy([action]), state_file, 1000.0).run()

    state = state_file.read()
    assert state.max_drawdown == pytest.approx(0.2)
    assert state.equity_peak == pytest.approx(1050.0)   # 500 cash + 5 units at 110


# A book upgraded mid-flight has never recorded a peak, and its equity today
# says nothing about where it has been. That it opened at its starting balance
# is not a guess, though, so the first reading is right rather than zero.
@pytest.mark.asyncio
async def test_an_older_book_measures_drawdown_from_its_starting_balance(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "pair": "BTC-USDT", "balance": 900.0, "position": None,
        "last_candle_start": 50, "realized_trades": 1,
    }))
    state_file = PaperStateFile(str(path))

    await PaperTick(FakeRows([_row(100, 100.0)]), _ScriptedStrategy([Action.HOLD]),
                    state_file, 1000.0).run()

    # 900 against a 1000 start it cannot have opened below.
    assert state_file.read().max_drawdown == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_a_new_book_is_dated_from_its_first_candle(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))

    await PaperTick(FakeRows([_row(100, 100.0)]), _ScriptedStrategy([Action.HOLD]),
                    state_file, 1000.0).run()

    assert state_file.read().opened_at == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_a_books_open_time_is_not_moved_by_a_later_tick(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 100.0), _row(200, 105.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()
    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()

    assert state_file.read().opened_at == pytest.approx(100.0)


# Nothing an older book carries dates it, and every candidate is LATER than
# its real open — which shortens the window an annualized figure divides by
# and so inflates it without limit. Undated is the honest answer.
@pytest.mark.asyncio
async def test_an_older_book_stays_undated_rather_than_claiming_today(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "pair": "BTC-USDT",
        "balance": 1000.0,
        "position": {
            "entry_price": 50.0, "size": 1.0, "direction": "SHORT",
            "entry_timestamp": 1788607800, "entry_fee": 0.5,
        },
        "last_candle_start": 1788607800,
        "realized_trades": 0,
    }))
    state_file = PaperStateFile(str(path))

    assert state_file.read().opened_at == pytest.approx(0.0)
    await PaperTick(FakeRows([_row(1788700000, 50.0)]), _ScriptedStrategy([Action.HOLD]),
                    state_file, 1000.0).run()

    assert state_file.read().opened_at == pytest.approx(0.0)


# starting_balance is config. A permanent floor under the peak would let a
# retrain that raises it write a drawdown the book never suffered.
@pytest.mark.asyncio
async def test_a_raised_starting_balance_does_not_lift_an_established_peak(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    rows       = FakeRows([_row(100, 100.0), _row(200, 100.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()
    # The same book, ticked again by an entry point configured far higher.
    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 5000.0).run()

    state = state_file.read()
    assert state.equity_peak == pytest.approx(1000.0)
    assert state.max_drawdown == pytest.approx(0.0)


# A trough between two closes is a real fall. Both extremes are already
# fetched for the liquidation check, so the book can sample the worse of them.
@pytest.mark.asyncio
async def test_drawdown_sees_a_trough_inside_a_candle(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "state.json"))
    # Enters long 5 units at 100, then a candle that dips to 60 and closes flat
    # back at 100: equity troughs at 500 cash + 5 units at 60 = 800.
    rows = FakeRows([_row(100, 100.0), _row(200, 100.0, high=100.0, low=60.0)])

    await PaperTick(rows, _ScriptedStrategy([Action.BUY]), state_file, 1000.0).run()
    await PaperTick(rows, _ScriptedStrategy([Action.HOLD]), state_file, 1000.0).run()

    assert state_file.read().max_drawdown == pytest.approx(0.2)
