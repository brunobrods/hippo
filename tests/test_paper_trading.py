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
