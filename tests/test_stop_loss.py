import json

import pandas as pd
import pytest

from coinbase.ga.paper_trading import (
    PaperState,
    PaperStateFile,
    PaperTick,
    TrainedRestingOrders,
)
from coinbase.ga.strategy_evaluator import (
    StrategyConfig,
    StrategyConfigFile,
    ValidatedStrategyConfig,
)
from coinbase.trading_strategy import (
    Action,
    AtrDistance,
    Backtest,
    Decision,
    Direction,
    FixedDistance,
    MarketRows,
    Position,
    StopLoss,
)


class _BuysOnceThenHolds:
    def decide(self, row: dict[str, float], position, balance: float) -> Decision:
        if position is None:
            return Decision(Action.BUY, 1.0)
        return Decision(Action.HOLD)


class _ShortsOnceThenHolds:
    def decide(self, row: dict[str, float], position, balance: float) -> Decision:
        if position is None:
            return Decision(Action.SHORT, 1.0)
        return Decision(Action.HOLD)


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["close", "high", "low", "atr_pct"],
    ).assign(timestamp=lambda f: range(0, 3600 * len(f), 3600))


# ── The order itself ─────────────────────────────────────────────────

def test_a_long_stops_below_entry_and_a_short_above_it():
    long_stop  = StopLoss(Position(100.0, 1.0, Direction.LONG), 0.06)
    short_stop = StopLoss(Position(100.0, 1.0, Direction.SHORT), 0.06)

    assert long_stop.price()  == pytest.approx(94.0)
    assert short_stop.price() == pytest.approx(106.0)


def test_only_an_adverse_move_reaches_it():
    stop = StopLoss(Position(100.0, 1.0, Direction.LONG), 0.06)

    assert stop.reached_by(high=120.0, low=94.0) is True
    assert stop.reached_by(high=120.0, low=94.1) is False


def test_a_zero_fraction_rests_nothing():
    assert StopLoss(Position(100.0, 1.0), 0.0).reached_by(high=100.0, low=0.01) is False


# ── In a backtest ────────────────────────────────────────────────────

def test_the_stop_rests_at_a_multiple_of_the_entry_candles_atr():
    # atr_pct 2%, multiple 3 -> a stop 6% below 100, reached by a low of 94.
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 100.0, 94.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, stop_loss=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(-6.0)


def test_a_short_is_stopped_by_a_rising_candle():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 106.0, 100.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _ShortsOnceThenHolds(), 1000.0, stop_loss=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(-6.0)


# Without a stop this position simply rides the drawdown and is unwound at its
# own entry price, which is how a loser has been hidden from the book until now.
def test_without_a_stop_the_same_candle_books_nothing():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 100.0, 94.0, 0.02)])
    result = Backtest(MarketRows(frame), _BuysOnceThenHolds(), 1000.0).run()

    assert result.trades()[0].profit() == pytest.approx(0.0)


# The trap the whole ordering exists for. OHLC has no chronology, so a candle
# that reaches both levels must be read as the adverse one first.
def test_a_candle_reaching_both_the_stop_and_the_target_takes_the_loss():
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 130.0, 90.0, 0.02),   # reaches the 106 target AND the 94 stop
    ])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0,
        take_profit=AtrDistance(3.0), stop_loss=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() < 0.0


def test_a_stop_and_a_target_can_bracket_the_same_position():
    # The target is reached and the stop is not, so the target closes it.
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 112.0, 99.0, 0.02),
    ])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0,
        take_profit=AtrDistance(3.0), stop_loss=AtrDistance(1.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(6.0)


# ── Config ───────────────────────────────────────────────────────────

def _section(**overrides: object) -> dict[str, object]:
    section = {
        "position_size_pct": 0.6,
        "buy_threshold":     0.6,
        "sell_threshold":    0.4,
        "starting_balance":  1000.0,
    }
    section.update(overrides)
    return {"strategy": section}


def test_the_multiple_is_read_and_defaults_to_off():
    assert StrategyConfigFile(_section()).config().stop_loss_atr_mult == 0.0
    assert StrategyConfigFile(_section(stop_loss_atr_mult=2.5)).config().stop_loss_atr_mult == 2.5


def test_a_negative_stop_is_rejected():
    with pytest.raises(ValueError, match="stop_loss_atr_mult"):
        ValidatedStrategyConfig(
            StrategyConfigFile(_section(stop_loss_atr_mult=-1.0)).config(),
        ).config()


# A stop at or beyond the target closes on whichever side moved first, which
# makes the genome a coin flip between two fixed levels.
def test_a_stop_at_or_beyond_the_target_is_rejected():
    with pytest.raises(ValueError, match="race"):
        ValidatedStrategyConfig(
            StrategyConfigFile(
                _section(take_profit_atr_mult=2.0, stop_loss_atr_mult=2.0),
            ).config(),
        ).config()


def test_a_stop_inside_the_target_is_accepted():
    config = ValidatedStrategyConfig(
        StrategyConfigFile(
            _section(take_profit_atr_mult=4.0, stop_loss_atr_mult=2.0),
        ).config(),
    ).config()

    assert (config.take_profit_atr_mult, config.stop_loss_atr_mult) == (4.0, 2.0)


# ── What a trained genome hands the paper book ───────────────────────

def _config(**overrides: object) -> StrategyConfig:
    return StrategyConfig(
        position_size_pct=0.5, buy_threshold=0.6, sell_threshold=0.4,
        starting_balance=1000.0, **overrides,
    )


def test_the_orders_follow_the_genomes_own_config():
    orders = TrainedRestingOrders(_config(take_profit_atr_mult=4.0, stop_loss_atr_mult=2.0))

    assert orders.take_profit().fraction({"atr_pct": 0.03}) == pytest.approx(0.12)
    assert orders.stop_loss().fraction({"atr_pct": 0.03}) == pytest.approx(0.06)


def test_a_genome_without_a_stop_never_needs_the_column():
    assert TrainedRestingOrders(_config()).stop_loss().fraction({}) == 0.0


def test_a_fixed_target_still_reaches_the_paper_book():
    orders = TrainedRestingOrders(_config(take_profit_pct=0.05))
    assert orders.take_profit().fraction({}) == pytest.approx(0.05)


# ── Paper trading, across process boundaries ─────────────────────────

class _Rows:
    def __init__(self, rows: list[dict[str, float]]) -> None:
        self._rows = rows

    def pair(self) -> str:
        return "BTC-USDT"

    async def latest(self) -> dict[str, float]:
        return self._rows.pop(0)


def _row(timestamp: int, close: float, high: float, low: float) -> dict[str, float]:
    return {"timestamp": timestamp, "close": close, "high": high, "low": low, "atr_pct": 0.02}


@pytest.mark.asyncio
async def test_a_paper_tick_records_the_distances_it_opened_with(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))
    orders     = TrainedRestingOrders(_config(take_profit_atr_mult=4.0, stop_loss_atr_mult=2.0))

    await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()

    saved = json.loads((tmp_path / "book.json").read_text(encoding="utf-8"))
    assert saved["take_profit_fraction"] == pytest.approx(0.08)
    assert saved["stop_loss_fraction"]   == pytest.approx(0.04)


# The point of persisting them: the tick that opens a position and the tick that
# closes it are different processes, and the entry candle is gone by then.
@pytest.mark.asyncio
async def test_a_later_tick_stops_the_position_the_earlier_one_opened(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))
    orders     = TrainedRestingOrders(_config(stop_loss_atr_mult=2.0))

    await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()
    # A fresh tick, as a relaunched engine builds — the 4% stop sits at 96.0.
    outcome = await PaperTick(
        _Rows([_row(3600, 99.0, 100.0, 95.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()

    assert outcome.closed_trades == 1
    assert state_file.read().balance == pytest.approx(996.0)   # 5 units short of entry


@pytest.mark.asyncio
async def test_a_tick_without_orders_leaves_the_book_exactly_as_it_was(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))

    await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
    ).run()
    outcome = await PaperTick(
        _Rows([_row(3600, 99.0, 100.0, 1.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
    ).run()

    assert outcome.closed_trades == 0
    assert state_file.read().position is not None


# A book written before these fields existed has no resting orders, and cannot
# have them reconstructed: the entry candle that priced them is long gone.
def test_a_book_predating_the_fields_reads_as_having_no_orders(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "balance": 1000.0, "position": None, "last_candle_start": 1800,
        "realized_trades": 0,
    }), encoding="utf-8")

    state = PaperStateFile(str(path)).read()
    assert (state.take_profit_fraction, state.stop_loss_fraction) == (0.0, 0.0)


def test_the_fractions_survive_a_write_and_read(tmp_path):
    path       = str(tmp_path / "book.json")
    state_file = PaperStateFile(path)
    state_file.write(
        PaperState(
            balance=1000.0, position=None, last_candle_start=1800, realized_trades=0,
            take_profit_fraction=0.08, stop_loss_fraction=0.04,
        ),
        "BTC-USDT",
    )

    restored = state_file.read()
    assert (restored.take_profit_fraction, restored.stop_loss_fraction) == (0.08, 0.04)
