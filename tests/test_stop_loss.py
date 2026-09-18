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
    PlacedStop,
    Position,
    StopLoss,
)


class _BuysOnceThenHolds:
    def decide(self, row: dict[str, float], position, balance: float) -> Decision:
        if position is None:
            return Decision(Action.BUY, 1.0)
        return Decision(Action.HOLD)


class _ShortsOnceThenHolds:
    def __init__(self, size: float = 1.0) -> None:
        self._size = size

    def decide(self, row: dict[str, float], position, balance: float) -> Decision:
        if position is None:
            return Decision(Action.SHORT, self._size)
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


# A falling candle is a short's PROFIT, and must not stop it. Without this, a
# stop reading `low <= price` for both directions passes the whole suite.
def test_a_falling_candle_never_stops_a_short():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 100.5, 80.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _ShortsOnceThenHolds(), 1000.0, stop_loss=AtrDistance(3.0),
    ).run()

    # Unwound by the window's end at its own entry price, not stopped at 106.
    assert result.trades()[0].profit() == pytest.approx(0.0)


def test_a_rising_candle_never_stops_a_long():
    frame  = _frame([(100.0, 100.0, 100.0, 0.02), (100.0, 130.0, 99.0, 0.02)])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, stop_loss=AtrDistance(3.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(0.0)


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


# A long stopped a full 100% below entry rests at zero or below, which no market
# reaches. It is reported as no stop rather than as an order that can never
# fill — and NOT raised, because it is a property of one candle's volatility:
# 90 of this universe's 57,624 six-hour candles put 8 x atr_pct past 1.0, and a
# raise would abort a 60-run sweep on the one candle where it crosses.
def test_an_unreachable_long_stop_is_reported_as_no_stop():
    placed = PlacedStop(Position(100.0, 1.0, Direction.LONG), 1.2)

    assert placed.unreachable() is True
    assert placed.fraction() == 0.0


def test_a_short_can_always_be_stopped_however_wide():
    placed = PlacedStop(Position(100.0, 1.0, Direction.SHORT), 1.2)

    assert placed.unreachable() is False
    assert placed.fraction() == pytest.approx(1.2)
    assert StopLoss(Position(100.0, 1.0, Direction.SHORT), 1.0).price() == pytest.approx(200.0)


def test_an_ordinary_stop_is_left_alone():
    assert PlacedStop(Position(100.0, 1.0, Direction.LONG), 0.3).fraction() == pytest.approx(0.3)


# The whole point of not raising: a wide arm meets such a candle mid-sweep, and
# the run has to finish rather than die.
def test_a_backtest_survives_a_candle_whose_stop_cannot_exist():
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.20),   # 8 x 20% = 160% below entry
        (100.0, 100.0, 50.0, 0.20),
    ])
    result = Backtest(
        MarketRows(frame), _BuysOnceThenHolds(), 1000.0, stop_loss=AtrDistance(8.0),
    ).run()

    assert result.trades()[0].profit() == pytest.approx(0.0)   # never stopped, never crashed


# ── Which adverse exit the market reached first ──────────────────────
# Both sit on the same side of entry for a short, and price is continuous: to
# reach the far one the market must pass the near one. Resolving liquidation
# first regardless books a liquidation the stop would have prevented.

def test_a_short_spiking_through_both_is_stopped_not_liquidated():
    # 6 units short at 100 against 1000 of collateral liquidates at 254.76. The
    # 6% stop sits at 106, which the market passed on its way there.
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 260.0, 100.0, 0.02),
    ])
    stopped = Backtest(
        MarketRows(frame), _ShortsOnceThenHolds(6.0), 1000.0, stop_loss=AtrDistance(3.0),
    ).run()
    unbounded = Backtest(MarketRows(frame), _ShortsOnceThenHolds(6.0), 1000.0).run()

    assert stopped.trades()[0].profit()   == pytest.approx(-36.0)
    assert unbounded.trades()[0].profit() < -900.0     # liquidated at 254.76 instead


# The other order, and it is not symmetric: a stop wide enough to sit BEYOND the
# liquidation is never reached, because the position is gone before the market
# gets there.
def test_a_stop_beyond_the_liquidation_leaves_the_liquidation_in_charge():
    frame  = _frame([
        (100.0, 100.0, 100.0, 0.02),
        (100.0, 260.0, 100.0, 0.02),
    ])
    # A 200% stop rests at 300, beyond the 254.76 liquidation.
    result = Backtest(
        MarketRows(frame), _ShortsOnceThenHolds(6.0), 1000.0, stop_loss=AtrDistance(100.0),
    ).run()

    assert result.trades()[0].profit() < -900.0


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


# The two sit on opposite sides of entry, so whichever the market reaches first
# closes the position — that is what a bracket is, at any ratio. A wide stop
# behind a near target is an ordinary configuration, not a contradiction.
@pytest.mark.parametrize("target,stop", [(4.0, 2.0), (2.0, 4.0), (2.0, 2.0)])
def test_any_bracket_of_target_and_stop_is_accepted(target, stop):
    config = ValidatedStrategyConfig(
        StrategyConfigFile(
            _section(take_profit_atr_mult=target, stop_loss_atr_mult=stop),
        ).config(),
    ).config()

    assert (config.take_profit_atr_mult, config.stop_loss_atr_mult) == (target, stop)


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


# ── What the journal records ─────────────────────────────────────────
# A resting order fires before the strategy is consulted, so the decision on
# that tick is HOLD. Reporting only the decision put a HOLD beside a jumped
# balance and no record of the exit.

@pytest.mark.asyncio
async def test_a_stopped_tick_reports_the_stop_not_the_hold(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))
    orders     = TrainedRestingOrders(_config(stop_loss_atr_mult=2.0))

    await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()
    outcome = await PaperTick(
        _Rows([_row(3600, 99.0, 100.0, 95.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()

    # The stop fired before the strategy was consulted, and this strategy then
    # re-entered on the same candle's close — so the tick carries BOTH legs, and
    # the decision alone would report only the entry.
    assert outcome.closed_by == "stop"
    assert outcome.decision.action is Action.BUY
    assert outcome.closed_trades == 1


@pytest.mark.asyncio
async def test_a_target_tick_reports_the_target(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))
    orders     = TrainedRestingOrders(_config(take_profit_atr_mult=2.0))

    await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()
    outcome = await PaperTick(
        _Rows([_row(3600, 101.0, 105.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
        take_profit=orders.take_profit(), stop_loss=orders.stop_loss(),
    ).run()

    assert outcome.closed_by == "target"


@pytest.mark.asyncio
async def test_an_ordinary_tick_reports_nothing_closed(tmp_path):
    state_file = PaperStateFile(str(tmp_path / "book.json"))
    outcome    = await PaperTick(
        _Rows([_row(1800, 100.0, 100.0, 100.0)]), _BuysOnceThenHolds(), state_file, 1000.0,
    ).run()

    assert outcome.closed_by == ""
