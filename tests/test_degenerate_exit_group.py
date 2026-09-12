"""The dual exit model: what a one-key group still cannot do, and what the
scaled, undirected P&L term now does correctly.

Both facts were found by retraining every pair on design "dual" with exit_keys
empty. All five BTC seeds returned bit-identical books, because the exit model
had no free parameters AND was comparing a raw fractional return against
thresholds meant for a [0, 1] column.
"""
import pytest

from coinbase.ga.ga_engine import Genome, GroupedL1Scaling
from coinbase.ga.strategy_evaluator import (
    DUAL_DESIGN,
    DualSignal,
    GaStrategy,
    ScaledReturn,
    SignalDesign,
    StrategyConfig,
)
from coinbase.trading_strategy import Action, Direction, Position

KEYS = ("entry:rsi", "entry:macd", "exit:position_pnl")


def config() -> StrategyConfig:
    return StrategyConfig(
        position_size_pct     = 0.6,
        buy_threshold         = 0.6,
        sell_threshold        = 0.4,
        starting_balance      = 10000.0,
        allow_short           = True,
        short_entry_threshold = 0.25,
        short_exit_threshold  = 0.45,
        design                = DUAL_DESIGN,
    )


def model() -> DualSignal:
    return DualSignal(
        Genome({"entry:rsi": 0.5, "entry:macd": 0.5, "exit:position_pnl": 1.0}), KEYS,
    )


def row(close: float) -> dict[str, float]:
    return {"norm_rsi": 0.5, "norm_macd": 0.5, "close": close}


class TestSingleKeyGroupStillHasNoFreedom:
    # Unchanged by the scaling fix, and the reason exit_keys must not be empty:
    # L1 over one key returns its sign, so the GA cannot move this weight.
    @pytest.mark.parametrize("proposed", [0.01, 1.0, 99.0])
    def test_a_single_key_group_is_forced_to_one(self, proposed: float) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "entry:macd": 1.0, "exit:position_pnl": proposed},
        )
        assert scaled["exit:position_pnl"] == pytest.approx(1.0)

    def test_two_keys_leave_the_exit_group_learnable(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "exit:position_pnl": 3.0, "exit:delta_1": 1.0},
        )
        assert scaled["exit:position_pnl"] == pytest.approx(0.75)
        assert scaled["exit:delta_1"] == pytest.approx(0.25)

    def test_the_shipped_default_does_not_produce_a_one_key_exit_group(self) -> None:
        import yaml

        with open("coinbase/ga/config.yaml", encoding="utf-8") as handle:
            section = yaml.safe_load(handle)["strategy"]
        if section.get("design") != DUAL_DESIGN:
            pytest.skip("default is not dual; nothing to guard")
        shape = SignalDesign(DUAL_DESIGN).keys(
            tuple(section["weight_keys"]), tuple(section.get("exit_keys") or ()),
        )
        exit_keys = [k for k in shape if k.startswith("exit:")]
        assert len(exit_keys) > 1, (
            f"dual is configured with a single-key exit group {exit_keys}, which "
            f"GroupedL1Scaling pins at 1.0 — the exit model would have no free "
            f"parameters"
        )


class TestScaledReturn:
    def test_flat_pnl_reads_neutral(self) -> None:
        assert ScaledReturn(0.0).value() == pytest.approx(0.5)

    def test_it_stays_inside_the_unit_interval(self) -> None:
        for r in (-10.0, -0.5, 0.0, 0.5, 10.0):
            assert 0.0 <= ScaledReturn(r).value() <= 1.0

    def test_small_moves_are_where_it_is_sensitive(self) -> None:
        near = ScaledReturn(0.02).value() - ScaledReturn(0.0).value()
        far = ScaledReturn(0.42).value() - ScaledReturn(0.40).value()
        assert near > 100 * far  # saturates, so a big move stops mattering

    def test_it_is_monotone_in_the_move(self) -> None:
        values = [ScaledReturn(r).value() for r in (-0.1, -0.01, 0.0, 0.01, 0.1)]
        assert values == sorted(values)


class TestTheExitTermActsAsAStopLossOnBothSides:
    # The old term was direction-agnostic, so "ahead" read high for a short —
    # and the short band closes on a HIGH score, cutting winners and riding
    # losers. The exit score now tracks price, so both sides behave alike.
    def test_a_falling_price_stops_out_a_long_and_holds_a_short(self) -> None:
        strategy = GaStrategy(model(), config())
        long_pos = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        short_pos = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
        assert strategy.decide(row(98.0), long_pos, 10000.0).action is Action.SELL
        assert strategy.decide(row(98.0), short_pos, 10000.0).action is Action.HOLD

    def test_a_rising_price_holds_a_long_and_stops_out_a_short(self) -> None:
        strategy = GaStrategy(model(), config())
        long_pos = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        short_pos = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
        assert strategy.decide(row(102.0), long_pos, 10000.0).action is Action.HOLD
        assert strategy.decide(row(102.0), short_pos, 10000.0).action is Action.COVER

    # The bug this replaces: every long was sold on the candle after it opened,
    # because 0.02 of raw return is far below a sell_threshold of 0.40.
    def test_a_long_slightly_ahead_is_no_longer_sold_immediately(self) -> None:
        strategy = GaStrategy(model(), config())
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        assert strategy.decide(row(102.0), held, 10000.0).action is Action.HOLD

    def test_the_exit_score_is_scaled_not_raw(self) -> None:
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        assert model().score(row(102.0), held) == pytest.approx(ScaledReturn(0.02).value())
        assert model().score(row(102.0), held) > 0.85  # raw would have been 0.02
