"""A one-key group has no free parameters, and the exit model must not be one.

Found by retraining every pair on design "dual" with exit_keys empty: all five
BTC seeds returned bit-identical books. The cause is in the scaling, not the
search — see test_a_single_key_group_is_forced_to_one below.
"""
import pytest

from coinbase.ga.ga_engine import Genome, GroupedL1Scaling
from coinbase.ga.strategy_evaluator import (
    DUAL_DESIGN,
    DualSignal,
    GaStrategy,
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


class TestSingleKeyExitGroupIsDegenerate:
    # Whatever the GA proposes, L1 over one key returns its sign. The exit
    # model has nothing to learn.
    @pytest.mark.parametrize("proposed", [0.01, 1.0, 99.0])
    def test_a_single_key_group_is_forced_to_one(self, proposed: float) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "entry:macd": 1.0, "exit:position_pnl": proposed},
        )
        assert scaled["exit:position_pnl"] == pytest.approx(1.0)

    def test_the_entry_group_still_has_freedom(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 3.0, "entry:macd": 1.0, "exit:position_pnl": 1.0},
        )
        assert scaled["entry:rsi"] == pytest.approx(0.75)
        assert scaled["entry:macd"] == pytest.approx(0.25)

    # With the weight pinned at 1.0 the exit score IS the unrealized return,
    # which is compared against thresholds meant for a [0, 1] score.
    def test_the_exit_score_is_the_raw_unrealized_return(self) -> None:
        model = DualSignal(Genome({"entry:rsi": 0.5, "entry:macd": 0.5,
                                   "exit:position_pnl": 1.0}), KEYS)
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        assert model.score({"norm_rsi": 1.0, "norm_macd": 1.0, "close": 110.0}, held) \
            == pytest.approx(0.10)

    def test_a_long_is_sold_on_the_very_next_candle(self) -> None:
        model = DualSignal(Genome({"entry:rsi": 0.5, "entry:macd": 0.5,
                                   "exit:position_pnl": 1.0}), KEYS)
        strategy = GaStrategy(model, config())
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        # Up 2% and still sold: 0.02 is below a sell_threshold of 0.40.
        row = {"norm_rsi": 1.0, "norm_macd": 1.0, "close": 102.0}
        assert strategy.decide(row, held, 10000.0).action is Action.SELL

    def test_a_short_is_never_covered_by_signal(self) -> None:
        model = DualSignal(Genome({"entry:rsi": 0.5, "entry:macd": 0.5,
                                   "exit:position_pnl": 1.0}), KEYS)
        strategy = GaStrategy(model, config())
        held = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
        # Even 30% in profit is below a short_exit_threshold of 0.45.
        row = {"norm_rsi": 0.0, "norm_macd": 0.0, "close": 70.0}
        assert strategy.decide(row, held, 10000.0).action is Action.HOLD

    # Two keys restore a degree of freedom, which is the minimum a dual exit
    # model needs before it is worth adopting again.
    def test_two_keys_leave_the_exit_group_learnable(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "exit:position_pnl": 3.0, "exit:delta_1": 1.0},
        )
        assert scaled["exit:position_pnl"] == pytest.approx(0.75)
        assert scaled["exit:delta_1"] == pytest.approx(0.25)

    def test_the_shipped_default_does_not_produce_a_one_key_exit_group(self) -> None:
        import yaml

        with open("coinbase/ga/config.yaml", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        section = raw["strategy"]
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
