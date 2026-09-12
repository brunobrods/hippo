import pytest

from coinbase.ga.ga_engine import Genome, GroupedL1Scaling, L1Scaling
from coinbase.ga.strategy_evaluator import (
    DUAL_DESIGN,
    LINEAR_DESIGN,
    POSITION_PNL_KEY,
    DualSignal,
    ScaledReturn,
    SignalDesign,
    StrategyConfig,
    TwoSidedModel,
)
from coinbase.trading_strategy import Direction, Position

ENTRY = ("entry:rsi", "entry:macd")
EXIT = ("exit:delta_1", "exit:position_pnl")
KEYS = ENTRY + EXIT


def row(rsi: float = 0.0, macd: float = 0.0, delta_1: float = 0.0, close: float = 100.0):
    return {"norm_rsi": rsi, "norm_macd": macd, "norm_delta_1": delta_1, "close": close}


def dual(**weights) -> DualSignal:
    return DualSignal(Genome(dict(weights)), KEYS)


class TestGroupedL1Scaling:
    def test_each_group_sums_to_one_independently(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:a": 3.0, "entry:b": 1.0, "exit:a": 1.0, "exit:b": 1.0},
        )
        assert scaled["entry:a"] + scaled["entry:b"] == pytest.approx(1.0)
        assert scaled["exit:a"] + scaled["exit:b"] == pytest.approx(1.0)
        assert scaled["entry:a"] == pytest.approx(0.75)

    # The whole point: weight given to the exit model no longer comes out of the
    # entry model's budget. Under flat L1 these four weights would sum to 1
    # together, leaving entry with half.
    def test_exit_weight_does_not_shrink_the_entry_budget(self) -> None:
        raw = {"entry:a": 1.0, "exit:a": 1.0, "exit:b": 8.0}
        grouped = GroupedL1Scaling().scaled(raw)
        flat = L1Scaling().scaled(raw)
        assert grouped["entry:a"] == pytest.approx(1.0)
        assert flat["entry:a"] == pytest.approx(0.1)

    def test_a_flat_genome_is_scaled_exactly_as_l1_does(self) -> None:
        raw = {"rsi": 2.0, "macd": 2.0}
        assert GroupedL1Scaling().scaled(raw) == L1Scaling().scaled(raw)

    def test_absolute_values_are_what_sum_to_one(self) -> None:
        scaled = GroupedL1Scaling().scaled({"entry:a": -3.0, "entry:b": 1.0})
        assert abs(scaled["entry:a"]) + abs(scaled["entry:b"]) == pytest.approx(1.0)
        assert scaled["entry:a"] < 0


class TestDualSignal:
    def test_while_flat_it_scores_only_the_entry_model(self) -> None:
        model = dual(**{"entry:rsi": 0.6, "entry:macd": 0.4, "exit:delta_1": 0.5, "exit:position_pnl": 0.5})
        assert model.score(row(rsi=1.0, macd=0.0, delta_1=1.0), None) == pytest.approx(0.6)

    def test_while_holding_it_scores_only_the_exit_model(self) -> None:
        model = dual(**{"entry:rsi": 0.6, "entry:macd": 0.4, "exit:delta_1": 0.5, "exit:position_pnl": 0.5})
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        # delta_1 reads 1.0, and price is unchanged so the scaled move is 0.5
        # (neutral) rather than 0. The entry columns contribute nothing.
        assert model.score(row(rsi=1.0, delta_1=1.0), held) == pytest.approx(0.75)

    # The move is SCALED onto [0, 1] and is UNDIRECTED — see ScaledReturn and
    # DualSignal._move. A raw return here is what made every long sell
    # immediately, and a direction-agnostic one made shorts cut their winners.
    def test_the_exit_model_reads_the_scaled_price_move(self) -> None:
        model = dual(**{"entry:rsi": 1.0, "entry:macd": 0.0, "exit:delta_1": 0.0, "exit:position_pnl": 1.0})
        held = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        assert model.score(row(close=110.0), held) == pytest.approx(ScaledReturn(0.10).value())
        assert model.score(row(close=110.0), held) > 0.99

    def test_a_short_reads_the_same_price_move_with_the_same_sign(self) -> None:
        model = dual(**{"entry:rsi": 1.0, "entry:macd": 0.0, "exit:delta_1": 0.0, "exit:position_pnl": 1.0})
        long_pos = Position(entry_price=100.0, size=1.0, direction=Direction.LONG)
        short_pos = Position(entry_price=100.0, size=1.0, direction=Direction.SHORT)
        # Price rose 10%: bad for the short, good for the long — but the exit
        # score describes the MARKET, so both read it the same way.
        assert model.score(row(close=110.0), short_pos) == pytest.approx(
            model.score(row(close=110.0), long_pos),
        )

    # The failure the design exists to remove: under linear, weight on
    # position_pnl lowered this. Here the entry group is its own budget.
    def test_the_flat_ceiling_is_one_however_much_weight_the_exit_model_holds(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "entry:macd": 1.0, "exit:delta_1": 1.0, "exit:position_pnl": 99.0},
        )
        assert DualSignal(Genome(scaled), KEYS).flat_score_ceiling() == pytest.approx(1.0)

    def test_a_dual_genome_is_never_refused_as_one_sided(self) -> None:
        scaled = GroupedL1Scaling().scaled(
            {"entry:rsi": 1.0, "entry:macd": 1.0, "exit:delta_1": 1.0, "exit:position_pnl": 99.0},
        )
        config = StrategyConfig(
            position_size_pct = 0.6,
            buy_threshold     = 0.6,
            sell_threshold    = 0.4,
            starting_balance  = 10000.0,
            design            = DUAL_DESIGN,
        )
        assert TwoSidedModel(DualSignal(Genome(scaled), KEYS), config).is_one_sided() is False

    def test_a_negative_weight_lifts_the_groups_floor_back_to_zero(self) -> None:
        model = dual(**{"entry:rsi": -0.5, "entry:macd": 0.5, "exit:delta_1": 1.0, "exit:position_pnl": 0.0})
        # Worst case for the entry model is rsi at 1.0, macd at 0.0: -0.5, lifted to 0.
        assert model.score(row(rsi=1.0, macd=0.0), None) == pytest.approx(0.0)
        assert model.score(row(rsi=0.0, macd=1.0), None) == pytest.approx(1.0)


class TestSignalDesignKeys:
    def test_linear_appends_position_pnl_to_one_flat_list(self) -> None:
        assert SignalDesign(LINEAR_DESIGN).keys(("rsi", "macd")) == ("rsi", "macd", POSITION_PNL_KEY)

    def test_linear_ignores_exit_keys(self) -> None:
        assert SignalDesign(LINEAR_DESIGN).keys(("rsi",), ("delta_1",)) == ("rsi", POSITION_PNL_KEY)

    def test_dual_prefixes_both_groups_and_gives_the_exit_model_position_pnl(self) -> None:
        assert SignalDesign(DUAL_DESIGN).keys(("rsi", "macd"), ("delta_1",)) == (
            "entry:rsi", "entry:macd", "exit:delta_1", "exit:position_pnl",
        )

    def test_dual_with_no_exit_columns_still_has_a_pnl_only_exit_model(self) -> None:
        assert SignalDesign(DUAL_DESIGN).keys(("rsi",)) == ("entry:rsi", "exit:position_pnl")

    def test_an_unknown_design_names_both_it_knows(self) -> None:
        with pytest.raises(ValueError, match="dual"):
            SignalDesign("transformer").keys(("rsi",))


class TestSignalDesignWiring:
    def test_dual_is_built_with_grouped_scaling(self) -> None:
        assert isinstance(SignalDesign(DUAL_DESIGN).scaling(), GroupedL1Scaling)

    def test_linear_keeps_flat_scaling(self) -> None:
        assert isinstance(SignalDesign(LINEAR_DESIGN).scaling(), L1Scaling)

    def test_dual_builds_a_dual_model(self) -> None:
        model = SignalDesign(DUAL_DESIGN).model(Genome({"entry:rsi": 1.0}), ("entry:rsi",))
        assert isinstance(model, DualSignal)
