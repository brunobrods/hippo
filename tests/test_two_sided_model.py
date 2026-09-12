import pytest

from coinbase.ga.ga_engine import Genome
from coinbase.ga.strategy_evaluator import (
    POSITION_PNL_KEY,
    LinearSignal,
    StrategyConfig,
    TwoSidedModel,
)

KEYS = ("rsi", "macd", POSITION_PNL_KEY)


def config(buy_threshold: float = 0.6) -> StrategyConfig:
    return StrategyConfig(
        position_size_pct     = 0.6,
        buy_threshold         = buy_threshold,
        sell_threshold        = 0.4,
        starting_balance      = 10000.0,
        allow_short           = True,
        short_entry_threshold = 0.25,
        short_exit_threshold  = 0.45,
    )


def model(rsi: float, macd: float, position_pnl: float) -> LinearSignal:
    return LinearSignal(
        Genome({"rsi": rsi, "macd": macd, POSITION_PNL_KEY: position_pnl}), KEYS,
    )


class TestTwoSidedModel:
    # The live ETH book: 0.4804 on position_pnl leaves a 0.5196 ceiling against
    # a 0.6 buy_threshold, so it could only ever short.
    def test_refuses_a_genome_that_can_never_reach_its_buy_threshold(self) -> None:
        guarded = TwoSidedModel(model(0.4516, 0.0680, 0.4804), config())
        with pytest.raises(ValueError, match="can never open a long"):
            guarded.model()

    def test_names_the_ceiling_and_the_threshold_it_fails(self) -> None:
        guarded = TwoSidedModel(model(0.4516, 0.0680, 0.4804), config())
        with pytest.raises(ValueError, match=r"0\.5196"):
            guarded.model()

    def test_passes_a_genome_with_headroom_over_the_threshold(self) -> None:
        guarded = TwoSidedModel(model(0.5, 0.143, 0.357), config())
        assert guarded.model().flat_score_ceiling() == pytest.approx(0.643)

    def test_returns_the_same_model_it_was_given(self) -> None:
        original = model(0.5, 0.143, 0.357)
        assert TwoSidedModel(original, config()).model() is original

    # Exactly at the threshold is still refused: decide() opens a long on
    # `score > buy_threshold`, so a ceiling equal to it never crosses either.
    # 0.3 + 0.3 lands on the same double as 0.6, where 0.4 + 0.2 does not.
    def test_refuses_a_ceiling_exactly_at_the_threshold(self) -> None:
        assert model(0.3, 0.3, 0.4).flat_score_ceiling() == 0.6
        guarded = TwoSidedModel(model(0.3, 0.3, 0.4), config())
        with pytest.raises(ValueError, match="can never open a long"):
            guarded.model()

    def test_is_one_sided_reports_without_raising(self) -> None:
        assert TwoSidedModel(model(0.4516, 0.0680, 0.4804), config()).is_one_sided() is True
        assert TwoSidedModel(model(0.5, 0.143, 0.357), config()).is_one_sided() is False

    # The ceiling is a property of the genome; whether it is fatal depends on
    # the threshold it is measured against.
    def test_a_lower_buy_threshold_admits_the_same_genome(self) -> None:
        one_sided = model(0.4516, 0.0680, 0.4804)
        assert TwoSidedModel(one_sided, config(0.6)).is_one_sided() is True
        assert TwoSidedModel(one_sided, config(0.5)).is_one_sided() is False

    # position_pnl is the only key excluded from the ceiling, because it is the
    # only one that scores zero while flat.
    def test_position_pnl_weight_is_what_lowers_the_ceiling(self) -> None:
        assert model(0.5, 0.5, 0.0).flat_score_ceiling() == pytest.approx(1.0)
        assert model(0.25, 0.25, 0.5).flat_score_ceiling() == pytest.approx(0.5)
