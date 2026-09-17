"""Exit thresholds taken from the exit model's own score distribution.

Splitting entry from exit broke a guarantee linear gets for free: under linear
one score decides both, so a position opens above buy_threshold and closes only
when THAT score falls to sell_threshold — it is born far from its own exit. Over
467 linear entries, 0% were already past their exit threshold on the candle they
opened. For dual it was 21% to 31%, and the median hold was two candles at every
exit_pnl_scale tried.
"""
import dataclasses

import pandas as pd
import pytest

from coinbase.ga.ga_engine import Genome, GroupedL1Scaling
from coinbase.ga.strategy_evaluator import (
    DUAL_DESIGN,
    LINEAR_DESIGN,
    CalibratedExit,
    DualSignal,
    LinearSignal,
    StrategyConfig,
)
from coinbase.trading_strategy import Direction, Position

KEYS = ("entry:rsi", "exit:delta_1", "exit:position_pnl")


def config(design: str = DUAL_DESIGN, **kw) -> StrategyConfig:
    return dataclasses.replace(
        StrategyConfig(
            position_size_pct     = 0.6,
            buy_threshold         = 0.6,
            sell_threshold        = 0.4,
            starting_balance      = 10000.0,
            allow_short           = True,
            short_entry_threshold = 0.25,
            short_exit_threshold  = 0.45,
            design                = design,
        ),
        **kw,
    )


def frame(delta_values) -> pd.DataFrame:
    return pd.DataFrame({
        "norm_rsi": [0.5] * len(delta_values),
        "norm_delta_1": list(delta_values),
        "close": [100.0] * len(delta_values),
    })


def dual_model() -> DualSignal:
    scaled = GroupedL1Scaling().scaled(
        {"entry:rsi": 1.0, "exit:delta_1": 1.0, "exit:position_pnl": 1.0},
    )
    return DualSignal(Genome(scaled), KEYS, 0.02)


class TestCalibratedExit:
    def test_linear_is_left_exactly_as_it_was(self) -> None:
        cfg = config(LINEAR_DESIGN)
        model = LinearSignal(Genome({"rsi": 1.0}), ("rsi",))
        assert CalibratedExit(model, frame([0.5] * 10), cfg).config() == cfg

    # The thresholds should land inside the distribution the model actually
    # produces, not at numbers picked for a different model.
    def test_dual_thresholds_come_from_the_models_own_scores(self) -> None:
        values = [i / 100 for i in range(101)]
        out = CalibratedExit(dual_model(), frame(values), config()).config()
        assert out.sell_threshold != 0.4
        assert out.short_exit_threshold != 0.45
        assert out.sell_threshold < out.short_exit_threshold

    def test_a_tighter_quantile_puts_the_bands_further_apart(self) -> None:
        values = [i / 100 for i in range(101)]
        wide = CalibratedExit(dual_model(), frame(values), config(exit_quantile=0.05)).config()
        narrow = CalibratedExit(dual_model(), frame(values), config(exit_quantile=0.40)).config()
        assert wide.sell_threshold < narrow.sell_threshold
        assert wide.short_exit_threshold > narrow.short_exit_threshold

    # The point of the whole exercise: a position born at a typical score must
    # not already be past the band that closes it.
    def test_a_typical_entry_is_not_born_already_exiting(self) -> None:
        values = [i / 100 for i in range(101)]
        rows = frame(values)
        model = dual_model()
        out = CalibratedExit(model, rows, config(exit_quantile=0.10)).config()
        births = [
            model.score(row, Position(row["close"], 1.0, Direction.LONG))
            for row in rows.to_dict("records")
        ]
        doomed = [b for b in births if b < out.sell_threshold]
        assert len(doomed) / len(births) <= 0.15, (
            f"{len(doomed)/len(births):.0%} of positions are born past their own "
            f"exit; a 0.10 quantile should leave about 10%"
        )

    # A flat distribution would otherwise produce bands the validator rejects,
    # aborting a whole sweep on one pathological genome.
    def test_a_degenerate_distribution_still_yields_a_valid_config(self) -> None:
        from coinbase.ga.strategy_evaluator import ValidatedStrategyConfig

        out = CalibratedExit(dual_model(), frame([0.5] * 20), config()).config()
        assert ValidatedStrategyConfig(out).config() is out

    def test_the_quantile_must_name_a_tail(self) -> None:
        from coinbase.ga.strategy_evaluator import ValidatedStrategyConfig

        with pytest.raises(ValueError, match="exit_quantile"):
            ValidatedStrategyConfig(config(exit_quantile=0.8)).config()
