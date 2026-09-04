import pandas as pd
import pytest

from coinbase.ga.ga_engine import Genome
from coinbase.ga.pair_selector import (
    AlignedFrames,
    BestRunPerPair,
    BuyAndHold,
    Candidate,
    Conviction,
    PairRanking,
    SelectionRun,
    WilsonInterval,
)
from coinbase.ga.paper_trading import BasisPointFee, NoFees
from coinbase.ga.strategy_evaluator import GaStrategy, SignalDesign, StrategyConfig
from coinbase.trading_strategy import Action, Direction


# ── Builders ───────────────────────────────────────────────────────────

def _config(buy: float = 0.6, sell: float = 0.4, short: bool = False,
            size: float = 1.0) -> StrategyConfig:
    return StrategyConfig(
        position_size_pct     = size,
        buy_threshold         = buy,
        sell_threshold        = sell,
        starting_balance      = 1000.0,
        allow_short           = short,
        short_entry_threshold = 0.25,
        short_exit_threshold  = 0.45,
    )


# One weight on one key, so signal_score is exactly the norm_rsi column and a
# test can dictate the score candle by candle.
def _strategy(config: StrategyConfig) -> GaStrategy:
    return GaStrategy(
        SignalDesign(config.design).model(Genome({"rsi": 1.0}), ("rsi",)), config,
    )


def _frame(scores: list[float], closes: list[float], step: int = 3600) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [i * step for i in range(len(scores))],
        "close":     closes,
        "high":      closes,
        "low":       closes,
        "norm_rsi":  scores,
    })


def _run(frames: dict[str, pd.DataFrame], configs: dict[str, StrategyConfig], fees=None) -> SelectionRun:
    run = SelectionRun(
        AlignedFrames(frames),
        {pair: _strategy(config) for pair, config in configs.items()},
        configs,
        1000.0,
        fees or NoFees(),
    )
    run.run()
    return run


# ── Conviction ─────────────────────────────────────────────────────────

def test_conviction_opens_a_long_above_the_buy_threshold():
    assert Conviction(0.8, _config(buy=0.6)).action() is Action.BUY


def test_conviction_holds_inside_the_band():
    assert Conviction(0.5, _config(buy=0.6)).action() is Action.HOLD


def test_conviction_will_not_short_when_shorting_is_disabled():
    assert Conviction(0.1, _config(short=False)).action() is Action.HOLD
    assert Conviction(0.1, _config(short=True)).action() is Action.SHORT


# The point of the class: a score is measured against its OWN genome's bar, so
# two pairs whose raw scores are not comparable can still be ranked.
def test_conviction_is_the_fraction_of_the_room_past_its_own_trigger():
    # 0.8 with a 0.6 bar has used half the 0.4 of room above it.
    assert Conviction(0.8, _config(buy=0.6)).value() == pytest.approx(0.5)
    # 0.8 with a 0.2 bar has used three quarters of its 0.8 of room.
    assert Conviction(0.8, _config(buy=0.2)).value() == pytest.approx(0.75)


# Weights sum to 1.0 including position_pnl, which contributes 0 when flat, so
# a genome carrying weight there can never reach 1.0. Measuring against 1.0
# made conviction a proxy for whose position_pnl weight was smallest.
def test_conviction_measures_against_the_genomes_real_ceiling_not_one():
    # Ceiling 0.7: a score of 0.7 is maximally bullish for THIS genome.
    assert Conviction(0.7, _config(buy=0.6), ceiling=0.7).value() == pytest.approx(1.0)
    # Against a notional ceiling of 1.0 the same score looks feeble.
    assert Conviction(0.7, _config(buy=0.6), ceiling=1.0).value() == pytest.approx(0.25)


def test_a_genome_whose_ceiling_is_below_its_buy_threshold_can_never_go_long():
    # FET-USDT's live genome: 0.488 on position_pnl leaves a 0.512 ceiling.
    assert not Conviction(0.5, _config(buy=0.6), ceiling=0.512).can_ever_go_long()
    assert Conviction(0.5, _config(buy=0.6), ceiling=0.933).can_ever_go_long()


def test_two_genomes_at_their_own_ceilings_rank_equally():
    # Neither should outrank the other for the accident of its pnl weight.
    high = Conviction(0.93, _config(buy=0.6), ceiling=0.93).value()
    low  = Conviction(0.63, _config(buy=0.6), ceiling=0.63).value()
    assert high == pytest.approx(low)


def test_conviction_ranks_a_lower_raw_score_higher_when_its_bar_is_lower():
    lenient = Conviction(0.65, _config(buy=0.2)).value()
    strict  = Conviction(0.85, _config(buy=0.8)).value()
    assert lenient > strict


def test_conviction_is_zero_when_holding():
    assert Conviction(0.5, _config()).value() == 0.0


def test_conviction_of_a_threshold_pinned_at_the_top_does_not_divide_by_zero():
    assert Conviction(1.5, _config(buy=1.0)).value() == 1.0


def test_short_conviction_grows_as_the_score_falls():
    shallow = Conviction(0.20, _config(short=True)).value()
    deep    = Conviction(0.05, _config(short=True)).value()
    assert deep > shallow


# ── PairRanking ────────────────────────────────────────────────────────

def test_ranking_picks_the_highest_conviction_candidate():
    best = PairRanking((
        Candidate("A-USDT", Action.BUY, 0.2, 0.7),
        Candidate("B-USDT", Action.BUY, 0.9, 0.8),
    )).best()
    assert best.pair == "B-USDT"


def test_ranking_ignores_pairs_that_are_not_signalling():
    best = PairRanking((
        Candidate("A-USDT", Action.HOLD, 0.0, 0.5),
        Candidate("B-USDT", Action.SHORT, 0.4, 0.1),
    )).best()
    assert best.pair == "B-USDT"


def test_ranking_returns_nothing_when_every_pair_is_holding():
    assert PairRanking((Candidate("A-USDT", Action.HOLD, 0.0, 0.5),)).best() is None


# ── AlignedFrames ──────────────────────────────────────────────────────

# Comparing a score from Tuesday against one from Monday would rank on
# staleness, so the index is the intersection, not the union.
def test_aligned_frames_use_only_timestamps_every_pair_shares():
    aligned = AlignedFrames({
        "A-USDT": _frame([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        "B-USDT": _frame([0.5, 0.5], [1.0, 1.0]),
    })
    assert aligned.timestamps == (0, 3600)


def test_aligned_frames_are_empty_when_no_timestamp_is_shared():
    a = pd.DataFrame({"timestamp": [0], "close": [1.0], "high": [1.0], "low": [1.0], "norm_rsi": [0.5]})
    b = pd.DataFrame({"timestamp": [999], "close": [1.0], "high": [1.0], "low": [1.0], "norm_rsi": [0.5]})
    assert AlignedFrames({"A-USDT": a, "B-USDT": b}).timestamps == ()


def test_aligned_frames_look_a_row_up_by_pair_and_timestamp():
    aligned = AlignedFrames({"A-USDT": _frame([0.1, 0.9], [10.0, 20.0])})
    assert aligned.row("A-USDT", 3600)["close"] == 20.0


# ── SelectionRun ───────────────────────────────────────────────────────

def test_the_selector_enters_the_pair_with_the_strongest_conviction():
    #  B signals harder than A on the first candle, then both go quiet.
    frames = {
        "A-USDT": _frame([0.65, 0.5, 0.5], [100.0, 110.0, 110.0]),
        "B-USDT": _frame([0.95, 0.5, 0.5], [100.0, 200.0, 200.0]),
    }
    run = _run(frames, {"A-USDT": _config(), "B-USDT": _config()})
    trades = run.result().trades()
    assert len(trades) == 1
    assert trades[0].pair == "B-USDT"


def test_the_selector_holds_one_position_at_a_time():
    frames = {
        "A-USDT": _frame([0.9, 0.9, 0.9], [100.0, 100.0, 100.0]),
        "B-USDT": _frame([0.9, 0.9, 0.9], [100.0, 100.0, 100.0]),
    }
    run = _run(frames, {"A-USDT": _config(), "B-USDT": _config()})
    # Both scream buy on every candle, but only one position ever opens, so at
    # most one trade closes across the window.
    assert len(run.result().trades()) <= 1


def test_a_position_is_held_until_its_own_exit_threshold_fires():
    frames = {"A-USDT": _frame([0.9, 0.9, 0.9, 0.1], [100.0, 100.0, 100.0, 130.0])}
    run    = _run(frames, {"A-USDT": _config(buy=0.6, sell=0.4)})
    trade  = run.result().trades()[0]
    # Entered at 100 on candle 0, sold at 130 on candle 3 when the score fell
    # through the sell threshold — not on candles 1 or 2.
    assert trade.trade.direction() is Direction.LONG
    assert trade.trade.profit() == pytest.approx(30.0 * (1000.0 / 100.0))


def test_no_trade_is_taken_when_nothing_signals():
    frames = {"A-USDT": _frame([0.5, 0.5, 0.5], [100.0, 100.0, 100.0])}
    assert _run(frames, {"A-USDT": _config()}).result().trades() == ()


# Backtest and Ledger are gross; a selector re-entering on every exit trades
# far more than a single-pair strategy, so the drag has to be charged.
def test_fees_are_charged_on_both_the_entry_and_the_exit():
    frames = {"A-USDT": _frame([0.9, 0.1], [100.0, 100.0])}
    run    = _run(frames, {"A-USDT": _config()}, fees=BasisPointFee(10.0))
    trade  = run.result().trades()[0]
    # 1000 notional in and 1000 out, 10 bps EACH SIDE = 1.0 + 1.0 = 2.0.
    assert run.result().fees_paid() == pytest.approx(2.0)
    assert trade.fee == pytest.approx(2.0)                # both legs on the trade
    assert trade.trade.profit() == pytest.approx(0.0)     # gross stays gross
    assert run.result().net_profit() == pytest.approx(-2.0)


# The reported total has to agree with the money that actually left the book;
# recording only the exit leg made them differ by every entry fee.
def test_reported_fees_match_what_left_the_ledger():
    frames = {"A-USDT": _frame([0.9, 0.1], [100.0, 100.0])}
    result = _run(frames, {"A-USDT": _config()}, fees=BasisPointFee(10.0)).result()
    assert result.equity_curve()[-1] == pytest.approx(1000.0 - result.fees_paid())


def test_a_gross_run_charges_nothing():
    frames = {"A-USDT": _frame([0.9, 0.1], [100.0, 100.0])}
    assert _run(frames, {"A-USDT": _config()}).result().fees_paid() == 0.0


def test_direction_accuracy_ignores_fees_while_the_win_rate_does_not():
    # A trade that moved the right way but not far enough to cover its costs.
    frames = {"A-USDT": _frame([0.9, 0.1], [100.0, 100.001])}
    result = _run(frames, {"A-USDT": _config()}, fees=BasisPointFee(50.0)).result()
    assert result.direction_accuracy() == 1.0    # price did go the bet way
    assert result.win_rate() == 0.0              # and it still lost money


def test_attribution_names_which_pair_earned_what():
    frames = {
        "A-USDT": _frame([0.9, 0.1, 0.5], [100.0, 120.0, 120.0]),
        "B-USDT": _frame([0.5, 0.5, 0.5], [100.0, 100.0, 100.0]),
    }
    attribution = _run(frames, {"A-USDT": _config(), "B-USDT": _config()}).result().attribution()
    assert list(attribution.index) == ["A-USDT"]
    assert attribution.loc["A-USDT", "trades"] == 1


def test_an_open_position_at_the_end_unwinds_at_its_entry_price():
    # Never exits, so the window cuts mid-hold; that must not be scored as a win.
    frames = {"A-USDT": _frame([0.9, 0.9, 0.9], [100.0, 500.0, 900.0])}
    result = _run(frames, {"A-USDT": _config()}).result()
    assert result.gross_profit() == pytest.approx(0.0)


# An unwound position was closed by the calendar, not by the strategy. It
# unwinds at exactly break-even, so leaving it in the denominator scores every
# such run as one guaranteed miss.
def test_a_position_unwound_by_the_window_is_excluded_from_the_rates():
    frames = {"A-USDT": _frame([0.9, 0.9, 0.9], [100.0, 500.0, 900.0])}
    result = _run(frames, {"A-USDT": _config()}).result()
    assert len(result.trades()) == 1
    assert result.trades()[0].unwound is True
    assert result.resolved_trades() == ()
    assert result.direction_accuracy() == 0.0     # no resolved trades, not "0% right"


def test_an_unwound_trade_does_not_drag_down_a_perfect_record():
    frames = {
        # A resolves correctly, then B is entered and never exits.
        "A-USDT": _frame([0.9, 0.1, 0.5, 0.5], [100.0, 120.0, 120.0, 120.0]),
        "B-USDT": _frame([0.5, 0.5, 0.9, 0.9], [100.0, 100.0, 100.0, 100.0]),
    }
    result = _run(frames, {"A-USDT": _config(), "B-USDT": _config()}).result()
    assert len(result.trades()) == 2
    assert len(result.resolved_trades()) == 1
    assert result.direction_accuracy() == 1.0     # not 0.5


# ── BuyAndHold ─────────────────────────────────────────────────────────

def test_buy_and_hold_measures_first_close_to_last():
    aligned = AlignedFrames({"A-USDT": _frame([0.5, 0.5], [100.0, 150.0])})
    assert BuyAndHold(aligned).returns["A-USDT"] == pytest.approx(0.5)


def test_equal_weight_is_the_mean_of_every_pair_held():
    aligned = AlignedFrames({
        "A-USDT": _frame([0.5, 0.5], [100.0, 150.0]),
        "B-USDT": _frame([0.5, 0.5], [100.0, 50.0]),
    })
    assert BuyAndHold(aligned).equal_weight() == pytest.approx(0.0)


def test_best_single_pair_is_reported_with_its_name():
    aligned = AlignedFrames({
        "A-USDT": _frame([0.5, 0.5], [100.0, 150.0]),
        "B-USDT": _frame([0.5, 0.5], [100.0, 300.0]),
    })
    assert BuyAndHold(aligned).best_pair() == ("B-USDT", pytest.approx(2.0))


# ── WilsonInterval ─────────────────────────────────────────────────────

# A bare "58% accurate" from 12 trades reads as evidence when it is
# indistinguishable from a coin — the interval is what says so.
def test_a_small_sample_gives_an_interval_too_wide_to_act_on():
    low, high = WilsonInterval(7, 12).bounds()
    assert low < 0.5 < high
    assert high - low > 0.4


def test_a_large_sample_narrows_the_interval():
    small = WilsonInterval(70, 120).bounds()
    large = WilsonInterval(700, 1200).bounds()
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_no_trades_gives_the_widest_possible_interval():
    assert WilsonInterval(0, 0).bounds() == (0.0, 1.0)


# ── BestRunPerPair ─────────────────────────────────────────────────────

def _index(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"run_id": r, "pair": p, "granularity": g, "annualized_yield": y} for r, p, g, y in rows]
    )


def test_best_run_per_pair_picks_the_highest_scoring_run():
    frame = _index([("r1", "A-USDT", "SIX_HOUR", 0.1), ("r2", "A-USDT", "SIX_HOUR", 0.9)])
    assert BestRunPerPair(frame, ("A-USDT",), "annualized_yield", "SIX_HOUR").run_ids() == {"A-USDT": "r2"}


# A genome trained on THIRTY_MINUTE candles reads its indicators off a
# different clock; applying it to SIX_HOUR rows is not the same strategy.
def test_best_run_per_pair_will_not_cross_granularities():
    frame = _index([("r1", "A-USDT", "THIRTY_MINUTE", 0.9), ("r2", "A-USDT", "SIX_HOUR", 0.1)])
    assert BestRunPerPair(frame, ("A-USDT",), "annualized_yield", "SIX_HOUR").run_ids() == {"A-USDT": "r2"}


def test_a_pair_with_no_trained_run_is_omitted_rather_than_guessed():
    frame = _index([("r1", "A-USDT", "SIX_HOUR", 0.9)])
    assert BestRunPerPair(frame, ("A-USDT", "B-USDT"), "annualized_yield", "SIX_HOUR").run_ids() == {"A-USDT": "r1"}
