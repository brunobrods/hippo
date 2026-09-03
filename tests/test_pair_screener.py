import asyncio
import os

import pandas as pd
import pytest

from coinbase.ga.market_data_processor import OhlcFrame
from coinbase.pair_screener import (
    DailyMoves,
    IsolatedPairScan,
    MetricRanking,
    PairCandles,
    PairMetrics,
    PairScan,
    RequiredAccuracy,
    RoundTripCost,
    ScreenFrame,
    ScreenerArguments,
    Shortlist,
    ShortlistFile,
    ThresholdHitRate,
    TradablePairs,
    UtcDayWindow,
    VenueFees,
    VenueSpreads,
)

DAY = 86400


# ── Test doubles ───────────────────────────────────────────────────────

# Records how many candle requests are ever in flight at once, mirroring
# _ConcurrencyRecordingAdapter in test_market_data_processor.
class _ConcurrencyRecordingAdapter:
    def __init__(self, candles: list[dict]) -> None:
        self._candles  = candles
        self.in_flight = 0
        self.peak      = 0
        self.calls     = 0

    async def get_product_candles(self, product_id: str, start: int, end: int, granularity: str) -> list[dict]:
        self.in_flight += 1
        self.peak       = max(self.peak, self.in_flight)
        self.calls     += 1
        await asyncio.sleep(0)
        self.in_flight -= 1
        return self._candles

    def max_candles_per_request(self) -> int:
        return 1000

    def name(self) -> str:
        return "binance"


class FailingCandles:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def raw(self) -> list[dict]:
        raise self._exc


class StubCandles:
    def __init__(self, candles: list[dict]) -> None:
        self._candles = candles

    async def raw(self) -> list[dict]:
        return self._candles


class SpreadAdapter:
    def __init__(self, books: dict[str, tuple[float, float]]) -> None:
        self._books    = books
        self.batches: list[tuple[str, ...]] = []

    async def get_best_bid_ask(self, *product_ids: str) -> dict:
        self.batches.append(product_ids)
        return {
            "pricebooks": [
                {
                    "product_id": pair,
                    "bids": [{"price": str(self._books[pair][0])}],
                    "asks": [{"price": str(self._books[pair][1])}],
                }
                for pair in product_ids if pair in self._books
            ]
        }


# ── Builders ───────────────────────────────────────────────────────────

def _candle(start: int, open_: float, high: float, low: float, close: float, volume: float = 1.0) -> dict:
    return {
        "start":  str(start),
        "open":   str(open_),
        "high":   str(high),
        "low":    str(low),
        "close":  str(close),
        "volume": str(volume),
    }


def _flat_candles(n: int, close: float = 100.0) -> list[dict]:
    return [_candle(i * DAY, close, close, close, close) for i in range(n)]


def _product(pair: str, quote: str = "USDT", volume: float = 10_000_000.0,
             tradable: bool = True, can_short: bool = True) -> dict:
    return {
        "product_id":       pair,
        "quote_currency":   quote,
        "tradable":         tradable,
        "can_short":        can_short,
        "volume_24h_quote": volume,
    }


def _metrics(pair: str, median_move: float, error: str = "") -> PairMetrics:
    return PairMetrics(
        pair=pair, exchange="binance", candles=180,
        move_hit_rate=0.6, range_hit_rate=0.8,
        median_move=median_move, mean_move=median_move,
        median_range=median_move * 2, atr_pct=median_move,
        up_day_rate=0.5, volume_24h_quote=1e7, can_short=True,
        error=error or None,
    )


# ── UtcDayWindow ───────────────────────────────────────────────────────

def test_utc_day_window_end_is_the_last_completed_midnight():
    window = UtcDayWindow(now=1000 * DAY + 3600 * 5, days=30)
    assert window.end() == 1000 * DAY


def test_utc_day_window_start_is_exactly_days_before_end():
    window = UtcDayWindow(now=1000 * DAY + 1, days=180)
    assert window.end() - window.start() == 180 * DAY


# The whole point of snapping: CandleCacheKey interpolates these ints into the
# filename, so two runs on the same day must produce the same key or every run
# is a cold refetch of the entire universe.
def test_utc_day_window_is_identical_for_two_times_within_the_same_utc_day():
    morning = UtcDayWindow(now=1000 * DAY + 60, days=90)
    evening = UtcDayWindow(now=1000 * DAY + DAY - 1, days=90)
    assert (morning.start(), morning.end()) == (evening.start(), evening.end())


def test_utc_day_window_advances_by_one_day_after_midnight():
    before = UtcDayWindow(now=1000 * DAY + DAY - 1, days=90)
    after  = UtcDayWindow(now=1001 * DAY, days=90)
    assert after.end() - before.end() == DAY


# ── TradablePairs ──────────────────────────────────────────────────────

def test_tradable_universe_drops_a_pair_below_the_minimum_quote_volume():
    products = [_product("BTC-USDT", volume=1e7), _product("DUST-USDT", volume=1_000.0)]
    kept = TradablePairs(products, "USDT", 5e6, False, 100).pairs()
    assert kept == ("BTC-USDT",)


def test_tradable_universe_drops_a_pair_whose_quote_currency_differs():
    products = [_product("BTC-USDT"), _product("BTC-BUSD", quote="BUSD")]
    assert TradablePairs(products, "USDT", 0.0, False, 100).pairs() == ("BTC-USDT",)


def test_tradable_universe_drops_a_pair_that_is_not_tradable():
    products = [_product("BTC-USDT"), _product("DEAD-USDT", tradable=False)]
    assert TradablePairs(products, "USDT", 0.0, False, 100).pairs() == ("BTC-USDT",)


def test_tradable_universe_drops_a_short_disabled_pair_when_shorts_are_required():
    products = [_product("BTC-USDT", can_short=True), _product("LONGONLY-USDT", can_short=False)]
    assert TradablePairs(products, "USDT", 0.0, True, 100).pairs() == ("BTC-USDT",)
    assert len(TradablePairs(products, "USDT", 0.0, False, 100).pairs()) == 2


def test_capped_universe_keeps_the_highest_volume_pairs():
    products = [
        _product("THIN-USDT", volume=6e6),
        _product("FAT-USDT",  volume=9e9),
        _product("MID-USDT",  volume=5e8),
    ]
    assert TradablePairs(products, "USDT", 0.0, False, 2).pairs() == ("FAT-USDT", "MID-USDT")


# ── Statistics ─────────────────────────────────────────────────────────

def test_daily_moves_range_fraction_is_high_minus_low_over_open():
    frame = OhlcFrame([_candle(0, open_=100.0, high=110.0, low=90.0, close=105.0)]).dataframe
    assert DailyMoves(frame).range_fraction.iloc[0] == pytest.approx(0.20)


def test_daily_moves_close_change_is_the_day_over_day_return():
    candles = [
        _candle(0 * DAY, 100.0, 100.0, 100.0, 100.0),
        _candle(1 * DAY, 100.0, 102.0, 100.0, 102.0),
    ]
    changes = DailyMoves(OhlcFrame(candles).dataframe).close_change
    assert len(changes) == 1                      # the first row has no predecessor
    assert changes.iloc[0] == pytest.approx(0.02)


def test_threshold_hit_rate_counts_moves_in_both_directions():
    series = pd.Series([0.02, -0.03, 0.001])
    assert ThresholdHitRate(series, 0.01).fraction() == pytest.approx(2 / 3)


def test_threshold_hit_rate_is_zero_for_a_flat_series():
    assert ThresholdHitRate(pd.Series([0.0, 0.0]), 0.01).fraction() == 0.0


def test_threshold_hit_rate_is_zero_for_an_empty_series():
    assert ThresholdHitRate(pd.Series([], dtype="float64"), 0.01).fraction() == 0.0


# ── Cost ───────────────────────────────────────────────────────────────

def test_maker_cost_earns_the_spread_while_taker_cost_pays_it():
    fees = VenueFees("binance", maker_bps=10.0, taker_bps=10.0, measured=True)
    assert RoundTripCost(fees, spread_bps=20.0, maker=False).bps() == pytest.approx(40.0)
    assert RoundTripCost(fees, spread_bps=20.0, maker=True).bps()  == pytest.approx(0.0)


def test_round_trip_cost_charges_the_fee_on_both_legs():
    fees = VenueFees("coinbase", maker_bps=60.0, taker_bps=120.0, measured=True)
    assert RoundTripCost(fees, spread_bps=0.0, maker=False).bps() == pytest.approx(240.0)
    assert RoundTripCost(fees, spread_bps=0.0, maker=True).bps()  == pytest.approx(120.0)


def test_required_accuracy_is_seventy_percent_for_a_three_percent_mover_at_twenty_bps():
    assert RequiredAccuracy(move=0.03, cost=0.002, target=0.01).value() == pytest.approx(0.70)


def test_required_accuracy_exceeds_one_when_the_cost_floor_exceeds_the_move():
    # Coinbase base tier: 240 bps round trip against a 3% mover.
    assert RequiredAccuracy(move=0.03, cost=0.024, target=0.01).value() > 1.0


def test_required_accuracy_is_infinite_for_a_pair_that_does_not_move():
    assert RequiredAccuracy(move=0.0, cost=0.002, target=0.01).value() == float("inf")


def test_fees_describe_says_when_the_rate_is_a_fallback_rather_than_measured():
    assert "measured" in VenueFees("binance", 10.0, 10.0, True).describe()
    assert "FALLBACK" in VenueFees("binance", 10.0, 10.0, False).describe()


# ── Spreads ────────────────────────────────────────────────────────────

async def test_supplied_spreads_are_used_without_any_request():
    adapter = SpreadAdapter({})
    spreads = VenueSpreads(adapter, ("BTC-USDT",), {"BTC-USDT": 4.0})
    await spreads.run()
    assert spreads.bps() == {"BTC-USDT": 4.0}
    assert adapter.batches == []          # ticker/24hr already carried it


async def test_missing_spreads_are_fetched_and_converted_to_basis_points():
    adapter = SpreadAdapter({"ETH-USDC": (99.0, 101.0)})
    spreads = VenueSpreads(adapter, ("ETH-USDC",), {})
    await spreads.run()
    assert spreads.bps()["ETH-USDC"] == pytest.approx(200.0)


async def test_a_failed_spread_batch_does_not_stop_the_scan():
    class Boom:
        async def get_best_bid_ask(self, *product_ids: str) -> dict:
            raise RuntimeError("rate limited")

    spreads = VenueSpreads(Boom(), ("ETH-USDC",), {})
    await spreads.run()
    assert spreads.bps() == {}


# ── Isolation ──────────────────────────────────────────────────────────

async def test_isolated_pair_scan_reports_the_error_instead_of_raising():
    scan = IsolatedPairScan(
        PairScan(FailingCandles(RuntimeError("delisted")), "DEAD-USDT", "binance", 0.01, {}),
        "DEAD-USDT", "binance",
    )
    await scan.run()
    assert scan.metrics().error == "RuntimeError: delisted"
    assert scan.metrics().pair == "DEAD-USDT"


async def test_a_pair_with_too_few_candles_is_an_error_not_a_zero_row():
    scan = IsolatedPairScan(
        PairScan(StubCandles(_flat_candles(5)), "NEW-USDT", "binance", 0.01, {}),
        "NEW-USDT", "binance",
    )
    await scan.run()
    assert "need at least" in scan.metrics().error


async def test_pair_scans_complete_every_healthy_pair_when_one_pair_fails():
    healthy = [
        IsolatedPairScan(
            PairScan(StubCandles(_flat_candles(40)), f"OK{i}-USDT", "binance", 0.01, _product(f"OK{i}-USDT")),
            f"OK{i}-USDT", "binance",
        )
        for i in range(3)
    ]
    broken = IsolatedPairScan(
        PairScan(FailingCandles(ValueError("boom")), "BAD-USDT", "binance", 0.01, {}),
        "BAD-USDT", "binance",
    )
    scans = healthy + [broken]
    await asyncio.gather(*(scan.run() for scan in scans))
    assert [scan.metrics().error for scan in healthy] == [None, None, None]
    assert broken.metrics().error is not None


# ── Throttling ─────────────────────────────────────────────────────────

# The regression test for the failure HistoricalCandles' own docstring
# describes: without one shared semaphore, N pairs each build a private
# Semaphore(8) and put N x 8 requests in flight.
async def test_every_pair_fetches_through_one_shared_semaphore():
    adapter = _ConcurrencyRecordingAdapter(_flat_candles(40))
    window  = UtcDayWindow(now=1000 * DAY, days=180)
    limit   = asyncio.Semaphore(4)
    fetches = [
        PairCandles(adapter, f"P{i}-USDT", "ONE_DAY", window, "", limit, use_cache=False)
        for i in range(20)
    ]
    await asyncio.gather(*(fetch.raw() for fetch in fetches))
    assert adapter.calls == 20
    assert adapter.peak <= 4
    assert adapter.in_flight == 0


# ── Closed candles only ────────────────────────────────────────────────

# Binance treats klines' endTime as inclusive of a candle's OPEN, so a window
# ending at midnight comes back carrying the candle that opened at midnight —
# the one still forming, whose close is just the current price. Wilder
# smoothing gives that newest bar ~7% of the ATR, so leaving it in moved
# atr_pct by ~11% on real BTC data.
async def test_the_still_forming_candle_at_the_window_end_is_dropped():
    window  = UtcDayWindow(now=1000 * DAY, days=3)
    candles = [_candle(start, 100.0, 100.0, 100.0, 100.0)
               for start in (997 * DAY, 998 * DAY, 999 * DAY, 1000 * DAY)]
    adapter = _ConcurrencyRecordingAdapter(candles)
    fetched = await PairCandles(
        adapter, "BTC-USDT", "ONE_DAY", window, "", asyncio.Semaphore(1), use_cache=False,
    ).raw()
    assert [int(c["start"]) for c in fetched] == [997 * DAY, 998 * DAY, 999 * DAY]
    assert window.end() not in [int(c["start"]) for c in fetched]


# A file written before the filter existed still holds the partial candle, so
# the correction has to happen on read rather than only before the cache write.
async def test_a_cached_partial_candle_is_still_dropped_on_read(tmp_path):
    window  = UtcDayWindow(now=1000 * DAY, days=3)
    candles = [_candle(start, 100.0, 100.0, 100.0, 100.0)
               for start in (998 * DAY, 999 * DAY, 1000 * DAY)]
    adapter = _ConcurrencyRecordingAdapter(candles)
    fetch   = lambda: PairCandles(
        adapter, "BTC-USDT", "ONE_DAY", window, str(tmp_path), asyncio.Semaphore(1),
    ).raw()
    await fetch()                                   # writes the cache
    assert [int(c["start"]) for c in await fetch()] == [998 * DAY, 999 * DAY]


# ── Frame and shortlist ────────────────────────────────────────────────

def _frame_of(*metrics: PairMetrics, spread: float = 0.0) -> pd.DataFrame:
    return ScreenFrame(
        metrics,
        {metric.pair: spread for metric in metrics},
        VenueFees("binance", 10.0, 10.0, True),
        target=0.01,
    ).dataframe


def test_screen_frame_has_one_row_per_pair_including_failed_ones():
    frame = _frame_of(_metrics("A-USDT", 0.03), _metrics("B-USDT", 0.0, error="boom"))
    assert len(frame) == 2
    assert set(frame["pair"]) == {"A-USDT", "B-USDT"}


def test_net_edge_subtracts_the_round_trip_cost_from_the_median_move():
    frame = _frame_of(_metrics("A-USDT", 0.03), spread=0.0)
    # 3% = 300 bps, taker round trip = 2 x 10 bps
    assert frame.loc[0, "net_taker_bps"] == pytest.approx(280.0)


def test_viability_is_judged_on_the_taker_floor_not_the_maker_one():
    # A move of 15 bps clears the maker floor (0 bps at this spread) but not
    # the taker one (40 bps), and a maker fill is never guaranteed.
    frame = _frame_of(_metrics("THIN-USDT", 0.0015), spread=20.0)
    assert frame.loc[0, "net_maker_bps"] > 0
    assert frame.loc[0, "net_taker_bps"] < 0
    assert not frame.loc[0, "viable"]


# Clearing the cost floor only says a pair moves more than it costs to trade.
# A pair needing better than 100% direction accuracy cannot reach the target
# however good the strategy is, and must not reach the phase-2 shortlist.
def test_a_pair_that_cannot_reach_the_target_is_not_viable():
    # 0.5% median move: comfortably over a 20 bps floor, nowhere near 1%/day.
    frame = _frame_of(_metrics("BTC-USDT", 0.005))
    assert frame.loc[0, "net_taker_bps"] > 0
    assert frame.loc[0, "req_accuracy_taker"] > 1.0
    assert not frame.loc[0, "viable"]


def test_a_pair_that_moves_far_more_than_the_target_stays_viable():
    frame = _frame_of(_metrics("DEXE-USDT", 0.039))
    assert frame.loc[0, "req_accuracy_taker"] < 1.0
    assert frame.loc[0, "viable"]


# A missing spread must never be costed as if the book were free — the failure
# direction has to be pessimistic, or a batch timeout admits pairs.
def test_a_pair_with_an_unreadable_spread_is_never_viable():
    frame = ScreenFrame(
        (_metrics("THIN-USDT", 0.05),), {}, VenueFees("binance", 10.0, 10.0, True), target=0.01,
    ).dataframe
    assert not frame.loc[0, "viable"]


def test_shortlist_excludes_pairs_whose_net_edge_is_negative():
    frame = _frame_of(_metrics("GOOD-USDT", 0.03), _metrics("FLAT-USDT", 0.0001))
    rows  = Shortlist(frame, "net_maker_bps").rows()
    assert list(rows["pair"]) == ["GOOD-USDT"]


def test_shortlist_excludes_failed_pairs():
    frame = _frame_of(_metrics("GOOD-USDT", 0.03), _metrics("BAD-USDT", 0.05, error="boom"))
    assert list(Shortlist(frame, "net_maker_bps").rows()["pair"]) == ["GOOD-USDT"]


def test_shortlist_ranks_by_the_requested_metric():
    frame = _frame_of(_metrics("SMALL-USDT", 0.02), _metrics("BIG-USDT", 0.06))
    assert list(Shortlist(frame, "net_maker_bps").rows()["pair"]) == ["BIG-USDT", "SMALL-USDT"]


# Leaderboard always sorts descending, so ranking a lower-is-better column
# through it puts the WORST candidates on top of the shortlist phase 2 reads.
def test_a_lower_is_better_metric_ranks_ascending():
    frame = _frame_of(_metrics("EASY-USDT", 0.05), _metrics("HARD-USDT", 0.012))
    ranked = MetricRanking(frame, "req_accuracy_taker").rows()
    # The pair needing the LOWER direction accuracy is the better candidate.
    assert list(ranked["pair"]) == ["EASY-USDT", "HARD-USDT"]
    assert ranked.iloc[0]["req_accuracy_taker"] < ranked.iloc[1]["req_accuracy_taker"]


def test_a_higher_is_better_metric_still_ranks_descending():
    frame  = _frame_of(_metrics("SMALL-USDT", 0.02), _metrics("BIG-USDT", 0.06))
    ranked = MetricRanking(frame, "net_taker_bps").rows()
    assert list(ranked["pair"]) == ["BIG-USDT", "SMALL-USDT"]


# --top caps what the console prints. Capping the persisted shortlist too
# handed phase 2 a subset while the banner reported the full count as viable.
def test_the_shortlist_keeps_every_viable_pair_regardless_of_top():
    frame = _frame_of(*[_metrics(f"P{i}-USDT", 0.02 + i / 1000) for i in range(8)])
    assert len(Shortlist(frame, "net_taker_bps").rows()) == 8


def test_an_empty_shortlist_is_not_an_error():
    frame = _frame_of(_metrics("FLAT-USDT", 0.0))
    assert Shortlist(frame, "net_maker_bps").rows().empty


# ── Shortlist file ─────────────────────────────────────────────────────

def test_shortlist_file_writes_atomically_and_reads_back(tmp_path):
    store = ShortlistFile(str(tmp_path / "screener"), "shortlist_binance_latest.json")
    store.write({"pairs": ["BTC-USDT"], "metric": "net_maker_bps"})
    assert store.read()["pairs"] == ["BTC-USDT"]


def test_shortlist_file_leaves_no_temporary_file_behind(tmp_path):
    directory = tmp_path / "screener"
    ShortlistFile(str(directory), "out.json").write({"pairs": []})
    assert [entry.name for entry in directory.iterdir()] == ["out.json"]


def test_shortlist_file_overwrites_the_previous_run(tmp_path):
    store = ShortlistFile(str(tmp_path), "out.json")
    store.write({"pairs": ["OLD-USDT"]})
    store.write({"pairs": ["NEW-USDT"]})
    assert store.read()["pairs"] == ["NEW-USDT"]


# ── CLI ────────────────────────────────────────────────────────────────

def test_screener_arguments_read_the_given_argv():
    config = ScreenerArguments(["--exchange", "binance", "--days", "365", "--top", "5"]).config()
    assert (config.exchange, config.days, config.top) == ("binance", 365, 5)


def test_screener_arguments_default_the_quote_currency_per_exchange():
    assert ScreenerArguments(["--exchange", "binance"]).config().quote  == "USDT"
    assert ScreenerArguments(["--exchange", "coinbase"]).config().quote == "USDC"


def test_screener_arguments_convert_percentages_to_fractions():
    config = ScreenerArguments(["--threshold", "1.0", "--target", "2.5"]).config()
    assert config.threshold == pytest.approx(0.01)
    assert config.target    == pytest.approx(0.025)


def test_screener_arguments_keep_the_screener_cache_separate_from_the_ga_cache():
    config = ScreenerArguments([]).config()
    assert config.cache_dir.endswith("screener_cache")
