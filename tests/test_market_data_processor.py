import pandas as pd
import pytest

from coinbase.coinbase_adapter import CoinbaseError
from coinbase.ga.market_data_processor import (
    AccountBalance,
    CachedHistoricalCandles,
    CandleCacheFile,
    CandleCacheKey,
    ChunkedTimeRange,
    Delta,
    HistoricalCandles,
    HistoricalMarketData,
    IndicatorFrame,
    IndicatorPeriods,
    IsoDate,
    LiveMarketState,
    Macd,
    MarketDataConfig,
    MinMaxColumn,
    NormalizedIndicators,
    Rsi,
    Sma,
    TrainTestSplit,
)


# ── Test double ────────────────────────────────────────────────────────

class FakeAdapter:
    """Stands in for CoinbaseAdapter — only the methods market data touches."""

    def __init__(self, candles_by_window: dict[tuple[int, int], list[dict]], accounts: list[dict]) -> None:
        self._candles_by_window = candles_by_window
        self._accounts          = accounts
        self.candle_calls: list[tuple[int, int]] = []

    async def get_product_candles(self, product_id: str, start: int, end: int, granularity: str) -> list[dict]:
        self.candle_calls.append((start, end))
        return self._candles_by_window.get((start, end), [])

    async def get_accounts(self, limit: int = 250) -> list[dict]:
        return self._accounts


def _candle(start: int, close: float, high: float = None, low: float = None, volume: float = 1.0) -> dict:
    return {
        "start":  str(start),
        "close":  str(close),
        "high":   str(high if high is not None else close),
        "low":    str(low if low is not None else close),
        "volume": str(volume),
    }


def _account(currency: str, available: str) -> dict:
    return {"currency": currency, "available_balance": {"value": available, "currency": currency}}


def _rising_candles(n: int, start_ts: int = 0, step: int = 3600) -> list[dict]:
    return [_candle(start_ts + i * step, close=100.0 + i) for i in range(n)]


# ── Sma / Rsi / Macd ───────────────────────────────────────────────────

def test_sma_series_matches_rolling_mean():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sma    = Sma(closes, period=2)
    assert sma.series.iloc[-1] == pytest.approx(4.5)
    assert pd.isna(sma.series.iloc[0])


def test_rsi_series_is_100_when_all_gains():
    closes = pd.Series([float(i) for i in range(1, 20)])
    rsi    = Rsi(closes, period=14)
    assert rsi.series.iloc[-1] == pytest.approx(100.0)


def test_rsi_series_is_0_when_all_losses():
    closes = pd.Series([float(20 - i) for i in range(1, 20)])
    rsi    = Rsi(closes, period=14)
    assert rsi.series.iloc[-1] == pytest.approx(0.0)


def test_rsi_series_is_neutral_when_price_is_flat():
    closes = pd.Series([100.0] * 20)
    rsi    = Rsi(closes, period=14)
    assert rsi.series.iloc[-1] == pytest.approx(50.0)


def test_macd_histogram_is_line_minus_signal():
    closes = pd.Series([float(i % 7) for i in range(60)])
    macd   = Macd(closes, fast=12, slow=26, signal=9)
    assert (macd.histogram == macd.line - macd.signal_line).all()


def test_delta_series_is_percent_change_over_n_periods():
    closes = pd.Series([100.0, 110.0, 121.0, 133.1])
    delta  = Delta(closes, period=2)
    assert delta.series.iloc[2] == pytest.approx((121.0 - 100.0) / 100.0)
    assert delta.series.iloc[3] == pytest.approx((133.1 - 110.0) / 110.0)
    assert pd.isna(delta.series.iloc[0])
    assert pd.isna(delta.series.iloc[1])


# ── MinMaxColumn ───────────────────────────────────────────────────────

def test_min_max_column_scales_to_unit_range():
    scaled = MinMaxColumn(pd.Series([10.0, 20.0, 30.0])).scaled()
    assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_column_constant_series_returns_midpoint():
    scaled = MinMaxColumn(pd.Series([5.0, 5.0, 5.0])).scaled()
    assert scaled.tolist() == pytest.approx([0.5, 0.5, 0.5])


# ── IndicatorFrame / NormalizedIndicators / TrainTestSplit ────────────

def test_indicator_frame_has_expected_columns_and_no_nan():
    raw   = _rising_candles(80)
    frame = IndicatorFrame(raw, IndicatorPeriods()).dataframe
    expected = {
        "timestamp", "close", "volume", "high", "low",
        "sma_short", "sma_long", "sma_extra",
        "rsi", "macd", "macd_signal", "macd_histogram",
        "delta_1", "delta_3", "delta_5", "delta_10",
    }
    assert expected.issubset(set(frame.columns))
    assert not frame.isna().any().any()


def test_indicator_frame_survives_when_delta_10_is_the_binding_warmup_constraint():
    # with every SMA/RSI/MACD period below 10, delta_10's own 10-row NaN
    # prefix becomes the tightest dropna() constraint instead of being masked
    # by sma_extra_period's usual (larger) warmup requirement
    periods = IndicatorPeriods(
        sma_short_period=2, sma_long_period=3, sma_extra_period=4,
        rsi_period=3, macd_fast=2, macd_slow=3, macd_signal=2,
    )
    raw   = _rising_candles(20)
    frame = IndicatorFrame(raw, periods).dataframe
    assert len(frame) > 0
    assert not frame["delta_10"].isna().any()


def test_indicator_frame_sorts_out_of_order_candles():
    raw   = list(reversed(_rising_candles(80)))
    frame = IndicatorFrame(raw, IndicatorPeriods()).dataframe
    assert frame["timestamp"].is_monotonic_increasing


_NORMALIZED_COLUMNS = ("sma_short", "sma_long", "sma_extra", "rsi", "macd")
_DELTA_COLUMNS      = ("delta_1", "delta_3", "delta_5", "delta_10")


def test_normalized_indicators_adds_norm_columns_in_unit_range():
    raw        = _rising_candles(80)
    frame      = IndicatorFrame(raw, IndicatorPeriods()).dataframe
    normalized = NormalizedIndicators(frame, _NORMALIZED_COLUMNS + _DELTA_COLUMNS).dataframe
    for column in (
        "norm_sma_short", "norm_sma_long", "norm_sma_extra", "norm_rsi", "norm_macd",
        "norm_delta_1", "norm_delta_3", "norm_delta_5", "norm_delta_10",
    ):
        assert normalized[column].between(0.0, 1.0).all()


def test_normalized_indicators_only_touches_the_injected_columns():
    raw        = _rising_candles(80)
    frame      = IndicatorFrame(raw, IndicatorPeriods()).dataframe
    normalized = NormalizedIndicators(frame, ("rsi",)).dataframe
    assert "norm_rsi" in normalized.columns
    assert "norm_delta_1" not in normalized.columns


def test_train_test_split_respects_fraction():
    frame = pd.DataFrame({"x": range(10)})
    split = TrainTestSplit(frame, test_fraction=0.2)
    assert len(split.train()) == 8
    assert len(split.test()) == 2


# ── ChunkedTimeRange ───────────────────────────────────────────────────

def test_chunked_time_range_splits_long_span():
    windows = ChunkedTimeRange(start=0, end=3600 * 1000, candle_seconds=3600, max_candles=300).windows()
    assert windows[0] == (0, 3600 * 300)
    assert windows[-1][1] == 3600 * 1000
    assert len(windows) == 4


def test_chunked_time_range_single_window_when_short():
    windows = ChunkedTimeRange(start=0, end=3600 * 10, candle_seconds=3600, max_candles=300).windows()
    assert windows == [(0, 3600 * 10)]


# ── HistoricalCandles ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_historical_candles_pages_across_chunks_and_caches():
    start, end = 0, 3600 * 700
    window_1   = (0, 3600 * 300)
    window_2   = (3600 * 300, 3600 * 600)
    window_3   = (3600 * 600, 3600 * 700)
    adapter = FakeAdapter(
        candles_by_window={
            window_1: _rising_candles(300, start_ts=0),
            window_2: _rising_candles(300, start_ts=3600 * 300),
            window_3: _rising_candles(100, start_ts=3600 * 600),
        },
        accounts=[],
    )
    candles = HistoricalCandles(adapter, "BTC-USDC", "ONE_HOUR", start, end)
    raw     = await candles.raw()
    assert len(raw) == 700
    assert len(adapter.candle_calls) == 3

    await candles.raw()  # second call must not re-fetch
    assert len(adapter.candle_calls) == 3


@pytest.mark.asyncio
async def test_historical_candles_dedupes_boundary_candle():
    start, end = 0, 3600 * 600
    window_1   = (0, 3600 * 300)
    window_2   = (3600 * 300, 3600 * 600)
    boundary   = _candle(3600 * 300, close=999.0)
    adapter = FakeAdapter(
        candles_by_window={
            window_1: _rising_candles(299, start_ts=0) + [boundary],
            window_2: [boundary] + _rising_candles(299, start_ts=3600 * 301),
        },
        accounts=[],
    )
    raw = await HistoricalCandles(adapter, "BTC-USDC", "ONE_HOUR", start, end).raw()
    assert len(raw) == 599  # boundary candle counted once, not twice


# ── Candle disk cache ──────────────────────────────────────────────────

def test_candle_cache_key_filename_is_stable_for_the_same_window():
    key = CandleCacheKey("BTC-USDC", "ONE_HOUR", 0, 3600 * 80)
    assert key.filename() == CandleCacheKey("BTC-USDC", "ONE_HOUR", 0, 3600 * 80).filename()


def test_candle_cache_key_filename_differs_for_a_different_window():
    a = CandleCacheKey("BTC-USDC", "ONE_HOUR", 0, 3600 * 80)
    b = CandleCacheKey("BTC-USDC", "ONE_HOUR", 0, 3600 * 90)
    assert a.filename() != b.filename()


def test_candle_cache_file_round_trips_candles(tmp_path):
    key        = CandleCacheKey("BTC-USDC", "ONE_HOUR", 0, 3600 * 80)
    cache_file = CandleCacheFile(str(tmp_path), key)
    candles    = _rising_candles(5)

    assert not cache_file.exists()
    cache_file.write(candles)
    assert cache_file.exists()
    assert cache_file.read() == candles


@pytest.mark.asyncio
async def test_cached_historical_candles_fetches_once_then_reads_disk(tmp_path):
    start, end = 0, 3600 * 80
    window     = (0, 3600 * 80)
    adapter = FakeAdapter(
        candles_by_window={window: _rising_candles(80, start_ts=0)},
        accounts=[],
    )
    key        = CandleCacheKey("BTC-USDC", "ONE_HOUR", start, end)
    cache_file = CandleCacheFile(str(tmp_path), key)

    first_raw = await CachedHistoricalCandles(
        HistoricalCandles(adapter, "BTC-USDC", "ONE_HOUR", start, end), cache_file,
    ).raw()
    assert len(adapter.candle_calls) == 1

    second_raw = await CachedHistoricalCandles(
        HistoricalCandles(adapter, "BTC-USDC", "ONE_HOUR", start, end), cache_file,
    ).raw()
    assert second_raw == first_raw
    assert len(adapter.candle_calls) == 1  # second instance never touched the adapter


# ── AccountBalance ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_account_balance_available_returns_matching_currency():
    adapter = FakeAdapter({}, accounts=[_account("BTC", "1.5"), _account("USDC", "1000")])
    assert await AccountBalance(adapter, "BTC").available() == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_account_balance_available_is_zero_when_missing():
    adapter = FakeAdapter({}, accounts=[_account("USDC", "1000")])
    assert await AccountBalance(adapter, "BTC").available() == 0.0


# ── HistoricalMarketData / LiveMarketState ─────────────────────────────

@pytest.mark.asyncio
async def test_historical_market_data_dataframe_end_to_end(tmp_path):
    start, end = 0, 3600 * 80
    window     = (0, 3600 * 80)
    adapter = FakeAdapter(
        candles_by_window={window: _rising_candles(80, start_ts=0)},
        accounts=[],
    )
    market_data = HistoricalMarketData(
        adapter, "BTC-USDC", "ONE_HOUR", start, end, IndicatorPeriods(),
        _NORMALIZED_COLUMNS + _DELTA_COLUMNS, str(tmp_path),
    )
    frame = await market_data.dataframe()
    assert "norm_rsi" in frame.columns
    assert len(frame) > 0


@pytest.mark.asyncio
async def test_historical_market_data_reuses_disk_cache_across_instances(tmp_path):
    start, end = 0, 3600 * 80
    window     = (0, 3600 * 80)
    adapter = FakeAdapter(
        candles_by_window={window: _rising_candles(80, start_ts=0)},
        accounts=[],
    )

    def market_data() -> HistoricalMarketData:
        return HistoricalMarketData(
            adapter, "BTC-USDC", "ONE_HOUR", start, end, IndicatorPeriods(),
            _NORMALIZED_COLUMNS + _DELTA_COLUMNS, str(tmp_path),
        )

    first  = await market_data().dataframe()
    second = await market_data().dataframe()
    assert len(adapter.candle_calls) == 1  # second instance read the disk cache, not the adapter
    assert first.equals(second)


@pytest.mark.asyncio
async def test_live_market_state_snapshot_uses_latest_candle_and_balances():
    adapter = FakeAdapter(candles_by_window={}, accounts=[_account("BTC", "0.5"), _account("USDC", "200")])

    # patch in the exact window LiveMarketState will request
    import time as time_module
    now = int(time_module.time())

    async def get_product_candles(product_id, start, end, granularity):
        adapter.candle_calls.append((start, end))
        return [_candle(now - 3600, close=42.0), _candle(now, close=43.0)]

    adapter.get_product_candles = get_product_candles

    snapshot = await LiveMarketState(adapter, "BTC-USDC", "ONE_HOUR").snapshot()
    assert snapshot.close == pytest.approx(43.0)
    assert snapshot.account_base == pytest.approx(0.5)
    assert snapshot.account_quote == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_live_market_state_raises_when_no_candles_available():
    adapter = FakeAdapter(candles_by_window={}, accounts=[])
    with pytest.raises(CoinbaseError):
        await LiveMarketState(adapter, "BTC-USDC", "ONE_HOUR").snapshot()


# ── IsoDate / MarketDataConfig ──────────────────────────────────────────

def test_iso_date_timestamp_is_midnight_utc():
    assert IsoDate("2024-01-01").timestamp() == 1704067200


def test_market_data_config_window_and_periods():
    raw = {
        "data": {
            "pair": "BTC-USDC",
            "granularity": "ONE_HOUR",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "test_split": 0.25,
        },
        "strategy": {
            "indicators": {
                "sma_short_period": 9,
                "sma_long_period": 20,
                "sma_extra_period": 50,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
            }
        },
        "market_data": {
            "cache_dir": "./candle_cache",
            "normalized_columns": ["sma_short", "rsi"],
            "delta_columns": ["delta_1"],
        },
    }
    config = MarketDataConfig(raw)
    window = config.window()
    assert window.pair == "BTC-USDC"
    assert window.granularity == "ONE_HOUR"
    assert window.test_split == pytest.approx(0.25)
    assert window.end > window.start
    assert config.periods() == IndicatorPeriods()
    assert config.normalized_columns() == ("sma_short", "rsi")
    assert config.delta_columns() == ("delta_1",)
    assert config.columns() == ("sma_short", "rsi", "delta_1")
    assert config.cache_dir() == "./candle_cache"
