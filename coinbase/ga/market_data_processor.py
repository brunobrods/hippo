import asyncio
import functools
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from coinbase.ga.config import GA_RESULTS_ROOT, ConfigFile
from coinbase.market_scanner import GRANULARITY_SECONDS
from exchange.adapter import ExchangeAdapter, ExchangeError

# Fallback only — the window is normally chunked by the adapter's own ceiling
# (adapter.max_candles_per_request()), which differs per exchange.
MAX_CANDLES_PER_REQUEST = 300


# ── Config data ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorPeriods:
    sma_short_period: int = 9
    sma_long_period:  int = 20
    sma_extra_period: int = 50
    rsi_period:        int = 14
    macd_fast:         int = 12
    macd_slow:         int = 26
    macd_signal:       int = 9


@dataclass(frozen=True)
class LiveSnapshot:
    close:         float
    high:          float
    low:           float
    volume:        float
    account_base:  float
    account_quote: float


@dataclass(frozen=True)
class TrainingWindow:
    pair:        str
    granularity: str
    start:       int
    end:         int
    test_split:  float


class IsoDate:
    def __init__(self, value: str) -> None:
        self._value = value

    def timestamp(self) -> int:
        parsed = datetime.strptime(self._value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())


class MarketDataConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def window(self) -> TrainingWindow:
        data = self._raw["data"]
        return TrainingWindow(
            pair        = data["pair"],
            granularity = data["granularity"],
            start       = IsoDate(data["start_date"]).timestamp(),
            end         = IsoDate(data["end_date"]).timestamp(),
            test_split  = float(data["test_split"]),
        )

    def periods(self) -> IndicatorPeriods:
        return IndicatorPeriods(**self._raw["strategy"]["indicators"])

    def normalized_columns(self) -> tuple[str, ...]:
        return tuple(self._raw["market_data"]["normalized_columns"])

    def delta_columns(self) -> tuple[str, ...]:
        return tuple(self._raw["market_data"]["delta_columns"])

    def columns(self) -> tuple[str, ...]:
        return self.normalized_columns() + self.delta_columns()

    def cache_dir(self) -> str:
        default = str(GA_RESULTS_ROOT / "candle_cache")
        return self._raw["market_data"].get("cache_dir", default)


# ── Indicators ─────────────────────────────────────────────────────────

class Sma:
    def __init__(self, closes: pd.Series, period: int) -> None:
        self._closes = closes
        self._period = period

    @functools.cached_property
    def series(self) -> pd.Series:
        return self._closes.rolling(self._period).mean()


class Rsi:
    def __init__(self, closes: pd.Series, period: int) -> None:
        self._closes = closes
        self._period = period

    @functools.cached_property
    def series(self) -> pd.Series:
        delta    = self._closes.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self._period, min_periods=self._period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self._period, min_periods=self._period, adjust=False).mean()
        rs       = avg_gain / avg_loss.mask(avg_loss == 0)
        value    = 100 - 100 / (1 + rs)
        value    = value.where(avg_loss != 0, 100.0)          # no losses -> maximally overbought
        return value.where((avg_loss != 0) | (avg_gain != 0), 50.0)  # no movement at all -> neutral


class Macd:
    def __init__(self, closes: pd.Series, fast: int, slow: int, signal: int) -> None:
        self._closes = closes
        self._fast   = fast
        self._slow   = slow
        self._signal = signal

    @functools.cached_property
    def line(self) -> pd.Series:
        return self._ema(self._fast) - self._ema(self._slow)

    @functools.cached_property
    def signal_line(self) -> pd.Series:
        return self.line.ewm(span=self._signal, adjust=False).mean()

    @functools.cached_property
    def histogram(self) -> pd.Series:
        return self.line - self.signal_line

    def _ema(self, span: int) -> pd.Series:
        return self._closes.ewm(span=span, adjust=False).mean()


class Delta:
    def __init__(self, closes: pd.Series, period: int) -> None:
        self._closes = closes
        self._period = period

    @functools.cached_property
    def series(self) -> pd.Series:
        return self._closes.pct_change(periods=self._period)


class MinMaxColumn:
    def __init__(self, series: pd.Series) -> None:
        self._series = series

    def scaled(self) -> pd.Series:
        span = self._series.max() - self._series.min()
        if span == 0:
            return pd.Series(0.5, index=self._series.index)
        return (self._series - self._series.min()) / span


# ── Frame builders ─────────────────────────────────────────────────────

class IndicatorFrame:
    def __init__(self, raw_candles: list[dict], periods: IndicatorPeriods) -> None:
        self._raw     = raw_candles
        self._periods = periods

    @functools.cached_property
    def dataframe(self) -> pd.DataFrame:
        candles = sorted(self._raw, key=lambda c: int(c["start"]))
        closes  = pd.Series([float(c["close"])  for c in candles])
        highs   = pd.Series([float(c["high"])   for c in candles])
        lows    = pd.Series([float(c["low"])    for c in candles])
        volumes = pd.Series([float(c["volume"]) for c in candles])
        starts  = pd.Series([int(c["start"])    for c in candles])

        macd = Macd(closes, self._periods.macd_fast, self._periods.macd_slow, self._periods.macd_signal)

        frame = pd.DataFrame({
            "timestamp":      starts,
            "close":          closes,
            "volume":         volumes,
            "high":           highs,
            "low":            lows,
            "sma_short":      Sma(closes, self._periods.sma_short_period).series,
            "sma_long":       Sma(closes, self._periods.sma_long_period).series,
            "sma_extra":      Sma(closes, self._periods.sma_extra_period).series,
            "rsi":            Rsi(closes, self._periods.rsi_period).series,
            "macd":           macd.line,
            "macd_signal":    macd.signal_line,
            "macd_histogram": macd.histogram,
            "delta_1":        Delta(closes, 1).series,
            "delta_3":        Delta(closes, 3).series,
            "delta_5":        Delta(closes, 5).series,
            "delta_10":       Delta(closes, 10).series,
        })
        return frame.dropna().reset_index(drop=True)


class NormalizedIndicators:
    def __init__(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        self._frame   = frame
        self._columns = columns

    @functools.cached_property
    def dataframe(self) -> pd.DataFrame:
        normalized = self._frame.copy()
        for column in self._columns:
            normalized[f"norm_{column}"] = MinMaxColumn(self._frame[column]).scaled()
        return normalized


class TrainTestSplit:
    def __init__(self, frame: pd.DataFrame, test_fraction: float) -> None:
        self._frame         = frame
        self._test_fraction = test_fraction

    def train(self) -> pd.DataFrame:
        return self._frame.iloc[: self._split_index()].reset_index(drop=True)

    def test(self) -> pd.DataFrame:
        return self._frame.iloc[self._split_index():].reset_index(drop=True)

    def _split_index(self) -> int:
        return int(len(self._frame) * (1 - self._test_fraction))


# ── Coinbase fetch ─────────────────────────────────────────────────────

class ChunkedTimeRange:
    def __init__(self, start: int, end: int, candle_seconds: int, max_candles: int = MAX_CANDLES_PER_REQUEST) -> None:
        self._start          = start
        self._end            = end
        self._candle_seconds = candle_seconds
        self._max_candles    = max_candles

    def windows(self) -> list[tuple[int, int]]:
        chunk_seconds = self._candle_seconds * self._max_candles
        windows: list[tuple[int, int]] = []
        chunk_start = self._start
        while chunk_start < self._end:
            chunk_end = min(chunk_start + chunk_seconds, self._end)
            windows.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        return windows


class ThrottledCandlePage:
    def __init__(
        self,
        adapter: CoinbaseAdapter,
        product_id: str,
        granularity: str,
        limit: asyncio.Semaphore,
    ) -> None:
        self._adapter     = adapter
        self._product_id  = product_id
        self._granularity = granularity
        self._limit       = limit

    async def fetch(self, start: int, end: int) -> list[dict]:
        async with self._limit:
            return await self._adapter.get_product_candles(
                self._product_id, start, end, self._granularity
            )


class HistoricalCandles:
    # A fine granularity over a long window chunks into hundreds of requests —
    # 876 days of THIRTY_MINUTE candles is 141 of them. Awaiting all of those at
    # once does not fetch faster; they queue against the adapter's connection
    # pool and the exchange's rate limit until each exceeds its own request
    # timeout, so the whole fetch fails. Bounded concurrency keeps every request
    # inside its timeout while still overlapping enough to stay quick.
    def __init__(
        self,
        adapter: ExchangeAdapter,
        product_id: str,
        granularity: str,
        start: int,
        end: int,
        max_concurrent_requests: int = 8,
    ) -> None:
        self._adapter                 = adapter
        self._product_id              = product_id
        self._granularity             = granularity
        self._start                   = start
        self._end                     = end
        self._max_concurrent_requests = max_concurrent_requests
        self._cache: Optional[list[dict]] = None

    async def raw(self) -> list[dict]:
        if self._cache is None:
            windows = ChunkedTimeRange(
                self._start, self._end, GRANULARITY_SECONDS[self._granularity],
                self._adapter.max_candles_per_request(),
            ).windows()
            page = ThrottledCandlePage(
                self._adapter, self._product_id, self._granularity,
                asyncio.Semaphore(self._max_concurrent_requests),
            )
            pages = await asyncio.gather(*(
                page.fetch(w_start, w_end) for w_start, w_end in windows
            ))
            by_start = {candle["start"]: candle for page_ in pages for candle in page_}
            self._cache = list(by_start.values())
        return self._cache


# ── Candle disk cache ──────────────────────────────────────────────────
# Persists raw candles across process runs, keyed by the exact window
# requested, so a parameter sweep that repeats a (pair, granularity,
# start, end) window many times only fetches it from Coinbase once.
# Deliberately not used by LiveMarketState, which needs the true latest
# candle on every call.

# Keyed by exchange first: "BTC-USDC"/"SIX_HOUR" name the same window on
# either venue but not the same prices, and a cached file is returned without
# ever consulting the adapter. Without this, flipping data.exchange and
# retraining would silently score Binance runs on Coinbase candles.
class CandleCacheKey:
    def __init__(self, exchange: str, pair: str, granularity: str, start: int, end: int) -> None:
        self._exchange    = exchange
        self._pair        = pair
        self._granularity = granularity
        self._start       = start
        self._end         = end

    def filename(self) -> str:
        return f"{self._exchange}_{self._pair}_{self._granularity}_{self._start}_{self._end}.json"


class CandleCacheFile:
    def __init__(self, directory: str, key: CandleCacheKey) -> None:
        self._directory = directory
        self._key       = key

    def exists(self) -> bool:
        return os.path.exists(self._path())

    def read(self) -> list[dict]:
        with open(self._path()) as handle:
            return json.load(handle)

    def write(self, candles: list[dict]) -> None:
        os.makedirs(self._directory, exist_ok=True)
        tmp_path = f"{self._path()}.tmp-{os.getpid()}"
        with open(tmp_path, "w") as handle:
            json.dump(candles, handle)
        os.replace(tmp_path, self._path())  # atomic — a crash or concurrent writer never leaves a truncated cache file

    def _path(self) -> str:
        return os.path.join(self._directory, self._key.filename())


class CachedHistoricalCandles:
    def __init__(self, candles: HistoricalCandles, cache_file: CandleCacheFile) -> None:
        self._candles    = candles
        self._cache_file = cache_file

    async def raw(self) -> list[dict]:
        if self._cache_file.exists():
            return self._cache_file.read()
        raw = await self._candles.raw()
        self._cache_file.write(raw)
        return raw


# Coinbase wallets are account-wide, so a currency names exactly one balance.
# Binance isolated margin holds a separate wallet per pair, so the same asset
# appears once per pair and its accounts carry a "product_id". Passing one
# scopes the lookup to that pair; accounts without the key match regardless,
# which leaves Coinbase's behaviour unchanged.
class AccountBalance:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        currency: str,
        product_id: Optional[str] = None,
    ) -> None:
        self._adapter    = adapter
        self._currency   = currency
        self._product_id = product_id

    async def available(self) -> float:
        accounts = await self._adapter.get_accounts(limit=250)
        for account in accounts:
            if account["currency"] != self._currency:
                continue
            if not self._in_scope(account):
                continue
            return float(account.get("available_balance", {}).get("value", 0.0) or 0.0)
        return 0.0

    def _in_scope(self, account: dict) -> bool:
        scope = account.get("product_id")
        return scope is None or self._product_id is None or scope == self._product_id


# ── Entry points ───────────────────────────────────────────────────────

class HistoricalMarketData:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        product_id: str,
        granularity: str,
        start: int,
        end: int,
        periods: IndicatorPeriods,
        normalized_columns: tuple[str, ...],
        cache_dir: Optional[str] = None,
    ) -> None:
        self._adapter            = adapter
        self._product_id         = product_id
        self._granularity        = granularity
        self._start              = start
        self._end                = end
        self._periods            = periods
        self._normalized_columns = normalized_columns
        self._cache_dir          = cache_dir

    async def dataframe(self) -> pd.DataFrame:
        base    = HistoricalCandles(self._adapter, self._product_id, self._granularity, self._start, self._end)
        candles = base if self._cache_dir is None else CachedHistoricalCandles(
            base,
            CandleCacheFile(
                self._cache_dir,
                CandleCacheKey(
                    self._adapter.name(), self._product_id, self._granularity,
                    self._start, self._end,
                ),
            ),
        )
        raw   = await candles.raw()
        frame = IndicatorFrame(raw, self._periods).dataframe
        return NormalizedIndicators(frame, self._normalized_columns).dataframe


class LiveMarketState:
    def __init__(self, adapter: ExchangeAdapter, product_id: str, granularity: str) -> None:
        self._adapter     = adapter
        self._product_id  = product_id
        self._granularity = granularity

    async def snapshot(self) -> LiveSnapshot:
        base, quote = self._product_id.split("-")
        end         = int(time.time())
        start       = end - GRANULARITY_SECONDS[self._granularity] * 2
        candles, base_balance, quote_balance = await asyncio.gather(
            HistoricalCandles(self._adapter, self._product_id, self._granularity, start, end).raw(),
            AccountBalance(self._adapter, base, self._product_id).available(),
            AccountBalance(self._adapter, quote, self._product_id).available(),
        )
        if not candles:
            raise ExchangeError(0, {"product_id": self._product_id}, f"No recent candles for {self._product_id}")
        latest = max(candles, key=lambda c: int(c["start"]))
        return LiveSnapshot(
            close         = float(latest["close"]),
            high          = float(latest["high"]),
            low           = float(latest["low"]),
            volume        = float(latest["volume"]),
            account_base  = base_balance,
            account_quote = quote_balance,
        )


# ── Smoke test ─────────────────────────────────────────────────────────
# Run (as a module, from the repo root — a direct script path fails with
# ModuleNotFoundError: No module named 'coinbase'):
#   python -m coinbase.ga.market_data_processor
# Fetches a small BTC-USDC window and prints the indicator frame's tail.

async def _main() -> None:
    from exchange.selection import ConfiguredExchange

    raw_config = ConfigFile("coinbase/ga/config.yaml").raw()
    config     = MarketDataConfig(raw_config)
    window     = config.window()

    async with ConfiguredExchange(raw_config).adapter() as adapter:
        market_data = HistoricalMarketData(
            adapter, window.pair, window.granularity, window.start, window.end,
            config.periods(), config.columns(), config.cache_dir(),
        )
        frame = await market_data.dataframe()
        print(frame.tail())

        snapshot = await LiveMarketState(adapter, window.pair, window.granularity).snapshot()
        print(snapshot)


if __name__ == "__main__":
    asyncio.run(_main())
