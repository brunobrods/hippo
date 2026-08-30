"""
Pair screener — which markets actually move, and does the move survive costs?
-----------------------------------------------------------------------------

Discovers a venue's whole tradable universe, measures how far each pair moved
per candle over a trailing window, subtracts what a round trip costs there, and
ranks what is left.

Two different numbers are reported, and they are never conflated:

  range      (high - low) / open — the PERFECT-TIMING bound. Capturing it means
             buying the low and selling the high. Nobody does this. Shown for
             context; never ranked on.
  move       |close-to-close return| — the PERFECT-DIRECTION bound, capturable
             by a hold that got the direction right. Typically half the range
             or less. This is the honest number, and the default ranking uses it.

Neither is a forecast. Volatility persists; direction does not. A pair ranking
well here has been shown to MOVE, not to be predictable — which is why the
shortlist feeds the GA sweep rather than a trading rule.

`required_accuracy` is the column to read: given the median move and a cost
floor, it says how often a strategy must call the direction correctly to hit
the target. It routinely lands near 70%, against published daily-direction
models in the low 50s. The median is used rather than the mean so one spike
cannot make a quiet pair look tradable; both are reported, and a wide gap
between them means the movement was a handful of events, not a property.

Usage (run as a module, from the repo root — a direct script path fails with
ModuleNotFoundError: No module named 'coinbase'):
    python -m coinbase.pair_screener
    python -m coinbase.pair_screener --exchange binance --quote USDT --days 180
    python -m coinbase.pair_screener --exchange binance --top 25 --require-short
    python -m coinbase.pair_screener --exchange coinbase --quote USDC
"""

import argparse
import asyncio
import functools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from coinbase.ga.config import GA_RESULTS_ROOT
from coinbase.ga.market_data_processor import (
    AverageTrueRange,
    CachedHistoricalCandles,
    CandleCacheFile,
    CandleCacheKey,
    CandleRange,
    Delta,
    HistoricalCandles,
    OhlcFrame,
    TrueRange,
)
from coinbase.ga.results import ConsoleTable, Leaderboard
from coinbase.market_scanner import GRANULARITY_SECONDS
from exchange.adapter import ExchangeAdapter
from exchange.pool import ExchangePool

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────

DEFAULT_GRANULARITY = "ONE_DAY"
ATR_PERIOD          = 14

# Ranked on the TAKER edge, deliberately, even though the maker figure is
# usually larger. net_maker_bps credits the whole spread back, so it RISES with
# the spread — ranking on it promotes illiquid pairs bps for bps, and hands
# phase 2 a shortlist sorted worst-first on liquidity. It is reported as a
# column because the maker/taker gap is the thing worth seeing; it is not what
# the shortlist is ordered by.
DEFAULT_METRIC = "net_taker_bps"

# Every numeric column a run can be ranked by. Declared up front so argparse
# rejects a typo before the scan runs — the shortlist is built after every
# candle has been fetched, so validating there would throw the whole scan away.
RANKABLE_COLUMNS = (
    "volume_24h_quote", "candles", "move_hit_rate", "range_hit_rate",
    "median_move_pct", "mean_move_pct", "median_range_pct", "atr_pct", "up_day_rate",
    "spread_bps", "cost_taker_bps", "cost_maker_bps",
    "net_taker_bps", "net_maker_bps",
    "req_accuracy_taker", "req_accuracy_maker",
)

# Columns where a SMALLER number is the better pair. Leaderboard always sorts
# descending, so ranking these through it puts the worst candidates on top —
# and that ordering is what gets written to the shortlist phase 2 consumes.
ASCENDING_COLUMNS = frozenset({
    "req_accuracy_taker", "req_accuracy_maker",
    "spread_bps", "cost_taker_bps", "cost_maker_bps",
})


class MetricRanking:
    def __init__(self, frame: pd.DataFrame, metric: str, top: Optional[int] = None) -> None:
        self._frame  = frame
        self._metric = metric
        self._top    = top

    def rows(self) -> pd.DataFrame:
        if self._frame.empty or self._metric not in self._frame.columns:
            return self._frame
        if self._metric in ASCENDING_COLUMNS:
            ranked = self._frame.sort_values(self._metric, ascending=True)
            return ranked if self._top is None else ranked.head(self._top)
        # Reuses the existing Leaderboard for the ordinary case, which is what
        # every other ranked view in this repo already goes through.
        return Leaderboard(self._frame, self._metric, self._top or len(self._frame)).rows()

# Below this many candles the median and the hit rate are noise, not statistics.
MIN_CANDLES = 30

DEFAULT_QUOTE = {"coinbase": "USDC", "binance": "USDT"}

# Published base-tier rates, per side, used ONLY when the venue's own fee
# endpoint cannot be reached. Every run says which of the two it used, so a
# fallback never quietly passes itself off as this account's measured rate.
FALLBACK_FEE_BPS = {
    "coinbase": (60.0, 120.0),
    "binance":  (10.0, 10.0),
}


@dataclass(frozen=True)
class ScreenConfig:
    exchange:                str
    quote:                   str
    days:                    int
    granularity:             str
    threshold:               float
    target:                  float
    min_volume:              float
    max_pairs:               int
    metric:                  str
    top:                     int
    require_short:           bool
    use_cache:               bool
    maker_bps:               Optional[float]
    taker_bps:               Optional[float]
    max_concurrent_requests: int
    out_dir:                 str
    cache_dir:               str


# ── Window ─────────────────────────────────────────────────────────────

# Snapped to whole UTC days, and this is load-bearing rather than tidy.
#
# CandleCacheKey.filename() interpolates the raw start/end ints, so handing it
# int(time.time()) mints a unique key every second — a guaranteed cache miss
# and a full cold refetch of every pair on every invocation. Snapping means the
# first run of a day fetches and every rerun that day is a pure disk read.
#
# Snapping alone does NOT make the window closed-only: Binance treats klines'
# endTime as inclusive of a candle's open, so a window ending at midnight still
# returns the candle that opened at midnight. PairCandles._closed drops it.
class UtcDayWindow:
    _DAY = 86400

    def __init__(self, now: float, days: int) -> None:
        self._now  = now
        self._days = days

    def end(self) -> int:
        return int(self._now) // self._DAY * self._DAY

    def start(self) -> int:
        return self.end() - self._days * self._DAY

    def days(self) -> int:
        return self._days


# ── Universe ───────────────────────────────────────────────────────────

class TradablePairs:
    def __init__(
        self,
        products: list[dict],
        quote: str,
        min_volume: float,
        require_short: bool,
        max_pairs: int,
    ) -> None:
        self._products      = products
        self._quote         = quote
        self._min_volume    = min_volume
        self._require_short = require_short
        self._max_pairs     = max_pairs

    # Ranked by liquidity before the cap, so trimming to max_pairs drops the
    # thinnest names rather than whichever the venue happened to list last.
    @functools.cached_property
    def selected(self) -> list[dict]:
        kept = [product for product in self._products if self._keeps(product)]
        kept.sort(key=lambda product: product.get("volume_24h_quote", 0.0), reverse=True)
        return kept[: self._max_pairs]

    def pairs(self) -> tuple[str, ...]:
        return tuple(product["product_id"] for product in self.selected)

    def _keeps(self, product: dict) -> bool:
        if product.get("quote_currency", "").upper() != self._quote.upper():
            return False
        if not product.get("tradable", False):
            return False
        # A pair can be listed and margin-enabled with its BUY leg halted.
        # Every strategy here opens with a buy or a borrow-and-sell, so one
        # whose long side is closed cannot be traded long at all.
        if not product.get("can_long", True):
            return False
        if product.get("volume_24h_quote", 0.0) < self._min_volume:
            return False
        if self._require_short and not product.get("can_short", False):
            return False
        return True


# ── Fees and cost ──────────────────────────────────────────────────────

class VenueFees:
    def __init__(
        self,
        exchange: str,
        maker_bps: float,
        taker_bps: float,
        measured: bool,
        overrides: tuple[str, ...] = (),
    ) -> None:
        self._exchange  = exchange
        self._maker_bps = maker_bps
        self._taker_bps = taker_bps
        self._measured  = measured
        self._overrides = overrides

    def maker_bps(self) -> float:
        return self._maker_bps

    def taker_bps(self) -> float:
        return self._taker_bps

    # True only when BOTH printed numbers came from the venue untouched, so a
    # reader of the persisted shortlist can trust the pair of them together.
    def measured(self) -> bool:
        return self._measured and not self._overrides

    # Names which rates were overridden rather than lumping them in with a
    # failed lookup — a number the user typed and a number nobody measured are
    # different kinds of unmeasured.
    def describe(self) -> str:
        source = "measured from the venue" if self._measured else "PUBLISHED FALLBACK — not this account's"
        if self._overrides:
            source = f"{source}; {' and '.join(self._overrides)} overridden on the command line"
        return (
            f"{self._exchange}: maker {self._maker_bps:.2f} bps/side, "
            f"taker {self._taker_bps:.2f} bps/side ({source})"
        )


class AccountFees:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        exchange: str,
        maker_override: Optional[float],
        taker_override: Optional[float],
    ) -> None:
        self._adapter        = adapter
        self._exchange       = exchange
        self._maker_override = maker_override
        self._taker_override = taker_override
        self._fees: Optional[VenueFees] = None

    async def run(self) -> None:
        maker, taker = FALLBACK_FEE_BPS.get(self._exchange, (0.0, 0.0))
        measured     = False
        try:
            maker, taker = await self._adapter.fee_rates()
            measured     = True
        except Exception:
            # A read-only key cannot see the fee endpoint on either venue. That
            # is not worth failing a scan over, but it must be visible in the
            # output rather than silently becoming a guess.
            logger.warning(
                "could not read %s fee rates — falling back to published base tier",
                self._exchange, exc_info=True,
            )
        overrides = tuple(
            name for name, value in (("maker", self._maker_override), ("taker", self._taker_override))
            if value is not None
        )
        self._fees = VenueFees(
            self._exchange,
            self._maker_override if self._maker_override is not None else maker,
            self._taker_override if self._taker_override is not None else taker,
            measured,
            overrides,
        )

    def fees(self) -> VenueFees:
        if self._fees is None:
            raise ValueError("fee rates have not been read yet")
        return self._fees


# A market round trip pays the fee on both legs and crosses the half-spread
# twice, i.e. one whole spread. A maker round trip pays its fee twice but
# EARNS that spread instead of paying it — resting at the bid to buy and at
# the ask to sell.
#
# The maker figure therefore assumes the order filled, which is exactly what a
# resting order does not guarantee. Read it as the best case, and read the
# adverse-selection warning in the report banner next to it.
class RoundTripCost:
    def __init__(self, fees: VenueFees, spread_bps: float, maker: bool) -> None:
        self._fees       = fees
        self._spread_bps = spread_bps
        self._maker      = maker

    def bps(self) -> float:
        if self._maker:
            return 2.0 * self._fees.maker_bps() - self._spread_bps
        return 2.0 * self._fees.taker_bps() + self._spread_bps


# How often a strategy must call the direction correctly to clear `target`,
# given a typical move of `move` against a cost floor of `cost`:
#     (2p - 1) * move - cost = target   =>   p = (target + cost + move) / (2 * move)
# A value above 1.0 means the target is arithmetically out of reach at that
# cost, however good the strategy is.
class RequiredAccuracy:
    def __init__(self, move: float, cost: float, target: float) -> None:
        self._move   = move
        self._cost   = cost
        self._target = target

    def value(self) -> float:
        if self._move <= 0.0:
            return float("inf")
        return (self._target + self._cost + self._move) / (2.0 * self._move)


# ── Statistics ─────────────────────────────────────────────────────────

class DailyMoves:
    def __init__(self, frame: pd.DataFrame, atr_period: int = ATR_PERIOD) -> None:
        self._frame      = frame
        self._atr_period = atr_period

    # Close-to-close, not close-minus-open: the GA decides on a closed candle
    # and both enters and exits at a close, so this is the return its trades
    # are actually made of.
    @functools.cached_property
    def close_change(self) -> pd.Series:
        return Delta(self._frame["close"], 1).series.dropna()

    @functools.cached_property
    def range_fraction(self) -> pd.Series:
        return CandleRange(
            self._frame["high"], self._frame["low"], self._frame["open"],
        ).fraction.dropna()

    @functools.cached_property
    def atr_fraction(self) -> pd.Series:
        true_range = TrueRange(
            self._frame["high"], self._frame["low"], self._frame["close"],
        ).series
        return AverageTrueRange(
            true_range, self._frame["close"], self._atr_period,
        ).percent.dropna()

    def up_day_rate(self) -> float:
        changes = self.close_change
        if changes.empty:
            return 0.0
        return float((changes > 0).mean())

    def latest_atr(self) -> float:
        atr = self.atr_fraction
        if atr.empty:
            return 0.0
        return float(atr.iloc[-1])


class ThresholdHitRate:
    def __init__(self, series: pd.Series, threshold: float) -> None:
        self._series    = series
        self._threshold = threshold

    # Absolute, so a 3% fall counts as a move exactly like a 3% rise — the
    # question is whether there was something to trade, not which way it went.
    def fraction(self) -> float:
        if self._series.empty:
            return 0.0
        return float((self._series.abs() >= self._threshold).mean())


# ── Spreads ────────────────────────────────────────────────────────────

# Binance's ticker/24hr already carries bid and ask, so list_products() hands
# those spreads over for free and nothing more is fetched. Coinbase's products
# endpoint does not, so those are filled in from best_bid_ask — which IS
# batched server-side there, one request for many products.
class VenueSpreads:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        pairs: tuple[str, ...],
        supplied: dict[str, float],
        batch: int = 50,
    ) -> None:
        self._adapter  = adapter
        self._pairs    = pairs
        self._supplied = supplied
        self._batch    = batch
        self._bps: Optional[dict[str, float]] = None

    async def run(self) -> None:
        resolved = {pair: self._supplied[pair] for pair in self._pairs if pair in self._supplied}
        missing  = [pair for pair in self._pairs if pair not in resolved]
        for index in range(0, len(missing), self._batch):
            chunk = missing[index : index + self._batch]
            try:
                book = await self._adapter.get_best_bid_ask(*chunk)
            except Exception:
                # A spread is an adjustment, not the measurement. Losing one
                # batch must not cost the whole scan its volatility numbers.
                logger.warning("could not read spreads for %s", chunk, exc_info=True)
                continue
            resolved.update(self._parsed(book))
        self._bps = resolved

    def bps(self) -> dict[str, float]:
        if self._bps is None:
            raise ValueError("spreads have not been read yet")
        return self._bps

    @staticmethod
    def _parsed(book: dict) -> dict[str, float]:
        spreads: dict[str, float] = {}
        for entry in book.get("pricebooks", []):
            bids = entry.get("bids") or [{}]
            asks = entry.get("asks") or [{}]
            bid  = float(bids[0].get("price", 0.0) or 0.0)
            ask  = float(asks[0].get("price", 0.0) or 0.0)
            if bid > 0.0 and ask > 0.0:
                spreads[entry["product_id"]] = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
        return spreads


# ── Fetch ──────────────────────────────────────────────────────────────

class PairCandles:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        pair: str,
        granularity: str,
        window: UtcDayWindow,
        cache_dir: str,
        limit: asyncio.Semaphore,
        use_cache: bool = True,
    ) -> None:
        self._adapter     = adapter
        self._pair        = pair
        self._granularity = granularity
        self._window      = window
        self._cache_dir   = cache_dir
        self._limit       = limit
        self._use_cache   = use_cache

    async def raw(self) -> list[dict]:
        # The shared semaphore is mandatory, not stylistic: HistoricalCandles
        # builds a private Semaphore(8) when none is passed, so N pairs would
        # put N x 8 requests in flight — at 200 pairs that is 1600, every one
        # of which times out. This is the failure that class exists to prevent.
        candles = HistoricalCandles(
            self._adapter, self._pair, self._granularity,
            self._window.start(), self._window.end(), limit=self._limit,
        )
        if not self._use_cache:
            return self._closed(await candles.raw())
        return self._closed(await CachedHistoricalCandles(
            candles,
            CandleCacheFile(
                self._cache_dir,
                CandleCacheKey(
                    self._adapter.name(), self._pair, self._granularity,
                    self._window.start(), self._window.end(),
                ),
            ),
        ).raw())

    # Binance treats klines' endTime as INCLUSIVE of a candle's open, so a
    # window ending at today's midnight comes back carrying the candle that
    # opened at midnight — the one still forming. Its close is just the current
    # price and its high/low are partial, and Wilder smoothing gives that newest
    # bar ~7% of the ATR, so leaving it in moved atr_pct by ~11% on real BTC data.
    #
    # Filtered on the way out rather than before the cache write, so a file
    # already holding a partial candle is corrected on read instead of staying
    # poisoned for the rest of the UTC day.
    def _closed(self, candles: list[dict]) -> list[dict]:
        return [candle for candle in candles if int(candle["start"]) < self._window.end()]


# ── Per-pair scan ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairMetrics:
    pair:                 str
    exchange:             str
    candles:              int
    move_hit_rate:        float
    range_hit_rate:       float
    median_move:          float
    mean_move:            float
    median_range:         float
    atr_pct:              float
    up_day_rate:          float
    volume_24h_quote:     float
    can_short:            bool
    error:                Optional[str] = None


class PairScan:
    def __init__(
        self,
        candles: PairCandles,
        pair: str,
        exchange: str,
        threshold: float,
        product: dict,
    ) -> None:
        self._candles   = candles
        self._pair      = pair
        self._exchange  = exchange
        self._threshold = threshold
        self._product   = product
        self._metrics: Optional[PairMetrics] = None

    async def run(self) -> None:
        raw = await self._candles.raw()
        if len(raw) < MIN_CANDLES:
            raise ValueError(f"only {len(raw)} candles — need at least {MIN_CANDLES}")
        moves = DailyMoves(OhlcFrame(raw).dataframe)
        self._metrics = PairMetrics(
            pair             = self._pair,
            exchange         = self._exchange,
            candles          = len(raw),
            move_hit_rate    = ThresholdHitRate(moves.close_change, self._threshold).fraction(),
            range_hit_rate   = ThresholdHitRate(moves.range_fraction, self._threshold).fraction(),
            median_move      = float(moves.close_change.abs().median()),
            mean_move        = float(moves.close_change.abs().mean()),
            median_range     = float(moves.range_fraction.median()),
            atr_pct          = moves.latest_atr(),
            up_day_rate      = moves.up_day_rate(),
            volume_24h_quote = self._product.get("volume_24h_quote", 0.0),
            can_short        = self._product.get("can_short", False),
        )

    def metrics(self) -> PairMetrics:
        if self._metrics is None:
            raise ValueError(f"scan for {self._pair} has not run")
        return self._metrics


# Failure as a value, not an exception, following IsolatedAlgo in paper_engine.
# A delisted, halted or barely-quoted pair must not take a 200-pair scan down
# with it. This is the one place the codebase's "raise, don't swallow" rule is
# deliberately set aside, and it matches the house precedent.
class IsolatedPairScan:
    _EMPTY = PairMetrics(
        pair="", exchange="", candles=0, move_hit_rate=0.0, range_hit_rate=0.0,
        median_move=0.0, mean_move=0.0, median_range=0.0, atr_pct=0.0,
        up_day_rate=0.0, volume_24h_quote=0.0, can_short=False,
    )

    def __init__(self, scan: PairScan, pair: str, exchange: str) -> None:
        self._scan     = scan
        self._pair     = pair
        self._exchange = exchange
        self._error: Optional[str] = None

    async def run(self) -> None:
        try:
            await self._scan.run()
        except Exception as exc:
            logger.warning("scan failed for %s: %s", self._pair, exc)
            self._error = f"{type(exc).__name__}: {exc}"

    def metrics(self) -> PairMetrics:
        if self._error is None:
            return self._scan.metrics()
        return replace(self._EMPTY, pair=self._pair, exchange=self._exchange, error=self._error)


# ── Frame, shortlist, report ───────────────────────────────────────────

class ScreenFrame:
    def __init__(
        self,
        metrics: tuple[PairMetrics, ...],
        spreads: dict[str, float],
        fees: VenueFees,
        target: float,
    ) -> None:
        self._metrics = metrics
        self._spreads = spreads
        self._fees    = fees
        self._target  = target

    @functools.cached_property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self._row(metric) for metric in self._metrics])

    def _row(self, metric: PairMetrics) -> dict[str, Any]:
        # NaN, not 0.0, when the spread could not be read — a failed batch or an
        # unquoted book. Zero would understate the cost floor and could flip a
        # pair to viable, which is the wrong direction for a cost model to fail
        # in. NaN propagates into every net figure and leaves `viable` False.
        spread = self._spreads.get(metric.pair, float("nan"))
        taker  = RoundTripCost(self._fees, spread, maker=False).bps()
        maker  = RoundTripCost(self._fees, spread, maker=True).bps()
        move     = metric.median_move
        accuracy = RequiredAccuracy(move, taker / 10_000.0, self._target).value()
        return {
            "pair":                metric.pair,
            "volume_24h_quote":    round(metric.volume_24h_quote, 0),
            "candles":             metric.candles,
            "move_hit_rate":       round(metric.move_hit_rate, 4),
            "range_hit_rate":      round(metric.range_hit_rate, 4),
            "median_move_pct":     round(move * 100.0, 4),
            "mean_move_pct":       round(metric.mean_move * 100.0, 4),
            "median_range_pct":    round(metric.median_range * 100.0, 4),
            "atr_pct":             round(metric.atr_pct * 100.0, 4),
            "up_day_rate":         round(metric.up_day_rate, 4),
            "spread_bps":          round(spread, 2),
            "cost_taker_bps":      round(taker, 2),
            "cost_maker_bps":      round(maker, 2),
            "net_taker_bps":       round(move * 10_000.0 - taker, 2),
            "net_maker_bps":       round(move * 10_000.0 - maker, 2),
            "req_accuracy_taker":  round(accuracy, 4),
            "req_accuracy_maker":  round(RequiredAccuracy(move, maker / 10_000.0, self._target).value(), 4),
            "can_short":           metric.can_short,
            # Two conditions, both on the TAKER floor — a maker fill is never
            # guaranteed, so calling a pair viable on the strength of one would
            # be optimism rather than measurement.
            #
            # The second condition is the one that matters for the question
            # being asked. Clearing the cost floor only means the pair moves
            # more than it costs to trade; it says nothing about reaching the
            # target. A pair needing better than 100% direction accuracy cannot
            # hit it however good the strategy is, and without this term BTC,
            # LTC, BNB, PAXG, XAUT and TRX all reached the phase-2 shortlist
            # while being arithmetically incapable of 1%/day.
            "viable":              bool(move * 10_000.0 - taker > 0.0 and accuracy <= 1.0),
            "error":               metric.error or "",
        }


# Every viable pair, ranked — not a --top slice of them. --top caps what the
# console prints; capping the persisted shortlist too would silently hand phase
# 2 a subset while the banner reported the full count as "viable".
class Shortlist:
    def __init__(self, frame: pd.DataFrame, metric: str) -> None:
        self._frame  = frame
        self._metric = metric

    def rows(self) -> pd.DataFrame:
        if self._frame.empty:
            return self._frame
        # Guarded rather than left to raise: the shortlist is built after the
        # whole network scan, so an unknown --metric would otherwise throw away
        # every fetched candle and print nothing at all.
        if self._metric not in self._frame.columns:
            raise ValueError(
                f"unknown --metric {self._metric!r}; available: "
                f"{', '.join(sorted(self._frame.columns))}"
            )
        healthy = self._frame[(self._frame["error"] == "") & self._frame["viable"]]
        if healthy.empty:
            return healthy
        return MetricRanking(healthy, self._metric).rows()


class ShortlistPayload:
    def __init__(
        self,
        rows: pd.DataFrame,
        config: ScreenConfig,
        window: UtcDayWindow,
        fees: VenueFees,
    ) -> None:
        self._rows   = rows
        self._config = config
        self._window = window
        self._fees   = fees

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "exchange":     self._config.exchange,
            "quote":        self._config.quote,
            "window": {
                "start":       self._window.start(),
                "end":         self._window.end(),
                "days":        self._window.days(),
                "granularity": self._config.granularity,
            },
            "assumptions": {
                "target_pct":       self._config.target * 100.0,
                "threshold_pct":    self._config.threshold * 100.0,
                "maker_bps":        self._fees.maker_bps(),
                "taker_bps":        self._fees.taker_bps(),
                "fees_measured":    self._fees.measured(),
            },
            "metric": self._config.metric,
            # A flat list, deliberately: this is literally the `values:` block
            # for a sweep axis, so phase 2 is a copy-paste and not a parse.
            "pairs":  list(self._rows["pair"]) if not self._rows.empty else [],
            "rows":   self._rows.to_dict("records") if not self._rows.empty else [],
        }


class ShortlistFile:
    def __init__(self, directory: str, filename: str) -> None:
        self._directory = directory
        self._filename  = filename

    def write(self, payload: dict[str, Any]) -> None:
        os.makedirs(self._directory, exist_ok=True)
        tmp_path = f"{self._path()}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self._path())  # atomic — never a truncated shortlist

    def read(self) -> dict[str, Any]:
        with open(self._path(), encoding="utf-8") as handle:
            return json.load(handle)

    def _path(self) -> str:
        return os.path.join(self._directory, self._filename)


class ScreenerReport:
    def __init__(
        self,
        frame: pd.DataFrame,
        shortlist: pd.DataFrame,
        fees: VenueFees,
        config: ScreenConfig,
        window: UtcDayWindow,
    ) -> None:
        self._frame     = frame
        self._shortlist = shortlist
        self._fees      = fees
        self._config    = config
        self._window    = window

    def print(self) -> None:
        rule = "─" * 78
        print(f"\n{rule}")
        print(
            f"{self._config.exchange} / {self._config.quote}  "
            f"{self._window.days()}d of {self._config.granularity}  "
            f"[{self._stamp(self._window.start())} → {self._stamp(self._window.end())}]"
        )
        print(f"fees      {self._fees.describe()}")
        print(
            f"target    {self._config.target * 100:.2f}% per candle    "
            f"threshold {self._config.threshold * 100:.2f}%"
        )
        print(rule)

        shown = min(self._config.top, len(self._frame))
        print(
            f"\nAll pairs ({len(self._frame)} scanned, showing {shown}), "
            f"ranked by {self._config.metric}:"
        )
        ConsoleTable(self._ranked()).print()

        print(
            f"\nShortlist ({len(self._shortlist)} viable, net of the taker cost floor"
            + (f"; showing {self._config.top}" if len(self._shortlist) > self._config.top else "")
            + "):"
        )
        ConsoleTable(self._shortlist.head(self._config.top)).print()

        self._print_caveats()

    def _ranked(self) -> pd.DataFrame:
        return MetricRanking(self._frame, self._config.metric, self._config.top).rows()

    # Printed every run, deliberately. These are the reasons a good-looking row
    # here is not yet a good-looking trade, and they do not stop being true
    # because the table looks encouraging.
    def _print_caveats(self) -> None:
        print(f"\n{'─' * 78}")
        print("Read before acting on this table:")
        print("  · Movement is not predictability. These pairs MOVED; nothing here says")
        print("    the direction was forecastable. req_accuracy is the bar a strategy")
        print("    must clear — compare it against ~55% for a realistic daily model.")
        print("  · median_range_pct is the perfect-timing bound (buy the low, sell the")
        print("    high). It is not capturable. median_move_pct is the honest number.")
        print("  · net_maker_bps assumes the resting order FILLED. A maker order fills")
        print("    preferentially when the market is going against it (adverse")
        print("    selection), and may not fill at all. Treat it as a best case.")
        print("    It also RISES with the spread, so it flatters illiquid pairs — which")
        print("    is why the shortlist is ranked on net_taker_bps by default.")
        print("  · A blank spread_bps means the book could not be read. Those rows are")
        print("    never counted viable rather than being costed as if free.")
        print("  · Borrow interest is not modelled anywhere in this repo; Binance")
        print("    charges it hourly on shorts. Nor is slippage past the top of book.")
        print(f"{'─' * 78}")

    @staticmethod
    def _stamp(seconds: int) -> str:
        return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d")


# ── Orchestration ──────────────────────────────────────────────────────

class PairScreener:
    def __init__(self, config: ScreenConfig, now: Optional[float] = None) -> None:
        self._config = config
        self._now    = now
        self._frame: Optional[pd.DataFrame] = None
        self._fees:  Optional[VenueFees]    = None

    async def run(self) -> None:
        window = self.window()
        async with ExchangePool(
            (self._config.exchange,), self._config.max_concurrent_requests,
        ) as pool:
            lane    = pool.lane(self._config.exchange)
            adapter = lane.adapter()

            universe = TradablePairs(
                await adapter.list_products(), self._config.quote,
                self._config.min_volume, self._config.require_short,
                self._config.max_pairs,
            )
            pairs = universe.pairs()
            if not pairs:
                # Named separately, because --require-short on Coinbase empties
                # the universe every time — there is no borrow there at all —
                # and blaming the volume floor for that sends the reader off
                # lowering a number that was never the problem.
                if self._config.require_short and self._config.exchange == "coinbase":
                    raise ValueError(
                        "--require-short excludes every Coinbase pair: Coinbase Advanced "
                        "Trade has no short side. Use --exchange binance for shorts."
                    )
                raise ValueError(
                    f"no {self._config.quote} pairs on {self._config.exchange} cleared "
                    f"a {self._config.min_volume:,.0f} 24h volume floor"
                    + (" with shorts enabled" if self._config.require_short else "")
                )
            logger.info("scanning %d pairs on %s", len(pairs), self._config.exchange)

            account_fees = AccountFees(
                adapter, self._config.exchange,
                self._config.maker_bps, self._config.taker_bps,
            )
            await account_fees.run()
            self._fees = account_fees.fees()

            spreads = VenueSpreads(
                adapter, pairs,
                {
                    product["product_id"]: product["spread_bps"]
                    for product in universe.selected if "spread_bps" in product
                },
            )
            await spreads.run()

            scans = [
                IsolatedPairScan(
                    PairScan(
                        PairCandles(
                            adapter, product["product_id"], self._config.granularity,
                            window, self._config.cache_dir, lane.limit(),
                            self._config.use_cache,
                        ),
                        product["product_id"], self._config.exchange,
                        self._config.threshold, product,
                    ),
                    product["product_id"], self._config.exchange,
                )
                for product in universe.selected
            ]
            await asyncio.gather(*(scan.run() for scan in scans))

            self._frame = ScreenFrame(
                tuple(scan.metrics() for scan in scans),
                spreads.bps(), self._fees, self._config.target,
            ).dataframe

    # Cached, so the clock is read once. The report header and the persisted
    # shortlist both ask for the window again after the scan; a cold run that
    # starts at 23:55 UTC and ends after midnight would otherwise label its
    # output with a window one day later than the candles it was computed from.
    @functools.cached_property
    def _window(self) -> UtcDayWindow:
        return UtcDayWindow(self._now if self._now is not None else time.time(), self._config.days)

    def window(self) -> UtcDayWindow:
        return self._window

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            raise ValueError("screener has not run")
        return self._frame

    def fees(self) -> VenueFees:
        if self._fees is None:
            raise ValueError("screener has not run")
        return self._fees


# ── CLI ────────────────────────────────────────────────────────────────

class ScreenerArguments:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    @functools.cached_property
    def parsed(self) -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Rank a venue's pairs by how far they move, net of trading costs",
        )
        p.add_argument("--exchange", "-e", default="coinbase", choices=["coinbase", "binance"])
        p.add_argument("--quote", "-q", default=None, help="Quote currency (default: USDC coinbase / USDT binance)")
        p.add_argument("--days", "-d", type=int, default=180, help="Trailing window in days (default: 180)")
        p.add_argument("--granularity", "-g", default=DEFAULT_GRANULARITY, choices=sorted(GRANULARITY_SECONDS))
        p.add_argument("--threshold", type=float, default=1.0, help="Move %% counted as a hit (default: 1.0)")
        p.add_argument("--target", type=float, default=1.0, help="Target %% per candle for req_accuracy (default: 1.0)")
        p.add_argument("--min-volume", type=float, default=5_000_000.0, help="Minimum 24h quote volume (default: 5e6)")
        p.add_argument("--max-pairs", type=int, default=200, help="Cap on pairs scanned, highest volume first")
        p.add_argument("--metric", default=DEFAULT_METRIC, choices=RANKABLE_COLUMNS,
                       help=f"Column to rank by (default: {DEFAULT_METRIC})")
        p.add_argument("--top", type=int, default=25, help="Rows to show (default: 25)")
        p.add_argument("--require-short", action="store_true", help="Only pairs whose short side is enabled")
        p.add_argument("--no-cache", action="store_true", help="Bypass the candle disk cache")
        p.add_argument("--maker-bps", type=float, default=None, help="Override the maker rate, per side")
        p.add_argument("--taker-bps", type=float, default=None, help="Override the taker rate, per side")
        p.add_argument("--max-concurrent", type=int, default=8, help="Shared in-flight request budget (default: 8)")
        p.add_argument("--out", default=None, help="Directory for the shortlist JSON")
        p.add_argument("--cache-dir", default=None, help="Candle cache directory")
        return p.parse_args(self._argv)

    def config(self) -> ScreenConfig:
        args = self.parsed
        return ScreenConfig(
            exchange                = args.exchange,
            quote                   = args.quote or DEFAULT_QUOTE[args.exchange],
            days                    = args.days,
            granularity             = args.granularity,
            # Percentages on the command line, fractions everywhere inside.
            threshold               = args.threshold / 100.0,
            target                  = args.target / 100.0,
            min_volume              = args.min_volume,
            max_pairs               = args.max_pairs,
            metric                  = args.metric,
            top                     = args.top,
            require_short           = args.require_short,
            use_cache               = not args.no_cache,
            maker_bps               = args.maker_bps,
            taker_bps               = args.taker_bps,
            max_concurrent_requests = args.max_concurrent,
            out_dir                 = args.out or str(GA_RESULTS_ROOT / "screener"),
            # Kept apart from the GA's candle_cache: a new UTC day mints a new
            # key for every pair, so this directory churns and would otherwise
            # bury the training cache's handful of long-lived windows.
            cache_dir               = args.cache_dir or str(GA_RESULTS_ROOT / "screener_cache"),
        )


async def _main(argv: list[str]) -> None:
    # The report rules its sections with box-drawing characters, and a Windows
    # console (or any redirected pipe) defaults to cp1252, which cannot encode
    # them — the scan would finish and then die printing its own results.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config   = ScreenerArguments(argv).config()
    screener = PairScreener(config)
    await screener.run()

    frame     = screener.frame()
    shortlist = Shortlist(frame, config.metric).rows()
    ScreenerReport(frame, shortlist, screener.fees(), config, screener.window()).print()

    payload = ShortlistPayload(shortlist, config, screener.window(), screener.fees()).as_dict()
    ShortlistFile(config.out_dir, f"shortlist_{config.exchange}_latest.json").write(payload)
    print(f"\nshortlist → {os.path.join(config.out_dir, f'shortlist_{config.exchange}_latest.json')}")


# Reads sys.argv, unlike market_scanner's __main__, which hardcodes its argv and
# silently ignores everything the user typed.
if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
