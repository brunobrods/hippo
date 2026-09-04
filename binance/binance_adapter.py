"""
Binance Isolated Margin Execution Adapter
------------------------------------------
API:  https://api.binance.com
Docs: https://developers.binance.com/docs/margin_trading

Auth: HMAC-SHA256 over the exact encoded query string, sent as `signature`.
      The API key travels in the `X-MBX-APIKEY` header. Keys are created at
      https://www.binance.com/en/my/settings/api-management — the key needs
      "Enable Margin" and "Enable Spot & Margin Trading".

Key format:
    api_key    = "64-char alphanumeric"
    api_secret = "64-char alphanumeric"   (a shared secret, not a PEM key)

Canonical formats (see exchange.adapter) — this adapter translates them:

    "BTC-USDC"  ->  symbol "BTCUSDC"          via BinanceSymbol
    "SIX_HOUR"  ->  interval "6h"             via GRANULARITY_INTERVALS
    seconds     ->  milliseconds              on every time parameter
    kline array ->  Coinbase-shaped candle    via KlineRow

Order model (isolated margin):
    Every order carries isIsolated=TRUE and a sideEffectType that decides
    whether Binance borrows or repays on your behalf:

        NO_SIDE_EFFECT  spend funds already in the isolated wallet
        MARGIN_BUY      auto-borrow whatever the order needs
        AUTO_REPAY      repay the outstanding borrow with the proceeds

    Longs are NO_SIDE_EFFECT, which makes them economically identical to a
    spot buy: nothing is borrowed, no interest accrues, and there is no
    liquidation price. Shorts must borrow the base asset to sell it, so
    market_short/limit_short use MARGIN_BUY and the matching cover uses
    AUTO_REPAY. That asymmetry is what trading_strategy.IsolatedMargin
    models — longs borrow nothing so they cannot be liquidated, while a short's
    liquidation price falls out of its collateral ratio.

    NOTE: Binance charges hourly interest on borrowed funds. Ledger/Backtest
    do not model it, so a short held for days scores better in the GA than it
    will trade live.

Divergences from CoinbaseAdapter's surface, forced by the API:
    cancel_orders   needs a product_id — Binance cancels per symbol, and the
                    batch is not atomic (per-order results, like Coinbase's).
    replace_order   stands in for edit_order — Binance has no in-place amend,
                    only cancelReplace, which needs the side and order type.
    get_fills       needs a product_id — myTrades is symbol-scoped, so there
                    is no account-wide fills query.

NOTE: Binance's spot testnet does not serve margin endpoints, so isolated
      margin is live-only. Use very small amounts when testing.

Usage:
    async with BinanceAdapter(api_key="...", api_secret="...") as adapter:
        ticker = await adapter.get_best_bid_ask("BTC-USDC")
        bid    = float(ticker["pricebooks"][0]["bids"][0]["price"])
        order  = await adapter.limit_buy(
            product_id  = "BTC-USDC",
            base_size   = "0.001",
            limit_price = str(round(bid * 0.98, 2)),
        )
        await adapter.cancel_orders("BTC-USDC", [order["orderId"]])
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import sys
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from exchange.adapter import ExchangeError

logger = logging.getLogger(__name__)

BASE_URL     = "https://api.binance.com"
RECV_WINDOW  = 5000   # ms a signed request stays valid after its timestamp
MAX_CANDLES_PER_REQUEST = 1000

# Canonical granularity names (Coinbase's, shared across this codebase) mapped
# to Binance kline intervals.
GRANULARITY_INTERVALS = {
    "ONE_MINUTE":     "1m",
    "FIVE_MINUTE":    "5m",
    "FIFTEEN_MINUTE": "15m",
    "THIRTY_MINUTE":  "30m",
    "ONE_HOUR":       "1h",
    "TWO_HOUR":       "2h",
    "SIX_HOUR":       "6h",
    "ONE_DAY":        "1d",
}

NO_SIDE_EFFECT = "NO_SIDE_EFFECT"
MARGIN_BUY     = "MARGIN_BUY"
AUTO_REPAY     = "AUTO_REPAY"


# ── Symbols ────────────────────────────────────────────────────────────
# Binance has no separator in its symbols, so "BTCUSDC" cannot be split back
# into base and quote without the exchange's own asset list. This codebase
# therefore keeps the dash form canonical and converts on the way out only.

class BinanceSymbol:
    def __init__(self, product_id: str) -> None:
        self._product_id = product_id

    def wire(self) -> str:
        return self._product_id.replace("-", "").upper()

    def base(self) -> str:
        return self._split()[0]

    def quote(self) -> str:
        return self._split()[1]

    def _split(self) -> tuple[str, str]:
        if "-" not in self._product_id:
            raise ValueError(
                f"{self._product_id!r} has no base/quote separator — "
                f"use the canonical 'BTC-USDC' form"
            )
        base, quote = self._product_id.split("-", 1)
        return base.upper(), quote.upper()


# ── Signing ────────────────────────────────────────────────────────────

class HmacSignature:
    def __init__(self, api_secret: str, payload: str) -> None:
        self._api_secret = api_secret
        self._payload    = payload

    def value(self) -> str:
        return hmac.new(
            self._api_secret.encode("utf-8"),
            self._payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()


class SignedQuery:
    def __init__(
        self,
        params: dict[str, Any],
        api_secret: str,
        recv_window: int = RECV_WINDOW,
    ) -> None:
        self._params      = params
        self._api_secret  = api_secret
        self._recv_window = recv_window

    # The signature covers the exact string sent, so the query is built once
    # here and handed to aiohttp already encoded — re-encoding a params dict
    # would risk a different ordering and a 401.
    def encoded(self) -> str:
        payload = urlencode(self._stamped())
        return f"{payload}&signature={HmacSignature(self._api_secret, payload).value()}"

    def _stamped(self) -> dict[str, Any]:
        stamped = {
            key: value for key, value in self._params.items()
            if value is not None and value != ""
        }
        stamped["recvWindow"] = self._recv_window
        stamped["timestamp"]  = int(time.time() * 1000)
        return stamped


# ── Response translation ───────────────────────────────────────────────

class KlineRow:
    def __init__(self, raw: list[Any]) -> None:
        self._raw = raw

    # [openTime, open, high, low, close, volume, closeTime, ...]
    def as_candle(self) -> dict[str, str]:
        return {
            "start":  str(int(self._raw[0]) // 1000),
            "open":   str(self._raw[1]),
            "high":   str(self._raw[2]),
            "low":    str(self._raw[3]),
            "close":  str(self._raw[4]),
            "volume": str(self._raw[5]),
        }


# klines returns the OLDEST `limit` candles from startTime, so a window wider
# than MAX_CANDLES_PER_REQUEST intervals loses its newest candles with no error
# — a caller would silently read a stale price as "current". HistoricalCandles
# chunks to avoid this; anything calling the adapter directly (the scanner) can
# not, so an incomplete page is raised rather than returned.
#
# A chunk that ends exactly on a candle boundary is short by 1ms, not by a
# candle, so the tolerance is one full interval.
class TruncatedKlines:
    def __init__(self, rows: list[list[Any]], requested_end: int) -> None:
        self._rows          = rows
        self._requested_end = requested_end

    def raise_if_incomplete(self, product_id: str, granularity: str) -> None:
        if not self._missing_span():
            return
        raise BinanceError(
            0,
            {"product_id": product_id, "granularity": granularity, "returned": len(self._rows)},
            f"klines truncated at {MAX_CANDLES_PER_REQUEST} candles — the window "
            f"asked for more than one page. Request a narrower window, or fetch "
            f"through HistoricalCandles, which chunks by max_candles_per_request().",
        )

    def _missing_span(self) -> bool:
        if len(self._rows) < MAX_CANDLES_PER_REQUEST:
            return False
        last          = self._rows[-1]
        interval_ms   = int(last[6]) - int(last[0]) + 1
        return (self._requested_end * 1000 - int(last[6])) > interval_ms


class IsolatedAccounts:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    # One entry per asset per pair, in the shape AccountBalance expects, with
    # a product_id so a balance can be scoped to its own isolated wallet —
    # the same asset appears once per pair it is held in.
    #
    # CAVEAT: available_balance maps to Binance's `free`, which is spendable
    # cash, NOT equity — with a short open it holds the sale proceeds while the
    # offsetting debt sits in the OTHER asset's `borrowed`. Sizing a new
    # position off it therefore overstates the account. `netAsset` does not fix
    # this either (it nets only same-asset debt); true equity is
    # quote.netAsset + base.netAsset * price, which needs a mark price and so
    # belongs in a separate object once order execution is actually wired.
    def flattened(self) -> list[dict[str, Any]]:
        accounts: list[dict[str, Any]] = []
        for pair in self._raw.get("assets", []):
            product_id = f"{pair['baseAsset']['asset']}-{pair['quoteAsset']['asset']}"
            accounts.append(self._account(product_id, pair["baseAsset"]))
            accounts.append(self._account(product_id, pair["quoteAsset"]))
        return accounts

    @staticmethod
    def _account(product_id: str, asset: dict[str, Any]) -> dict[str, Any]:
        currency = asset["asset"]
        return {
            "currency":          currency,
            "product_id":        product_id,
            "available_balance": {"value": asset["free"],   "currency": currency},
            "hold":              {"value": asset["locked"], "currency": currency},
            "borrowed":          asset["borrowed"],
            "interest":          asset["interest"],
            "net_asset":         asset["netAsset"],
        }


# Binance's own risk numbers for one isolated pair — the real liquidation
# price. IsolatedMargin computes the same figure from the ledger's own
# collateral; this is the exchange's authoritative version.
class IsolatedRisk:
    def __init__(self, pair: dict[str, Any]) -> None:
        self._pair = pair

    def margin_level(self) -> float:
        return float(self._pair["marginLevel"])

    def liquidation_price(self) -> float:
        return float(self._pair["liquidatePrice"])


class BinanceProduct:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    # exchangeInfo filters, renamed to the increment fields CoinbaseAdapter's
    # callers already read off get_product().
    def normalized(self) -> dict[str, Any]:
        filters = {item["filterType"]: item for item in self._raw.get("filters", [])}
        lot     = filters.get("LOT_SIZE", {})
        return {
            "product_id":       f"{self._raw['baseAsset']}-{self._raw['quoteAsset']}",
            "base_currency":    self._raw["baseAsset"],
            "quote_currency":   self._raw["quoteAsset"],
            "quote_increment":  filters.get("PRICE_FILTER", {}).get("tickSize", "0.01"),
            "base_increment":   lot.get("stepSize", "0.00000001"),
            "base_min_size":    lot.get("minQty", "0.00000001"),
            "base_max_size":    lot.get("maxQty", "0"),
            "min_market_funds": self._min_notional(filters),
            "status":           self._raw.get("status", ""),
            "tradable":         self._raw.get("status", "") == "TRADING",
            "raw":              self._raw,
        }

    @staticmethod
    def _min_notional(filters: dict[str, Any]) -> str:
        # Binance renamed MIN_NOTIONAL to NOTIONAL; both still appear.
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        return notional.get("minNotional", "0")


# ── Universe ───────────────────────────────────────────────────────────

# One row of /sapi/v1/margin/isolated/allPairs. This endpoint is the only
# place the isolated-margin universe is enumerated, and — unlike exchangeInfo's
# symbol string — it carries base and quote as separate fields, so the
# canonical dashed id comes from the exchange's own data rather than a guess
# at where to cut "BTCUSDT". See BinanceSymbol above on why that matters.
class IsolatedPair:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def symbol(self) -> str:
        return self._raw["symbol"]

    def product_id(self) -> str:
        return f"{self._raw['base']}-{self._raw['quote']}"

    def can_long(self) -> bool:
        return bool(self._raw.get("isMarginTrade")) and bool(self._raw.get("isBuyAllowed"))

    # A short must borrow the base asset to sell it, so it needs the sell leg
    # specifically — a pair can be margin-enabled and still have shorts halted.
    def can_short(self) -> bool:
        return bool(self._raw.get("isMarginTrade")) and bool(self._raw.get("isSellAllowed"))


# /api/v3/ticker/24hr with no symbol: every pair's rolling stats in one
# unauthenticated call. Also carries bidPrice/askPrice, so a spread comes free
# from a request the universe scan already makes.
class TwentyFourHourStats:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    @functools.cached_property
    def by_symbol(self) -> dict[str, dict[str, Any]]:
        return {row["symbol"]: row for row in self._rows}

    def volume(self, symbol: str) -> float:
        return float(self.by_symbol.get(symbol, {}).get("quoteVolume", 0.0) or 0.0)

    def change_percent(self, symbol: str) -> float:
        return float(self.by_symbol.get(symbol, {}).get("priceChangePercent", 0.0) or 0.0)

    # Basis points between best bid and best ask. NaN — not zero — when either
    # side of the book is empty: an unquoted pair has no spread to measure, and
    # zero would read as a perfectly tight book and understate its cost floor.
    # NaN keeps one dead pair from failing the scan while still refusing to
    # claim a number nobody quoted.
    def spread_bps(self, symbol: str) -> float:
        row = self.by_symbol.get(symbol, {})
        bid = float(row.get("bidPrice", 0.0) or 0.0)
        ask = float(row.get("askPrice", 0.0) or 0.0)
        if bid <= 0.0 or ask <= 0.0:
            return float("nan")
        return (ask - bid) / ((ask + bid) / 2.0) * 10_000.0


class IsolatedCatalog:
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        stats: TwentyFourHourStats,
    ) -> None:
        self._pairs   = pairs
        self._symbols = symbols
        self._stats   = stats

    def products(self) -> list[dict]:
        by_wire = {entry["symbol"]: entry for entry in self._symbols}
        products: list[dict] = []
        for raw in self._pairs:
            pair  = IsolatedPair(raw)
            entry = by_wire.get(pair.symbol())
            # An isolated pair absent from exchangeInfo is mid-delisting, not a
            # bug: skip it rather than raise and lose the other ~200 rows.
            if entry is None:
                continue
            product = BinanceProduct(entry).normalized()
            product.update({
                "can_long":             pair.can_long(),
                "can_short":            pair.can_short(),
                "volume_24h_quote":     self._stats.volume(pair.symbol()),
                "price_change_24h_pct": self._stats.change_percent(pair.symbol()),
                "spread_bps":           self._stats.spread_bps(pair.symbol()),
            })
            products.append(product)
        return products


# /sapi/v1/asset/tradeFee returns this account's real rates per symbol, as
# fractions ("0.001" = 10 bps). Rates are uniform across spot symbols for most
# accounts, so the median is representative and immune to one odd row.
class TradeFeeRates:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def maker_bps(self) -> float:
        return self._median("makerCommission")

    def taker_bps(self) -> float:
        return self._median("takerCommission")

    def _median(self, key: str) -> float:
        values = sorted(
            float(row[key]) * 10_000.0 for row in self._rows if row.get(key) is not None
        )
        if not values:
            raise BinanceError(0, self._rows, f"tradeFee response carried no {key}")
        return values[len(values) // 2]


# Coinbase's order_status vocabulary mapped onto Binance's.
class OrderStatuses:
    _BY_NAME = {
        "OPEN":      ("NEW", "PARTIALLY_FILLED"),
        "FILLED":    ("FILLED",),
        "CANCELLED": ("CANCELED", "EXPIRED", "REJECTED"),
        "CANCELED":  ("CANCELED", "EXPIRED", "REJECTED"),
    }

    def __init__(self, requested: list[str]) -> None:
        self._requested = requested

    def only_open(self) -> bool:
        return [status.upper() for status in self._requested] == ["OPEN"]

    def matches(self, status: str) -> bool:
        return status in self.native()

    def native(self) -> tuple[str, ...]:
        native: tuple[str, ...] = ()
        for name in self._requested:
            native += self._BY_NAME.get(name.upper(), (name.upper(),))
        return native


# ── Order shapes ───────────────────────────────────────────────────────

class LimitOrder:
    def __init__(
        self,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        time_in_force: str = "GTC",
    ) -> None:
        self._base_size     = base_size
        self._limit_price   = limit_price
        self._post_only     = post_only
        self._time_in_force = time_in_force

    # LIMIT_MAKER is Binance's post-only: rejected outright if it would take.
    # It carries no timeInForce — sending one is an error.
    def params(self) -> dict[str, Any]:
        if self._post_only:
            return {
                "type":     "LIMIT_MAKER",
                "quantity": self._base_size,
                "price":    self._limit_price,
            }
        return {
            "type":        "LIMIT",
            "timeInForce": self._time_in_force,
            "quantity":    self._base_size,
            "price":       self._limit_price,
        }


class MarketOrder:
    def __init__(self, base_size: str = "", quote_size: str = "") -> None:
        self._base_size  = base_size
        self._quote_size = quote_size

    # Compared as numbers, not truthiness: "0" and "0.00" are strings Binance
    # would reject, so they must fail here rather than on the wire.
    def params(self) -> dict[str, Any]:
        if self._positive(self._quote_size):
            return {"type": "MARKET", "quoteOrderQty": self._quote_size}
        if not self._positive(self._base_size):
            raise ValueError("MarketOrder needs a positive base_size or quote_size")
        return {"type": "MARKET", "quantity": self._base_size}

    @staticmethod
    def _positive(size: str) -> bool:
        return bool(size) and float(size) > 0


# ── Adapter ────────────────────────────────────────────────────────────

class BinanceAdapter:
    """
    Async execution adapter for Binance isolated margin.

    Auth: HMAC-SHA256 signed query strings — no token refresh needed.
    Orders: POST to /sapi/v1/margin/order with isIsolated=TRUE.
    Market data: GET on the public /api/v3 endpoints (no auth).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        request_timeout: float = 10.0,
    ) -> None:
        self.api_key    = api_key
        self.api_secret = api_secret
        self.timeout    = aiohttp.ClientTimeout(total=request_timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        logger.info("Binance adapter connected.")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "BinanceAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ── HTTP helpers ───────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _url(self, path: str, params: dict[str, Any], auth: bool) -> str:
        if auth:
            return f"{BASE_URL}{path}?{SignedQuery(params, self.api_secret).encoded()}"
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        return f"{BASE_URL}{path}?{urlencode(clean)}" if clean else f"{BASE_URL}{path}"

    async def _get(self, path: str, params: Optional[dict] = None, auth: bool = True) -> Any:
        url = self._url(path, params or {}, auth)
        async with self._session.get(url, headers=self._headers() if auth else {}) as resp:
            return await self._parse(resp)

    async def _post(self, path: str, params: dict) -> Any:
        url = self._url(path, params, auth=True)
        async with self._session.post(url, headers=self._headers()) as resp:
            return await self._parse(resp)

    async def _delete(self, path: str, params: dict) -> Any:
        url = self._url(path, params, auth=True)
        async with self._session.delete(url, headers=self._headers()) as resp:
            return await self._parse(resp)

    @staticmethod
    async def _parse(resp: aiohttp.ClientResponse) -> Any:
        data = await resp.json()
        if not resp.ok:
            raise BinanceError(resp.status, data)
        # A 200 can still carry an error code on some SAPI endpoints.
        if isinstance(data, dict) and data.get("code", 0) not in (0, 200):
            raise BinanceError(resp.status, data)
        return data

    # ── Orders: longs (NO_SIDE_EFFECT — nothing is borrowed) ───────────

    async def limit_buy(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "BUY",
            LimitOrder(base_size, limit_price, post_only).params(),
            NO_SIDE_EFFECT, client_order_id,
        )

    async def limit_sell(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "SELL",
            LimitOrder(base_size, limit_price, post_only).params(),
            NO_SIDE_EFFECT, client_order_id,
        )

    async def market_buy(
        self,
        product_id: str,
        quote_size: str,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "BUY",
            MarketOrder(quote_size=quote_size).params(),
            NO_SIDE_EFFECT, client_order_id,
        )

    async def market_sell(
        self,
        product_id: str,
        base_size: str,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "SELL",
            MarketOrder(base_size=base_size).params(),
            NO_SIDE_EFFECT, client_order_id,
        )

    async def limit_buy_ioc(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "BUY",
            LimitOrder(base_size, limit_price, time_in_force="IOC").params(),
            NO_SIDE_EFFECT, client_order_id,
        )

    # ── Orders: shorts (MARGIN_BUY borrows, AUTO_REPAY settles) ────────
    #
    # Binance enforces a MINIMUM borrow per asset, and a short that would
    # borrow less than it is rejected with -11007 "Exceeding the maximum
    # borrowable limit" — the message names the wrong bound entirely.
    # Verified live on BTCUSDT: borrowing 0.00006 BTC was rejected while
    # maxBorrowable reported 0.00495; 0.00019 BTC went through untouched.
    # A short must therefore clear BOTH the pair's minNotional and the base
    # asset's minimum borrow, and the second is not in exchangeInfo.

    async def market_short(
        self,
        product_id: str,
        base_size: str,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "SELL",
            MarketOrder(base_size=base_size).params(),
            MARGIN_BUY, client_order_id,
        )

    async def limit_short(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "SELL",
            LimitOrder(base_size, limit_price, post_only).params(),
            MARGIN_BUY, client_order_id,
        )

    async def market_cover(
        self,
        product_id: str,
        base_size: str,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "BUY",
            MarketOrder(base_size=base_size).params(),
            AUTO_REPAY, client_order_id,
        )

    async def limit_cover(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        return await self._create_order(
            product_id, "BUY",
            LimitOrder(base_size, limit_price, post_only).params(),
            AUTO_REPAY, client_order_id,
        )

    async def _create_order(
        self,
        product_id: str,
        side: str,
        order_params: dict[str, Any],
        side_effect: str,
        client_order_id: str = "",
    ) -> dict:
        params = {
            "symbol":           BinanceSymbol(product_id).wire(),
            "isIsolated":       "TRUE",
            "side":             side,
            "sideEffectType":   side_effect,
            "newClientOrderId": client_order_id or uuid.uuid4().hex,
            "newOrderRespType": "FULL",
            **order_params,
        }
        result = await self._post("/sapi/v1/margin/order", params)
        logger.info(
            "%-4s  %-12s  %-14s  qty=%s  price=%s  id=%s",
            side,
            product_id,
            side_effect,
            order_params.get("quantity") or order_params.get("quoteOrderQty", "?"),
            order_params.get("price", "MARKET"),
            result.get("orderId", ""),
        )
        return result

    # ── Order management ───────────────────────────────────────────────

    # Binance cancels one order per call, so this is a fan-out rather than a
    # true batch: it is NOT atomic, and a failure on one order leaves the
    # others cancelled. Per-order outcomes are reported the way Coinbase's
    # batch_cancel reports them.
    async def cancel_orders(self, product_id: str, order_ids: list[str]) -> dict:
        outcomes = await asyncio.gather(
            *(self._cancel_one(product_id, order_id) for order_id in order_ids),
            return_exceptions=True,
        )
        results = [
            {"order_id": order_id, "success": not isinstance(outcome, Exception),
             "failure_reason": str(outcome) if isinstance(outcome, Exception) else ""}
            for order_id, outcome in zip(order_ids, outcomes)
        ]
        logger.info(
            "Cancelled %d/%d order(s) on %s",
            sum(1 for r in results if r["success"]), len(order_ids), product_id,
        )
        return {"results": results}

    async def cancel_all_orders(self, product_id: str) -> list[dict]:
        result = await self._delete(
            "/sapi/v1/margin/openOrders",
            {"symbol": BinanceSymbol(product_id).wire(), "isIsolated": "TRUE"},
        )
        logger.info("Cancelled all open orders on %s", product_id)
        return result

    # Binance has no in-place amend — cancelReplace atomically cancels and
    # re-places, so the order gets a NEW orderId. STOP_ON_FAILURE means a
    # failed cancel aborts before the replacement is placed.
    async def replace_order(
        self,
        product_id: str,
        order_id: str,
        side: str,
        size: str,
        price: str,
        side_effect: str = NO_SIDE_EFFECT,
        time_in_force: str = "GTC",
    ) -> dict:
        params = {
            "symbol":            BinanceSymbol(product_id).wire(),
            "isIsolated":        "TRUE",
            "side":              side,
            "type":              "LIMIT",
            "cancelReplaceMode": "STOP_ON_FAILURE",
            "cancelOrderId":     order_id,
            "timeInForce":       time_in_force,
            "quantity":          size,
            "price":             price,
            "sideEffectType":    side_effect,
        }
        result = await self._post("/sapi/v1/margin/order/cancelReplace", params)
        logger.info("Replaced %s → size=%s price=%s", order_id, size, price)
        return result

    async def _cancel_one(self, product_id: str, order_id: str) -> dict:
        return await self._delete(
            "/sapi/v1/margin/order",
            {
                "symbol":     BinanceSymbol(product_id).wire(),
                "isIsolated": "TRUE",
                "orderId":    order_id,
            },
        )

    # ── Account ────────────────────────────────────────────────────────

    # Every isolated pair's balances, flattened to one entry per asset. The
    # `limit` parameter is accepted for ExchangeAdapter compatibility and
    # ignored — Binance returns every isolated pair in one response.
    async def get_accounts(self, limit: int = 50) -> list[dict]:
        result = await self._get("/sapi/v1/margin/isolated/account")
        return IsolatedAccounts(result).flattened()

    async def get_isolated_account(self, product_id: str) -> dict:
        result = await self._get(
            "/sapi/v1/margin/isolated/account",
            {"symbols": BinanceSymbol(product_id).wire()},
        )
        assets = result.get("assets", [])
        if not assets:
            raise BinanceError(
                0, result, f"No isolated margin account for {product_id} — transfer funds in first"
            )
        pair = assets[0]
        # A pair that was never created still answers, with a placeholder row
        # rather than an empty list — isolatedCreated/enabled/tradeEnabled all
        # false, and marginLevel "999" / liquidatePrice "0". Those risk numbers
        # read as a perfectly healthy account, so existence has to be taken from
        # the flag. Verified live: BTCUSDC returned exactly this shape while
        # myTrades and openOrders rejected the same symbol with -11001.
        if not pair.get("isolatedCreated", False):
            raise BinanceError(
                0, pair,
                f"No isolated margin account for {product_id} — it has never been "
                f"created. Transfer funds in (transfer_in) to create it; its "
                f"marginLevel/liquidatePrice are placeholders until then.",
            )
        return pair

    async def get_order(self, product_id: str, order_id: str) -> dict:
        return await self._get(
            "/sapi/v1/margin/order",
            {
                "symbol":     BinanceSymbol(product_id).wire(),
                "isIsolated": "TRUE",
                "orderId":    order_id,
            },
        )

    async def list_orders(
        self,
        product_id: str,
        order_status: Optional[list[str]] = None,
        limit: int = 500,
    ) -> list[dict]:
        symbol   = BinanceSymbol(product_id).wire()
        statuses = OrderStatuses(order_status or [])

        # openOrders is far cheaper than allOrders, so use it when only open
        # orders were asked for.
        if order_status and statuses.only_open():
            return await self._get(
                "/sapi/v1/margin/openOrders", {"symbol": symbol, "isIsolated": "TRUE"},
            )

        orders = await self._get(
            "/sapi/v1/margin/allOrders",
            {"symbol": symbol, "isIsolated": "TRUE", "limit": limit},
        )
        if not order_status:
            return orders
        return [order for order in orders if statuses.matches(order.get("status", ""))]

    # myTrades is symbol-scoped — there is no account-wide fills query, so
    # product_id is required.
    #
    # Paging starts from fromId=0 rather than from an unseeded first call:
    # without fromId Binance returns the MOST RECENT page, and asking for
    # "everything after the newest trade" would return nothing. Seeded at 0 it
    # walks the whole history forward, oldest first, which is what a cost-basis
    # reconstruction needs.
    async def get_fills(
        self,
        product_id: str,
        order_id: str = "",
        limit: int = 500,
        max_pages: int = 20,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "symbol":     BinanceSymbol(product_id).wire(),
            "isIsolated": "TRUE",
            "limit":      limit,
        }
        # One order's fills never exceed a page, so scoping skips the walk.
        if order_id:
            return await self._get(
                "/sapi/v1/margin/myTrades", {**params, "orderId": order_id}
            )

        fills:   list[dict] = []
        from_id: int        = 0
        for _ in range(max_pages):
            page = await self._get("/sapi/v1/margin/myTrades", {**params, "fromId": from_id})
            fills.extend(page)
            if len(page) < limit:
                break
            from_id = int(page[-1]["id"]) + 1
        return fills

    # ── Isolated wallet transfers ──────────────────────────────────────
    # An isolated pair starts empty and cannot trade until funds are moved in
    # from the spot wallet, and profits are stuck there until moved out.

    # Uses the margin-scoped transfer rather than the newer universal transfer
    # (/sapi/v1/asset/transfer, type=MAIN_ISOLATED_MARGIN). Binance has marked
    # this one deprecated, but it still serves and — unlike the universal
    # endpoint — needs no separate "Permits Universal Transfer" key permission
    # beyond the margin permission these orders already require.
    #
    # Verified live: a 2 USDT round trip out and back settled exactly, with no
    # fee and no dust, on a key reporting permitsUniversalTransfer=false and
    # enableInternalTransfer=false — the universal endpoint would have been
    # rejected on that key. Revisit only if this starts erroring.
    async def transfer_in(self, product_id: str, asset: str, amount: str) -> dict:
        return await self._transfer(product_id, asset, amount, "SPOT", "ISOLATED_MARGIN")

    async def transfer_out(self, product_id: str, asset: str, amount: str) -> dict:
        return await self._transfer(product_id, asset, amount, "ISOLATED_MARGIN", "SPOT")

    async def _transfer(
        self,
        product_id: str,
        asset: str,
        amount: str,
        trans_from: str,
        trans_to: str,
    ) -> dict:
        result = await self._post(
            "/sapi/v1/margin/isolated/transfer",
            {
                "asset":     asset.upper(),
                "symbol":    BinanceSymbol(product_id).wire(),
                "transFrom": trans_from,
                "transTo":   trans_to,
                "amount":    amount,
            },
        )
        logger.info("Transferred %s %s  %s → %s", amount, asset, trans_from, trans_to)
        return result

    # ── Market data ────────────────────────────────────────────────────

    async def get_product(self, product_id: str) -> dict:
        result = await self._get(
            "/api/v3/exchangeInfo",
            {"symbol": BinanceSymbol(product_id).wire()},
            auth=False,
        )
        symbols = result.get("symbols", [])
        if not symbols:
            raise BinanceError(0, result, f"Unknown product {product_id}")
        return BinanceProduct(symbols[0]).normalized()

    # The isolated-margin universe, joined to its trading filters and rolling
    # 24h stats. Three requests rather than one per pair: allPairs must be
    # signed, the other two are unauthenticated and return every symbol in a
    # single response, so the whole ~200-pair universe costs three round trips.
    async def list_products(self) -> list[dict]:
        pairs   = await self._get("/sapi/v1/margin/isolated/allPairs")
        symbols = [pair["symbol"] for pair in pairs]
        info, stats = await asyncio.gather(
            self._filtered("/api/v3/exchangeInfo", "symbols", symbols),
            self._filtered("/api/v3/ticker/24hr", "symbols", symbols),
        )
        return IsolatedCatalog(
            pairs,
            [entry for page in info for entry in page.get("symbols", [])],
            TwentyFourHourStats([row for page in stats for row in page]),
        ).products()

    # Both endpoints return the WHOLE venue when unfiltered — exchangeInfo alone
    # measured 17.5 MB over 3685 symbols and took ~6 s, against this session's
    # 10 s total timeout shared with every other request in flight. Filtering to
    # the isolated universe cuts it by an order of magnitude.
    #
    # Chunked because the filter travels in the query string: several hundred
    # symbols in one URL runs past the 8 KB many proxies cap at.
    async def _filtered(self, path: str, key: str, symbols: list[str], chunk: int = 100) -> list[Any]:
        pages = await asyncio.gather(*(
            self._page(path, key, symbols[index : index + chunk])
            for index in range(0, len(symbols), chunk)
        ))
        return [page for page in pages if page is not None]

    # Binance rejects the WHOLE request with -1121 if any one symbol in the
    # array is unknown to spot — it does not quietly omit it. An isolated pair
    # mid-delisting is exactly that case, and it would otherwise take the
    # entire ~200-pair scan down with it, making IsolatedCatalog's skip guard
    # unreachable. Bisecting isolates the bad symbol in ~log2(chunk) requests
    # instead of failing the run or re-fetching every symbol one at a time.
    async def _page(self, path: str, key: str, symbols: list[str]) -> Optional[Any]:
        if not symbols:
            return None
        try:
            return await self._get(
                path, {key: json.dumps(symbols, separators=(",", ":"))}, auth=False,
            )
        except BinanceError:
            if len(symbols) == 1:
                logger.warning("%s rejected symbol %s — skipping it", path, symbols[0])
                return None
            middle = len(symbols) // 2
            halves = await asyncio.gather(
                self._page(path, key, symbols[:middle]),
                self._page(path, key, symbols[middle:]),
            )
            return self._merged([half for half in halves if half is not None])

    # The two endpoints this serves return different shapes: exchangeInfo a
    # dict under "symbols", ticker/24hr a bare list.
    @staticmethod
    def _merged(pages: list[Any]) -> Optional[Any]:
        if not pages:
            return None
        if isinstance(pages[0], dict):
            symbols: list[Any] = []
            for page in pages:
                symbols.extend(page.get("symbols", []))
            return {"symbols": symbols}
        merged: list[Any] = []
        for page in pages:
            merged.extend(page)
        return merged

    async def fee_rates(self) -> tuple[float, float]:
        rows = await self._get("/sapi/v1/asset/tradeFee")
        fees = TradeFeeRates(rows)
        return fees.maker_bps(), fees.taker_bps()

    # Shaped like Coinbase's pricebooks response so callers written against
    # CoinbaseAdapter read the same path: ["pricebooks"][0]["bids"][0]["price"].
    async def get_best_bid_ask(self, *product_ids: str) -> dict:
        pricebooks = await asyncio.gather(
            *(self._book_ticker(product_id) for product_id in product_ids)
        )
        return {"pricebooks": list(pricebooks)}

    async def get_product_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
    ) -> list[dict]:
        rows = await self._get(
            "/api/v3/klines",
            {
                "symbol":    BinanceSymbol(product_id).wire(),
                "interval":  GRANULARITY_INTERVALS[granularity],
                "startTime": start * 1000,
                "endTime":   end * 1000,
                "limit":     MAX_CANDLES_PER_REQUEST,
            },
            auth=False,
        )
        TruncatedKlines(rows, end).raise_if_incomplete(product_id, granularity)
        return [KlineRow(row).as_candle() for row in rows]

    def max_candles_per_request(self) -> int:
        return MAX_CANDLES_PER_REQUEST

    def name(self) -> str:
        return "binance"

    async def get_market_trades(self, product_id: str, limit: int = 10) -> list[dict]:
        trades = await self._get(
            "/api/v3/trades",
            {"symbol": BinanceSymbol(product_id).wire(), "limit": limit},
            auth=False,
        )
        return [
            {
                "trade_id":   str(trade["id"]),
                "product_id": product_id,
                "price":      trade["price"],
                "size":       trade["qty"],
                "time":       str(trade["time"]),
                # isBuyerMaker=True means the taker sold into a resting bid.
                "side":       "SELL" if trade.get("isBuyerMaker") else "BUY",
            }
            for trade in trades
        ]

    async def _book_ticker(self, product_id: str) -> dict:
        raw = await self._get(
            "/api/v3/ticker/bookTicker",
            {"symbol": BinanceSymbol(product_id).wire()},
            auth=False,
        )
        return {
            "product_id": product_id,
            "bids": [{"price": raw["bidPrice"], "size": raw["bidQty"]}],
            "asks": [{"price": raw["askPrice"], "size": raw["askQty"]}],
        }


# ── Exception ──────────────────────────────────────────────────────────

class BinanceError(ExchangeError):
    pass


# ── Utility ────────────────────────────────────────────────────────────
# Binance's tickSize/stepSize are decimal strings ("0.01000000"), so the
# increment's own text carries the precision — no log10 rounding needed.

class SnappedValue:
    def __init__(self, value: float, increment: str) -> None:
        self._value     = value
        self._increment = increment

    def as_string(self) -> str:
        step     = float(self._increment)
        decimals = self._decimals()
        snapped  = round(round(self._value / step) * step, decimals)
        return f"{snapped:.{decimals}f}"

    def _decimals(self) -> int:
        trimmed = self._increment.rstrip("0")
        if "." not in trimmed:
            return 0
        return len(trimmed.split(".", 1)[1])


# ── Smoke test ─────────────────────────────────────────────────────────
# Run (as a module, from the repo root — `python binance/binance_adapter.py`
# directly fails with ModuleNotFoundError: No module named 'binance', since
# Python only adds the script's own directory to sys.path, not the repo root):
#   python -m binance.binance_adapter              # defaults to BTC-USDC
#   python -m binance.binance_adapter BTC-USDT     # any isolated pair
#
# Places a passive limit BUY 2% below best bid, confirms it's open, then
# cancels it.
#
# WARNING: Binance's spot testnet does not serve margin endpoints, so this
#          runs against live. Use a key with margin trading enabled, keep
#          amounts tiny, and transfer QUOTE currency into that pair's isolated
#          wallet first (adapter.transfer_in) — the order is a BUY, so holding
#          only the base asset is not enough and it will be rejected for
#          insufficient balance.

DEFAULT_SMOKE_TEST_PRODUCT = "BTC-USDC"


async def _smoke_test(product: str = DEFAULT_SMOKE_TEST_PRODUCT) -> None:
    from binance.credentials_file import CredentialsFile

    credentials = CredentialsFile().credentials()

    PRODUCT = product

    async with BinanceAdapter(credentials.api_key, credentials.api_secret) as adapter:

        # 1. Isolated balances
        print("\n=== Isolated margin balances ===")
        accounts = await adapter.get_accounts()
        for acct in accounts:
            free = float(acct["available_balance"]["value"])
            if free > 0 or float(acct["borrowed"]) > 0:
                print(
                    f"  {acct['product_id']:12s} {acct['currency']:6s} "
                    f"free={free:<16.8f} borrowed={float(acct['borrowed']):<16.8f} "
                    f"interest={float(acct['interest']):.8f}"
                )

        # 2. Pair risk — the exchange's own liquidation price
        print(f"\n=== Isolated risk: {PRODUCT} ===")
        pair = await adapter.get_isolated_account(PRODUCT)
        risk = IsolatedRisk(pair)
        print(f"  Margin level      : {risk.margin_level()}")
        print(f"  Liquidation price : {risk.liquidation_price()}")

        # 3. Product spec — gives us tickSize/stepSize for rounding
        print(f"\n=== Product: {PRODUCT} ===")
        product         = await adapter.get_product(PRODUCT)
        quote_increment = product["quote_increment"]
        base_increment  = product["base_increment"]
        print(f"  Quote increment : {quote_increment}")
        print(f"  Base increment  : {base_increment}")
        print(f"  Base min size   : {product['base_min_size']}")
        print(f"  Min notional    : {product['min_market_funds']}")

        # 4. Best bid/ask
        print(f"\n=== Best Bid/Ask: {PRODUCT} ===")
        bba  = await adapter.get_best_bid_ask(PRODUCT)
        book = bba["pricebooks"][0]
        bid  = float(book["bids"][0]["price"])
        ask  = float(book["asks"][0]["price"])
        print(f"  Bid : {bid}")
        print(f"  Ask : {ask}")

        # 5. Limit buy 2% below bid (passive — won't fill). NO_SIDE_EFFECT, so
        #    nothing is borrowed: this is economically a spot buy.
        limit_price = SnappedValue(bid * 0.98, quote_increment).as_string()
        #    Size up to the pair's min notional — Binance rejects dust orders.
        min_notional = float(product["min_market_funds"])
        raw_size     = max(float(product["base_min_size"]), min_notional / float(limit_price) * 1.05)
        base_size    = SnappedValue(raw_size, base_increment).as_string()

        symbol = BinanceSymbol(PRODUCT)
        print(
            f"\n=== Placing limit BUY: {base_size} {symbol.base()} "
            f"@ {limit_price} {symbol.quote()} ==="
        )
        result   = await adapter.limit_buy(
            product_id  = PRODUCT,
            base_size   = base_size,
            limit_price = limit_price,
            post_only   = True,
        )
        order_id = result.get("orderId", "")
        print(f"  Order ID : {order_id}")
        print(f"  Status   : {result.get('status')}")

        # 6. Confirm open
        print(f"\n=== Open orders on {PRODUCT} ===")
        open_orders = await adapter.list_orders(PRODUCT, order_status=["OPEN"])
        for order in open_orders:
            print(f"  {order['orderId']}  {order['side']}  {order['origQty']} @ {order['price']}")

        # 7. Cancel
        print(f"\n=== Cancelling {order_id} ===")
        cancelled = await adapter.cancel_orders(PRODUCT, [order_id])
        for entry in cancelled["results"]:
            print(
                f"  {entry['order_id']}  success={entry['success']}  "
                f"reason={entry['failure_reason']}"
            )

        print("\n✓ Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_smoke_test(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SMOKE_TEST_PRODUCT))
