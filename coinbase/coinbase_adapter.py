"""
Coinbase Advanced Trade Execution Adapter
------------------------------------------
API: https://api.coinbase.com/api/v3/brokerage
Docs: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/

Auth: CDP API keys using short-lived JWTs (ES256, 2-minute expiry).
      Keys are created at https://portal.cdp.coinbase.com/

Key format:
    api_key    = "organizations/{org_id}/apiKeys/{key_id}"
    api_secret = "-----BEGIN EC PRIVATE KEY-----\\nYOUR KEY\\n-----END EC PRIVATE KEY-----\\n"

Order model (Coinbase Advanced):
    All orders use a nested `order_configuration` dict.
    Amount fields are always strings (Coinbase rejects floats).

    Limit GTC buy/sell:
        order_configuration = {"limit_limit_gtc": {"baseSize": "0.001", "limitPrice": "80000"}}

    Market buy (spend quote currency):
        order_configuration = {"market_market_ioc": {"quoteSize": "10"}}   # spend $10 USDC

    Market sell (sell base currency):
        order_configuration = {"market_market_ioc": {"baseSize": "0.001"}} # sell 0.001 BTC

NOTE: Coinbase has no official sandbox for Advanced Trade.
      Use very small amounts on live when testing.

Usage:
    async with CoinbaseAdapter(api_key="...", api_secret="...") as adapter:
        ticker  = await adapter.get_best_bid_ask("BTC-USDC")
        bid     = float(ticker["pricebooks"][0]["bids"][0]["price"])
        order   = await adapter.limit_buy(
            product_id = "BTC-USDC",
            base_size  = "0.001",
            limit_price= str(round(bid * 0.98, 2)),
        )
        await adapter.cancel_orders([order["order_id"]])
"""

import asyncio
import time
import uuid
import logging
from typing import Any, Optional
import aiohttp
import jwt as pyjwt      # pip install "PyJWT[crypto]"

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coinbase.com"
_JWT_TTL = 120  # seconds — Coinbase rejects JWTs older than 2 minutes


# ── JWT helpers ────────────────────────────────────────────────────────

def _build_jwt(api_key: str, api_secret: str, method: str, path: str) -> str:
    """
    Build a short-lived ES256 JWT for one request.
    path should be just the path component, e.g. "/api/v3/brokerage/orders".
    """
    now = int(time.time())
    payload = {
        "sub": api_key,
        "iss": "cdp",
        "nbf": now,
        "exp": now + _JWT_TTL,
        "uri": f"{method} api.coinbase.com{path}",
    }
    # nonce header prevents replay
    headers = {"kid": api_key, "nonce": uuid.uuid4().hex}
    return pyjwt.encode(payload, api_secret, algorithm="ES256", headers=headers)


# ── Adapter ────────────────────────────────────────────────────────────

class CoinbaseAdapter:
    """
    Async execution adapter for the Coinbase Advanced Trade REST API.

    Auth: per-request ES256 JWTs — no token refresh needed.
    Orders: POST JSON to /api/v3/brokerage/orders.
    Market data: GET (no auth required for most endpoints).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        request_timeout: float = 10.0,
    ):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.timeout    = aiohttp.ClientTimeout(total=request_timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        logger.info("Coinbase adapter connected.")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()

    # ── HTTP helpers ───────────────────────────────────────────────────

    def _auth_headers(self, method: str, path: str) -> dict:
        token = _build_jwt(self.api_key, self.api_secret, method, path)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    async def _get(self, path: str, params: Optional[dict] = None, auth: bool = True) -> Any:
        url     = f"{BASE_URL}{path}"
        headers = self._auth_headers("GET", path) if auth else {}
        async with self._session.get(url, params=params, headers=headers) as resp:
            return await self._parse(resp)

    async def _post(self, path: str, body: dict) -> Any:
        url     = f"{BASE_URL}{path}"
        headers = self._auth_headers("POST", path)
        async with self._session.post(url, json=body, headers=headers) as resp:
            return await self._parse(resp)

    async def _delete(self, path: str, body: Optional[dict] = None) -> Any:
        url     = f"{BASE_URL}{path}"
        headers = self._auth_headers("DELETE", path)
        async with self._session.delete(url, json=body or {}, headers=headers) as resp:
            return await self._parse(resp)

    @staticmethod
    async def _parse(resp: aiohttp.ClientResponse) -> Any:
        # Status is checked against a body that may not be JSON at all: a 429
        # or a gateway 5xx often comes back as plain text or HTML from an edge
        # proxy, and decoding that first raises aiohttp's ContentTypeError
        # instead of CoinbaseError — throwing away the status code that says
        # what actually went wrong.
        try:
            data = await resp.json()
        except aiohttp.ContentTypeError:
            data = {"body": (await resp.text())[:500]}
        if not resp.ok:
            raise CoinbaseError(resp.status, data)
        # Some endpoints wrap in {"order": ...}, others return directly
        return data

    # ── Orders ─────────────────────────────────────────────────────────

    async def limit_buy(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        """
        Limit buy — GTC by default.

        product_id:    e.g. "BTC-USDC", "ETH-USDC"
        base_size:     amount of base currency to buy, as a string e.g. "0.001"
        limit_price:   max price to pay, as a string e.g. "80000.00"
        post_only:     maker-only; order rejected if it would immediately fill
        """
        cfg = {
            "limit_limit_gtc": {
                "baseSize":   base_size,
                "limitPrice": limit_price,
                "postOnly":   post_only,
            }
        }
        return await self._create_order(product_id, "BUY", cfg, client_order_id)

    async def limit_sell(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        post_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        """
        Limit sell — GTC.

        base_size:   amount of base currency to sell e.g. "0.001"
        limit_price: min price to accept e.g. "90000.00"
        """
        cfg = {
            "limit_limit_gtc": {
                "baseSize":   base_size,
                "limitPrice": limit_price,
                "postOnly":   post_only,
            }
        }
        return await self._create_order(product_id, "SELL", cfg, client_order_id)

    async def market_buy(
        self,
        product_id: str,
        quote_size: str,
        client_order_id: str = "",
    ) -> dict:
        """
        Market buy — spend a fixed amount of quote currency.

        quote_size: USDC (or USD) amount to spend e.g. "10.00"
        """
        cfg = {"market_market_ioc": {"quoteSize": quote_size}}
        return await self._create_order(product_id, "BUY", cfg, client_order_id)

    async def market_sell(
        self,
        product_id: str,
        base_size: str,
        client_order_id: str = "",
    ) -> dict:
        """
        Market sell — sell a fixed amount of base currency.

        base_size: BTC/ETH amount to sell e.g. "0.001"
        """
        cfg = {"market_market_ioc": {"baseSize": base_size}}
        return await self._create_order(product_id, "SELL", cfg, client_order_id)

    async def limit_buy_ioc(
        self,
        product_id: str,
        base_size: str,
        limit_price: str,
        client_order_id: str = "",
    ) -> dict:
        """Limit buy IOC — cancels unfilled remainder immediately."""
        cfg = {
            "sor_limit_ioc": {
                "baseSize":   base_size,
                "limitPrice": limit_price,
            }
        }
        return await self._create_order(product_id, "BUY", cfg, client_order_id)

    async def _create_order(
        self,
        product_id: str,
        side: str,
        order_configuration: dict,
        client_order_id: str = "",
    ) -> dict:
        body = {
            "client_order_id":     client_order_id or uuid.uuid4().hex,
            "product_id":          product_id,
            "side":                side,
            "order_configuration": order_configuration,
        }
        result  = await self._post("/api/v3/brokerage/orders", body)
        order   = result.get("order_id") or result.get("success_response", {})
        success = result.get("success", False)

        if not success:
            err = result.get("error_response", result)
            raise CoinbaseError(0, err, f"Order placement failed: {err}")

        order_id = result.get("success_response", {}).get("order_id", "")
        logger.info(
            "%-4s  %-12s  base=%s  price=%s  id=%s",
            side,
            product_id,
            order_configuration.get(
                list(order_configuration.keys())[0], {}
            ).get("baseSize") or order_configuration.get(
                list(order_configuration.keys())[0], {}
            ).get("quoteSize", "?"),
            order_configuration.get(
                list(order_configuration.keys())[0], {}
            ).get("limitPrice", "MARKET"),
            order_id,
        )
        return result

    async def cancel_orders(self, order_ids: list[str]) -> dict:
        """Cancel one or more orders by ID."""
        result = await self._post(
            "/api/v3/brokerage/orders/batch_cancel",
            {"order_ids": order_ids},
        )
        logger.info("Cancelled %d order(s)", len(order_ids))
        return result

    async def edit_order(
        self,
        order_id: str,
        size: str,
        price: str,
    ) -> dict:
        """Amend an open limit order's size and/or price."""
        body = {"order_id": order_id, "size": size, "price": price}
        result = await self._post("/api/v3/brokerage/orders/edit", body)
        logger.info("Edited %s → size=%s price=%s", order_id, size, price)
        return result

    # ── Account ────────────────────────────────────────────────────────

    async def get_accounts(self, limit: int = 50) -> list[dict]:
        """Return all brokerage accounts (spot wallets)."""
        result = await self._get("/api/v3/brokerage/accounts", {"limit": limit})
        return result.get("accounts", result)

    async def get_account(self, account_uuid: str) -> dict:
        """Return one account by its UUID."""
        return await self._get(f"/api/v3/brokerage/accounts/{account_uuid}")

    async def get_order(self, order_id: str) -> dict:
        """Return the current state of one order."""
        result = await self._get(f"/api/v3/brokerage/orders/historical/{order_id}")
        return result.get("order", result)

    async def list_orders(
        self,
        product_id: Optional[str] = None,
        order_status: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        List orders with optional filters.
        order_status: ["OPEN"] | ["FILLED"] | ["CANCELLED"] etc.
        """
        params: dict[str, Any] = {"limit": limit}
        if product_id:
            params["product_id"] = product_id
        if order_status:
            params["order_status"] = order_status
        result = await self._get("/api/v3/brokerage/orders/historical/batch", params)
        return result.get("orders", result)

    async def get_fills(
        self,
        order_id: str = "",
        product_id: str = "",
        limit: int = 250,
        max_pages: int = 20,
    ) -> list[dict]:
        """
        Return fill records, optionally filtered by order or product.
        Follows Coinbase cursor pagination up to max_pages.
        """
        params: dict[str, Any] = {"limit": limit}
        if order_id:
            params["order_id"] = order_id
        if product_id:
            params["product_id"] = product_id

        fills: list[dict] = []
        for _ in range(max_pages):
            result = await self._get(
                "/api/v3/brokerage/orders/historical/fills", params
            )
            page = result.get("fills", [])
            fills.extend(page)
            cursor = result.get("cursor", "")
            if not cursor or not page:
                break
            params["cursor"] = cursor
        return fills

    # ── Market data ────────────────────────────────────────────────────

    async def get_product(self, product_id: str) -> dict:
        """Product details including quote/base increment (tick size)."""
        return await self._get(f"/api/v3/brokerage/products/{product_id}")

    async def get_best_bid_ask(self, *product_ids: str) -> dict:
        """Best bid and ask for one or more products."""
        return await self._get(
            "/api/v3/brokerage/best_bid_ask",
            {"product_ids": list(product_ids)},
        )

    async def get_product_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
    ) -> list[dict]:
        """
        OHLCV candles.
        granularity: ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE |
                     ONE_HOUR | TWO_HOUR | SIX_HOUR | ONE_DAY
        """
        result = await self._get(
            f"/api/v3/brokerage/products/{product_id}/candles",
            {"start": start, "end": end, "granularity": granularity},
        )
        return result.get("candles", result)

    async def get_market_trades(self, product_id: str, limit: int = 10) -> list[dict]:
        """Recent public trades for a product."""
        result = await self._get(
            f"/api/v3/brokerage/products/{product_id}/ticker",
            {"limit": limit},
        )
        return result.get("trades", result)


# ── Exception ──────────────────────────────────────────────────────────

class CoinbaseError(Exception):
    def __init__(self, status: int, raw: Any, message: str = ""):
        msg = message or str(raw)
        super().__init__(f"[HTTP {status}] {msg} | raw={raw}")
        self.status = status
        self.raw    = raw


# ── Utility ────────────────────────────────────────────────────────────

def snap_to_increment(value: float, increment: float) -> str:
    """
    Round a price or size to the nearest valid increment and return as string.
    Uses the product's quote_increment (prices) or base_increment (sizes).

    e.g. snap_to_increment(83241.7, 0.01) -> "83241.70"
    """
    import math
    decimals = max(0, -int(math.floor(math.log10(increment))))
    snapped  = round(round(value / increment) * increment, decimals)
    return f"{snapped:.{decimals}f}"


# ── Smoke test ─────────────────────────────────────────────────────────
# Run (as a module, from the repo root — `python coinbase/coinbase_adapter.py`
# directly fails with ModuleNotFoundError: No module named 'coinbase', since
# Python only adds the script's own directory to sys.path, not the repo root):
#   python -m coinbase.coinbase_adapter
#
# Places a passive limit BUY on BTC-USDC 2% below best bid,
# confirms it's open, then cancels it.
#
# WARNING: Coinbase Advanced Trade has no sandbox. This runs against live.
#          Use a key with trade:read_write and keep amounts tiny.

async def _smoke_test():
    from coinbase.credentials_file import CredentialsFile

    credentials = CredentialsFile().credentials()

    PRODUCT = "BTC-USDC"

    async with CoinbaseAdapter(credentials.api_key, credentials.api_secret) as adapter:

        # 1. Accounts
        print("\n=== Accounts ===")
        accounts = await adapter.get_accounts()
        for acct in accounts:
            bal = acct.get("available_balance", {})
            if float(bal.get("value", 0)) > 0:
                print(f"  {acct['name']:20s}  {bal.get('value'):>14s} {bal.get('currency')}")

        # 2. Product spec — gives us quote_increment for price rounding
        print(f"\n=== Product: {PRODUCT} ===")
        product         = await adapter.get_product(PRODUCT)
        quote_increment = float(product["quote_increment"])   # e.g. 0.01
        base_increment  = float(product["base_increment"])    # e.g. 0.00000001
        base_min        = float(product.get("base_min_size", base_increment))
        print(f"  Quote increment : {quote_increment}")
        print(f"  Base increment  : {base_increment}")
        print(f"  Base min size   : {base_min}")

        # 3. Best bid/ask
        print(f"\n=== Best Bid/Ask: {PRODUCT} ===")
        bba     = await adapter.get_best_bid_ask(PRODUCT)
        book    = bba["pricebooks"][0]
        bid     = float(book["bids"][0]["price"])
        ask     = float(book["asks"][0]["price"])
        print(f"  Bid : {bid}")
        print(f"  Ask : {ask}")

        # 4. Limit buy 2% below bid (passive — won't fill)
        limit_price = snap_to_increment(bid * 0.98, quote_increment)
        base_size   = snap_to_increment(base_min, base_increment)

        print(f"\n=== Placing limit BUY: {base_size} BTC @ {limit_price} USDC ===")
        result   = await adapter.limit_buy(
            product_id  = PRODUCT,
            base_size   = base_size,
            limit_price = limit_price,
            post_only   = True,
        )
        order_id = result.get("success_response", {}).get("order_id", "")
        print(f"  Order ID : {order_id}")
        print(f"  Success  : {result.get('success')}")

        # 5. Confirm open
        print(f"\n=== Open orders on {PRODUCT} ===")
        open_orders = await adapter.list_orders(product_id=PRODUCT, order_status=["OPEN"])
        for o in open_orders:
            cfg = o.get("order_configuration", {})
            print(f"  {o['order_id']}  {o['side']}  {cfg}")

        # 6. Cancel
        print(f"\n=== Cancelling {order_id} ===")
        cancel_result = await adapter.cancel_orders([order_id])
        results = cancel_result.get("results", [])
        for r in results:
            print(f"  {r.get('order_id')}  success={r.get('success')}  reason={r.get('failure_reason','')}")

        print("\n✓ Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_smoke_test())
