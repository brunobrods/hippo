import hashlib
import hmac
from collections import deque
from urllib.parse import parse_qs, urlparse

import pytest

from binance.binance_adapter import (
    AUTO_REPAY,
    MARGIN_BUY,
    NO_SIDE_EFFECT,
    BinanceAdapter,
    BinanceError,
    BinanceProduct,
    BinanceSymbol,
    HmacSignature,
    IsolatedAccounts,
    IsolatedRisk,
    KlineRow,
    LimitOrder,
    MarketOrder,
    OrderStatuses,
    SignedQuery,
    SnappedValue,
)
from exchange.adapter import ExchangeError


# ── Test doubles ───────────────────────────────────────────────────────
# The adapter builds its own aiohttp session in connect(), so tests swap in a
# fake one directly rather than opening a real connection.

class FakeResponse:
    def __init__(self, payload, status: int = 200) -> None:
        self._payload = payload
        self.status   = status

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def json(self):
        return self._payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *_) -> bool:
        return False


class FakeSession:
    def __init__(self, *payloads, status: int = 200) -> None:
        self._payloads = deque(payloads)
        self._status   = status
        self.requests: list[tuple[str, str, dict]] = []

    def get(self, url, headers=None):
        return self._respond("GET", url, headers)

    def post(self, url, headers=None):
        return self._respond("POST", url, headers)

    def delete(self, url, headers=None):
        return self._respond("DELETE", url, headers)

    # The last payload repeats, so a fan-out (cancel_orders, get_best_bid_ask)
    # needs only one entry unless the responses differ.
    def _respond(self, method: str, url: str, headers) -> FakeResponse:
        self.requests.append((method, url, headers or {}))
        payload = self._payloads.popleft() if len(self._payloads) > 1 else self._payloads[0]
        return FakeResponse(payload, self._status)

    def params(self, index: int = 0) -> dict[str, str]:
        _, url, _ = self.requests[index]
        return {key: value[0] for key, value in parse_qs(urlparse(url).query).items()}

    def path(self, index: int = 0) -> str:
        return urlparse(self.requests[index][1]).path


def _adapter(session: FakeSession) -> BinanceAdapter:
    adapter = BinanceAdapter("test-key", "test-secret")
    adapter._session = session
    return adapter


# ── BinanceSymbol ──────────────────────────────────────────────────────

def test_symbol_strips_the_separator_for_the_wire():
    assert BinanceSymbol("BTC-USDC").wire() == "BTCUSDC"


def test_symbol_uppercases_the_wire_form():
    assert BinanceSymbol("btc-usdc").wire() == "BTCUSDC"


def test_symbol_splits_base_and_quote():
    symbol = BinanceSymbol("ETH-USDC")
    assert symbol.base() == "ETH"
    assert symbol.quote() == "USDC"


def test_symbol_passes_through_an_already_joined_wire_form():
    assert BinanceSymbol("BTCUSDC").wire() == "BTCUSDC"


def test_symbol_refuses_to_guess_base_and_quote_without_a_separator():
    with pytest.raises(ValueError, match="canonical"):
        BinanceSymbol("BTCUSDC").base()


# ── Signing ────────────────────────────────────────────────────────────

def test_hmac_signature_matches_the_reference_digest():
    expected = hmac.new(b"secret", b"symbol=BTCUSDC", hashlib.sha256).hexdigest()
    assert HmacSignature("secret", "symbol=BTCUSDC").value() == expected


def test_signed_query_appends_timestamp_recv_window_and_signature():
    encoded = SignedQuery({"symbol": "BTCUSDC"}, "secret").encoded()
    params  = {k: v[0] for k, v in parse_qs(encoded).items()}

    assert params["symbol"] == "BTCUSDC"
    assert params["recvWindow"] == "5000"
    assert int(params["timestamp"]) > 0
    assert len(params["signature"]) == 64


def test_signed_query_signs_the_exact_string_it_sends():
    encoded          = SignedQuery({"symbol": "BTCUSDC"}, "secret").encoded()
    payload, _, sig  = encoded.rpartition("&signature=")

    assert HmacSignature("secret", payload).value() == sig


def test_signed_query_drops_empty_parameters():
    encoded = SignedQuery({"symbol": "BTCUSDC", "orderId": "", "cursor": None}, "secret").encoded()

    assert "orderId" not in encoded
    assert "cursor" not in encoded


# ── Response translation ───────────────────────────────────────────────

def test_kline_row_converts_milliseconds_to_the_canonical_second_start():
    candle = KlineRow([1700000000000, "1.0", "3.0", "0.5", "2.0", "10.0", 1700003599999]).as_candle()

    assert candle == {
        "start": "1700000000", "open": "1.0", "high": "3.0",
        "low": "0.5", "close": "2.0", "volume": "10.0",
    }


def test_kline_row_produces_the_keys_the_indicator_frame_reads():
    candle = KlineRow([1700000000000, "1", "1", "1", "1", "1"]).as_candle()

    assert set(candle) >= {"start", "close", "high", "low", "volume"}


def _isolated_pair(symbol: str = "BTCUSDC", free_base: str = "0.5", free_quote: str = "100") -> dict:
    return {
        "symbol":     symbol,
        "isolatedCreated": True,
        "enabled":         True,
        "marginLevel": "3.5",
        "liquidatePrice": "41000.0",
        "baseAsset":  {"asset": "BTC", "free": free_base, "locked": "0.1",
                       "borrowed": "0.2", "interest": "0.0001", "netAsset": "0.2"},
        "quoteAsset": {"asset": "USDC", "free": free_quote, "locked": "10",
                       "borrowed": "0", "interest": "0", "netAsset": "110"},
    }


def test_isolated_accounts_flatten_to_one_entry_per_asset():
    accounts = IsolatedAccounts({"assets": [_isolated_pair()]}).flattened()

    assert [a["currency"] for a in accounts] == ["BTC", "USDC"]


def test_isolated_accounts_carry_the_pair_they_are_scoped_to():
    accounts = IsolatedAccounts({"assets": [_isolated_pair()]}).flattened()

    assert {a["product_id"] for a in accounts} == {"BTC-USDC"}


def test_isolated_accounts_expose_balances_in_the_account_shape():
    base = IsolatedAccounts({"assets": [_isolated_pair()]}).flattened()[0]

    assert base["available_balance"] == {"value": "0.5", "currency": "BTC"}
    assert base["hold"] == {"value": "0.1", "currency": "BTC"}
    assert base["borrowed"] == "0.2"
    assert base["interest"] == "0.0001"


def test_isolated_accounts_of_an_empty_response_are_empty():
    assert IsolatedAccounts({}).flattened() == []


def test_isolated_risk_reads_binances_own_numbers():
    risk = IsolatedRisk(_isolated_pair())

    assert risk.margin_level() == pytest.approx(3.5)
    assert risk.liquidation_price() == pytest.approx(41000.0)


def _exchange_info_symbol(notional_key: str = "NOTIONAL") -> dict:
    return {
        "symbol": "BTCUSDC", "baseAsset": "BTC", "quoteAsset": "USDC", "status": "TRADING",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.00001000",
             "minQty": "0.00001000", "maxQty": "9000.00000000"},
            {"filterType": notional_key, "minNotional": "5.00000000"},
        ],
    }


def test_product_renames_filters_to_the_increment_fields_callers_read():
    product = BinanceProduct(_exchange_info_symbol()).normalized()

    assert product["quote_increment"] == "0.01000000"
    assert product["base_increment"] == "0.00001000"
    assert product["base_min_size"] == "0.00001000"
    assert product["min_market_funds"] == "5.00000000"


def test_product_rebuilds_the_canonical_product_id():
    assert BinanceProduct(_exchange_info_symbol()).normalized()["product_id"] == "BTC-USDC"


def test_product_accepts_the_legacy_min_notional_filter_name():
    product = BinanceProduct(_exchange_info_symbol("MIN_NOTIONAL")).normalized()

    assert product["min_market_funds"] == "5.00000000"


def test_product_falls_back_when_a_filter_is_absent():
    product = BinanceProduct({"baseAsset": "BTC", "quoteAsset": "USDC", "filters": []}).normalized()

    assert product["quote_increment"] == "0.01"
    assert product["min_market_funds"] == "0"


# ── OrderStatuses ──────────────────────────────────────────────────────

def test_open_maps_to_both_of_binances_live_statuses():
    assert OrderStatuses(["OPEN"]).native() == ("NEW", "PARTIALLY_FILLED")


def test_cancelled_covers_expired_and_rejected_too():
    assert OrderStatuses(["CANCELLED"]).matches("EXPIRED")
    assert OrderStatuses(["CANCELLED"]).matches("REJECTED")


def test_only_open_is_false_when_other_statuses_are_requested():
    assert OrderStatuses(["OPEN"]).only_open()
    assert not OrderStatuses(["OPEN", "FILLED"]).only_open()


def test_an_unmapped_status_passes_through_verbatim():
    assert OrderStatuses(["PARTIALLY_FILLED"]).matches("PARTIALLY_FILLED")


# ── Order shapes ───────────────────────────────────────────────────────

def test_limit_order_is_gtc_by_default():
    assert LimitOrder("0.001", "80000").params() == {
        "type": "LIMIT", "timeInForce": "GTC", "quantity": "0.001", "price": "80000",
    }


def test_post_only_becomes_limit_maker_without_a_time_in_force():
    params = LimitOrder("0.001", "80000", post_only=True).params()

    assert params["type"] == "LIMIT_MAKER"
    assert "timeInForce" not in params


def test_limit_order_carries_an_explicit_time_in_force():
    assert LimitOrder("0.001", "80000", time_in_force="IOC").params()["timeInForce"] == "IOC"


def test_market_order_spends_quote_when_given_a_quote_size():
    assert MarketOrder(quote_size="10").params() == {"type": "MARKET", "quoteOrderQty": "10"}


def test_market_order_sells_base_when_given_a_base_size():
    assert MarketOrder(base_size="0.001").params() == {"type": "MARKET", "quantity": "0.001"}


def test_market_order_without_a_size_raises():
    with pytest.raises(ValueError, match="positive base_size or quote_size"):
        MarketOrder().params()


def test_a_zero_size_is_rejected_locally_rather_than_by_the_exchange():
    with pytest.raises(ValueError, match="positive"):
        MarketOrder(base_size="0").params()
    with pytest.raises(ValueError, match="positive"):
        MarketOrder(quote_size="0.00").params()


# ── SnappedValue ───────────────────────────────────────────────────────

def test_snapped_value_rounds_to_the_increment():
    assert SnappedValue(83241.7891, "0.01000000").as_string() == "83241.79"


def test_snapped_value_keeps_the_increments_own_precision():
    assert SnappedValue(0.123456789, "0.00001000").as_string() == "0.12346"


def test_snapped_value_of_a_whole_number_increment_has_no_decimals():
    assert SnappedValue(41.7, "1").as_string() == "42"


# ── Orders over the wire ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_limit_buy_is_isolated_and_borrows_nothing():
    session = FakeSession({"orderId": 1, "status": "NEW"})

    await _adapter(session).limit_buy("BTC-USDC", "0.001", "80000")

    params = session.params()
    assert session.path() == "/sapi/v1/margin/order"
    assert params["symbol"] == "BTCUSDC"
    assert params["isIsolated"] == "TRUE"
    assert params["side"] == "BUY"
    assert params["sideEffectType"] == NO_SIDE_EFFECT
    assert params["type"] == "LIMIT"
    assert params["quantity"] == "0.001"
    assert params["price"] == "80000"


@pytest.mark.asyncio
async def test_market_buy_spends_quote_currency():
    session = FakeSession({"orderId": 1})

    await _adapter(session).market_buy("BTC-USDC", "10.00")

    params = session.params()
    assert params["type"] == "MARKET"
    assert params["quoteOrderQty"] == "10.00"
    assert params["sideEffectType"] == NO_SIDE_EFFECT


@pytest.mark.asyncio
async def test_market_sell_closing_a_long_borrows_nothing():
    session = FakeSession({"orderId": 1})

    await _adapter(session).market_sell("BTC-USDC", "0.001")

    params = session.params()
    assert params["side"] == "SELL"
    assert params["sideEffectType"] == NO_SIDE_EFFECT


@pytest.mark.asyncio
async def test_limit_buy_ioc_does_not_rest_on_the_book():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_buy_ioc("BTC-USDC", "0.001", "80000")

    assert session.params()["timeInForce"] == "IOC"


@pytest.mark.asyncio
async def test_a_short_sells_and_auto_borrows_the_base_asset():
    session = FakeSession({"orderId": 1})

    await _adapter(session).market_short("BTC-USDC", "0.001")

    params = session.params()
    assert params["side"] == "SELL"
    assert params["sideEffectType"] == MARGIN_BUY


@pytest.mark.asyncio
async def test_a_limit_short_also_auto_borrows():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_short("BTC-USDC", "0.001", "90000")

    params = session.params()
    assert params["side"] == "SELL"
    assert params["sideEffectType"] == MARGIN_BUY
    assert params["price"] == "90000"


@pytest.mark.asyncio
async def test_a_cover_buys_and_repays_the_borrow():
    session = FakeSession({"orderId": 1})

    await _adapter(session).market_cover("BTC-USDC", "0.001")

    params = session.params()
    assert params["side"] == "BUY"
    assert params["sideEffectType"] == AUTO_REPAY


@pytest.mark.asyncio
async def test_a_limit_cover_repays_too():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_cover("BTC-USDC", "0.001", "70000")

    assert session.params()["sideEffectType"] == AUTO_REPAY


@pytest.mark.asyncio
async def test_an_explicit_client_order_id_is_sent_verbatim():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_buy("BTC-USDC", "0.001", "80000", client_order_id="mine-1")

    assert session.params()["newClientOrderId"] == "mine-1"


@pytest.mark.asyncio
async def test_orders_without_a_client_order_id_still_get_one():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_buy("BTC-USDC", "0.001", "80000")

    assert len(session.params()["newClientOrderId"]) == 32


@pytest.mark.asyncio
async def test_the_api_key_travels_in_the_header_not_the_query():
    session = FakeSession({"orderId": 1})

    await _adapter(session).limit_buy("BTC-USDC", "0.001", "80000")

    _, url, headers = session.requests[0]
    assert headers["X-MBX-APIKEY"] == "test-key"
    assert "test-secret" not in url


# ── Order management ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_orders_reports_one_result_per_order():
    session = FakeSession({"orderId": 1, "status": "CANCELED"})

    result = await _adapter(session).cancel_orders("BTC-USDC", ["1", "2"])

    assert [entry["success"] for entry in result["results"]] == [True, True]
    assert len(session.requests) == 2


@pytest.mark.asyncio
async def test_a_failed_cancel_is_reported_rather_than_hiding_the_others():
    session = FakeSession({"code": -2011, "msg": "Unknown order sent."}, status=400)

    result = await _adapter(session).cancel_orders("BTC-USDC", ["1"])

    entry = result["results"][0]
    assert entry["success"] is False
    assert "Unknown order" in entry["failure_reason"]


@pytest.mark.asyncio
async def test_list_orders_uses_the_cheap_endpoint_for_open_orders_only():
    session = FakeSession([{"orderId": 1, "status": "NEW"}])

    await _adapter(session).list_orders("BTC-USDC", order_status=["OPEN"])

    assert session.path() == "/sapi/v1/margin/openOrders"


@pytest.mark.asyncio
async def test_list_orders_filters_history_by_the_requested_status():
    session = FakeSession([
        {"orderId": 1, "status": "FILLED"},
        {"orderId": 2, "status": "CANCELED"},
    ])

    orders = await _adapter(session).list_orders("BTC-USDC", order_status=["FILLED"])

    assert session.path() == "/sapi/v1/margin/allOrders"
    assert [order["orderId"] for order in orders] == [1]


@pytest.mark.asyncio
async def test_list_orders_without_a_status_returns_everything():
    session = FakeSession([{"orderId": 1, "status": "FILLED"}, {"orderId": 2, "status": "CANCELED"}])

    orders = await _adapter(session).list_orders("BTC-USDC")

    assert len(orders) == 2


@pytest.mark.asyncio
async def test_replace_order_cancels_and_replaces_atomically():
    session = FakeSession({"cancelResult": "SUCCESS", "newOrderResponse": {"orderId": 9}})

    await _adapter(session).replace_order("BTC-USDC", "1", "BUY", "0.002", "79000")

    params = session.params()
    assert session.path() == "/sapi/v1/margin/order/cancelReplace"
    assert params["cancelOrderId"] == "1"
    assert params["cancelReplaceMode"] == "STOP_ON_FAILURE"
    assert params["quantity"] == "0.002"


# ── Account ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_accounts_flattens_every_isolated_pair():
    session = FakeSession({"assets": [_isolated_pair()]})

    accounts = await _adapter(session).get_accounts()

    assert [a["currency"] for a in accounts] == ["BTC", "USDC"]


@pytest.mark.asyncio
async def test_get_isolated_account_asks_for_one_symbol():
    session = FakeSession({"assets": [_isolated_pair()]})

    pair = await _adapter(session).get_isolated_account("BTC-USDC")

    assert session.params()["symbols"] == "BTCUSDC"
    assert pair["marginLevel"] == "3.5"


@pytest.mark.asyncio
async def test_an_untraded_isolated_pair_raises_rather_than_returning_nothing():
    session = FakeSession({"assets": []})

    with pytest.raises(BinanceError, match="transfer funds in first"):
        await _adapter(session).get_isolated_account("BTC-USDC")


@pytest.mark.asyncio
async def test_a_never_created_pairs_placeholder_row_raises_instead_of_reassuring():
    # Verified live: Binance answers for a pair that does not exist with a full
    # row whose risk fields read as healthy, not with an empty list.
    session = FakeSession({"assets": [{
        "symbol": "BTCUSDC",
        "isolatedCreated": False,
        "enabled": False,
        "tradeEnabled": False,
        "marginLevel": "999",
        "liquidatePrice": "0",
        "baseAsset":  {"asset": "BTC", "free": "0", "locked": "0",
                       "borrowed": "0", "interest": "0", "netAsset": "0"},
        "quoteAsset": {"asset": "USDC", "free": "0", "locked": "0",
                       "borrowed": "0", "interest": "0", "netAsset": "0"},
    }]})

    with pytest.raises(BinanceError, match="never been created"):
        await _adapter(session).get_isolated_account("BTC-USDC")


@pytest.mark.asyncio
async def test_get_fills_stops_when_a_page_is_short():
    session = FakeSession([{"id": 1}, {"id": 2}])

    fills = await _adapter(session).get_fills("BTC-USDC", limit=500)

    assert len(fills) == 2
    assert len(session.requests) == 1


@pytest.mark.asyncio
async def test_get_fills_starts_from_the_earliest_trade():
    # Unseeded, Binance returns only the MOST RECENT page, and asking for
    # trades after the newest one would return nothing at all.
    session = FakeSession([{"id": 1}])

    await _adapter(session).get_fills("BTC-USDC")

    assert session.params()["fromId"] == "0"


@pytest.mark.asyncio
async def test_get_fills_pages_forward_from_the_last_trade_id():
    session = FakeSession([{"id": 1}, {"id": 2}], [{"id": 3}])

    fills = await _adapter(session).get_fills("BTC-USDC", limit=2)

    assert [fill["id"] for fill in fills] == [1, 2, 3]
    assert session.params(1)["fromId"] == "3"


@pytest.mark.asyncio
async def test_get_fills_walks_the_whole_history_not_just_one_page():
    session = FakeSession([{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}], [{"id": 5}])

    fills = await _adapter(session).get_fills("BTC-USDC", limit=2)

    assert [fill["id"] for fill in fills] == [1, 2, 3, 4, 5]
    assert len(session.requests) == 3


@pytest.mark.asyncio
async def test_get_fills_stops_at_max_pages():
    session = FakeSession([{"id": 1}, {"id": 2}])   # always a full page

    fills = await _adapter(session).get_fills("BTC-USDC", limit=2, max_pages=3)

    assert len(session.requests) == 3
    assert len(fills) == 6


@pytest.mark.asyncio
async def test_get_fills_can_be_scoped_to_one_order():
    session = FakeSession([])

    await _adapter(session).get_fills("BTC-USDC", order_id="42")

    assert session.params()["orderId"] == "42"


@pytest.mark.asyncio
async def test_scoping_to_one_order_needs_no_paging_walk():
    session = FakeSession([{"id": 1}])

    await _adapter(session).get_fills("BTC-USDC", order_id="42")

    assert len(session.requests) == 1
    assert "fromId" not in session.params()


# ── Transfers ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transfer_in_moves_funds_from_spot_to_the_isolated_wallet():
    session = FakeSession({"tranId": 1})

    await _adapter(session).transfer_in("BTC-USDC", "USDC", "100")

    params = session.params()
    assert params["transFrom"] == "SPOT"
    assert params["transTo"] == "ISOLATED_MARGIN"
    assert params["symbol"] == "BTCUSDC"
    assert params["amount"] == "100"


@pytest.mark.asyncio
async def test_transfer_out_moves_funds_back_to_spot():
    session = FakeSession({"tranId": 1})

    await _adapter(session).transfer_out("BTC-USDC", "USDC", "100")

    params = session.params()
    assert params["transFrom"] == "ISOLATED_MARGIN"
    assert params["transTo"] == "SPOT"


# ── Market data ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_candles_are_requested_in_milliseconds_and_binances_interval():
    session = FakeSession([])

    await _adapter(session).get_product_candles("BTC-USDC", 1700000000, 1700003600, "SIX_HOUR")

    params = session.params()
    assert params["interval"] == "6h"
    assert params["startTime"] == "1700000000000"
    assert params["endTime"] == "1700003600000"
    assert params["limit"] == "1000"


@pytest.mark.asyncio
async def test_candles_come_back_in_the_canonical_shape():
    session = FakeSession([[1700000000000, "1.0", "3.0", "0.5", "2.0", "10.0", 1700003599999]])

    candles = await _adapter(session).get_product_candles("BTC-USDC", 0, 1, "ONE_HOUR")

    assert candles == [{
        "start": "1700000000", "open": "1.0", "high": "3.0",
        "low": "0.5", "close": "2.0", "volume": "10.0",
    }]


@pytest.mark.asyncio
async def test_public_market_data_is_sent_unsigned():
    session = FakeSession([])

    await _adapter(session).get_product_candles("BTC-USDC", 0, 1, "ONE_HOUR")

    _, url, headers = session.requests[0]
    assert "signature" not in url
    assert headers == {}


@pytest.mark.asyncio
async def test_best_bid_ask_is_shaped_like_coinbases_pricebooks():
    session = FakeSession({"bidPrice": "80000.0", "bidQty": "1.0", "askPrice": "80001.0", "askQty": "2.0"})

    result = await _adapter(session).get_best_bid_ask("BTC-USDC")

    book = result["pricebooks"][0]
    assert book["product_id"] == "BTC-USDC"
    assert book["bids"][0]["price"] == "80000.0"
    assert book["asks"][0]["price"] == "80001.0"


@pytest.mark.asyncio
async def test_best_bid_ask_fans_out_over_several_products():
    session = FakeSession({"bidPrice": "1", "bidQty": "1", "askPrice": "2", "askQty": "1"})

    result = await _adapter(session).get_best_bid_ask("BTC-USDC", "ETH-USDC")

    assert [book["product_id"] for book in result["pricebooks"]] == ["BTC-USDC", "ETH-USDC"]


@pytest.mark.asyncio
async def test_get_product_normalizes_exchange_info():
    session = FakeSession({"symbols": [_exchange_info_symbol()]})

    product = await _adapter(session).get_product("BTC-USDC")

    assert product["quote_increment"] == "0.01000000"


@pytest.mark.asyncio
async def test_an_unknown_product_raises():
    session = FakeSession({"symbols": []})

    with pytest.raises(BinanceError, match="Unknown product"):
        await _adapter(session).get_product("NOPE-USDC")


@pytest.mark.asyncio
async def test_market_trades_report_the_takers_side():
    session = FakeSession([
        {"id": 1, "price": "80000", "qty": "0.1", "time": 1700000000000, "isBuyerMaker": True},
        {"id": 2, "price": "80001", "qty": "0.2", "time": 1700000001000, "isBuyerMaker": False},
    ])

    trades = await _adapter(session).get_market_trades("BTC-USDC")

    assert [trade["side"] for trade in trades] == ["SELL", "BUY"]
    assert trades[0]["product_id"] == "BTC-USDC"


def test_binance_pages_candles_in_thousands():
    assert BinanceAdapter("k", "s").max_candles_per_request() == 1000


def test_the_adapter_names_its_exchange_for_cache_keying():
    assert BinanceAdapter("k", "s").name() == "binance"


def _klines(count: int, start_ms: int = 1700000000000, interval_ms: int = 3600000) -> list[list]:
    return [
        [start_ms + i * interval_ms, "1", "1", "1", "1", "1",
         start_ms + (i + 1) * interval_ms - 1]
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_a_window_wider_than_one_page_raises_instead_of_truncating():
    # klines returns the OLDEST `limit` candles, so the newest would be lost.
    session = FakeSession(_klines(1000))
    start   = 1700000000
    end     = start + 2000 * 3600   # asks for 2000 hourly candles

    with pytest.raises(BinanceError, match="truncated"):
        await _adapter(session).get_product_candles("BTC-USDC", start, end, "ONE_HOUR")


@pytest.mark.asyncio
async def test_a_full_page_that_covers_its_whole_window_is_not_an_error():
    # What ChunkedTimeRange produces: a chunk ending exactly on a boundary is
    # short by 1ms, not by a candle.
    session = FakeSession(_klines(1000))
    start   = 1700000000
    end     = start + 1000 * 3600

    candles = await _adapter(session).get_product_candles("BTC-USDC", start, end, "ONE_HOUR")

    assert len(candles) == 1000


# ── Errors ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_http_error_raises_with_the_status_and_body():
    session = FakeSession({"code": -1121, "msg": "Invalid symbol."}, status=400)

    with pytest.raises(BinanceError) as raised:
        await _adapter(session).get_accounts()

    assert raised.value.status == 400
    assert raised.value.raw["code"] == -1121


@pytest.mark.asyncio
async def test_an_error_code_on_a_200_response_still_raises():
    session = FakeSession({"code": -3020, "msg": "Transfer out amount exceeds max amount."})

    with pytest.raises(BinanceError):
        await _adapter(session).transfer_out("BTC-USDC", "USDC", "999999")


def test_binance_errors_are_exchange_errors():
    assert issubclass(BinanceError, ExchangeError)
