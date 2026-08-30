import aiohttp
import pytest

from coinbase.coinbase_adapter import CoinbaseAdapter, CoinbaseError, CoinbaseProduct


# ── Test double ──────────────────────────────────────────────────────
# Stands in for aiohttp.ClientResponse — only what _parse touches.

class _FakeResponse:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self._status  = status
        self._payload = payload
        self._text    = text

    @property
    def status(self) -> int:
        return self._status

    @property
    def ok(self) -> bool:
        return 200 <= self._status < 400

    async def json(self):
        if self._payload is None:
            raise aiohttp.ContentTypeError(None, ())
        return self._payload

    async def text(self) -> str:
        return self._text


# ── _parse ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_returns_the_decoded_body_on_success():
    resp = _FakeResponse(200, payload={"candles": [1, 2, 3]})
    assert await CoinbaseAdapter._parse(resp) == {"candles": [1, 2, 3]}


@pytest.mark.asyncio
async def test_parse_raises_coinbase_error_carrying_the_status():
    resp = _FakeResponse(400, payload={"error": "INVALID_ARGUMENT"})
    with pytest.raises(CoinbaseError) as caught:
        await CoinbaseAdapter._parse(resp)
    assert caught.value.status == 400
    assert caught.value.raw == {"error": "INVALID_ARGUMENT"}


@pytest.mark.asyncio
async def test_parse_raises_coinbase_error_when_the_error_body_is_not_json():
    # a rate-limited response from an edge proxy comes back as plain text, and
    # decoding it first would surface aiohttp's ContentTypeError instead of the
    # status code that says the request was throttled
    resp = _FakeResponse(429, payload=None, text="Too Many Requests")
    with pytest.raises(CoinbaseError) as caught:
        await CoinbaseAdapter._parse(resp)
    assert caught.value.status == 429
    assert "Too Many Requests" in str(caught.value.raw)


@pytest.mark.asyncio
async def test_parse_truncates_a_huge_non_json_error_body():
    resp = _FakeResponse(502, payload=None, text="x" * 10_000)
    with pytest.raises(CoinbaseError) as caught:
        await CoinbaseAdapter._parse(resp)
    assert len(caught.value.raw["body"]) == 500


# ── CoinbaseProduct ──────────────────────────────────────────────────

def _product(**overrides) -> dict:
    raw = {
        "product_id":                  "BTC-USDC",
        "base_currency_id":            "BTC",
        "quote_currency_id":           "USDC",
        "price":                       "50000",
        "volume_24h":                  "100",
        "price_percentage_change_24h": "2.5",
        "base_increment":              "0.00000001",
        "quote_increment":             "0.01",
        "base_min_size":               "0.0001",
        "quote_min_size":              "1",
        "status":                      "online",
        "trading_disabled":            False,
    }
    raw.update(overrides)
    return raw


def test_coinbase_product_keeps_the_canonical_dashed_product_id():
    assert CoinbaseProduct(_product()).normalized()["product_id"] == "BTC-USDC"


# Coinbase reports volume_24h in BASE units while Binance's quoteVolume is
# already in quote — unconverted, a liquidity filter would compare a BTC count
# against a USDT count and rank on nothing at all.
def test_coinbase_product_reports_quote_volume_as_base_volume_times_price():
    product = CoinbaseProduct(_product(volume_24h="100", price="50000")).normalized()
    assert product["volume_24h_quote"] == pytest.approx(5_000_000.0)


def test_coinbase_product_is_not_tradable_when_trading_is_disabled():
    assert CoinbaseProduct(_product(trading_disabled=True)).normalized()["tradable"] is False


def test_coinbase_product_is_not_tradable_when_it_is_offline():
    assert CoinbaseProduct(_product(status="offline")).normalized()["tradable"] is False


def test_coinbase_product_is_not_tradable_in_cancel_only_mode():
    assert CoinbaseProduct(_product(cancel_only=True)).normalized()["tradable"] is False


# Coinbase Advanced Trade has no borrow, so no strategy can be shorted here.
def test_coinbase_product_can_never_short():
    assert CoinbaseProduct(_product()).normalized()["can_short"] is False


def test_coinbase_product_tolerates_a_missing_price_without_raising():
    assert CoinbaseProduct(_product(price="")).normalized()["volume_24h_quote"] == 0.0


def test_coinbase_product_matches_the_binance_normalized_key_set():
    product = CoinbaseProduct(_product()).normalized()
    assert {
        "product_id", "base_currency", "quote_currency", "quote_increment",
        "base_increment", "base_min_size", "min_market_funds", "status",
        "tradable", "can_long", "can_short", "volume_24h_quote",
        "price_change_24h_pct",
    } <= set(product)
