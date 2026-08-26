import aiohttp
import pytest

from coinbase.coinbase_adapter import CoinbaseAdapter, CoinbaseError


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
