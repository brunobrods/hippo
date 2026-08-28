import asyncio

import pytest

from exchange.pool import ExchangeLane, ExchangePool


# ── Test doubles ─────────────────────────────────────────────────────

class _FakeAdapter:
    def __init__(self, name: str, fail_on_close: bool = False) -> None:
        self._name         = name
        self._fail_on_close = fail_on_close
        self.connected     = False
        self.closed        = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True
        if self._fail_on_close:
            raise RuntimeError(f"{self._name} refused to close")

    def name(self) -> str:
        return self._name


def _patched_pool(monkeypatch, adapters: dict) -> None:
    class _Selection:
        def __init__(self, raw) -> None:
            self._name = raw["data"]["exchange"]

        def adapter(self):
            return adapters[self._name]

    monkeypatch.setattr("exchange.pool.ConfiguredExchange", _Selection)


# ── ExchangeLane ─────────────────────────────────────────────────────

def test_an_unopened_lane_raises_rather_than_handing_back_none():
    with pytest.raises(ValueError):
        ExchangeLane("coinbase").adapter()


@pytest.mark.asyncio
async def test_a_lane_shares_one_semaphore_across_every_call():
    lane = ExchangeLane("coinbase", max_concurrent_requests=4)

    assert lane.limit() is lane.limit()


@pytest.mark.asyncio
async def test_closing_a_lane_releases_its_adapter(monkeypatch):
    adapter = _FakeAdapter("coinbase")
    _patched_pool(monkeypatch, {"coinbase": adapter})
    lane = ExchangeLane("coinbase")

    await lane.open()
    assert lane.adapter() is adapter

    await lane.close()
    assert adapter.closed is True
    with pytest.raises(ValueError):
        lane.adapter()


# ── ExchangePool ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_pool_opens_one_lane_per_distinct_venue(monkeypatch):
    adapters = {"coinbase": _FakeAdapter("coinbase"), "binance": _FakeAdapter("binance")}
    _patched_pool(monkeypatch, adapters)

    # Three algos, two venues -> two adapters, not three.
    async with ExchangePool(("coinbase", "binance", "coinbase")) as pool:
        assert pool.lane("coinbase").adapter() is adapters["coinbase"]
        assert pool.lane("binance").adapter() is adapters["binance"]

    assert adapters["coinbase"].closed and adapters["binance"].closed


@pytest.mark.asyncio
async def test_every_algo_on_one_venue_shares_that_venues_budget(monkeypatch):
    _patched_pool(monkeypatch, {"coinbase": _FakeAdapter("coinbase")})

    async with ExchangePool(("coinbase", "coinbase")) as pool:
        # Both algos resolve to the same lane, so to the same semaphore — the
        # bound is per venue, not per algo.
        assert pool.lane("coinbase").limit() is pool.lane("coinbase").limit()


@pytest.mark.asyncio
async def test_asking_for_an_unopened_venue_raises(monkeypatch):
    _patched_pool(monkeypatch, {"coinbase": _FakeAdapter("coinbase")})

    async with ExchangePool(("coinbase",)) as pool:
        with pytest.raises(KeyError):
            pool.lane("binance")


@pytest.mark.asyncio
async def test_a_lane_failing_to_open_does_not_strand_the_ones_already_open(monkeypatch):
    # __aexit__ never runs if __aenter__ raises, so a second venue with missing
    # credentials would otherwise leak the first venue's aiohttp session.
    opened = _FakeAdapter("coinbase")

    class _Selection:
        def __init__(self, raw) -> None:
            self._name = raw["data"]["exchange"]

        def adapter(self):
            if self._name == "binance":
                raise ValueError("no binance credentials")
            return opened

    monkeypatch.setattr("exchange.pool.ConfiguredExchange", _Selection)

    with pytest.raises(ValueError):
        async with ExchangePool(("coinbase", "binance")):
            pass

    assert opened.closed is True


@pytest.mark.asyncio
async def test_one_lane_failing_to_close_does_not_leak_the_others(monkeypatch):
    adapters = {
        "coinbase": _FakeAdapter("coinbase", fail_on_close=True),
        "binance":  _FakeAdapter("binance"),
    }
    _patched_pool(monkeypatch, adapters)

    with pytest.raises(RuntimeError):
        async with ExchangePool(("coinbase", "binance")):
            pass

    # The failure propagates, but only after every session was shut down.
    assert adapters["binance"].closed is True
