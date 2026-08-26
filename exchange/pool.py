"""
One adapter per venue, shared by every algo trading on it.
-----------------------------------------------------------

Opening an adapter per algo would open a connection pool and a credential
signer per algo, and — worse — give each its own fetch budget. The candle
semaphore inside HistoricalCandles bounds one fetch object; N algos fetching
concurrently would run N of those, so the bound that keeps requests inside
their timeout would be N times looser than it reads.

A lane owns both the adapter and the one semaphore every fetch on that venue
shares, so the budget is per exchange, which is the thing that actually rate
limits.
"""

import asyncio
from typing import Any, Optional

from exchange.adapter import ExchangeAdapter
from exchange.selection import ConfiguredExchange


# ── Lane ───────────────────────────────────────────────────────────────

class ExchangeLane:
    def __init__(self, name: str, max_concurrent_requests: int = 8) -> None:
        self._name                    = name
        self._max_concurrent_requests = max_concurrent_requests
        self._adapter: Optional[ExchangeAdapter] = None
        self._limit: Optional[asyncio.Semaphore] = None

    def name(self) -> str:
        return self._name

    async def open(self) -> None:
        adapter = ConfiguredExchange({"data": {"exchange": self._name}}).adapter()
        await adapter.connect()
        self._adapter = adapter

    async def close(self) -> None:
        if self._adapter is not None:
            await self._adapter.close()
            self._adapter = None

    def adapter(self) -> ExchangeAdapter:
        if self._adapter is None:
            raise ValueError(f"exchange lane {self._name!r} is not open")
        return self._adapter

    # Built on first use rather than in open(), so a lane constructed outside a
    # running loop does not bind a semaphore to the wrong one.
    def limit(self) -> asyncio.Semaphore:
        if self._limit is None:
            self._limit = asyncio.Semaphore(self._max_concurrent_requests)
        return self._limit


# ── Pool ───────────────────────────────────────────────────────────────

class ExchangePool:
    def __init__(self, names: tuple[str, ...], max_concurrent_requests: int = 8) -> None:
        self._names                   = names
        self._max_concurrent_requests = max_concurrent_requests
        self._lanes: dict[str, ExchangeLane] = {}

    # A lane that fails to open — a missing credentials file for the second
    # venue, say — must not strand the sessions already connected for the
    # first. __aexit__ never runs if __aenter__ raises, so the unwind is here.
    async def __aenter__(self) -> "ExchangePool":
        try:
            for name in dict.fromkeys(self._names):
                lane = ExchangeLane(name, self._max_concurrent_requests)
                await lane.open()
                self._lanes[name] = lane
        except Exception:
            await self._close_all()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        failure = await self._close_all()
        if failure is not None:
            raise failure

    # Every lane closes even if one raises, so a failing adapter cannot leak the
    # others' sessions; the first failure is handed back rather than thrown, so
    # the caller decides whether it outranks whatever it was already unwinding.
    async def _close_all(self) -> Optional[BaseException]:
        failure: Optional[BaseException] = None
        for lane in self._lanes.values():
            try:
                await lane.close()
            except Exception as exc:
                failure = failure or exc
        self._lanes = {}
        return failure

    def lane(self, name: str) -> ExchangeLane:
        if name not in self._lanes:
            raise KeyError(f"no open exchange lane for {name!r}")
        return self._lanes[name]
