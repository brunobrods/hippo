# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See @WORKFLOW.md for process rules (when to ask clarifying questions, when to spawn verification sub-agents).

## Project

Async Python toolkit for the Coinbase Advanced Trade REST API and Binance
isolated margin. Python 3.11+.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
git config core.hooksPath .githooks  # see "Branching" below — not automatic
```

Credentials go in `~/.coinbase/credentials.yaml` and `~/.binance/credentials.yaml`
(outside the repo — shared across every worktree of this checkout, never committed):
```yaml
api_key: "organizations/<org_id>/apiKeys/<key_id>"
api_secret: |
  -----BEGIN EC PRIVATE KEY-----
  ...
  -----END EC PRIVATE KEY-----
```
Keys created at <https://portal.cdp.coinbase.com/>. Loaded via `CredentialsFile` in
`coinbase/credentials_file.py`.

Binance uses HMAC-SHA256 with a plain shared secret, so both fields are ordinary
strings:
```yaml
api_key:    "<64-char key>"
api_secret: "<64-char secret>"
```
Keys created at <https://www.binance.com/en/my/settings/api-management> with
**Enable Margin** + **Enable Spot & Margin Trading**. Loaded via `CredentialsFile`
in `binance/credentials_file.py`; both share `exchange/credentials_file.py`.

## Branching

**Never commit directly to `master`.** Work lands through a branch and a pull
request — create one with `git switch -c <name>` before the first commit.

`.githooks/pre-commit` enforces this, and worktrees share the common `.git`
directory so the single hook covers every worktree at once. Two caveats worth
knowing:

- It is **not** active until `git config core.hooksPath .githooks` has been run
  in that clone (see Setup) — git will not pick up a tracked hook by itself.
- A tracked hook only exists on branches that contain it, so it cannot protect
  a branch whose own tree lacks the file.

## Commands

Run every script below as a module, from the repo root — a direct script path
(`python coinbase/market_scanner.py`) fails with `ModuleNotFoundError: No module
named 'coinbase'`, since Python only adds the script's own directory to `sys.path`,
not the repo root.

```bash
# Market scanner — live snapshot (--exchange coinbase|binance)
python -m coinbase.market_scanner
python -m coinbase.market_scanner --exchange binance --pairs BTC-USDC
python -m coinbase.market_scanner --granularity ONE_HOUR --candles 200 --pairs BTC-USDC ETH-USDC
python -m coinbase.market_scanner --at 2026-05-22T14:30
python -m coinbase.market_scanner --week --step 6

# Adapter smoke tests (live — no sandbox on either exchange)
python -m coinbase.coinbase_adapter   # requires trade:read_write key
python -m binance.binance_adapter            # defaults to BTC-USDC
python -m binance.binance_adapter BTC-USDT   # any isolated pair
# both need a margin-enabled key and QUOTE currency in that pair's isolated wallet

# Tests
pytest
pytest tests/test_config.py   # single file
```

## Architecture

Two exchanges behind one contract. `exchange/adapter.py` defines the
`ExchangeAdapter` Protocol; the GA pipeline, live/paper runs and scanner are all
typed against it, never against a concrete adapter.

**Canonical formats** — product IDs, granularities and timestamps stay in
Coinbase's vocabulary everywhere (`"BTC-USDC"`, `"SIX_HOUR"`, UNIX seconds).
Each adapter translates on the wire. Never leak an exchange's own dialect
(`BTCUSDC`, `6h`, milliseconds) past its adapter boundary.

- `exchange/adapter.py` — `ExchangeAdapter` Protocol + `ExchangeError` base. `CoinbaseError` and `BinanceError` both subclass it, so shared code catches the base.
- `exchange/selection.py` — `ConfiguredExchange` builds the adapter named by `data.exchange` in `config.yaml`. Adapter imports are deferred inside the method to avoid an import cycle.
- `exchange/credentials_file.py` — shared YAML parsing; the per-exchange modules own only the default path.
- `binance/binance_adapter.py` — `BinanceAdapter`, mirroring `CoinbaseAdapter`'s surface over `/sapi/v1/margin/*`. Auth via HMAC-SHA256 over the exact encoded query string. Every order carries `isIsolated=TRUE`. Longs use `sideEffectType=NO_SIDE_EFFECT` (nothing borrowed — economically a spot buy); `market_short`/`limit_short` use `MARGIN_BUY` and covers use `AUTO_REPAY`. Amounts are strings here too.
- `coinbase/coinbase_adapter.py` — `CoinbaseAdapter` async context manager. Auth via per-request ES256 JWTs (2-min TTL). All order amount fields must be **strings** — Coinbase rejects floats.
- `coinbase/market_scanner.py` — fetches OHLCV candles concurrently via `asyncio.gather`, computes RSI (EWM, 48-period) / MACD (12/26/9) / Bollinger (20-period ±2σ) using pandas, prints a snapshot table.
- `coinbase/strategy.py` — stub; `Strategy.onTimer()` not yet implemented.
- `coinbase/credentials_file.py` — `CredentialsFile` loads `~/.coinbase/credentials.yaml`. No sandbox exists for Coinbase Advanced Trade, and Binance's spot testnet does not serve margin endpoints — all testing is live.

### Margin caveats

- Isolated margin balances are **per pair**, not account-wide. `AccountBalance` takes an optional `product_id` to scope the lookup; accounts without that key (Coinbase's) match regardless.
- An isolated pair starts empty — `transfer_in` before trading it, or orders fail on insufficient balance.
- **Shorts have a minimum borrow size.** Below it Binance returns `-11007 "Exceeding the maximum borrowable limit"`, which names the opposite bound and is badly misleading. Verified live: a short borrowing 0.00006 BTC was rejected while `maxBorrowable` reported 0.00495 BTC; 0.00019 BTC succeeded. The minimum is not published in `exchangeInfo`, so a strategy sizing shorts off a small balance can emit orders that can never fill.
- **You cannot always sell back what you just bought.** Buy fees are taken in the base asset, so a buy sized near `minNotional` leaves you holding slightly less than you ordered; flooring that to `stepSize` can drop the sell under `minNotional` and fail with `-1013 Filter failure: NOTIONAL`. Size entries with headroom over the floor, not at it.
- **Binance's real liquidation price is not the backtest's 2x entry.** Verified live: a short entered at ~78,266 with 42 USDT of collateral reported `liquidatePrice` 290,022 — the model's 2x rule (156,532) only corresponds to a short whose notional is the *entire* isolated wallet. Read `IsolatedRisk.liquidation_price()` for the real number.
- `Ledger`/`Backtest` charge **no borrow interest**, but Binance does, hourly. Shorts held for days score better in the GA than they trade live. `IsolatedRisk` exposes the exchange's real `marginLevel`/`liquidatePrice`; prefer those over `IsolatedMargin`'s 2×-entry approximation.

## Code Style

This codebase follows Yegor Bugayenko's object-oriented philosophy ("Elegant Objects").

### Everything is an object

No standalone utility functions. Every piece of behaviour lives inside a class.
Wrap primitives and data in objects that know how to act on themselves.

```python
# Wrong — utility function
def snap_to_increment(value: float, increment: float) -> str: ...

# Right — object that knows how to snap itself
class SnappedPrice:
    def __init__(self, value: float, increment: float) -> None:
        self._value     = value
        self._increment = increment

    def as_string(self) -> str: ...
```

### Constructors do no work

`__init__` only assigns parameters to private instance variables. No computation,
no validation, no I/O, no method calls. All work is deferred to methods.

```python
# Wrong
def __init__(self, raw: list[dict]) -> None:
    self._closes = pd.Series([float(c["close"]) for c in raw])  # work!

# Right
def __init__(self, raw: list[dict]) -> None:
    self._raw = raw  # store only

@functools.cached_property
def _closes(self) -> pd.Series:
    return pd.Series([float(c["close"]) for c in self._raw])
```

### Command-Query Separation

Every method either **does a job** (returns `None`, causes a side effect) or
**returns a value** (pure, no side effects) — never both.

```python
# Wrong — both returns a value AND logs (side effect)
async def place_order(self, ...) -> dict:
    result = await self._post(...)
    logger.info("placed %s", result)   # side effect mixed in
    return result                       # also returns

# Right — split into two
async def place(self, ...) -> None:         # does the job
    self._result = await self._post(...)

def placed_order(self) -> dict:             # returns the value
    return self._result
```

### Small classes — few public methods

Each class has one responsibility and ideally ≤ 5 public methods.
If a class needs more, extract a new object.

### Caching

Use `functools.cached_property` for expensive sync computations that should be
evaluated lazily and only once. For async results, cache with a private
attribute initialised to `None`:

```python
import functools

class Rsi:
    def __init__(self, closes: pd.Series, period: int = 48) -> None:
        self._closes = closes
        self._period = period

    @functools.cached_property
    def value(self) -> float:           # computed once, reused forever
        delta = self._closes.diff()
        ...

class CoinbaseCandles:
    def __init__(self, session: aiohttp.ClientSession, product_id: str) -> None:
        self._session    = session
        self._product_id = product_id
        self._cache: Optional[list[dict]] = None   # async cache slot

    async def raw(self) -> list[dict]:
        if self._cache is None:
            self._cache = await self._fetch()
        return self._cache
```

### Composition over inheritance

Build complex behaviour by wrapping simpler objects, not by subclassing.
Use `typing.Protocol` to define contracts between objects.

```python
from typing import Protocol

class Indicator(Protocol):
    def value(self) -> float: ...
```

### Type hints

Every method signature must have parameter and return type hints.
Use `from typing import Any, Optional` for compatibility.
Lowercase generics: `list[str]`, `dict[str, Any]`, `tuple[float, float]`.

### Naming

Name classes as **nouns** (what they are), not verbs (what they do):
`JwtToken`, `CoinbaseCandles`, `BollingerBands` — not `TokenBuilder`, `FetchCandles`.

### Formatting conventions

Section dividers between logical groups:
```python
# ── Section Name ───────────────────────────────────────────────────────
```

Align `=` in `__init__` bodies and constant blocks:
```python
self._api_key    = api_key
self._api_secret = api_secret
self._timeout    = aiohttp.ClientTimeout(total=timeout)
```

Use `logger = logging.getLogger(__name__)` at module level; `print` only in
entry points and smoke tests.

### Other non-negotiables

- **Async everywhere.** All I/O via `async`/`await`. No blocking calls.
- **Raise, don't swallow.** Propagate exceptions. Follow the `CoinbaseError`
  pattern for domain errors — HTTP status + raw body.
- **No None returns.** Raise an exception instead of returning `None` to signal
  absence or failure.
- **No docstrings.** Names and types carry the meaning.

### Tests

Write `pytest` tests for every new class. Place them in `tests/` mirroring the
package (e.g. `tests/test_coinbase_adapter.py`). Mock `aiohttp.ClientSession`
for unit tests; use `pytest-asyncio` for async test methods.

### Exchange specifics

All price and size values sent to either API must be strings — both reject
floats. Snap to the venue's increment before placing an order, using a snapping
**object**: `SnappedValue` for Binance (`tickSize` / `stepSize`, whose own decimal
text carries the precision). Coinbase's `snap_to_increment` is still a free
function — it is the "Wrong — utility function" example above, and ROADMAP tracks
replacing it with a `SnappedPrice`; follow `SnappedValue`'s shape for new code
rather than copying it. Binance additionally rejects anything under the pair's
`minNotional`.
