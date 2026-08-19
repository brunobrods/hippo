# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See @WORKFLOW.md for process rules (when to ask clarifying questions, when to spawn verification sub-agents).

## Project

Async Python toolkit for the Coinbase Advanced Trade REST API. Python 3.11+.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

Credentials go in `~/.coinbase/credentials.yaml` (outside the repo — shared across
every worktree of this checkout, never committed):
```yaml
api_key: "organizations/<org_id>/apiKeys/<key_id>"
api_secret: |
  -----BEGIN EC PRIVATE KEY-----
  ...
  -----END EC PRIVATE KEY-----
```
Keys created at <https://portal.cdp.coinbase.com/>. Loaded via `CredentialsFile` in
`coinbase/credentials_file.py`.

## Commands

Run every script below as a module, from the repo root — a direct script path
(`python coinbase/market_scanner.py`) fails with `ModuleNotFoundError: No module
named 'coinbase'`, since Python only adds the script's own directory to `sys.path`,
not the repo root.

```bash
# Market scanner — live snapshot
python -m coinbase.market_scanner
python -m coinbase.market_scanner --granularity ONE_HOUR --candles 200 --pairs BTC-USDC ETH-USDC
python -m coinbase.market_scanner --at 2026-05-22T14:30
python -m coinbase.market_scanner --week --step 6

# Adapter smoke test (live — requires trade:read_write key)
python -m coinbase.coinbase_adapter

# Tests
pytest
pytest tests/test_config.py   # single file
```

## Architecture

- `coinbase/coinbase_adapter.py` — `CoinbaseAdapter` async context manager. Auth via per-request ES256 JWTs (2-min TTL). All order amount fields must be **strings** — Coinbase rejects floats.
- `coinbase/market_scanner.py` — fetches OHLCV candles concurrently via `asyncio.gather`, computes RSI (EWM, 48-period) / MACD (12/26/9) / Bollinger (20-period ±2σ) using pandas, prints a snapshot table.
- `coinbase/strategy.py` — stub; `Strategy.onTimer()` not yet implemented.
- `coinbase/credentials_file.py` — `CredentialsFile` loads `~/.coinbase/credentials.yaml`. No sandbox exists for Coinbase Advanced Trade — all testing is live.

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

### Coinbase specifics

All price and size values sent to the API must be strings — Coinbase rejects
floats. Use a `SnappedPrice` (or equivalent) object to enforce rounding before
any order is placed.
