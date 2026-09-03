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

# Paper trading — ONE tick per invocation, then exits (for Task Scheduler).
# Acts on the last CLOSED candle and refuses to act on the same candle twice,
# so scheduling it more often than the granularity is safe and free.
python -m coinbase.ga.paper_trading
./scripts/register_paper_task.ps1                    # register, every 30 min
./scripts/register_paper_task.ps1 -Remove            # unregister

# Pair screener — rank a venue's whole universe by how far it moves, net of costs
python -m coinbase.pair_screener
python -m coinbase.pair_screener --exchange binance --quote USDT --days 180 --top 25
python -m coinbase.pair_screener --exchange binance --require-short --min-volume 1e7
python -m coinbase.pair_screener --metric net_maker_bps   # see the maker/taker gap

# Cross-sectional selector — one book, picks which pair to hold and which way
python -m coinbase.ga.pair_selector --pairs FET-USDT ETH-USDT --exchange binance
python -m coinbase.ga.pair_selector --from-shortlist binance --fee-bps 10
python -m coinbase.ga.pair_selector --pairs FET-USDT --no-fees   # gross, like Backtest

# Selector paper trading — ONE tick per invocation, then exits (Task Scheduler)
python -m coinbase.ga.selection_paper --pairs FET-USDT ETH-USDT --exchange binance
python -m coinbase.ga.selection_paper --from-shortlist binance

# Tests
pytest
pytest tests/test_config.py   # single file
```

## Paper trading

`coinbase/ga/dry_run.py` holds a process open and sleeps between ticks;
`coinbase/ga/paper_trading.py` runs one tick and exits, which is what makes it
schedulable. Two properties make the scheduled form safe:

- **State persists.** Balance and any open position are written to
  `paper_state.json` after every tick and reloaded on the next, so a reboot or
  a closed lid does not silently reset the book to flat. `dry_run.py` builds a
  fresh `Ledger` on every launch and does restart flat — that is fine for a
  process you watch, wrong for a scheduled one.
- **Idempotent per candle.** A tick records `last_candle_start` and refuses to
  act on that candle again, so the task can be scheduled far more often than
  the granularity. Every 30 minutes against `SIX_HOUR` candles means 11 of 12
  runs exit as no-ops. This is deliberately preferred over aligning a
  local-time schedule to UTC boundaries: no DST arithmetic, and a machine
  asleep at the boundary acts late rather than skipping the candle.

Decisions are taken on the last **closed** candle (`ClosedMarketRow`), matching
what `Backtest` sees — `LiveMarketRow`'s last row is the still-forming candle,
whose close is just the current price and whose high/low are partial.

The `paper:` section of `config.yaml` sets the market independently of `data:`,
so a genome trained on Coinbase BTC-USDC can be papered against Binance
BTC-USDT without disturbing training config.

**Known gap:** `LiveMarketRow` min-max normalizes against its trailing window
while the GA trained against the full training window's range, so `norm_*`
columns do not mean quite the same thing live as in scoring. Deferred
deliberately; it affects any live or paper run.

## Pair screening

`coinbase/pair_screener.py` answers "which pairs move enough to be worth
trading, after costs?" — discovering the venue's universe via the
`ProductCatalog` half of the adapter Protocol, measuring 180 days of `ONE_DAY`
candles per pair, and ranking on movement net of the round-trip cost floor.

Three things about it are load-bearing rather than stylistic:

- **The window is snapped to whole UTC days.** `CandleCacheKey` interpolates the
  raw `start`/`end` ints into the filename, so an unsnapped `time.time()` mints
  a unique key every second — a guaranteed cache miss and a full cold refetch of
  every pair on every run. Snapped, the first run of a day fetches and the rest
  are disk reads. Snapping alone does **not** make the window closed-only:
  Binance treats klines' `endTime` as inclusive of a candle's *open*, so
  `PairCandles._closed` drops the still-forming candle. Leaving it in moved
  `atr_pct` by ~11% on real BTC data.
- **One semaphore is shared across every pair.** `HistoricalCandles` builds a
  private `Semaphore(8)` when none is passed, so N pairs would put N×8 requests
  in flight. The screener threads `ExchangeLane.limit()` into all of them.
- **Ranking defaults to `net_taker_bps`, not `net_maker_bps`.** The maker figure
  credits the whole spread back, so it *rises* with the spread and promotes
  illiquid pairs — it once ranked MANTRA above pairs that objectively moved more,
  purely for having the widest spread. `net_maker_bps` stays as a column.

**Read `required_accuracy`, not the volatility.** Movement is necessary and not
sufficient: the screener proves a pair moved, never that the direction was
forecastable. `required_accuracy` converts a move and a cost floor into the
direction accuracy a strategy would need — `p = (target + cost + move) / 2·move`.
It lands near 70% on the good pairs, against published daily-direction models in
the low 50s. `viable` requires both clearing the cost floor **and**
`required_accuracy <= 1.0`, because a pair can move more than it costs to trade
while still being arithmetically incapable of the target (BTC-USDT is exactly
this case at 1%/day).

Fee rates are read from the venue per account (`/sapi/v1/asset/tradeFee`,
`/api/v3/brokerage/transaction_summary`), never hardcoded — tiers move with
volume, and the measured rates here differed sharply from the published base
tier. The output says whether a rate was measured or is a fallback.

**Handoff to the GA.** The shortlist JSON's flat `pairs` array is the `values:`
list for a sweep axis, which `DottedPathOverride` already supports:

```yaml
axes:
  - path: "data.pair"
    values: ["DEXE-USDT", "ZEC-USDT"]     # paste from the shortlist
```

`Sweep` builds one adapter from `data.exchange` **outside** the point loop, so
set `data.exchange: "binance"` in `config.yaml` before sweeping Binance pairs —
otherwise every point trains against Coinbase, where those pairs do not exist.
Then `python -m coinbase.ga.results --group-by pair` compares them; read the
`std` column, not just `mean`.

**Maker execution is not modelled.** Post-only exists on both adapters
(Binance `LIMIT_MAKER`, Coinbase `postOnly`) and nothing calls it, because
`Ledger.apply` fills every decision unconditionally at the candle close — there
is no resting-order state and no way to express a non-fill. `MakerTakerFee` in
`paper_trading.py` prices the two sides differently but is not yet wired into
`paper_engine`. Until a fill model exists, treat `net_maker_bps` as a best case
that assumes the order filled, which is exactly what a resting order does not
guarantee.

## Wider-market context (`index_z`)

`market_data.index_pairs` sets an equal-weight basket standing in for "the
market". With it set, every frame gains an **`index_z`** column: the pair's
return minus the basket's, scored against its own recent spread. So +2 means
this pair is diverging from the market by more than it normally does, and 0
means it is going along with it.

Equal weight, not volume weight — the basket is meant to be the menu a
strategy picked from, so every item gets the same say. The z-score rather than
the raw excess because 2% of excess is ordinary on a wild pair and
extraordinary on a quiet one; scaling by each pair's own spread makes the
number mean the same thing everywhere. `MarketIndex` is keyed by **timestamp**,
not row order, because the pairs it averages list at different dates and drop
candles independently.

**Off by default, and that is load-bearing.** Turning it on changes the frame
every genome is scored on. To use it: set the basket, add `index_z` to *both*
`market_data.normalized_columns` and `strategy.weight_keys`, then retrain.
Genomes saved before it existed carry no such weight — `BackfilledGenome` runs
them with it at zero (the honest reading: they could not have used it) and
logs which keys it filled. Training still fails loudly on an unknown key,
because there a missing weight is a bug rather than history.

## Cross-sectional pair selection

`coinbase/ga/pair_selector.py` asks the question one level above a genome:
given N genomes each watching its own pair, which pair is worth holding right
now, and in which direction? One position at a time, taken in whichever pair
has the strongest conviction, held until that pair's own exit threshold fires.
No rotation mid-hold — a round trip costs real basis points.

Two things it does that `Backtest` does not, deliberately:

- **Fees are charged**, via `Ledger.charge`. A selector re-entering on every
  exit trades far more than a single-pair strategy. `Trade.profit()` stays
  gross so `experiments/index.csv` comparisons across the whole history remain
  valid; the selector charges on top.
- **Direction accuracy is measured** — the number `required_accuracy` from the
  screener is a bar for. Reported with a Wilson 95% interval, because "63%"
  from 11 trades is indistinguishable from a coin and the bar sits inside the
  interval. Trades the window cut short unwind at their own entry price and are
  **excluded** from both rates: closed by the calendar, not by the strategy, and
  scoring one as a miss invents an outcome the data never produced.

**Cross-pair ranking is the weakest link.** `signal_score` sums `norm_*`
columns min-max scaled inside each pair's *own* window, so a raw 0.8 on one
pair and 0.8 on another are not the same quantity. `Conviction` measures each
score against its own genome's trigger — `(score − buy_threshold) / (1 −
buy_threshold)` — which removes the differing-thresholds problem and much of
the differing-scale one. "Much of" is the honest phrase; do not read a good
result here as proof the ranking is sound.

**Always read the benchmark and the attribution, not the headline.** The report
prints equal-weight buy-and-hold over the identical window, and flags when one
pair carries more than 60% of net profit. On the first real run the selector
returned +60% against −11.9% for holding everything equally — but FET-USDT
alone accounted for 134% of net profit, and stripping it left −20.7%. That is
one pair moving, not selection skill.

**Paper trading it** — `coinbase/ga/selection_paper.py`, one tick per
invocation like `paper_trading.py`, with the same two safety properties. The
state file adds one field that matters: **which pair the position is in**. A
size and an entry price restored against the wrong market would be marked at
an unrelated price and exited using a different pair's genome. A tick acts on
the oldest candle *every* pair has closed — one lagging pair holds the whole
tick rather than letting a stale score into the ranking. A single tick can
both exit one pair and enter another, so the outcome reports `exited` and
`entered` separately rather than one action field that would hide a leg.

`BestRunPerPair` picks each pair's highest-scoring run out of
`experiments/index.csv`, filtered to the **same granularity** — a genome
trained on `THIRTY_MINUTE` candles reads its indicators off a different clock.
This is in-sample selection twice over (the run scored best on the data it is
then selected over), so treat any result as an upper bound.

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
- **Liquidation depends on the collateral ratio, not a fixed multiple of entry.** `IsolatedMargin` computes `(collateral + size * entry) / (1.05 * size)`, matching Binance: verified live, a short of 0.00019022 BTC entered at 78,266 against 57.93 USDT of quote assets reported `marginLevel` 3.890 and `liquidatePrice` 290,022.60, putting the trigger at exactly 1.05. A flat 2x-entry rule is right only for a short whose notional is the *entire* wallet. `IsolatedRisk.liquidation_price()` reads the exchange's own number when trading live.
- `Ledger`/`Backtest` charge **no borrow interest**, but Binance does, hourly. Shorts held for days score better in the GA than they trade live.

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
