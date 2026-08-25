# Crypto Trading Toolkit — Coinbase & Binance

Async Python scripts for the
[Coinbase Advanced Trade REST API](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/)
and [Binance isolated margin](https://developers.binance.com/docs/margin_trading).

| File | Purpose |
|---|---|
| `coinbase/coinbase_adapter.py` | Async HTTP adapter for Coinbase spot — orders, accounts, market data |
| `binance/binance_adapter.py` | Async HTTP adapter for Binance **isolated margin** — the same surface, plus shorting and wallet transfers |
| `exchange/adapter.py` | The `ExchangeAdapter` Protocol both adapters satisfy, so the GA pipeline is exchange-agnostic |
| `exchange/selection.py` | Builds the adapter named by `data.exchange` in `config.yaml` |
| `coinbase/market_scanner.py` | Fetches OHLCV candles and prints RSI / MACD / Bollinger Bands (either exchange, via `--exchange`) |
| `*/credentials_file.py` | Loads **your** API credentials from `~/.coinbase/` or `~/.binance/` (outside the repo, never committed) |

### Canonical formats

Product IDs, granularities and timestamps stay in Coinbase's vocabulary
everywhere in this codebase — `"BTC-USDC"`, `"SIX_HOUR"`, UNIX **seconds**.
`BinanceAdapter` translates them on the wire (`BTCUSDC`, `6h`, milliseconds) and
normalizes klines back into Coinbase-shaped candle dicts, so one config and one
trained strategy drive either exchange.

### Long vs short on Binance

Longs are sent with `sideEffectType=NO_SIDE_EFFECT`: nothing is borrowed, no
interest accrues and there is no liquidation price, which makes them
economically identical to a Coinbase spot buy. Shorts must borrow the base
asset, so `market_short`/`limit_short` use `MARGIN_BUY` and the matching cover
uses `AUTO_REPAY`. That asymmetry is exactly what `trading_strategy.IsolatedMargin`
models.

> **Interest is not modelled.** Binance charges hourly interest on borrowed
> funds; `Ledger`/`Backtest` do not, so a short held for days scores better in
> the GA than it will trade live.

---

## Requirements

- Python **3.11+** (developed on 3.13)
- For Coinbase: a CDP API key with at least `trade:read` scope
  → create one at <https://portal.cdp.coinbase.com/>
- For Binance: an API key with **Enable Margin** and **Enable Spot & Margin
  Trading** → create one at <https://www.binance.com/en/my/settings/api-management>

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
```

Stay at the cloned repo's root — don't `cd` into the `coinbase/` package directory.
Every script below is run as a module (`python -m coinbase.xxx`) *from the repo root*,
since Python only resolves the `coinbase` package from there.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API credentials

Create `~/.coinbase/credentials.yaml` (outside the project directory, so it's never
committed and every git worktree/checkout of this repo shares the one file):

```yaml
# ~/.coinbase/credentials.yaml
api_key: "organizations/<org_id>/apiKeys/<key_id>"
api_secret: |
  -----BEGIN EC PRIVATE KEY-----
  ...
  -----END EC PRIVATE KEY-----
```

Keys are created at <https://portal.cdp.coinbase.com/>. `api_secret` is a YAML block
scalar (`|`) holding the PEM string across its real newlines, not an escaped one-liner.

For Binance, create `~/.binance/credentials.yaml` in the same way. Binance uses
HMAC-SHA256 with a plain shared secret, so both values are ordinary strings:

```yaml
# ~/.binance/credentials.yaml
api_key: "<64-char key>"
api_secret: "<64-char secret>"
```

### Choosing the exchange

`coinbase/ga/config.yaml` selects which one the GA trains and dry-runs against:

```yaml
data:
  exchange: "binance"   # coinbase | binance
  pair: "BTC-USDC"      # canonical either way — Binance sees BTCUSDC
```

---

## Usage

### Market scanner

```bash
python -m coinbase.market_scanner
```

Options:

```
--granularity  Candle interval (default: FIVE_MINUTE)
               ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE
               ONE_HOUR | TWO_HOUR | SIX_HOUR | ONE_DAY

--candles      Number of candles to fetch per pair
               (default: 100; max 300 on coinbase, 1000 on binance)

--pairs        Space-separated list of pairs (default: BTC/ETH/SOL/XRP/DOGE/ADA/AVAX/LINK vs USDC)

--exchange     coinbase | binance (default: coinbase)
```

Examples:

```bash
# 1-minute candles, last 60 bars
python -m coinbase.market_scanner --granularity ONE_MINUTE --candles 60

# Custom pair list
python -m coinbase.market_scanner --pairs BTC-USDC ETH-USDC SOL-USDC

# Same scan against Binance
python -m coinbase.market_scanner --exchange binance --pairs BTC-USDC ETH-USDC
```

### Adapter smoke tests

Each places a passive limit buy 2 % below best bid, confirms it is open, then
cancels it. **Both run against live** — neither exchange offers a sandbox for
these endpoints (Binance's spot testnet does not serve margin).

```bash
python -m coinbase.coinbase_adapter     # requires trade:read_write
python -m binance.binance_adapter       # requires margin trading enabled
python -m binance.binance_adapter BTC-USDT   # or any other isolated pair
```

The Binance smoke test needs **quote currency** in that pair's **isolated**
wallet. An isolated pair starts empty and cannot trade until you move money in,
and the test places a BUY — holding only the base asset is not enough:

```python
await adapter.transfer_in("BTC-USDT", "USDT", "50")
```

A pair that was never created still answers `get_isolated_account` with a
placeholder row whose `marginLevel` reads 999; the adapter raises on it rather
than reporting a healthy account that does not exist.

---

## Dependencies

| Package | Purpose |
|---|---|
| `aiohttp` | Async HTTP client |
| `PyJWT[crypto]` | ES256 JWT generation for Coinbase auth (Binance signs with `hmac` from the stdlib) |
| `pandas` | OHLCV calculations (RSI, MACD, Bollinger Bands) |

All pinned in `requirements.txt`.

---

## Security notes

- Credentials live in `~/.coinbase/credentials.yaml` and `~/.binance/credentials.yaml`,
  outside the repo entirely, so there's nothing credential-related for `.gitignore`
  to protect against.
- Neither exchange offers a usable sandbox here — Coinbase Advanced Trade has none,
  and Binance's spot testnet does not serve margin endpoints. Test with the smallest
  possible amounts, against the pair's `minNotional`.
- Binance shorts borrow real funds and can be liquidated. `IsolatedRisk` reads the
  exchange's own `marginLevel` and `liquidatePrice` — trust those over the
  backtest's 2×-entry approximation.
