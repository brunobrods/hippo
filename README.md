# Coinbase Advanced Trade Toolkit

Async Python scripts for the [Coinbase Advanced Trade REST API](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/).

| File | Purpose |
|---|---|
| `coinbase_adapter.py` | Async HTTP adapter — orders, accounts, market data |
| `market_scanner.py` | Fetches OHLCV candles and prints RSI / MACD / Bollinger Bands |
| `credentials.py` | **Your** API credentials (not committed to version control) |

---

## Requirements

- Python **3.11+** (developed on 3.13)
- A Coinbase CDP API key with at least `trade:read` scope  
  → create one at <https://portal.cdp.coinbase.com/>

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd coinbase
```

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

Create `credentials.py` in the project directory (it is intentionally excluded from version control):

```python
# credentials.py
api_key    = "organizations/<org_id>/apiKeys/<key_id>"
api_secret = "-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
```

Keys are created at <https://portal.cdp.coinbase.com/>.  
The secret must be the raw PEM string with literal `\n` newlines.

---

## Usage

### Market scanner

```bash
python market_scanner.py
```

Options:

```
--granularity  Candle interval (default: FIVE_MINUTE)
               ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE
               ONE_HOUR | TWO_HOUR | SIX_HOUR | ONE_DAY

--candles      Number of candles to fetch per pair (default: 100, max: 300)

--pairs        Space-separated list of pairs (default: BTC/ETH/SOL/XRP/DOGE/ADA/AVAX/LINK vs USDC)
```

Examples:

```bash
# 1-minute candles, last 60 bars
python market_scanner.py --granularity ONE_MINUTE --candles 60

# Custom pair list
python market_scanner.py --pairs BTC-USDC ETH-USDC SOL-USDC
```

### Adapter smoke test

Places a passive limit buy 2 % below best bid, confirms it is open, then cancels it.  
**Runs against live Coinbase** — requires a key with `trade:read_write`.

```bash
python coinbase_adapter.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `aiohttp` | Async HTTP client |
| `PyJWT[crypto]` | ES256 JWT generation for Coinbase auth |
| `pandas` | OHLCV calculations (RSI, MACD, Bollinger Bands) |

All pinned in `requirements.txt`.

---

## Security notes

- `credentials.py` must **never** be committed. Add it to `.gitignore`:
  ```
  credentials.py
  .venv/
  __pycache__/
  ```
- Coinbase Advanced Trade has no sandbox. Test with the smallest possible amounts.
