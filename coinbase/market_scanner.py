"""
Market Scanner
--------------
Fetches OHLCV candles for a set of Coinbase pairs concurrently and
prints a snapshot table of technical indicators:

  RSI      — momentum oscillator; >70 overbought, <30 oversold
  MACD     — 12/26/9 EMA crossover line and signal line
  Bollinger — 20-period ±2σ bands; flags price above/below bands
  Volume   — average of the last 5 candles (normalised to base currency)

Usage:
    python market_scanner.py                        # live snapshot
    python market_scanner.py --at 2026-05-22        # snapshot at a past date
    python market_scanner.py --at 2026-05-22T14:30  # snapshot at a past datetime
    python market_scanner.py --week                 # daily snapshots for last 7 days
    python market_scanner.py --week --step 6        # every 6 hours for last 7 days
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from coinbase.coinbase_adapter import CoinbaseAdapter
from coinbase.credentials_file import CredentialsFile


# ── Configuration ──────────────────────────────────────────────────────

PAIRS = [
    # "BTC-USDC",
    # "ETH-USDC",
    # "SOL-USDC",
    # "XRP-USDC",
    # "DOGE-USDC",
    # "ADA-USDC",
    # "AVAX-USDC",
    # "LINK-USDC",
    "FET-USDC",
    # "COMP-USDC",
    # "AMP-USDC",
]

# Seconds per candle for each supported granularity
GRANULARITY_SECONDS = {
    "ONE_MINUTE":      60,
    "FIVE_MINUTE":    300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR":      3600,
    "TWO_HOUR":      7200,
    "SIX_HOUR":     21600,
    "ONE_DAY":      86400,
}

RSI_PERIOD  = 48
BB_PERIOD   = 20
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9


# ── Indicator calculations ─────────────────────────────────────────────

def rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("inf"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


def macd(closes: pd.Series) -> tuple[float, float]:
    ema_fast    = closes.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow    = closes.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def bollinger(closes: pd.Series, period: int = BB_PERIOD) -> tuple[float, float]:
    sma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return float((sma + 2 * std).iloc[-1]), float((sma - 2 * std).iloc[-1])


# ── Per-pair result ────────────────────────────────────────────────────

@dataclass
class Snapshot:
    pair:      str
    price:     float = 0.0
    rsi:       float = 0.0
    macd_line: float = 0.0
    macd_sig:  float = 0.0
    bb_upper:  float = 0.0
    bb_lower:  float = 0.0
    vol_avg:   float = 0.0
    error:     Optional[str] = None


async def fetch_snapshot(
    adapter: CoinbaseAdapter,
    pair: str,
    granularity: str,
    n_candles: int,
    ref_time: int,
) -> Snapshot:
    candle_secs = GRANULARITY_SECONDS[granularity]
    start       = ref_time - n_candles * candle_secs

    try:
        raw     = await adapter.get_product_candles(pair, start, ref_time, granularity)
        candles = sorted(raw, key=lambda c: int(c["start"]))  # oldest → newest
        closes  = pd.Series([float(c["close"])  for c in candles])
        volumes = pd.Series([float(c["volume"]) for c in candles])

        bb_up, bb_lo   = bollinger(closes)
        macd_l, macd_s = macd(closes)

        return Snapshot(
            pair      = pair,
            price     = float(closes.iloc[-1]),
            rsi       = rsi(closes),
            macd_line = macd_l,
            macd_sig  = macd_s,
            bb_upper  = bb_up,
            bb_lower  = bb_lo,
            vol_avg   = float(volumes.iloc[-5:].mean()),
        )
    except Exception as exc:
        return Snapshot(pair=pair, error=str(exc))


# ── Display ────────────────────────────────────────────────────────────

def _rsi_label(value: float) -> str:
    if value > 70:
        return f"{value:5.1f} OB"
    if value < 30:
        return f"{value:5.1f} OS"
    return f"{value:5.1f}   "


def _bb_position(price: float, upper: float, lower: float) -> str:
    if price > upper:
        return "ABOVE"
    if price < lower:
        return "BELOW"
    bb_range = upper - lower
    if bb_range == 0:
        return "MID"
    pct = (price - lower) / bb_range * 100
    return f"{pct:4.0f}%"


def print_table(snapshots: list[Snapshot], granularity: str, ref_time: int) -> None:
    ts  = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ref_time))
    hdr = (
        f"{'Pair':<14} {'Price':>13} {f'RSI-{RSI_PERIOD}':>9}  "
        f"{'MACD':>9} {'Signal':>9}  "
        f"{'BB Upper':>12} {'BB Lower':>12}  "
        f"{'BB Pos':>6}  {'Vol(5c avg)':>12}"
    )
    print(f"\nMarket snapshot  [{granularity}]  {ts}\n")
    print(hdr)
    print("─" * len(hdr))

    for s in snapshots:
        if s.error:
            print(f"{s.pair:<14}  ERROR: {s.error}")
            continue

        print(
            f"{s.pair:<14} {s.price:>13.4f} {_rsi_label(s.rsi):>9}  "
            f"{s.macd_line:>9.4f} {s.macd_sig:>9.4f}  "
            f"{s.bb_upper:>12.4f} {s.bb_lower:>12.4f}  "
            f"{_bb_position(s.price, s.bb_upper, s.bb_lower):>6}  "
            f"{s.vol_avg:>12.4f}"
        )

    print()


# ── Time helpers ───────────────────────────────────────────────────────

def parse_at(value: str) -> int:
    """Parse an ISO date or datetime string (UTC) into a unix timestamp."""
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM"
    )


def build_reference_times(args: argparse.Namespace) -> list[int]:
    """Return the ordered list of unix timestamps to run the scanner at."""
    if args.at:
        return [parse_at(args.at)]

    now = int(time.time())

    if args.week:
        step_secs  = args.step * 3600
        week_secs  = 7 * 24 * 3600
        n_steps    = week_secs // step_secs
        return [now - (n_steps - i) * step_secs for i in range(n_steps + 1)]

    return [now]


# ── Entry point ────────────────────────────────────────────────────────

def parse_args(args_in) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coinbase market scanner with RSI/MACD/BB")
    p.add_argument(
        "--granularity", "-g",
        default="THIRTY_MINUTE",
        choices=list(GRANULARITY_SECONDS),
        help="Candle interval (default: THIRTY_MINUTE)",
    )
    p.add_argument(
        "--candles", "-n",
        type=int,
        default=100,
        help="Number of candles to fetch per pair (default: 100, max: 300)",
    )
    p.add_argument(
        "--pairs", "-p",
        nargs="+",
        default=PAIRS,
        metavar="PAIR",
        help="Space-separated list of pairs, e.g. BTC-USDC ETH-USDC",
    )

    time_group = p.add_mutually_exclusive_group()
    time_group.add_argument(
        "--at",
        metavar="DATETIME",
        help="Historical snapshot at a specific time, e.g. 2026-05-22 or 2026-05-22T14:30",
    )
    time_group.add_argument(
        "--week",
        action="store_true",
        help="Print daily snapshots across the last 7 days",
    )
    p.add_argument(
        "--step",
        type=int,
        default=24,
        metavar="HOURS",
        help="Hours between snapshots when using --week (default: 24)",
    )
    return p.parse_args(args_in)


async def main(args_in) -> None:
    args        = parse_args(args_in)
    ref_times   = build_reference_times(args)
    credentials = CredentialsFile().credentials()

    async with CoinbaseAdapter(credentials.api_key, credentials.api_secret) as adapter:
        for ref_time in ref_times:
            tasks     = [fetch_snapshot(adapter, p, args.granularity, args.candles, ref_time) for p in args.pairs]
            snapshots = await asyncio.gather(*tasks)
            print_table(snapshots, args.granularity, ref_time)


if __name__ == "__main__":
    asyncio.run(main(["--week", "--step", "1"]))
