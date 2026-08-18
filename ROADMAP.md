# Roadmap

Where this project stands and the next big steps, in dependency order. Written
2026-08-17 from the current state of the repo (last commit `beb0562`, "Feed
current position PnL into the GA strategy's signal").

## Where things stand

The **training pipeline is complete and tested**: `coinbase/ga/main.py` fetches
historical candles, computes indicators, evolves a `GaStrategy` genome against
an annualized-yield fitness function, backtests it out-of-sample, and saves
`best_strategy.json`. See `coinbase/ga/README.md` for the architecture.

The **live-trading side is a decision engine with no execution behind it**.
`coinbase/strategy.py`'s `LiveTradingRun.on_timer()` pulls a live market row
and account balance, asks a `Strategy` for a `Decision`, and updates its own
in-memory `Position` — but the comment on `on_timer()` says it plainly: *"it
decides but does not place a real order."* Nothing calls `on_timer()` on a
schedule, nothing sends the `Decision` to `CoinbaseAdapter`, and `pnl.py`'s
account-derived position tracking (`OpenPositions`/`EntryPrice`) is never
consulted by the live loop, which instead trusts only its own in-memory state.

That gap — decision without execution, and in-memory state without
reconciliation — is the shape of everything below.

## 0. Fix a bug blocking reliable runs

`coinbase/ga/main.py:112` hardcodes an absolute path to `config.yaml`:

```python
raw_config = ConfigFile("D:\project\scratches\coinbase\ga\config.yaml").raw()
```

Two problems: it's not a raw string (the backslash sequences aren't valid
escapes and will warn), and it points at the **main checkout**, not whichever
worktree `main.py` is actually run from — so from this worktree it would
silently load the wrong `config.yaml`. Every other entry point in the module
(`market_data_processor.py`'s smoke test) uses the relative path
`"coinbase/ga/config.yaml"`; `main.py` should match.

## 1. De-risk the strategy before it touches money

Nothing yet establishes that the GA strategy generalizes rather than
overfitting to one historical window on one pair (`FET-USDC`,
2025-06-01→2026-07-01 per `config.yaml`). Before wiring up execution:

- **Walk-forward validation** — retrain/re-evaluate across several rolling
  historical windows (not just one train/test split) and check performance is
  consistent, not a fluke of this particular period.
- **Cross-pair sanity check** — run the same pipeline against a couple of
  other pairs to see whether the learned weights are pair-specific noise or a
  real signal.
- **A dry-run / paper-trading mode** — drive `LiveTradingRun` off real-time
  data, log every `Decision` it would have made, but never call the adapter.
  Given Coinbase Advanced Trade has no sandbox, this is the only safe way to
  watch the live loop behave before it risks real capital.

## 2. Wire up real execution

Once the strategy has been paper-traded and trusted:

- **Send `Decision` to `CoinbaseAdapter`** — `on_timer()` computes a
  `Decision` and throws it away; connect `BUY`/`SELL` to
  `market_buy`/`market_sell` (or `limit_buy`/`limit_sell` with a
  `SnappedPrice`-style rounding object per `CLAUDE.md`'s Coinbase-specifics
  rule — all order amounts must be strings).
- **Reconcile position state from the account, not from memory** —
  `LiveTradingRun._position` is rebuilt purely from this process's own past
  decisions. A restart, a manual trade, or a partial fill desyncs it silently.
  `pnl.py`'s `OpenPositions`/`EntryPrice` already know how to derive real
  position state from account balances and fill history — the live loop
  should ask them, not trust its own memory.
- **Risk controls** — nothing currently caps position size, sets a stop-loss,
  or provides a kill-switch independent of the GA's learned thresholds. The
  GA optimizes for yield, not for bounding downside; that has to be enforced
  outside the genome.

## 3. Make it operable

- **A real scheduler** — nothing currently calls `on_timer()` periodically.
  Needs an actual loop (interval matched to the trained `granularity`) or
  integration with a task runner, plus handling for a missed/delayed tick.
- **Observability** — the live loop will run unattended against real money;
  it needs structured logging of every decision and order (not just
  `print`/console GA progress) and some alerting path for errors or
  unexpected states, not just `logger.info`.
- **End-to-end tests for the loop itself** — current tests (`test_strategy.py`,
  `test_trading_strategy.py`) cover `decide()` and the backtest engine in
  isolation, but nothing exercises the scheduling/execution wiring end-to-end
  against a fake adapter before it runs live.

## 4. Scale out (later)

- **Multi-pair / portfolio allocation** — today one genome is trained per
  pair via `config.yaml`; running several strategies concurrently raises
  capital-allocation and correlated-risk questions not addressed yet.
- **Live re-training cadence** — decide whether/how often a deployed
  strategy gets retrained on fresh data, and how to roll out a new genome
  without discontinuity in an open position.

---

Suggested next conversation: pick one item from §0–1 (the bug fix is a
five-minute change; the dry-run mode is the highest-leverage next feature)
and scope it properly per `WORKFLOW.md` before implementing.
