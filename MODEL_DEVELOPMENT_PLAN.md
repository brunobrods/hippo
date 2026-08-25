# Model Development Plan

How to go from "run `main.py` once, eyeball the printed summary" to a
systematic search over parameters, with every run's config, result, and
trained model kept and comparable. Written 2026-08-18 against the GA module
at commit `acf3dbd`. Complements `ROADMAP.md` (which covers the path to live
execution) — this covers what happens *before* a strategy is trusted enough
to reach that roadmap.

## Where this sits relative to what exists

`TrainingRun.train()` in `coinbase/ga/main.py` already takes a plain
`raw_config: dict` and does fetch → split → train → evaluate → save. That's
the right seam: a sweep doesn't need to touch the GA engine, the evaluator,
or the market data processor — it just needs to build many config dicts and
call `TrainingRun(adapter, variant).train()` once per variant. Everything
below is about what wraps *around* that call, not what's inside it.

Four gaps make sweeping impractical today:

1. **No data cache.** `HistoricalCandles.raw()` hits live Coinbase on every
   run. There's no sandbox, so every sweep run pays a real network/rate-limit
   cost, even when the sweep only changes a GA hyperparameter that has
   nothing to do with the fetched candles.
2. **No run history.** `output.strategy_filepath` and `log_filepath` are
   fixed paths in `config.yaml` — each run overwrites the last. There is
   currently no way to look back at run #12 after run #13 finishes.
3. **No variance control.** The GA is stochastic (population init, crossover,
   mutation, tournament selection). A single run per config conflates "this
   parameter matters" with "this seed got lucky."
4. **No cross-pair / cross-window check.** Every run trains and tests on one
   window of one pair (`FET-USDC`, per current `config.yaml`). A config that
   looks good may just be fit to that one window — `ROADMAP.md` already flags
   this as a prerequisite before live money is at risk.

## 1. Separate the two costs

Data fetch is network-bound and keyed only by `(pair, granularity, start,
end)`. GA training is CPU-bound and keyed by everything else. Right now a
sweep over, say, `mutation_rate` would re-fetch identical candles ten times
for no reason.

**Add a local candle cache** — raw candles written to disk (parquet/CSV) the
first time a `(pair, granularity, start, end)` window is fetched, read from
disk on every subsequent run that needs the same window. This is the
highest-leverage piece of infrastructure here: it's what makes wide sweeps
over GA/strategy/column parameters cheap, and it's what makes a cross-pair
sweep (which *does* need fresh fetches, one per pair) bounded and predictable
instead of open-ended live-API traffic.

Indicator/column changes (different SMA periods, added delta columns) still
require recomputing `IndicatorFrame` — but that's pure pandas over cached
candles, not a network call.

## 2. What counts as "a parameter" — group them

| Group | Parameters | Effect of changing it |
|---|---|---|
| Data-defining | `pair`, `granularity`, `start_date`/`end_date`, `test_split` | Changes which candles get fetched — cache key |
| Feature-defining | `normalized_columns`, `delta_columns`, indicator periods (`sma_short/long/extra`, `rsi_period`, `macd_fast/slow/signal`) | Changes the shape of the indicator frame; interacts with `weight_keys` |
| Strategy-defining | `buy_threshold`, `sell_threshold`, `position_size_pct` | Changes trade timing/sizing for a fixed signal |
| Search-defining (GA) | `population_size`, `generations`, `mutation_rate`, `crossover_rate`, `tournament_size`, `elitism_count`, `mutation_sigma`, `seed` | Changes how well/reliably the GA finds a good genome, not the genome's meaning |

`weight_keys` must stay a subset of `normalized_columns` — already enforced
by `ValidatedWeightKeys` in `strategy_evaluator.py`, so a sweep generator just
needs to respect that invariant when it builds column-subset variants rather
than re-implement the check.

## 3. Sweep methodology

Full grid search over all of the above is combinatorially hopeless (five
indicator columns alone give 2⁵ subsets, before multiplying by periods,
thresholds, and GA hyperparameters). Use cheaper methods matched to each
group:

- **One-factor-at-a-time (OFAT)** for the strategy and GA-hyperparameter
  groups: hold everything else at the current `config.yaml` baseline, vary
  one parameter across a small grid (e.g. `mutation_rate` ∈
  {0.05, 0.1, 0.2, 0.4}), plot test-set `annualized_yield` against it. Cheap,
  and tells you sensitivity and direction per parameter.
- **Ablation, not grid, for columns**: baseline = all current
  `normalized_columns`. Run drop-one-column and add-one-candidate-column
  variants. The marginal change in test yield tells you what a column is
  actually contributing, without enumerating every subset.
- **Cross-pair sweep**: same fixed strategy/GA config, run across a handful
  of liquid pairs (this is where the candle cache pays for itself — N pairs
  means N fetches, not N × (number of other sweeps)). A config that only
  works on one pair is overfit to that pair, not a real signal.
- **Multiple seeds per config that matters**: rerun the shortlisted configs
  across ~5 seeds and report mean ± std of test yield, not a single number.
  A parameter's "effect" that's smaller than the seed-to-seed spread isn't a
  real effect.
- **Joint search only after OFAT narrows the field**: once OFAT/ablation
  identify the 2-3 parameters the result is actually sensitive to, a small
  random search over just those (jointly) catches interaction effects that
  OFAT misses, without paying for a full grid over everything.
- **Always score comparisons on the held-out test split** — already the
  existing design (`test_evaluator` never sees `train_df`). Don't let a sweep
  quietly start comparing train fitness across configs.
- **Reserve a third, walk-forward window, touched once**: after a sweep
  produces a shortlist, evaluate only the shortlist against a validation
  window that's been untouched by every prior comparison. Trying hundreds of
  configs against the same test split risks overfitting the *sweep* to that
  split, even though each individual GA run only sees train data — this is
  the walk-forward check `ROADMAP.md` already calls for, positioned here as
  the last step of a sweep rather than a one-off.

## 4. Capturing results

Give every run a **`run_id`**: a UTC timestamp plus a short hash of the fully
resolved config dict (the one actually passed to `TrainingRun`, not just
`config.yaml` — since a sweep overrides it in memory).

Persist, per run, under `experiments/<run_id>/`:
- the resolved config (so the run is exactly reproducible)
- `strategy.json` — the existing `StrategyJson` schema (weights,
  hyperparameters, performance) already has almost everything needed
- the GA per-generation log (existing `GaRunLog` format)

Add one **experiment index** — a flat, append-only CSV or JSON-lines file,
one row per run: `run_id`, `timestamp`, `git_commit`, `pair`, `granularity`,
window, the parameters that were varied, `annualized_yield`, `win_rate`,
`max_drawdown`, `seed`. This is the leaderboard: cheap to load into a
DataFrame and sort/filter, without opening every `strategy.json`. Record
`git rev-parse HEAD` per row — the GA engine or evaluator logic itself can
change between sweeps, and without a commit hash old results become silently
incomparable to new ones.

## 5. Model history

Stop overwriting `best_strategy.json`. Each run writes to
`experiments/<run_id>/strategy.json`; the index from §4 is the lookup table
across all of them — that alone gives full history for free.

Keep one small, explicit pointer file (e.g. `models/current_run_id.txt`) for
"whichever model is presently deployed/paper-traded." It only changes when
you deliberately promote a run, never as a side effect of training — so
`ROADMAP.md`'s dry-run/live-execution stage always has one unambiguous answer
to "what's running right now," while the full history stays queryable
separately.

## 6. Comparing results

A small loader over the experiment index (CSV/JSON-lines → DataFrame) is
enough to answer "what was the effect of parameter X" — group by the varied
parameter, compare mean/std test yield across its values. Plotting
`best_fitness`/`avg_fitness` from a run's GA log is a cheap sanity check that
the GA actually converged for that run, rather than reading the final number
in isolation.

## 7. Suggested sequencing

1. Candle cache — unblocks everything else being cheap enough to run wide.
2. `run_id` + `experiments/` layout + index file — unblocks history before
   any sweep produces results worth losing.
3. A thin `Sweep` driver that builds config variants (+ repeats per seed) and
   calls the existing `TrainingRun.train()` unchanged — no GA/evaluator
   changes needed for this part.
4. OFAT sweeps over the strategy and GA-hyperparameter groups.
5. Column ablation, then cross-pair sweep.
6. Shortlist → one walk-forward validation pass → promote via
   `models/current_run_id.txt`.

Each of 1-3 is infrastructure and independent of the others; 4-6 are the
actual experiments and depend on 1-3 being in place first. Suggest picking
one of 1-3 to scope properly (per `WORKFLOW.md`) as the next implementation
conversation, rather than building all of it before running a single sweep.
