import argparse
import asyncio
import copy
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from coinbase.ga.config import ConfigFile
from coinbase.ga.main import TrainingRun, TrainingSummary
from coinbase.ga.market_data_processor import (
    CachedHistoricalCandles,
    CandleCacheFile,
    CandleCacheKey,
    HistoricalCandles,
    MarketDataConfig,
)
from coinbase.ga.strategy_output import OutputConfigFile, ParentDirectory
from exchange.adapter import ExchangeAdapter

logger = logging.getLogger(__name__)

# genetic_algorithm.seed is overridden per seed repeat regardless of which
# axis is varying, so every axis value gets trained once per configured seed.
SEED_PATH = "genetic_algorithm.seed"

# TrainingRun.train() unconditionally overwrites output.strategy_filepath —
# the single "current" strategy dry_run.py reloads. Every sweep point gets
# this redirected to a shared scratch file instead, so a sweep never clobbers
# that canonical file with whatever point happened to train last; each
# point's real record is still its own experiments/<run_id>/strategy.json.
STRATEGY_FILEPATH_PATH = "output.strategy_filepath"


# ── Spec ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepAxis:
    path:   str
    values: tuple[Any, ...]


class SweepConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def base_config_path(self) -> str:
        return self._raw["base_config"]

    def seeds(self) -> tuple[int, ...]:
        return tuple(self._raw["seeds"])

    # Dotted paths pinned for every point of this sweep, applied to base_config
    # before any axis is. Not an axis: nothing here varies, so it does not
    # multiply the point count and does not break the OFAT property below.
    # Optional — a sweep file without a `fixed:` section pins nothing.
    def fixed(self) -> tuple[tuple[str, Any], ...]:
        return tuple((self._raw.get("fixed") or {}).items())

    def axes(self) -> tuple[SweepAxis, ...]:
        return tuple(
            SweepAxis(path=axis["path"], values=tuple(axis["values"]))
            for axis in self._raw["axes"]
        )


# ── Config variants ──────────────────────────────────────────────────────

class DottedPathOverride:
    def __init__(self, raw_config: dict[str, Any], path: str, value: Any) -> None:
        self._raw_config = raw_config
        self._path       = path
        self._value      = value

    def applied(self) -> dict[str, Any]:
        result = copy.deepcopy(self._raw_config)
        target = result
        keys   = self._path.split(".")
        for key in keys[:-1]:
            # A YAML mapping whose every child is commented out (e.g. a fully
            # optional `output:` section) parses to None, not {} — treat a
            # *present* key with that value as an empty section so overriding
            # one of its paths doesn't crash on a section nobody has
            # customized. A genuinely absent key still raises KeyError below,
            # same as before.
            if target[key] is None:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = self._value
        return result


class FixedOverrides:
    # Folds a sweep file's `fixed:` paths onto the base config, in order, so a
    # later path can refine one an earlier one set. The result is what both the
    # SweepPlan and the adapter are built from — pinning `data.exchange` here
    # has to reach ConfiguredExchange too, or a sweep would train Binance
    # configs against a Coinbase adapter.
    def __init__(self, raw_config: dict[str, Any], overrides: tuple[tuple[str, Any], ...]) -> None:
        self._raw_config = raw_config
        self._overrides  = overrides

    def applied(self) -> dict[str, Any]:
        # Copied even when nothing is pinned, so the result is never an alias
        # of the caller's config — DottedPathOverride makes the same promise,
        # and a `fixed:`-less sweep must not be the one case that breaks it.
        result = copy.deepcopy(self._raw_config)
        for path, value in self._overrides:
            result = DottedPathOverride(result, path, value).applied()
        return result


@dataclass(frozen=True)
class SweepPoint:
    axis_path:  str
    axis_value: Any
    seed:       int
    raw_config: dict[str, Any]


class ScratchStrategyPath:
    # TrainingRun writes output.strategy_filepath and immediately reads it back
    # to verify the round trip, so any two runs sharing the path would have each
    # reading the other's strategy — or a half-written one. Two things can
    # collide, and the name has to separate both:
    #
    #   process_id  keeps concurrent SWEEPS apart, which ExperimentIndex's
    #               atomic header creation already assumes is supported.
    #   ordinal     keeps concurrent POINTS of one sweep apart, which
    #               ParallelSweep makes possible — the plan is built in the
    #               parent, so every point would otherwise inherit one pid.
    def __init__(self, experiments_dir: str, process_id: int, ordinal: int) -> None:
        self._experiments_dir = experiments_dir
        self._process_id      = process_id
        self._ordinal         = ordinal

    def value(self) -> str:
        return os.path.join(
            self._experiments_dir, "_sweep_scratch",
            f"best_strategy_{self._process_id}_{self._ordinal}.json",
        )


class SweepPlan:
    # One-factor-at-a-time by construction: each axis value is applied to the
    # unmodified base_config, never combined with another axis's value, so
    # every point isolates the effect of a single parameter. Joint/combined
    # sweeps are deliberately out of scope here — see MODEL_DEVELOPMENT_PLAN.md
    # step 3, which reserves those for a later, narrower random search.
    #
    # process_id names the scratch file this plan's points write through; it
    # defaults to the running process, and is injectable so a test can assert
    # the path without depending on its own pid.
    def __init__(
        self,
        base_config: dict[str, Any],
        axes: tuple[SweepAxis, ...],
        seeds: tuple[int, ...],
        process_id: Optional[int] = None,
    ) -> None:
        self._base_config = base_config
        self._axes        = axes
        self._seeds       = seeds
        self._process_id  = process_id

    def points(self) -> tuple[SweepPoint, ...]:
        points: list[SweepPoint] = []
        for axis in self._axes:
            for value in axis.values:
                with_axis = DottedPathOverride(self._base_config, axis.path, value).applied()
                for seed in self._seeds:
                    config = DottedPathOverride(with_axis, SEED_PATH, seed).applied()
                    config = DottedPathOverride(
                        config, STRATEGY_FILEPATH_PATH, self._scratch_strategy_path(len(points)),
                    ).applied()
                    points.append(SweepPoint(axis.path, value, seed, config))
        return tuple(points)

    def ensure_scratch_directory(self) -> None:
        ParentDirectory(self._scratch_strategy_path(0)).ensure()

    # One path per point AND per sweep process. Every run writes its best
    # strategy here and reads it straight back to verify the saved JSON reloads
    # into an identical backtest, so any sharing has a run verifying whichever
    # write landed last. Sequential sweeps are unaffected; they just no longer
    # reuse one file.
    def _scratch_strategy_path(self, ordinal: int) -> str:
        experiments_dir = OutputConfigFile(self._base_config).config().experiments_dir
        owner = os.getpid() if self._process_id is None else self._process_id
        return ScratchStrategyPath(experiments_dir, owner, ordinal).value()


# ── Console progress ─────────────────────────────────────────────────────

class ConsoleSweepProgress:
    def __init__(self, total: int) -> None:
        self._total = total

    def point_started(self, index: int, point: SweepPoint) -> None:
        print(f"[{index}/{self._total}] {point.axis_path}={point.axis_value} seed={point.seed}")

    def point_finished(self, summary: TrainingSummary) -> None:
        print(summary.as_text())
        print()

    # Parallel points finish out of order, so there is no "started" line to
    # pair a result with — the count and the text carry it instead.
    def point_completed(self, done: int, text: str) -> None:
        print(f"[{done}/{self._total}] done")
        print(text)
        print(flush=True)


# ── Orchestration ────────────────────────────────────────────────────────

class Sweep:
    # Sequential, not concurrent: points sharing a data window (the common
    # case — only strategy/GA hyperparameters vary across most axes) rely on
    # the first point populating market_data_processor's candle disk cache
    # before the rest read it. Running points concurrently would race several
    # cold fetches of the same window instead of reusing one.
    def __init__(self, adapter: ExchangeAdapter, points: tuple[SweepPoint, ...], progress: ConsoleSweepProgress) -> None:
        self._adapter  = adapter
        self._points   = points
        self._progress = progress

    async def run(self) -> tuple[TrainingSummary, ...]:
        summaries = []
        for index, point in enumerate(self._points, start=1):
            self._progress.point_started(index, point)
            summary = await TrainingRun(self._adapter, point.raw_config).train()
            self._progress.point_finished(summary)
            summaries.append(summary)
        return tuple(summaries)


# Runs ONE point, in its own process, with its own adapter.
#
# A callable object rather than a bare function so it stays in the codebase's
# idiom while remaining picklable — ProcessPoolExecutor has to ship it and its
# argument across a process boundary, which rules out closures and lambdas.
# The aiohttp session is built inside __call__ for the same reason: a
# connection pool cannot be pickled, let alone shared between processes.
class SweepWorker:
    def __call__(self, point: SweepPoint) -> tuple[str, str]:
        from exchange.selection import ConfiguredExchange

        async def train() -> tuple[str, str]:
            async with ConfiguredExchange(point.raw_config).adapter() as adapter:
                summary = await TrainingRun(adapter, point.raw_config).train()
            return summary.run_id(), summary.as_text()

        return asyncio.run(train())


# True parallelism, not asyncio: a training run is ~5000 pandas backtests, so
# it is CPU-bound and the GIL makes concurrent coroutines strictly slower than
# a loop. Processes sidestep it. Measured at 70 s a run, 120 runs go from
# ~2h20m sequential to roughly 20 minutes across 8 workers.
#
# Two pieces of shared state had to be made safe first: each point now writes
# its own scratch strategy file (SweepPlan._scratch_strategy_path), and
# index.csv appends take an IndexLock. The candle cache was already safe —
# CandleCacheFile writes tmp-then-rename.
class ParallelSweep:
    def __init__(
        self,
        points: tuple[SweepPoint, ...],
        progress: "ConsoleSweepProgress",
        workers: int,
    ) -> None:
        self._points   = points
        self._progress = progress
        self._workers  = workers

    async def run(self) -> tuple[str, ...]:
        loop    = asyncio.get_running_loop()
        worker  = SweepWorker()
        results: list[str] = []
        with ProcessPoolExecutor(max_workers=self._workers) as pool:
            tasks = [loop.run_in_executor(pool, worker, point) for point in self._points]
            for done, task in enumerate(asyncio.as_completed(tasks), start=1):
                try:
                    _, text = await task
                    results.append(text)
                except Exception as exc:
                    # One point failing must not lose the other 119. The pair
                    # is not recoverable from the future, so the message is
                    # what identifies it.
                    logger.error("a sweep point failed: %s", exc)
                    results.append(f"FAILED: {exc}")
                self._progress.point_completed(done, results[-1])
        return tuple(results)


# ── Cache warming ──────────────────────────────────────────────────────

# Fetched once, in the parent, before any worker starts.
#
# Workers do not share memory, so N of them starting cold would each fetch the
# same windows — N identical cold fetches racing instead of one, which is the
# very thing Sweep's sequential comment says it was avoiding. Warming first
# keeps the parallel path as cache-friendly as the sequential one.
class WarmedCache:
    def __init__(self, points: tuple[SweepPoint, ...]) -> None:
        self._points = points

    async def run(self) -> int:
        from exchange.selection import ConfiguredExchange

        wanted: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
        for point in self._points:
            market = MarketDataConfig(point.raw_config)
            window = market.window()
            key    = (point.raw_config["data"]["exchange"], window.pair,
                      window.granularity, window.start, window.end)
            wanted.setdefault(key, point.raw_config)
            for pair in market.index_pairs():
                wanted.setdefault(
                    (point.raw_config["data"]["exchange"], pair,
                     window.granularity, window.start, window.end),
                    point.raw_config,
                )

        first = self._points[0].raw_config
        async with ConfiguredExchange(first).adapter() as adapter:
            limit  = asyncio.Semaphore(8)
            market = MarketDataConfig(first)
            await asyncio.gather(*(
                CachedHistoricalCandles(
                    HistoricalCandles(adapter, pair, granularity, start, end, limit=limit),
                    CandleCacheFile(
                        market.cache_dir(),
                        CandleCacheKey(exchange, pair, granularity, start, end),
                    ),
                ).raw()
                for exchange, pair, granularity, start, end in wanted
            ))
        return len(wanted)


# ── Entry point ────────────────────────────────────────────────────────
# Run:  python coinbase/ga/sweep.py [sweep_config_path]
# Trains one full GA strategy per (axis value, seed) point in sweep.yaml (or
# the sweep config passed as the first CLI argument), holding every other
# parameter at base_config's value (one-factor-at-a-time). Each point is an
# ordinary TrainingRun — same experiments/<run_id>/ history and
# experiments/index.csv leaderboard row as a single `main.py` run, so
# comparing sweep results needs no sweep-specific tooling. Passing a
# narrower sweep config (e.g. a subset of axes/values) is also how a large
# sweep gets split into batches that finish within one process's lifetime —
# already-recorded points in experiments/ are unaffected either way.

async def _main(argv: list[str]) -> None:
    from exchange.selection import ConfiguredExchange

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train one GA strategy per sweep point")
    parser.add_argument("sweep_config", nargs="?", default="coinbase/ga/sweep.yaml")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Training processes to run at once (default: 1, sequential). "
             "A run is CPU-bound, so this scales with cores, not with IO.",
    )
    args = parser.parse_args(argv)

    sweep_config = SweepConfigFile(ConfigFile(args.sweep_config).raw())
    # `fixed:` is folded on before any axis, and the result is what BOTH the
    # plan and the adapter are built from — pinning data.exchange has to reach
    # ConfiguredExchange too, or a sweep trains Binance configs against a
    # Coinbase adapter.
    fixed        = sweep_config.fixed()
    base_config  = FixedOverrides(ConfigFile(sweep_config.base_config_path()).raw(), fixed).applied()
    plan         = SweepPlan(base_config, sweep_config.axes(), sweep_config.seeds())
    plan.ensure_scratch_directory()
    points       = plan.points()

    print(f"Sweeping {len(points)} points across {len(sweep_config.axes())} axes, "
          f"{len(sweep_config.seeds())} seeds each, {args.workers} at a time")
    if fixed:
        print("Fixed for every point: " + ", ".join(f"{path}={value}" for path, value in fixed))

    if args.workers <= 1:
        async with ConfiguredExchange(base_config).adapter() as adapter:
            await Sweep(adapter, points, ConsoleSweepProgress(len(points))).run()
        return

    # Warm first: workers share no memory, so starting them cold would race N
    # identical fetches of the same windows.
    windows = await WarmedCache(points).run()
    print(f"Cache warmed for {windows} windows; starting {args.workers} workers")
    await ParallelSweep(points, ConsoleSweepProgress(len(points)), args.workers).run()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
