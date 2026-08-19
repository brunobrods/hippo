import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any

from coinbase.coinbase_adapter import CoinbaseAdapter
from coinbase.ga.config import ConfigFile
from coinbase.ga.main import TrainingRun, TrainingSummary

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
            target = target[key]
        target[keys[-1]] = self._value
        return result


@dataclass(frozen=True)
class SweepPoint:
    axis_path:  str
    axis_value: Any
    seed:       int
    raw_config: dict[str, Any]


class SweepPlan:
    # One-factor-at-a-time by construction: each axis value is applied to the
    # unmodified base_config, never combined with another axis's value, so
    # every point isolates the effect of a single parameter. Joint/combined
    # sweeps are deliberately out of scope here — see MODEL_DEVELOPMENT_PLAN.md
    # step 3, which reserves those for a later, narrower random search.
    def __init__(self, base_config: dict[str, Any], axes: tuple[SweepAxis, ...], seeds: tuple[int, ...]) -> None:
        self._base_config = base_config
        self._axes        = axes
        self._seeds       = seeds

    def points(self) -> tuple[SweepPoint, ...]:
        scratch_strategy_path = self._scratch_strategy_path()
        points: list[SweepPoint] = []
        for axis in self._axes:
            for value in axis.values:
                with_axis = DottedPathOverride(self._base_config, axis.path, value).applied()
                for seed in self._seeds:
                    config = DottedPathOverride(with_axis, SEED_PATH, seed).applied()
                    config = DottedPathOverride(config, STRATEGY_FILEPATH_PATH, scratch_strategy_path).applied()
                    points.append(SweepPoint(axis.path, value, seed, config))
        return tuple(points)

    def ensure_scratch_directory(self) -> None:
        os.makedirs(os.path.dirname(self._scratch_strategy_path()), exist_ok=True)

    def _scratch_strategy_path(self) -> str:
        experiments_dir = self._base_config["output"]["experiments_dir"]
        return os.path.join(experiments_dir, "_sweep_scratch", "best_strategy.json")


# ── Console progress ─────────────────────────────────────────────────────

class ConsoleSweepProgress:
    def __init__(self, total: int) -> None:
        self._total = total

    def point_started(self, index: int, point: SweepPoint) -> None:
        print(f"[{index}/{self._total}] {point.axis_path}={point.axis_value} seed={point.seed}")

    def point_finished(self, summary: TrainingSummary) -> None:
        print(summary.as_text())
        print()


# ── Orchestration ────────────────────────────────────────────────────────

class Sweep:
    # Sequential, not concurrent: points sharing a data window (the common
    # case — only strategy/GA hyperparameters vary across most axes) rely on
    # the first point populating market_data_processor's candle disk cache
    # before the rest read it. Running points concurrently would race several
    # cold fetches of the same window instead of reusing one.
    def __init__(self, adapter: CoinbaseAdapter, points: tuple[SweepPoint, ...], progress: ConsoleSweepProgress) -> None:
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


# ── Entry point ────────────────────────────────────────────────────────
# Run:  python coinbase/ga/sweep.py
# Trains one full GA strategy per (axis value, seed) point in sweep.yaml,
# holding every other parameter at base_config's value (one-factor-at-a-time).
# Each point is an ordinary TrainingRun — same experiments/<run_id>/ history
# and experiments/index.csv leaderboard row as a single `main.py` run, so
# comparing sweep results needs no sweep-specific tooling.

async def _main() -> None:
    from coinbase.credentials_file import CredentialsFile

    credentials  = CredentialsFile().credentials()
    sweep_config = SweepConfigFile(ConfigFile("coinbase/ga/sweep.yaml").raw())
    base_config  = ConfigFile(sweep_config.base_config_path()).raw()
    plan         = SweepPlan(base_config, sweep_config.axes(), sweep_config.seeds())
    plan.ensure_scratch_directory()
    points       = plan.points()

    print(f"Sweeping {len(points)} points across {len(sweep_config.axes())} axes, "
          f"{len(sweep_config.seeds())} seeds each")

    async with CoinbaseAdapter(credentials.api_key, credentials.api_secret) as adapter:
        await Sweep(adapter, points, ConsoleSweepProgress(len(points))).run()


if __name__ == "__main__":
    asyncio.run(_main())
