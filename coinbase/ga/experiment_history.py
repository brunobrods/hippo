import asyncio
import csv
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from coinbase.ga.ga_engine import GaConfig
from coinbase.ga.strategy_evaluator import StrategyConfig
from coinbase.ga.strategy_output import ParentDirectory

# One row per training run — a flat, greppable/pandas-loadable leaderboard so
# comparing runs doesn't require opening every experiments/<run_id>/strategy.json.
_INDEX_FIELDS = (
    "run_id", "started_at", "git_commit",
    "pair", "granularity", "start_date", "end_date", "test_split",
    "seed", "population_size", "generations", "mutation_rate", "crossover_rate",
    "tournament_size", "elitism_count", "mutation_sigma",
    "buy_threshold", "sell_threshold", "position_size_pct", "unwind_at_entry_price",
    "gross_profit", "annualized_yield", "total_trades", "win_rate", "max_drawdown",
)


# ── Identity ───────────────────────────────────────────────────────────

class GitCommitHash:
    async def value(self) -> str:
        process = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"git rev-parse HEAD failed: {stderr.decode().strip()}")
        return stdout.decode().strip()


class RunId:
    # Microsecond-precision timestamp plus a random suffix — not a hash of the
    # config — because two runs of the *same* config (e.g. a sweep re-running a
    # fixed seed) starting within the same wall-clock second must still get
    # distinct ids, or they'd silently interleave into the same experiments/
    # subdirectory instead of each getting their own history entry.
    def __init__(self, started_at: str) -> None:
        self._started_at = started_at

    def value(self) -> str:
        stamp = datetime.fromisoformat(self._started_at).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{stamp}-{secrets.token_hex(4)}"


# ── Per-run directory ────────────────────────────────────────────────────

class ExperimentDirectory:
    def __init__(self, base_dir: str, run_id: str) -> None:
        self._base_dir = base_dir
        self._run_id   = run_id

    def path(self) -> str:
        return os.path.join(self._base_dir, self._run_id)

    def config_path(self) -> str:
        return os.path.join(self.path(), "config.json")

    def strategy_path(self) -> str:
        return os.path.join(self.path(), "strategy.json")

    def log_path(self) -> str:
        return os.path.join(self.path(), "run_log.txt")

    def ensure(self) -> None:
        os.makedirs(self.path(), exist_ok=True)


class ResolvedConfigFile:
    def __init__(self, filepath: str, raw_config: dict[str, Any]) -> None:
        self._filepath   = filepath
        self._raw_config = raw_config

    def save(self) -> None:
        # default=str: an unquoted YAML scalar like `start_date: 2025-06-01` parses
        # to datetime.date, not str — stringify rather than let json.dump raise.
        with open(self._filepath, "w") as handle:
            json.dump(self._raw_config, handle, indent=2, default=str)


# ── Index ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentRecord:
    run_id:      str
    started_at:  str
    git_commit:  str
    pair:        str
    granularity: str
    start_date:  str
    end_date:    str
    test_split:  float
    ga_config:       GaConfig
    strategy_config: StrategyConfig
    performance:     dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        return {
            "run_id":                self.run_id,
            "started_at":            self.started_at,
            "git_commit":            self.git_commit,
            "pair":                  self.pair,
            "granularity":           self.granularity,
            "start_date":            self.start_date,
            "end_date":              self.end_date,
            "test_split":            self.test_split,
            "seed":                  self.ga_config.seed,
            "population_size":       self.ga_config.population_size,
            "generations":           self.ga_config.generations,
            "mutation_rate":         self.ga_config.mutation_rate,
            "crossover_rate":        self.ga_config.crossover_rate,
            "tournament_size":       self.ga_config.tournament_size,
            "elitism_count":         self.ga_config.elitism_count,
            "mutation_sigma":        self.ga_config.mutation_sigma,
            "buy_threshold":         self.strategy_config.buy_threshold,
            "sell_threshold":        self.strategy_config.sell_threshold,
            "position_size_pct":     self.strategy_config.position_size_pct,
            "unwind_at_entry_price": self.strategy_config.unwind_at_entry_price,
            "gross_profit":          self.performance["gross_profit"],
            "annualized_yield":      self.performance["annualized_yield"],
            "total_trades":          self.performance["total_trades"],
            "win_rate":              self.performance["win_rate"],
            "max_drawdown":          self.performance["max_drawdown"],
        }


class ExperimentIndex:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def append(self, record: ExperimentRecord) -> None:
        ParentDirectory(self._filepath).ensure()
        is_new = self._create_exclusively()
        with open(self._filepath, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_INDEX_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(record.as_row())

    def _create_exclusively(self) -> bool:
        # os.O_CREAT|os.O_EXCL is atomic — unlike an exists()-then-open() check,
        # only one of several concurrent sweep processes can ever "win" this and
        # be the one that writes the header, however closely they race.
        try:
            os.close(os.open(self._filepath, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False
