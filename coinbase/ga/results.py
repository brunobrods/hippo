import argparse
import json
import os
import sys
from typing import Any, Optional

import pandas as pd

from coinbase.ga.config import ConfigFile
from coinbase.ga.experiment_history import ExperimentDirectory
from coinbase.ga.strategy_evaluator import LINEAR_DESIGN
from coinbase.ga.strategy_output import OutputConfigFile

DEFAULT_METRIC = "annualized_yield"

# Groupable, but not present in index.csv — read back per run from config.json.
_CONFIG_COLUMNS = ("weight_keys", "index_pairs", "negative_weights", "fitness_confidence")


# ── Loading ──────────────────────────────────────────────────────────────

class ExperimentIndexFile:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath)


class RunLogFile:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def generations(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        with open(self._filepath, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3 and parts[0].isdigit():
                    rows.append({
                        "generation":   int(parts[0]),
                        "best_fitness": float(parts[1]),
                        "avg_fitness":  float(parts[2]),
                    })
        return pd.DataFrame(rows)


# Columns the index cannot carry, recovered from each run's own config.json.
#
# index.csv has a fixed header and ExperimentIndex refuses to append rows built
# from a different one, so adding a column for every config knob would break
# every existing file. ResolvedConfigFile already saves the whole resolved
# config per run, which makes it the right place to read a knob back from —
# and it means comparing runs that differ in `weight_keys` (a feature ablation,
# say) needs no schema change at all.
class ExperimentConfigs:
    def __init__(self, frame: pd.DataFrame, experiments_dir: str) -> None:
        self._frame           = frame
        self._experiments_dir = experiments_dir

    def enriched(self) -> pd.DataFrame:
        frame = self._frame.copy()
        frame["weight_keys"]        = frame["run_id"].map(self._weight_keys)
        frame["index_pairs"]        = frame["run_id"].map(self._index_pairs)
        # The two knobs that change what the model can express and what the GA
        # optimizes. Neither is an index column, and comparing runs across
        # either without knowing which is which produces a meaningless average.
        frame["negative_weights"]   = frame["run_id"].map(self._negative_weights)
        frame["fitness_confidence"] = frame["run_id"].map(self._fitness_confidence)
        # Which model design scored the run. Averaging a linear run together
        # with one from another design compares two different functions and
        # reports the mean as if it meant something.
        frame["design"]             = frame["run_id"].map(self._design)
        return frame

    def _design(self, run_id: str) -> str:
        return str((self._raw(run_id).get("strategy") or {}).get("design", LINEAR_DESIGN))

    def _negative_weights(self, run_id: str) -> bool:
        return bool((self._raw(run_id).get("genetic_algorithm") or {}).get("allow_negative_weights", False))

    def _fitness_confidence(self, run_id: str) -> float:
        return float((self._raw(run_id).get("strategy") or {}).get("fitness_confidence", 0.0))

    def _weight_keys(self, run_id: str) -> str:
        raw = self._raw(run_id)
        keys = ((raw.get("strategy") or {}).get("weight_keys")) or []
        return ",".join(keys) if keys else "unknown"

    def _index_pairs(self, run_id: str) -> int:
        raw = self._raw(run_id)
        return len(((raw.get("market_data") or {}).get("index_pairs")) or [])

    def _raw(self, run_id: str) -> dict[str, Any]:
        path = ExperimentDirectory(self._experiments_dir, run_id).config_path()
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


# ── Views ────────────────────────────────────────────────────────────────

class Leaderboard:
    def __init__(self, frame: pd.DataFrame, metric: str, top: int) -> None:
        self._frame  = frame
        self._metric = metric
        self._top    = top

    def rows(self) -> pd.DataFrame:
        return self._frame.sort_values(self._metric, ascending=False).head(self._top)


class GroupedComparison:
    def __init__(self, frame: pd.DataFrame, group_by: str, metric: str) -> None:
        self._frame    = frame
        self._group_by = group_by
        self._metric   = metric

    def summary(self) -> pd.DataFrame:
        grouped = self._frame.groupby(self._group_by)[self._metric]
        summary = grouped.agg(["count", "mean", "std", "min", "max"])
        return summary.sort_values("mean", ascending=False)


class PairFilter:
    def __init__(self, frame: pd.DataFrame, pair: Optional[str]) -> None:
        self._frame = frame
        self._pair  = pair

    def applied(self) -> pd.DataFrame:
        if self._pair is None:
            return self._frame
        return self._frame[self._frame["pair"] == self._pair]


# ── Console output ─────────────────────────────────────────────────────

class ConsoleTable:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def print(self) -> None:
        if self._frame.empty:
            print("(no matching rows)")
            return
        print(self._frame.to_string())


# ── Entry point ────────────────────────────────────────────────────────
# Run (as a module, from the repo root, so the coinbase package resolves —
# `python coinbase/ga/results.py` directly does NOT work):
#   python -m coinbase.ga.results                          # top 20 by annualized_yield
#   python -m coinbase.ga.results --top 10 --pair FET-USDC
#   python -m coinbase.ga.results --group-by mutation_rate  # compare a swept parameter
#   python -m coinbase.ga.results --run-log <run_id>        # one run's GA convergence
# Reads experiments/index.csv (or a single run's run_log.txt) — no network access,
# no credentials required.

def parse_args(args_in: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare GA training runs recorded in experiments/index.csv")
    p.add_argument("--config", default="coinbase/ga/config.yaml", help="Path to config.yaml (for output paths)")
    p.add_argument("--metric", default=DEFAULT_METRIC, help=f"Performance column to rank/aggregate by (default: {DEFAULT_METRIC})")
    p.add_argument("--pair", default=None, help="Only include runs for this pair")
    # Exact match, not "on or after": annualized_yield is only comparable
    # between runs scored on the SAME window, so a range filter would invite
    # exactly the comparison that means nothing.
    p.add_argument("--window", default=None, metavar="END_DATE",
                   help="Only runs whose data window ends on this ISO date")
    p.add_argument("--with-config", action="store_true",
                   help="Add weight_keys/index_pairs columns read from each run's config.json")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--top", type=int, metavar="N", help="Show the top N runs by --metric (default mode)")
    mode.add_argument("--group-by", metavar="COLUMN", help="Group runs by COLUMN, compare mean/std of --metric per value")
    mode.add_argument("--run-log", metavar="RUN_ID", help="Print one run's per-generation best/avg fitness")

    return p.parse_args(args_in)


def main(args_in: list[str]) -> None:
    args       = parse_args(args_in)
    raw_config = ConfigFile(args.config).raw()
    output     = OutputConfigFile(raw_config).config()

    if args.run_log:
        run_log_path = ExperimentDirectory(output.experiments_dir, args.run_log).log_path()
        ConsoleTable(RunLogFile(run_log_path).generations()).print()
        return

    frame = PairFilter(ExperimentIndexFile(output.index_filepath).dataframe(), args.pair).applied()
    # Only when asked for: reading one config.json per run costs a stat and a
    # parse per row, which is wasted on the ordinary leaderboard.
    if args.group_by in _CONFIG_COLUMNS or args.with_config:
        frame = ExperimentConfigs(frame, output.experiments_dir).enriched()
    if args.window:
        frame = frame[frame["end_date"].astype(str) == args.window]

    if args.group_by:
        ConsoleTable(GroupedComparison(frame, args.group_by, args.metric).summary()).print()
    else:
        top = args.top if args.top is not None else 20
        ConsoleTable(Leaderboard(frame, args.metric, top).rows()).print()


if __name__ == "__main__":
    main(sys.argv[1:])
