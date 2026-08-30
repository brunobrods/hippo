import csv
import json

import pandas as pd
import pytest
import yaml

from coinbase.ga.results import (
    ConsoleTable,
    ExperimentConfigs,
    ExperimentIndexFile,
    GroupedComparison,
    Leaderboard,
    PairFilter,
    RunLogFile,
    main,
    parse_args,
)

_INDEX_FIELDS = ("run_id", "pair", "mutation_rate", "annualized_yield")


def _write_index(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── ExperimentIndexFile ──────────────────────────────────────────────

def test_experiment_index_file_loads_csv_into_a_dataframe(tmp_path):
    path = tmp_path / "index.csv"
    _write_index(path, [
        {"run_id": "r1", "pair": "BTC-USDC", "mutation_rate": 0.1, "annualized_yield": 0.2},
        {"run_id": "r2", "pair": "BTC-USDC", "mutation_rate": 0.2, "annualized_yield": 0.5},
    ])
    frame = ExperimentIndexFile(str(path)).dataframe()
    assert list(frame["run_id"]) == ["r1", "r2"]
    assert frame["annualized_yield"].tolist() == pytest.approx([0.2, 0.5])


# ── RunLogFile ───────────────────────────────────────────────────────

def test_run_log_file_extracts_only_generation_rows(tmp_path):
    path = tmp_path / "run_log.txt"
    path.write_text(
        "=== run 2026-07-21T21:15:03+00:00 ===\n"
        "pair=FET-USDC granularity=TWO_HOUR window=2026-01-01..2026-07-01 test_split=0.5\n"
        "buy_threshold=0.60 sell_threshold=0.40 position_size_pct=0.10 unwind_at_entry_price=True\n"
        "population=20 generations=2 mutation_rate=0.2 crossover_rate=0.7 "
        "tournament_size=3 elitism_count=2 mutation_sigma=0.15 seed=42\n"
        "generation\tbest_fitness\tavg_fitness\n"
        "1\t1.500000\t0.800000\n"
        "2\t2.000000\t1.100000\n"
    )
    frame = RunLogFile(str(path)).generations()
    assert frame["generation"].tolist() == [1, 2]
    assert frame["best_fitness"].tolist() == pytest.approx([1.5, 2.0])
    assert frame["avg_fitness"].tolist() == pytest.approx([0.8, 1.1])


def test_run_log_file_handles_a_log_with_multiple_runs_appended(tmp_path):
    path = tmp_path / "run_log.txt"
    path.write_text(
        "=== run A ===\nheader\nheader\nheader\ngeneration\tbest_fitness\tavg_fitness\n1\t1.0\t0.5\n"
        "=== run B ===\nheader\nheader\nheader\ngeneration\tbest_fitness\tavg_fitness\n1\t2.0\t1.0\n"
    )
    frame = RunLogFile(str(path)).generations()
    assert len(frame) == 2  # both runs' generation rows extracted, headers skipped


# ── Leaderboard ──────────────────────────────────────────────────────

def test_leaderboard_sorts_descending_and_limits_to_top_n():
    frame = pd.DataFrame({"run_id": ["a", "b", "c"], "annualized_yield": [0.1, 0.9, 0.5]})
    rows = Leaderboard(frame, "annualized_yield", top=2).rows()
    assert rows["run_id"].tolist() == ["b", "c"]


# ── GroupedComparison ────────────────────────────────────────────────

def test_grouped_comparison_aggregates_mean_per_group_sorted_descending():
    frame = pd.DataFrame({
        "mutation_rate":    [0.1, 0.1, 0.2, 0.2],
        "annualized_yield": [0.1, 0.3, 0.9, 0.7],
    })
    summary = GroupedComparison(frame, "mutation_rate", "annualized_yield").summary()
    assert summary.index.tolist() == [0.2, 0.1]  # 0.2's mean (0.8) beats 0.1's (0.2)
    assert summary.loc[0.2, "mean"] == pytest.approx(0.8)
    assert summary.loc[0.1, "count"] == 2


# ── PairFilter ───────────────────────────────────────────────────────

def test_pair_filter_keeps_only_matching_rows():
    frame = pd.DataFrame({"pair": ["BTC-USDC", "ETH-USDC"], "annualized_yield": [0.1, 0.2]})
    filtered = PairFilter(frame, "ETH-USDC").applied()
    assert filtered["pair"].tolist() == ["ETH-USDC"]


def test_pair_filter_passes_through_unfiltered_when_pair_is_none():
    frame = pd.DataFrame({"pair": ["BTC-USDC", "ETH-USDC"]})
    assert len(PairFilter(frame, None).applied()) == 2


# ── ConsoleTable ─────────────────────────────────────────────────────

def test_console_table_prints_frame_contents(capsys):
    ConsoleTable(pd.DataFrame({"run_id": ["r1"], "annualized_yield": [0.42]})).print()
    out = capsys.readouterr().out
    assert "r1" in out
    assert "0.42" in out


def test_console_table_prints_placeholder_for_empty_frame(capsys):
    ConsoleTable(pd.DataFrame()).print()
    assert "no matching rows" in capsys.readouterr().out


# ── parse_args ───────────────────────────────────────────────────────

def test_parse_args_defaults():
    args = parse_args([])
    assert args.metric == "annualized_yield"
    assert args.pair is None
    assert args.top is None
    assert args.group_by is None
    assert args.run_log is None


def test_parse_args_top_and_group_by_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--top", "5", "--group-by", "mutation_rate"])


# ── main (end-to-end) ──────────────────────────────────────────────────

def _write_config(tmp_path, index_path, experiments_dir) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "output": {"experiments_dir": str(experiments_dir), "index_filepath": str(index_path)},
    }))
    return str(config_path)


def test_main_leaderboard_mode_prints_top_rows(tmp_path, capsys):
    index_path = tmp_path / "index.csv"
    _write_index(index_path, [
        {"run_id": "r1", "pair": "BTC-USDC", "mutation_rate": 0.1, "annualized_yield": 0.1},
        {"run_id": "r2", "pair": "BTC-USDC", "mutation_rate": 0.2, "annualized_yield": 0.9},
    ])
    config_path = _write_config(tmp_path, index_path, tmp_path / "experiments")

    main(["--config", config_path, "--top", "1"])

    out = capsys.readouterr().out
    assert "r2" in out
    assert "r1" not in out


def test_main_leaderboard_top_zero_shows_no_rows(tmp_path, capsys):
    # --top 0 must mean "zero rows", not fall back to the default of 20
    # (Python treats 0 as falsy, an easy footgun with `args.top or 20`)
    index_path = tmp_path / "index.csv"
    _write_index(index_path, [{"run_id": "r1", "pair": "BTC-USDC", "mutation_rate": 0.1, "annualized_yield": 0.1}])
    config_path = _write_config(tmp_path, index_path, tmp_path / "experiments")

    main(["--config", config_path, "--top", "0"])

    out = capsys.readouterr().out
    assert "r1" not in out


def test_main_group_by_mode(tmp_path, capsys):
    index_path = tmp_path / "index.csv"
    _write_index(index_path, [
        {"run_id": "r1", "pair": "BTC-USDC", "mutation_rate": 0.1, "annualized_yield": 0.1},
        {"run_id": "r2", "pair": "BTC-USDC", "mutation_rate": 0.2, "annualized_yield": 0.9},
    ])
    config_path = _write_config(tmp_path, index_path, tmp_path / "experiments")

    main(["--config", config_path, "--group-by", "mutation_rate"])

    out = capsys.readouterr().out
    assert "mutation_rate" in out


def test_main_run_log_mode(tmp_path, capsys):
    experiments_dir = tmp_path / "experiments"
    run_dir = experiments_dir / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_log.txt").write_text(
        "=== run x ===\nh\nh\nh\ngeneration\tbest_fitness\tavg_fitness\n1\t1.0\t0.5\n"
    )
    config_path = _write_config(tmp_path, tmp_path / "index.csv", experiments_dir)

    main(["--config", config_path, "--run-log", "run-1"])

    out = capsys.readouterr().out
    assert "best_fitness" in out
    assert "1.0" in out


# ── ExperimentConfigs ────────────────────────────────────────────────
# index.csv has a fixed header and ExperimentIndex refuses rows built from a
# different one, so a knob like weight_keys can only be recovered from each
# run's own config.json.

def _run_config(experiments_dir, run_id: str, keys: list, index_pairs: list = None) -> None:
    directory = experiments_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "strategy":    {"weight_keys": keys},
        "market_data": {"index_pairs": index_pairs or []},
    }))


def test_weight_keys_are_read_back_from_each_runs_config(tmp_path):
    experiments = tmp_path / "experiments"
    _run_config(experiments, "run-1", ["rsi", "macd"])
    _run_config(experiments, "run-2", ["rsi", "macd", "index_z"], ["BTC-USDT"])
    frame = pd.DataFrame({"run_id": ["run-1", "run-2"]})

    enriched = ExperimentConfigs(frame, str(experiments)).enriched()
    assert list(enriched["weight_keys"]) == ["rsi,macd", "rsi,macd,index_z"]
    assert list(enriched["index_pairs"]) == [0, 1]


# A run recorded before ResolvedConfigFile existed, or a directory pruned by
# hand, must not take the whole leaderboard down.
def test_a_run_with_no_saved_config_is_marked_unknown(tmp_path):
    frame    = pd.DataFrame({"run_id": ["ghost"]})
    enriched = ExperimentConfigs(frame, str(tmp_path / "experiments")).enriched()
    assert enriched.loc[0, "weight_keys"] == "unknown"
    assert enriched.loc[0, "index_pairs"] == 0


def test_enriching_leaves_the_original_frame_alone(tmp_path):
    experiments = tmp_path / "experiments"
    _run_config(experiments, "run-1", ["rsi"])
    frame = pd.DataFrame({"run_id": ["run-1"]})
    ExperimentConfigs(frame, str(experiments)).enriched()
    assert "weight_keys" not in frame.columns


# The point of the class: it makes a feature ablation groupable with the
# existing GroupedComparison, without touching index.csv's schema.
def test_the_enriched_column_groups_an_ablation(tmp_path):
    experiments = tmp_path / "experiments"
    _run_config(experiments, "a", ["rsi"])
    _run_config(experiments, "b", ["rsi"])
    _run_config(experiments, "c", ["rsi", "index_z"])
    frame = pd.DataFrame({"run_id": ["a", "b", "c"], "annualized_yield": [0.1, 0.3, 0.9]})

    enriched = ExperimentConfigs(frame, str(experiments)).enriched()
    summary  = GroupedComparison(enriched, "weight_keys", "annualized_yield").summary()
    assert summary.loc["rsi,index_z", "mean"] == pytest.approx(0.9)
    assert summary.loc["rsi", "mean"] == pytest.approx(0.2)
