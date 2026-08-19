import csv
import json
import subprocess
from datetime import date

import pytest

from coinbase.ga.experiment_history import (
    ExperimentDirectory,
    ExperimentIndex,
    ExperimentRecord,
    GitCommitHash,
    ResolvedConfigFile,
    RunId,
)
from coinbase.ga.ga_engine import GaConfig
from coinbase.ga.strategy_evaluator import StrategyConfig


def _ga_config() -> GaConfig:
    return GaConfig(
        population_size=20, generations=5, mutation_rate=0.2, crossover_rate=0.7,
        tournament_size=3, elitism_count=2, mutation_sigma=0.15, seed=42,
    )


def _strategy_config() -> StrategyConfig:
    return StrategyConfig(position_size_pct=0.10, buy_threshold=0.6, sell_threshold=0.4, starting_balance=1000.0)


# ── RunId ──────────────────────────────────────────────────────────────

def test_run_id_differs_across_calls_even_with_the_same_started_at():
    # e.g. a sweep re-running the same fixed-seed config within the same second
    started_at = "2026-07-21T21:15:03.500000+00:00"
    a = RunId(started_at).value()
    b = RunId(started_at).value()
    assert a != b


def test_run_id_starts_with_a_microsecond_precision_timestamp():
    run_id = RunId("2026-07-21T21:15:03.123456+00:00").value()
    assert run_id.startswith("20260721T211503123456Z-")


# ── ExperimentDirectory ──────────────────────────────────────────────────

def test_experiment_directory_paths_are_nested_under_run_id(tmp_path):
    directory = ExperimentDirectory(str(tmp_path), "20260721T211503Z-abcd1234")
    assert directory.path() == str(tmp_path / "20260721T211503Z-abcd1234")
    assert directory.config_path() == str(tmp_path / "20260721T211503Z-abcd1234" / "config.json")
    assert directory.strategy_path() == str(tmp_path / "20260721T211503Z-abcd1234" / "strategy.json")
    assert directory.log_path() == str(tmp_path / "20260721T211503Z-abcd1234" / "run_log.txt")


def test_experiment_directory_ensure_creates_the_directory(tmp_path):
    directory = ExperimentDirectory(str(tmp_path), "run-1")
    assert not (tmp_path / "run-1").exists()
    directory.ensure()
    assert (tmp_path / "run-1").is_dir()


# ── ResolvedConfigFile ───────────────────────────────────────────────────

def test_resolved_config_file_saves_the_exact_raw_config(tmp_path):
    raw_config = {"data": {"pair": "BTC-USDC"}, "strategy": {"buy_threshold": 0.6}}
    filepath   = tmp_path / "config.json"
    ResolvedConfigFile(str(filepath), raw_config).save()
    assert json.loads(filepath.read_text()) == raw_config


def test_resolved_config_file_stringifies_values_json_cannot_serialize(tmp_path):
    # an unquoted YAML scalar like `start_date: 2025-06-01` parses to
    # datetime.date, not str — this must not crash the save.
    raw_config = {"data": {"start_date": date(2025, 6, 1)}}
    filepath   = tmp_path / "config.json"
    ResolvedConfigFile(str(filepath), raw_config).save()
    assert json.loads(filepath.read_text()) == {"data": {"start_date": "2025-06-01"}}


# ── GitCommitHash ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_git_commit_hash_matches_git_rev_parse_head():
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert await GitCommitHash().value() == expected


# ── ExperimentIndex ──────────────────────────────────────────────────────

def _record(run_id: str = "run-1") -> ExperimentRecord:
    return ExperimentRecord(
        run_id=run_id, started_at="2026-07-21T21:15:03+00:00", git_commit="deadbeef",
        pair="BTC-USDC", granularity="ONE_HOUR", start_date="2024-01-01", end_date="2024-01-02",
        test_split=0.3, ga_config=_ga_config(), strategy_config=_strategy_config(),
        performance={
            "gross_profit": 12.5, "annualized_yield": 0.15, "total_trades": 3,
            "win_rate": 0.66, "max_drawdown": 0.1,
        },
    )


def test_experiment_index_writes_header_and_row_on_first_append(tmp_path):
    filepath = tmp_path / "index.csv"
    ExperimentIndex(str(filepath)).append(_record())

    rows = list(csv.DictReader(filepath.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["pair"] == "BTC-USDC"
    assert rows[0]["gross_profit"] == "12.5"
    assert rows[0]["seed"] == "42"


def test_experiment_index_appends_without_repeating_the_header(tmp_path):
    filepath = tmp_path / "index.csv"
    index    = ExperimentIndex(str(filepath))
    index.append(_record("run-1"))
    index.append(_record("run-2"))

    lines = filepath.read_text().splitlines()
    assert lines[0].startswith("run_id,")
    rows = list(csv.DictReader(lines))
    assert [row["run_id"] for row in rows] == ["run-1", "run-2"]


def test_experiment_index_creates_missing_parent_directory(tmp_path):
    filepath = tmp_path / "nested" / "index.csv"
    ExperimentIndex(str(filepath)).append(_record())
    assert filepath.exists()
