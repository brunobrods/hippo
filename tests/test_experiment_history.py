import csv
import os
import time
import json
import subprocess
from datetime import date

import pytest

from coinbase.ga.experiment_history import (
    ExperimentDirectory,
    ExperimentIndex,
    ExperimentRecord,
    IndexLock,
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


def test_experiment_index_ensure_appendable_rejects_before_any_row_is_built(tmp_path):
    # callable up front so a caller can fail fast, rather than discovering the
    # mismatch only after a full training run has finished
    filepath = tmp_path / "index.csv"
    filepath.write_text("run_id,started_at,pair\nold-1,2026-01-01,BTC-USDC\n")

    with pytest.raises(ValueError, match="misalign"):
        ExperimentIndex(str(filepath)).ensure_appendable()


def test_experiment_index_ensure_appendable_accepts_a_missing_file(tmp_path):
    ExperimentIndex(str(tmp_path / "not-created-yet.csv")).ensure_appendable()


def test_experiment_index_refuses_to_append_to_an_older_column_set(tmp_path):
    # the header is only written once, so appending newer rows to an index
    # written with fewer columns would silently shift every value sideways
    filepath = tmp_path / "index.csv"
    filepath.write_text("run_id,started_at,pair\nold-1,2026-01-01,BTC-USDC\n")

    with pytest.raises(ValueError, match="misalign"):
        ExperimentIndex(str(filepath)).append(_record())


def test_experiment_index_still_appends_when_the_header_matches(tmp_path):
    filepath = tmp_path / "index.csv"
    index    = ExperimentIndex(str(filepath))
    index.append(_record("run-1"))
    index.append(_record("run-2"))  # exercises the verify path on an unchanged schema
    assert len(list(csv.DictReader(filepath.read_text().splitlines()))) == 2


# ── IndexLock ────────────────────────────────────────────────────────
# A parallel sweep appends to one index.csv from N processes; two
# unsynchronized appends can interleave into a single corrupt row.

def test_the_lock_is_released_when_the_block_exits(tmp_path):
    target = str(tmp_path / "index.csv")
    with IndexLock(target):
        assert os.path.exists(f"{target}.lock")
    assert not os.path.exists(f"{target}.lock")


def test_the_lock_is_released_even_when_the_block_raises(tmp_path):
    target = str(tmp_path / "index.csv")
    with pytest.raises(RuntimeError):
        with IndexLock(target):
            raise RuntimeError("boom")
    assert not os.path.exists(f"{target}.lock")


def test_a_second_holder_waits_rather_than_writing_alongside(tmp_path):
    target = str(tmp_path / "index.csv")
    lock   = IndexLock(target, timeout=0.2, poll=0.01)
    lock.__enter__()
    # The holder never releases, so the waiter must break the stale lock and
    # take it rather than blocking the sweep forever.
    started = time.time()
    with IndexLock(target, timeout=0.2, poll=0.01):
        assert time.time() - started >= 0.2
    lock.__exit__()


def test_appending_leaves_no_lock_behind(tmp_path):
    filepath = tmp_path / "index.csv"
    ExperimentIndex(str(filepath)).append(_record("run-1"))
    assert not os.path.exists(f"{filepath}.lock")
    assert len(list(csv.DictReader(filepath.read_text().splitlines()))) == 1
