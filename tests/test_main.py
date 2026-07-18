import json
import math

import pytest

from coinbase.ga.ga_engine import GaConfig, Genome
from coinbase.ga.main import ConsoleGenerationLog, TrainingRun, TrainingSummary
from coinbase.ga.strategy_evaluator import BacktestResult, StrategyConfig, Trade
from coinbase.ga.strategy_output import (
    GaRunLog,
    PerformanceReport,
    StrategyJson,
    StrategyMetadata,
    TrainedStrategy,
    TrainingPeriod,
)


# ── Test double ──────────────────────────────────────────────────────

class FakeAdapter:
    """Stands in for CoinbaseAdapter — only the method market data touches."""

    def __init__(self, candles: list[dict]) -> None:
        self._candles = candles

    async def get_product_candles(self, product_id: str, start: int, end: int, granularity: str) -> list[dict]:
        return self._candles


def _candle(start: int, close: float) -> dict:
    return {"start": str(start), "close": str(close), "high": str(close), "low": str(close), "volume": "1.0"}


def _oscillating_candles(n: int, step: int = 3600) -> list[dict]:
    return [_candle(i * step, close=100.0 + 10.0 * math.sin(i / 5.0)) for i in range(n)]


def _raw_config(output_dir) -> dict:
    return {
        "data": {
            "pair": "BTC-USDC",
            "granularity": "ONE_HOUR",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "test_split": 0.3,
        },
        "strategy": {
            "position_size_pct": 0.10,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "starting_balance": 1000.0,
            "indicators": {
                "sma_short_period": 2,
                "sma_long_period": 3,
                "sma_extra_period": 5,
                "rsi_period": 5,
                "macd_fast": 2,
                "macd_slow": 3,
                "macd_signal": 2,
            },
        },
        "genetic_algorithm": {
            "population_size": 6,
            "generations": 2,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "tournament_size": 3,
            "elitism_count": 1,
            "seed": 11,
        },
        "output": {
            "strategy_filepath": str(output_dir / "best_strategy.json"),
            "log_filepath":      str(output_dir / "ga_run_log.txt"),
        },
    }


# ── ConsoleGenerationLog ─────────────────────────────────────────────

def test_console_generation_log_writes_file_and_prints(tmp_path, capsys):
    log_path = tmp_path / "log.txt"
    console_log = ConsoleGenerationLog(GaRunLog(str(log_path)))

    console_log.append(1, best_fitness=2.5, average_fitness=1.0)

    assert log_path.read_text().strip() == "1\t2.500000\t1.000000"
    assert "generation" in capsys.readouterr().out


# ── TrainingSummary ──────────────────────────────────────────────────

def test_training_summary_as_text_reports_saved_path_and_metrics():
    metadata = StrategyMetadata(
        pair="BTC-USDC",
        granularity="ONE_HOUR",
        training_period=TrainingPeriod("2024-01-01", "2024-01-02"),
        ga_config=GaConfig(
            population_size=6, generations=2, mutation_rate=0.2, crossover_rate=0.8,
            tournament_size=3, elitism_count=1,
        ),
        created_at="2026-07-18T00:00:00+00:00",
    )
    strategy = TrainedStrategy(
        Genome({"sma_short": 1.0}),
        StrategyConfig(position_size_pct=0.1, buy_threshold=0.6, sell_threshold=0.4, starting_balance=1000.0),
    )
    performance = PerformanceReport(BacktestResult([Trade(100.0, 110.0, 1.0)], [1000.0, 1010.0]))
    strategy_json = StrategyJson(metadata, strategy, performance)

    summary = TrainingSummary(strategy_json, output_path="/tmp/best_strategy.json", round_trip_matches=True)
    text = summary.as_text()

    assert "Saved strategy to /tmp/best_strategy.json" in text
    assert "Test-set gross profit: 10.00" in text
    assert "Reload round-trip:     OK" in text


def test_training_summary_as_text_reports_round_trip_mismatch():
    metadata = StrategyMetadata(
        pair="BTC-USDC", granularity="ONE_HOUR",
        training_period=TrainingPeriod("2024-01-01", "2024-01-02"),
        ga_config=GaConfig(
            population_size=6, generations=2, mutation_rate=0.2, crossover_rate=0.8,
            tournament_size=3, elitism_count=1,
        ),
        created_at="2026-07-18T00:00:00+00:00",
    )
    strategy = TrainedStrategy(
        Genome({"sma_short": 1.0}),
        StrategyConfig(position_size_pct=0.1, buy_threshold=0.6, sell_threshold=0.4, starting_balance=1000.0),
    )
    performance = PerformanceReport(BacktestResult([], [1000.0]))
    strategy_json = StrategyJson(metadata, strategy, performance)

    summary = TrainingSummary(strategy_json, output_path="/tmp/x.json", round_trip_matches=False)
    assert "Reload round-trip:     MISMATCH" in summary.as_text()


# ── TrainingRun (end-to-end, fake adapter) ───────────────────────────

@pytest.mark.asyncio
async def test_training_run_trains_saves_and_round_trips(tmp_path):
    adapter    = FakeAdapter(_oscillating_candles(60))
    raw_config = _raw_config(tmp_path)

    summary = await TrainingRun(adapter, raw_config).train()

    strategy_path = tmp_path / "best_strategy.json"
    log_path      = tmp_path / "ga_run_log.txt"

    assert strategy_path.exists()
    on_disk = json.loads(strategy_path.read_text())
    assert set(on_disk) == {"metadata", "strategy", "performance"}
    assert on_disk["metadata"]["pair"] == "BTC-USDC"
    assert set(on_disk["strategy"]["weights"]) == {"sma_short", "sma_long", "sma_extra", "rsi", "macd"}

    assert log_path.exists()
    assert len(log_path.read_text().splitlines()) == 2  # one line per generation

    assert "Reload round-trip:     OK" in summary.as_text()
