import asyncio

from exchange.selection import ConfiguredExchange
from coinbase.ga.config import ConfigFile
from coinbase.ga.ga_engine import BackfilledGenome, Genome
from coinbase.ga.market_data_processor import LiveBasket, MarketDataConfig
from coinbase.ga.strategy_evaluator import (
    GaStrategy,
    POSITION_PNL_KEY,
    StrategyConfigFile,
    ValidatedStrategyConfig,
    ValidatedWeightKeys,
    WeightKeysConfig,
)
from coinbase.ga.strategy_output import DryRunLog, OutputConfigFile, StrategyJsonFile, UtcNow
from coinbase.market_scanner import GRANULARITY_SECONDS
from coinbase.strategy import LiveMarketRow, PaperTradingRun
from coinbase.trading_strategy import Decision, Ledger


# ── Console progress ─────────────────────────────────────────────────────

class ConsoleDryRunLog:
    def __init__(self, log: DryRunLog) -> None:
        self._log = log

    def append(self, timestamp: str, decision: Decision, balance: float, equity: float) -> None:
        self._log.append(timestamp, decision, balance, equity)
        print(
            f"{timestamp}  {decision.action.value:<4}  size {decision.size:>12.6f}  "
            f"balance {balance:>14.2f}  equity {equity:>14.2f}"
        )


# ── Loop ───────────────────────────────────────────────────────────────

class DryRun:
    def __init__(self, run: PaperTradingRun, log: ConsoleDryRunLog, interval_seconds: int) -> None:
        self._run              = run
        self._log              = log
        self._interval_seconds = interval_seconds

    async def forever(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self._interval_seconds)

    async def tick(self) -> None:
        decision = await self._run.on_timer()
        ledger   = self._run.ledger()
        price    = self._run.last_price()
        self._log.append(UtcNow().iso(), decision, ledger.balance(), ledger.equity(price))


# ── Entry point ────────────────────────────────────────────────────────
# Run (as a module, from the repo root — a direct script path fails with
# ModuleNotFoundError: No module named 'coinbase'):
#   python -m coinbase.ga.dry_run
# Reloads the last trained strategy and drives it against live market data on
# a loop matched to its trained granularity, logging every decision it would
# make against a simulated balance seeded from config.yaml's starting_balance.
# Never places a real order — safe to leave running against live prices.
# Requires only a read-scoped key for the configured exchange (candles only,
# no account or order calls).
# Ctrl+C to stop.

async def _main() -> None:
    raw_config      = ConfigFile("coinbase/ga/config.yaml").raw()
    market_config   = MarketDataConfig(raw_config)
    window          = market_config.window()
    strategy_config = ValidatedStrategyConfig(StrategyConfigFile(raw_config).config()).config()
    output_config   = OutputConfigFile(raw_config).config()
    weight_keys     = ValidatedWeightKeys(
        WeightKeysConfig(raw_config).keys(), market_config.normalized_columns(),
    ).keys()
    keys = weight_keys + (POSITION_PNL_KEY,)

    reloaded = StrategyJsonFile(output_config.strategy_filepath)
    # Backfilled for the same reason paper_trading does it: a genome saved
    # before a weight key existed carries no weight for it, and Genome.weight
    # raises rather than guessing.
    genome   = BackfilledGenome(Genome(reloaded.weights()), keys)
    for key in genome.missing():
        print(f"note: genome predates {key} — running it weighted zero; retrain to use it")
    strategy = GaStrategy(genome.filled(), strategy_config, keys)

    async with ConfiguredExchange(raw_config).adapter() as adapter:
        basket      = LiveBasket(
            adapter, market_config.index_pairs(), window.granularity,
        ) if market_config.index_pairs() else None
        market_row  = LiveMarketRow(
            adapter, window.pair, window.granularity, market_config.periods(), market_config.normalized_columns(),
            basket=basket, index_period=market_config.index_period(),
        )
        ledger      = Ledger(strategy_config.starting_balance)
        run         = PaperTradingRun(market_row, strategy, ledger)
        console_log = ConsoleDryRunLog(DryRunLog(output_config.dry_run_log_filepath))
        interval    = GRANULARITY_SECONDS[window.granularity]

        print(f"Dry-running {window.pair} every {interval}s against simulated balance {ledger.balance():.2f}")
        await DryRun(run, console_log, interval).forever()


if __name__ == "__main__":
    asyncio.run(_main())
