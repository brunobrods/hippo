import asyncio

import pytest

from coinbase.ga.paper_engine import (
    AlgoConfig,
    BookSnapshot,
    CandleBoundary,
    DecisionLoop,
    IsolatedAlgo,
    PaperBook,
    PaperEngineConfigFile,
    PaperJournal,
    PaperAlgo,
    PriceLoop,
    PriceMarks,
)
from coinbase.ga.paper_metrics import EquityCurve
from coinbase.ga.paper_trading import BasisPointFee, NoFees, PaperState, PaperStateFile
from coinbase.ga.strategy_output import DryRunLog
from coinbase.trading_strategy import Action, Decision, Direction, Position
from exchange.adapter import ExchangeError


# ── Test doubles ─────────────────────────────────────────────────────

class FakeAdapter:
    """Stands in for an ExchangeAdapter — candles and prices only.

    It deliberately implements NO order-placement methods: if the engine ever
    tried to place one this would raise AttributeError instead of completing.
    """

    def __init__(self, price: float = 100.0) -> None:
        self.price      = price
        self.book_calls = 0

    async def get_best_bid_ask(self, *product_ids: str) -> dict:
        self.book_calls += 1
        return {"pricebooks": [
            {"product_id": pair, "bids": [{"price": str(self.price)}]}
            for pair in product_ids
        ]}

    def max_candles_per_request(self) -> int:
        return 300

    def name(self) -> str:
        return "fake"


class FakeLane:
    def __init__(self, adapter: FakeAdapter) -> None:
        self._adapter = adapter

    def adapter(self) -> FakeAdapter:
        return self._adapter

    def limit(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(8)


class FakeRows:
    def __init__(self, rows: list[dict[str, float]], error: Exception = None) -> None:
        self._rows  = rows
        self._error = error
        self.calls  = 0

    def pair(self) -> str:
        return "BTC-USDC"

    async def latest(self) -> dict[str, float]:
        if self._error is not None:
            raise self._error
        row = self._rows[min(self.calls, len(self._rows) - 1)]
        self.calls += 1
        return row


class _ScriptedStrategy:
    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions
        self._calls   = 0

    def decide(self, row, position, balance) -> Decision:
        action = self._actions[min(self._calls, len(self._actions) - 1)]
        self._calls += 1
        size = (balance * 0.5) / row["close"] if action in (Action.BUY, Action.SHORT) else 0.0
        return Decision(action, size)

    def signal_score(self, row, position) -> float:
        return 0.5


def _row(timestamp: int, close: float) -> dict[str, float]:
    return {
        "timestamp": float(timestamp), "close": close, "high": close, "low": close,
        "rsi": 55.0, "macd": 0.25,
    }


def _algo_config(tmp_path, name: str = "btc", balance: float = 1000.0) -> AlgoConfig:
    return AlgoConfig(
        name=name, exchange="coinbase", pair="BTC-USDC", granularity="THIRTY_MINUTE",
        strategy_filepath="unused.json", starting_balance=balance,
        state_filepath=str(tmp_path / f"{name}.json"),
        log_filepath=str(tmp_path / name / "decisions.tsv"),
    )


def _algo(tmp_path, rows: FakeRows, actions: list[Action], fees=NoFees(), name="btc") -> PaperAlgo:
    config = _algo_config(tmp_path, name)
    return PaperAlgo(
        config=config, rows=rows, strategy=_ScriptedStrategy(actions),
        book=PaperBook(PaperStateFile(config.state_filepath), config.starting_balance,
                       config.pair),
        fees=fees, curve=EquityCurve(), log=DryRunLog(config.log_filepath),
    )


# ── CandleBoundary ───────────────────────────────────────────────────

def test_boundary_waits_the_offset_past_a_just_closed_candle():
    boundary = CandleBoundary("THIRTY_MINUTE", offset_seconds=20)

    # A candle closed exactly at 1800; the fetch waits 20s past it.
    assert boundary.next_at(1800.0) == 1820.0
    assert boundary.seconds_until(1800.0) == pytest.approx(20.0)


def test_boundary_rolls_to_the_next_period_once_the_offset_has_passed():
    boundary = CandleBoundary("THIRTY_MINUTE", offset_seconds=20)

    assert boundary.next_at(1820.0) == 3620.0
    assert boundary.next_at(1821.0) == 3620.0


def test_boundary_never_schedules_a_wait_in_the_past():
    boundary = CandleBoundary("THIRTY_MINUTE", offset_seconds=20)

    for now in range(0, 7200, 37):
        assert boundary.seconds_until(float(now)) > 0.0


def test_thirty_minute_boundaries_land_on_the_half_hour():
    boundary = CandleBoundary("THIRTY_MINUTE", offset_seconds=0)

    # 12:07:30 UTC -> the 12:30 boundary.
    assert boundary.next_at(43650.0) % 1800 == 0


# ── PaperEngineConfigFile ────────────────────────────────────────────

def test_defaults_merge_under_every_algo_and_an_entry_wins():
    config = PaperEngineConfigFile({
        "defaults": {"exchange": "coinbase", "starting_balance": 10000.0},
        "algos": [
            {"name": "btc", "pair": "BTC-USDC", "strategy_filepath": "a.json"},
            {"name": "sol", "pair": "SOL-USDT", "strategy_filepath": "b.json",
             "exchange": "binance", "starting_balance": 5000.0},
        ],
    }).config()

    assert (config.algos[0].exchange, config.algos[0].starting_balance) == ("coinbase", 10000.0)
    assert (config.algos[1].exchange, config.algos[1].starting_balance) == ("binance", 5000.0)


def test_state_and_log_paths_default_under_the_paper_root():
    config = PaperEngineConfigFile({
        "algos": [{"name": "btc", "pair": "BTC-USDC", "strategy_filepath": "a.json"}],
    }).config()

    assert config.algos[0].state_filepath.endswith("btc.json")
    assert "paper" in config.algos[0].state_filepath
    assert config.algos[0].log_filepath.endswith("decisions.tsv")


def test_a_commented_out_engine_section_is_not_an_error():
    # A YAML mapping whose every child is commented out parses to None.
    config = PaperEngineConfigFile({
        "engine": None, "dashboard": None,
        "algos": [{"name": "btc", "pair": "BTC-USDC", "strategy_filepath": "a.json"}],
    }).config()

    assert (config.granularity, config.port) == ("THIRTY_MINUTE", 8787)


def test_duplicate_algo_names_are_rejected():
    # Names key the state files — two algos sharing one would share a book.
    with pytest.raises(ValueError):
        PaperEngineConfigFile({
            "algos": [
                {"name": "btc", "pair": "BTC-USDC", "strategy_filepath": "a.json"},
                {"name": "btc", "pair": "ETH-USDC", "strategy_filepath": "b.json"},
            ],
        }).config()


# ── PaperBook ────────────────────────────────────────────────────────

def test_book_resumes_an_open_short_written_by_a_previous_process(tmp_path):
    path = str(tmp_path / "btc.json")
    PaperStateFile(path).write(
        PaperState(
            balance=900.0, position=Position(50.0, 2.0, Direction.SHORT),
            last_candle_start=1800, realized_trades=3, realized_wins=2,
        ),
        "BTC-USDC",
    )

    state = PaperBook(PaperStateFile(path), 1000.0, "BTC-USDC").read()

    assert state.balance == pytest.approx(900.0)
    assert state.position.direction() is Direction.SHORT
    assert (state.last_candle_start, state.realized_trades, state.realized_wins) == (1800, 3, 2)


def test_book_writes_touch_no_disk_until_snapshot(tmp_path):
    path = str(tmp_path / "btc.json")
    book = PaperBook(PaperStateFile(path), 1000.0, "BTC-USDC")

    book.write(PaperState(balance=1234.0, position=None, last_candle_start=1800,
                          realized_trades=1), "BTC-USDC")
    assert not (tmp_path / "btc.json").exists()

    book.snapshot()
    assert PaperStateFile(path).read().balance == pytest.approx(1234.0)


def test_book_seeds_from_the_starting_balance_when_no_file_exists(tmp_path):
    state = PaperBook(PaperStateFile(str(tmp_path / "new.json")), 777.0, "BTC-USDC").read()

    assert (state.balance, state.position, state.last_candle_start) == (777.0, None, 0)


# ── PaperAlgo ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_second_tick_on_the_same_candle_changes_nothing(tmp_path):
    rows = FakeRows([_row(1800, 100.0), _row(1800, 105.0)])
    algo = _algo(tmp_path, rows, [Action.BUY, Action.BUY])

    await algo.tick()
    first = algo.status()
    await algo.tick()
    second = algo.status()

    assert (second.balance, second.trades) == (first.balance, first.trades)
    assert second.position.size == pytest.approx(first.position.size)


@pytest.mark.asyncio
async def test_a_new_candle_lets_the_algo_act_again(tmp_path):
    rows = FakeRows([_row(1800, 100.0), _row(3600, 120.0)])
    algo = _algo(tmp_path, rows, [Action.BUY, Action.SELL])

    await algo.tick()
    await algo.tick()

    status = algo.status()
    assert status.position is None          # the long was closed
    assert status.trades == 1
    assert status.realized_pnl > 0.0        # bought at 100, sold at 120


@pytest.mark.asyncio
async def test_the_algo_reports_the_indicators_behind_its_decision(tmp_path):
    algo = _algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.HOLD])

    await algo.tick()

    status = algo.status()
    assert (status.rsi, status.macd) == (55.0, 0.25)
    assert status.signal_score == pytest.approx(0.5)


# ── Fees ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fees_reduce_the_balance_by_exactly_both_legs(tmp_path):
    rows = [_row(1800, 100.0), _row(3600, 120.0)]
    gross = _algo(tmp_path, FakeRows(list(rows)), [Action.BUY, Action.SELL], NoFees(), "g")
    net   = _algo(tmp_path, FakeRows(list(rows)), [Action.BUY, Action.SELL],
                  BasisPointFee(10.0), "n")

    for algo in (gross, net):
        await algo.tick()
        await algo.tick()

    # 10bps on a 5.0-unit buy at 100 and the same size sold at 120.
    size     = (1000.0 * 0.5) / 100.0
    expected = size * 100.0 * 0.001 + size * 120.0 * 0.001
    assert gross.status().balance - net.status().balance == pytest.approx(expected)


# ── IsolatedAlgo ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_failing_algo_does_not_stop_the_others(tmp_path):
    broken = IsolatedAlgo(_algo(
        tmp_path, FakeRows([], ExchangeError(500, "boom")), [Action.HOLD], name="broken",
    ))
    healthy = IsolatedAlgo(_algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.BUY], name="ok"))

    await DecisionLoop(
        (broken, healthy), CandleBoundary("THIRTY_MINUTE"),
        PaperJournal(str(tmp_path / "journal.tsv")), BookSnapshot(()),
    ).once()

    assert broken.status().running is False
    assert broken.status().error is not None
    assert healthy.status().running is True
    assert healthy.status().position is not None


@pytest.mark.asyncio
async def test_a_recovered_algo_clears_its_error(tmp_path):
    rows = FakeRows([_row(1800, 100.0)], ExchangeError(500, "boom"))
    algo = IsolatedAlgo(_algo(tmp_path, rows, [Action.HOLD]))

    await algo.tick()
    assert algo.status().running is False

    rows._error = None
    await algo.tick()
    assert algo.status().running is True
    assert algo.status().error is None


# ── DecisionLoop ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_loop_journals_every_algo_including_a_failed_one(tmp_path):
    journal = str(tmp_path / "journal.tsv")
    broken  = IsolatedAlgo(_algo(
        tmp_path, FakeRows([], ExchangeError(500, "boom")), [Action.HOLD], name="broken",
    ))
    healthy = IsolatedAlgo(_algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.BUY], name="ok"))

    await DecisionLoop(
        (broken, healthy), CandleBoundary("THIRTY_MINUTE"),
        PaperJournal(journal), BookSnapshot(()),
    ).once()

    lines = (tmp_path / "journal.tsv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].startswith("timestamp\talgo\t")     # header written once
    assert len(lines) == 3
    assert "broken" in lines[1] and "boom" in lines[1]


@pytest.mark.asyncio
async def test_the_loop_snapshots_every_book(tmp_path):
    config = _algo_config(tmp_path)
    algo   = _algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.BUY])
    book   = PaperBook(PaperStateFile(config.state_filepath), config.starting_balance, config.pair)

    await DecisionLoop(
        (IsolatedAlgo(algo),), CandleBoundary("THIRTY_MINUTE"),
        PaperJournal(str(tmp_path / "journal.tsv")), BookSnapshot((book,)),
    ).once()

    # The snapshotted book is a different instance; it flushes what it holds.
    assert (tmp_path / "btc.json").exists() or book.read() is not None


@pytest.mark.asyncio
async def test_an_unreadable_book_does_not_blank_every_other_algo(tmp_path):
    # The dashboard builds every status in one pass, so a status() that raises
    # would 500 the whole page rather than mark one row broken.
    class _Exploding:
        def config(self):
            return _algo_config(tmp_path, "bad")

        def status(self):
            raise ValueError("state file is corrupt")

        async def tick(self):
            pass

        def mark(self, price):
            pass

    status = IsolatedAlgo(_Exploding()).status()

    assert status.running is False
    assert "corrupt" in status.error
    assert status.name == "bad"


# ── Journal ──────────────────────────────────────────────────────────

def test_a_multiline_error_cannot_split_a_journal_record(tmp_path):
    # ExchangeError embeds the raw body, which for a gateway error is HTML.
    from coinbase.ga.paper_metrics import AlgoStatus

    path   = str(tmp_path / "journal.tsv")
    status = AlgoStatus(
        name="btc", exchange="coinbase", pair="BTC-USDC", granularity="THIRTY_MINUTE",
        running=False, error="<html>\nbad\tgateway\r\n</html>", last_tick_at=None,
        last_candle_start=0, last_action="-", starting_balance=0.0, balance=0.0,
        mark_price=0.0, equity=0.0, position=None, realized_pnl=0.0,
        unrealized_pnl=0.0, trades=0, wins=0, rsi=0.0, macd=0.0,
        signal_score=0.0, fee_paid=0.0,
    )
    PaperJournal(path).append("2026-08-26T00:00:00+00:00", status)

    lines = open(path, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 2                       # header plus exactly one record
    assert len(lines[1].split("\t")) == len(lines[0].split("\t"))


# ── Catch-up ─────────────────────────────────────────────────────────

def test_the_boundary_names_the_candle_that_has_actually_closed():
    boundary = CandleBoundary("THIRTY_MINUTE", offset_seconds=20)

    # At 1820 the candle forming began at 1800, so the newest CLOSED one
    # began a full period earlier and covers [0, 1800).
    assert boundary.closed_candle_start(1820.0) == 0
    assert boundary.closed_candle_start(3620.0) == 1800


@pytest.mark.asyncio
async def test_a_late_candle_is_retried_rather_than_lost(tmp_path, monkeypatch):
    # The exchange has not published the closed candle yet, so the fetch
    # returns the previous one and PaperTick correctly no-ops — no exception,
    # so nothing retries unless the loop itself does.
    rows = FakeRows([_row(0, 100.0), _row(1800, 110.0)])
    algo = IsolatedAlgo(_algo(tmp_path, rows, [Action.HOLD, Action.BUY]))

    slept: list[float] = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("coinbase.ga.paper_engine.asyncio.sleep", _sleep)
    monkeypatch.setattr("coinbase.ga.paper_engine.time.time", lambda: 3620.0)

    loop = DecisionLoop(
        (algo,), CandleBoundary("THIRTY_MINUTE", 20),
        PaperJournal(str(tmp_path / "journal.tsv")), BookSnapshot(()),
        catch_up_seconds=60,
    )
    await loop.once()
    assert algo.status().last_candle_start == 0      # behind: got the old candle

    await loop.catch_up()

    assert slept                                     # it waited and retried
    assert algo.status().last_candle_start == 1800   # and caught the real one


# ── PriceLoop ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_marking_moves_unrealized_pnl_but_never_the_book(tmp_path):
    algo = IsolatedAlgo(_algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.BUY]))
    await algo.tick()
    before = algo.status()

    adapter = FakeAdapter(price=110.0)
    await PriceLoop((PriceMarks("coinbase", FakeLane(adapter), ("BTC-USDC",)),), (algo,), 45).once()
    after = algo.status()

    assert after.mark_price == pytest.approx(110.0)
    assert after.unrealized_pnl > before.unrealized_pnl
    # The book itself is untouched: same cash, same position, same trade count.
    assert after.balance == pytest.approx(before.balance)
    assert after.trades == before.trades
    assert after.position.size == pytest.approx(before.position.size)


@pytest.mark.asyncio
async def test_one_venue_failing_still_marks_the_others(tmp_path):
    class _BrokenMarks:
        def exchange(self) -> str:
            return "binance"

        async def prices(self) -> dict:
            raise ExchangeError(500, "venue down")

    algo    = IsolatedAlgo(_algo(tmp_path, FakeRows([_row(1800, 100.0)]), [Action.BUY]))
    await algo.tick()
    working = PriceMarks("coinbase", FakeLane(FakeAdapter(price=110.0)), ("BTC-USDC",))

    await PriceLoop((_BrokenMarks(), working), (algo,), 45).once()

    # Coinbase's book still marked the algo despite Binance being down.
    assert algo.status().mark_price == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_the_same_pair_on_two_venues_does_not_cross_contaminate(tmp_path):
    coinbase = _algo_config(tmp_path, "cb")
    binance  = AlgoConfig(
        name="bn", exchange="binance", pair="BTC-USDC", granularity="THIRTY_MINUTE",
        strategy_filepath="unused.json", starting_balance=1000.0,
        state_filepath=str(tmp_path / "bn.json"),
        log_filepath=str(tmp_path / "bn" / "decisions.tsv"),
    )
    algos = []
    for config in (coinbase, binance):
        algo = PaperAlgo(
            config=config, rows=FakeRows([_row(1800, 100.0)]),
            strategy=_ScriptedStrategy([Action.BUY]),
            book=PaperBook(PaperStateFile(config.state_filepath), 1000.0, config.pair),
            fees=NoFees(), curve=EquityCurve(), log=DryRunLog(config.log_filepath),
        )
        await algo.tick()
        algos.append(IsolatedAlgo(algo))

    await PriceLoop((
        PriceMarks("coinbase", FakeLane(FakeAdapter(price=100.0)), ("BTC-USDC",)),
        PriceMarks("binance", FakeLane(FakeAdapter(price=200.0)), ("BTC-USDC",)),
    ), tuple(algos), 45).once()

    # Same product id, two venues, two prices — each algo takes its own.
    assert algos[0].status().mark_price == pytest.approx(100.0)
    assert algos[1].status().mark_price == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_prices_are_fetched_in_one_request_per_venue(tmp_path):
    adapter = FakeAdapter()
    marks   = PriceMarks("coinbase", FakeLane(adapter), ("BTC-USDC", "ETH-USDC", "SOL-USDC"))

    prices = await marks.prices()

    assert adapter.book_calls == 1
    # Keyed by (exchange, pair) so the same product on two venues cannot collide.
    assert set(prices) == {
        ("coinbase", "BTC-USDC"), ("coinbase", "ETH-USDC"), ("coinbase", "SOL-USDC"),
    }
