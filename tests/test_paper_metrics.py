import json
import math

import pytest

from coinbase.ga.paper_metrics import (
    AlgoPerformance,
    AlgoStatus,
    EquityCurve,
    PortfolioStatus,
    PositionView,
    StatusPayload,
)


# ── Builders ─────────────────────────────────────────────────────────

def _status(
    name: str = "btc",
    trades: int = 0,
    wins: int = 0,
    balance: float = 1000.0,
    equity: float = 1000.0,
    starting: float = 1000.0,
    position: PositionView = None,
    running: bool = True,
    error: str = None,
) -> AlgoStatus:
    return AlgoStatus(
        name=name, exchange="coinbase", pair="BTC-USDC", granularity="THIRTY_MINUTE",
        running=running, error=error, last_tick_at=None, last_candle_start=1800,
        last_action="HOLD", starting_balance=starting, balance=balance,
        mark_price=100.0, equity=equity, position=position,
        realized_pnl=balance - starting, unrealized_pnl=equity - balance,
        trades=trades, wins=wins, rsi=55.0, macd=0.25, signal_score=0.5, fee_paid=0.0,
    )


def _curve(values: list[float]) -> EquityCurve:
    curve = EquityCurve()
    for value in values:
        curve.record(value)
    return curve


# ── AlgoPerformance ──────────────────────────────────────────────────

def test_win_rate_is_wins_over_trades():
    performance = AlgoPerformance(_status(trades=5, wins=3), _curve([]), 86400.0)

    assert performance.win_rate() == pytest.approx(0.6)


def test_win_rate_of_a_book_that_never_traded_is_zero_not_an_error():
    performance = AlgoPerformance(_status(trades=0, wins=0), _curve([]), 86400.0)

    assert performance.win_rate() == 0.0


def test_max_drawdown_measures_the_worst_peak_to_trough():
    # Peaks at 120, troughs at 90 -> (120 - 90) / 120.
    performance = AlgoPerformance(_status(), _curve([100.0, 120.0, 90.0, 110.0]), 86400.0)

    assert performance.max_drawdown() == pytest.approx(0.25)


def test_total_return_is_measured_against_the_starting_balance():
    performance = AlgoPerformance(_status(equity=1250.0, starting=1000.0), _curve([]), 86400.0)

    assert performance.total_return() == pytest.approx(0.25)


def test_a_short_run_reports_plain_return_rather_than_annualizing_it():
    # Annualizing 60 seconds raises (1 + r) to ~525,960 and overflows.
    performance = AlgoPerformance(_status(equity=1010.0, starting=1000.0), _curve([]), 60.0)

    assert performance.annualized_yield() == pytest.approx(0.01)


def test_annualizing_a_full_year_returns_the_period_return():
    year        = 365.25 * 24 * 3600
    performance = AlgoPerformance(_status(equity=1100.0, starting=1000.0), _curve([]), year)

    assert performance.annualized_yield() == pytest.approx(0.1, rel=1e-6)


def test_annualizing_an_hour_old_book_does_not_overflow():
    # The exponent over one hour is 8766, so any gain above ~8% raises
    # OverflowError — which propagated all the way out of the status endpoint
    # and blanked the whole dashboard, not just this algo.
    performance = AlgoPerformance(_status(equity=1200.0, starting=1000.0), _curve([]), 3600.0)

    assert performance.annualized_yield() == pytest.approx(0.2)


def test_an_unbounded_annualized_yield_is_encoded_as_absent():
    # A real gain over just past the minimum window can still overflow; the
    # payload must stay valid JSON rather than emitting a bare `Infinity`.
    performance = AlgoPerformance(_status(equity=1e6, starting=1000.0), _curve([]), 86400.0)

    assert performance.as_dict()["annualized_yield"] is None
    assert "Infinity" not in json.dumps(performance.as_dict())


def test_a_wiped_out_book_reports_total_loss_rather_than_a_complex_number():
    # 1 + total_return <= 0 would raise a negative base to a fractional power.
    performance = AlgoPerformance(_status(equity=0.0, starting=1000.0), _curve([]), 86400.0)

    assert performance.annualized_yield() == -1.0


# ── EquityCurve ──────────────────────────────────────────────────────

def test_the_curve_drops_the_oldest_samples_past_its_capacity():
    curve = EquityCurve(capacity=3)
    for value in (1.0, 2.0, 3.0, 4.0):
        curve.record(value)

    assert curve.values() == [2.0, 3.0, 4.0]


def test_the_curve_hands_back_a_copy_not_its_own_list():
    curve = EquityCurve()
    curve.record(1.0)

    curve.values().append(99.0)

    assert curve.values() == [1.0]


# ── PortfolioStatus ──────────────────────────────────────────────────

def test_the_portfolio_sums_equity_and_pnl_across_algos():
    portfolio = PortfolioStatus((
        _status(name="a", balance=1100.0, equity=1150.0),
        _status(name="b", balance=900.0, equity=880.0),
    ))

    assert portfolio.equity() == pytest.approx(2030.0)
    assert portfolio.realized_pnl() == pytest.approx(0.0)      # +100 and -100
    assert portfolio.unrealized_pnl() == pytest.approx(30.0)   # +50 and -20


def test_the_portfolio_win_rate_pools_trades_rather_than_averaging_rates():
    # 3/4 and 1/6 pooled is 4/10, not the mean of 0.75 and 0.167.
    portfolio = PortfolioStatus((
        _status(name="a", trades=4, wins=3),
        _status(name="b", trades=6, wins=1),
    ))

    assert portfolio.win_rate() == pytest.approx(0.4)


def test_an_errored_algo_is_counted_but_does_not_break_the_portfolio():
    portfolio = PortfolioStatus((
        _status(name="ok"),
        _status(name="bad", running=False, error="boom", equity=0.0, balance=0.0),
    ))

    assert portfolio.as_dict()["algos_ok"] == 1
    assert portfolio.as_dict()["algos_errored"] == 1


# ── StatusPayload ────────────────────────────────────────────────────

def _payload(statuses, curves=None) -> dict:
    curves = curves or [_curve([]) for _ in statuses]
    return StatusPayload(
        statuses=tuple(statuses),
        performances=tuple(
            AlgoPerformance(status, curve, 86400.0)
            for status, curve in zip(statuses, curves)
        ),
        portfolio=PortfolioStatus(tuple(statuses)),
        started_at="2026-08-26T00:00:00+00:00",
        generated_at="2026-08-26T12:00:00+00:00",
        next_tick_in=120.0,
        seconds_since_price=5.0,
    ).as_dict()


def test_the_payload_is_json_serializable():
    payload = _payload([_status()])

    assert json.loads(json.dumps(payload))["algos"][0]["name"] == "btc"


def test_the_payload_reports_raw_indicators_and_never_normalized_ones():
    # norm_* is min-max scaled per pair's own trailing window, so it is not
    # comparable across pairs — it stays strategy input, never display.
    payload = _payload([_status()])
    algo    = payload["algos"][0]

    assert (algo["rsi"], algo["macd"]) == (55.0, 0.25)
    assert not [key for key in algo if key.startswith("norm_")]


def test_an_infinite_liquidation_price_is_encoded_as_absent_not_as_infinity():
    # json.dumps would emit a bare `Infinity`, which JSON.parse rejects and
    # which would take the whole dashboard down.
    position = PositionView(
        direction="SHORT", size=0.0, entry_price=100.0,
        unrealized=0.0, unrealized_return=0.0, liquidation_price=math.inf,
    )
    payload = _payload([_status(position=position)])

    assert payload["algos"][0]["position"]["liquidation_price"] is None
    assert "Infinity" not in json.dumps(payload)


def test_the_payload_carries_the_position_when_one_is_open():
    position = PositionView(
        direction="LONG", size=2.5, entry_price=100.0,
        unrealized=25.0, unrealized_return=0.1, liquidation_price=0.0,
    )
    payload = _payload([_status(position=position)])

    assert payload["algos"][0]["position"]["direction"] == "LONG"
    assert payload["algos"][0]["position"]["size"] == pytest.approx(2.5)


def test_a_flat_algo_reports_no_position():
    assert _payload([_status()])["algos"][0]["position"] is None
