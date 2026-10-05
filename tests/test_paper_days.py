"""Replication: our trades on the paper's example days (Figures 2, 4, 5) and its yearly returns."""

import pytest

from src.config import load_research, load_strategies
from src.engine.backtest import MarketData, run
from src.evaluation.replication import paper_yearly

R = load_research()
PAPER_YEARLY = paper_yearly().loc[2016:2022]   # paper's Q24 table, final strategy


@pytest.fixture(scope="module")
def results(real_market):
    data = MarketData.load()
    return {name: run(cfg, R, data, "2021-06-01", "2022-12-31")
            for name, cfg in load_strategies().items()}


def trades_on(results, name, day):
    t = results[name].trades
    return t[t["date"] == day].reset_index(drop=True)


def test_jan_20_2022_base_long_reversal_loses_about_2_19pct(results):
    """Fig. 4: long after the morning rally, reversal below the noise area, -2.19%."""
    t = trades_on(results, "base", "2022-01-20")
    assert list(t["side"]) == [1] and t.loc[0, "entry_time"] == "10:00"
    assert results["base"].daily.loc["2022-01-20", "ret"] == pytest.approx(-0.0219, abs=0.001)


def test_jan_20_2022_vwap_stop_exits_early_near_break_even(results):
    """Fig. 5b: VWAP trailing stop closes the long around 13:00 near break-even.
    (Our data puts the 13:00 price a hair above VWAP, so the exit comes at 13:30.)"""
    t = trades_on(results, "vwap_stop", "2022-01-20")
    assert t.loc[0, "side"] == 1 and t.loc[0, "exit_time"] in ("13:00", "13:30")
    assert abs(results["vwap_stop"].daily.loc["2022-01-20", "ret"]) < 0.005


@pytest.mark.parametrize("day, side", [("2022-01-31", 1), ("2022-04-29", -1)])
def test_fig2_trend_days_enter_at_10_30_and_hold_to_close(results, day, side):
    """Fig. 2: the breakout is acted on at 10:30 and held to the close."""
    t = trades_on(results, "base", day)
    assert len(t) == 1
    assert (t.loc[0, "side"], t.loc[0, "entry_time"], t.loc[0, "exit_time"]) == (side, "10:30", "close")


def test_yearly_returns_track_the_paper(real_market):
    r = run(load_strategies()["final"], R, MarketData.load(), "2016-01-01", "2022-12-31")
    ours = (1 + r.daily["ret"]).groupby(r.daily.index.year).prod() - 1
    assert ours.corr(PAPER_YEARLY) > 0.95
    assert (ours - PAPER_YEARLY).abs().max() < 0.10
