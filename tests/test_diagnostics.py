"""Diagnostics reconcile with the backtest."""

import numpy as np
import pandas as pd
import pytest

from src import features
from src.config import StrategyConfig
from src.engine.backtest import MarketData, prepare, simulate
from src.evaluation import diagnostics as dg
from tests.conftest import make_market
from tests.test_synthetic_days import RESEARCH


@pytest.fixture(scope="module")
def backtest():
    minute, daily = make_market(n_days=80, seed=3)
    data = MarketData(minute, daily, *features.build_features(minute, daily))
    prep = prepare(StrategyConfig("final", stop="band_vwap", sizing="fixed"), data)
    return prep, simulate(prep, prep.rule_positions(), RESEARCH)


def test_time_of_day_legs_add_up_to_daily_return(backtest):
    prep, result = backtest
    tod = dg.time_of_day(prep, result, RESEARCH.cost_per_share)
    assert (tod["gross_bps"] - tod["cost_bps"]).sum() == pytest.approx(result.daily["ret"].mean() * dg.BPS)
    assert (tod["long_bps"] + tod["short_bps"]).values == pytest.approx(tod["gross_bps"].values)


def test_long_and_short_contributions_add_up_to_daily_return(backtest):
    _, result = backtest
    assert len(result.trades) > 0
    sides = dg.side_contributions(result)
    assert sides.sum(axis=1).values == pytest.approx(result.daily["ret"].values)


def test_bucket_stats_cover_every_day_once():
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300))
    groups = pd.Series(np.repeat(["a", "b", "c"], 100))
    table = dg.bucket_stats(ret, groups, ret != 0)
    assert table["days"].sum() == 300
    assert table["share_of_pnl"].sum() == pytest.approx(1)


def test_trade_sequence_labels():
    trades = pd.DataFrame({"date": pd.to_datetime(["2021-01-04"] * 3 + ["2021-01-05"]),
                           "entry_time": ["10:00", "11:00", "12:30", "10:30"], "side": [1, 1, -1, -1]})
    t = dg.trade_sequence(trades)
    assert list(t["sequence"]) == ["1st trade", "2nd trade", "3rd+ trade", "1st trade"]
    assert list(t["reentry"]) == ["first of day", "same direction again", "reversed direction", "first of day"]
