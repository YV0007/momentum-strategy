"""Each feature checked against a slow, obvious re-implementation of the paper's formula."""

import numpy as np
import pandas as pd
import pytest

from src import features
from tests.conftest import make_market


@pytest.fixture(scope="module")
def market():
    return make_market(n_days=30, dividend_days={"2021-01-25": 0.8})


def test_sigma_matches_brute_force(market):
    minute, daily = market
    sigma = features.noise_sigma(minute, daily, lookback=14)
    dates = daily.index
    for day_i, m in [(14, 0), (20, 29), (29, 389)]:
        moves = []
        for prev in dates[day_i - 14:day_i]:
            bars = minute[minute["date"] == prev]
            moves.append(abs(bars["close"].iloc[m] / bars["open"].iloc[0] - 1))
        row = (minute["date"] == dates[day_i]) & (minute["minute"] == m)
        assert sigma[row].item() == pytest.approx(np.mean(moves))


def test_sigma_skips_invalid_days(market):
    minute, daily = market
    daily = daily.copy()
    bad_day = daily.index[10]
    daily.loc[bad_day, "is_valid"] = False
    sigma = features.noise_sigma(minute, daily, lookback=14)
    # Day 15's window is days 1..14 and contains the bad day: only 13 days are averaged.
    day, m = daily.index[15], 100
    moves = [abs(minute.loc[minute["date"] == d, "close"].iloc[m] / daily.loc[d, "open"] - 1)
             for d in daily.index[1:15] if d != bad_day]
    row = (minute["date"] == day) & (minute["minute"] == m)
    assert len(moves) == 13
    assert sigma[row].item() == pytest.approx(np.mean(moves))


def test_bands_use_gap_and_dividend_adjusted_close(market):
    minute, daily = market
    sigma = features.noise_sigma(minute, daily)
    bands = features.noise_bands(minute, daily, sigma, vm=1.5)
    for day in [daily.index[20], pd.Timestamp("2021-01-25")]:  # second one is ex-dividend
        o, pc = daily.loc[day, "open"], daily.loc[day, "prev_close_adj"]
        row = (minute["date"] == day) & (minute["minute"] == 200)
        s = sigma[row].item()
        assert bands.loc[row, "upper"].item() == pytest.approx(max(o, pc) * (1 + 1.5 * s))
        assert bands.loc[row, "lower"].item() == pytest.approx(min(o, pc) * (1 - 1.5 * s))
    ex = pd.Timestamp("2021-01-25")
    assert daily.loc[ex, "prev_close_adj"] == pytest.approx(daily.loc[ex, "prev_close"] - 0.8)


def test_vwap_matches_brute_force(market):
    minute, _ = market
    v = features.vwap(minute)
    day = minute[minute["date"] == minute["date"].iloc[0]]
    for m in [0, 45, 389]:
        bars = day.iloc[: m + 1]
        expected = (bars["vwap_bar"] * bars["volume"]).sum() / bars["volume"].sum()
        assert v.loc[day.index[m]] == pytest.approx(expected)


def test_daily_volatility_matches_brute_force(market):
    _, daily = market
    vol = features.daily_volatility(daily, lookback=14)
    day_i = 25
    expected = np.std(daily["ret_cc"].iloc[day_i - 14:day_i], ddof=1)
    assert vol.iloc[day_i] == pytest.approx(expected)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 120, 30))
    down = pd.Series(np.linspace(120, 100, 30))
    assert features.rsi(up).iloc[-1] == pytest.approx(100)
    assert features.rsi(down).iloc[-1] == pytest.approx(0)
