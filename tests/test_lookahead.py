"""Look-ahead test: scramble everything after a cutoff moment and rebuild the features.
Anything computed for a moment at or before the cutoff must not change.

At a cutoff inside day D, these are still unknown and get scrambled:
  - every minute bar after the cutoff (same day and later days),
  - day D's end-of-day data: close, high, low, VIX close, validity flag,
  - everything about later days, including their open, dividends and VIX open.
"""

import numpy as np
import pandas as pd
import pytest

from src import features
from src.data.clean import add_returns


def scramble_after(minute: pd.DataFrame, daily: pd.DataFrame, cutoff: pd.Timestamp, seed: int):
    rng = np.random.default_rng(seed)
    minute, daily = minute.copy(), daily.copy()
    day = cutoff.tz_localize(None).normalize()

    later_bars = minute.index > cutoff
    shock = rng.uniform(0.9, 1.1, later_bars.sum())
    for col in ["open", "high", "low", "close", "vwap_bar"]:
        minute.loc[later_bars, col] *= shock
    minute.loc[later_bars, "volume"] = rng.integers(1, 1_000_000, later_bars.sum())

    today, later = daily.index == day, daily.index > day
    for col in ["close", "high", "low", "vix_close"]:
        daily.loc[today | later, col] *= rng.uniform(0.9, 1.1, (today | later).sum())
    for col in ["open", "vix_open"]:
        daily.loc[later, col] *= rng.uniform(0.9, 1.1, later.sum())
    daily.loc[later, "dividend"] = rng.uniform(0, 2, later.sum())
    daily.loc[today | later, "is_valid"] = rng.random((today | later).sum()) > 0.5
    return minute, add_returns(daily)


@pytest.mark.parametrize("day_i, minute_i, seed", [(20, 0, 1), (30, 29, 2), (40, 200, 3), (55, 389, 4)])
def test_no_lookahead(real_market, day_i, minute_i, seed):
    minute, daily = real_market
    day = daily.index[day_i]
    cutoff = minute.index[(minute["date"] == day) & (minute["minute"] == minute_i)][0]

    base_min, base_day = features.build_features(minute, daily)
    pert_min, pert_day = features.build_features(*scramble_after(minute, daily, cutoff, seed))

    # Sanity: the scramble really changed later features, so the test has teeth.
    assert not base_min.loc[base_min.index > cutoff, "vwap"].equals(pert_min.loc[pert_min.index > cutoff, "vwap"])

    known = base_min.index <= cutoff
    pd.testing.assert_frame_equal(base_min[known], pert_min[known])
    pd.testing.assert_frame_equal(base_day.loc[:day], pert_day.loc[:day])
