"""Entry features must only use information available at the decision time."""

import numpy as np

from src.engine.backtest import MarketData
from src.evaluation.trade_features import intraday_state
from tests.test_lookahead import scramble_after

KS = np.arange(30, 390, 30)   # decisions 10:00 ... 15:30


def test_intraday_state_has_no_lookahead(real_market):
    minute, daily = real_market
    day_i, last_known_bar = 40, 149                          # cutoff: end of the 11:59 bar
    day = daily.index[day_i]
    cutoff = minute.index[(minute["date"] == day) & (minute["minute"] == last_known_bar)][0]

    base = intraday_state(MarketData(minute, daily, None, None), KS)
    scrambled = intraday_state(MarketData(*scramble_after(minute, daily, cutoff, seed=7), None, None), KS)

    known_decisions = KS - 1 <= last_known_bar               # 10:00 ... 12:00 on the cutoff day
    for name in base:
        np.testing.assert_allclose(base[name][:day_i], scrambled[name][:day_i], equal_nan=True, err_msg=name)
        np.testing.assert_allclose(base[name][day_i, known_decisions], scrambled[name][day_i, known_decisions],
                                   equal_nan=True, err_msg=name)
    # and the scramble does reach later decisions, so the test has teeth
    assert not np.allclose(base["rel_volume_30m"][day_i, ~known_decisions],
                           scrambled["rel_volume_30m"][day_i, ~known_decisions], equal_nan=True)
