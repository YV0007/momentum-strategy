"""Hand-made price paths give the expected trades."""

import numpy as np
import pytest

from src import features
from src.config import ResearchConfig, StrategyConfig
from src.engine.backtest import MarketData, run
from tests.conftest import make_market

RESEARCH = ResearchConfig(train_start="2021-01-01", train_end="2021-12-31", test_start="2022-01-01",
                          post_publication_start="2022-01-01", initial_aum=100_000,
                          commission=0.0035, slippage=0.001, trading_days=252, risk_free_rate=0.0)
BASE = StrategyConfig("base", stop="opposite_band", sizing="fixed")
VWAP = StrategyConfig("vwap", stop="band_vwap", sizing="fixed")

m = np.arange(390)
STEADY_CLIMB = 1 + 0.0002 * m
UP_THEN_CRASH = np.where(m < 150, 1 + 0.0002 * m, 1.03 - 0.0004 * (m - 150))
FLAT = np.ones(390)


def last_day_result(path, config, n_days=20):
    minute, daily = make_market(n_days=n_days, seed=1, last_day=path)
    minute_feats, daily_feats = features.build_features(minute, daily)
    data = MarketData(minute, daily, minute_feats, daily_feats)
    result = run(config, RESEARCH, data)
    last = daily.index[-1]
    return result, result.trades[result.trades["date"] == last], minute[minute["date"] == last], daily.loc[last]


@pytest.mark.parametrize("config", [BASE, VWAP])
def test_steady_climb_is_one_long_from_10_to_close(config):
    result, trades, bars, day = last_day_result(STEADY_CLIMB, config)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert (t.side, t.entry_time, t.exit_time) == (1, "10:00", "close")
    assert t.entry_price == pytest.approx(bars["open"].iloc[30])
    assert t.exit_price == pytest.approx(day["close"])


def test_flat_day_has_no_trades():
    for config in (BASE, VWAP):
        _, trades, _, _ = last_day_result(FLAT, config)
        assert trades.empty


def test_base_flips_long_to_short_on_opposite_band():
    _, trades, _, _ = last_day_result(UP_THEN_CRASH, BASE)
    assert list(trades["side"]) == [1, -1]
    assert trades.iloc[0]["exit_time"] == trades.iloc[1]["entry_time"]
    assert trades.iloc[1]["exit_time"] == "close"


def test_vwap_stop_exits_long_before_the_base_model_would():
    _, base_trades, _, _ = last_day_result(UP_THEN_CRASH, BASE)
    _, vwap_trades, _, _ = last_day_result(UP_THEN_CRASH, VWAP)
    base_exit, vwap_exit = base_trades.iloc[0]["exit_time"], vwap_trades.iloc[0]["exit_time"]
    assert vwap_trades.iloc[0]["side"] == 1
    assert vwap_exit < base_exit
    assert vwap_trades.iloc[0]["pnl"] > base_trades.iloc[0]["pnl"]


def test_costs_and_pnl_of_a_single_trade():
    result, trades, bars, day = last_day_result(STEADY_CLIMB, BASE)
    last = result.daily.iloc[-1]
    shares = int(last["aum_start"] // day["open"])
    entry = bars["open"].iloc[30]
    assert last["shares"] == shares
    assert last["costs"] == pytest.approx(2 * shares * 0.0045)
    assert last["pnl_gross"] == pytest.approx(shares * (day["close"] - entry))


def test_vol_target_leverage_is_capped():
    capped = StrategyConfig("final", stop="band_vwap", sizing="vol_target", target_vol=1.0, max_leverage=4)
    result, _, _, day = last_day_result(STEADY_CLIMB, capped)
    last = result.daily.iloc[-1]
    assert last["leverage"] == 4
    assert last["shares"] == int(last["aum_start"] * 4 // day["open"])


def test_band_only_stop_exits_when_price_falls_back_inside_the_band():
    config = StrategyConfig("band", stop="band", sizing="fixed")
    _, trades, _, _ = last_day_result(UP_THEN_CRASH, config)
    _, base_trades, _, _ = last_day_result(UP_THEN_CRASH, BASE)
    assert trades.iloc[0]["side"] == 1
    assert trades.iloc[0]["exit_time"] <= base_trades.iloc[0]["exit_time"]


def test_vwap_only_stop_holds_a_long_until_price_crosses_vwap():
    config = StrategyConfig("vwap", stop="vwap", sizing="fixed")
    _, trades, _, _ = last_day_result(STEADY_CLIMB, config)
    t = trades.iloc[0]
    assert (t.side, t.entry_time, t.exit_time) == (1, "10:00", "close")


def test_without_gap_adjustment_both_bands_are_built_around_the_open():
    minute, daily = make_market(n_days=20, seed=4)
    sigma = features.noise_sigma(minute, daily)
    bands = features.noise_bands(minute, daily, sigma, gap_adjust=False)
    last = daily.index[-1]
    row = (minute["date"] == last) & (minute["minute"] == 100)
    o, s = daily.loc[last, "open"], sigma[row].item()
    assert bands.loc[row, "upper"].item() == pytest.approx(o * (1 + s))
    assert bands.loc[row, "lower"].item() == pytest.approx(o * (1 - s))
