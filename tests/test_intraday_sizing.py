"""Per-trade sizing, its accounting and the ML model's data."""

import numpy as np
import pytest

from src import features
from src.config import load_research, load_strategies
from src.engine.backtest import MarketData, hold_from_entry, prepare, run, simulate
from src.intraday import PANEL_FEATURES, decision_panel
from tests.test_lookahead import scramble_after

R = load_research()
S = load_strategies()
KS = np.arange(30, 390, 30)


def test_size_is_fixed_at_entry_and_held_to_exit():
    position = np.array([[0, 1, 1, 1, 0, -1, -1, 1, 1]], dtype=float)
    size = np.array([[9, 0.5, 2, 3, 9, 0.7, 4, 1.2, 5]], dtype=float)
    expected = np.array([[0, 0.5, 0.5, 0.5, 0, -0.7, -0.7, 1.2, 1.2]])
    np.testing.assert_allclose(hold_from_entry(position, size), expected)


@pytest.fixture(scope="module")
def data():
    try:
        return MarketData.load()
    except FileNotFoundError:
        pytest.skip("processed data not built")


def test_half_size_trades_hold_half_the_shares(data):
    full = run(S["final"], R, data, "2016-01-01", "2017-12-31")
    prep = prepare(S["final"], data, "2016-01-01", "2017-12-31", multiplier=np.full((len(data.daily), len(KS)), 0.5))
    half = simulate(prep, prep.positions(), R)
    t_full, t_half = full.trades, half.trades
    assert len(t_full) == len(t_half)
    day_shares = t_half["date"].map(half.daily["shares"])
    np.testing.assert_array_equal(t_half["shares"], np.floor(day_shares * 0.5))
    expected = t_half["shares"] * t_half["side"] * (t_half["exit_price"] - t_half["entry_price"]) \
        - 2 * t_half["shares"] * R.cost_per_share
    np.testing.assert_allclose(t_half["pnl"], expected)


@pytest.fixture(scope="module")
def sized(data):
    return run(S["own_turbulence"], R, data, "2016-01-01", "2019-12-31")


def test_sized_backtest_reconciles(sized):
    d, t = sized.daily, sized.trades
    assert d["aum"].iloc[-1] == pytest.approx(R.initial_aum + (d["pnl_gross"] - d["costs"]).sum())
    assert t["pnl"].sum() == pytest.approx((d["pnl_gross"] - d["costs"]).sum())
    assert d["trades"].sum() == len(t)
    assert (t["shares"] % 1 == 0).all()


def test_sized_backtest_respects_bounds_and_leverage_cap(sized, data):
    d, t, cfg = sized.daily, sized.trades, sized.config
    assert t["size"].between(cfg.size_floor - 1e-12, cfg.size_cap + 1e-12).all()
    assert (t["size"] < 1).any() and (t["size"] > 1).any()
    notional = d["peak_shares"] * data.daily.loc[d.index, "open"]
    assert (notional <= d["aum_start"] * cfg.max_leverage + 1e-6).all()


def test_same_trades_as_final(sized, data):
    final = run(S["final"], R, data, "2016-01-01", "2019-12-31").trades
    cols = ["date", "entry_time", "exit_time", "side"]
    assert final[cols].equals(sized.trades[cols])


def test_decision_panel_has_no_lookahead(real_market):
    minute, daily = real_market
    day_i, last_known_bar = 40, 149
    day = daily.index[day_i]
    cutoff = minute.index[(minute["date"] == day) & (minute["minute"] == last_known_bar)][0]

    def panel(m, d):
        min_feats, day_feats = features.build_features(m, d)
        return decision_panel(MarketData(m, d, min_feats, day_feats), KS)

    base = panel(minute, daily)
    scrambled = panel(*scramble_after(minute, daily, cutoff, seed=11))
    known = np.zeros(len(base), bool)
    known[: day_i * len(KS)] = True
    known[day_i * len(KS): day_i * len(KS) + len(KS)] = KS - 1 <= last_known_bar
    np.testing.assert_allclose(base.loc[known, PANEL_FEATURES], scrambled.loc[known, PANEL_FEATURES], equal_nan=True)
    today = base.index.get_level_values("date") == day
    assert not np.allclose(base.loc[today, "target"], scrambled.loc[today, "target"], equal_nan=True)


def test_frozen_ml_model_never_sees_the_test_period(data):
    from src.strategies import ml_sizing
    if not ml_sizing.SETTINGS_FILE.exists():
        pytest.skip("config/ml_sizing.yaml not written yet (scripts/05a_tune_ml_sizing.py --write)")
    end = R.train_end
    cut = MarketData(data.minute[data.minute["date"] <= end], data.daily.loc[:end],
                     data.minute_feats[data.minute["date"].to_numpy() <= np.datetime64(end)],
                     data.daily_feats.loc[:end])
    full_fc = ml_sizing.frozen_forecast(data, KS)
    cut_fc = ml_sizing.frozen_forecast(cut, KS)
    np.testing.assert_allclose(full_fc[: len(cut.daily)], cut_fc, equal_nan=True)
