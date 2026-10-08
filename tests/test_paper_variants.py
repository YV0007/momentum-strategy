"""Alternative cost models and daily patterns."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_research, load_strategies
from src.engine import costs
from src.engine.backtest import MarketData, run
from src.evaluation.paper_tables import daily_patterns

R = load_research()


def test_istar_impact_formula():
    istar = replace(R, slippage_model="istar")
    fills, prices = np.array([1_000.0, 0.0, 2_000.0]), np.array([400.0, 401.0, 410.0])
    expected = sum(687 * (q / 1e8) ** 0.70 * 0.2 ** 0.72 / 1e4 * p * q for q, p in zip(fills, prices))
    assert costs.slippage(istar, fills, prices, adv=1e8, vol_annual=0.2) == pytest.approx(expected)
    assert costs.slippage(istar, np.zeros(3), prices, adv=np.nan, vol_annual=np.nan) == 0.0
    assert costs.slippage(R, fills, prices, adv=1e8, vol_annual=0.2) == pytest.approx(R.slippage * 3_000)


def test_tiered_commission_threshold():
    tiered = replace(R, commission_tiered=True)
    assert costs.commission_rate(tiered, 300_001) == 0.002
    assert costs.commission_rate(tiered, 300_000) == R.commission
    assert costs.commission_rate(R, 10_000_000) == R.commission


@pytest.fixture(scope="module")
def data():
    try:
        return MarketData.load()
    except FileNotFoundError:
        pytest.skip("processed data not built")


def test_fill_by_fill_accounting_matches_the_fast_path(data):
    final = load_strategies()["final"]
    fast = run(final, R, data, "2016-01-01", "2017-12-31").daily
    slow = run(final, replace(R, commission_tiered=True), data, "2016-01-01", "2017-12-31").daily
    assert slow["shares_traded"].rolling(costs.TIER_DAYS).sum().max() < costs.TIER_SHARES
    np.testing.assert_allclose(slow["aum"], fast["aum"])


def test_istar_backtest_reconciles(data):
    result = run(load_strategies()["final"], replace(R, slippage_model="istar"), data, "2016-01-01", "2017-12-31")
    d = result.daily
    assert result.trades["pnl"].sum() == pytest.approx((d["pnl_gross"] - d["costs"]).sum())
    assert (d["costs"] >= d["shares_traded"] * R.commission - 1e-9).all()


def test_daily_patterns_are_flagged_the_next_day():
    daily = pd.DataFrame({"high": [10.0, 11.0, 12.0, 10.9, 13.0, 12.0],
                          "low": [9.0, 9.5, 10.0, 10.5, 9.0, 11.0],
                          "open": [9.5, 10.0, 11.0, 10.7, 12.8, 11.5],
                          "close": [9.5, 10.25, 11.0, 10.7, 12.9, 11.5]},
                         index=pd.bdate_range("2024-01-01", periods=6))
    flagged_on = {name: list(col[col].index.day) for name, col in daily_patterns(daily).items()}
    assert flagged_on["ID (inside day)"] == [5]
    assert flagged_on["Triangle"] == [5]
    assert flagged_on["NR4"] == [5]
    assert flagged_on["OD (outside day)"] == [8]
    assert flagged_on["Big tail"] == [8]
    assert flagged_on["Strong/weak closure"] == [8]
    assert flagged_on["Trend day"] == []
