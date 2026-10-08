"""Portfolio maths and the pooled ML model."""

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import MarketData
from src.evaluation import portfolio
from src.evaluation.split import walk_forward_folds
from src.intraday import decision_panel
from src.strategies import ml_sizing

KS = np.arange(30, 390, 30)


def test_portfolio_is_the_average_sleeve_with_idle_days_as_cash():
    days = pd.bdate_range("2024-01-01", periods=3)
    sleeves = {"A": pd.Series([0.02, -0.01, 0.00], index=days), "B": pd.Series([0.00, 0.03], index=days[:2])}
    np.testing.assert_allclose(portfolio.combine(sleeves), [0.01, 0.01, 0.0])


def test_independent_bets_follow_the_average_correlation():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000))
    same = portfolio.diversification({"A": x, "B": x, "C": x})
    assert same["average_correlation"] == pytest.approx(1.0) and same["independent_bets"] == pytest.approx(1.0)
    unrelated = portfolio.diversification({k: pd.Series(rng.normal(size=2000)) for k in "ABCD"})
    assert unrelated["independent_bets"] == pytest.approx(4.0, rel=0.1)


def test_pooled_walk_forward_never_uses_the_block_it_forecasts_or_later():
    try:
        data = MarketData.load()
    except FileNotFoundError:
        pytest.skip("processed data not built")
    days = data.daily.loc["2016-01-01":"2019-12-31"].index
    folds = walk_forward_folds(days, n_folds=2, min_train_years=2)
    panel = decision_panel(data, KS)
    settings = ml_sizing.asset_settings()
    scrambled = panel.copy()
    later = scrambled.index.get_level_values("date") >= folds[0][1][0]
    scrambled.loc[later, "target"] = np.random.default_rng(1).normal(size=later.sum())
    base = ml_sizing.walk_forward_pooled({"A": panel, "B": panel}, folds, settings)
    pert = ml_sizing.walk_forward_pooled({"A": panel, "B": scrambled}, folds, settings)
    first_block = data.daily.index.isin(folds[0][1])
    np.testing.assert_allclose(base["A"][first_block], pert["A"][first_block])
    np.testing.assert_allclose(base["B"][first_block], pert["B"][first_block])
    second_block = data.daily.index.isin(folds[1][1])
    assert not np.allclose(base["B"][second_block], pert["B"][second_block], equal_nan=True)
