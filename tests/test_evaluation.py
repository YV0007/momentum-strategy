"""Metrics, statistics and the placebo, checked on cases with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import baselines, metrics, stats
from src.evaluation.split import walk_forward_folds


def test_metrics_on_a_known_series():
    ret = pd.Series([0.01, -0.01] * 126)                   # one year, alternating
    assert metrics.annual_volatility(ret) == pytest.approx(ret.std() * np.sqrt(252))
    assert metrics.max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert metrics.annual_return(pd.Series([0.0] * 252)) == 0


def test_sharpe_se_close_to_normal_formula_for_normal_returns():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0.0005, 0.01, 5000))
    sr_daily = ret.mean() / ret.std()
    expected = np.sqrt((1 + 0.5 * sr_daily**2) / len(ret)) * np.sqrt(252)
    assert stats.sharpe_se(ret) == pytest.approx(expected, rel=0.05)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.0005, 0.01, 2000))
    lo, hi = stats.bootstrap_ci(ret)
    assert lo < metrics.sharpe(ret) < hi


def test_random_direction_keeps_timing_and_size_of_every_trade():
    position = np.array([[0, 1, 1, 0, -1, -1, 1], [0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, 1, 1, 0, 0]], float)
    for seed in range(20):
        flipped = baselines.random_direction(position, np.random.default_rng(seed))
        assert np.array_equal(np.abs(flipped), np.abs(position))       # same timing and size
        assert flipped[0, 1] == flipped[0, 2] and flipped[0, 4] == flipped[0, 5]   # one sign per trade


def test_placebo_p_value():
    null = np.arange(100.0)
    assert stats.empirical_p_value(1000, null) == pytest.approx(1 / 101)
    assert stats.empirical_p_value(-1, null) == 1


def test_walk_forward_folds_never_train_on_the_future():
    days = pd.bdate_range("2016-01-01", "2022-12-31")
    folds = walk_forward_folds(days, n_folds=4)
    assert len(folds) == 4
    for train, val in folds:
        assert train.max() < val.min()
    assert folds[-1][1].max() == days.max()
