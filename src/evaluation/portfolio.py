"""Combine single-asset sleeves into one equal-capital portfolio. (Phase 9)

Every sleeve runs the strategy on its own ETF with the same rules, its own volatility target, the
4x cap and the same capital. The strategy is flat overnight, so rebalancing to equal capital every
morning costs nothing: the portfolio's daily return is the average of the sleeves' daily returns
(a sleeve that does not trade that day contributes 0).
"""

import numpy as np
import pandas as pd


def combine(sleeves: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame(sleeves).fillna(0.0).mean(axis=1).rename("ret")


def diversification(sleeves: dict[str, pd.Series]) -> dict:
    """Average pairwise correlation of the sleeves' daily returns, and the number of independent bets
    it is worth for equally risky sleeves: N / (1 + (N - 1) * average correlation)."""
    corr = pd.DataFrame(sleeves).fillna(0.0).corr().to_numpy()
    n = len(corr)
    avg = (corr.sum() - n) / (n * (n - 1))
    return {"assets": n, "average_correlation": avg, "independent_bets": n / (1 + (n - 1) * avg)}
