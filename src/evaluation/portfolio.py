"""Combines single-ETF results into an equal-capital portfolio."""

import numpy as np
import pandas as pd


def combine(sleeves: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame(sleeves).fillna(0.0).mean(axis=1).rename("ret")


def diversification(sleeves: dict[str, pd.Series]) -> dict:
    corr = pd.DataFrame(sleeves).fillna(0.0).corr().to_numpy()
    n = len(corr)
    avg = (corr.sum() - n) / (n * (n - 1))
    return {"assets": n, "average_correlation": avg, "independent_bets": n / (1 + (n - 1) * avg)}
