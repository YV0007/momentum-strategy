"""Performance metrics from daily returns."""

import numpy as np
import pandas as pd


def annual_return(ret: pd.Series, n: int = 252) -> float:
    return (1 + ret).prod() ** (n / len(ret)) - 1


def annual_volatility(ret: pd.Series, n: int = 252) -> float:
    return ret.std() * np.sqrt(n)


def sharpe(ret: pd.Series, n: int = 252, rf: float = 0.0) -> float:
    excess = ret - rf / n
    return excess.mean() / excess.std() * np.sqrt(n)


def max_drawdown(ret: pd.Series) -> float:
    equity = (1 + ret).cumprod()
    return (1 - equity / equity.cummax()).max()


def hit_ratio(ret: pd.Series, traded: pd.Series) -> float:
    return (ret[traded] > 0).mean()


def return_at_vol(ret: pd.Series, target: float = 0.10, n: int = 252) -> float:
    return annual_return(ret * target / annual_volatility(ret, n), n)


def summary(daily: pd.DataFrame, n: int = 252, rf: float = 0.0) -> dict:
    ret = daily["ret"]
    traded = daily["trades"] > 0
    if not traded.any():
        traded[:] = True
    out = {
        "total_return": (1 + ret).prod() - 1,
        "annual_return": annual_return(ret, n),
        "annual_volatility": annual_volatility(ret, n),
        "sharpe": sharpe(ret, n, rf),
        "max_drawdown": max_drawdown(ret),
        "hit_ratio": hit_ratio(ret, traded),
        "return_at_10pct_vol": return_at_vol(ret, 0.10, n),
        "skew": ret.skew(),
        "worst_day": ret.min(),
        "best_day": ret.max(),
        "trades": int(daily["trades"].sum()),
        "days": len(daily),
    }
    if "costs" in daily:
        out["cost_share_of_gross"] = daily["costs"].sum() / daily["pnl_gross"].sum()
    return out
