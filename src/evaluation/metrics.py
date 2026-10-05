"""Performance metrics on daily returns. (Phase 4; core set needed from Phase 3 on)

Definitions (daily returns r, N = 252 trading days per year):
  annual return      geometric: (prod(1 + r)) ** (N / days) - 1
  annual volatility  std(r) * sqrt(N)
  Sharpe ratio       (mean(r) - rf/N) / std(r) * sqrt(N)
  max drawdown       largest peak-to-trough fall of the equity curve
  hit ratio          share of traded days with a positive return
  return at 10% vol  annual return after scaling daily returns to 10% annual volatility,
                     so strategies with different leverage compare like for like
                     (an ex-post comparison device, not a tradable strategy)
"""

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
    """Headline metrics from a backtest's daily table (strategy or baseline)."""
    ret = daily["ret"]
    traded = daily["trades"] > 0
    if not traded.any():          # always-invested benchmark: every day counts
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
    if "costs" in daily:   # costs as a share of gross trading profit
        out["cost_share_of_gross"] = daily["costs"].sum() / daily["pnl_gross"].sum()
    return out
