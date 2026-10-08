"""The paper's Section 4 and FAQ analyses on our data."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.engine.backtest import BacktestResult
from src.evaluation.diagnostics import BPS, side_contributions, with_contribution
from src.features import narrow_range


def daily_patterns(daily: pd.DataFrame) -> pd.DataFrame:
    high, low, open_, close = daily["high"], daily["low"], daily["open"], daily["close"]
    day_range = high - low
    open_pos, close_pos = (open_ - low) / day_range, (close - low) / day_range
    wide = day_range > day_range.rolling(14).mean().shift(1)
    patterns = pd.DataFrame({
        "NR4": narrow_range(daily, 4),
        "NR7": narrow_range(daily, 7),
        "ID (inside day)": (high < high.shift(1)) & (low > low.shift(1)),
        "OD (outside day)": (high > high.shift(1)) & (low < low.shift(1)),
        "Triangle": (high < high.shift(1)) & (high < high.shift(2)) & (low > low.shift(1)) & (low > low.shift(2)),
        "Trend day": wide & (((open_pos < 0.15) & (close_pos > 0.85)) | ((open_pos > 0.85) & (close_pos < 0.15))),
        "Big tail": ((open_pos > 0.75) & (close_pos > 0.75)) | ((open_pos < 0.25) & (close_pos < 0.25)),
        "Strong/weak closure": (close_pos > 0.9) | (close_pos < 0.1),
    })
    return patterns.shift(1, fill_value=False)


def conditional_stats(ret: pd.Series, traded: pd.Series, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, mask in {"Unconditional": pd.Series(True, index=ret.index), **groups}.items():
        r = ret[traded & mask.reindex(ret.index, fill_value=False)]
        rows[name] = {"observations": len(r), "avg_bps": r.mean() * BPS,
                      "t_stat": r.mean() / r.std() * np.sqrt(len(r)), "hit_ratio": (r > 0).mean(),
                      "sharpe": r.mean() / r.std() * np.sqrt(252)}
    return pd.DataFrame(rows).T


def trade_stats(result: BacktestResult) -> dict:
    t = with_contribution(result)
    reversals = (t["entry_time"] == t.groupby("date")["exit_time"].shift()).sum()
    worst, best = t.loc[t["contribution"].idxmin()], t.loc[t["contribution"].idxmax()]
    return {"trades": len(t), "trades_per_day": len(t) / len(result.daily),
            "orders_per_day": (2 * len(t) - reversals) / len(result.daily), "hit_ratio": (t["pnl"] > 0).mean(),
            "avg_pnl_per_share": (t["pnl"] / t["shares"]).mean(),
            "max_loss_trade": f"{worst['contribution']:.1%} ({worst['date']:%Y-%m-%d})",
            "max_gain_trade": f"{best['contribution']:.1%} ({best['date']:%Y-%m-%d})"}


def pnl_per_share_by_year(result: BacktestResult) -> pd.Series:
    t = result.trades
    return (t["pnl"] / t["shares"]).groupby(t["date"].dt.year).mean()


def legs(result: BacktestResult) -> pd.DataFrame:
    sides = side_contributions(result)
    return sides.assign(both=result.daily["ret"])


def compounded(ret: pd.Series) -> float:
    return (1 + ret).prod() - 1


def shorts_above_vix(legs_ret: pd.DataFrame, vix_open: pd.Series, thresholds: list[float]) -> pd.Series:
    return pd.Series({f">= {t:g}": compounded(legs_ret["short"].where(vix_open >= t, 0.0)) for t in thresholds})


def shorts_below_sma(legs_ret: pd.DataFrame, close: pd.Series, windows: list[int]) -> pd.DataFrame:
    rows = {"all shorts (strategy as is)": legs_ret["both"]}
    for n in windows:
        bear = (close < close.rolling(n).mean()).shift(1, fill_value=False).reindex(legs_ret.index)
        rows[f"shorts only below SMA{n}"] = legs_ret["long"] + legs_ret["short"].where(bear, 0.0)
    return pd.DataFrame({name: {"total_return": compounded(r), "sharpe": r.mean() / r.std() * np.sqrt(252)}
                         for name, r in rows.items()}).T


def short_trades_vs_vix(result: BacktestResult, vix_mean: pd.Series) -> dict:
    t = result.trades[result.trades["side"] < 0]
    bps = -(t["exit_price"] / t["entry_price"] - 1) * BPS
    fit = sm.OLS(bps.to_numpy(), sm.add_constant(t["date"].map(vix_mean).to_numpy())).fit()
    return {"short_trades": len(t), "beta_bps_per_vix_point": fit.params[1], "t_stat": fit.tvalues[1],
            "p_value": fit.pvalues[1]}


def worst_quarters(ret: pd.Series, spy_ret: pd.Series, k: int = 10) -> pd.DataFrame:
    q = pd.DataFrame({"spy": spy_ret, "strategy": ret}).groupby(ret.index.to_period("Q")).apply(
        lambda g: (1 + g).prod() - 1)
    return q.nsmallest(k, "spy")
