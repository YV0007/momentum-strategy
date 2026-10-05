"""Checkpoint diagnostics: where does the edge come from, and where does it fail? (Phase 5)

Run on the TRAIN period only. Bucket tables report the mean daily return with a 95% interval
and t-stat: with a few hundred days per bucket, one bucket's Sharpe has a standard error near
1, so only large, consistent patterns are worth acting on.

All returns are daily strategy returns on start-of-day AUM; days without a trade count as 0.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.engine.backtest import BacktestResult, Prepared, clock

BPS = 1e4


# ---------------------------------------------------------------- conditional returns

def quantile_buckets(x: pd.Series, q: int = 5, fmt: str = "{:.0f}") -> pd.Series:
    """Equal-sized buckets with readable range labels, e.g. '12.1-15.3'."""
    edges = pd.qcut(x, q, retbins=True, duplicates="drop")[1]
    labels = [f"{fmt.format(a)}–{fmt.format(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.qcut(x, q, labels=labels, duplicates="drop")


def bucket_stats(ret: pd.Series, groups: pd.Series, traded: pd.Series, n: int = 252) -> pd.DataFrame:
    """Per group: days, share of days traded, mean daily return (bps) with 95% half-width and
    t-stat, hit ratio on traded days, annualized Sharpe, and share of total P&L."""
    df = pd.DataFrame({"ret": ret, "group": groups, "traded": traded}).dropna(subset=["group"])
    g = df.groupby("group", observed=True)["ret"]
    se = g.std() / np.sqrt(g.size())
    hit = df[df["traded"]].groupby("group", observed=True)["ret"].apply(lambda r: (r > 0).mean())
    return pd.DataFrame({
        "days": g.size(),
        "traded_share": df.groupby("group", observed=True)["traded"].mean(),
        "mean_bps": g.mean() * BPS,
        "ci95_bps": 1.96 * se * BPS,
        "t_stat": g.mean() / se,
        "hit_ratio": hit,
        "sharpe": g.mean() / g.std() * np.sqrt(n),
        "share_of_pnl": g.sum() / df["ret"].sum(),
    })


def above_thresholds(ret: pd.Series, x: pd.Series, thresholds: list[float], n: int = 252) -> pd.DataFrame:
    """Paper Fig. 8 style: performance on the days where x >= threshold."""
    rows = {}
    for t in thresholds:
        r = ret[x >= t]
        rows[f">= {t:g}"] = {"days": len(r), "mean_bps": r.mean() * BPS,
                             "sharpe": r.mean() / r.std() * np.sqrt(n) if len(r) > 1 else np.nan}
    return pd.DataFrame(rows).T


def regression(ret: pd.Series, x: pd.Series, per: float = 1.0) -> dict:
    """ret (bps) = a + b * x with autocorrelation-robust t-stats; slope reported per `per` units."""
    df = pd.concat([ret * BPS, x.astype(float)], axis=1, keys=["y", "x"]).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {"slope_bps": fit.params["x"] * per, "t_stat": fit.tvalues["x"],
            "p_value": fit.pvalues["x"], "days": len(df)}


# ---------------------------------------------------------------- inside the day

def time_of_day(prep: Prepared, result: BacktestResult, cost_per_share: float) -> pd.DataFrame:
    """Average P&L of each half-hour holding leg (10:00-10:30, ..., 15:30-close), in bps of AUM.
    Trading costs are booked on the leg where the trade happens (closing costs on the last leg).
    Summed over legs, gross minus costs equals the average daily return."""
    position = prep.rule_positions()
    shares = result.daily["shares"].to_numpy()[:, None]
    aum = result.daily["aum_start"].to_numpy()[:, None]

    gross = shares * position * np.diff(prep.legs, axis=1) / aum
    prev = np.hstack([np.zeros((len(position), 1)), position[:, :-1]])
    cost = shares * np.abs(position - prev) * cost_per_share / aum
    cost[:, -1] += shares[:, 0] * np.abs(position[:, -1]) * cost_per_share / aum[:, 0]

    times = clock(prep.ks)
    labels = [f"{a}–{b}" for a, b in zip(times, times[1:] + ["close"])]
    n_days = len(position)
    return pd.DataFrame({
        "gross_bps": gross.mean(axis=0) * BPS,
        "ci95_bps": 1.96 * gross.std(axis=0, ddof=1) / np.sqrt(n_days) * BPS,
        "long_bps": np.where(position > 0, gross, 0).mean(axis=0) * BPS,
        "short_bps": np.where(position < 0, gross, 0).mean(axis=0) * BPS,
        "cost_bps": cost.mean(axis=0) * BPS,
        "long_share": (position > 0).mean(axis=0),
        "short_share": (position < 0).mean(axis=0),
    }, index=pd.Index(labels, name="leg"))


# ---------------------------------------------------------------- trades and sides

def with_contribution(result: BacktestResult) -> pd.DataFrame:
    """Trade log plus each trade's net P&L as a return on that day's starting AUM."""
    t = result.trades.copy()
    t["contribution"] = t["pnl"] / t["date"].map(result.daily["aum_start"])
    t["exit_type"] = np.where(t["exit_time"] == "close", "held to close", "stopped out")
    t["side_name"] = t["side"].map({1: "long", -1: "short"})
    return t


def side_contributions(result: BacktestResult) -> pd.DataFrame:
    """Daily return split into the part earned by long trades and by short trades."""
    t = with_contribution(result)
    daily = t.pivot_table(index="date", columns="side_name", values="contribution", aggfunc="sum")
    return daily.reindex(result.daily.index).fillna(0.0)[["long", "short"]]


def trade_breakdown(trades: pd.DataFrame, by: str, n: int = 252) -> pd.DataFrame:
    """Per group of trades: count, hit ratio, average and total contribution."""
    g = trades.groupby(by)["contribution"]
    return pd.DataFrame({"trades": g.size(), "hit_ratio": g.apply(lambda c: (c > 0).mean()),
                         "avg_bps": g.mean() * BPS, "total_contribution": g.sum()})


# ---------------------------------------------------------------- over time and extremes

def yearly(result: BacktestResult, market_ret: pd.Series, vix: pd.Series, max_leverage: float,
           n: int = 252) -> pd.DataFrame:
    d = result.daily.assign(spy=market_ret, vix=vix)
    rows = {}
    for year, g in d.groupby(d.index.year):
        traded = g[g["trades"] > 0]
        equity = (1 + g["ret"]).cumprod()
        rows[year] = {
            "return": equity.iloc[-1] - 1,
            "spy_return": (1 + g["spy"]).prod() - 1,
            "sharpe": g["ret"].mean() / g["ret"].std() * np.sqrt(n),
            "max_drawdown": (1 - equity / equity.cummax()).max(),
            "trades_per_day": g["trades"].mean(),
            "avg_leverage": traded["leverage"].mean(),
            "days_at_cap": (g["leverage"] >= max_leverage).mean(),
            "avg_vix": g["vix"].mean(),
        }
    return pd.DataFrame(rows).T


def extreme_days(result: BacktestResult, daily: pd.DataFrame, daily_feats: pd.DataFrame,
                 k: int = 10, worst: bool = True) -> pd.DataFrame:
    """The k worst (or best) days with market context and what the strategy did."""
    ret = result.daily["ret"]
    days = ret.nsmallest(k).index if worst else ret.nlargest(k).index
    paths = (result.trades.assign(leg=lambda t: t["side"].map({1: "L", -1: "S"}) + " "
                                  + t["entry_time"] + "→" + t["exit_time"])
             .groupby("date")["leg"].agg(", ".join))
    return pd.DataFrame({
        "ret": ret[days],
        "spy_open_to_close": daily.loc[days, "ret_oc"],
        "spy_gap": daily_feats.loc[days, "gap"],
        "vix_open": daily_feats.loc[days, "vix_open"],
        "leverage": result.daily.loc[days, "leverage"],
        "trades": paths.reindex(days),
    })


def concentration(ret: pd.Series) -> dict:
    """How much of the total P&L comes from a few days (typical of trend following)."""
    total = ret.sum()
    ordered = ret.sort_values(ascending=False)
    n = len(ret)
    return {"best_1pct_days_share": ordered.iloc[: max(1, n // 100)].sum() / total,
            "best_5pct_days_share": ordered.iloc[: max(1, n // 20)].sum() / total,
            "total_return": (1 + ret).prod() - 1,
            "total_return_without_best_10_days": (1 + ordered.iloc[10:]).prod() - 1}


def trade_sequence(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades labeled by their order within the day and how they relate to the previous trade:
    the marginal value of re-entering after a stop-out."""
    t = trades.sort_values(["date", "entry_time"]).copy()
    number = t.groupby("date").cumcount() + 1
    t["sequence"] = number.clip(upper=3).map({1: "1st trade", 2: "2nd trade", 3: "3rd+ trade"})
    prev_side = t.groupby("date")["side"].shift(1)
    t["reentry"] = np.where(prev_side.isna(), "first of day",
                            np.where(prev_side == t["side"], "same direction again", "reversed direction"))
    return t


def stop_counterfactual(trades: pd.DataFrame, result: BacktestResult, close: pd.Series,
                        cost_per_share: float) -> pd.DataFrame:
    """Stopped-out trades vs the same trades held to the close (explanatory: uses the close).
    Many stopped trades that would have won at the close = stops too tight."""
    s = trades[trades["exit_type"] == "stopped out"].copy()
    aum = s["date"].map(result.daily["aum_start"])
    held = s["shares"] * s["side"] * (s["date"].map(close) - s["entry_price"]) - 2 * s["shares"] * cost_per_share
    s["held_contribution"] = held / aum
    return pd.DataFrame({
        "trades": [len(s)],
        "actual_avg_bps": [s["contribution"].mean() * BPS],
        "held_to_close_avg_bps": [s["held_contribution"].mean() * BPS],
        "share_profitable_if_held": [(s["held_contribution"] > 0).mean()],
        "actual_total": [s["contribution"].sum()],
        "held_to_close_total": [s["held_contribution"].sum()],
    }, index=["stopped-out trades"])


def expansion_ratio(daily: pd.DataFrame, minute_feats: pd.DataFrame) -> pd.Series:
    """Today's open-to-close move in units of the recent normal move: |close / open - 1| divided
    by the noise-area sigma at the last minute (the 14-day average move from the open at the
    close). Above 1 = the day closed outside the noise area. Explanatory only: known at the close."""
    sigma_at_close = minute_feats.groupby("date")["sigma"].last().reindex(daily.index)
    return ((daily["close"] / daily["open"] - 1).abs() / sigma_at_close).rename("expansion")



# ---------------------------------------------------------------- own-version research (Stage 1)

def pnl_split(ret: pd.Series, expansion: pd.Series, leverage: pd.Series, spy_open_to_close: pd.Series,
              n: int = 252) -> pd.DataFrame:
    """Each year's result as opportunity x capture + cost.

    opportunity  share of days that close outside the noise area (expansion > 1)
    capture      mean return on those days; capture_ratio compares it with a perfect-hindsight
                 trade at the same leverage from open to close in the right direction
    cost         mean return on the other days (failed breakouts)
    The three *_effect columns split each year's deviation from the whole-period mean daily
    return exactly: (p - p0)(cE0 - cN0) + p(cE - cE0) + (1 - p)(cN - cN0)."""
    exp = expansion > 1
    p0, ce0, cn0 = exp.mean(), ret[exp].mean(), ret[~exp].mean()
    rows = {}
    for year, idx in ret.groupby(ret.index.year).groups.items():
        r, e = ret.loc[idx], exp.loc[idx]
        p, ce, cn = e.mean(), r[e].mean(), r[~e].mean()
        perfect = (leverage.loc[idx][e] * spy_open_to_close.loc[idx][e].abs()).sum()
        rows[year] = {
            "days": len(r), "opportunity": p, "capture_bps": ce * BPS, "capture_ratio": r[e].sum() / perfect,
            "cost_bps": cn * BPS, "mean_bps": r.mean() * BPS, "annual_return": (1 + r).prod() ** (n / len(r)) - 1,
            "opportunity_effect_bps": (p - p0) * (ce0 - cn0) * BPS,
            "capture_effect_bps": p * (ce - ce0) * BPS,
            "cost_effect_bps": (1 - p) * (cn - cn0) * BPS,
        }
    table = pd.DataFrame(rows).T
    perfect_all = (leverage[exp] * spy_open_to_close[exp].abs()).sum()
    table.loc["all"] = {"days": len(ret), "opportunity": p0, "capture_bps": ce0 * BPS,
                        "capture_ratio": ret[exp].sum() / perfect_all, "cost_bps": cn0 * BPS,
                        "mean_bps": ret.mean() * BPS, "annual_return": (1 + ret).prod() ** (n / len(ret)) - 1,
                        "opportunity_effect_bps": 0.0, "capture_effect_bps": 0.0, "cost_effect_bps": 0.0}
    return table


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """False-discovery-rate adjusted p-values (q-values) for testing many features at once."""
    order = p_values.sort_values()
    ranked = order * len(order) / np.arange(1, len(order) + 1)
    return ranked[::-1].cummin()[::-1].clip(upper=1).reindex(p_values.index)


def feature_screen(trades: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """How well each entry feature separates good trades from bad ones, on two questions:

    hit rate  auc = chance that a random winning trade has a higher value than a random losing
              one (0.5 = no information; below 0.5 = LOWER values win more often)
    payoff    rho = rank correlation with the trade's return at 1x leverage (sizing removed)
    q-values  p-values adjusted for testing all features together (Benjamini-Hochberg)
    bottom/top_bps  average 1x return of trades in the lowest and highest fifth of the feature
    years_same_sign years in which rho has the same sign as overall
    Trades on the same day are not independent, so p-values are somewhat optimistic."""
    from scipy.stats import mannwhitneyu, spearmanr
    rows = {}
    for f in features:
        df = trades[[f, "win", "return_1x", "year"]].dropna()
        won, lost = df.loc[df["win"], f], df.loc[~df["win"], f]
        u, p_auc = mannwhitneyu(won, lost, alternative="two-sided")
        rho, p_rho = spearmanr(df[f], df["return_1x"])
        quintile = pd.qcut(df[f].rank(method="first"), 5, labels=False)
        yearly_rho = df.groupby("year").apply(lambda g: spearmanr(g[f], g["return_1x"])[0], include_groups=False)
        rows[f] = {"auc": u / (len(won) * len(lost)), "p_auc": p_auc, "rho": rho, "p_rho": p_rho,
                   "bottom_bps": df.loc[quintile == 0, "return_1x"].mean() * BPS,
                   "top_bps": df.loc[quintile == 4, "return_1x"].mean() * BPS,
                   "years_same_sign": f"{int((np.sign(yearly_rho) == np.sign(rho)).sum())}/{len(yearly_rho)}"}
    table = pd.DataFrame(rows).T
    table.insert(2, "q_auc", _benjamini_hochberg(table["p_auc"].astype(float)))
    table.insert(5, "q_rho", _benjamini_hochberg(table["p_rho"].astype(float)))
    return table.drop(columns=["p_auc", "p_rho"]).sort_values("q_rho")


def worst_trade_patterns(trades: pd.DataFrame, flags: dict[str, pd.Series], k: int = 50) -> pd.DataFrame:
    """How often each pattern appears among the k worst trades vs all trades."""
    worst = trades["contribution"].nsmallest(k).index
    return pd.DataFrame({
        "worst_trades": {name: flag.loc[worst].mean() for name, flag in flags.items()},
        "all_trades": {name: flag.mean() for name, flag in flags.items()},
    }).assign(ratio=lambda t: t["worst_trades"] / t["all_trades"])
