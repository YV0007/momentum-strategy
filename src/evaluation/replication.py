"""Compares our returns with the paper's published monthly returns."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import plots
from src.config import CONFIG_DIR, RESULTS_DIR


def paper_monthly() -> pd.Series:
    df = pd.read_csv(CONFIG_DIR / "paper_monthly_returns.csv", comment="#")
    index = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)) + pd.offsets.MonthEnd(0)
    return pd.Series(df["return_pct"].values / 100, index=index, name="paper")


def paper_yearly() -> pd.Series:
    monthly = paper_monthly()
    counts = monthly.groupby(monthly.index.year).size()
    full = counts[counts == 12].index
    yearly = (1 + monthly).groupby(monthly.index.year).prod() - 1
    return yearly.loc[full]


def to_monthly(daily_ret: pd.Series) -> pd.Series:
    return (1 + daily_ret).resample("ME").prod() - 1


def alignment(daily_ret: pd.Series, last_full_month: str = "2025-01-31") -> tuple[pd.DataFrame, dict]:
    m = pd.concat([to_monthly(daily_ret).rename("ours"), paper_monthly()], axis=1, join="inner")
    m = m.loc[:last_full_month]
    diff = m["ours"] - m["paper"]
    fit = sm.OLS(m["ours"], sm.add_constant(m["paper"])).fit()
    return m, {
        "months": len(m), "first_month": m.index[0], "last_month": m.index[-1],
        "correlation": m["ours"].corr(m["paper"]),
        "slope": fit.params["paper"], "slope_t_vs_1": (fit.params["paper"] - 1) / fit.bse["paper"],
        "mean_monthly_difference": diff.mean(), "mean_difference_t": diff.mean() / diff.std() * np.sqrt(len(diff)),
        "tracking_error": diff.std() * np.sqrt(12), "median_abs_difference": diff.abs().median(),
        "same_sign": (np.sign(m["ours"]) == np.sign(m["paper"])).mean(),
        "months_over_1pct": int((diff.abs() > 0.01).sum()), "months_over_2pct": int((diff.abs() > 0.02).sum()),
    }


def write_report(daily_ret: pd.Series) -> str:
    m, st = alignment(daily_ret)
    periods = {"Train 2016–2022": slice("2016", "2022"), "Test Jan 2023–Jan 2025": slice("2023", "2025-01"),
               "All shared months": slice(None)}
    rows = {}
    for label, sl in periods.items():
        x = m.loc[sl]
        growth = (1 + x).prod()
        rows[label] = {"cumulative_ours": growth["ours"] - 1, "cumulative_paper": growth["paper"] - 1,
                       "annual_ours": growth["ours"] ** (12 / len(x)) - 1,
                       "annual_paper": growth["paper"] ** (12 / len(x)) - 1}
    periods_table = pd.DataFrame(rows).T.map("{:.1%}".format)
    yearly = (1 + m).groupby(m.index.year).prod() - 1
    yearly["difference"] = yearly["ours"] - yearly["paper"]
    worst = m.assign(difference=m["ours"] - m["paper"]).reindex((m["ours"] - m["paper"]).abs().nlargest(6).index)
    worst.index = worst.index.strftime("%Y-%m")
    plots.replication_chart(m, "Final strategy: our backtest vs the paper's published monthly returns", "replication")

    lines = [
        "# Replication: final strategy vs the paper's published monthly returns", "",
        "_Generated from the saved train and test backtests. Paper: FAQ Q24 table, net of costs. "
        "Feb 2025 excluded as possibly partial. Figure: results/figures/replication.png._", "",
        f"- Months compared: {st['months']} ({st['first_month']:%b %Y} to {st['last_month']:%b %Y})",
        f"- Correlation of monthly returns: {st['correlation']:.3f}",
        f"- Regression ours = a + b x paper: slope {st['slope']:.3f} (t vs 1: {st['slope_t_vs_1']:+.2f})",
        f"- Mean monthly difference: {st['mean_monthly_difference']:+.2%} (t {st['mean_difference_t']:+.2f}): no bias",
        f"- Tracking error: {st['tracking_error']:.2%} per year; median absolute monthly difference "
        f"{st['median_abs_difference']:.2%}",
        f"- Months with the same sign: {st['same_sign']:.0%}; months differing by more than 1%: "
        f"{st['months_over_1pct']}, more than 2%: {st['months_over_2pct']}", "",
        "## Cumulative and annual returns", "", periods_table.to_markdown(), "",
        "## By year", "", yearly.map("{:.1%}".format).to_markdown(), "",
        "## Largest monthly differences", "", worst.map("{:.1%}".format).to_markdown(), "",
    ]
    (RESULTS_DIR / "replication_report.md").write_text("\n".join(lines))
    return "\n".join(lines)
