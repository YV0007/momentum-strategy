"""All report figures."""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

COLORS = {
    "final": "#2a78d6", "vwap_stop": "#eb6834", "base": "#1baf7a", "own": "#4a3aa7",
    "own_turbulence": "#4a3aa7", "own_ml_vol": "#c2407f", "long_only": "#1d8a5b", "short_only": "#c2402f",
    "spy_alone": "#e0a106",
    "buy_and_hold": "#52514e", "open_to_close": "#a3a29d",
}
LABELS = {"final": "Final (VWAP stop + vol sizing)", "vwap_stop": "VWAP stop, 1x",
          "base": "Base (opposite band), 1x", "own": "Own strategy",
          "own_turbulence": "Own A: turbulence sizing", "own_ml_vol": "Own B: ML volatility sizing",
          "long_only": "Long trades only", "short_only": "Short trades only",
          "spy_alone": "SPY alone (paper's final)",
          "buy_and_hold": "SPY buy & hold", "open_to_close": "SPY open-to-close"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _color(name: str) -> str:
    return COLORS.get(name, COLORS["own"])


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def equity_curves(returns: dict[str, pd.Series], title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, ret in returns.items():
        equity = (1 + ret).cumprod()
        dashed = key in ("buy_and_hold", "open_to_close")
        ax.plot(equity.index, equity, color=_color(key), lw=1.6, ls="--" if dashed else "-",
                label=LABELS.get(key, key))
        ax.annotate(f"{equity.iloc[-1]:.2f}x", (equity.index[-1], equity.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(loc="upper left")
    _save(fig, name)


def drawdowns(returns: dict[str, pd.Series], title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for key, ret in returns.items():
        equity = (1 + ret).cumprod()
        ax.plot(equity.index, equity / equity.cummax() - 1, color=_color(key), lw=1.2,
                label=LABELS.get(key, key))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncol=3)
    _save(fig, name)


def metric_bars(table: pd.DataFrame, title: str, name: str) -> None:
    metrics = [("sharpe", "Sharpe ratio", "{:.2f}"), ("annual_return", "Annualized return", "{:.1%}"),
               ("annual_volatility", "Annualized volatility", "{:.1%}")]
    has_periods = isinstance(table.index, pd.MultiIndex)
    strategies = table.index.get_level_values(0).unique() if has_periods else table.index
    periods = table.index.get_level_values(1).unique() if has_periods else [None]
    width = 0.8 / len(periods)

    stacked = len(strategies) > 5
    fig, axes = plt.subplots(3, 1, figsize=(12, 11)) if stacked else plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (col, label, fmt) in zip(axes, metrics):
        for p_i, period in enumerate(periods):
            values = [table.loc[(s, period), col] if has_periods else table.loc[s, col] for s in strategies]
            x = np.arange(len(strategies)) + (p_i - (len(periods) - 1) / 2) * width
            bars = ax.bar(x, values, width * 0.92, color=[_color(s) for s in strategies],
                          alpha=1.0 if p_i == 0 else 0.55, label=period)
            for bar, v in zip(bars, values):
                ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 2 if v >= 0 else -10), textcoords="offset points",
                            ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(np.arange(len(strategies)))
        names = [LABELS.get(s, s).split(" (")[0] for s in strategies]
        if stacked:
            ax.set_xticklabels([textwrap.fill(n.replace("_", " "), 16) for n in names], fontsize=8)
        else:
            ax.set_xticklabels(names, rotation=30, ha="right")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
        if col != "sharpe":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if has_periods:
        axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def sweep_bars(tables: dict[str, pd.DataFrame], metrics: list[tuple[str, str, str]], title: str, name: str,
               highlight=None) -> None:
    periods = list(tables)
    values = tables[periods[0]].index
    width = 0.8 / len(periods)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4))
    for ax, (col, label, fmt) in zip(np.atleast_1d(axes), metrics):
        for p_i, period in enumerate(periods):
            x = np.arange(len(values)) + (p_i - (len(periods) - 1) / 2) * width
            ys = tables[period][col].to_numpy(dtype=float)
            bars = ax.bar(x, ys, width * 0.92, color=COLORS["final"], alpha=1.0 if p_i == 0 else 0.5, label=period)
            for bar, v, value in zip(bars, ys, values):
                if value == highlight:
                    bar.set_edgecolor(TEXT)
                    bar.set_linewidth(1.4)
                ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2 if v >= 0 else -10),
                            textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in values])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
        if "%" in fmt:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title + ("" if highlight is None else "  (outlined: the paper's setting)"),
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def placebo_histogram(placebo_sharpes: pd.Series, real_sharpe: float, p_value: float,
                      title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(placebo_sharpes, bins=40, color=COLORS["open_to_close"], edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(real_sharpe, color=COLORS["final"], lw=2)
    ax.annotate(f"Real strategy: {real_sharpe:.2f}\nplacebo p-value: {p_value:.3f}",
                (real_sharpe, ax.get_ylim()[1] * 0.9), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="top", fontsize=9)
    ax.set_xlabel("Sharpe ratio of random-direction versions")
    ax.set_ylabel("Runs")
    ax.set_title(title, loc="left")
    _save(fig, name)


def monthly_heatmap(ret: pd.Series, title: str, name: str) -> None:
    monthly = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    table = monthly.unstack()
    lim = np.nanmax(np.abs(table.values))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(table) + 1.2))
    ax.imshow(table.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(table.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(range(table.shape[1]), [pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns])
    ax.set_yticks(range(len(table)), table.index)
    ax.grid(False)
    ax.set_title(title, loc="left")
    _save(fig, name)


SIDE_COLORS = {"long": "#eda100", "short": "#e87ba4"}


def bucket_panels(panels: dict[str, pd.DataFrame], title: str, name: str, key: str = "final") -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(max(3.6 * len(panels), 11), 3.9), sharey=True)
    for ax, (label, t) in zip(np.atleast_1d(axes), panels.items()):
        x = np.arange(len(t))
        ax.bar(x, t["mean_bps"], 0.7, color=_color(key))
        ax.errorbar(x, t["mean_bps"], yerr=t["ci95_bps"], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3)
        ax.set_xticks(x, [str(i) for i in t.index], rotation=30, ha="right", fontsize=8.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Mean daily return, bps (95% CI)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def time_of_day_bars(table: pd.DataFrame, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = np.arange(len(table))
    ax.bar(x, table["gross_bps"], 0.62, color=COLORS["final"], label="Gross P&L of the leg")
    ax.errorbar(x, table["gross_bps"], yerr=table["ci95_bps"], fmt="none", ecolor=MUTED,
                elinewidth=1, capsize=3)
    ax.bar(x, -table["cost_bps"], 0.62, color=COLORS["open_to_close"], label="Trading costs")
    ax.set_xticks(x, table.index, rotation=30, ha="right", fontsize=8.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean per day, bps of AUM")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right")
    _save(fig, name)


def side_contribution_lines(contrib: pd.DataFrame, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    total = contrib.sum(axis=1).cumsum()
    ax.plot(total.index, total, color=COLORS["final"], lw=1.6, label="Total")
    for side in ("long", "short"):
        c = contrib[side].cumsum()
        ax.plot(c.index, c, color=SIDE_COLORS[side], lw=1.6, label=f"{side.capitalize()} trades")
    for label, series in [("Total", total), ("Long", contrib["long"].cumsum()), ("Short", contrib["short"].cumsum())]:
        ax.annotate(f"{label} {series.iloc[-1]:.0%}", (series.index[-1], series.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Cumulative sum of daily returns")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, name)


def yearly_bars(table: pd.DataFrame, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = np.arange(len(table))
    for offset, (col, key) in zip([-0.2, 0.2], [("return", "final"), ("spy_return", "buy_and_hold")]):
        bars = ax.bar(x + offset, table[col], 0.38, color=_color(key), label=LABELS[key])
        for bar, v in zip(bars, table[col]):
            ax.annotate(f"{v:.0%}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2 if v >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, table.index)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, name)


def payoff_scatter(strategy_ret: pd.Series, spy_intraday: pd.Series, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(spy_intraday, strategy_ret, s=10, color=COLORS["final"], alpha=0.45, linewidths=0)
    bins = pd.qcut(spy_intraday, 15)
    means = pd.DataFrame({"x": spy_intraday, "y": strategy_ret}).groupby(bins, observed=True).mean()
    ax.plot(means["x"], means["y"], color=TEXT, lw=1.6, marker="o", ms=4, label="Average per bucket")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("SPY open-to-close return (same day)")
    ax.set_ylabel("Strategy daily return")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center")
    _save(fig, name)


def _draw_day(ax, minute: pd.DataFrame, minute_feats: pd.DataFrame, trades: pd.DataFrame, day) -> None:
    bars = minute[minute["date"] == day].join(minute_feats[["upper", "lower", "vwap"]])
    t = bars.index.tz_localize(None) + pd.Timedelta(minutes=1)
    ax.fill_between(t, bars["lower"], bars["upper"], color="#e4e3df", label="Noise area")
    ax.plot(t, bars["close"], color=TEXT, lw=1, label="SPY")
    ax.plot(t, bars["vwap"], color=COLORS["vwap_stop"], lw=1, label="VWAP")
    day_str = f"{pd.Timestamp(day):%Y-%m-%d}"
    for _, tr in trades[trades["date"] == day].iterrows():
        entry_t = pd.Timestamp(f"{day_str} {tr['entry_time']}")
        exit_t = t[-1] if tr["exit_time"] == "close" else pd.Timestamp(f"{day_str} {tr['exit_time']}")
        ax.scatter(entry_t, tr["entry_price"], marker="^" if tr["side"] > 0 else "v", s=70, color=TEXT, zorder=5)
        ax.scatter(exit_t, tr["exit_price"], marker="x", s=55, color=TEXT, zorder=5)
    for hh in pd.date_range(f"{day_str} 10:00", f"{day_str} 15:30", freq="30min"):
        ax.axvline(hh, color="#c9c8c3", ls=":", lw=0.7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.grid(False)


def day_panels(minute: pd.DataFrame, minute_feats: pd.DataFrame, trades: pd.DataFrame,
               day_returns: pd.Series, title: str, name: str) -> None:
    days = list(day_returns.index)
    rows = int(np.ceil(len(days) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), squeeze=False)
    for ax, day in zip(axes.ravel(), days):
        _draw_day(ax, minute, minute_feats, trades, day)
        ax.set_title(f"{day:%Y-%m-%d}   strategy {day_returns[day]:+.1%}", loc="left", fontsize=10)
    for ax in axes.ravel()[len(days):]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def noise_area_explainer(minute: pd.DataFrame, minute_feats: pd.DataFrame, daily: pd.DataFrame,
                         trades: pd.DataFrame, day: str, name: str, lookback: int = 14) -> None:
    day = pd.Timestamp(day)
    prev_days = daily.index[(daily.index < day) & daily["is_valid"]][-lookback:]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.3))
    clock_times = pd.Timestamp("2000-01-01 09:31") + pd.to_timedelta(np.arange(390), unit="min")
    for d in prev_days:
        bars = minute[minute["date"] == d]
        move = (bars["close"] / daily.loc[d, "open"] - 1).abs().to_numpy()
        left.plot(clock_times[:len(move)], move, color="#c9c8c3", lw=0.8)
    sigma = minute_feats.loc[minute["date"] == day, "sigma"].to_numpy()
    left.plot(clock_times[:len(sigma)], sigma, color=COLORS["final"], lw=2.2,
              label=f"Average = sigma used on {day:%Y-%m-%d}")
    left.plot([], [], color="#c9c8c3", lw=0.8, label=f"Each of the previous {lookback} days")
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    left.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    left.set_ylabel("|price / open - 1|")
    left.set_title("1. Past days: absolute move from the open", loc="left")
    left.legend(loc="upper left", fontsize=8.5)

    _draw_day(right, minute, minute_feats, trades, day)
    right.set_title(f"2. Today: open ± sigma = noise area, trade on a break ({day:%Y-%m-%d})", loc="left")
    right.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, name)


def leverage_timeline(vol_daily: pd.Series, leverage: pd.Series, cap: float, title: str, name: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    top.plot(vol_daily.index, vol_daily, color=COLORS["buy_and_hold"], lw=1.2)
    top.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    top.set_title("14-day volatility of daily returns (known at the open)", loc="left", fontsize=10)
    lev = leverage.where(leverage > 0)
    bottom.plot(lev.index, lev, color=COLORS["final"], lw=1.0)
    bottom.axhline(cap, color=MUTED, ls="--", lw=1)
    bottom.annotate(f"cap {cap:g}x", (lev.index[0], cap), xytext=(0, 4), textcoords="offset points",
                    fontsize=8.5, color=MUTED)
    bottom.set_ylabel("Leverage")
    bottom.set_title("Leverage = min(cap, 2% / volatility)", loc="left", fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def replication_chart(monthly: pd.DataFrame, title: str, name: str) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    growth = (1 + monthly[["ours", "paper"]]).cumprod()
    higher = growth.iloc[-1].idxmax()
    for col, color, label in [("ours", COLORS["final"], "Our backtest"), ("paper", COLORS["buy_and_hold"], "Paper (FAQ Q24)")]:
        left.plot(growth.index, growth[col], color=color, lw=1.8, ls="-" if col == "ours" else "--", label=label)
        left.annotate(f"{label.split(' (')[0]} {growth[col].iloc[-1]:.2f}x", (growth.index[-1], growth[col].iloc[-1]),
                      xytext=(4, 7 if col == higher else -7), textcoords="offset points", va="center",
                      fontsize=9, color=MUTED)
    left.set_title("Growth of $1, net of costs", loc="left")
    left.legend(loc="upper left")
    lim = max(monthly.abs().max()) * 1.1
    right.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    right.scatter(monthly["paper"], monthly["ours"], s=16, color=COLORS["final"], alpha=0.75, linewidths=0)
    for axis in (right.xaxis, right.yaxis):
        axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    right.set_xlabel("Paper monthly return")
    right.set_ylabel("Our monthly return")
    right.set_title(f"Each month (correlation {monthly['ours'].corr(monthly['paper']):.3f})", loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def pnl_split_chart(split: pd.DataFrame, title: str, name: str) -> None:
    years = split.drop("all")
    panels = [("opportunity", "Opportunity: days closing outside the noise area", True),
              ("capture_bps", "Capture: mean return on those days, bps", False),
              ("cost_bps", "Cost: mean return on the other days, bps", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(years))
    for ax, (col, label, is_share) in zip(axes, panels):
        ax.bar(x, years[col], 0.7, color=COLORS["final"])
        ax.axhline(split.loc["all", col], color=MUTED, ls="--", lw=1)
        for xi, v in zip(x, years[col]):
            ax.annotate(f"{v:.0%}" if is_share else f"{v:.0f}", (xi, v), xytext=(0, 2 if v >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED)
        if is_share:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(x, years.index.astype(int))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title + "   (dashed = whole-period average)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def feature_screen_chart(screen: pd.DataFrame, title: str, name: str, q_max: float = 0.05) -> None:
    s = screen.sort_values("rho")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 1.2))
    y = np.arange(len(s))
    passed = s["q_rho"].astype(float) < q_max
    colors = [matplotlib.colors.to_rgba(COLORS["final"], 1.0 if ok else 0.35) for ok in passed]
    ax.barh(y, s["rho"].astype(float), 0.65, color=colors)
    for yi, (rho, ok) in enumerate(zip(s["rho"].astype(float), passed)):
        ax.annotate(f"{rho:+.3f}" + ("" if ok else "  n.s."), (rho, yi), xytext=(4 if rho >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if rho >= 0 else "right", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, s.index)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Rank correlation with the trade's return at 1x leverage")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, name)
