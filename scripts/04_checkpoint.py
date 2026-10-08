"""Diagnostics of the final strategy on the train period."""

import numpy as np
import pandas as pd

from src import plots
from src.config import RESULTS_DIR, load_research, load_strategies
from src.engine.backtest import MarketData, prepare, run, simulate
from src.evaluation import baselines, diagnostics as dg

PCT = {"traded_share", "hit_ratio", "share_of_pnl", "total_contribution", "return", "spy_return",
       "max_drawdown", "days_at_cap", "long_share", "short_share", "ret", "spy_open_to_close",
       "spy_gap", "best_1pct_days_share", "best_5pct_days_share", "total_return",
       "total_return_without_best_10_days", "share_profitable_if_held", "actual_total",
       "held_to_close_total"}


def fmt(df: pd.DataFrame) -> str:
    out = df.copy().astype(object)
    if isinstance(df.index, pd.DatetimeIndex):
        out.index = df.index.strftime("%Y-%m-%d")
    for col in df.columns:
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        pattern = ("{:.1%}" if col in PCT else "{:.0f}" if col in ("days", "trades")
                   else "{:.1f}" if col.endswith("_bps") else "{:.3f}" if col == "p_value" else "{:.2f}")
        out[col] = values.map(lambda v: "" if pd.isna(v) else pattern.format(v))
    return out.to_markdown()


def decision_gate() -> str:
    path = RESULTS_DIR / "train_metrics.csv"
    if not path.exists():
        raise SystemExit("Run scripts/03_run_backtests.py first (needs results/train_metrics.csv)")
    final = pd.read_csv(path, index_col=0).loc["final"]
    clear = final["sharpe_ci_low"] > 0 and final["placebo_p_value"] < 0.05
    return (f"Final strategy on train: Sharpe {final['sharpe']:.2f}, 95% CI "
            f"[{final['sharpe_ci_low']:.2f}, {final['sharpe_ci_high']:.2f}], placebo p-value "
            f"{final['placebo_p_value']:.3f}. "
            + ("The edge is clearly measurable, so the data is NOT extended back to 2007."
               if clear else "The edge is too weak to judge: extend the data back to 2007 before going on."))


def main() -> None:
    research = load_research()
    strategies = load_strategies()
    data = MarketData.load()
    start, end, n = research.train_start, research.train_end, research.trading_days

    prep = prepare(strategies["final"], data, start, end)
    final = simulate(prep, prep.rule_positions(), research)
    one_x = run(strategies["vwap_stop"], research, data, start, end)

    days = final.daily.index
    ret, traded = final.daily["ret"], final.daily["trades"] > 0
    daily, feats = data.daily.loc[days], data.daily_feats.loc[days]
    market = baselines.buy_and_hold(daily)["ret"]
    trades = dg.with_contribution(final)

    vix_q = dg.quantile_buckets(feats["vix_open"], 5, "{:.1f}")
    regimes = {
        "VIX at open": dg.bucket_stats(ret, vix_q, traded, n),
        "RSI(5) yesterday": dg.bucket_stats(ret, dg.quantile_buckets(feats["rsi5_prev"], 5), traded, n),
        "|Overnight gap|": dg.bucket_stats(ret, pd.cut(feats["gap"].abs(), [0, 0.0025, 0.005, 0.01, np.inf],
                                                       labels=["<0.25%", "0.25–0.5%", "0.5–1%", ">1%"],
                                                       include_lowest=True), traded, n),
        "NR4 yesterday": dg.bucket_stats(ret, feats["nr4_prev"].map({False: "no", True: "yes"}), traded, n),
    }
    vix_1x = dg.bucket_stats(one_x.daily["ret"], vix_q, one_x.daily["trades"] > 0, n)
    vix_thresholds = dg.above_thresholds(ret, feats["vix_open"], [0, 15, 20, 25, 30, 40], n)
    regressions = pd.DataFrame({
        "VIX at open (per 10 points)": dg.regression(ret, feats["vix_open"], 10),
        "RSI(5) yesterday (per 10 points)": dg.regression(ret, feats["rsi5_prev"], 10),
        "|Overnight gap| (per 1%)": dg.regression(ret, feats["gap"].abs() * 100),
        "NR4 yesterday (yes vs no)": dg.regression(ret, feats["nr4_prev"]),
    }).T

    tod = dg.time_of_day(prep, final, research.cost_per_share)
    assert np.isclose((tod["gross_bps"] - tod["cost_bps"]).sum(), ret.mean() * dg.BPS)
    by_entry = dg.trade_breakdown(trades, "entry_time", n)
    by_exit = dg.trade_breakdown(trades, "exit_type", n)
    sequenced = dg.trade_sequence(trades)
    by_sequence = dg.trade_breakdown(sequenced, "sequence", n)
    by_reentry = dg.trade_breakdown(sequenced, "reentry", n)
    stops = dg.stop_counterfactual(trades, final, daily["close"], research.cost_per_share)
    per_day = dg.bucket_stats(ret, final.daily["trades"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"}),
                              traded, n)

    sides = dg.side_contributions(final)
    side_table = dg.trade_breakdown(trades, "side_name", n)
    side_table["sharpe_of_daily_contribution"] = sides.mean() / sides.std() * np.sqrt(n)
    years = dg.yearly(final, market, feats["vix_open"], strategies["final"].max_leverage, n)
    worst = dg.extreme_days(final, daily, feats, k=10, worst=True)
    best = dg.extreme_days(final, daily, feats, k=10, worst=False)
    conc = pd.DataFrame([dg.concentration(ret)])
    payoff = dg.bucket_stats(ret, dg.quantile_buckets(daily["ret_oc"] * 100, 5, "{:.2f}"), traded, n)
    payoff.index = [f"{i} %" for i in payoff.index]

    expansion = dg.expansion_ratio(daily, data.minute_feats)
    vol_level = dg.bucket_stats(ret, dg.quantile_buckets(feats["vol_daily"] * 100, 5, "{:.2f}"), traded, n)
    vol_level.index = [f"{i} %" for i in vol_level.index]
    vol_expansion = dg.bucket_stats(ret, dg.quantile_buckets(expansion, 5, "{:.2f}"), traded, n)
    vol_expansion.index = [f"{i}x" for i in vol_expansion.index]

    label = "final strategy, train 2016–2022"
    plots.bucket_panels(regimes, f"Mean daily return by market regime ({label})", "checkpoint_regimes")
    plots.time_of_day_bars(tod, f"P&L by half-hour holding leg ({label})", "checkpoint_time_of_day")
    plots.side_contribution_lines(sides, f"Long vs short trades ({label})", "checkpoint_long_short")
    plots.yearly_bars(years, f"Yearly returns ({label})", "checkpoint_yearly")
    plots.payoff_scatter(ret, daily["ret_oc"], f"Payoff vs SPY's intraday move ({label})", "checkpoint_payoff")
    plots.bucket_panels({"Recent volatility level (known at the open)": vol_level,
                         "Today's move ÷ recent normal (known at the close)": vol_expansion},
                        f"Volatility level vs volatility expansion ({label})", "checkpoint_volatility")
    plots.noise_area_explainer(data.minute, data.minute_feats, data.daily, final.trades, "2022-04-29",
                               "checkpoint_noise_area_explained")
    plots.leverage_timeline(feats["vol_daily"], final.daily["leverage"], strategies["final"].max_leverage,
                            f"How past volatility sets the position size ({label})", "checkpoint_leverage")
    plots.day_panels(data.minute, data.minute_feats, final.trades, ret[worst.index[:4]],
                     "The four worst days (train)", "checkpoint_worst_days")
    plots.day_panels(data.minute, data.minute_feats, final.trades, ret[best.index[:4]],
                     "The four best days (train)", "checkpoint_best_days")

    sections = [
        (2, "Decision gate", decision_gate(), None),
        (2, "1. Market regimes, known at the open",
         "Mean daily return per bucket (bps of AUM) with 95% interval. Figure: checkpoint_regimes.png.", None),
        *[(3, f"1.{i} {name}", "", table) for i, (name, table) in enumerate(regimes.items(), 1)],
        (3, "1.5 VIX quintiles for the 1x VWAP-stop version (no vol sizing)",
         "Without vol targeting, high-VIX days carry far more risk, which shows how sizing reshapes "
         "the regime profile.", vix_1x),
        (3, "1.6 Paper Fig. 8 style: days with VIX at or above a threshold", "", vix_thresholds),
        (3, "1.7 Regressions of daily return (bps) on each variable",
         "Newey-West t-stats. Paper section 4.5 reports a significantly negative RSI slope.", regressions),
        (2, "2. Inside the day", "", None),
        (3, "2.1 P&L by half-hour holding leg",
         "Average per day over all days; gross minus costs summed over legs equals the average daily "
         "return. Figure: checkpoint_time_of_day.png.", tod),
        (3, "2.2 Trades by entry time", "", by_entry),
        (3, "2.3 Trades by exit type", "", by_exit),
        (3, "2.4 Stopped-out trades vs holding them to the close (explanatory, uses the close)", "", stops),
        (3, "2.5 Trades by order within the day", "The marginal value of each re-entry.", by_sequence),
        (3, "2.6 Re-entries: same direction again vs reversed", "", by_reentry),
        (3, "2.7 Days by number of trades (ex post: the count is only known at the close)", "", per_day),
        (2, "3. Sides, years and extremes", "", None),
        (3, "3.1 Long vs short trades", "Figure: checkpoint_long_short.png.", side_table),
        (3, "3.2 By year", "Figure: checkpoint_yearly.png.", years),
        (3, "3.3 Payoff vs SPY's same-day open-to-close move (explanatory, uses same-day information)",
         "Figure: checkpoint_payoff.png.", payoff),
        (3, "3.4 Concentration of P&L", "", conc),
        (3, "3.5 Ten worst days", "Figure: checkpoint_worst_days.png.", worst),
        (3, "3.6 Ten best days", "Figure: checkpoint_best_days.png.", best),
        (2, "4. How past volatility enters the strategy",
         "Past volatility is used twice, both times to NORMALIZE, never to forecast: (1) the noise "
         "area, the 14-day average move from the open at each minute, sets the entry threshold "
         "(checkpoint_noise_area_explained.png); (2) the 14-day volatility of daily returns sets the "
         "leverage, min(4, 2% / vol) (checkpoint_leverage.png).", None),
        (3, "4.1 By recent volatility level (known at the open)", "", vol_level),
        (3, "4.2 By today's move relative to the recent normal (explanatory: known at the close)",
         "Ratio = |open-to-close| / noise-area sigma at the close; above 1x the day closed outside "
         "the noise area. Figure: checkpoint_volatility.png.", vol_expansion),
    ]
    lines = [f"# Checkpoint diagnostics: final strategy, train period ({start} to {end})", "",
             "_Generated by `scripts/04_checkpoint.py`; do not edit by hand._", ""]
    for level, title, note, table in sections:
        lines += ["#" * level + " " + title, ""]
        if note:
            lines += [note, ""]
        if table is not None:
            lines += [fmt(table), ""]
    (RESULTS_DIR / "checkpoint_report.md").write_text("\n".join(lines))
    print(decision_gate())
    print("Report: results/checkpoint_report.md, figures: results/figures/checkpoint_*.png")


if __name__ == "__main__":
    main()
