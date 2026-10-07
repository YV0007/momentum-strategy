"""The paper's variations and further investigations, on the train and the test period. (Phase 8)

    python -m scripts.07_robustness

Nothing here is a candidate: the paper's settings stay as they are, and every run is logged with
note "robustness: ..." or "ablation" (not counted as a trial by the Deflated Sharpe).
1. Paper versions: every stop rule at 1x and with volatility sizing (Tables 1-3, Fig. 5a, FAQ Q22)
2. Sweeps of the final strategy: volatility multiplier (4.4, Fig. 9), noise-area lookback (FAQ Q6)
3. Costs: commission levels (4.6, Fig. 10), IB tiered commission (4.6), I-Star impact (FAQ Q15)
4. Conditional results of the final strategy: VIX at the open (4.1, Fig. 8), daily patterns
   (4.2, Table 5), weekday (4.3, Table 6)
5. Trades and legs: Table 4, profit per share by year (Q23), long/short legs (Q5), short trades
   vs VIX (Q19, Q20), shorts only below moving averages (Q21), SPY's worst quarters (Q7)
Writes results/robustness_report.md and results/figures/robustness_*.png.
"""

from dataclasses import replace

import pandas as pd

from src import plots
from src.config import RESULTS_DIR, load_research, load_strategies
from src.engine.backtest import MarketData, run
from src.evaluation import baselines, metrics, paper_tables as pt
from src.evaluation.ablation import run_ablation
from src.experiment_log import log_run

VM_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
LOOKBACKS = [5, 10, 14, 20, 25, 30, 60, 90]
COMMISSIONS = [0.0, 0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01]
VIX_THRESHOLDS = [10, 15, 20, 25, 30, 35, 40]
SMA_WINDOWS = [100, 150, 200]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SUMMARY = ["annual_return", "annual_volatility", "sharpe", "max_drawdown", "trades"]
SWEEP_METRICS = [("sharpe", "Sharpe ratio", "{:.2f}"), ("annual_return", "Annualized return", "{:.1%}"),
                 ("annual_volatility", "Annualized volatility", "{:.1%}")]
RATES = {"total_return", "annual_return", "annual_volatility", "max_drawdown", "cost_share_of_gross",
         "hit_ratio", "spy", "strategy"}
COUNTS = {"trades", "observations", "short_trades"}


def side_by_side(tables: dict[str, pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    """One row per variant, `cols` for each period next to each other."""
    return pd.concat({period: t[cols] for period, t in tables.items()}, axis=1)


def md(df: pd.DataFrame) -> str:
    """Markdown with rates as percentages and counts as integers (by column name, or by the
    metric level of side-by-side columns)."""
    def fmt(col: pd.Series) -> pd.Series:
        name = col.name[-1] if isinstance(col.name, tuple) else col.name
        if name in RATES:
            return col.map(lambda v: f"{v:.1%}")
        return col.astype(int) if name in COUNTS else col
    df = df.apply(fmt)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(map(str, c)) for c in df.columns]
    if isinstance(df.index, pd.MultiIndex):
        df.index = [" ".join(map(str, i)) for i in df.index]
    return df.to_markdown(floatfmt=".2f")


def main() -> None:
    research = load_research()
    data = MarketData.load()
    strategies = load_strategies()
    final = strategies["final"]
    periods = {"train": (research.train_start, research.train_end), "test": (research.test_start, None)}
    last_day = f"{data.daily.index[-1]:%Y-%m-%d}"

    def sweep(variants: dict, label: str) -> dict[str, pd.DataFrame]:
        """variants: value -> (strategy config, research config). One summary row per value and period."""
        tables = {}
        for period, (start, end) in periods.items():
            rows = {}
            for value, (cfg, res) in variants.items():
                rows[value] = metrics.summary(run(cfg, res, data, start, end).daily)
                log_run(cfg, period, start, end or last_day, rows[value], note=f"robustness: {label} {value}")
            tables[period] = pd.DataFrame(rows).T
        return tables

    # ---- 1. paper versions
    ladder = load_strategies("ablation.yaml")
    versions = {p: run_ablation(ladder, research, data, p, start, end) for p, (start, end) in periods.items()}
    described = versions["train"].apply(lambda r: f"{r.name}: {r['stop']} stop, {r['sizing']}", axis=1)
    version_table = side_by_side(versions, SUMMARY).rename(index=described)
    plots.metric_bars(pd.concat({p: t.drop(index="SPY open-to-close") for p, t in versions.items()})
                      .swaplevel().loc[list(versions["train"].index.drop("SPY open-to-close"))],
                      "Paper versions: train (solid) and test (faded)", "robustness_paper_versions")

    # ---- 2. parameter sweeps and 3. costs
    vm = sweep({v: (replace(final, name=f"final_vm_{v:g}", vm=v), research) for v in VM_VALUES}, "VM")
    lookback = sweep({n: (replace(final, name=f"final_lookback_{n}", lookback=n), research) for n in LOOKBACKS},
                     "lookback")
    commission = sweep({c: (final, replace(research, commission=c)) for c in COMMISSIONS}, "commission")
    cost_models = sweep({"flat (default)": (final, research),
                         "IB tiered commission": (final, replace(research, commission_tiered=True)),
                         "I-Star market impact": (final, replace(research, slippage_model="istar")),
                         "tiered + I-Star": (final, replace(research, commission_tiered=True, slippage_model="istar"))},
                        "cost model")
    for name, tables, paper_value, title in [
            ("vm", vm, 1.0, "Volatility multiplier VM (paper Fig. 9)"),
            ("lookback", lookback, 14, "Noise-area lookback in days (paper FAQ Q6)"),
            ("commission", commission, 0.0035, "Commission per share, slippage $0.001 (paper Fig. 10)")]:
        plots.sweep_bars(tables, SWEEP_METRICS, title, f"robustness_{name}", paper_value)

    # ---- 4. and 5. the final strategy, analysed as in the paper
    results = {p: run(final, research, data, start, end) for p, (start, end) in periods.items()}
    vix, patterns = data.daily["vix_open"], pt.daily_patterns(data.daily)
    vix_mean = (data.daily["vix_open"] + data.daily["vix_close"]) / 2
    conditional, trade_rows, legs_ret = {}, {}, {}
    for p, result in results.items():
        ret, traded = result.daily["ret"], result.daily["trades"] > 0
        weekday = {d: pd.Series(ret.index.dayofweek == i, index=ret.index) for i, d in enumerate(WEEKDAYS)}
        conditional[p] = {
            "vix": pt.conditional_stats(ret, traded, {f"VIX >= {t}": vix >= t for t in VIX_THRESHOLDS}),
            "patterns": pt.conditional_stats(ret, traded, dict(patterns.items())),
            "weekday": pt.conditional_stats(ret, traded, weekday)}
        legs_ret[p] = pt.legs(result)
        for name in ("base", "vwap_stop", "final"):
            r = result if name == "final" else run(strategies[name], research, data, *periods[p])
            trade_rows[(p, name)] = pt.trade_stats(r)

    plots.sweep_bars({p: c["vix"].iloc[1:].rename(index=lambda x: x.replace("VIX >= ", "≥ "))
                      for p, c in conditional.items()},
                     [("sharpe", "Sharpe ratio (traded days)", "{:.2f}"), ("avg_bps", "Mean daily return, bps", "{:.1f}")],
                     "Final strategy on days with VIX at the open at or above a threshold (paper Fig. 8); "
                     "above 30 the bars rest on few days", "robustness_vix")
    all_legs = pd.concat(legs_ret.values())
    plots.equity_curves({"long_only": all_legs["long"], "short_only": all_legs["short"], "final": all_legs["both"]},
                        "Long and short legs of the final strategy, 2016 – Oct 2026 (paper FAQ Q5)", "robustness_legs")

    leg_table = pd.concat({p: pd.DataFrame({leg: {"total_return": pt.compounded(r[leg]),
                                                  "annual_return": metrics.annual_return(r[leg]),
                                                  "sharpe": metrics.sharpe(r[leg])} for leg in ("long", "short", "both")}).T
                           for p, r in legs_ret.items()}, axis=1)
    shorts_vix = pd.DataFrame({p: pt.shorts_above_vix(r, vix, VIX_THRESHOLDS) for p, r in legs_ret.items()})
    shorts_sma = pd.concat({p: pt.shorts_below_sma(r, data.daily["close"], SMA_WINDOWS) for p, r in legs_ret.items()},
                           axis=1)
    shorts_reg = pd.DataFrame({p: pt.short_trades_vs_vix(r, vix_mean) for p, r in results.items()}).T
    per_share = pd.concat([pt.pnl_per_share_by_year(r) for r in results.values()]).rename("avg net P&L per share ($)")
    per_share.index.name = "year"
    full_ret = pd.concat([r.daily["ret"] for r in results.values()])
    quarters = pt.worst_quarters(full_ret, baselines.buy_and_hold(data.daily.loc[full_ret.index])["ret"])
    quarters.index = quarters.index.astype(str).rename("quarter")

    commission_shown = {p: t.rename(index=lambda c: f"${c:.4f}") for p, t in commission.items()}
    conditional_cols = ["observations", "avg_bps", "t_stat", "hit_ratio", "sharpe"]
    cond = lambda key: md(side_by_side({p: c[key] for p, c in conditional.items()}, conditional_cols))
    report = "\n\n".join([
        "# The paper's variations and further investigations, train and test\n\n"
        f"_Generated by `scripts/07_robustness.py`. Train {research.train_start} – {research.train_end}, test "
        f"{research.test_start} – {last_day}. Net of costs ($0.0035 commission + $0.001 slippage per share unless "
        "stated). The paper's settings are not changed by any of this; nothing here is a candidate. Conditional "
        "tables use the days the strategy traded (paper convention); Sharpe annualized with sqrt(252)._",
        "## 1. Paper versions (Tables 1–3, Fig. 5a, FAQ Q22)\n\nFigure: robustness_paper_versions.png.\n\n"
        + md(version_table),
        "## 2. Volatility multiplier VM (Section 4.4, Fig. 9)\n\nFigure: robustness_vm.png. Paper: VM = 1 used, "
        "best Sharpe near 1.5.\n\n" + md(side_by_side(vm, SUMMARY)),
        "## 3. Noise-area lookback in days (FAQ Q6)\n\nFigure: robustness_lookback.png. Paper: Sharpe 1.23–1.35 "
        "for 5–60 days, 1.50 at 90. Longer lookbacks need a longer warm-up, so they start trading later in 2016.\n\n"
        + md(side_by_side(lookback, SUMMARY)),
        "## 4. Commission per share (Section 4.6, Fig. 10)\n\nFigure: robustness_commission.png. Slippage stays "
        "$0.001 per share.\n\n" + md(side_by_side(commission_shown, ["total_return", *SUMMARY])),
        "## 5. Cost models (Section 4.6, FAQ Q15)\n\nTiered: $0.002 per share once the previous 21 days traded "
        "more than 300,000 shares (the paper qualifies on ~20% of its days with a much larger account). I-Star: "
        "687 × (shares / ADV)^0.70 × σ^0.72 bps with the large-cap parameters (paper: Sharpe 1.33 → 1.17).\n\n"
        + md(side_by_side(cost_models, ["annual_return", "sharpe", "max_drawdown", "cost_share_of_gross"])),
        "## 6. VIX at the open (Section 4.1, Fig. 8)\n\nFigure: robustness_vix.png.\n\n" + cond("vix"),
        "## 7. Daily patterns of the previous day (Section 4.2, Table 5)\n\n" + cond("patterns"),
        "## 8. Day of the week (Section 4.3, Table 6)\n\n" + cond("weekday"),
        "## 9. Trade-level statistics (Table 4)\n\nA trade is a round trip; orders count every entry and exit "
        "(a reversal is one order). The paper's 'trades' per day (1.3 base, 1.8 band + VWAP) match our orders per "
        "day. Max loss / gain trade in % of the account.\n\n"
        + md(pd.DataFrame(trade_rows).T),
        "## 10. Average net profit per share by year (FAQ Q23)\n\n" + md(per_share.to_frame()),
        "## 11. Long and short legs (FAQ Q5)\n\nFigure: robustness_legs.png.\n\n" + md(leg_table),
        "## 12. Short trades vs VIX (FAQ Q19, Q20)\n\nQ19: return of each short trade (bps of price) regressed "
        "on the day's average VIX.\n\n" + md(shorts_reg)
        + "\n\nQ20: total return of the short trades if taken only when the VIX at the open was at or above "
        "the threshold.\n\n" + md(shorts_vix.map(lambda v: f"{v:.1%}")),
        "## 13. Shorts only in bear markets (FAQ Q21)\n\nLong trades always kept; short trades only when "
        "yesterday's close was below its n-day moving average.\n\n" + md(shorts_sma),
        "## 14. SPY's ten worst quarters, 2016 – Oct 2026 (FAQ Q7)\n\n" + md(quarters),
    ]) + "\n"
    (RESULTS_DIR / "robustness_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
