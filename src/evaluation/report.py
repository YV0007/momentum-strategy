"""Evaluate a set of strategies on one period, identically for train and test. (Phase 4)

For each strategy: headline metrics, Sharpe standard error and bootstrap interval,
alpha/beta vs SPY, placebo test, Deflated Sharpe. Plus the baselines and all figures,
written to results/figures/ and results/<period>_report.md.
"""

import json

import numpy as np
import pandas as pd

from src import plots
from src.config import BACKTEST_DIR, RESULTS_DIR, ResearchConfig, StrategyConfig
from src.engine.backtest import MarketData, prepare, simulate
from src.evaluation import baselines, metrics, stats
from src.experiment_log import LOG_FILE, log_run

N_PLACEBO = 1000


def trial_sharpes(period: str) -> list[float]:
    """Daily Sharpe of every distinct strategy logged for this period (Deflated Sharpe). Counted
    by name, not by parameter text: the config schema gained fields over time, so the same design
    was logged with different parameter texts. Every new design gets a new name."""
    if not LOG_FILE.exists():
        return []
    log = pd.read_csv(LOG_FILE)
    # ablation and robustness runs decompose or stress fixed designs; they are not candidates
    log = log[(log["period"] == period) & ~log["note"].fillna("").str.startswith(("ablation", "robustness"))]
    log = log.drop_duplicates("strategy", keep="last")
    return list(log["sharpe"] / np.sqrt(252))


def evaluate(strategies: dict[str, StrategyConfig], research: ResearchConfig, data: MarketData,
             period: str, start: str, end: str | None) -> pd.DataFrame:
    n, rf = research.trading_days, research.risk_free_rate
    daily = data.daily.loc[start:end]
    market = baselines.buy_and_hold(daily)

    returns, rows, placebo_runs = {}, {}, {}
    for name, config in strategies.items():
        prep = prepare(config, data, start, end)
        result = simulate(prep, prep.positions(), research)
        result.daily.to_parquet(BACKTEST_DIR / f"{name}_{period}_daily.parquet")
        result.trades.to_parquet(BACKTEST_DIR / f"{name}_{period}_trades.parquet")
        ret = result.daily["ret"]

        row = metrics.summary(result.daily, n, rf)
        log_run(config, period, start, str(end or daily.index[-1].date()), row)
        lo, hi = stats.bootstrap_ci(ret, n)
        placebo = baselines.placebo(prep, research, N_PLACEBO).apply(lambda c: metrics.sharpe(c, n, rf))
        row |= {"sharpe_se": stats.sharpe_se(ret, n), "sharpe_ci_low": lo, "sharpe_ci_high": hi,
                **stats.alpha_beta(ret, market["ret"], n),
                "placebo_sharpe_95pct": placebo.quantile(0.95),
                "placebo_p_value": stats.empirical_p_value(row["sharpe"], placebo.to_numpy())}
        returns[name], rows[name], placebo_runs[name] = ret, row, placebo

    deflated = {name: stats.deflated_sharpe(returns[name], trial_sharpes(period)) for name in strategies}
    for name in strategies:
        rows[name]["deflated_sharpe_prob"] = deflated[name]

    for name, bench in [("buy_and_hold", market), ("open_to_close", baselines.open_to_close(daily, research))]:
        returns[name] = bench["ret"]
        rows[name] = metrics.summary(bench, n, rf) | {
            "sharpe_se": stats.sharpe_se(bench["ret"], n),
            **dict(zip(["sharpe_ci_low", "sharpe_ci_high"], stats.bootstrap_ci(bench["ret"], n)))}

    table = pd.DataFrame(rows).T
    table.to_csv(RESULTS_DIR / f"{period}_metrics.csv")
    _figures(table, returns, placebo_runs, period)
    _write_markdown(table, period, start, daily.index[-1])
    return table


def _figures(table, returns, placebo_runs, period):
    label = period.replace("_", " ")
    plots.equity_curves(returns, f"Equity curves, {label} period", f"{period}_equity")
    plots.drawdowns({k: v for k, v in returns.items() if k != "open_to_close"},
                    f"Drawdowns, {label} period", f"{period}_drawdowns")
    plots.metric_bars(table, f"Required metrics, {label} period", f"{period}_metric_bars")
    if "final" in placebo_runs:
        plots.placebo_histogram(placebo_runs["final"], table.loc["final", "sharpe"],
                                table.loc["final", "placebo_p_value"],
                                f"Final strategy vs {N_PLACEBO} random-direction placebos, {label}",
                                f"{period}_placebo")
        plots.monthly_heatmap(returns["final"], f"Final strategy monthly returns, {label}",
                              f"{period}_monthly_final")


def _write_markdown(table: pd.DataFrame, period: str, start: str, last_day) -> None:
    pct = ["total_return", "annual_return", "annual_volatility", "max_drawdown", "hit_ratio",
           "return_at_10pct_vol", "worst_day", "best_day", "alpha", "cost_share_of_gross"]
    shown = table.copy().astype(object)
    for col in shown:
        if col in pct:
            shown[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif col in ("trades", "days"):
            shown[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        else:
            shown[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    lines = [f"# Evaluation: {period} period ({start} to {last_day:%Y-%m-%d})", "",
             "_Generated by the evaluation scripts; do not edit by hand._", "",
             shown.T.to_markdown(), "",
             f"Figures: `results/figures/{period}_*.png`."]
    (RESULTS_DIR / f"{period}_report.md").write_text("\n".join(lines))
    (RESULTS_DIR / f"{period}_metrics.json").write_text(json.dumps(table.to_dict(), default=float, indent=1))
